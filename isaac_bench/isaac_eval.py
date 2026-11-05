"""Closed-loop success-rate benchmark of our VLA on the NVIDIA Sim-to-Real SO-101 workshop task (Isaac Lab).

Same protocol as the workshop's GR00T eval (scripts/lerobot_eval.py): task Lerobot-So101-Teleop-Vials-To-Rack-Eval
(or -DR-Eval), 10 warm-up steps at the initial pose, 450-step (7.5 s @ 60 Hz) episode limit, success = the env's
`vial_placed_on_rack` termination before time-out. The policy runs in a separate process (scratch_vla/
isaac_policy_server.py); every query returns a 16-step chunk spaced `step` env steps apart, the first `exec`
chunk steps are executed with linear interpolation, then it re-plans.
"""
import argparse, json, os, sys, time, io
sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-Eval")
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--seed", type=int, default=1984)
ap.add_argument("--port", type=int, default=6000)
ap.add_argument("--exec", type=int, default=4)
ap.add_argument("--out", default=os.path.expanduser("~/VLA/isaac_bench/results/run.json"))
ap.add_argument("--videos", type=int, default=0, help="save the first K episodes of env 0 as mp4")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a).app

import random
import gymnasium as gym, numpy as np, torch
from PIL import Image
from multiprocessing.connection import Client
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401

# LeRobot <-> USD joint mapping (copied from the workshop's LeRobotSO101Interface.SO101_USD_MAPPING)
JMIN = np.array([-110, -100, -100, -95, -160, -10], float); JMAX = np.array([110, 100, 90, 95, 160, 100], float)
INIT = np.array([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465])


def rad_to_raw(q):
    n = (np.degrees(q) - JMIN) / (JMAX - JMIN)
    r = n * 200 - 100; r[..., 5] = n[..., 5] * 100
    return r


def raw_to_rad(r):
    n = (r + 100) / 200; n[..., 5] = r[..., 5] / 100
    return np.radians(JMIN + n * (JMAX - JMIN))


def small(img):
    """640x480 render -> 160x120 JPEG round-trip, matching how the training frames were made"""
    b = io.BytesIO(); Image.fromarray(img).resize((160, 120), Image.BILINEAR).save(b, format="JPEG", quality=90)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))


def main():
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=a.num_envs); cfg.seed = a.seed
    env = gym.make(a.task, cfg=cfg); u = env.unwrapped; n = a.num_envs
    conn = Client(("localhost", a.port), authkey=b"vla-isaac"); conn.send("meta"); meta = conn.recv(); step = meta["step"]
    obs, _ = env.reset()
    t = np.zeros(n, int)                     # env step within episode
    plan = [None] * n; k_plan = np.zeros(n, int)  # current interpolated command sequence (rad) and index
    cur = np.tile(INIT, (n, 1))
    res, ep_started = [], 0
    vid, vids = [], []
    grasped = np.zeros(n, bool)
    t0 = time.time()
    while len(res) < a.episodes:
        q = obs["policy"]["joint_pos_obs"].cpu().numpy().astype(np.float64)
        need = [i for i in range(n) if t[i] >= 10 and (plan[i] is None or k_plan[i] >= len(plan[i]))]
        if need:
            f = obs["visual"]["rgb_external_D455"][..., :3].cpu().numpy().astype(np.uint8)
            w = obs["visual"]["rgb_ego"][..., :3].cpu().numpy().astype(np.uint8)
            conn.send(dict(state=rad_to_raw(q[need]), front=[small(f[i]) for i in need], wrist=[small(w[i]) for i in need]))
            chunks = raw_to_rad(conn.recv())
            for j, i in enumerate(need):
                seq, prev = [], cur[i]
                for kk in range(a.exec):
                    nxt = chunks[j, kk]
                    seq += [prev + (nxt - prev) * (s + 1) / step for s in range(step)]
                    prev = nxt
                plan[i], k_plan[i] = seq, 0
        act = np.empty((n, 6))
        for i in range(n):
            if t[i] < 10:
                act[i] = INIT
            else:
                act[i] = plan[i][k_plan[i]]; k_plan[i] += 1
        cur = act.copy()
        if a.videos and len(vids) < a.videos:
            vid.append(np.concatenate([obs["visual"]["rgb_external_D455"][0, ..., :3].cpu().numpy(), obs["visual"]["rgb_ego"][0, ..., :3].cpu().numpy()], 1).astype(np.uint8)[::2, ::2])
        obs, _, term, trunc, _ = env.step(torch.tensor(act, dtype=torch.float32, device=u.device))
        t += 1
        if "subtask_terms" in obs:
            grasped |= obs["subtask_terms"]["vial_grasped"].cpu().numpy().reshape(-1) > 0.5
        done = (term | trunc).cpu().numpy()
        for i in np.nonzero(done)[0]:
            succ = bool(term[i].item() and not trunc[i].item())
            res.append(dict(success=succ, grasped=bool(grasped[i] or succ), steps=int(t[i]), env=int(i)))
            print(json.dumps(dict(ep=len(res), **res[-1], sr=round(np.mean([r["success"] for r in res]), 3))), flush=True)
            if i == 0 and a.videos and len(vids) < a.videos:
                vids.append((np.stack(vid), succ)); vid = []
            t[i] = 0; plan[i] = None; cur[i] = INIT; grasped[i] = False
    sr = float(np.mean([r["success"] for r in res[:a.episodes]]))
    out = dict(task=a.task, episodes=a.episodes, success_rate=sr, grasp_rate=float(np.mean([r["grasped"] for r in res[:a.episodes]])),
               mean_steps_success=float(np.mean([r["steps"] for r in res if r["success"]])) if any(r["success"] for r in res) else None,
               exec=a.exec, num_envs=n, seed=a.seed, wall_s=round(time.time() - t0), results=res[:a.episodes])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    if vids:
        import av
        for j, (v, s) in enumerate(vids):
            p = a.out.replace(".json", f"_vid{j}_{'success' if s else 'fail'}.mp4")
            c = av.open(p, "w"); st = c.add_stream("libx264", rate=30); st.width, st.height, st.pix_fmt = v.shape[2], v.shape[1], "yuv420p"
            for fr in v:
                for pk in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    c.mux(pk)
            for pk in st.encode():
                c.mux(pk)
            c.close()
    print("RESULT", json.dumps({k: v for k, v in out.items() if k != "results"}), flush=True)
    conn.close(); env.close(); app.close()


if __name__ == "__main__":
    main()
