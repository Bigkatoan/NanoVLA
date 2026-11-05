"""Closed-loop success rate of a NanoVLA policy in the Isaac tabletop (batched envs).

The policy runs in a separate process (vla/isaac_policy_server.py --bench <traj dataset>), fine-tuned on demos recorded
by traj_tool.py; this client renders the two cameras (160x120), sends joint state (rad) + images + the episode's
instruction, and executes the returned chunk: chunk element k is reached in `step` control steps (linear
interpolation), the first `exec` elements are executed, then it re-plans. Success = the tabletop task check
(object placed / lifted, gripper open where required). Each env runs its own episodes asynchronously.

    python tabletop_eval.py --tasks bowl --episodes 50 --num_envs 4 --port 6000 --out results/bowl_N20.json
"""
import argparse, json, os, sys, time
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="pick,pad,tray,bowl")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--seed", type=int, default=10_000, help="layout seed (keep disjoint from the demo seeds)")
ap.add_argument("--port", type=int, default=6000)
ap.add_argument("--exec", type=int, default=4)
ap.add_argument("--max_s", type=float, default=25.0)
ap.add_argument("--out", default="results/tabletop_eval.json")
ap.add_argument("--video", default="", help="mp4 of env 0's first episodes")
ap.add_argument("--video_eps", type=int, default=3)
a = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from multiprocessing.connection import Client
import tabletop as tt


def main():
    env = tt.Tabletop(num_envs=a.num_envs, seed=a.seed, task_mix=tuple(a.tasks.split(",")))
    conn = Client(("localhost", a.port), authkey=b"vla-isaac"); conn.send("meta"); meta = conn.recv(); step = meta["step"]
    n = env.n
    env.reset()
    for _ in range(10):
        env.step(np.tile(tt.HOME, (n, 1)))
    env.t[:] = 0
    cur = env.state().copy()                   # last commanded target per env
    plan = [None] * n; kp = np.zeros(n, int)
    res, frames, ep0 = [], [], 0
    t0 = time.time()
    while len(res) < a.episodes:
        S = env.state()
        need = [i for i in range(n) if plan[i] is None or kp[i] >= len(plan[i])]
        if need or (a.video and ep0 < a.video_eps):
            f, w = env.images()
        if need:
            conn.send(dict(state=S[need], front=[f[i] for i in need], wrist=[w[i] for i in need], text=[env.ep[i]["text"] for i in need]))
            chunks = conn.recv()
            for j, i in enumerate(need):
                seq, prev = [], cur[i]
                for k in range(a.exec):
                    nxt = chunks[j, k]
                    seq += [prev + (nxt - prev) * (s + 1) / step for s in range(step)]
                    prev = nxt
                plan[i], kp[i] = seq, 0
        cmd = np.stack([plan[i][kp[i]] for i in range(n)]); kp += 1; cur = cmd.copy()
        if a.video and ep0 < a.video_eps:
            frames.append(np.concatenate([f[0], w[0]], 1))
        env.step(cmd)
        succ = env.success()
        done = [i for i in range(n) if succ[i] or env.t[i] >= a.max_s * tt.CTRL_HZ]
        if not done:
            continue
        for i in done:
            E = env.ep[i]
            res.append(dict(success=bool(succ[i]), task=E["task"], obj=E["obj"], text=E["text"], t=round(env.t[i] / tt.CTRL_HZ, 2)))
            if i == 0:
                ep0 += 1
            print(json.dumps(res[-1] | {"n": len(res), "sr": round(float(np.mean([r["success"] for r in res])), 3), "el": round(time.time() - t0)}), flush=True)
        env.reset(env_ids=done, settle=0)
        for i in done:
            plan[i] = None; cur[i] = env.state()[i]
    res = res[: a.episodes]
    by = {}
    for r in res:
        for key in (r["task"], r["obj"]):
            b = by.setdefault(key, [0, 0]); b[0] += r["success"]; b[1] += 1
    out = dict(tasks=a.tasks, episodes=len(res), success_rate=float(np.mean([r["success"] for r in res])), by=by, exec=a.exec, seed=a.seed,
               wall_s=round(time.time() - t0), results=res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    if frames:
        import av
        c = av.open(a.video, "w"); s = c.add_stream("libx264", rate=tt.CTRL_HZ); s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()
    print("RESULT", json.dumps({k: v for k, v in out.items() if k != "results"}), flush=True)
    conn.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)
