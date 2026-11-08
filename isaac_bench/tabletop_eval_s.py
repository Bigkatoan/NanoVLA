"""Closed-loop eval of the state-based NanoVLA-S (vla/vlas_server.py) with a non-learned IK servo.

Every --query_every control steps (default 3 = 10 Hz; any value works: the policy is time-free) the client sends the
wrist frame + joints + instruction; the server answers the task state, the target point in the wrist-camera frame, the
yaw to turn and the jaw. The client turns the target into the world with the camera pose (gripper link x mount) and,
at every control step (30 Hz), moves the grasp centre towards it at <= --speed m/s with DLS IK; the jaw ramps to
open/closed. --jitter J randomises the query spacing in [q-J, q+J] steps to test robustness to timing.

    python tabletop_eval_s.py --tasks pick --episodes 24 --port 6000 --out results/s/pick.json --diag 1
"""
import argparse, json, math, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="pick,pad,tray,container,push,topple,rotate,rel")
ap.add_argument("--objects", default="")
ap.add_argument("--episodes", type=int, default=24)
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--seed", type=int, default=30000)
ap.add_argument("--port", type=int, default=6000)
ap.add_argument("--max_s", type=float, default=30.0)
ap.add_argument("--query_every", type=int, default=3)
ap.add_argument("--jitter", type=int, default=0)
ap.add_argument("--speed", type=float, default=0.18)
ap.add_argument("--out", default="results/s/eval.json")
ap.add_argument("--video", default="")
ap.add_argument("--video_eps", type=int, default=3)
ap.add_argument("--diag", type=int, default=1)
ap.add_argument("--oracle_modules", default="", help="with --oracle_spec: the MODEL runs, but these modules get the true answer (grounder,state,grasp)")
ap.add_argument("--nano_ik", default="", help="use the learned NanoIK (tools/nano_ik.py checkpoint) instead of the analytic DLS IK")
ap.add_argument("--hold_on_jaw", type=int, default=1)
ap.add_argument("--oracle_spec", default="", help="controller check: the expert's own state / target / jaw (TrajRunner) replace the model")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing.connection import Client
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
import torch
import tabletop as tt
import policy_io as pio
from tabletop_expert import Expert, quat_from_axes, OPEN, CLOSE
from traj_core import GRIP_T

FLIP = np.diag([1.0, -1.0, -1.0])


def q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    env = tt.Tabletop(num_envs=a.num_envs, seed=a.seed, task_mix=tuple(a.tasks.split(",")))
    if a.objects:
        only = set(a.objects.split(",")); _orig = tt.Tabletop._sample_once
        tt.GRASP[:] = sorted(set(tt.GRASP) | only)
        tt.Tabletop._sample_once = lambda self, _o=_orig: (lambda E: E if (E is not None and E["obj"] in only) else None)(_o(self))
    conn = None
    if not a.oracle_spec or a.oracle_modules:
        conn = Client(("localhost", a.port), authkey=b"vla-isaac"); conn.send("meta"); conn.recv()
    from isaaclab.utils import math as mu
    n = env.n; rng = np.random.default_rng(a.seed + 1)
    env.reset()
    for _ in range(10):
        env.step(np.tile(tt.HOME, (n, 1)))
    env.t[:] = 0
    kin = Expert(env, rng); kin.q_cmd = env.robot.data.joint_pos.clone()
    nik = None
    if a.nano_ik:
        sys.path.insert(0, os.path.expanduser("~/VLA/tools")); from nano_ik import NanoIK
        nik = NanoIK().to(env.dev); nik.load_state_dict(torch.load(a.nano_ik, map_location=env.dev)["ik"]); nik.eval()
    runner = None
    if a.oracle_spec:
        from traj_core import TrajRunner
        runner = TrajRunner(env, json.load(open(a.oracle_spec)), np.random.default_rng(3)); runner.reset()
    bi = env.robot.body_names.index("gripper")
    tgt = kin.tcp_pose()[0].cpu().numpy().copy(); yaw_t = np.zeros(n); g = env.state()[:, 5].copy(); g_t = g.copy()
    next_q = np.zeros(n, int); fresh = np.ones(n, bool); states = [[] for _ in range(n)]
    D = [dict(dmin=9.0, close_t=None, close_d=None, was_open=False, lift=0.0) for _ in range(n)]
    res, frames, ep0, t0 = [], [], 0, time.time()

    def dest_point(i):
        E = env.ep[i]; t = E["task"]
        if t == "pad": return env.pos(E["pad"])[i]
        if t == "tray": c, _ = env.tray_center(i); return np.r_[c, 0.0]
        if t == "container": return env.pos(E["container"])[i]
        if t == "bowl": return env.pos("bowl")[i]
        if t == "rel": return np.r_[np.asarray(E["rel_xy"], float), 0.0]
        if t == "push": return np.r_[np.asarray(E["p0"], float) + E.get("push_len", 0.10) * np.asarray(E["dir"], float), 0.0]
        g_ = env.pos(E["obj"])[i]
        return g_ + np.array([0, 0, 0.10]) if t == "pick" else g_

    def cam_pose():
        pb = env.robot.data.body_pos_w[:, bi].cpu().numpy() - env.origins.cpu().numpy(); Rb = mu.matrix_from_quat(env.robot.data.body_quat_w[:, bi]).cpu().numpy()
        out = []
        for i in range(n):
            tq = env.dr.cam[i]["t"].Get(); oq = env.dr.cam[i]["o"].Get(); ql = np.r_[oq.GetReal(), list(oq.GetImaginary())]
            out.append((pb[i] + Rb[i] @ np.array(tq, float), Rb[i] @ q2m(ql / np.linalg.norm(ql)) @ FLIP, math.atan2(Rb[i][1, 0], Rb[i][0, 0])))
        return out

    while len(res) < a.episodes:
        S = env.state()
        need = [i for i in range(n) if env.t[i] >= next_q[i]]
        if need or (a.video and ep0 < a.video_eps):
            imgs = pio.get_images(env, "wrist")
        if runner is not None:
            runner.targets()                                   # advance the expert's node machine (its sub-goals are the oracle answers)
            cur_ = kin.tcp_pose()[0].cpu().numpy()
            for i in range(n):
                goal = runner.p1[i] if runner.phase[i] in ("move", "settle") else runner.p_hold[i]
                runner.err[i] = float(np.linalg.norm(cur_[i] - goal))
        if need and runner is not None and not a.oracle_modules:
            for i in need:
                tgt[i] = runner.p_hold[i] if (runner.done[i] or runner.phase[i] == "grip") else runner.p1[i]; yaw_t[i] = runner.yaw[i]
                k_ = min(int(runner.k[i]), len(runner.spec["trajectory"]) - 1)
                g_t[i] = OPEN if runner.spec["trajectory"][k_]["gripper"] == "open" else CLOSE
                states[i].append(int(runner.k[i])); fresh[i] = False
                next_q[i] = env.t[i] + max(1, a.query_every + (int(rng.integers(-a.jitter, a.jitter + 1)) if a.jitter else 0))
        elif need:
            req = dict(state=S[need], wrist=[imgs["wrist"][i] for i in need], text=[env.ep[i]["text"] for i in need], ids=[int(i) for i in need], new=[bool(fresh[i]) for i in need])
            cpg = cam_pose(); tcw = kin.tcp_pose()[0].cpu().numpy()
            req["tcp_cam"] = [((tcw[i] - cpg[i][0]) @ cpg[i][1]).tolist() for i in need]
            tru = []                                               # ground truth, only logged by the server (diagnostics)
            for i in need:
                E = env.ep[i]; tc, Rc, _ = cpg[i]
                gpt = env.grasp_point(E["obj"], i)[0] if E["obj"] in tt.GRASP else env.pos(E["obj"])[i]
                tru.append(dict(obj=((gpt - tc) @ Rc).tolist(), dest=((dest_point(i) - tc) @ Rc).tolist()))
            req["truth"] = tru
            if a.oracle_modules:                                   # true answers for the modules under test
                cp0 = cam_pose(); orc = []
                for i in need:
                    tc, Rc, cy = cp0[i]; E = env.ep[i]
                    gpt = env.grasp_point(E["obj"], i)[0] if E["obj"] in tt.GRASP else env.pos(E["obj"])[i]
                    k_ = len(runner.spec["trajectory"]) if runner.done[i] else int(runner.k[i])
                    orc.append(dict(obj=((gpt - tc) @ Rc).tolist(), dest=((dest_point(i) - tc) @ Rc).tolist(), state=k_,
                                    dyaw=float((runner.yaw[i] - cy + math.pi) % (2 * math.pi) - math.pi)))
                req["oracle"] = orc
            conn.send(req)
            ans = conn.recv(); cp = cam_pose()
            for j, i in enumerate(need):
                r_ = ans[j]; tc, Rc, cy = cp[i]
                tgt[i] = tc + Rc @ np.asarray(r_["target_cam"]); yaw_t[i] = cy + r_["dyaw"]; g_t[i] = OPEN if r_["jaw_open"] > 0.5 else CLOSE
                states[i].append(r_["state"]); fresh[i] = False
                next_q[i] = env.t[i] + max(1, a.query_every + (int(rng.integers(-a.jitter, a.jitter + 1)) if a.jitter else 0))
        # ---- servo: speed-limited step towards the target, DLS IK, jaw ramp
        cur = kin.tcp_pose()[0].cpu().numpy()
        step = tgt - cur; nrm = np.linalg.norm(step, axis=1, keepdims=True); lim = a.speed / tt.CTRL_HZ
        pos_des = cur + step * np.minimum(1.0, lim / np.maximum(nrm, 1e-9))
        if a.hold_on_jaw:                                         # like the expert's grip phase: the arm holds while the jaw ramps
            ramp = np.abs(g_t - g) > 1e-3
            pos_des[ramp] = cur[ramp]
        quat = np.stack([quat_from_axes(np.array([math.cos(y), math.sin(y), 0.0]), np.array([0, 0, 1.0])) for y in yaw_t])
        if nik is None:
            dq, _ = kin.ik_step(torch.tensor(pos_des, device=env.dev, dtype=torch.float32), torch.tensor(quat, device=env.dev, dtype=torch.float32), gain=1.0)
        else:                                                   # learned IK: displacement asked in the CURRENT wrist-camera frame
            cp = cam_pose(); qa = env.robot.data.joint_pos
            dp_cam = np.stack([cp[i][1].T @ (pos_des[i] - cur[i]) for i in range(n)])
            dyw = np.clip([(yaw_t[i] - cp[i][2] + math.pi) % (2 * math.pi) - math.pi for i in range(n)], -0.05, 0.05)
            with torch.no_grad():
                dq = nik(qa.float(), torch.tensor(dp_cam, device=env.dev, dtype=torch.float32), torch.tensor(dyw, device=env.dev, dtype=torch.float32))
            kin.q_cmd[:, :5] = qa[:, :5]                        # differential IK from the actual joints
        kin.q_cmd[:, :5] += dq
        g = g + np.clip(g_t - g, -(OPEN - CLOSE) / GRIP_T, (OPEN - CLOSE) / GRIP_T)
        kin.q_cmd[:, 5] = torch.tensor(g, device=env.dev, dtype=torch.float32)
        lim_ = env.robot.data.soft_joint_pos_limits; kin.q_cmd = torch.clamp(kin.q_cmd, lim_[..., 0], lim_[..., 1])
        if a.diag:
            tcp = kin.tcp_pose()[0].cpu().numpy()
            for i in range(n):
                E = env.ep[i]
                if E["obj"] not in tt.GRASP:
                    continue
                gp, _ = env.grasp_point(E["obj"], i); d = float(np.linalg.norm(tcp[i] - gp)); dd = D[i]
                dd["dmin"] = min(dd["dmin"], d)
                if S[i, 5] > 0.5: dd["was_open"] = True
                if dd["was_open"] and dd["close_t"] is None and S[i, 5] < 0.25:
                    dd["close_t"], dd["close_d"] = round(env.t[i] / tt.CTRL_HZ, 2), round(d, 4)
                dd["lift"] = max(dd["lift"], float(env.pos(E["obj"])[i][2] + tt.OBJ_META[E["obj"]]["zmin"]))
        if a.video and ep0 < a.video_eps:
            frames.append(imgs["wrist"][0])
        env.step(kin.q_cmd.cpu().numpy().astype(np.float64))
        succ = env.success()
        done = [i for i in range(n) if (succ[i] and env.t[i] > 30) or env.t[i] >= a.max_s * tt.CTRL_HZ]
        if not done:
            continue
        for i in done:
            E = env.ep[i]; sq = states[i]
            r = dict(success=bool(succ[i]), task=E["task"], obj=E["obj"], text=E["text"], t=round(env.t[i] / tt.CTRL_HZ, 2),
                     state_seq=[sq[0]] + [s_ for p_, s_ in zip(sq[:-1], sq[1:]) if s_ != p_] if sq else [])
            if a.diag and E["obj"] in tt.GRASP:
                dd = D[i]; r.update(dmin_cm=round(100 * dd["dmin"], 1), close_t=dd["close_t"], close_d_cm=None if dd["close_d"] is None else round(100 * dd["close_d"], 1), lift_cm=round(100 * dd["lift"], 1))
            res.append(r); ep0 += (i == 0)
            print(json.dumps(r | {"n": len(res), "sr": round(float(np.mean([x["success"] for x in res])), 3), "el": round(time.time() - t0)}), flush=True)
            D[i] = dict(dmin=9.0, close_t=None, close_d=None, was_open=False, lift=0.0); states[i] = []
        env.reset(env_ids=done, settle=0)
        jp = env.robot.data.joint_pos.clone()
        for i in done:
            kin.q_cmd[i] = jp[i]; fresh[i] = True; next_q[i] = 0; g[i] = env.state()[i, 5]; g_t[i] = g[i]
            if runner is not None:
                runner.reset_env(i)
        cur = kin.tcp_pose()[0].cpu().numpy()
        for i in done:
            tgt[i] = cur[i]; yaw_t[i] = cam_pose()[i][2]
    res = res[: a.episodes]; by = {}
    for r in res:
        for key in (r["task"], r["obj"]):
            b = by.setdefault(key, [0, 0]); b[0] += r["success"]; b[1] += 1
    out = dict(tasks=a.tasks, episodes=len(res), success_rate=float(np.mean([r["success"] for r in res])), by=by, query_every=a.query_every, jitter=a.jitter, seed=a.seed, results=res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    if frames and a.video:
        import av
        c = av.open(a.video, "w"); s = c.add_stream("libx264", rate=tt.CTRL_HZ); s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")): c.mux(p)
        for p in s.encode(): c.mux(p)
        c.close()
    print("RESULT", json.dumps({k: v for k, v in out.items() if k != "results"}), flush=True)
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)
