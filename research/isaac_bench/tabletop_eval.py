"""Closed-loop success rate of a NanoVLA policy in the Isaac tabletop (batched envs).

The policy runs in a separate process (vla/isaac_policy_server.py --bench <traj dataset>), fine-tuned on demos recorded
by traj_tool.py; this client renders the two cameras (160x120), sends joint state (rad) + images + the episode's
instruction, and executes the returned chunk: chunk element k is reached in `step` control steps (linear
interpolation), the first `exec` elements are executed, then it re-plans. Success = the tabletop task check
(object placed / lifted, gripper open where required). Each env runs its own episodes asynchronously.

    python tabletop_eval.py --tasks bowl --episodes 50 --num_envs 4 --port 6000 --out results/bowl_N20.json
    # wrist-camera-only policy (server started with --cams wrist): only env.scene["wrist"] is rendered/read
    python tabletop_eval.py --cams wrist --tasks bowl --episodes 50 --num_envs 4 --port 6000 --out results/bowl_wrist.json
    # no Isaac: fake wrist-only scene, in-process fake policy (--port 0) or a real policy server (--port N)
    python tabletop_eval.py --dry_run --cams wrist --port 0 --episodes 6
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="pick,pad,tray,bowl")
ap.add_argument("--objects", default="", help="comma list: only episodes whose target object is one of these (e.g. held-out objects)")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--seed", type=int, default=10_000, help="layout seed (keep disjoint from the demo seeds)")
ap.add_argument("--port", type=int, default=6000)
ap.add_argument("--exec", type=int, default=4)
ap.add_argument("--max_s", type=float, default=25.0)
ap.add_argument("--out", default="results/tabletop_eval.json")
ap.add_argument("--video", default="", help="mp4 of env 0's first episodes")
ap.add_argument("--video_eps", type=int, default=3)
ap.add_argument("--cams", default="both", choices=("both", "wrist"), help="wrist: render/send the wrist camera only (scene may lack 'front')")
ap.add_argument("--diag", type=int, default=0, help="log per-episode grasp diagnostics: min gripper-object distance, where the jaw closed, lift")
ap.add_argument("--warmup_spec", default="", help="hypothesis test: the scripted expert (this spec) drives the first --warmup_steps after each reset, then the policy (fresh memory) takes over")
ap.add_argument("--warmup_steps", type=int, default=30)
ap.add_argument("--jpeg", type=int, default=0, help="JPEG-encode the policy images at this quality (training frames are JPEG q90)")
ap.add_argument("--dry_run", action="store_true", help="no Isaac: fake wrist-only scene (policy_io.FakeTabletop); --port 0 = fake policy")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing.connection import Client
import policy_io as pio
if a.dry_run:
    tt = pio.FakeTabletop
else:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    import torch
    import tabletop as tt


def main():
    if a.dry_run:
        env = pio.FakeTabletop(num_envs=a.num_envs, seed=a.seed, task_mix=tuple(a.tasks.split(",")))
    else:
        if a.objects:                                   # restrict target objects (rejection sampling on top of the tabletop sampler)
            only = set(a.objects.split(",")); _orig = tt.Tabletop._sample_once
            tt.GRASP[:] = sorted(set(tt.GRASP) | only)       # held-out objects may have been removed by TABLETOP_HOLDOUT
            def _restricted(self, _orig=_orig, only=only):
                E = _orig(self)
                return E if (E is not None and E["obj"] in only) else None
            tt.Tabletop._sample_once = _restricted
        env = tt.Tabletop(num_envs=a.num_envs, seed=a.seed, task_mix=tuple(a.tasks.split(",")))
    conn = pio.FakeConn(a.cams) if a.dry_run and a.port == 0 else Client(("localhost", a.port), authkey=b"vla-isaac")
    conn.send("meta"); meta = conn.recv(); step = meta["step"]
    if meta.get("cams", "both") != a.cams:
        raise SystemExit(f"policy server runs with cams={meta.get('cams', 'both')} but the client has --cams {a.cams}")
    n = env.n
    env.reset()
    for _ in range(10):
        env.step(np.tile(tt.HOME, (n, 1)))
    env.t[:] = 0
    kin = runner = None
    if a.diag or a.warmup_spec:
        from tabletop_expert import Expert
        kin = Expert(env, np.random.default_rng(0))
    if a.warmup_spec:
        from traj_core import TrajRunner
        runner = TrajRunner(env, json.load(open(a.warmup_spec)), np.random.default_rng(1)); runner.reset()
    D = [dict(dmin=9.0, t_dmin=None, close_t=None, close_d=None, was_open=False, lift=0.0) for _ in range(env.n)]
    cur = env.state().copy()                   # last commanded target per env
    plan = [None] * n; kp = np.zeros(n, int); fresh = np.ones(n, bool)     # fresh: episode started, memory must be reset
    res, frames, ep0 = [], [], 0
    t0 = time.time()
    while len(res) < a.episodes:
        S = env.state()
        warm = np.array([runner is not None and env.t[i] < a.warmup_steps for i in range(n)])
        wcmd = runner.step() if runner is not None else None
        if kin is not None:                          # grasp diagnostics (graspable target objects)
            tcp = kin.tcp_pose()[0].cpu().numpy()
            for i in range(n):
                E = env.ep[i]
                if E["obj"] not in tt.GRASP:
                    continue
                g, _ = env.grasp_point(E["obj"], i); d = float(np.linalg.norm(tcp[i] - g)); dd = D[i]
                if d < dd["dmin"]:
                    dd["dmin"], dd["t_dmin"] = d, round(env.t[i] / tt.CTRL_HZ, 2)
                if S[i, 5] > 0.5:
                    dd["was_open"] = True
                if dd["was_open"] and dd["close_t"] is None and S[i, 5] < 0.25 and not warm[i]:
                    dd["close_t"], dd["close_d"] = round(env.t[i] / tt.CTRL_HZ, 2), round(d, 4)
                dd["lift"] = max(dd["lift"], float(env.pos(E["obj"])[i][2] + tt.OBJ_META[E["obj"]]["zmin"]))
        need = [i for i in range(n) if (plan[i] is None or kp[i] >= len(plan[i])) and not warm[i]]
        for i in range(n):
            if warm[i] and env.t[i] == a.warmup_steps - 1:
                fresh[i] = True; plan[i] = None
        if need or (a.video and ep0 < a.video_eps):
            imgs = pio.get_images(env, a.cams)
            if a.jpeg:
                import io as _io
                from PIL import Image as _I
                def _j(x):
                    b_ = _io.BytesIO(); _I.fromarray(x).save(b_, format="JPEG", quality=a.jpeg); return np.asarray(_I.open(b_).convert("RGB"))
                imgs = {k_: np.stack([_j(x) for x in v_]) for k_, v_ in imgs.items()}
        if need:
            conn.send(pio.build_request(S, need, imgs, [E["text"] for E in env.ep], a.cams, fresh)); fresh[need] = False
            chunks = np.asarray(conn.recv(), dtype=np.float64)
            for j, i in enumerate(need):
                seq, prev = [], cur[i]
                for k in range(a.exec):
                    nxt = chunks[j, k]
                    seq += [prev + (nxt - prev) * (s + 1) / step for s in range(step)]
                    prev = nxt
                plan[i], kp[i] = seq, 0
        cmd = np.stack([wcmd[i] if warm[i] else plan[i][kp[i]] for i in range(n)])
        for i in range(n):
            if not warm[i]:
                kp[i] += 1
        cur = cmd.copy()
        if a.video and ep0 < a.video_eps:
            frames.append(pio.video_frame(imgs, a.cams))
        env.step(cmd)
        succ = env.success()
        done = [i for i in range(n) if succ[i] or env.t[i] >= a.max_s * tt.CTRL_HZ]
        if not done:
            continue
        for i in done:
            E = env.ep[i]
            # grounding: another graspable object was lifted (the policy acted on the wrong object)
            wrong = [o for o in E.get("present", []) if o != E["obj"] and o in tt.OBJ_META
                     and env.pos(o)[i][2] + tt.OBJ_META[o]["zmin"] > 0.03]
            res.append(dict(success=bool(succ[i]), task=E["task"], obj=E["obj"], text=E["text"], t=round(env.t[i] / tt.CTRL_HZ, 2),
                            wrong_obj=wrong, n_present=len(E.get("present", []))))
            if kin is not None:
                dd = D[i]; res[-1].update(dmin_cm=round(100 * dd["dmin"], 1), t_dmin=dd["t_dmin"], close_t=dd["close_t"],
                                          close_d_cm=None if dd["close_d"] is None else round(100 * dd["close_d"], 1), lift_cm=round(100 * dd["lift"], 1))
                D[i] = dict(dmin=9.0, t_dmin=None, close_t=None, close_d=None, was_open=False, lift=0.0)
            if i == 0:
                ep0 += 1
            print(json.dumps(res[-1] | {"n": len(res), "sr": round(float(np.mean([r["success"] for r in res])), 3), "el": round(time.time() - t0)}), flush=True)
        env.reset(env_ids=done, settle=0)
        for i in done:
            plan[i] = None; cur[i] = env.state()[i]; fresh[i] = True
            if runner is not None:
                runner.reset_env(i)
    res = res[: a.episodes]
    by = {}
    for r in res:
        for key in (r["task"], r["obj"]):
            b = by.setdefault(key, [0, 0]); b[0] += r["success"]; b[1] += 1
    wrong_rate = float(np.mean([bool(r.get("wrong_obj")) for r in res])) if res else 0.0
    out = dict(wrong_object_rate=wrong_rate, tasks=a.tasks, cams=a.cams, dry_run=a.dry_run, episodes=len(res), success_rate=float(np.mean([r["success"] for r in res])), by=by, exec=a.exec, seed=a.seed,
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
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)   # Kit would hang on exit
    sys.stdout.flush(); os._exit(0)
