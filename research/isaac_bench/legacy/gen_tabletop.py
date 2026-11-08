"""Generate scripted-expert demos in the multi-task tabletop (Isaac Lab), saved in the few-shot bench format:
  <out>/frames/isaac__tabletop.parquet (ep, cam, fidx, jpeg, task, task_orig, fps=30)  -- every 3rd control step (10 Hz)
  <out>/actions/isaac__tabletop.npz    (ep<k>_action, ep<k>_state; rad, 30 Hz)
  <out>/episodes.json                  (task type, instruction, objects, layout, texture, success)
Only successful expert episodes are kept; the expert success rate per task type is reported.
"""
import argparse, io, json, os, sys, time
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=6)
ap.add_argument("--n", type=int, default=200, help="successful demos to collect")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tasks", default="pick,pad,tray,stack")
ap.add_argument("--out", default=os.path.expanduser("~/VLA/data/isaac_tabletop"))
ap.add_argument("--video", type=int, default=0, help="also write the first K attempts (all envs, front+wrist) to mp4")
ap.add_argument("--max_steps", type=int, default=330)
a = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from PIL import Image
import tabletop as tt
from tabletop_expert import Expert


def jpg(x):
    b = io.BytesIO(); Image.fromarray(x).save(b, format="JPEG", quality=90); return b.getvalue()


def main():
    rng = np.random.default_rng(a.seed + 7)
    env = tt.Tabletop(num_envs=a.num_envs, seed=a.seed, task_mix=tuple(a.tasks.split(",")))
    ex = Expert(env, rng)
    n = env.n
    env.reset()
    for i in range(n):
        ex.start(i)
    buf = [dict(S=[], A=[], F=[]) for _ in range(n)]
    rows, acts, meta = [], {}, []
    tried, stats = 0, {}
    vid = [] if a.video else None
    t0 = time.time()
    while len(meta) < a.n:
        cmd, status, err = ex.act(np.ones(n, bool))
        S = env.state()
        rec = env.t % 3 == 0
        if rec.any() or vid is not None:
            f, w = env.images()
        for i in range(n):
            buf[i]["S"].append(S[i].astype(np.float32)); buf[i]["A"].append(cmd[i].astype(np.float32))
            if rec[i]:
                buf[i]["F"].append((int(env.t[i]), jpg(f[i]), jpg(w[i])))
        if vid is not None and tried < a.video:
            vid.append(np.concatenate([np.concatenate([f[i], w[i]], 1) for i in range(min(n, 4))], 0))
        env.step(cmd)
        succ = env.success()
        finished = [i for i in range(n) if status[i] in ("done", "fail") or env.t[i] >= a.max_steps]
        for i in finished:
            E = env.ep[i]; ok = bool(succ[i]) and status[i] == "done"
            tried += 1; st = stats.setdefault(E["task"], [0, 0]); st[0] += ok; st[1] += 1
            if ok and len(meta) < a.n:
                e = len(meta)
                acts[f"ep{e}_action"] = np.stack(buf[i]["A"]); acts[f"ep{e}_state"] = np.stack(buf[i]["S"])
                for fidx, fj, wj in buf[i]["F"]:
                    rows.append((e, "front", fidx, fj, E["text"], E["text"])); rows.append((e, "wrist", fidx, wj, E["text"], E["text"]))
                meta.append(dict(ep=e, task=E["task"], text=E["text"], obj=E["obj"], present=E["present"], pads=E["pads"], tray=E["tray"],
                                 texture=os.path.basename(tt.TEX[E["table_mat"]]), steps=int(env.t[i]),
                                 layout={k: [float(v[0][0]), float(v[0][1]), float(v[1])] for k, v in E["layout"].items()}))
            print(json.dumps(dict(dt_reset=round(getattr(env, "_last_reset_s", 0), 1), tried=tried, kept=len(meta), task=E["task"], ok=ok, status=status[i], steps=int(env.t[i]),
                                  rate={k: round(v[0] / v[1], 2) for k, v in stats.items()}, el=round(time.time() - t0))), flush=True)
            buf[i] = dict(S=[], A=[], F=[])
        if finished:
            env.reset(env_ids=finished)
            for i in finished:
                ex.reset_env(i); ex.start(i)
        if vid is not None and tried >= a.video and len(vid):
            import av
            os.makedirs(a.out, exist_ok=True)
            c = av.open(os.path.join(a.out, "expert_preview.mp4"), "w"); s = c.add_stream("libx264", rate=30)
            s.width, s.height, s.pix_fmt = vid[0].shape[1], vid[0].shape[0], "yuv420p"
            for fr in vid:
                for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
            c.close(); vid = None
            print("preview written", flush=True)
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True); os.makedirs(os.path.join(a.out, "actions"), exist_ok=True)
    F = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig"]); F["fps"] = 30
    F.to_parquet(os.path.join(a.out, "frames", "isaac__tabletop.parquet"))
    np.savez_compressed(os.path.join(a.out, "actions", "isaac__tabletop.npz"), **acts)
    json.dump(["isaac/tabletop"], open(os.path.join(a.out, "repos.json"), "w"))
    json.dump(dict(episodes=meta, expert_rate={k: v[0] / v[1] for k, v in stats.items()}, tried=tried), open(os.path.join(a.out, "episodes.json"), "w"), indent=1)
    print("DONE", json.dumps({k: round(v[0] / v[1], 3) for k, v in stats.items()}), "kept", len(meta), "tried", tried, flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)
