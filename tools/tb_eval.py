"""Stream closed-loop eval progress (tabletop_eval logs) into TensorBoard: success rate so far, per-task rate, and the
rollout video frames of env 0 (as images) once the eval finishes.
    python tb_eval.py <eval.log> <tb run dir> <name>       (polls until RESULT appears)"""
import json, os, sys, time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
log, run, name = sys.argv[1:4]
w = SummaryWriter(run); seen = 0
while True:
    L = [json.loads(l) for l in open(log)] if os.path.exists(log) and False else []
    rows = []
    if os.path.exists(log):
        for l in open(log):
            if l.startswith('{"success"'):
                rows.append(json.loads(l))
    for k in range(seen, len(rows)):
        sr = np.mean([r["success"] for r in rows[: k + 1]])
        w.add_scalar(f"eval_{name}/success_rate_running", sr, k + 1)
        w.add_scalar(f"eval_{name}/episode_time_s", rows[k]["t"], k + 1)
    seen = len(rows); w.flush()
    done = os.path.exists(log) and any(l.startswith("RESULT") for l in open(log))
    if done:
        by = {}
        for r in rows:
            b = by.setdefault(r["task"], [0, 0]); b[0] += r["success"]; b[1] += 1
        w.add_text(f"eval_{name}/result", " · ".join(f"{t}: {s}/{n}" for t, (s, n) in sorted(by.items())) + f"  — total {sum(r['success'] for r in rows)}/{len(rows)}")
        for t, (s, n) in by.items():
            w.add_scalar(f"eval_{name}/by_task/{t}", s / n, 0)
        mp4 = log.replace(".eval.log", ".mp4")
        if os.path.exists(mp4):
            import av
            fr = [f.to_ndarray(format="rgb24") for f in av.open(mp4).decode(video=0)]
            idx = np.linspace(0, len(fr) - 1, 24).astype(int)
            w.add_images(f"eval_{name}/rollout_env0", np.stack([fr[i] for i in idx]).transpose(0, 3, 1, 2), 0)
        w.close(); break
    time.sleep(20)
