"""Stream DAgger data generation (traj_tool logs) into TensorBoard: kept / attempted episodes, share of steps driven by
the policy, expert recovery rate, per chunk.   python tb_gen.py <tb run dir> <glob of chunk logs> <stop-file>"""
import glob, json, os, sys, time
from torch.utils.tensorboard import SummaryWriter
run, pat, stop = sys.argv[1:4]
w = SummaryWriter(run); seen = {}
while True:
    tot_k = tot_a = 0
    for f in sorted(glob.glob(pat)):
        if f.endswith(".server.log"):
            continue
        rows = [json.loads(l) for l in open(f) if l.startswith('{"ep"')]
        name = os.path.basename(f)[:-4]; n0 = seen.get(name, 0)
        for k in range(n0, len(rows)):
            R = rows[: k + 1]
            w.add_scalar(f"dagger_{name}/kept", sum(r["kept"] for r in R), k + 1)
            w.add_scalar(f"dagger_{name}/expert_recovered_rate", sum(r["success_check"] for r in R) / len(R), k + 1)
            w.add_scalar(f"dagger_{name}/policy_driven_share", rows[k].get("policy_frac", 0), k + 1)
        seen[name] = len(rows); tot_k += sum(r["kept"] for r in rows); tot_a += len(rows)
    w.add_scalar("dagger_total/kept_episodes", tot_k, int(time.time())); w.add_scalar("dagger_total/attempts", tot_a, int(time.time())); w.flush()
    if os.path.exists(stop):
        w.close(); break
    time.sleep(30)
