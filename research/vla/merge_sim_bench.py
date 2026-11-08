"""Merge several sim_data.py outputs (same task/repo name) into one bench dir with renumbered episodes."""
import sys, os, json, glob
import numpy as np, pandas as pd
out = sys.argv[1]; srcs = sys.argv[2:]
F, A, off = [], {}, 0
for s in srcs:
    name = json.load(open(os.path.join(s, "repos.json")))[0].replace("/", "__")
    f = pd.read_parquet(os.path.join(s, "frames", name + ".parquet")); z = np.load(os.path.join(s, "actions", name + ".npz"))
    eps = sorted(int(k[2:].split("_")[0]) for k in z.files if k.endswith("_action"))
    for e in eps:
        A[f"ep{e + off}_action"] = z[f"ep{e}_action"]; A[f"ep{e + off}_state"] = z[f"ep{e}_state"]
        if f"ep{e}_pol" in z.files:                   # DAgger: which steps were driven by the policy
            A[f"ep{e + off}_pol"] = z[f"ep{e}_pol"]
    f["ep"] = f["ep"] + off; off += max(eps) + 1
    if "loc" not in f.columns:                    # GCT labels: older sources have none
        f["loc"] = None
    F.append(f)
    print(s, len(eps))
os.makedirs(os.path.join(out, "frames"), exist_ok=True); os.makedirs(os.path.join(out, "actions"), exist_ok=True)
(pd.concat(F) if any(x["loc"].notna().any() for x in F) else pd.concat(F).drop(columns=["loc"])).to_parquet(os.path.join(out, "frames", name + ".parquet"))
np.savez_compressed(os.path.join(out, "actions", name + ".npz"), **A)
json.dump([name.replace("__", "/")], open(os.path.join(out, "repos.json"), "w"))
print("total episodes", off)
