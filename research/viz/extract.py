import json, io, glob, os, collections, numpy as np, pandas as pd
from PIL import Image
R = "/home/bigkatoan/VLA/data/isaac_v2"
D = [d for d in sorted(glob.glob(f"{R}/*_c*")) if os.path.isdir(d)] + [f"{R}/fs_bowl"]
cnt = collections.Counter(); thumbs = {}; texts = {}
for d in D:
    M = json.load(open(f"{d}/episodes.json")); cnt.update(e["task"] for e in M["episodes"])
    F = None
    for e in M["episodes"]:
        t = e["task"]
        if t in thumbs: continue
        if F is None: F = pd.read_parquet(glob.glob(f"{d}/frames/*.parquet")[0])
        g = F[(F.ep == int(e["ep"])) & (F.cam == "wrist")].sort_values("fidx")
        if len(g) < 5: continue
        idx = np.linspace(0, len(g) - 1, 6).astype(int)
        thumbs[t] = np.stack([np.asarray(Image.open(io.BytesIO(g.jpeg.iloc[i])).convert("RGB").resize((96, 72))) for i in idx]); texts[t] = e["text"]
print(cnt, sorted(thumbs))
F = pd.read_parquet(f"{R}/pick_c5/frames/traj__pick.parquet"); g = F[(F.ep == 0) & (F.cam == "wrist")].sort_values("fidx")
seq = np.stack([np.asarray(Image.open(io.BytesIO(j)).convert("RGB").resize((80, 60))) for j in g.jpeg.iloc[np.linspace(0, len(g) - 1, 24).astype(int)]])
z = np.load(f"{R}/pick_c5/actions/traj__pick.npz"); st, ac = z["ep0_state"], z["ep0_action"]
dr = []
for d in D[:12]:
    F = pd.read_parquet(glob.glob(f"{d}/frames/*.parquet")[0]); F = F[F.cam == "wrist"]
    for ep in sorted(F.ep.unique())[:2]:
        g = F[F.ep == ep].sort_values("fidx"); dr.append(np.asarray(Image.open(io.BytesIO(g.jpeg.iloc[len(g)//3])).convert("RGB").resize((64, 48))))
order = ["pick", "pad", "tray", "container", "rel", "push", "topple", "rotate", "bowl"]
res = {}
for n in ("A_zeroshot_multitask", "B_bowl_isaac_N10", "B_bowl_isaac_N50", "B_bowl_v1_N50", "C_unseen_zeroshot", "C_unseen_ft"):
    J = json.load(open(f"/home/bigkatoan/VLA/research/results/final_v1/{n}.json")); res[n] = dict(sr=J["success_rate"], n=J["episodes"], by={k: v for k, v in J["by"].items() if k in order}, wrong=J.get("wrong_object_rate", 0))
json.dump(dict(cnt=dict(cnt), texts=texts, res=res), open("data.json", "w"), indent=1)
np.savez_compressed("data.npz", seq=seq, st=st, ac=ac, dr=np.stack(dr), **{f"th_{k}": v for k, v in thumbs.items()})
print(st.shape, len(dr), res["A_zeroshot_multitask"]["by"])
