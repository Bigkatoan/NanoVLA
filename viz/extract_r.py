"""Real per-tick internals of NanoVLA-R (CPU): pooling attention, LSTM memory trajectory, flow sampling path, chunk vs label."""
import os, sys, json, glob, io, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.expanduser("~/VLA/vla"))
import nanovla_r as R, nanovla2 as v2
from train_text import clean
from PIL import Image
ck = torch.load("vlar_s2500.pt", map_location="cpu", weights_only=False); st = ck["st"]
m, tok = R.build("cpu", "own"); m.load_state_dict(ck["model"]); m.eval()
D = os.path.expanduser("~/VLA/data/isaac_v2/place_c8")            # an episode with grasp + release
M = json.load(open(f"{D}/episodes.json")); E = next(e for e in M["episodes"] if e["task"] == "container")
ep = int(E["ep"]); F = pd.read_parquet(glob.glob(f"{D}/frames/*.parquet")[0]); g = F[(F.ep == ep) & (F.cam == "wrist")].sort_values("fidx")
z = np.load(glob.glob(f"{D}/actions/*.npz")[0]); A, S = z[f"ep{ep}_action"], z[f"ep{ep}_state"]
# same calibration as training (fewshot.prepare applies coef to the actions)
A = A * ck["coef"][:, 0] + ck["coef"][:, 1]
fr = [(int(f), np.asarray(Image.open(io.BytesIO(j)).convert("RGB"))) for f, j in zip(g.fidx, g.jpeg) if f + 3 < len(A)]
ph = v2.phases(A)
with torch.no_grad():
    u = m.text_vec(tok([clean(E["text"])]))
    s_prev = None; out = dict(h=[], phase=[], chunk=[], label=[], att=[], flow=[], f=[])
    hc = None
    for k, (f, im) in enumerate(fr):
        x = torch.from_numpy(im).permute(2, 0, 1)[None, None]
        img = R.gpu_images(x, v2.SIZE["own"], False)[0]
        sn = torch.tensor((S[f] - st["s_mean"]) / st["s_std"], dtype=torch.float32)[None]
        ds = torch.zeros_like(sn) if s_prev is None else torch.tensor((S[f] - s_prev) / st["s_std"] * 5, dtype=torch.float32)[None]
        s12 = torch.cat([sn, ds], -1).clamp(-5, 5); s_prev = S[f]
        if hc is None: hc = m.init_state(u, s12)
        # pooling attention (re-run the pooling block with weights)
        tokv = m.vproj(m.vis(img)) + m.vpos
        gg, bb = m.qfilm(u).chunk(2, -1); q = m.pool_q * (1 + gg[:, None]) + bb[:, None]
        _, w = m.pool(m.pool_n1(q), tokv, tokv, need_weights=True)
        v = m.image_vec(img, u)
        h, hc = m.lstm(torch.cat([v, s12, u], -1)[:, None], hc); h = h[:, 0]
        c = m.cond(h, v, s12, u)
        xs = [torch.zeros(1, 16, 6)]
        for j in range(5):
            xs.append(xs[-1] + m.velocity(c, xs[-1], torch.full((1,), j / 5)) / 5)
        p = xs[-1][0].numpy()
        fut = np.minimum(f + 3 * np.arange(1, 17), len(A) - 1)
        chunk = np.c_[S[f, :5] + p[:, :5] * st["dstd"], (p[:, 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]]
        out["h"].append(h[0].numpy()); out["phase"].append(int(ph[f])); out["chunk"].append(chunk); out["label"].append(A[fut]); out["f"].append(f)
        out["att"].append(w[0].numpy().reshape(4, 8, 10)); out["flow"].append(np.stack([x[0].numpy() for x in xs]))
H_ = np.array(out["h"]); H_ = H_ - H_.mean(0); U, Sv, Vt = np.linalg.svd(H_, full_matrices=False); pc = H_ @ Vt[:2].T
crops = np.stack([np.asarray(Image.fromarray(im).resize((160, 128))) for _, im in fr])
np.savez("rec.npz", frames=crops, pc=pc, phase=np.array(out["phase"]), chunk=np.array(out["chunk"]), label=np.array(out["label"]), att=np.array(out["att"]),
         flow=np.array(out["flow"]), state=S[np.array(out["f"])], dstd=st["dstd"])
json.dump(dict(text=E["text"], n=len(fr), var2=float((Sv[:2] ** 2).sum() / (Sv ** 2).sum()), l1=float(np.mean(np.abs(np.array(out["chunk"]) - np.array(out["label"]))))), open("rec.json", "w"))
print(E["text"], len(fr), np.bincount(out["phase"]), json.load(open("rec.json")))
