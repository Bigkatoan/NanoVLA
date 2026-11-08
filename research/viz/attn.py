"""Real forward pass of ckpt/vla_isaac_wrist_v1 on dataset frames: decoder cross-attention over the 49 wrist tokens,
predicted chunk vs ground truth. CPU only."""
import os, sys, io, json, glob, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.expanduser("~/VLA/research/vla"))
os.environ["VLA_CKPT"] = os.path.expanduser("~/VLA/ckpt/vla_isaac_wrist_v1/vla.pt")
import fewshot as fs
from sim_fewshot import prep_img
from train_text import clean
from PIL import Image
ck = torch.load(os.environ["VLA_CKPT"], map_location="cpu", weights_only=False); st = ck["st"]
model, tok, row = fs.make_model("A", "cpu"); model.eval()
store = []
for L in model.dec.layers:
    mha = L.multihead_attn; f0 = mha.forward
    def fw(*a, _f=f0, **k):
        k["need_weights"] = True; k["average_attn_weights"] = True
        o = _f(*a, **k); store.append(o[1].detach()); return o
    mha.forward = fw
R = os.path.expanduser("~/VLA/data/isaac_v2")
cases = [("place_c8", "container"), ("place_c0", "tray"), ("pick_c5", "pick"), ("push_c6", "push")]
out = []
for d, task in cases:
    M = json.load(open(f"{R}/{d}/episodes.json")); E = next(e for e in M["episodes"] if e["task"] == task)
    ep = int(E["ep"]); F = pd.read_parquet(glob.glob(f"{R}/{d}/frames/*.parquet")[0]); g = F[(F.ep == ep) & (F.cam == "wrist")].sort_values("fidx")
    z = np.load(glob.glob(f"{R}/{d}/actions/*.npz")[0]); A, S = z[f"ep{ep}_action"], z[f"ep{ep}_state"]
    for q in (0.1, 0.45, 0.8):
        r = g.iloc[int(q * (len(g) - 1))]; f = int(r.fidx)
        im = np.asarray(Image.open(io.BytesIO(r.jpeg)).convert("RGB"))
        x = prep_img(im)[None]; ext, wri, hw = fs.wrist_views(x, "ext")
        s = torch.from_numpy(np.clip((S[f] - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32))[None]
        store.clear()
        with torch.no_grad():
            p = model(ext, wri, hw, s, tok([clean(E["text"])]), torch.full((1,), row)).numpy()[0]
        att = torch.stack(store)[:, 0]               # layers, 16 queries, mem
        a49 = att[-1, :, :49].mean(0).numpy().reshape(7, 7)
        mod = att.mean((0, 1)).numpy(); L_t = mod.shape[0] - 100
        share = dict(vision=float(mod[:49].sum()), wrist_masked=float(mod[49:98].sum()), text=float(mod[98:98 + L_t].sum()), state=float(mod[-2]), emb=float(mod[-1]))
        idx = np.minimum(f + 3 * np.arange(1, 17), len(A) - 1)
        pred = np.empty((16, 6)); pred[:, :5] = S[f, :5] + p[:, :5] * st["dstd"]; pred[:, 5] = (p[:, 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
        crop = np.asarray(Image.fromarray(im).resize((149, 112)))[:, 18:130]
        out.append(dict(task=task, text=E["text"], q=q, img=im, crop=crop, att=a49, share=share, pred=pred, gt=A[idx], state=S[f]))
        print(task, q, {k: round(v, 3) for k, v in share.items()}, "L1", round(float(np.abs(pred - A[idx]).mean()), 3))
np.save("attn.npy", np.array(out, dtype=object), allow_pickle=True)
