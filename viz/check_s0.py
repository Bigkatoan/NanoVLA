"""Check the NanoVLA-S labels on the test chunk: step sequence, phase, jaw label vs actual jaw, waypoint projected on the image."""
import glob, io, json, os, sys
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/VLA/data/s0/place")
M = json.load(open(f"{D}/episodes.json")); F = pd.read_parquet(glob.glob(f"{D}/frames/*.parquet")[0]); z = np.load(glob.glob(f"{D}/actions/*.npz")[0])
sheet = []
for e in M["episodes"][:4]:
    ep = int(e["ep"]); g = F[(F.ep == ep) & (F.cam == "wrist")].sort_values("fidx"); L = np.stack(g["loc"].values); S = z[f"ep{ep}_state"]
    ks = L[:, 11].astype(int); ph = L[:, 12].astype(int)
    seq = [int(ks[0])] + [int(k) for a_, k in zip(ks[:-1], ks[1:]) if k != a_]
    jaw = S[g.fidx.values, 5]; lab = L[:, 10]
    agree = float(np.mean((jaw > 0.4) == (lab > 0.5)))
    print(json.dumps(dict(ep=ep, task=e["task"], text=e["text"], frames=len(g), steps_seq=seq, phases=sorted(set(ph.tolist())),
                          jaw_label_agreement=round(agree, 2), way_dist_cm=[round(float(np.linalg.norm(L[i, 6:9])) * 100, 1) for i in np.linspace(0, len(g) - 1, 6).astype(int)])))
    for fi in np.linspace(0, len(g) - 1, 4).astype(int):
        row = g.iloc[fi]; im = Image.open(io.BytesIO(row.jpeg)).convert("RGB").resize((320, 240)); dr = ImageDraw.Draw(im); l = L[fi]
        for k, col in ((0, (255, 50, 50)), (1, (50, 255, 50)), (2, (255, 220, 0))):
            x, y, zz = l[3 * k: 3 * k + 3]
            if zz > 0.01:
                u, v = 160 + 206 * x / zz, 120 + 206 * y / zz
                dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=col, width=3)
        dr.text((4, 4), f"ep{ep} k={int(l[11])} ph={int(l[12])} jaw={'open' if l[10] > .5 else 'close'} dyaw={np.degrees(l[9]):.0f}", fill=(255, 255, 0))
        sheet.append(np.asarray(im))
rows = [np.concatenate(sheet[i:i + 4], 1) for i in range(0, len(sheet), 4)]
Image.fromarray(np.concatenate(rows, 0)).save(os.path.expanduser("~/VLA/viz/s0_sheet.png"))
