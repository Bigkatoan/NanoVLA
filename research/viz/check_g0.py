"""Gate G0 check on the test chunks: frames from the start, start states, labels consistent with the images."""
import glob, io, json, os, sys
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
R = os.path.expanduser("~/VLA/data/g0")
HOME = np.array([0.0, -0.6, -0.07, 1.51, -1.57, -0.1])
K = None
report, sheet = {}, []
for d in ("pick", "place", "push"):
    D = f"{R}/{d}"
    if not os.path.exists(f"{D}/episodes.json"):
        report[d] = "missing"; continue
    M = json.load(open(f"{D}/episodes.json")); F = pd.read_parquet(glob.glob(f"{D}/frames/*.parquet")[0]); z = np.load(glob.glob(f"{D}/actions/*.npz")[0])
    r = dict(kept=M["kept"], attempts=M["attempts"], first_fidx=[], start_off=[], grip0=[], start_noise=[e.get("start_noise") for e in M["episodes"]],
             dart=[e.get("dart") for e in M["episodes"]], close_label_dist_cm=[])
    for e in M["episodes"]:
        ep = int(e["ep"]); g = F[(F.ep == ep) & (F.cam == "wrist")].sort_values("fidx"); S = z[f"ep{ep}_state"]; A = z[f"ep{ep}_action"]
        f0 = int(g.fidx.iloc[0]); r["first_fidx"].append(f0); r["start_off"].append(round(float(np.abs(S[f0, :5] - HOME[:5]).mean()), 3)); r["grip0"].append(round(float(S[f0, 5]), 2))
        # at the moment the jaw closes on the object, the grasp point should sit near the gripper (small, in front of the camera)
        gc = A[:, 5]; op = np.nonzero(gc > 0.5)[0]
        if len(op):
            cl = np.nonzero(gc[op[0]:] < 0.3)[0]
            if len(cl):
                tcl = op[0] + cl[0]; row = g.iloc[np.argmin(np.abs(g.fidx.values - tcl))]; loc = np.asarray(row["loc"])
                r["close_label_dist_cm"].append(round(float(np.linalg.norm(loc[:3])) * 100, 1))
        # overlay: project the grasp point with an approximate pinhole (fx from 160 px width, ~90 deg hfov)
        if len(sheet) < 12:
            for fi in (0, len(g) // 3, 2 * len(g) // 3):
                row = g.iloc[fi]; im = Image.open(io.BytesIO(row.jpeg)).convert("RGB").resize((320, 240)); dr = ImageDraw.Draw(im); loc = np.asarray(row["loc"])
                for k, col in ((0, (255, 40, 40)), (1, (40, 255, 40))):
                    x, y, zz = loc[3 * k: 3 * k + 3]
                    if zz > 0.01:
                        f = 206.0   # 13.5 mm / 20.955 mm * 320 px (DR may change the focal slightly)
                        u, v = 160 + f * x / zz, 120 + f * y / zz
                        dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=col, width=3)
                dr.text((4, 4), f"{d} ep{ep} f{int(row.fidx)} z={loc[2]:.2f}", fill=(255, 255, 0))
                sheet.append(np.asarray(im))
    report[d] = r
print(json.dumps(report, indent=1))
if sheet:
    rows = [np.concatenate(sheet[i:i + 3], 1) for i in range(0, len(sheet) - len(sheet) % 3, 3)]
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.expanduser("~/VLA/research/viz/g0_sheet.png"))
