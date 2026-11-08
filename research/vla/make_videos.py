"""Before/after cleaning videos (side by side) + one compilation.

left : original episode (all sampled frames), red border + reason when the frame is removed
right: cleaned stream (only kept frames), same playback rate -> ends earlier
bottom: timeline of the original episode, green = kept, colour = removed reason, white cursor
Output: data/clean/videos/*.mp4  (H.264, 2x real time)
"""
import os, io, sys, argparse
import numpy as np, pandas as pd, av
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from clean_data import IDLE_EDGE, IDLE_PAUSE, SPIKE, FROZEN, EP_DROP
BLANK = 32

ROOT = os.path.expanduser("~/VLA/data")
CL = os.environ.get("CLEAN_DIR", "clean")
OUT = os.path.join(ROOT, CL, "videos")
W, H = 1280, 560
PW, PH = 600, 450
COL = {BLANK: (90, 90, 90), IDLE_EDGE: (220, 50, 50), IDLE_PAUSE: (240, 150, 40), SPIKE: (150, 60, 200), FROZEN: (60, 120, 220), EP_DROP: (120, 0, 0)}
NAME = {BLANK: "corrupt / blank frame", IDLE_EDGE: "idle start/end", IDLE_PAUSE: "pause > 1s", SPIKE: "servo spike", FROZEN: "frozen video", EP_DROP: "episode dropped"}
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except Exception:
    FONT = FONT_S = FONT_B = ImageFont.load_default()

I = pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["repo_id", "ep", "cam", "wrist", "fidx", "off", "len", "task_id"])
F = pd.read_parquet(os.path.join(ROOT, CL, "frames.parquet"))
E = pd.read_parquet(os.path.join(ROOT, CL, "episodes.parquet"))
R = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet")).set_index("repo_id")
T = pd.read_parquet(os.path.join(ROOT, "packed/tasks.parquet")).set_index("task_id")
blob = np.memmap(os.path.join(ROOT, "packed/frames.bin"), dtype=np.uint8, mode="r")


def reason_of(bits):
    for b in (EP_DROP, BLANK, FROZEN, SPIKE, IDLE_PAUSE, IDLE_EDGE):
        if bits & b:
            return b
    return 0


def load(row):
    return Image.open(io.BytesIO(bytes(blob[row.off: row.off + row.len]))).convert("RGB").resize((PW, PH), Image.BILINEAR)


def card(text, sub=""):
    im = Image.new("RGB", (W, H), (20, 20, 24))
    d = ImageDraw.Draw(im)
    d.text((W // 2, H // 2 - 30), text, font=FONT_B, fill="white", anchor="mm")
    if sub:
        d.text((W // 2, H // 2 + 20), sub, font=FONT, fill=(200, 200, 200), anchor="mm")
    return im


def episode_frames(s, title):
    repo, fps = R.loc[s.repo_id].repo, int(R.loc[s.repo_id].fps)
    Ie = I[(I.repo_id == s.repo_id) & (I.ep == s.ep)]
    ext = Ie[Ie.wrist == 0].cam.unique()
    cam = ext[0] if len(ext) else Ie.cam.iloc[0]
    Ie = Ie[Ie.cam == cam].sort_values("fidx")
    keep = F.keep.to_numpy()[Ie.index.to_numpy()]
    bits = F.reasons.to_numpy()[Ie.index.to_numpy()]
    w = float(F.weight.to_numpy()[Ie.index.to_numpy()][0])
    imgs = [load(r) for r in Ie.itertuples()]
    kept_imgs = [im for im, k in zip(imgs, keep) if k]
    n = len(imgs)
    task = T.loc[Ie.task_id.iloc[0]].task
    dur_o = (Ie.fidx.iloc[-1] - Ie.fidx.iloc[0]) / fps
    dur_c = dur_o * keep.mean()
    yield from [card(title, f"{repo}  ep{s.ep}")] * 8
    for i in range(n):
        im = Image.new("RGB", (W, H), (245, 245, 245))
        d = ImageDraw.Draw(im)
        d.text((20, 8), f"{title}   |   {repo} ep{s.ep}", font=FONT, fill="black")
        d.text((20, 32), f"task: {task[:120]}", font=FONT_S, fill=(60, 60, 60))
        # left: original
        x0, y0 = 20, 58
        im.paste(imgs[i], (x0, y0))
        b = reason_of(bits[i])
        col = (40, 170, 60) if keep[i] else COL.get(b, (200, 0, 0))
        d.rectangle([x0 - 4, y0 - 4, x0 + PW + 3, y0 + PH + 3], outline=col, width=6)
        d.text((x0, y0 + PH + 8), f"BEFORE  t={(Ie.fidx.iloc[i]-Ie.fidx.iloc[0])/fps:5.1f}s / {dur_o:.1f}s", font=FONT, fill="black")
        if not keep[i]:
            lbl = "REMOVED: " + (NAME.get(b, "") if b != EP_DROP else f"episode dropped ({s['drop']})")
            d.rectangle([x0, y0, x0 + PW, y0 + 34], fill=(0, 0, 0))
            d.text((x0 + 8, y0 + 6), lbl, font=FONT, fill=COL.get(b, (255, 80, 80)) if b != EP_DROP else (255, 90, 90))
        # right: cleaned stream
        x1 = 660
        k_i = int(keep[: i + 1].sum()) - 1
        j = min(i, len(kept_imgs) - 1)
        if len(kept_imgs) == 0:
            d.rectangle([x1, y0, x1 + PW, y0 + PH], fill=(40, 40, 40))
            d.text((x1 + PW // 2, y0 + PH // 2), "episode removed from training", font=FONT, fill="white", anchor="mm")
        elif i < len(kept_imgs):
            im.paste(kept_imgs[i], (x1, y0))
        else:
            g = kept_imgs[-1].convert("L").convert("RGB")
            im.paste(g, (x1, y0))
            d.text((x1 + PW // 2, y0 + PH // 2), f"END  (cleaned {dur_c:.1f}s vs original {dur_o:.1f}s)", font=FONT, fill="white", anchor="mm",
                   stroke_width=2, stroke_fill="black")
        d.rectangle([x1 - 4, y0 - 4, x1 + PW + 3, y0 + PH + 3], outline=(40, 170, 60), width=6)
        extra = f"   loss weight {w:g}" if 0 < w < 1 else ""
        d.text((x1, y0 + PH + 8), f"AFTER  kept {keep.sum()}/{n} frames{extra}", font=FONT, fill="black")
        # timeline
        ty, tx0, tx1 = H - 22, 20, W - 20
        for k in range(n):
            a = tx0 + (tx1 - tx0) * k / n; z = tx0 + (tx1 - tx0) * (k + 1) / n
            c = (40, 170, 60) if keep[k] else COL.get(reason_of(bits[k]), (200, 0, 0))
            d.rectangle([a, ty, z, ty + 14], fill=c)
        cx = tx0 + (tx1 - tx0) * (i + 0.5) / n
        d.polygon([(cx - 6, ty - 8), (cx + 6, ty - 8), (cx, ty)], fill="black")
        yield im
    # hold the end a little
    for _ in range(6):
        yield im


def write(frames, path, fps):
    c = av.open(path, "w")
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.options = {"crf": "23", "preset": "veryfast"}
    for im in frames:
        for p in s.encode(av.VideoFrame.from_image(im)):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_random", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    Eh = E[E.has_actions.fillna(False)].copy()
    Eh["drop"] = Eh["drop"].fillna(""); Eh["outlier"] = Eh["outlier"].fillna("")
    pz = (F.reasons.to_numpy() & IDLE_PAUSE) > 0
    pk = I.loc[pz, ["repo_id", "ep"]].drop_duplicates()
    Eh["has_pause"] = Eh.set_index(["repo_id", "ep"]).index.isin(pk.set_index(["repo_id", "ep"]).index)
    rs = np.random.RandomState(7)
    pick = lambda d: d.iloc[rs.randint(len(d))]
    cases = [
        ("Idle start/end trimmed", Eh[(Eh["drop"] == "") & (Eh.idle_cut_frac > 0.35) & (Eh.dur > 8)]),
        ("Mid-episode pause removed", Eh[(Eh["drop"] == "") & Eh.has_pause & (Eh.dur.between(10, 40))]),
        ("Dropped: corrupted / uncorrelated camera", Eh[Eh["drop"].str.contains("cam_uncorrelated") & (Eh.dur > 5)]),
        ("Dropped: frozen video", Eh[Eh["drop"].str.contains("video_frozen") & (Eh.dur > 5)]),
        ("Dropped: action/state mismatch", Eh[(Eh["drop"] == "action_state_mismatch") & (Eh.dur > 4)]),
        ("Dropped: follower servo stuck", Eh[Eh["drop"].str.contains("follower_joint_stuck") & (Eh.dur > 4)]),
        ("Dropped: corrupt video frames", Eh[Eh["drop"].str.contains("video_corrupt") & (Eh.dur > 4)]),
        ("Down-weighted: within-repo outlier", Eh[(Eh.outlier != "") & (Eh.dur > 5)]),
    ]
    sel = [(t, pick(d)) for t, d in cases if len(d)]
    sel += [("Random episode", s) for _, s in Eh[Eh.dur > 5].sample(a.n_random, random_state=1).iterrows()]
    allf = []
    for i, (title, s) in enumerate(sel):
        frames = list(episode_frames(s, title))
        name = f"{i+1:02d}_{title.split(':')[0].replace(' ', '_').replace('/', '')}.mp4"
        write(frames, os.path.join(OUT, name), 10)
        allf += frames
        print("wrote", name, R.loc[s.repo_id].repo, s.ep, len(frames), "frames", flush=True)
    write([card("Data cleaning: before vs after", "left = original, right = cleaned, 2x speed")] * 20 + allf, os.path.join(OUT, "00_compilation.mp4"), 10)
    print("wrote 00_compilation.mp4")


if __name__ == "__main__":
    main()
