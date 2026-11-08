"""Turn Cosmos-Transfer outputs back into a bench-format dataset: same actions and joint states as the source
episodes, camera frames replaced by the generated ones (one new episode per style).

    python cosmos/import_cosmos.py --src data/isaac_pre/place_c0 --export data/cosmos_export/place_c0 \
        --specs cosmos/specs/place_c0 --out data/isaac_cosmos/place_c0
"""
import argparse, glob, io, json, os
import av, numpy as np, pandas as pd
from PIL import Image


def read_video(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--export", required=True); ap.add_argument("--specs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name = json.load(open(os.path.join(a.src, "repos.json")))[0].replace("/", "__")
    F = pd.read_parquet(os.path.join(a.src, "frames", name + ".parquet")); Z = np.load(os.path.join(a.src, "actions", name + ".npz"))
    rows, acts, new_ep = [], {}, 0
    by_style = {}
    for s in sorted(glob.glob(os.path.join(a.specs, "*.json"))):
        spec = json.load(open(s)); ep, cam, style = spec["name"].rsplit("_", 2)
        vids = glob.glob(os.path.join(spec["output_dir"], "**", "*.mp4"), recursive=True)
        if vids:
            by_style.setdefault((ep, style), {})[cam] = vids[0]
    for (ep, style), cams in sorted(by_style.items()):
        if set(cams) != {"front", "wrist"}:
            continue
        k = int(ep[2:]); meta = json.load(open(os.path.join(a.export, ep, "episode.json"))); fidx = meta["fidx"]
        gen = {c: read_video(p) for c, p in cams.items()}
        src = F[F.ep == k]
        for r in src[src.cam == "front"].itertuples():
            j = min(range(len(fidx)), key=lambda q: abs(fidx[q] - r.fidx))           # 15 fps export -> 10 Hz dataset frames
            for c in ("front", "wrist"):
                im = gen[c][min(j * len(gen[c]) // max(len(fidx), 1), len(gen[c]) - 1)]
                b = io.BytesIO(); Image.fromarray(im).resize((160, 120), Image.BILINEAR).save(b, format="JPEG", quality=90)
                rows.append((new_ep, c, int(r.fidx), b.getvalue(), r.task, r.task_orig))
        acts[f"ep{new_ep}_action"] = Z[f"ep{k}_action"]; acts[f"ep{new_ep}_state"] = Z[f"ep{k}_state"]; new_ep += 1
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True); os.makedirs(os.path.join(a.out, "actions"), exist_ok=True)
    D = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig"]); D["fps"] = 30
    D.to_parquet(os.path.join(a.out, "frames", name + "_cosmos.parquet"))
    np.savez_compressed(os.path.join(a.out, "actions", name + "_cosmos.npz"), **acts)
    json.dump([name.replace("__", "/") + "_cosmos"], open(os.path.join(a.out, "repos.json"), "w"))
    print(new_ep, "augmented episodes ->", a.out)


if __name__ == "__main__":
    main()
