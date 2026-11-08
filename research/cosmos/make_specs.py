"""Write Cosmos-Transfer2.5 spec files for every exported episode x camera x style.

    python cosmos/make_specs.py --export data/cosmos_export/place_c0 --styles 3 --out cosmos/specs/place_c0
Each spec uses the episode's sim RGB as input video, sim depth + segmentation as controls (geometry and object
identity stay fixed) plus on-the-fly edge, and a prompt = base scene description + one style.
"""
import argparse, glob, json, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--styles", type=int, default=3, help="styles sampled per episode and camera")
    ap.add_argument("--cams", default="front,wrist")
    ap.add_argument("--depth_w", type=float, default=0.6); ap.add_argument("--seg_w", type=float, default=0.6); ap.add_argument("--edge_w", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")))
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True); n = 0
    for ep_dir in sorted(glob.glob(os.path.join(a.export, "ep*"))):
        meta = json.load(open(os.path.join(ep_dir, "episode.json")))
        for cam in a.cams.split(","):
            for k, style in enumerate(rng.sample(P["styles"], a.styles)):
                name = f"{os.path.basename(ep_dir)}_{cam}_s{k}"
                spec = dict(name=name, prompt=f"{P['base']} {style} Task: {meta.get('text', '')}.",
                            video_path=os.path.abspath(os.path.join(ep_dir, f"{cam}_rgb.mp4")), output_dir=os.path.abspath(os.path.join(a.out, "outputs", name)),
                            guidance=3, depth=dict(control_path=os.path.abspath(os.path.join(ep_dir, f"{cam}_depth.mp4")), control_weight=a.depth_w),
                            seg=dict(control_path=os.path.abspath(os.path.join(ep_dir, f"{cam}_seg.mp4")), control_weight=a.seg_w),
                            edge=dict(control_weight=a.edge_w))
                json.dump(spec, open(os.path.join(a.out, name + ".json"), "w"), indent=1); n += 1
    print(n, "specs ->", a.out)


if __name__ == "__main__":
    main()
