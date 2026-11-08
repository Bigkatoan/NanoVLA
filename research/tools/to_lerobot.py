"""Convert NanoVLA sim chunks (frames/*.parquet + actions/*.npz) into a LeRobotDataset (v3) for SmolVLA fine-tuning.

Frames are recorded every 3 control steps (30 Hz control) -> fps 10. Per frame:
  observation.images.camera1 : RGB, resized to --hw (default 240x320)
  observation.state        : the 6 joint positions at that control step (rad)
  action                   : the expert's joint target at that step (rad)
  task                     : the instruction text

    ~/VLA/envs/lerobot/bin/python tools/to_lerobot.py --out ~/VLA/data/lerobot/sim_v1 --src ~/VLA/data/isaac_s/*_c* ~/VLA/data/isaac_gct/*_c*
"""
import argparse, glob, io, json, os, shutil, time
import numpy as np, pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--hw", default="240x320"); ap.add_argument("--max_eps", type=int, default=0); ap.add_argument("--repo_id", default="local/nanovla_sim")
    ap.add_argument("--scale", type=float, default=1.0, help="joint unit factor, 57.29578 = radians -> degrees (the real SO-101 convention)")
    ap.add_argument("--cams", default="wrist", help="wrist | wrist,front  (wrist -> camera1, front -> camera2); episodes missing a camera are skipped")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    H, W = (int(x) for x in a.hw.split("x"))
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    CAMS = a.cams.split(","); SLOT = {"wrist": "observation.images.camera1", "front": "observation.images.camera2"}
    feats = {SLOT[c]: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]} for c in CAMS}
    feats.update({
             "observation.state": {"dtype": "float32", "shape": (6,), "names": ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "jaw"]},
             "action": {"dtype": "float32", "shape": (6,), "names": ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "jaw"]}})
    ds = LeRobotDataset.create(a.repo_id, fps=10, features=feats, root=a.out, robot_type="so101_sim", use_videos=True, image_writer_threads=4)
    n_ep, n_fr, t0 = 0, 0, time.time()
    for src in sorted(d for d in a.src if os.path.exists(os.path.join(d, "episodes.json"))):
        meta = {e["ep"]: e for e in json.load(open(os.path.join(src, "episodes.json")))["episodes"]}
        A = {}
        for f in glob.glob(os.path.join(src, "actions", "*.npz")):
            z = np.load(f); A.update({k: z[k] for k in z.files})
        for pq in glob.glob(os.path.join(src, "frames", "*.parquet")):
            df = pd.read_parquet(pq, columns=["ep", "cam", "fidx", "jpeg", "task"])
            df = df[df.cam.isin(CAMS)]
            for e, g in df.groupby("ep"):
                e = int(e)
                if f"ep{e}_action" not in A or (meta.get(e, {}).get("success_check") is False) or set(g.cam) != set(CAMS):
                    continue
                act, st = A[f"ep{e}_action"], A[f"ep{e}_state"]
                byf = {}
                for c_, f_, jpg, txt in zip(g.cam, g.fidx, g.jpeg, g.task):
                    byf.setdefault(int(f_), {})[c_] = jpg; byf[int(f_)]["_t"] = txt
                k0 = n_fr
                for f_ in sorted(byf):
                    if f_ >= len(act) or any(c not in byf[f_] for c in CAMS):
                        continue
                    fr = {"observation.state": (st[f_] * a.scale).astype(np.float32), "action": (act[f_] * a.scale).astype(np.float32), "task": str(byf[f_]["_t"])}
                    for c in CAMS:
                        im = Image.open(io.BytesIO(byf[f_][c])).convert("RGB")
                        if im.size != (W, H):
                            im = im.resize((W, H), Image.BICUBIC)
                        fr[SLOT[c]] = np.asarray(im)
                    ds.add_frame(fr); n_fr += 1
                if n_fr == k0:
                    continue
                ds.save_episode(); n_ep += 1
                if n_ep % 100 == 0:
                    print(json.dumps(dict(episodes=n_ep, frames=n_fr, s=round(time.time() - t0))), flush=True)
                if a.max_eps and n_ep >= a.max_eps:
                    break
            if a.max_eps and n_ep >= a.max_eps:
                break
        if a.max_eps and n_ep >= a.max_eps:
            break
    ds.finalize()
    print(json.dumps(dict(done=a.out, episodes=n_ep, frames=n_fr, s=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
