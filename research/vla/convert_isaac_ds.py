"""Convert a LeRobot v3.0 dataset (Isaac SO-101 workshop teleop, 1 file per episode) into the few-shot bench format:
  <out>/frames/<name>.parquet  (ep, cam, fidx, jpeg, task, task_orig, fps) -- every 3rd frame (10 Hz in frame units), 160x120
  <out>/actions/<name>.npz     (ep<k>_action, ep<k>_state; LeRobot degree units, gripper 0..100)
  <out>/repos.json
Cameras: external_D455 -> 'front', ego -> 'wrist' (wrist keyword is what fewshot.prepare uses to find the wrist cam).
"""
import argparse, glob, io, json, os
import av, numpy as np, pandas as pd
from PIL import Image

CAMS = {"observation.images.external_D455": "front", "observation.images.ego": "wrist"}


def frames(path, every=3, size=(160, 120)):
    out = []
    with av.open(path) as c:
        for k, f in enumerate(c.decode(video=0)):
            if k % every == 0:
                b = io.BytesIO(); f.to_image().resize(size, Image.BILINEAR).save(b, format="JPEG", quality=90)
                out.append((k, b.getvalue()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/VLA/data/isaac/so101_teleop_vials_rack_left"))
    ap.add_argument("--out", default=os.path.expanduser("~/VLA/data/isaac_bench"))
    ap.add_argument("--name", default="isaac/so101_vials_rack_left")
    a = ap.parse_args()
    E = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(a.src, "meta/episodes/*/*.parquet"))]).sort_values("episode_index")
    rows, acts = [], {}
    for _, R in E.iterrows():
        e = int(R["episode_index"])
        d = pd.read_parquet(os.path.join(a.src, f"data/chunk-{R['data/chunk_index']:03d}/file-{R['data/file_index']:03d}.parquet"))
        d = d[d.episode_index == e].sort_values("frame_index")
        acts[f"ep{e}_action"] = np.stack(d.action).astype(np.float32); acts[f"ep{e}_state"] = np.stack(d["observation.state"]).astype(np.float32)
        task = R["tasks"][0] if len(R["tasks"]) else ""
        for key, cam in CAMS.items():
            vp = os.path.join(a.src, f"videos/{key}/chunk-{R[f'videos/{key}/chunk_index']:03d}/file-{R[f'videos/{key}/file_index']:03d}.mp4")
            assert R[f"videos/{key}/from_timestamp"] == 0.0, "expects one episode per video file"
            for k, jp in frames(vp):
                if k < len(d):
                    rows.append((e, cam, k, jp, task, task))
        print(e, len(d), flush=True)
    name = a.name.replace("/", "__")
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True); os.makedirs(os.path.join(a.out, "actions"), exist_ok=True)
    F = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig"]); F["fps"] = 30
    F.to_parquet(os.path.join(a.out, "frames", name + ".parquet"))
    np.savez_compressed(os.path.join(a.out, "actions", name + ".npz"), **acts)
    json.dump([a.name], open(os.path.join(a.out, "repos.json"), "w"))
    print("episodes", len(E), "frames", len(F))


if __name__ == "__main__":
    main()
