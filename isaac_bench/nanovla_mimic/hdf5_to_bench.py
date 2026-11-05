"""Convert an Isaac Lab Mimic dataset (HDF5: data/demo_k/{actions, obs/...}) into the NanoVLA few-shot bench format.

state  = obs/joint_pos (rad, 6), action = obs/joint_target (the joint targets the differential-IK action produced),
frames = obs/front, obs/wrist (160x120) every 3rd control step, instruction from the env config (task/obj/target).
Only demos marked successful are converted.

    python nanovla_mimic/hdf5_to_bench.py --input datasets/tray_marker_gen.hdf5 --text "put the marker in the tray" \
        --out ~/VLA/data/mimic/tray_marker
"""
import argparse, io, json, os
import h5py, numpy as np, pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True); ap.add_argument("--text", required=True)
    ap.add_argument("--name", default="mimic/tabletop")
    a = ap.parse_args()
    rows, acts, e = [], {}, 0
    with h5py.File(a.input, "r") as f:
        for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1])):
            d = f["data"][k]
            if "success" in d.attrs and not bool(d.attrs["success"]):
                continue
            o = d["obs"]
            S = np.asarray(o["joint_pos"], np.float32); A = np.asarray(o["joint_target"], np.float32)
            fr, wr = o["front"], o["wrist"]
            for t in range(0, len(S), 3):
                for cam, arr in (("front", fr), ("wrist", wr)):
                    b = io.BytesIO(); Image.fromarray(np.asarray(arr[t], np.uint8)).save(b, format="JPEG", quality=90)
                    rows.append((e, cam, t, b.getvalue(), a.text, a.text))
            acts[f"ep{e}_action"] = A; acts[f"ep{e}_state"] = S; e += 1
    name = a.name.replace("/", "__")
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True); os.makedirs(os.path.join(a.out, "actions"), exist_ok=True)
    D = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig"]); D["fps"] = 30
    D.to_parquet(os.path.join(a.out, "frames", name + ".parquet"))
    np.savez_compressed(os.path.join(a.out, "actions", name + ".npz"), **acts)
    json.dump([a.name], open(os.path.join(a.out, "repos.json"), "w"))
    print(e, "episodes ->", a.out)


if __name__ == "__main__":
    main()
