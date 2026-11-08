"""Unify downloaded LeRobot v3 SO-101 datasets (in place) so lerobot's aggregate_datasets can merge them:
  wrist camera -> observation.images.camera1, the overview camera -> observation.images.camera2
  same robot_type, same state/action names, same video feature metadata; other features must already match.
Then aggregates them into one dataset.

    ~/VLA/envs/lerobot/bin/python tools/hf_unify.py --out ~/VLA/data/lerobot/so101_real_v1 ~/VLA/data/hf/<owner>__<name> ...
"""
import argparse, glob, json, os, shutil
import pandas as pd

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
WRIST = ("wrist", "hand", "gripper", "handeye")


def unify(root, canon_video=None):
    info_p = os.path.join(root, "meta", "info.json"); info = json.load(open(info_p)); F = info["features"]
    cams = [k for k, v in F.items() if v.get("dtype") == "video"]
    wr = [c for c in cams if any(w in c.split(".")[-1].lower() for w in WRIST)]; ov = [c for c in cams if c not in wr]
    if cams and set(cams) <= {"observation.images.camera1", "observation.images.camera2"}:
        return info                                                # already unified
    assert len(wr) == 1 and len(ov) == 1, f"{root}: cameras {cams}"
    ren = {wr[0]: "observation.images.camera1", ov[0]: "observation.images.camera2"}
    for old, new in ren.items():
        assert F[old]["shape"][:2] == [480, 640], f"{root}: {old} shape {F[old]['shape']}"
        F[new] = F.pop(old)
        if canon_video is not None:
            F[new]["info"] = canon_video
        if os.path.isdir(os.path.join(root, "videos", old)):
            os.rename(os.path.join(root, "videos", old), os.path.join(root, "videos", new))
    for k in ("observation.state", "action"):
        F[k]["names"] = NAMES
    info["robot_type"] = "so101_follower"
    info["features"] = {k: F[k] for k in sorted(F)}
    json.dump(info, open(info_p, "w"), indent=4)
    st_p = os.path.join(root, "meta", "stats.json")
    if os.path.exists(st_p):
        st = json.load(open(st_p)); json.dump({ren.get(k, k): v for k, v in st.items()}, open(st_p, "w"), indent=4)
    for f in glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")):
        df = pd.read_parquet(f); cols = {}
        for c in df.columns:
            for old, new in ren.items():
                if c.startswith(f"videos/{old}/") or c.startswith(f"stats/{old}/"):
                    cols[c] = c.replace(old, new, 1)
        df.rename(columns=cols).to_parquet(f)
    return info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("roots", nargs="+"); a = ap.parse_args()
    infos = [unify(r) for r in a.roots]
    import re
    for r in a.roots:                                              # some uploads carry stale meta/episodes/{chunk,file}_index: point them at the file they live in
        for f in glob.glob(os.path.join(r, "meta", "episodes", "*", "*.parquet")):
            c, i = (int(x) for x in re.findall(r"chunk-(\d+)/file-(\d+)", f)[0])
            df = pd.read_parquet(f)
            if "meta/episodes/chunk_index" in df and ((df["meta/episodes/chunk_index"] != c).any() or (df["meta/episodes/file_index"] != i).any()):
                df["meta/episodes/chunk_index"] = c; df["meta/episodes/file_index"] = i; df.to_parquet(f); print(json.dumps(dict(fixed_meta_index=f)), flush=True)
    canon = infos[0]["features"]["observation.images.camera1"].get("info")
    for r in a.roots:                                              # identical video metadata everywhere (decoding reads the file itself)
        p = os.path.join(r, "meta", "info.json"); info = json.load(open(p))
        for c in ("observation.images.camera1", "observation.images.camera2"):
            info["features"][c]["info"] = canon
        json.dump(info, open(p, "w"), indent=4)
    ref = json.load(open(os.path.join(a.roots[0], "meta", "info.json")))["features"]
    for r in a.roots:
        f = json.load(open(os.path.join(r, "meta", "info.json")))["features"]
        if f != ref:
            raise SystemExit(f"feature mismatch in {r}: {set(f) ^ set(ref)} / {[k for k in f if k in ref and f[k] != ref[k]]}")
    from lerobot.datasets.aggregate import aggregate_datasets
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    aggregate_datasets([os.path.basename(r) for r in a.roots], "local/so101_real", roots=[os.path.abspath(r) for r in a.roots], aggr_root=a.out)
    info = json.load(open(os.path.join(a.out, "meta", "info.json")))
    print(json.dumps(dict(done=a.out, episodes=info["total_episodes"], frames=info["total_frames"], tasks=info["total_tasks"])), flush=True)


if __name__ == "__main__":
    main()
