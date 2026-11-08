"""Dataset preparation: one camera layout and one metadata convention for every LeRobot v3 SO-101 dataset.

  prepare(root)          in place: wrist camera -> observation.images.camera1, overview -> observation.images.camera2,
                         robot_type / joint names unified, stale episode-metadata indices fixed. Joint values untouched.
  merge(roots, out)      prepare each, then lerobot's aggregate_datasets into one dataset
  describe(root)         episodes, frames, tasks, cameras, fps, joint-unit guess, per-task episode lists
  default_split(desc)    hold out one episode per task (the last one) for testing, train on the rest

    python -m nanovla.merge --out data/lerobot/teacher_mix data/hf/<owner>__<name> ...
"""
import argparse, glob, json, os, re, shutil
import numpy as np, pandas as pd

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
WRIST = ("wrist", "hand", "gripper", "handeye")
CAM1, CAM2 = "observation.images.camera1", "observation.images.camera2"


def _info(root):
    return json.load(open(os.path.join(root, "meta", "info.json")))


def _episodes(root):
    fs = sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs]).sort_values("episode_index") if fs else pd.DataFrame()


def prepare(root):
    info_p = os.path.join(root, "meta", "info.json"); info = json.load(open(info_p)); F = info["features"]
    cams = [k for k, v in F.items() if v.get("dtype") == "video"]
    if not (cams and set(cams) <= {CAM1, CAM2}):
        wr = [c for c in cams if any(w in c.split(".")[-1].lower() for w in WRIST)]; ov = [c for c in cams if c not in wr]
        if len(wr) != 1 or len(ov) != 1:
            raise ValueError(f"{root}: need exactly one wrist camera and one overview camera, found {cams}")
        ren = {wr[0]: CAM1, ov[0]: CAM2}
        for old, new in ren.items():
            F[new] = F.pop(old)
            if os.path.isdir(os.path.join(root, "videos", old)):
                os.rename(os.path.join(root, "videos", old), os.path.join(root, "videos", new))
        st_p = os.path.join(root, "meta", "stats.json")
        if os.path.exists(st_p):
            st = json.load(open(st_p)); json.dump({ren.get(k, k): v for k, v in st.items()}, open(st_p, "w"), indent=4)
        for f in glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")):
            df = pd.read_parquet(f)
            cols = {c: c.replace(o, n, 1) for c in df.columns for o, n in ren.items() if c.startswith(f"videos/{o}/") or c.startswith(f"stats/{o}/")}
            df.rename(columns=cols).to_parquet(f)
    for k in ("observation.state", "action"):
        F[k]["names"] = NAMES
    info["robot_type"] = "so101_follower"; info["features"] = {k: F[k] for k in sorted(F)}
    json.dump(info, open(info_p, "w"), indent=4)
    for f in glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")):     # stale meta/episodes indices (seen in public uploads)
        c, i = (int(x) for x in re.findall(r"chunk-(\d+)/file-(\d+)", f)[0]); df = pd.read_parquet(f)
        if "meta/episodes/chunk_index" in df and ((df["meta/episodes/chunk_index"] != c).any() or (df["meta/episodes/file_index"] != i).any()):
            df["meta/episodes/chunk_index"] = c; df["meta/episodes/file_index"] = i; df.to_parquet(f)
    return info


def merge(roots, out):
    infos = [prepare(r) for r in roots]
    canon = infos[0]["features"][CAM1].get("info")
    for r in roots:                                                # identical video metadata (the decoder reads the file itself)
        p = os.path.join(r, "meta", "info.json"); info = json.load(open(p))
        for c in (CAM1, CAM2):
            if info["features"][c]["shape"][:2] != [480, 640]:
                raise ValueError(f"{r}: {c} is {info['features'][c]['shape']}, all merged datasets must be 480x640")
            info["features"][c]["info"] = canon
        json.dump(info, open(p, "w"), indent=4)
    ref = _info(roots[0])["features"]
    for r in roots:
        f = _info(r)["features"]
        if f != ref:
            raise ValueError(f"feature mismatch in {r}: {sorted(set(f) ^ set(ref))}")
    from lerobot.datasets.aggregate import aggregate_datasets
    if os.path.exists(out):
        shutil.rmtree(out)
    aggregate_datasets([os.path.basename(r) for r in roots], "local/merged", roots=[os.path.abspath(r) for r in roots], aggr_root=out)
    return describe(out)


def describe(root):
    info = _info(root); F = info["features"]; E = _episodes(root)
    by_task = {}
    for e, t in zip(E.get("episode_index", []), E.get("tasks", [])):
        t = t[0] if len(t) else ""; by_task.setdefault(str(t), []).append(int(e))
    unit = "unknown"
    try:
        s = json.load(open(os.path.join(root, "meta", "stats.json")))["observation.state"]
        mx = float(np.abs(np.r_[np.ravel(s["min"])[:5], np.ravel(s["max"])[:5]]).max())
        unit = "radians" if mx < 4 else ("degrees" if mx > 105 else "degrees or -100..100")
    except Exception:
        pass
    return dict(name=os.path.basename(root.rstrip("/")), root=root, episodes=info.get("total_episodes", 0), frames=info.get("total_frames", 0),
                fps=info.get("fps"), cameras={k: v["shape"] for k, v in F.items() if v.get("dtype") == "video"}, joint_unit=unit,
                tasks={t: len(v) for t, v in by_task.items()}, episodes_by_task=by_task,
                ready=set(k for k, v in F.items() if v.get("dtype") == "video") == {CAM1, CAM2})


def default_split(desc):
    """test = the last episode of every task that has >= 2 episodes; train = everything else"""
    test = sorted(v[-1] for v in desc["episodes_by_task"].values() if len(v) >= 2)
    train = sorted(e for v in desc["episodes_by_task"].values() for e in v if e not in test)
    return train, test


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("roots", nargs="+"); a = ap.parse_args()
    print(json.dumps({k: v for k, v in merge(a.roots, a.out).items() if k != "episodes_by_task"}))


if __name__ == "__main__":
    main()
