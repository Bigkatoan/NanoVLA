"""Pack per-repo frame shards into one JPEG blob + index for fast memmap access.

Outputs (data/packed/):
  frames.bin       concatenated JPEG bytes
  index.parquet    one row per frame: repo_id, ep, cam, wrist, fidx, off, len, state[6], action[6], task_id, seq, pos, partner
  repos.parquet    repo_id, repo, n_frames, dstate_std[6]  (std of per-step state delta, for IDM normalisation)
  tasks.parquet    task_id, task, task_orig
"""
import glob, os, numpy as np, pandas as pd

ROOT = os.path.expanduser("~/VLA/data")
OUT = os.path.join(ROOT, "packed")
WRIST_KEYS = ("wrist", "gripper", "hand", "on_robot", "arm")


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "frames/*.parquet")))
    blob = open(os.path.join(OUT, "frames.bin"), "wb")
    off = 0
    idx, repos, task_ids = [], [], {}
    for rid, f in enumerate(files):
        d = pd.read_parquet(f)
        repo = os.path.basename(f)[:-8].replace("__", "/", 1)
        d = d.sort_values(["ep", "cam", "fidx"]).reset_index(drop=True)
        lens = d.jpeg.map(len).to_numpy()
        for b in d.jpeg:
            blob.write(b)
        offs = off + np.concatenate([[0], np.cumsum(lens)[:-1]])
        off += int(lens.sum())
        st = np.stack(d.state.to_numpy()).astype(np.float32)
        ac = np.stack(d.action.to_numpy()).astype(np.float32)
        # per-step state deltas within (ep, cam) sequences -> robust std per joint
        same = (d.ep.values[1:] == d.ep.values[:-1]) & (d.cam.values[1:] == d.cam.values[:-1])
        ds = (st[1:] - st[:-1])[same]
        dstd = (np.percentile(np.abs(ds), 90, axis=0) + 1e-3) if len(ds) else np.ones(6, np.float32)
        tids = []
        for t, o in zip(d.task, d.task_orig):
            k = (t, o)
            if k not in task_ids:
                task_ids[k] = len(task_ids)
            tids.append(task_ids[k])
        e = pd.DataFrame(dict(repo_id=rid, ep=d.ep.values, cam=d.cam.values, fidx=d.fidx.values, off=offs, len=lens, task_id=tids))
        e["wrist"] = e.cam.map(lambda c: any(k in c.lower() for k in WRIST_KEYS)).astype(np.int8)
        e["state"] = list(st)
        e["action"] = list(ac)
        idx.append(e)
        repos.append(dict(repo_id=rid, repo=repo, n_frames=len(d), fps=int(d.fps.iloc[0]), dstate_std=dstd.astype(np.float32)))
        if rid % 100 == 0:
            print(rid, len(files), off / 1e9, "GB", flush=True)
    blob.close()
    I = pd.concat(idx, ignore_index=True)
    # sequence id per (repo, ep, cam) and position within it
    I["seq"] = I.groupby(["repo_id", "ep", "cam"], sort=False).ngroup()
    I["pos"] = I.groupby("seq").cumcount()
    I["seq_len"] = I.groupby("seq").fidx.transform("size")
    # partner = same (repo, ep, fidx) other camera, -1 if none
    I["row"] = np.arange(len(I))
    key = I[["repo_id", "ep", "fidx", "cam", "row"]]
    m = key.merge(key, on=["repo_id", "ep", "fidx"], suffixes=("", "_o"))
    m = m[m.cam != m.cam_o].drop_duplicates("row")
    I["partner"] = -1
    I.loc[m.row.values, "partner"] = m.row_o.values
    I = I.drop(columns=["row"])
    I.to_parquet(os.path.join(OUT, "index.parquet"))
    pd.DataFrame(repos).to_parquet(os.path.join(OUT, "repos.parquet"))
    pd.DataFrame([dict(task_id=v, task=k[0], task_orig=k[1]) for k, v in task_ids.items()]).to_parquet(os.path.join(OUT, "tasks.parquet"))
    print("frames", len(I), "repos", len(repos), "tasks", len(task_ids), "blob GB %.2f" % (off / 1e9), "with partner %.2f" % (I.partner >= 0).mean())


if __name__ == "__main__":
    main()
