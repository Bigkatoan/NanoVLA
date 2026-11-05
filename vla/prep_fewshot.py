"""Two jobs, both network-bound (run while the GPU trains):

  --mode actions : for every pretraining repo, re-download the per-episode data parquets of the episodes already
                   in data/frames/ and store full-rate state/action -> data/actions/<repo>.npz  (keys ep<k>_state, ep<k>_action)
  --mode bench   : for held-out few-shot benchmark repos, download ALL episodes, decode frames at 10 fps for up to 2 cams,
                   store frames shard (data/bench/frames/<repo>.parquet) + full-rate actions (data/bench/actions/<repo>.npz)
"""
import argparse, json, os, random, shutil, time, io, glob
import multiprocessing as mp
import numpy as np, pandas as pd
from huggingface_hub import snapshot_download
from prep_data import pick_cams, decode_video, ROOT

RAW = os.path.join(ROOT, "raw_fs")


def dl(repo, pats, local):
    for k in range(10):
        try:
            snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=local, max_workers=4)
            return
        except Exception as ex:
            s = repr(ex)
            if any(x in s for x in ("404 Client Error", "403 Client Error", "RepositoryNotFound", "GatedRepo")) or k == 9:
                raise
            time.sleep(min(300, 30 * (k + 1)) + random.random() * 10)


def load_ep(local, info, e):
    cs = info.get("chunks_size", 1000)
    dp = os.path.join(local, info["data_path"].format(episode_chunk=e // cs, episode_index=e))
    if not os.path.exists(dp):
        return None
    df = pd.read_parquet(dp)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    ac = np.stack(df["action"].to_numpy()).astype(np.float32) if "action" in df else st
    ti = int(df["task_index"].iloc[0]) if "task_index" in df else -1
    return st, ac, ti


def job_actions(repo):
    name = repo.replace("/", "__")
    out = os.path.join(ROOT, "actions", name + ".npz")
    if os.path.exists(out):
        return repo, "cached"
    try:
        info = json.load(open(os.path.join(ROOT, "info", name + ".json")))
        eps = sorted(pd.read_parquet(os.path.join(ROOT, "frames", name + ".parquet"), columns=["ep"]).ep.unique().tolist())
        cs = info.get("chunks_size", 1000)
        pats = [info["data_path"].format(episode_chunk=e // cs, episode_index=e) for e in eps]
        local = os.path.join(RAW, name)
        for attempt in range(8):  # snapshot_download silently returns a partial dir on 429 -> verify completeness
            dl(repo, pats, local)
            d = {}
            for e in eps:
                r = load_ep(local, info, e)
                if r is not None:
                    d[f"ep{e}_state"], d[f"ep{e}_action"] = r[0], r[1]
            if len(d) == 2 * len(eps):
                break
            time.sleep(min(300, 45 * (attempt + 1)) + random.random() * 10)
        else:
            raise RuntimeError(f"incomplete {len(d)//2}/{len(eps)}")
        shutil.rmtree(local, ignore_errors=True)
        np.savez_compressed(out + ".tmp.npz", **d)
        os.replace(out + ".tmp.npz", out)
        return repo, f"ok {len(eps)}"
    except Exception as ex:
        shutil.rmtree(os.path.join(RAW, name), ignore_errors=True)
        return repo, "ERR " + repr(ex)[:150]


def job_bench(repo):
    name = repo.replace("/", "__")
    out = os.path.join(ROOT, "bench/frames", name + ".parquet")
    if os.path.exists(out):
        return repo, "cached"
    try:
        info = json.load(open(os.path.join(ROOT, "info", name + ".json")))
        cams = pick_cams([k for k, v in info["features"].items() if v.get("dtype") == "video"])
        eps = list(range(info["total_episodes"]))
        cs = info.get("chunks_size", 1000)
        pats = ["meta/tasks.jsonl", "meta/episodes.jsonl"]
        for e in eps:
            pats.append(info["data_path"].format(episode_chunk=e // cs, episode_index=e))
            for c in cams:
                pats.append(info["video_path"].format(episode_chunk=e // cs, video_key=c, episode_index=e))
        local = os.path.join(RAW, name)
        dl(repo, pats, local)
        annf = os.path.join(ROOT, "meta/molmoact2/language_annotations", repo, "tasks_annotated.parquet")
        ann = pd.read_parquet(annf) if os.path.exists(annf) else None
        torig = {}
        tp = os.path.join(local, "meta/tasks.jsonl")
        if os.path.exists(tp):
            for line in open(tp):
                j = json.loads(line); torig[j["task_index"]] = j["task"]
        stride = max(1, round(info["fps"] / 10))
        rows, acts = [], {}
        for e in eps:
            r = load_ep(local, info, e)
            if r is None:
                continue
            st, ac, ti = r
            acts[f"ep{e}_state"], acts[f"ep{e}_action"] = st, ac
            task_o = torig.get(ti, "")
            task_a = ann.loc[e, "task"] if (ann is not None and e in ann.index) else task_o
            for c in cams:
                vp = os.path.join(local, info["video_path"].format(episode_chunk=e // cs, video_key=c, episode_index=e))
                if not os.path.exists(vp):
                    continue
                for fidx, jpg in decode_video(vp, stride):
                    if fidx >= len(st):
                        break
                    rows.append((e, c.split(".")[-1], fidx, jpg, task_a, task_o))
        shutil.rmtree(local, ignore_errors=True)
        d = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig"])
        d["fps"] = info["fps"]
        np.savez_compressed(os.path.join(ROOT, "bench/actions", name + ".npz"), **acts)
        d.to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)
        return repo, f"ok {len(acts)//2}ep {len(d)}fr"
    except Exception as ex:
        shutil.rmtree(os.path.join(RAW, name), ignore_errors=True)
        return repo, "ERR " + repr(ex)[:150]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["actions", "bench"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min_eps", type=int, default=50)
    a = ap.parse_args()
    for d in ["actions", "bench/frames", "bench/actions"]:
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    R = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet"))
    val = set(json.load(open(os.path.expanduser("~/VLA/ckpt/vision/val_repos.json"))))
    if a.mode == "actions":
        repos, fn = R.repo.tolist(), job_actions
    else:
        info = pd.read_parquet(os.path.join(ROOT, "meta/repos_info.parquet"))
        v = R[R.repo_id.isin(val)].merge(info, on="repo")
        repos, fn = v[v.eps >= a.min_eps].repo.tolist(), job_bench
        json.dump(repos, open(os.path.join(ROOT, "bench/repos.json"), "w"))
    log = open(os.path.join(ROOT, f"fewshot_{a.mode}_log.txt"), "a")
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, (r, s) in enumerate(pool.imap_unordered(fn, repos)):
            log.write(f"{r}\t{s}\n"); log.flush()
            if i % 50 == 0:
                print(f"[{i+1}/{len(repos)}] {time.time()-t0:.0f}s {r} {s}", flush=True)
    print("done", a.mode, time.time() - t0, flush=True)
