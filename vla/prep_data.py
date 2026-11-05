"""Download a selected subset of SO-100/101 LeRobot v2.1 repos (MolmoAct2 manifest),
decode videos to small JPEG frames, and write one parquet shard per repo.

Shard columns: ep, cam, fidx, jpeg, state[6], action[6], task (annotated), task_orig
"""
import argparse, json, os, random, shutil, time, io, glob, traceback
import multiprocessing as mp
import numpy as np, pandas as pd
import av
from PIL import Image
from huggingface_hub import snapshot_download

ROOT = os.path.expanduser("~/VLA/data")
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "frames")
W, H = 160, 120
TARGET_FPS = 5
WRIST_KEYS = ("wrist", "gripper", "hand", "on_robot", "arm")


def pick_cams(cams):
    wrist = [c for c in cams if any(k in c.split(".")[-1].lower() for k in WRIST_KEYS)]
    ext = [c for c in cams if c not in wrist]
    out = ext[:1] + wrist[:1]
    if not out:
        out = cams[:1]
    if len(out) < 2 and len(cams) > 1:
        out = (out + [c for c in cams if c not in out])[:2]
    return out


def pick_eps(ann, total_eps, cap, seed):
    rng = random.Random(seed)
    eps = list(range(total_eps))
    if ann is None or len(ann) == 0:
        rng.shuffle(eps)
        return sorted(eps[:cap])
    ann = ann[ann.index < total_eps]
    chosen = []
    for _, g in ann.groupby("task"):
        chosen.append(rng.choice(list(g.index)))
    rng.shuffle(chosen)
    chosen = chosen[:cap]
    rest = [e for e in eps if e not in set(chosen)]
    rng.shuffle(rest)
    chosen += rest[: max(0, cap - len(chosen))]
    return sorted(chosen)


def decode_video(path, stride):
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            if i % stride:
                continue
            img = fr.to_image().resize((W, H), Image.BILINEAR)
            b = io.BytesIO()
            img.save(b, format="JPEG", quality=90)
            frames.append((i, b.getvalue()))
    return frames


def process(args):
    repo, cap = args
    name = repo.replace("/", "__")
    out = os.path.join(OUT, name + ".parquet")
    if os.path.exists(out):
        return repo, "cached"
    try:
        info = json.load(open(os.path.join(ROOT, "info", name + ".json")))
        if info.get("codebase_version") != "v2.1":
            return repo, "skip_ver"
        cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        if not cams:
            return repo, "skip_nocam"
        cams = pick_cams(cams)
        annf = os.path.join(ROOT, "meta/molmoact2/language_annotations", repo, "tasks_annotated.parquet")
        ann = pd.read_parquet(annf) if os.path.exists(annf) else None
        eps = pick_eps(ann, info["total_episodes"], cap, hash(repo) & 0xFFFF)
        cs = info.get("chunks_size", 1000)
        pats = ["meta/tasks.jsonl"]
        for e in eps:
            ch = e // cs
            pats.append(info["data_path"].format(episode_chunk=ch, episode_index=e))
            for c in cams:
                pats.append(info["video_path"].format(episode_chunk=ch, video_key=c, episode_index=e))
        local = os.path.join(RAW, name)
        for k in range(10):
            try:
                snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=local, max_workers=4)
                break
            except Exception as ex:
                s = repr(ex)
                if any(x in s for x in ("404 Client Error", "403 Client Error", "RepositoryNotFound", "GatedRepo")):
                    raise
                if k == 9:
                    raise
                time.sleep(min(300, 30 * (k + 1)) + random.random() * 10)
        # original tasks
        torig = {}
        tp = os.path.join(local, "meta/tasks.jsonl")
        if os.path.exists(tp):
            for line in open(tp):
                j = json.loads(line)
                torig[j["task_index"]] = j["task"]
        stride = max(1, round(info["fps"] / TARGET_FPS))
        rows = []
        for e in eps:
            ch = e // cs
            dp = os.path.join(local, info["data_path"].format(episode_chunk=ch, episode_index=e))
            if not os.path.exists(dp):
                continue
            df = pd.read_parquet(dp)
            st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            ac = np.stack(df["action"].to_numpy()).astype(np.float32) if "action" in df else st
            ti = int(df["task_index"].iloc[0]) if "task_index" in df else -1
            task_o = torig.get(ti, "")
            task_a = ann.loc[e, "task"] if (ann is not None and e in ann.index) else task_o
            for c in cams:
                vp = os.path.join(local, info["video_path"].format(episode_chunk=ch, video_key=c, episode_index=e))
                if not os.path.exists(vp):
                    continue
                for fidx, jpg in decode_video(vp, stride):
                    if fidx >= len(st):
                        break
                    rows.append((e, c.split(".")[-1], fidx, jpg, st[fidx].tolist(), ac[fidx].tolist(), task_a, task_o))
        shutil.rmtree(local, ignore_errors=True)
        if not rows:
            return repo, "empty"
        d = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "state", "action", "task", "task_orig"])
        d["fps"] = info["fps"]
        d["robot"] = info.get("robot_type", "")
        tmp = out + ".tmp"
        d.to_parquet(tmp)
        os.replace(tmp, out)
        return repo, f"ok {len(eps)}ep {len(d)}fr"
    except Exception as ex:
        shutil.rmtree(os.path.join(RAW, name), ignore_errors=True)
        return repo, "ERR " + repr(ex)[:200]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    ri = pd.read_parquet(os.path.join(ROOT, "meta/repos_info.parquet"))
    repos = ri[ri.ver == "v2.1"].sort_values("robot", ascending=False).repo.tolist()  # so101 first
    if a.limit:
        repos = repos[: a.limit]
    log = open(os.path.join(ROOT, "prep_log.txt"), "a")
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, (r, s) in enumerate(pool.imap_unordered(process, [(r, a.cap) for r in repos])):
            log.write(f"{r}\t{s}\n"); log.flush()
            if i % 20 == 0:
                print(f"[{i+1}/{len(repos)}] {time.time()-t0:.0f}s {r} {s}", flush=True)
    print("done", time.time() - t0)
