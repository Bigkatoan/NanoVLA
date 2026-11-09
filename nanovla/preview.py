"""Dataset previews for the web app: episode table, frames from the LeRobot v3 videos, joint trajectories, source.

Frames are decoded with PyAV at a fraction of the episode (cached as JPEG under jobs/.cache), so the app never loads a
whole dataset into memory.
"""
import glob, hashlib, io, json, os, threading
import numpy as np, pandas as pd
from nanovla import paths as P

CACHE = os.path.join(P.JOBS, ".cache", "frames"); os.makedirs(CACHE, exist_ok=True)
_eps, _lock = {}, threading.Lock()


def source(root):
    """where the data came from: recorded on the real robot, generated in Isaac Sim, or downloaded from the Hub"""
    p = os.path.join(root, "meta", "nanovla_source.json")
    if os.path.exists(p):
        return json.load(open(p))
    name = os.path.basename(root.rstrip("/"))
    if name.startswith("sim_"):
        return dict(kind="sim", label="Isaac Sim")
    if "__" in name:
        return dict(kind="hub", label="Hugging Face", repo=name.replace("__", "/", 1))
    if name in ("so101_deg_v1",):
        return dict(kind="hub", label="Hugging Face (5 datasets merged)")
    if name in ("ttt_v1",):
        return dict(kind="hub", label="Hugging Face", repo="fport/tic-tac-toe-so101-x-round1-clean-v1")
    return dict(kind="robot", label="Real robot")


def episodes(root):
    key = (root, os.path.getmtime(os.path.join(root, "meta", "info.json")))
    with _lock:
        if key not in _eps:
            fs = sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))
            cols = None; parts = []
            for f in fs:
                d = pd.read_parquet(f); keep = [c for c in d.columns if not c.startswith("stats/")]; parts.append(d[keep])
            _eps[key] = pd.concat(parts).sort_values("episode_index").set_index("episode_index") if parts else pd.DataFrame()
        return _eps[key]


def summary(root):
    info = json.load(open(os.path.join(root, "meta", "info.json"))); E = episodes(root); fps = info["fps"]
    L = E["length"].to_numpy() / fps if len(E) else np.zeros(0)
    hist, edges = np.histogram(L, bins=min(12, max(3, len(L) // 4))) if len(L) else (np.zeros(0), np.zeros(0))
    return dict(source=source(root), fps=fps, episodes=int(len(E)), hours=round(float(L.sum()) / 3600, 2), mean_s=round(float(L.mean()), 1) if len(L) else 0,
                min_s=round(float(L.min()), 1) if len(L) else 0, max_s=round(float(L.max()), 1) if len(L) else 0,
                length_hist=dict(counts=hist.astype(int).tolist(), edges=[round(float(x), 1) for x in edges]),
                cameras=[k for k, v in info["features"].items() if v.get("dtype") == "video"])


def episode_list(root, offset=0, limit=60):
    E = episodes(root); fps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]; out = []
    for e, r in E.iloc[offset: offset + limit].iterrows():
        t = r["tasks"]; out.append(dict(ep=int(e), task=str(t[0]) if len(t) else "", seconds=round(float(r["length"]) / fps, 1)))
    return dict(total=int(len(E)), items=out)


def frame(root, ep, cam, frac, width=320):
    """JPEG bytes of camera `cam` at fraction `frac` (0..1) of episode `ep`"""
    frac = float(min(max(frac, 0.0), 0.999)); k = hashlib.md5(f"{root}|{ep}|{cam}|{frac:.3f}|{width}".encode()).hexdigest()
    cp = os.path.join(CACHE, k + ".jpg")
    if os.path.exists(cp):
        return open(cp, "rb").read()
    info = json.load(open(os.path.join(root, "meta", "info.json"))); r = episodes(root).loc[int(ep)]
    key = f"observation.images.{cam}"
    path = os.path.join(root, info["video_path"].format(video_key=key, chunk_index=int(r[f"videos/{key}/chunk_index"]), file_index=int(r[f"videos/{key}/file_index"])))
    t0, t1 = float(r[f"videos/{key}/from_timestamp"]), float(r[f"videos/{key}/to_timestamp"]); ts = t0 + frac * (t1 - t0)
    import av
    from PIL import Image
    with av.open(path) as c:
        s = c.streams.video[0]; c.seek(int(max(0.0, ts - 1.0) / s.time_base), stream=s, backward=True)
        img = None
        for f in c.decode(s):
            img = f
            if f.time is not None and f.time >= ts - 0.5 / info["fps"]:
                break
    im = img.to_image(); im.thumbnail((width, width), Image.BILINEAR)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=82); data = b.getvalue(); open(cp, "wb").write(data)
    return data


def trajectory(root, ep, points=200):
    """state and action of one episode, downsampled, for the joint plot"""
    info = json.load(open(os.path.join(root, "meta", "info.json"))); r = episodes(root).loc[int(ep)]
    f = os.path.join(root, info["data_path"].format(chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])))
    d = pd.read_parquet(f, columns=["episode_index", "observation.state", "action"]); d = d[d.episode_index == int(ep)]
    S, A = np.stack(d["observation.state"].to_numpy()), np.stack(d["action"].to_numpy())
    idx = np.linspace(0, len(S) - 1, min(points, len(S))).round().astype(int)
    names = [n.replace(".pos", "") for n in info["features"]["observation.state"].get("names") or [f"j{i}" for i in range(S.shape[1])]]
    return dict(fps=info["fps"], n=int(len(S)), t=[round(float(i) / info["fps"], 2) for i in idx], names=names,
                state=np.round(S[idx], 2).T.tolist(), action=np.round(A[idx], 2).T.tolist(), task=str(r["tasks"][0]) if len(r["tasks"]) else "")
