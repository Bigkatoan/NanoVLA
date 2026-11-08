"""Pick public SO-101 LeRobot datasets that share one joint convention (actuator angle in degrees) for the teacher.

1. search the Hub for SO-101 datasets; keep v3.0, 30 fps, 6-dof action, exactly one wrist camera + one 480x640 overview camera
2. unit mode from meta/stats.json: radians (|q| < 4), degrees (some arm joint beyond +-105), otherwise the -100..100 range mode
3. among the degree datasets keep those whose per-joint means are within --max_dev degrees of the degree-mode median
   (a different calibration zero or joint direction shows up as a large offset)

    python3 tools/hf_select.py --out data/hf/selected.json          # prints the selection; download with snapshot_download
"""
import argparse, json, urllib.request, concurrent.futures as cf
import numpy as np

HUB = "https://huggingface.co"
WRIST = ("wrist", "hand", "gripper", "handeye")


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--max_dev", type=float, default=25)
    ap.add_argument("--min_eps", type=int, default=20); ap.add_argument("--search", default="so101,so-101")
    ap.add_argument("--exclude", default="fport/tic-tac-toe", help="comma list of id prefixes kept out (held-out few-shot proxy tasks)"); a = ap.parse_args()
    ids = []
    for q in a.search.split(","):
        for d in get(f"{HUB}/api/datasets?search={q}&limit=1000&sort=downloads&direction=-1") or []:
            if d["id"] not in ids and not any(k in d["id"].lower() for k in ("sim", "mujoco", "isaac")) and not any(d["id"].startswith(x) for x in a.exclude.split(",") if x):
                ids.append(d["id"])

    def probe(i):
        info = get(f"{HUB}/datasets/{i}/resolve/main/meta/info.json"); st = get(f"{HUB}/datasets/{i}/resolve/main/meta/stats.json")
        return i, info, st
    rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        for i, info, st in ex.map(probe, ids):
            if not info or not st or info.get("codebase_version") != "v3.0" or info.get("fps") != 30 or info.get("total_episodes", 0) < a.min_eps:
                continue
            F = info["features"]; cams = [k for k, v in F.items() if v.get("dtype") == "video"]
            wr = [c for c in cams if any(w in c.split(".")[-1].lower() for w in WRIST)]; ov = [c for c in cams if c not in wr]
            if len(wr) != 1 or len(ov) != 1 or F.get("action", {}).get("shape") != [6] or any(F[c]["shape"][:2] != [480, 640] for c in cams):
                continue
            s = st.get("observation.state") or {}
            v = lambda k: np.array(s.get(k, s.get("min")), float).ravel()
            if not s or len(v("mean")) != 6:
                continue
            lo, hi = v("q01") if "q01" in s else v("min"), v("q99") if "q99" in s else v("max")
            mx = np.abs(np.r_[v("min")[:5], v("max")[:5]]).max()
            mode = "rad" if mx < 4 else ("deg" if np.abs(np.r_[lo[:5], hi[:5]]).max() > 105 else "range")
            rows.append(dict(id=i, mode=mode, episodes=info["total_episodes"], frames=info["total_frames"], mean=np.round(v("mean"), 1).tolist()))
    deg = [r for r in rows if r["mode"] == "deg"]; med = np.median([r["mean"] for r in deg], 0)
    for r in deg:
        r["dev"] = round(float(np.abs(np.array(r["mean"]) - med)[:5].max()), 1)
    sel = sorted([r for r in deg if r["dev"] <= a.max_dev], key=lambda r: -r["frames"])
    json.dump(dict(median_mean=med.round(1).tolist(), selected=sel, all=rows), open(a.out, "w"), indent=1)
    print(json.dumps(dict(candidates=len(rows), degree_mode=len(deg), selected=len(sel), frames=sum(r["frames"] for r in sel))))
    for r in sel:
        print(r["id"], r["episodes"], r["frames"], "dev", r["dev"])


if __name__ == "__main__":
    main()
