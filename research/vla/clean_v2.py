"""Cleaning v2: repair first, drop only what cannot be repaired.

Per repo, full-rate (native fps) arrays from data/actions/<repo>.npz:
  1 glitch repair   : non-finite / absurd values (robust z > 12) and revert-spikes -> linear interpolation   [FIX]
  2 leader->follower: per-repo lag L and per-joint affine map  S[t+L] ~ a*A[t] + b  (pooled over episodes)
                      action_fixed = a*A + b  (action expressed in follower/state units, same direction)       [FIX]
                      episode dropped if a joint is commanded but the follower does not move (stuck servo),
                      or tracking R^2 of the repo map on this episode < 0.5                                     [DROP]
  3 video<->state lag (from image-motion / joint-speed correlation)                                            [FIX]
  4 idle start/end, pauses > 1 s                                                                               [CUT]
  5 blank/corrupt frames (low contrast), frozen frames                                                         [DROP frame; episode if > 30 %]
  6 duplicates (identical repaired action arrays), incl. copies of few-shot benchmark episodes               [DROP]
  7 labels: annotated instruction is the label; template / conflicting original tasks are discarded;
            colour words normalised to the repo majority when the grasped object looks identical            [FIX]
            degenerate labels ('<ALL FROZEN FRAMES>')                                                         [DROP]
  8 within-repo outlier episodes                                                                               [WEIGHT]
Outputs data/clean_v2/: actions_fixed/<repo>.npz (ep<k>_action, ep<k>_state, fixed, native fps), frames.parquet,
episodes.parquet, repo_calib.parquet, report.json
"""
import os, io, json, hashlib, sys, re
import numpy as np, pandas as pd, torch
from torchvision.io import decode_jpeg
sys.path.insert(0, os.path.dirname(__file__))
from clean_data import frame_flags, resample, robust_z, smooth, IDLE_EDGE, IDLE_PAUSE, SPIKE, FROZEN, EP_DROP

ROOT = os.path.expanduser("~/VLA/data")
OUT = os.path.join(ROOT, os.environ.get("CLEAN_OUT", "clean_v2"))
BLANK = 32
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]


def repair(X):
    """returns repaired copy and number of repaired samples"""
    X = X.astype(np.float64).copy()
    bad = ~np.isfinite(X)
    for j in range(X.shape[1]):
        x = X[:, j]
        ok = np.isfinite(x)
        if ok.sum() < 3:
            continue
        p1, p99 = np.percentile(x[ok], 1), np.percentile(x[ok], 99)
        rng = p99 - p1 + 1e-6
        med = np.median(x[ok])
        b = ~ok | (x < p1 - 0.5 * rng - 1.0) | (x > p99 + 0.5 * rng + 1.0)   # physically implausible jumps out of range
        d = np.diff(np.where(b, med, x))
        jump = np.abs(d) > 0.3 * rng
        for i in np.nonzero(jump)[0]:            # revert spike: up then back within 3 samples
            back = d[i + 1: i + 4]
            if len(back) and np.abs(d[i] + back.sum()) < 0.3 * np.abs(d[i]):
                b[i + 1: i + 1 + max(1, np.argmax(np.abs(np.cumsum(back) + d[i]) < 0.3 * abs(d[i])) + 1)] = True
        bad[:, j] |= b
    n = int(bad.sum())
    t = np.arange(len(X))
    for j in range(X.shape[1]):
        b = bad[:, j]
        if b.any() and (~b).sum() >= 2:
            X[b, j] = np.interp(t[b], t[~b], X[~b, j])
    return X.astype(np.float32), n


def fit_calib(eps, max_lag):
    """pooled per-joint affine S[t+L] = a*A[t] + b, best common lag L"""
    best = None
    for L in range(0, max_lag + 1):
        Aa = np.concatenate([A[: len(A) - L] for A, S in eps if len(A) > L + 5])
        Ss = np.concatenate([S[L:] for A, S in eps if len(A) > L + 5])
        coef, r2 = [], []
        for j in range(6):
            X = np.c_[Aa[:, j], np.ones(len(Aa))]
            w, *_ = np.linalg.lstsq(X, Ss[:, j], rcond=None)
            p = X @ w
            r2.append(1 - ((Ss[:, j] - p) ** 2).sum() / (((Ss[:, j] - Ss[:, j].mean()) ** 2).sum() + 1e-9))
            coef.append(w)
        sc = float(np.mean(r2[:5]))
        if best is None or sc > best[0]:
            best = (sc, L, np.array(coef, np.float32), np.array(r2, np.float32))
    return best


def ep_track(A, S, coef, L):
    """per-joint R^2 of the repo map on this episode + stuck-joint test"""
    if len(A) <= L + 5:
        return np.zeros(6), False
    a, s = A[: len(A) - L], S[L:]
    p = a * coef[:, 0] + coef[:, 1]
    r2 = 1 - ((s - p) ** 2).sum(0) / (((s - s.mean(0)) ** 2).sum(0) + 1e-9)
    ra, rs = np.ptp(a, 0), np.ptp(s, 0)
    cmd_range = np.abs(ra * coef[:, 0])
    stuck = bool(((cmd_range[:5] > 0.05 * (np.abs(coef[:5, 0]) * ra[:5]).max() + 1e-6) & (rs[:5] < 0.1 * cmd_range[:5])).any())
    return r2, stuck


def object_color(t):
    ws = re.findall(r"[a-z]+", str(t).lower())
    stop = next((i for i, w in enumerate(ws) if w in ("in", "into", "inside", "on", "onto", "to", "towards", "near")), len(ws))
    for i, w in enumerate(ws[:stop]):
        if w in COLORS and not set(ws[i + 1:i + 3]) & {"arm", "gripper", "robot", "robotic"}:
            return w
    return None


def main():
    os.makedirs(os.path.join(OUT, "actions_fixed"), exist_ok=True)
    I = pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["repo_id", "ep", "cam", "wrist", "fidx", "off", "len", "task_id"])
    R = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet"))
    T = pd.read_parquet(os.path.join(ROOT, "packed/tasks.parquet")).set_index("task_id")
    LA = pd.read_parquet(os.path.join(ROOT, "clean/label_audit.parquet")).set_index(["repo_id", "ep"])
    blob = np.memmap(os.path.join(ROOT, "packed/frames.bin"), dtype=np.uint8, mode="r")
    bench = set(json.load(open(os.path.join(ROOT, "bench/repos.json"))))
    # hashes of benchmark episodes (to remove any copy from training data)
    bench_hash = set()
    for b in bench:
        f = os.path.join(ROOT, "bench/actions", b.replace("/", "__") + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            for k in z.files:
                if k.endswith("_action"):
                    bench_hash.add(hashlib.md5(np.round(repair(z[k])[0], 2).tobytes()).hexdigest())
    reasons = np.zeros(len(I), np.int32); lag_col = np.zeros(len(I), np.int16); weight = np.ones(len(I), np.float32)
    seen_hash = {}
    E_rows, C_rows = [], []
    grp = I.groupby("repo_id").indices
    for _, rr in R.head(int(os.environ.get("LIMIT", len(R)))).iterrows():
        rid, repo, fps = int(rr.repo_id), rr.repo, int(rr.fps)
        Ir = I.iloc[grp[rid]]
        af = os.path.join(ROOT, "actions", repo.replace("/", "__") + ".npz")
        if repo in bench:
            reasons[Ir.index] |= EP_DROP
            E_rows += [dict(repo_id=rid, repo=repo, ep=e, drop="benchmark_repo") for e in Ir.ep.unique()]
            continue
        if not os.path.exists(af):
            reasons[Ir.index] |= EP_DROP
            E_rows += [dict(repo_id=rid, repo=repo, ep=e, drop="no_actions") for e in Ir.ep.unique()]
            continue
        z = np.load(af)
        eps = {}
        for e in Ir.ep.unique():
            if f"ep{e}_action" in z.files:
                A, na = repair(z[f"ep{e}_action"]); S, ns = repair(z[f"ep{e}_state"])
                eps[e] = (A, S, na + ns)
        good = [(A, S) for A, S, _ in eps.values() if len(A) > 2 * fps]
        if not good:
            reasons[Ir.index] |= EP_DROP
            E_rows += [dict(repo_id=rid, repo=repo, ep=e, drop="too_short") for e in Ir.ep.unique()]
            continue
        sc, L, coef, jr2 = fit_calib(good, max_lag=int(0.6 * fps))
        C_rows.append(dict(repo_id=rid, repo=repo, lag=L, r2=sc, scale=coef[:, 0], offset=coef[:, 1], joint_r2=jr2))
        fixed = {}
        stats = []
        pend = []
        for e, Ie in Ir.groupby("ep"):
            st = dict(repo_id=rid, repo=repo, ep=e)
            drop = []
            if e not in eps:
                st["drop"] = "no_actions"; stats.append(st); reasons[Ie.index] |= EP_DROP; continue
            A, S, nrep = eps[e]
            st.update(T=len(A), dur=len(A) / fps, repaired_samples=nrep)
            Af = A * coef[:, 0] + coef[:, 1] if sc > 0.6 else A.copy()   # action in follower units
            if sc <= 0.6:
                Af[:, :] = A
            r2e, stuck = ep_track(A, S, coef, L)
            moving_j = np.ptp(S[:, :5], 0) > 0.05 * np.ptp(S[:, :5], 0).max()
            st["track_r2"] = float(np.min(r2e[:5][moving_j])) if moving_j.any() else 0.0
            if len(A) < fps:
                drop.append("too_short")
            if stuck:
                drop.append("follower_joint_stuck")
            elif sc > 0.6 and st["track_r2"] < 0.5:
                drop.append("tracking_failure")
            h = hashlib.md5(np.round(A, 2).tobytes()).hexdigest()
            if h in bench_hash:
                drop.append("copy_of_benchmark")
            elif h in seen_hash:
                drop.append(f"duplicate_of:{seen_hash[h]}")
            else:
                seen_hash[h] = f"{repo}#{e}"
            flags, moving = frame_flags(Af, fps)
            st["moving_frac"] = float(moving.mean()); st["has_actions"] = True
            st["idle_cut_frac"] = float((flags & (IDLE_EDGE | IDLE_PAUSE)).astype(bool).mean())
            if moving.mean() < 0.15:
                drop.append("no_motion")
            # labels
            la = LA.loc[(rid, e)] if (rid, e) in LA.index else None
            task = T.loc[Ie.task_id.iloc[0]].task
            lf = la["flags"] if la is not None else ""
            if "degenerate" in lf:
                drop.append("label_degenerate")
            st["label"] = task
            st["label_src"] = "annotation"
            st["orig_discarded"] = bool(la is not None and ("orig_template" in lf or "orig_conflict" in lf))
            st["obj_color"] = object_color(task)
            # gripper events for outlier stats
            g = Af[:, -1]; gt = (g - g.min()) / (np.ptp(g) + 1e-6)
            closed = smooth(gt, max(1, fps // 5)) < 0.35
            st["grasps"] = int(np.sum(np.diff(closed.astype(int)) == 1))
            st["final"] = Af[-max(1, fps // 2):].mean(0) / (np.ptp(Af, 0) + 1e-6)
            st["traj"] = resample((Af - Af.min(0)) / (np.ptp(Af, 0) + 1e-6))
            # images
            for cam, Ic in Ie.groupby("cam"):
                Ic = Ic.sort_values("fidx")
                ims = torch.stack([decode_jpeg(torch.from_numpy(np.array(blob[o:o + l]))).float().mean(0) for o, l in zip(Ic.off, Ic.len)])
                bits = np.zeros(len(Ic), np.int32)
                std = ims.flatten(1).std(1).numpy(); mean = ims.flatten(1).mean(1).numpy()
                blank = (std < 8) | (mean < 12) | (mean > 245)
                # isolated corrupt frames: very different from both neighbours while neighbours agree
                if len(ims) >= 3:
                    dp = (ims[1:-1] - ims[:-2]).abs().mean((1, 2)); dn = (ims[1:-1] - ims[2:]).abs().mean((1, 2)); nn = (ims[2:] - ims[:-2]).abs().mean((1, 2))
                    iso = ((dp > 25) & (dn > 25) & (nn < 0.5 * torch.minimum(dp, dn))).numpy()
                    blank[1:-1] |= iso
                bits[blank] |= BLANK
                fi = np.clip(Ic.fidx.to_numpy(), 0, len(Af) - 1)
                T_ = len(Af)
                if len(Ic) >= 6:
                    md = (ims[1:] - ims[:-1]).abs().mean((1, 2)).numpy()
                    js = (np.abs(np.diff(Af[fi], axis=0)) / (np.ptp(Af, 0) + 1e-6)).max(1)
                    fr = (md < 0.3) & (js > 0.02)
                    bits[1:][fr] |= FROZEN
                    def corr(Lg):
                        a_, b_ = md[max(0, Lg): len(md) + min(0, Lg)], js[max(0, -Lg): len(js) - max(0, Lg)]
                        return np.corrcoef(a_, b_)[0, 1] if len(a_) > 6 and a_.std() > 0 and b_.std() > 0 else np.nan
                    cs = {Lg: corr(Lg) for Lg in range(-6, 7)}
                else:
                    cs = None
                if blank.mean() > 0.3:
                    drop.append("video_corrupt")
                if (bits & FROZEN).astype(bool).mean() > 0.3:
                    drop.append("video_frozen")
                step = int(np.median(np.diff(Ic.fidx.to_numpy()))) if len(Ic) > 1 else 1
                pend.append((e, cam, Ic.index.to_numpy(), bits, fi, step, len(Af), flags, cs))
            st["drop"] = ",".join(sorted(set(drop)))
            if st["drop"]:
                reasons[Ie.index.to_numpy()] |= EP_DROP
            else:
                fixed[f"ep{e}_action"], fixed[f"ep{e}_state"] = Af.astype(np.float32), S.astype(np.float32)
            stats.append(st)
        # video<->state lag: one value per (repo, camera), pooled over episodes; must agree on even and odd episodes
        cam_lag = {}
        for cam in {p[1] for p in pend}:
            curves = [(p[0], p[8]) for p in pend if p[1] == cam and p[8] is not None]
            def pooled(sub):
                M = np.array([[c[Lg] for Lg in range(-6, 7)] for _, c in sub], float)
                return np.nanmean(M, 0) if len(M) else np.full(13, np.nan)
            if len(curves) >= 4:
                al, ev, od = pooled(curves), pooled([c for c in curves if c[0] % 2 == 0]), pooled([c for c in curves if c[0] % 2 == 1])
                if np.isfinite(al).all():
                    Lb = int(np.nanargmax(al)) - 6
                    ok_eo = np.isfinite(ev).all() and np.isfinite(od).all() and int(np.nanargmax(ev)) - 6 == Lb and int(np.nanargmax(od)) - 6 == Lb
                    cam_lag[cam] = Lb if (Lb != 0 and abs(Lb) < 6 and ok_eo and al[Lb + 6] - al[6] > 0.03) else 0
        for e, cam, idx, bits, fi, step, T_, flags, cs in pend:
            lag = cam_lag.get(cam, 0)
            bits = bits | flags[np.clip(fi + lag * step, 0, T_ - 1)]
            reasons[idx] |= bits
            lag_col[idx] = lag
        # colour label normalisation within repo (grasped object looks the same, labels disagree)
        ok = [s for s in stats if not s.get("drop")]
        cols = [s["obj_color"] for s in ok if s.get("obj_color")]
        la_r = LA.loc[rid] if rid in LA.index.get_level_values(0) else None
        if la_r is not None and len(cols) >= 4:
            gc = la_r.get("grasp_color")
            if gc is not None and gc.notna().sum() >= 4 and gc.value_counts().iloc[0] / gc.notna().sum() >= 0.8:
                major = pd.Series(cols).value_counts().index[0]
                for s in ok:
                    if s.get("obj_color") and s["obj_color"] != major:
                        s["label"] = re.sub(rf"\b{s['obj_color']}\b", major, s["label"], count=1); s["label_src"] = "annotation+color_fixed"
        # outliers -> weights
        if len(ok) >= 5:
            Tz = robust_z(np.log([s["T"] for s in ok])); Gz = robust_z([s["grasps"] for s in ok])
            Tr = np.stack([s["traj"] for s in ok]); Dz = robust_z(np.sqrt(((Tr - np.median(Tr, 0)) ** 2).mean((1, 2))))
            Fn = np.stack([s["final"] for s in ok]); Fz = robust_z(np.sqrt(((Fn - np.median(Fn, 0)) ** 2).sum(1)))
            for s, tz, gz, dz, fz in zip(ok, Tz, Gz, Dz, Fz):
                bad = [n for n, c in [("len", abs(tz) > 3), ("regrasp", gz > 3 and s["grasps"] >= 3), ("traj", dz > 3.5), ("final", fz > 3.5)] if c]
                s["outlier"] = ",".join(bad); s["weight"] = 0.25 if len(bad) >= 2 else (0.5 if bad else 1.0)
        for s in stats:
            s.pop("traj", None); s.pop("final", None)
            if s.get("weight", 1.0) < 1:
                weight[Ir.index[(Ir.ep == s["ep"]).to_numpy()]] = s["weight"]
        if fixed:
            np.savez_compressed(os.path.join(OUT, "actions_fixed", repo.replace("/", "__") + ".npz"), **fixed)
        E_rows += stats
        if rid % 100 == 0:
            print(rid, len(R), flush=True)
    E = pd.DataFrame(E_rows); E["drop"] = E["drop"].fillna("")
    F = pd.DataFrame(dict(keep=reasons == 0, reasons=reasons, weight=weight, lag=lag_col))
    C = pd.DataFrame(C_rows)
    E.to_parquet(os.path.join(OUT, "episodes.parquet")); F.to_parquet(os.path.join(OUT, "frames.parquet")); C.to_parquet(os.path.join(OUT, "repo_calib.parquet"))
    dr = pd.Series([("duplicate" if r.startswith("duplicate_of") else r) for d in E["drop"] for r in d.split(",") if r]).value_counts().to_dict()
    rep = dict(
        episodes=len(E), episodes_kept=int((E["drop"] == "").sum()), episodes_dropped=int((E["drop"] != "").sum()), drop_reasons=dr,
        fixed=dict(repaired_glitch_samples=int(E.repaired_samples.fillna(0).sum()), episodes_with_glitch_repair=int((E.repaired_samples.fillna(0) > 0).sum()),
                   repos_leader_follower_rescaled=int(((C.r2 > 0.6) & C.scale.map(lambda s: np.abs(np.abs(np.asarray(s)[:5]) - 1).max() > 0.1)).sum()),
                   repos_leader_follower_inverted_joint=int(C.scale.map(lambda s: (np.asarray(s)[:5] < -0.1).any()).sum()),
                   median_leader_follower_lag_s=float(np.median(C.lag / R.set_index("repo_id").loc[C.repo_id].fps.values)),
                   video_lag_fixed_seqs=int(I.assign(lag=F.lag.values)[F.lag.values != 0].drop_duplicates(["repo_id", "ep", "cam"]).shape[0]),
                   labels_original_task_discarded=int(E.get("orig_discarded", pd.Series(dtype=bool)).fillna(False).sum()),
                   labels_color_fixed=int((E.get("label_src", pd.Series(dtype=str)) == "annotation+color_fixed").sum())),
        weighted_outliers=int(E.get("weight", pd.Series(dtype=float)).between(0.01, 0.99).sum()),
        frames=len(F), frames_kept=int(F.keep.sum()),
        frame_reasons={n: int(((F.reasons & b) > 0).sum()) for n, b in [("idle_edge", IDLE_EDGE), ("idle_pause", IDLE_PAUSE), ("spike", SPIKE), ("frozen", FROZEN), ("blank_corrupt", BLANK), ("episode_drop", EP_DROP)]},
    )
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
