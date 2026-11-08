"""Data cleaning for SO-100/101 LeRobot episodes.

Frame-level (hard drop, reason bit-flags):
  IDLE_EDGE   leading/trailing segment where the arm does not move
  IDLE_PAUSE  mid-episode pause longer than PAUSE_S (keeps a margin around motion)
  SPIKE       servo/comm glitch: single-step jump that immediately returns
  FROZEN      image does not change while joints move (video freeze)
Episode-level:
  drop   : annotated '<ALL FROZEN FRAMES>', action/state unit or calibration mismatch, almost no motion,
           external camera not correlated with arm motion even after lag search
  fix    : video<->state lag estimated with confidence -> image index shifted (stored as 'lag')
  weight : within-repo outliers (length, re-grasps, trajectory distance to medoid, final pose) -> weight < 1
Outputs (data/clean/):
  frames.parquet   per packed-index row: keep(bool), reasons(int bitmask), weight(float), lag(int, in frame steps)
  episodes.parquet per (repo, ep): stats, flags, weight
  report.json      summary;  samples/*.jpg contact sheets of flagged cases
"""
import os, glob, json, io
import numpy as np, pandas as pd, torch
from torchvision.io import decode_jpeg
from PIL import Image, ImageDraw

ROOT = os.path.expanduser("~/VLA/data")
OUT = os.path.join(ROOT, "clean")
IDLE_EDGE, IDLE_PAUSE, SPIKE, FROZEN, EP_DROP = 1, 2, 4, 8, 16
MOVE_THR = 0.004      # per-30Hz-step speed, fraction of the episode joint range, smoothed over 0.5 s
PAUSE_S = 1.0         # pauses longer than this are cut
MARGIN_S = 0.3        # keep this much around motion onsets/offsets (approach / settle)
SPIKE_THR = 0.25      # single-step jump > 25 % of joint range that reverts within 3 steps


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def motion_mask(A, fps):
    """returns per-step moving mask (len T), and spike mask"""
    rng = A.max(0) - A.min(0) + 1e-6
    d = np.diff(A, axis=0) / rng
    sp = np.abs(d).max(1)
    # spikes: big jump followed by a jump back
    spike = np.zeros(len(A), bool)
    big = np.nonzero(sp > SPIKE_THR)[0]
    for i in big:
        back = d[i + 1: i + 4].sum(0) if i + 1 < len(d) else 0
        if np.abs(d[i] + back).max() < 0.5 * np.abs(d[i]).max():
            spike[max(0, i - 2): i + 5] = True
    sp = np.where(spike[1:], 0, sp)
    ms = smooth(np.concatenate([[sp[0]], sp]), max(1, int(0.5 * fps)))
    grip = np.abs(np.gradient(A[:, -1] / rng[-1])) > MOVE_THR  # gripper motion counts as motion
    return (ms > MOVE_THR) | grip, spike


def frame_flags(A, fps):
    T = len(A)
    if T < 3:
        return np.full(T, IDLE_EDGE, np.int32), np.zeros(T, bool)
    moving, spike = motion_mask(A, fps)
    flags = np.zeros(T, np.int32)
    idx = np.nonzero(moving)[0]
    if len(idx) == 0:
        flags[:] = IDLE_EDGE
        return flags, moving
    m = int(MARGIN_S * fps)
    flags[: max(0, idx[0] - m)] |= IDLE_EDGE
    flags[min(T, idx[-1] + m + 1):] |= IDLE_EDGE
    # mid pauses
    still = ~moving
    i = idx[0]
    while i < idx[-1]:
        if still[i]:
            j = i
            while j < idx[-1] and still[j]:
                j += 1
            if (j - i) > PAUSE_S * fps:
                flags[i + m: j - m] |= IDLE_PAUSE
            i = j
        else:
            i += 1
    flags[spike] |= SPIKE
    return flags, moving


def resample(x, n=50):
    t = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, k]) for k in range(x.shape[1])], 1)


def robust_z(v):
    v = np.asarray(v, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-9
    return (v - med) / mad


def main():
    os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
    I = pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["repo_id", "ep", "cam", "wrist", "fidx", "off", "len", "seq", "pos"])
    R = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet"))
    tasks = pd.read_parquet(os.path.join(ROOT, "packed/tasks.parquet"))
    I = I.merge(pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["task_id"]), left_index=True, right_index=True)
    frozen_task = set(tasks[tasks.task.str.contains("FROZEN FRAMES", case=False, na=False)].task_id)
    blob = np.memmap(os.path.join(ROOT, "packed/frames.bin"), dtype=np.uint8, mode="r")
    keep_reason = np.zeros(len(I), np.int32)
    lag_col = np.zeros(len(I), np.int16)
    weight_col = np.ones(len(I), np.float32)
    ep_rows = []
    for _, rr in R.iterrows():
        rid, repo, fps = rr.repo_id, rr.repo, int(rr.fps)
        af = os.path.join(ROOT, "actions", repo.replace("/", "__") + ".npz")
        Ir = I[I.repo_id == rid]
        if not os.path.exists(af):
            ep_rows += [dict(repo_id=rid, repo=repo, ep=e, has_actions=False) for e in Ir.ep.unique()]
            continue
        z = np.load(af)
        stats = []
        for e, Ie in Ir.groupby("ep"):
            if f"ep{e}_action" not in z.files:
                stats.append(dict(repo_id=rid, repo=repo, ep=e, has_actions=False)); continue
            A, S = z[f"ep{e}_action"], z[f"ep{e}_state"]
            T = len(A)
            st = dict(repo_id=rid, repo=repo, ep=e, has_actions=True, T=T, dur=T / fps)
            drop = []
            if T < fps:  # < 1 s
                drop.append("too_short")
            flags, moving = frame_flags(A, fps)
            st["moving_frac"] = float(moving.mean())
            st["idle_cut_frac"] = float((flags & (IDLE_EDGE | IDLE_PAUSE)).astype(bool).mean())
            st["spike_frac"] = float((flags & SPIKE).astype(bool).mean())
            if moving.mean() < 0.15:
                drop.append("no_motion")
            rng = A.max(0) - A.min(0) + 1e-6
            trk = np.median(np.abs(A - S) / rng, 0)
            st["track_err"] = float(trk[:5].max())
            if not np.isfinite(A).all() or not np.isfinite(S).all() or trk[:5].max() > 0.5:
                drop.append("action_state_mismatch")
            if Ie.task_id.isin(frozen_task).any():
                drop.append("annot_frozen")
            # gripper re-grasps: closing events
            g = A[:, -1]
            gt = (g - g.min()) / (g.max() - g.min() + 1e-6)
            closed = smooth(gt, max(1, fps // 5)) < 0.35
            st["grasps"] = int(np.sum(np.diff(closed.astype(int)) == 1))
            st["final"] = A[-max(1, fps // 2):].mean(0) / rng
            st["traj"] = resample((A - A.min(0)) / rng)
            # image checks per camera
            for cam, Ic in Ie.groupby("cam"):
                Ic = Ic.sort_values("fidx")
                if len(Ic) < 6:
                    continue
                ims = torch.stack([decode_jpeg(torch.from_numpy(np.array(blob[o:o + l]))).float().mean(0) for o, l in zip(Ic.off, Ic.len)])
                md = (ims[1:] - ims[:-1]).abs().mean((1, 2)).numpy()
                fi = np.clip(Ic.fidx.to_numpy(), 0, T - 1)
                js = np.array([np.abs(A[min(b, T - 1)] - A[a]).max() / 1 for a, b in zip(fi[:-1], fi[1:])]) / rng.max()
                js = (np.abs(np.diff(A[fi], axis=0)) / rng).max(1)
                fr = (md < 0.3) & (js > 0.02)
                frame_bits = np.zeros(len(Ic), np.int32)
                frame_bits[1:][fr] |= FROZEN
                # lag search (in sampled steps)
                def corr(L):
                    a_, b_ = md[max(0, L): len(md) + min(0, L)], js[max(0, -L): len(js) - max(0, L)]
                    return np.corrcoef(a_, b_)[0, 1] if len(a_) > 6 and a_.std() > 0 and b_.std() > 0 else np.nan
                cs = {L: corr(L) for L in range(-3, 4)}
                c0 = cs[0] if np.isfinite(cs[0]) else 0.0
                Lb = max(cs, key=lambda L: cs[L] if np.isfinite(cs[L]) else -9)
                lag = Lb if (np.isfinite(cs[Lb]) and cs[Lb] - c0 > 0.15 and cs[Lb] > 0.3) else 0
                st[f"corr_{Ic.wrist.iloc[0]}"] = c0
                st[f"lag_{Ic.wrist.iloc[0]}"] = lag
                if Ic.wrist.iloc[0] == 0 and np.isfinite(cs[Lb]) and cs[Lb] < 0.05 and moving.mean() > 0.3:
                    drop.append("cam_uncorrelated")
                if fr.mean() > 0.3:
                    drop.append("video_frozen")
                # map 30 Hz frame flags to sampled frames (after lag shift)
                step = np.median(np.diff(Ic.fidx.to_numpy())) if len(Ic) > 1 else 1
                fsrc = np.clip(fi + int(lag * step), 0, T - 1)
                frame_bits |= flags[fsrc]
                keep_reason[Ic.index.to_numpy()] = frame_bits
                lag_col[Ic.index.to_numpy()] = lag
            st["drop"] = ",".join(sorted(set(drop)))
            stats.append(st)
        # within-repo outlier weighting
        ok = [s for s in stats if s.get("has_actions") and not s.get("drop")]
        if len(ok) >= 5:
            Tz = robust_z(np.log([s["T"] for s in ok]))
            Gz = robust_z([s["grasps"] for s in ok])
            Tr = np.stack([s["traj"] for s in ok]); med = np.median(Tr, 0)
            Dz = robust_z(np.sqrt(((Tr - med) ** 2).mean((1, 2))))
            Fz = robust_z(np.sqrt(((np.stack([s["final"] for s in ok]) - np.median(np.stack([s["final"] for s in ok]), 0)) ** 2).sum(1)))
            for s, tz, gz, dz, fz in zip(ok, Tz, Gz, Dz, Fz):
                s.update(z_len=tz, z_grasp=gz, z_traj=dz, z_final=fz)
                bad = []
                if abs(tz) > 3: bad.append("len")
                if gz > 3 and s["grasps"] >= 3: bad.append("regrasp")
                if dz > 3.5: bad.append("traj")
                if fz > 3.5: bad.append("final")
                s["outlier"] = ",".join(bad)
                s["weight"] = float(0.25 if len(bad) >= 2 else (0.5 if bad else 1.0))
        for s in stats:
            s.pop("traj", None); s.pop("final", None)
            if s.get("drop"):
                s["weight"] = 0.0
            m = ((I.repo_id == rid) & (I.ep == s["ep"])).to_numpy()
            w = s.get("weight", 1.0)
            weight_col[m] = w
            if s.get("drop"):
                keep_reason[m] |= EP_DROP
        ep_rows += stats
        if rid % 50 == 0:
            print(rid, len(R), flush=True)
    E = pd.DataFrame(ep_rows)
    F = pd.DataFrame(dict(keep=(keep_reason == 0) & (weight_col > 0), reasons=keep_reason, weight=weight_col, lag=lag_col))
    has = E.has_actions.fillna(False).to_numpy()
    F["has_actions"] = I.set_index(["repo_id", "ep"]).index.isin(E[has].set_index(["repo_id", "ep"]).index)
    F.loc[~F.has_actions, "keep"] = False
    E.to_parquet(os.path.join(OUT, "episodes.parquet")); F.to_parquet(os.path.join(OUT, "frames.parquet"))
    Eh = E[has]
    rep = dict(
        episodes=int(len(E)), episodes_with_actions=int(has.sum()),
        ep_drop=int((Eh["drop"].fillna("") != "").sum()),
        ep_drop_reasons=pd.Series([r for d in Eh["drop"].fillna("") for r in d.split(",") if r]).value_counts().to_dict(),
        ep_outlier_downweighted=int((Eh.get("weight", pd.Series(dtype=float)) .between(0.01, 0.99)).sum()),
        outlier_reasons=pd.Series([r for d in Eh.get("outlier", pd.Series(dtype=str)).fillna("") for r in d.split(",") if r]).value_counts().to_dict(),
        lag_fixed_seqs=int((F.groupby(I.seq).lag.first() != 0).sum()),
        frames=int(len(F)), frames_with_actions=int(F.has_actions.sum()),
        frames_kept=int(F.keep.sum()),
        frame_reasons={n: int(((F.reasons & b) > 0)[F.has_actions].sum()) for n, b in
                       [("idle_edge", IDLE_EDGE), ("idle_pause", IDLE_PAUSE), ("spike", SPIKE), ("frozen", FROZEN), ("episode_drop", EP_DROP)]},
    )
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))
    # contact sheets: for each reason, 6 random examples x 8 frames over the episode with red border on dropped frames
    rng = np.random.default_rng(0)
    for name, cond in [("drop_episode", Eh["drop"].fillna("") != ""), ("outlier", Eh.get("outlier", pd.Series("", index=Eh.index)).fillna("") != ""),
                       ("clean", (Eh["drop"].fillna("") == "") & (Eh.get("outlier", pd.Series("", index=Eh.index)).fillna("") == ""))]:
        sel = Eh[cond]
        if not len(sel):
            continue
        sheet = Image.new("RGB", (8 * 160, 6 * 140), "white")
        dr = ImageDraw.Draw(sheet)
        for r_, (_, s) in enumerate(sel.sample(min(6, len(sel)), random_state=0).iterrows()):
            Ie = I[(I.repo_id == s.repo_id) & (I.ep == s.ep) & (I.wrist == 0)].sort_values("fidx")
            if not len(Ie):
                Ie = I[(I.repo_id == s.repo_id) & (I.ep == s.ep)].sort_values("fidx")
            pick = np.linspace(0, len(Ie) - 1, 8).astype(int)
            for c_, p in enumerate(pick):
                row = Ie.iloc[p]
                im = Image.open(io.BytesIO(bytes(blob[row.off: row.off + row.len])))
                sheet.paste(im, (c_ * 160, r_ * 140))
                if not F.keep.iloc[row.name]:
                    dr.rectangle([c_ * 160, r_ * 140, c_ * 160 + 159, r_ * 140 + 119], outline="red", width=3)
            lbl = f"{s.repo[:40]} ep{s.ep} {s.get('drop') or ''} {s.get('outlier') or ''}"
            dr.text((2, r_ * 140 + 122), lbl, fill="black")
        sheet.save(os.path.join(OUT, "samples", f"{name}.jpg"))


if __name__ == "__main__":
    main()
