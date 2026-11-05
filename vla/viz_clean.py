"""Before/after cleaning figures: per example episode
  row 1  BEFORE: 10 frames evenly over the original episode (red border = dropped)
  row 2  joint trajectories (30 Hz) with removed spans shaded
  row 3  AFTER : 10 frames evenly over the kept frames
"""
import os, io, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from clean_data import frame_flags, IDLE_EDGE, IDLE_PAUSE, SPIKE, FROZEN, EP_DROP

ROOT = os.path.expanduser("~/VLA/data")
OUT = os.path.join(ROOT, "clean/samples")
I = pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["repo_id", "ep", "cam", "wrist", "fidx", "off", "len", "task_id"])
F = pd.read_parquet(os.path.join(ROOT, "clean/frames.parquet"))
E = pd.read_parquet(os.path.join(ROOT, "clean/episodes.parquet"))
R = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet")).set_index("repo_id")
T = pd.read_parquet(os.path.join(ROOT, "packed/tasks.parquet")).set_index("task_id")
blob = np.memmap(os.path.join(ROOT, "packed/frames.bin"), dtype=np.uint8, mode="r")
NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]


def img(row):
    return Image.open(io.BytesIO(bytes(blob[row.off: row.off + row.len])))


def figure(s, title, fname):
    repo = R.loc[s.repo_id].repo
    fps = int(R.loc[s.repo_id].fps)
    z = np.load(os.path.join(ROOT, "actions", repo.replace("/", "__") + ".npz"))
    A = z[f"ep{s.ep}_action"]
    flags, _ = frame_flags(A, fps)
    Ie = I[(I.repo_id == s.repo_id) & (I.ep == s.ep)]
    ext = Ie[Ie.wrist == 0].cam.unique()
    cam = ext[0] if len(ext) else Ie.cam.iloc[0]
    Ie = Ie[Ie.cam == cam].sort_values("fidx")
    keep = F.keep.to_numpy()[Ie.index.to_numpy()]
    drop_ep = bool(s.get("drop"))
    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(3, 10, height_ratios=[1, 1.15, 1], hspace=0.35, wspace=0.04)
    task = T.loc[Ie.task_id.iloc[0]].task
    fig.suptitle(f"{title}\n{repo}  ep{s.ep}  ({len(A)/fps:.1f}s @ {fps}Hz)   task: \"{task[:90]}\"", fontsize=11)
    pick = np.linspace(0, len(Ie) - 1, 10).astype(int)
    for c, p in enumerate(pick):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(img(Ie.iloc[p])); ax.set_xticks([]); ax.set_yticks([])
        col = "red" if not keep[p] else "green"
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3)
        ax.set_title(f"t={Ie.iloc[p].fidx/fps:.1f}s", fontsize=8)
        if c == 0:
            ax.set_ylabel("BEFORE", fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    t = np.arange(len(A)) / fps
    rng = A.max(0) - A.min(0) + 1e-6
    for k in range(A.shape[1]):
        ax.plot(t, (A[:, k] - A[:, k].min()) / rng[k] + 1.1 * k, lw=1, label=NAMES[k])
    for bit, color, lab in [(IDLE_EDGE, "tab:red", "idle (start/end)"), (IDLE_PAUSE, "tab:orange", "pause > 1s"), (SPIKE, "tab:purple", "spike")]:
        m = (flags & bit) > 0
        if m.any():
            ax.fill_between(t, -0.1, 6.7, where=m, color=color, alpha=0.25, step="mid", label=lab)
    if drop_ep:
        ax.text(0.5, 0.5, f"WHOLE EPISODE DROPPED: {s['drop']}", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="red", alpha=0.8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="red"))
    ax.set_yticks([1.1 * k + 0.5 for k in range(6)]); ax.set_yticklabels(NAMES, fontsize=8)
    ax.set_xlim(0, t[-1]); ax.set_xlabel("time (s)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    kept = Ie[keep]
    for c in range(10):
        ax = fig.add_subplot(gs[2, c]); ax.set_xticks([]); ax.set_yticks([])
        if len(kept):
            p = np.linspace(0, len(kept) - 1, 10).astype(int)[c]
            ax.imshow(img(kept.iloc[p])); ax.set_title(f"t={kept.iloc[p].fidx/fps:.1f}s", fontsize=8)
        else:
            ax.text(0.5, 0.5, "—", ha="center", va="center")
        if c == 0:
            ax.set_ylabel("AFTER", fontsize=10)
    info = f"kept {keep.sum()}/{len(keep)} sampled frames"
    if s.get("outlier"):
        info += f"   |   within-repo outlier [{s['outlier']}] -> loss weight {s['weight']}"
    lag = F.lag.to_numpy()[Ie.index.to_numpy()][0]
    if lag:
        info += f"   |   video<->state lag corrected by {lag} frames"
    fig.text(0.5, 0.015, info, ha="center", fontsize=10)
    fig.savefig(os.path.join(OUT, fname), dpi=80, bbox_inches="tight"); plt.close(fig)


def main():
    Eh = E[E.has_actions.fillna(False)].copy()
    Eh["drop"] = Eh["drop"].fillna(""); Eh["outlier"] = Eh["outlier"].fillna("")
    rs = np.random.RandomState(3)
    cases = [
        ("1) Idle start/end trimmed", Eh[(Eh["drop"] == "") & (Eh.idle_cut_frac > 0.35)]),
        ("2) Mid-episode pause removed", Eh[(Eh["drop"] == "") & (Eh.idle_cut_frac.between(0.1, 0.3))]),
        ("3) Dropped: camera does not see the arm moving", Eh[Eh["drop"].str.contains("cam_uncorrelated")]),
        ("4) Dropped: action/state mismatch", Eh[Eh["drop"].str.contains("action_state_mismatch") & (Eh["drop"] == "action_state_mismatch") & (Eh.dur > 4)]),
        ("5) Down-weighted: re-grasp outlier", Eh[Eh.outlier.str.contains("regrasp")]),
        ("6) Clean episode (almost untouched)", Eh[(Eh["drop"] == "") & (Eh.outlier == "") & (Eh.idle_cut_frac < 0.05)]),
    ]
    for i, (title, d) in enumerate(cases):
        if len(d):
            s = d.iloc[rs.randint(len(d))]
            figure(s, title, f"before_after_{i+1}.png")
            print("saved", title, s.repo, s.ep)


if __name__ == "__main__":
    main()
