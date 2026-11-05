"""Does the policy perceive the scene or replay a memorised trajectory?

Fine-tune (same recipe as the benchmark), roll out R episodes and log per episode: cube start, target, end-effector
position at the first gripper-close command, at release, success. Then:
  * regression of grasp xy on cube xy (slope ~1, R^2 high = tracks the cube; R^2 ~0 = blind replay)
  * same for release xy vs target xy
  * 'blind replay' reference: grasp positions of the N training demos (mean) -> distance to actual cube
Writes MP4 clips (3 success + 3 fail, annotated) and a scatter figure to ckpt_scratch/sim_diag/.
"""
import argparse, os, json, random, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, av, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import fewshot as fs
import sim_env as se
from sim_fewshot import finetune, prep_img, CK, ROOT
from train_vla import H

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)


@torch.no_grad()
def rollout_logged(model, tok, row, P, env, text, dev, exec_steps, max_s=25.0):
    st, coef = P["st"], P["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    env.reset()
    c0, tg = env.cube_pos(), env.target_pos()
    cur = env.state().copy(); ids = tok([text]).to(dev)
    frames, grasp, release, closed = [], None, None, False
    while env.t < int(max_s * 30):
        S = env.state()
        f, w = env.render("front"), env.render("wrist")
        ext, wri = prep_img(f)[None].to(dev), prep_img(w)[None].to(dev)
        s_n = torch.from_numpy(np.clip((S - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(ext, wri, torch.tensor([True], device=dev), s_n, ids, torch.tensor([row], device=dev)).float()[0].cpu().numpy()
        tgt = np.empty((H, 6)); tgt[:, :5] = S[:5] + p[:, :5] * st["dstd"]; tgt[:, 5] = (p[:, 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
        tgt = (tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)
        for k in range(exec_steps):
            for i in range(3):
                cmd = cur + (tgt[k] - cur) * (i + 1) / 3
                env.step(cmd)
                if env.t % 3 == 0:
                    frames.append((np.concatenate([env.render("front"), env.render("wrist")], 1), env.attached is not None))
                if not closed and cmd[5] < 0.4:
                    closed = True; grasp = env.ee_pos()
                if closed and release is None and cmd[5] > 0.8 and grasp is not None:
                    release = env.ee_pos()
                if env.success():
                    return dict(success=True, cube=c0, target=tg, grasp=grasp, release=release, t=env.t / 30, frames=frames)
            cur = tgt[k]
    return dict(success=False, cube=c0, target=tg, grasp=grasp, release=release, t=max_s, frames=frames)


def write_mp4(frames, path, title, info):
    c = av.open(path, "w"); s = c.add_stream("libx264", rate=10)
    h, w = frames[0][0].shape[:2]; W, Hh = w * 2, h * 2 + 40
    s.width, s.height, s.pix_fmt = W, Hh, "yuv420p"
    for k, (im, att) in enumerate(frames):
        canvas = Image.new("RGB", (W, Hh), "white"); canvas.paste(Image.fromarray(im).resize((W, h * 2)), (0, 40))
        d = ImageDraw.Draw(canvas)
        d.text((6, 4), title, font=FONT, fill="black"); d.text((6, 21), info + f"   t={k/10:4.1f}s   {'HOLDING' if att else ''}", font=FONT, fill=(60, 60, 60))
        for p in s.encode(av.VideoFrame.from_image(canvas)):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def fit(x, y):
    """per-axis slope and R^2 of y ~ x"""
    out = []
    for k in range(2):
        A = np.c_[x[:, k], np.ones(len(x))]; w, *_ = np.linalg.lstsq(A, y[:, k], rcond=None)
        r2 = 1 - ((y[:, k] - A @ w) ** 2).sum() / (((y[:, k] - y[:, k].mean()) ** 2).sum() + 1e-12)
        out.append((round(float(w[0]), 2), round(float(r2), 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="A_v1"); ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--rollouts", type=int, default=30); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--exec_steps", type=int, default=4); ap.add_argument("--enc_lr", type=float, default=3e-5)
    a = ap.parse_args()
    dev = "cuda"; out = os.path.join(CK, "sim_diag", f"{a.cond}_N{a.N}_e{a.exec_steps}"); os.makedirs(out, exist_ok=True)
    fs.ROOT_BENCH = os.path.join(ROOT, "sim_bench")
    repo = json.load(open(os.path.join(ROOT, "sim_bench/repos.json")))[0]
    text = "Pick up the red cube and place it in the blue square."
    ck = {"A_v1": "vla/vla.pt", "A_v2": "vla_v2/vla.pt", "A_v2enc": "vla_v2/vla.pt", "A_v1enc": "vla/vla.pt", "B": "vla/vla.pt", "C": "vla/vla.pt"}
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    P = fs.prepare(repo, a.N, 0, 0)
    model, tok, row = finetune(P, a.cond, a.steps, 1e-4 if a.cond != "C" else 3e-4, 64, dev, os.path.join(CK, ck[a.cond]), a.enc_lr)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    # demo grasp positions (blind-replay reference): ee position where the demo first commands close
    env = se.SO101Env(n_cubes=1, seed=10_000)
    R = [rollout_logged(model, tok, row, P, env, text, dev, a.exec_steps) for _ in range(a.rollouts)]
    env.close()
    succ = np.array([r["success"] for r in R])
    cube = np.array([r["cube"][:2] for r in R]); tgt = np.array([r["target"][:2] for r in R])
    has_g = np.array([r["grasp"] is not None for r in R]); has_r = np.array([r["release"] is not None for r in R])
    G = np.array([r["grasp"][:2] if r["grasp"] is not None else [np.nan, np.nan] for r in R])
    Rl = np.array([r["release"][:2] if r["release"] is not None else [np.nan, np.nan] for r in R])
    res = dict(cond=a.cond, N=a.N, exec_steps=a.exec_steps, success_rate=float(succ.mean()), n=len(R), grasp_attempted=float(has_g.mean()),
               grasp_vs_cube_slope_r2_xy=fit(cube[has_g], G[has_g]) if has_g.sum() > 3 else None,
               grasp_err_cm=float(np.nanmean(np.linalg.norm(G - cube, axis=1)) * 100),
               grasp_spread_cm=float(np.nanmean(np.linalg.norm(G - np.nanmean(G, 0), axis=1)) * 100),
               cube_spread_cm=float(np.mean(np.linalg.norm(cube - cube.mean(0), axis=1)) * 100),
               blind_replay_err_cm=float(np.mean(np.linalg.norm(cube - np.nanmean(G, 0), axis=1)) * 100),
               release_vs_target_slope_r2_xy=fit(tgt[has_r], Rl[has_r]) if has_r.sum() > 3 else None,
               release_err_cm=float(np.nanmean(np.linalg.norm(Rl - tgt, axis=1)) * 100) if has_r.any() else None)
    json.dump(res, open(os.path.join(out, "diag.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    # scatter figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for k, nm in enumerate(["x", "y"]):
        ax[k].scatter(cube[has_g, k] * 100, G[has_g, k] * 100, c=["tab:green" if s else "tab:red" for s in succ[has_g]], s=30)
        lo, hi = cube[:, k].min() * 100 - 2, cube[:, k].max() * 100 + 2
        ax[k].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect tracking")
        ax[k].axhline(np.nanmean(G[:, k]) * 100, color="gray", lw=1, ls=":", label="constant (blind replay)")
        ax[k].set_xlabel(f"cube {nm} at start (cm)"); ax[k].set_ylabel(f"gripper {nm} at first close (cm)"); ax[k].legend(fontsize=8)
        sl, r2 = res["grasp_vs_cube_slope_r2_xy"][k] if res["grasp_vs_cube_slope_r2_xy"] else (np.nan, np.nan)
        ax[k].set_title(f"{nm}: slope {sl}, R² {r2}  (green=success, red=fail)")
    fig.suptitle(f"{a.cond}, N={a.N} demos: does the grasp follow the cube?  SR={succ.mean():.0%}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "grasp_vs_cube.png"), dpi=90)
    # clips
    si, fi = [i for i in range(len(R)) if succ[i]][:3], [i for i in range(len(R)) if not succ[i]][:3]
    for tag, ids in (("success", si), ("fail", fi)):
        for j, i in enumerate(ids):
            r = R[i]
            ge = (np.linalg.norm(r["grasp"][:2] - r["cube"][:2]) * 100) if r["grasp"] is not None else None
            info = f"cube ({r['cube'][0]*100:.0f},{r['cube'][1]*100:.0f})cm  target ({r['target'][0]*100:.0f},{r['target'][1]*100:.0f})cm  grasp error {ge:.1f}cm" if ge is not None else "never closed the gripper"
            write_mp4(r["frames"], os.path.join(out, f"{tag}_{j+1}.mp4"), f"{a.cond} N={a.N}: {tag.upper()} (rollout {i})", info)
    print("clips:", out)


if __name__ == "__main__":
    main()
