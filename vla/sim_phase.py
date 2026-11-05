"""Phase analysis of closed-loop rollouts with a saved fine-tuned model: approach / grasp / transport / place."""
import os, json, sys, numpy as np, torch
import fewshot as fs, sim_env as se
from sim_fewshot import prep_img
from train_vla import H
TEXT = "Pick up the red cube and place it in the blue square."


@torch.no_grad()
def run(model, tok, row, P, env, dev, exec_steps=4, max_s=25.0):
    st, coef = P["st"], P["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    env.reset(); c0, tg = env.cube_pos(), env.target_pos()
    cur = env.state().copy(); ids = tok([TEXT]).to(dev)
    opened = closed_at = attach_t = None; min_d = 9; near_t = 0; n_close = 0; was_open = False
    while env.t < int(max_s * 30):
        S = env.state()
        ext, wri = prep_img(env.render("front"))[None].to(dev), prep_img(env.render("wrist"))[None].to(dev)
        s_n = torch.from_numpy(np.clip((S - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(ext, wri, torch.tensor([True], device=dev), s_n, ids, torch.tensor([row], device=dev)).float()[0].cpu().numpy()
        tgt = np.empty((H, 6)); tgt[:, :5] = S[:5] + p[:, :5] * st["dstd"]; tgt[:, 5] = (p[:, 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
        tgt = (tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)
        for k in range(exec_steps):
            for i in range(3):
                cmd = cur + (tgt[k] - cur) * (i + 1) / 3; env.step(cmd)
                d = np.linalg.norm(env.ee_pos()[:2] - env.cube_pos()[:2]); min_d = min(min_d, d)
                if d < 0.05 and env.attached is None: near_t += 1
                if cmd[5] > 0.8: was_open = True
                if was_open and cmd[5] < 0.4:
                    n_close += 1; was_open = False
                    if closed_at is None: closed_at = (env.ee_pos(), env.cube_pos())
                if env.attached is not None and attach_t is None: attach_t = env.t / 30
                if env.success():
                    return dict(success=True, cube=c0, target=tg, min_d=min_d, closed=closed_at, n_close=n_close, attach_t=attach_t, near_s=near_t / 30, t=env.t / 30)
            cur = tgt[k]
    return dict(success=False, cube=c0, target=tg, min_d=min_d, closed=closed_at, n_close=n_close, attach_t=attach_t, near_s=near_t / 30, t=max_s,
                final_cube_to_target=float(np.linalg.norm(env.cube_pos()[:2] - tg[:2])))


def analyse(R, name):
    n = len(R); s = [r for r in R if r["success"]]; f = [r for r in R if not r["success"]]
    reached = [r for r in R if r["min_d"] < 0.03]; grasped = [r for r in R if r["attach_t"] is not None]
    out = dict(name=name, n=n, success_rate=len(s) / n,
               reached_cube_within_3cm=len(reached) / n, closed_gripper_ever=sum(r["closed"] is not None for r in R) / n,
               grasped=len(grasped) / n, success_given_grasp=(len([r for r in grasped if r["success"]]) / len(grasped)) if grasped else None,
               fail_modes=dict(never_reached=sum(r["min_d"] >= 0.03 for r in f), reached_no_close=sum(r["min_d"] < 0.03 and r["closed"] is None for r in f),
                               closed_but_missed=sum(r["closed"] is not None and r["attach_t"] is None for r in f), grasped_but_not_placed=sum(r["attach_t"] is not None for r in f)),
               mean_hover_near_cube_s=dict(success=float(np.mean([r["near_s"] for r in s])) if s else None, fail=float(np.mean([r["near_s"] for r in f])) if f else None))
    cl = [r for r in R if r["closed"] is not None]
    if len(cl) > 3:
        G = np.array([r["closed"][0][:2] for r in cl]); C = np.array([r["closed"][1][:2] for r in cl])
        out["grasp_xy_err_cm"] = float(np.linalg.norm(G - C, axis=1).mean() * 100)
        out["grasp_tracks_cube_slope_xy"] = [round(float(np.polyfit(C[:, k], G[:, k], 1)[0]), 2) for k in range(2)]
    return out


if __name__ == "__main__":
    mdir, cond, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); n_roll = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    dev = "cuda"; CK = os.path.expanduser("~/VLA/ckpt")
    fs.ROOT_BENCH = os.path.expanduser("~/VLA/data/sim_bench"); P = fs.prepare("sim/pick_red_to_blue", N, 0, 0)
    os.environ["VLA_CKPT"] = f"{CK}/vla/vla.pt" if "v1" in cond or cond in ("B", "C") else f"{CK}/vla_v2/vla.pt"
    model, tok, row = fs.make_model("A" if cond.startswith("A") else cond[0], dev); model.load_state_dict(torch.load(os.path.join(mdir, "model.pt"))); model.eval()
    env = se.SO101Env(n_cubes=1, seed=10_000); R = [run(model, tok, row, P, env, dev) for _ in range(n_roll)]; env.close()
    import pickle; pickle.dump(R, open(os.path.join(mdir, "rollouts.pkl"), "wb"))
    res = analyse(R, f"{cond} N={N}"); print(json.dumps(res, indent=1, default=float)); json.dump(res, open(os.path.join(mdir, "phases.json"), "w"), indent=1, default=float)
