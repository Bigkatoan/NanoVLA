"""Scripted expert demos for the SO-101 MuJoCo pick-and-place task, written in the few-shot bench format.

Output dir (default data/sim_bench/<task>): frames/<repo>.parquet (ep, cam, fidx, jpeg, task, task_orig, fps)
                                            actions/<repo>.npz (ep<k>_action, ep<k>_state, rad, 30 Hz)
Only successful expert episodes are kept (the expert's own success rate is reported).
"""
import argparse, os, io, json
import numpy as np, pandas as pd
from PIL import Image
import sim_env as se

TASKS = {
    "pick_red_to_blue": dict(n_cubes=1, text="Pick up the red cube and place it in the blue square."),
}


def _unused_lerp_path(env, q_start, p0, p1, grip0, grip1, T, rng):
    """straight Cartesian line p0->p1 in T control steps; returns list of joint targets"""
    out, q = [], q_start.copy()
    for k in range(1, T + 1):
        s = k / T
        s = 3 * s ** 2 - 2 * s ** 3  # smoothstep
        p = p0 + (p1 - p0) * s
        g = grip0 + (grip1 - grip0) * s
        q, _ = env.ik(p, q, g, iters=30)
        q[4] = q_start[4]
        out.append(np.r_[q[:5], g])
    return out, q


def site_axes(env, q, grip):
    import mujoco
    d2 = mujoco.MjData(env.m); d2.qpos[:] = env.d.qpos; d2.qpos[env.qadr] = np.r_[q[:5], grip]
    mujoco.mj_kinematics(env.m, d2)
    return d2.site_xmat[env.site].reshape(3, 3)


def grasp_roll(env, q, grip, yaw):
    """adjust wrist roll so the jaw closing axis (site z) is parallel to a cube face normal"""
    q = q.copy()
    for _ in range(3):
        z = site_axes(env, q, grip)[:, 2]
        phi = np.arctan2(z[1], z[0])
        d = (yaw - phi + np.pi / 4) % (np.pi / 2) - np.pi / 4
        q2 = q.copy(); q2[4] += 0.05
        z2 = site_axes(env, q2, grip)[:, 2]
        sgn = np.sign(np.arctan2(z2[1], z2[0]) - phi) or 1.0
        q[4] = np.clip(q[4] + sgn * d, -2.7, 2.8)
    return q


def expert_episode(env, rng, record=True):
    env.reset()
    q = env.state()
    c, t = env.cube_pos(), env.target_pos()
    sp = rng.uniform(0.8, 1.25)
    j = lambda: rng.normal(0, 0.003, 3)
    OPEN, CLOSE = 1.3 + rng.uniform(-0.1, 0.15), -0.17
    ja = env.m.joint(f"cube{env.cube_idx}_free").qposadr[0]
    qw = env.d.qpos[ja + 3:ja + 7]; yaw = 2 * np.arctan2(qw[3], qw[0])
    above = c + [0, 0, 0.08]
    qa, _ = env.ik(above, q, OPEN, iters=200)
    qa = grasp_roll(env, qa, OPEN, yaw)
    zs = site_axes(env, qa, OPEN)[:, 2]
    zs[2] = 0; zs /= np.linalg.norm(zs) + 1e-9
    off = -(se.CUBE + 0.004) * zs                       # fixed jaw 4 mm beside one block face (no push on descent)
    g_pre = c + off + [0, 0, 0.08] + j()
    g_at = c + off + [0, 0, -(se.CUBE_H - 0.012)] + j() * 0.3   # site ~12 mm above the table
    lift = c + off + [0, 0, 0.10]
    t_hi = "PLACE_HI"   # resolved after the grasp from the measured block offset
    t_lo = "PLACE_LO"
    segs = [("move", g_pre, OPEN, int(38 * sp)), ("move", g_at, OPEN, int(24 * sp)), ("grip", None, CLOSE, int(14 * sp)),
            ("hold", None, CLOSE, 8), ("move", lift, CLOSE, int(34 * sp)), ("move", t_hi, CLOSE, int(36 * sp)),
            ("move", t_lo, CLOSE, int(18 * sp)), ("grip", None, OPEN, int(12 * sp)), ("move", "RETREAT", OPEN, int(20 * sp)), ("hold", None, OPEN, 12)]
    states, actions, frames = [], [], []
    qcur = np.r_[q[:5], 0.0]
    roll_target = qa[4]
    p_cur, g_cur = env.ee_pos(), 0.0
    place = {}
    for kind, p1, g1, T in segs:
        p0, g0 = p_cur.copy(), g_cur
        if isinstance(p1, str):
            if not place:
                delta = env.cube_pos() - env.ee_pos()          # block centre relative to the gripper site (world)
                hi = t - delta + [0, 0, 0]; hi[2] = env.ee_pos()[2] + 0.0
                lo = t - delta; lo[2] = (se.CUBE_H + 0.004) - delta[2]
                place = dict(PLACE_HI=np.r_[(t - delta)[:2] + rng.normal(0, 0.003, 2), 0.13], PLACE_LO=lo, RETREAT=lo + [0, 0, 0.08])
            p1 = place[p1]
        for k in range(1, T + 1):
            s_ = k / T; s_ = 3 * s_ ** 2 - 2 * s_ ** 3
            g = g0 + (g1 - g0) * s_
            if kind == "move":
                p = p0 + (np.asarray(p1) - p0) * s_
                qn, _ = env.ik(p, qcur, g, iters=30)
                qn[4] = qcur[4] + (roll_target - qcur[4]) * min(1.0, s_ * 2) if len(actions) < 40 else roll_target
                qcur = np.r_[qn[:5], g]
            else:
                qcur = np.r_[qcur[:5], g]
            states.append(env.state()); actions.append(qcur.copy())
            if record and env.t % 3 == 0:
                frames.append((env.t, env.render("front"), env.render("wrist")))
            env.step(qcur)
        if kind == "move":
            p_cur = np.asarray(p1, float)
        g_cur = g1
    return env.success(), np.array(states, np.float32), np.array(actions, np.float32), frames


def jpg(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=90); return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_red_to_blue")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/VLA/data/sim_bench"))
    a = ap.parse_args()
    cfg = TASKS[a.task]
    env = se.SO101Env(n_cubes=cfg["n_cubes"], seed=a.seed)
    rng = np.random.default_rng(a.seed + 1000)
    rows, acts, tried, ok = [], {}, 0, 0
    while ok < a.n:
        tried += 1
        succ, S, A, fr = expert_episode(env, rng)
        if not succ:
            continue
        e = ok; ok += 1
        acts[f"ep{e}_action"], acts[f"ep{e}_state"] = A, S
        for t, f, w in fr:
            rows.append((e, "front", t, jpg(f), cfg["text"], cfg["text"]))
            rows.append((e, "wrist", t, jpg(w), cfg["text"], cfg["text"]))
        if ok % 10 == 0:
            print(f"{ok}/{tried} expert success {ok / tried:.2f}", flush=True)
    env.close()
    repo = f"sim/{a.task}"
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True); os.makedirs(os.path.join(a.out, "actions"), exist_ok=True)
    d = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig"]); d["fps"] = 30
    d.to_parquet(os.path.join(a.out, "frames", repo.replace("/", "__") + ".parquet"))
    np.savez_compressed(os.path.join(a.out, "actions", repo.replace("/", "__") + ".npz"), **acts)
    json.dump([repo], open(os.path.join(a.out, "repos.json"), "w"))
    print(f"done: {ok} demos, expert success rate {ok / tried:.3f} ({tried} tried)")


if __name__ == "__main__":
    main()
