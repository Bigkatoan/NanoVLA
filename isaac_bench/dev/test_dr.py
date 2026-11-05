"""Test of isaac_bench/dr.py: per-env DR mechanisms, random base pose (teleport + kinematics), continuous variation,
and a scripted pick (traj_core.TrajRunner + specs/pick.json) with a moved / rotated base.
    ~/VLA/tools/gpu.sh 600 python -u isaac_bench/dev/test_dr.py [--rounds 4] [--picks 2]
Writes isaac_bench/out/dr/*.png and prints a JSON summary."""
import os, sys, json, time, math, argparse, traceback
os.environ.setdefault("TABLETOP_DR", "0")          # dr.py replaces Tabletop._randomize
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--picks", type=int, default=2)
ap.add_argument("--seed", type=int, default=3); ap.add_argument("--probe", type=int, default=1); ap.add_argument("--aa", type=int, default=-1, help="set /rtx/post/aa/op (2 = FXAA, no temporal ghosting)")
a = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, torch
from PIL import Image, ImageDraw
OUT = os.path.join(HERE, "out", "dr"); os.makedirs(OUT, exist_ok=True)


def main():
    import tabletop as tt
    import dr as DR
    from traj_core import TrajRunner
    t0 = time.time()
    env = tt.Tabletop(num_envs=2, seed=a.seed, task_mix=("pick",))
    rng = np.random.default_rng(a.seed + 100)
    dr = DR.DomainRandomizer(env, rng)
    print("init", round(time.time() - t0, 1), "s", flush=True)
    import carb
    cs = carb.settings.get_settings()
    print("AA", {k: cs.get(k) for k in ["/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/rendermode", "/rtx/directLighting/sampledLighting/enabled"]}, flush=True)
    if a.aa >= 0:
        cs.set("/rtx/post/aa/op", a.aa)
    summary = {}
    home = np.tile(tt.HOME, (env.n, 1))
    wrist = lambda: env.scene["wrist"].data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
    front = lambda: env.scene["front"].data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    def overview_cam():
        eyes = env.origins + torch.tensor([0.25, -0.85, 0.95], device=env.dev)
        tg = env.origins + torch.tensor([0.15, 0.0, 0.0], device=env.dev)
        env.scene["front"].set_world_poses_from_view(eyes, tg)

    def sample(i):
        while True:
            pose = dr.sample_base_pose()
            E = dr.transform_episode(env.sample_episode(), pose)
            if E is not None:
                return pose, E

    def do_reset(ids, poses, eps):
        dr.apply_base_pose(ids, poses)
        env.reset(env_ids=ids, episodes=eps)
        for i, E in zip(ids, eps):
            env.ep[i] = E; env.t[i] = 0          # (Tabletop.reset only sets ep of the last env)
        return dr.randomize_episode(ids, eps)

    def steps(k, cmd=None):
        for _ in range(k):
            env.step(home if cmd is None else cmd)

    def label(im, txt):
        im = Image.fromarray(im); d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0)); d.text((3, 1), txt, fill=(255, 255, 255)); return np.asarray(im)

    def grid(rows, path):
        h = max(r_.shape[0] for row in rows for r_ in row)
        R = [np.concatenate([np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0))) for x in row], 1) for row in rows]
        w = max(r_.shape[1] for r_ in R); R = [np.pad(r_, ((0, 0), (0, w - r_.shape[1]), (0, 0))) for r_ in R]
        Image.fromarray(np.concatenate(R, 0)).save(path); print("saved", path, flush=True)

    # ------------------------------------------------ 0. global DR + nominal reference
    g = dr.randomize_global(); print("GLOBAL", json.dumps(g), flush=True)
    pose0 = [DR.NOMINAL_BASE] * 2
    eps = []
    for i in range(2):
        while True:
            E = dr.transform_episode(env.sample_episode(), DR.NOMINAL_BASE)
            if E: eps.append(E); break
    do_reset([0, 1], pose0, eps); overview_cam(); steps(20)
    gi = env.robot.body_names.index("gripper")
    def grip_in_base(i):
        p = (env.robot.data.body_pos_w[i, gi] - env.origins[i]).cpu().numpy()
        return np.r_[DR.to_base_frame(p[:2], dr.base[i]), p[2]]
    ref = grip_in_base(0)
    def jac_fd(tag):
        J = env.robot.root_physx_view.get_jacobians()[:, gi - 1, :3, :5].clone()
        q0 = env.robot.data.joint_pos.clone(); p0 = env.robot.data.body_pos_w[:, gi].clone()
        cmd = q0.clone(); cmd[:, :5] += torch.tensor([0.03, -0.025, 0.025, 0.02, 0.0], device=env.dev)
        for _ in range(40):
            env.step(cmd.cpu().numpy())
        dp = env.robot.data.body_pos_w[:, gi] - p0; dqa = env.robot.data.joint_pos[:, :5] - q0[:, :5]
        pred = torch.bmm(J, dqa.unsqueeze(-1)).squeeze(-1)
        r_ = dict(actual_mm=(1000 * dp).round().tolist(), pred_mm=(1000 * pred).round().tolist(),
                  rel_err=[round(float((dp[i] - pred[i]).norm() / dp[i].norm()), 3) for i in range(2)])
        print("JAC", tag, r_, flush=True); summary["jacobian_fd_" + tag] = r_
        steps(30)
    jac_fd("nominal")
    print("gripper in base frame (nominal)", ref.round(4).tolist(), flush=True)

    # ------------------------------------------------ 1. per-env mechanism probe: change env 0 only, measure image change
    if a.probe:
        from pxr import Gf
        def diff(fn, undo, name, n=4):
            steps(n); w0, f0 = wrist().astype(float), front().astype(float)
            fn(); steps(n); w1, f1 = wrist().astype(float), front().astype(float)
            res = dict(wrist_env0=round(float(np.abs(w1[0] - w0[0]).mean()), 2), wrist_env1=round(float(np.abs(w1[1] - w0[1]).mean()), 2),
                       front_env0=round(float(np.abs(f1[0] - f0[0]).mean()), 2), front_env1=round(float(np.abs(f1[1] - f0[1]).mean()), 2),
                       mean_env0=[round(float(w0[0].mean()), 1), round(float(w1[0].mean()), 1)])
            print("PROBE", name, res, flush=True); summary.setdefault("probe", {})[name] = res
            pr.append(label(np.concatenate([w0[0], w1[0]], 1).astype(np.uint8), name))
            undo(); steps(n)
        pr = []
        C = dr.cam[0]; C0 = dr.cam0[0]
        diff(lambda: C["cam"].GetFocalLengthAttr().Set(C0["f"] * 1.3), lambda: C["cam"].GetFocalLengthAttr().Set(C0["f"]), "focal x1.3")
        q = DR._quat_mul(DR._quat_rotvec([math.radians(8), 0, 0]), C0["q"])
        diff(lambda: C["o"].Set(C["otype"](*map(float, q))), lambda: C["o"].Set(C["otype"](*map(float, C0["q"]))), "cam mount rot 8deg")
        diff(lambda: C["t"].Set(Gf.Vec3d(*map(float, C0["t"] + [0, 0, 0.01]))), lambda: C["t"].Set(Gf.Vec3d(*map(float, C0["t"]))), "cam mount +1cm z")
        L = dr.lamp[0]; I0 = L.GetIntensityAttr().Get()
        diff(lambda: L.GetIntensityAttr().Set(I0 * 4), lambda: L.GetIntensityAttr().Set(I0), "lamp I x4")
        K0 = L.GetColorTemperatureAttr().Get()
        diff(lambda: L.GetColorTemperatureAttr().Set(2000.0), lambda: L.GetColorTemperatureAttr().Set(K0), "lamp 2000K")
        F = dr.fill[0]["light"]; fI0 = F.GetIntensityAttr().Get()
        diff(lambda: F.GetIntensityAttr().Set(20000.0), lambda: F.GetIntensityAttr().Set(fI0), "fill I 20000")
        sk = dr.skin_ops[0][eps[0]["table_mat"]]
        diff(lambda: sk["rz"].Set(90.0), lambda: sk["rz"].Set(0.0), "skin rotZ 90")
        cl = dr.clutter[0][0]
        def cl_on():
            cl["t"].Set(Gf.Vec3d(float(dr.base[0][0] + 0.17), 0.0, 0.03)); cl["s"].Set(Gf.Vec3f(0.06, 0.06, 0.06)); cl["col"].Set(Gf.Vec3f(0.9, 0.1, 0.1))
        diff(cl_on, lambda: None, "clutter red cube in view")
        diff(lambda: cl["col"].Set(Gf.Vec3f(0.1, 0.2, 0.9)), lambda: cl["t"].Set(Gf.Vec3d(-1.5, 0, -0.7)), "clutter colour -> blue")
        szk = sk["size"].Get(); sc_ = sk["s"].Get()
        diff(lambda: (sk["size"].Set(2.0 * szk), sk["s"].Set(Gf.Vec3f(*[v / 2 for v in sc_]))), lambda: (sk["size"].Set(szk), sk["s"].Set(sc_)), "skin tex scale x2 (Cube.size)")
        # focal of env 0 only: is env 1 affected (shared tiled render product)?
        steps(10); w0 = wrist(); C["cam"].GetFocalLengthAttr().Set(C0["f"] * 1.4); steps(6); w1 = wrist()
        C["cam"].GetFocalLengthAttr().Set(C0["f"]); steps(6); w2 = wrist()
        res = dict(env0=round(float(np.abs(w1[0].astype(float) - w0[0]).mean()), 2), env1=round(float(np.abs(w1[1].astype(float) - w0[1]).mean()), 2),
                   env1_back=round(float(np.abs(w2[1].astype(float) - w0[1]).mean()), 2))
        print("PROBE focal env0 only", res, flush=True); summary.setdefault("probe", {})["focal env0 x1.4 only"] = res
        grid([[label(w0[0], "e0 ref"), label(w1[0], "e0 f x1.4"), label(w0[1], "e1 ref"), label(w1[1], "e1 while e0 zoomed")]], os.path.join(OUT, "probe_focal.png"))
        grid([pr[k:k + 3] for k in range(0, len(pr), 3)], os.path.join(OUT, "probe_env0_before_after.png"))

    # ------------------------------------------------ 2. random base pose + full DR rounds
    bright = []
    rows_w, rows_o, base_err, kin_err, t_reset = [], [], [], [], []
    for rd in range(a.rounds):
        if rd == 2:
            g = dr.randomize_global(); print("GLOBAL", json.dumps(g), flush=True)
        pe = [sample(i) for i in range(2)]
        poses, eps = [p for p, _ in pe], [E for _, E in pe]
        tr = time.time(); meta = do_reset([0, 1], poses, eps); t_reset.append(time.time() - tr)
        overview_cam(); steps(15)
        W, O = wrist(), front()
        for i in range(2):
            rp = (env.robot.data.root_pos_w[i] - env.origins[i]).cpu().numpy(); rq = env.robot.data.root_quat_w[i].cpu().numpy()
            yaw = 2 * math.atan2(rq[3], rq[0]) - math.pi / 2
            base_err.append([float(np.linalg.norm(rp[:2] - poses[i][:2])), abs((yaw - poses[i][2] + math.pi) % (2 * math.pi) - math.pi)])
            kin_err.append(float(np.linalg.norm(grip_in_base(i) - ref)))
            print(f"ROUND {rd} env {i} base={np.round(poses[i], 3).tolist()} root={rp.round(3).tolist()} yaw={math.degrees(yaw):.1f} "
                  f"grip_in_base_err={kin_err[-1] * 1000:.2f}mm obj={eps[i]['obj']} q={env.state()[i].round(2).tolist()}", flush=True)
            print("DR", json.dumps(meta[i])[:1500], flush=True)
        bright.append([round(float(W[i].mean()), 1) for i in range(2)])
        rows_w.append([label(W[i], f"r{rd} e{i} yaw{math.degrees(poses[i][2]):+.0f} f{meta[i]['wrist_cam']['focal']}") for i in range(2)])
        rows_o.append([label(O[i], f"r{rd} e{i} base({poses[i][0]:+.2f},{poses[i][1]:+.2f})") for i in range(2)])
    grid(rows_w, os.path.join(OUT, "wrist_rounds.png")); grid(rows_o, os.path.join(OUT, "overview_rounds.png"))
    summary.update(base_err_max_m=max(e[0] for e in base_err), base_yaw_err_max_rad=max(e[1] for e in base_err),
                   grip_in_base_err_max_mm=1000 * max(kin_err), reset_total_s=[round(x, 2) for x in t_reset], wrist_mean_brightness=bright)

    # ------------------------------------------------ 3. Jacobian check at a moved base (finite difference of the gripper body)
    jac_fd("moved")

    # ------------------------------------------------ 4. continuous variation
    steps(5); strip = []
    for k in range(90):
        env.step(home); dr.step(env.t)
        if k % 15 == 0:
            strip.append(label(wrist()[0], f"t={env.t[0]}"))
    grid([strip], os.path.join(OUT, "continuous_env0.png"))
    print("dyn env0", {k: dr.dyn[0][k] for k in ("amp", "hz", "w", "flick_rate", "flicks")}, flush=True)

    # ------------------------------------------------ 5. pick with moved / rotated base
    spec = json.load(open(os.path.join(HERE, "specs", "pick.json")))
    res, prow = [], []
    for rd in range(a.picks):
        pe = [sample(i) for i in range(2)]
        poses, eps = [p for p, _ in pe], [E for _, E in pe]
        do_reset([0, 1], poses, eps); steps(10); env.t[:] = 0
        runner = TrajRunner(env, spec, np.random.default_rng(rd)); runner.reset()
        frames = []
        for k in range(900):
            cmd = runner.step(); env.step(cmd); dr.step(env.t)
            if k % 60 == 0 and len(frames) < 8:
                frames.append(label(wrist()[0], f"pick r{rd} t={k}"))
            if runner.done.all():
                break
        succ = env.success()
        for i in range(2):
            bot = env.pos(eps[i]["obj"])[i][2] + tt.OBJ_META[eps[i]["obj"]]["zmin"]
            res.append(dict(round=rd, env=i, base=[round(v, 3) for v in poses[i]], yaw_deg=round(math.degrees(poses[i][2]), 1), obj=eps[i]["obj"],
                            done=bool(runner.done[i]), success=bool(succ[i]), retries=int(runner.retries[i]), lift_cm=round(100 * float(bot), 1), steps=k + 1))
            print("PICK", res[-1], flush=True)
        prow.append(frames)
        ov = front(); prow.append([label(ov[i], f"pick r{rd} e{i} end succ={bool(succ[i])}") for i in range(2)])
    grid(prow, os.path.join(OUT, "pick_moved_base.png"))
    summary["pick"] = res; summary["pick_success_rate"] = float(np.mean([r_["success"] for r_ in res]))
    summary["overhead"] = dr.overhead(); summary["total_s"] = round(time.time() - t0, 1)
    print("SUMMARY", json.dumps(summary), flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush(); os._exit(1)
sys.stdout.flush(); os._exit(0)
