"""Data for NanoIK (learned differential IK of the SO-101, wrist-camera frame).

A robot-only scene (no cameras, no objects, N arms). For random joint configurations q (half drawn around the joint
states of the recorded demos, half uniform in the limits) and small random arm-joint changes dq, kinematics only
(write state + forward, no dynamics) give the grasp-centre displacement expressed in the WRIST-CAMERA frame at q and
the change of gripper yaw. NanoIK learns (q, dp_cam, dyaw) -> dq.

    python ik_data.py --samples 2000000 --out ~/VLA/data/nano_ik/ik.npz
"""
import argparse, glob, json, math, os, sys, time
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--samples", type=int, default=2_000_000); ap.add_argument("--envs", type=int, default=512)
ap.add_argument("--dq", type=float, default=0.04); ap.add_argument("--out", default=os.path.expanduser("~/VLA/data/nano_ik/ik.npz")); ap.add_argument("--check", type=int, default=1)
a = ap.parse_args()
app = AppLauncher(headless=True).app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))
import numpy as np, torch


def main():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass, math as mu
    from sim_to_real_so101.assets.so101 import SO101_CFG
    from tabletop_expert import FIXED_TIP, MOVING_TIP

    @configclass
    class Cfg(InteractiveSceneCfg):
        robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dev = "cuda"
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=dev))
    scene = InteractiveScene(Cfg(num_envs=a.envs, env_spacing=1.0)); sim.reset()
    rob = scene["robot"]; n = a.envs; o = scene.env_origins
    bi, ji = rob.body_names.index("gripper"), rob.body_names.index("jaw")
    tip_f = torch.tensor(FIXED_TIP, device=dev).repeat(n, 1); tip_m = torch.tensor(MOVING_TIP, device=dev).repeat(n, 1)
    lim = rob.data.soft_joint_pos_limits[0]                      # [6, 2]
    t_loc = torch.tensor([-0.005, 0.06, -0.062], device=dev); q_loc = torch.tensor([0.92388, -0.38268, 0.0, 0.0], device=dev)
    FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=dev))

    def kin(q):
        rob.write_joint_state_to_sim(q, torch.zeros_like(q)); sim.forward(); rob.update(0.0)
        pb, qb = rob.data.body_pos_w[:, bi] - o, rob.data.body_quat_w[:, bi]; pj, qj = rob.data.body_pos_w[:, ji] - o, rob.data.body_quat_w[:, ji]
        fixed = pb + mu.quat_apply(qb, tip_f); moving = pj + mu.quat_apply(qj, tip_m)
        tcp = torch.cat([((fixed + moving) / 2)[:, :2], fixed[:, 2:3]], 1)
        Rb = mu.matrix_from_quat(qb); Rc = Rb @ mu.matrix_from_quat(q_loc[None].expand(n, -1)) @ FLIP
        yaw = torch.atan2(Rb[:, 1, 0], Rb[:, 0, 0])
        return tcp, Rc, yaw

    # joint states of recorded demos: realistic configurations
    demo = []
    for f in sorted(glob.glob(os.path.expanduser("~/VLA/data/isaac_gct/*_c*/actions/*.npz")))[:10]:
        z = np.load(f); demo += [z[k][::5] for k in z.files if k.endswith("_state")]
    demo = torch.tensor(np.concatenate(demo), device=dev, dtype=torch.float32); print(json.dumps(dict(demo_states=len(demo))), flush=True)
    if a.check:                                                   # forward() must give the same pose as a settled physics step
        q = rob.data.default_joint_pos.clone(); t1, _, _ = kin(q)
        rob.set_joint_position_target(q)
        for _ in range(30):
            sim.step(render=False); rob.update(sim.get_physics_dt())
        t2, _, _ = kin(rob.data.joint_pos.clone())
        print(json.dumps(dict(check_forward_vs_step_mm=round(float((t1 - t2).norm(dim=1).max()) * 1000, 3))), flush=True)
    Q, P, Y, D = [], [], [], []
    t0 = time.time(); got = 0
    while got < a.samples:
        half = n // 2
        q = torch.empty(n, 6, device=dev)
        q[:half] = demo[torch.randint(len(demo), (half,), device=dev)] + 0.15 * torch.randn(half, 6, device=dev)
        q[half:] = lim[:, 0] + (lim[:, 1] - lim[:, 0]) * torch.rand(n - half, 6, device=dev)
        q = torch.maximum(torch.minimum(q, lim[:, 1]), lim[:, 0])
        dq = torch.zeros_like(q); dq[:, :5] = a.dq * torch.randn(n, 5, device=dev) * torch.rand(n, 1, device=dev).sqrt()
        q2 = torch.maximum(torch.minimum(q + dq, lim[:, 1]), lim[:, 0]); dq = q2 - q
        t1, Rc, y1 = kin(q); t2, _, y2 = kin(q2)
        dp = torch.bmm(Rc.transpose(1, 2), (t2 - t1)[..., None])[..., 0]
        dy = (y2 - y1 + math.pi) % (2 * math.pi) - math.pi
        Q.append(q.cpu()); P.append(dp.cpu()); Y.append(dy.cpu()); D.append(dq[:, :5].cpu()); got += n
        if got % (n * 200) == 0:
            print(json.dumps(dict(samples=got, s=round(time.time() - t0))), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, q=torch.cat(Q).numpy(), dp_cam=torch.cat(P).numpy(), dyaw=torch.cat(Y).numpy(), dq=torch.cat(D).numpy())
    print(json.dumps(dict(saved=a.out, samples=got, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)
