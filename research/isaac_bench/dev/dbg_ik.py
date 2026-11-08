import sys, os
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import tabletop as tt, tabletop_expert as te
env = tt.Tabletop(num_envs=1, seed=0); env.reset()
for _ in range(20): env.step(tt.HOME[None])
ex = te.Expert(env, np.random.default_rng(0))
p, q = ex.tcp_pose(); from isaaclab.utils import math as mu
R = mu.matrix_from_quat(q)[0].cpu().numpy(); print("HOME tcp", p[0].cpu().numpy().round(3), "gripper axes x,y,z cols\n", R.round(2))
J = ex.jac()[0].cpu().numpy(); print("J\n", J.round(3))
# numerical check of position Jacobian: perturb joints via write_joint_state
q0 = env.robot.data.joint_pos.clone()
for j in range(5):
    dq = q0.clone(); dq[0, j] += 0.02
    env.robot.write_joint_state_to_sim(dq, torch.zeros_like(dq)); env.sim.forward() if hasattr(env.sim, "forward") else None
    env.scene.update(0.0); p2, _ = ex.tcp_pose(); print("num col", j, ((p2 - p)[0].cpu().numpy() / 0.02).round(3))
env.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
sys.stdout.flush(); os._exit(0)
