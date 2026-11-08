import sys, os
sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from sim_to_real_so101.assets.so101 import SO101_CFG
@configclass
class C(InteractiveSceneCfg):
    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
sc = InteractiveScene(C(num_envs=1, env_spacing=2.0)); sim.reset()
r = sc["robot"]
print("BODIES", r.body_names); print("JOINTS", r.joint_names)
print("LIMITS", r.data.soft_joint_pos_limits[0].cpu().numpy().round(3).tolist())
print("ROOT", r.data.root_pos_w[0].cpu().numpy().round(3), r.data.root_quat_w[0].cpu().numpy().round(3))
for _ in range(60):
    sc.write_data_to_sim(); sim.step(); sc.update(1/120)
for n in r.body_names:
    i = r.body_names.index(n); print("BODY", n, r.data.body_pos_w[0, i].cpu().numpy().round(4).tolist(), r.data.body_quat_w[0, i].cpu().numpy().round(3).tolist())
print("Q", r.data.joint_pos[0].cpu().numpy().round(3).tolist())
J = r.root_physx_view.get_jacobians(); print("JAC", tuple(J.shape))
from pxr import Usd
st = sim.stage
for p in st.Traverse():
    s = str(p.GetPath())
    if s.startswith("/World/envs/env_0/Robot") and s.count("/") <= 6: print("PRIM", s, p.GetTypeName())
sys.stdout.flush(); os._exit(0)
