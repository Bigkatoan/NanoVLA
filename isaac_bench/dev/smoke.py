import argparse, time, sys, os
sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-Eval"); ap.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(ap); a = ap.parse_args(); a.enable_cameras = True; a.headless = True
app = AppLauncher(a).app
import gymnasium as gym, torch, numpy as np
from PIL import Image
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa
cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=a.num_envs)
env = gym.make(a.task, cfg=cfg)
u = env.unwrapped
print("joints", u.scene["robot"].joint_names, "dt", u.step_dt, "max_len", u.max_episode_length)
obs, _ = env.reset()
print({g: {k: tuple(v.shape) for k, v in o.items()} for g, o in obs.items()})
act = torch.tensor([[-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]], device="cuda:0").repeat(a.num_envs, 1)
t0 = time.time()
for i in range(60):
    obs, r, term, trunc, info = env.step(act)
print("60 steps", round(time.time() - t0, 2), "s")
for k in ("rgb_ego", "rgb_external_D455"):
    Image.fromarray(obs["visual"][k][0].cpu().numpy().astype(np.uint8)[..., :3]).save(os.path.expanduser(f"~/VLA/isaac_bench/out/{k}.png"))
print("subtask", {k: v.cpu().numpy().tolist() for k, v in obs.get("subtask_terms", {}).items()})
print("DONE"); sys.stdout.flush(); os._exit(0)
