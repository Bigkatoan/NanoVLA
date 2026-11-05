import sys, os, json
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import tabletop as tt
env = tt.Tabletop(num_envs=4, seed=1)
eps = [env.sample_episode() for _ in range(env.n)]
for E, t in zip(eps, ['checker', 'wood_oak', 'marble_black', 'cutting_mat']): E['table_mat'] = [i for i, p in enumerate(tt.TEX) if t in p][0]
eps = env.reset(episodes=eps)
for _ in range(15):
    env.step(np.tile(tt.HOME, (env.n, 1)))
f, w = env.images()
rows = [np.concatenate([f[i], w[i]], 1) for i in range(env.n)]
Image.fromarray(np.concatenate(rows, 0)).resize((640, 960)).save("out/scene_grid.png")
hf = env.scene["front"].data.output["rgb"][0, ..., :3].cpu().numpy(); Image.fromarray(hf).save("out/scene_front_hi.png")
for E in eps: print("EP", E["task"], "|", E["text"], "| objs", E["present"], E["pads"], E["tray"], "tex", tt.TEX[E["table_mat"]].split("/")[-1])
print("tray center", env.tray_center(0), "pos", {k: env.pos(k)[0].round(3).tolist() for k in eps[0]["present"]})
print("TCP-ish gripper body", env.robot.data.body_pos_w[0, env.robot.body_names.index("gripper")].cpu().numpy() - env.origins[0].cpu().numpy())
print("Q", env.state()[0].round(3))
sys.stdout.flush(); os._exit(0)
