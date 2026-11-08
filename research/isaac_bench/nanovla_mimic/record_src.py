"""Record source demonstrations for Isaac Lab Mimic in the HDF5 format of Isaac Lab's record_demos.py.

The demos are driven by the node-based trajectory tool (spec templates in specs/), but through the manager-based
env's differential-IK pose action, so the recorded actions are end-effector poses that Mimic can re-target.
Only successful episodes are exported. Later, human teleoperation demos can replace these as Mimic sources.

    python nanovla_mimic/record_src.py --task NanoVLA-SO101-tray-marker-Mimic-v0 --spec specs/place.json --demos 10 \
        --out datasets/tray_marker_src.hdf5
"""
import argparse, json, os, sys
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--spec", default="specs/place.json")
ap.add_argument("--demos", type=int, default=10)
ap.add_argument("--out", default="datasets/src.hdf5")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=1200)
a = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))
import gymnasium as gym, numpy as np, torch
import isaaclab.utils.math as PoseUtils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg
import nanovla_mimic  # noqa: F401  (registers the tasks)
from traj_core import TrajRunner


def main():
    cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=1)
    cfg.seed = a.seed
    success_term = cfg.terminations.success; cfg.terminations.success = None      # judged here, as in record_demos.py
    out_dir, out_name = os.path.dirname(os.path.abspath(a.out)), os.path.splitext(os.path.basename(a.out))[0]
    cfg.recorders = ActionStateRecorderManagerCfg(); cfg.recorders.dataset_export_dir_path = out_dir; cfg.recorders.dataset_filename = out_name
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = gym.make(a.task, cfg=cfg).unwrapped
    spec = json.load(open(os.path.join(HERE, a.spec)))
    nv = cfg.nanovla; spec["tasks"] = [nv["task"]]
    view = env._tabletop_view
    runner = TrajRunner(view, spec, np.random.default_rng(a.seed))
    kept, tried = 0, 0
    env.reset()
    while kept < a.demos:
        runner.reset()
        ok = False
        for t in range(a.max_steps):
            pos, quat, grip = runner.targets()
            rp = env.scene["robot"].data.root_pos_w - env.scene.env_origins; rq = env.scene["robot"].data.root_quat_w
            pb, qb = PoseUtils.subtract_frame_transforms(rp, rq, torch.tensor(pos, device=env.device, dtype=torch.float32),
                                                         torch.tensor(quat, device=env.device, dtype=torch.float32))
            act = torch.cat([pb, qb, torch.tensor(grip, device=env.device, dtype=torch.float32)[:, None]], 1)
            env.step(act)
            eef = env.obs_buf["policy"]["eef_pos"]
            runner.err = torch.linalg.norm(eef - torch.tensor(pos, device=env.device, dtype=torch.float32), dim=1).cpu().numpy()
            if runner.done[0]:
                ok = bool(success_term.func(env, **success_term.params)[0]); break
        tried += 1
        if ok:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
            env.recorder_manager.export_episodes([0]); kept += 1
        print(json.dumps(dict(tried=tried, kept=kept, ok=ok, text=view.ep[0]["text"])), flush=True)
        env.recorder_manager.reset(); env.reset()
    print("saved", a.out, flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)   # Kit would hang on exit
    sys.stdout.flush(); os._exit(0)
