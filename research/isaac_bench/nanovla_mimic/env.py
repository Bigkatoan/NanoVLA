"""Manager-based SO-101 tabletop env + Isaac Lab Mimic wrapper (see __init__.py)."""
import math, os, sys
from collections.abc import Sequence

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tabletop as tt                                   # noqa: E402
from tabletop_expert import FIXED_TIP, OPEN, CLOSE       # noqa: E402

import isaaclab.utils.math as PoseUtils                  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv   # noqa: E402
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig       # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm                  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup            # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm              # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm             # noqa: E402
from isaaclab.utils import configclass                                   # noqa: E402
import isaaclab.envs.mdp as mdp                                          # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg   # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg                                   # noqa: E402

ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
# Mimic end-effector = grasp centre with the jaw open: fixed-jaw tip shifted ~3.9 cm along the gripper x axis
# (measured between the fixed and moving tip meshes at the open position); a constant offset on the gripper body.
EEF_OFFSET = (FIXED_TIP[0] + 0.039, FIXED_TIP[1], FIXED_TIP[2])


# ------------------------------------------------------------------ tabletop helpers on top of the manager env
class TabletopView(tt.Tabletop):
    """Reuses the tabletop sampler / pose writer / success checks on a manager-based env (no own SimulationContext)."""

    def __init__(self, env, task, obj, target, seed=0):
        self.torch, self.env = torch, env
        self.n, self.dev = env.num_envs, env.device
        self.scene = env.scene; self.robot = env.scene["robot"]
        self.objs = {k: env.scene[k] for k in tt.GRASP + tt.CONTAINERS + tt.DISTRACT + ["tray"] + [f"{c}_pad" for c in tt.PADS]}
        self.origins = env.scene.env_origins
        self.rng = np.random.default_rng(seed)
        self.dec = 1; self.task_mix = (task,); self.table_mats = [0]
        self.want = (obj, target)
        self.ep = [None] * self.n; self.t = np.zeros(self.n, int)
        self.home = torch.tensor(tt.HOME, device=self.dev, dtype=torch.float32)

    def _show_skin(self, i, k):
        pass

    def _randomize(self, ids, episodes):
        pass

    def sample_fixed(self):
        obj, target = self.want
        for _ in range(2000):
            E = self.sample_episode()
            if E["obj"] != obj:
                continue
            if target and E["task"] == "pad" and E.get("pad") != target:
                continue
            return E
        raise RuntimeError(f"could not sample a layout with {obj} -> {target}")


def _view(env):
    return env._tabletop_view


# ------------------------------------------------------------------ MDP terms
def reset_layout(env, env_ids):
    v = _view(env)
    ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
    v.reset(env_ids=ids, episodes=[v.sample_fixed() for _ in ids], settle=0)


def eef_pose_env(env):
    """end-effector (grasp centre) pose in the env frame: (N, 7) pos + quat(w,x,y,z)"""
    r = env.scene["robot"]; bi = r.body_names.index("gripper")
    q = r.data.body_quat_w[:, bi]
    p = r.data.body_pos_w[:, bi] + PoseUtils.quat_apply(q, torch.tensor(EEF_OFFSET, device=env.device).repeat(env.num_envs, 1)) - env.scene.env_origins
    return torch.cat([p, q], 1)


def eef_pos(env):
    return eef_pose_env(env)[:, :3]


def eef_quat(env):
    return eef_pose_env(env)[:, 3:]


def joint_target(env):
    return env.scene["robot"].data.joint_pos_target


def cam_rgb(env, cam):
    x = env.scene[cam].data.output["rgb"][..., :3].permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(x, size=tt.RES, mode="area")
    return x.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)


def grasped(env):
    v = _view(env); out = torch.zeros(env.num_envs, device=env.device)
    jaw = env.scene["robot"].data.joint_pos[:, 5]
    for i, E in enumerate(v.ep):
        if E is None:
            continue
        m = tt.OBJ_META[E["obj"]]; bottom = float(v.objs[E["obj"]].data.root_pos_w[i, 2] - env.scene.env_origins[i, 2]) + m["zmin"]
        out[i] = float(bottom > 0.03 and float(jaw[i]) < 0.45)
    return out


def task_success(env):
    return torch.tensor(_view(env).success(), device=env.device)


# ------------------------------------------------------------------ cfg factory
def make_cfg_class(task, obj, target, num_envs=1):
    scene = tt.scene_cfg(num_envs)

    @configclass
    class Actions:
        arm = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=ARM, body_name="gripper", scale=1.0,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=EEF_OFFSET))
        jaw = JointPositionActionCfg(asset_name="robot", joint_names=["Jaw"], scale=1.0, use_default_offset=False)

    @configclass
    class Obs:
        @configclass
        class Policy(ObsGroup):
            joint_pos = ObsTerm(func=mdp.joint_pos)
            joint_target = ObsTerm(func=joint_target)
            eef_pos = ObsTerm(func=eef_pos)
            eef_quat = ObsTerm(func=eef_quat)
            front = ObsTerm(func=cam_rgb, params={"cam": "front"})
            wrist = ObsTerm(func=cam_rgb, params={"cam": "wrist"})

            def __post_init__(self):
                self.enable_corruption = False; self.concatenate_terms = False

        @configclass
        class Subtask(ObsGroup):
            grasp = ObsTerm(func=grasped)

            def __post_init__(self):
                self.enable_corruption = False; self.concatenate_terms = False

        policy: Policy = Policy()
        subtask_terms: Subtask = Subtask()

    @configclass
    class Events:
        reset = EventTerm(func=reset_layout, mode="reset")

    @configclass
    class Terms:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        success = DoneTerm(func=task_success)

    @configclass
    class Cfg(ManagerBasedRLEnvCfg, MimicEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene = scene; self.actions = Actions(); self.observations = Obs(); self.events = Events(); self.terminations = Terms()
            self.rewards = None; self.commands = None; self.curriculum = None
            self.decimation = tt.PHYS_HZ // tt.CTRL_HZ; self.episode_length_s = 40.0
            self.sim.dt = 1 / tt.PHYS_HZ; self.sim.render_interval = self.decimation
            self.nanovla = dict(task=task, obj=obj, target=target)
            self.datagen_config.name = f"nanovla_{task}_{obj}"
            self.datagen_config.generation_guarantee = True
            self.datagen_config.generation_keep_failed = False
            self.datagen_config.generation_num_trials = 50
            self.datagen_config.generation_select_src_per_subtask = True
            self.datagen_config.generation_transform_first_robot_pose = False
            self.datagen_config.generation_interpolate_from_last_target_pose = True
            self.datagen_config.max_num_failures = 50
            self.datagen_config.seed = 1
            sub = [SubTaskConfig(object_ref=obj, subtask_term_signal="grasp" if task != "pick" else None, subtask_term_offset_range=(5, 10),
                                 selection_strategy="nearest_neighbor_object", selection_strategy_kwargs={"nn_k": 3}, action_noise=0.003,
                                 num_interpolation_steps=8, num_fixed_steps=0, apply_noise_during_interpolation=False)]
            if task != "pick":
                sub.append(SubTaskConfig(object_ref=target, subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                                         selection_strategy="nearest_neighbor_object", selection_strategy_kwargs={"nn_k": 3}, action_noise=0.003,
                                         num_interpolation_steps=8, num_fixed_steps=0, apply_noise_during_interpolation=False))
            self.subtask_configs["so101"] = sub

    Cfg.__name__ = f"SO101_{task}_{obj}_MimicCfg"
    return Cfg


# ------------------------------------------------------------------ Mimic env
class SO101TabletopMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper: eef poses in the env frame (same frame as Mimic's object poses); actions in the robot root frame."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        nv = cfg.nanovla
        self._tabletop_view = None
        # the reset event needs the view; it is created lazily on the first reset via load_managers hook below
        self._nv = nv
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self):
        self._tabletop_view = TabletopView(self, self._nv["task"], self._nv["obj"], self._nv["target"], seed=int(self.cfg.seed or 0))
        super().load_managers()

    def _root(self):
        r = self.scene["robot"]
        return r.data.root_pos_w - self.scene.env_origins, r.data.root_quat_w

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        p = self.obs_buf["policy"]["eef_pos"][env_ids]; q = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(p, PoseUtils.matrix_from_quat(q))

    def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict, noise: float | None = None, env_id: int = 0) -> torch.Tensor:
        (pose,) = target_eef_pose_dict.values(); pos, rot = PoseUtils.unmake_pose(pose); quat = PoseUtils.quat_from_matrix(rot)
        rp, rq = self._root()
        pb, qb = PoseUtils.subtract_frame_transforms(rp[env_id:env_id + 1], rq[env_id:env_id + 1], pos[None], quat[None])
        act = torch.cat([pb[0], qb[0]], 0)
        if noise is not None:
            act[:3] += noise * torch.randn_like(act[:3])
        (g,) = gripper_action_dict.values()
        return torch.cat([act, g], 0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        rp, rq = self._root(); n = action.shape[0]
        p, q = PoseUtils.combine_frame_transforms(rp[:n], rq[:n], action[:, :3], action[:, 3:7])
        return {"so101": PoseUtils.make_pose(p, PoseUtils.matrix_from_quat(q))}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"so101": actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        return {"grasp": self.obs_buf["subtask_terms"]["grasp"][env_ids]}


_CFG_CACHE = {}


def __getattr__(name):
    """module-level lazy attributes Cfg__<task>__<obj>__<target> (used by the gym registry entry points)"""
    if name.startswith("Cfg__"):
        if name not in _CFG_CACHE:
            _, task, obj, target = name.split("__")
            _CFG_CACHE[name] = make_cfg_class(task, obj, None if target == "none" else target)
        return _CFG_CACHE[name]
    raise AttributeError(name)
