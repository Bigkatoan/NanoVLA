"""SO-101 tabletop workspace in Isaac Sim / Isaac Lab (batched, N parallel envs).

Scene per env: SO-101 (official TheRobotStudio URDF incl. wrist camera, imported with fixed joints kept),
table (top at z=0, robot base at the origin), back wall, red 4 cm cube (real PhysX contact grasp, no attachment
assist), thin blue target square (visual only), front + wrist RGB cameras (160x120, same poses/FOV as the MuJoCo
bench). Control at 30 Hz (physics 120 Hz x 4), actions = joint position targets (rad) in JOINTS order.

Must be imported after the Isaac Sim app is launched (see make_app()).
"""
import os, math
import numpy as np

URDF = os.path.expanduser("~/VLA/third_party/sim_assets/SO-ARM100/Simulation/SO101/so101_new_calib_camera.urdf")
USD_DIR = os.path.expanduser("~/VLA/isaac_bench/usd")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CUBE = 0.02            # half size (4 cm cube), same as MuJoCo bench
CTRL_HZ, PHYS_HZ = 30, 120
RES = (120, 160)
# MuJoCo camera poses (OpenGL convention, quat w,x,y,z) -- see sim_env.build
FRONT_POS, FRONT_ROT = (0.55, 0.0, 0.42), (0.66193, 0.2487, 0.2487, 0.66193)
WRIST_POS, WRIST_ROT = (-0.0175, -0.0121, 0.0278), (0.03111, 0.72345, 0.61154, 0.31885)


def make_app(headless=True):
    from isaaclab.app import AppLauncher
    return AppLauncher(headless=headless, enable_cameras=True).app


def _aperture(fovy_deg, f=24.0):
    """horizontal aperture (mm) for a pinhole with vertical FOV fovy at 4:3"""
    return 2 * f * math.tan(math.radians(fovy_deg) / 2) * RES[1] / RES[0]


def scene_cfg(num_envs, spacing=2.5, distractors=False):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass

    vis = lambda rgb, rough=0.6: sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=rough)
    static = lambda size, rgb, pos, coll=True: AssetBaseCfg(
        spawn=sim_utils.CuboidCfg(size=size, visual_material=vis(rgb), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=coll)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos))

    sts3215 = dict(stiffness=998.22, damping=2.731, effort_limit_sim=2.94, velocity_limit_sim=6.0, armature=0.028, friction=0.052)

    @configclass
    class Cfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.27)),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.76)))
        dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 0.98, 0.95)))
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=2200.0, angle=2.0),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=(0.92, 0.2, -0.32, 0.0)))
        # workbench: top (z in [-0.04, 0]), legs, back wall behind the robot
        table = static((1.0, 1.2, 0.04), (0.75, 0.70, 0.62), (0.25, 0.0, -0.02)).replace(prim_path="{ENV_REGEX_NS}/table")
        leg0 = static((0.05, 0.05, 0.72), (0.3, 0.3, 0.32), (-0.2, -0.55, -0.40), False).replace(prim_path="{ENV_REGEX_NS}/leg0")
        leg1 = static((0.05, 0.05, 0.72), (0.3, 0.3, 0.32), (-0.2, 0.55, -0.40), False).replace(prim_path="{ENV_REGEX_NS}/leg1")
        leg2 = static((0.05, 0.05, 0.72), (0.3, 0.3, 0.32), (0.7, -0.55, -0.40), False).replace(prim_path="{ENV_REGEX_NS}/leg2")
        leg3 = static((0.05, 0.05, 0.72), (0.3, 0.3, 0.32), (0.7, 0.55, -0.40), False).replace(prim_path="{ENV_REGEX_NS}/leg3")
        wall = static((0.03, 1.2, 0.8), (0.82, 0.83, 0.85), (-0.27, 0.0, 0.38), False).replace(prim_path="{ENV_REGEX_NS}/wall")

        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=URDF, usd_dir=USD_DIR, fix_base=True, merge_fixed_joints=False, self_collision=False,
                collider_type="convex_decomposition", make_instanceable=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=998.22, damping=2.731)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), joint_pos={j: 0.0 for j in JOINTS}),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=JOINTS, **sts3215)},
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            spawn=sim_utils.CuboidCfg(size=(2 * CUBE,) * 3, visual_material=vis((0.85, 0.1, 0.1), 0.5),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.3, friction_combine_mode="max", restitution=0.0),
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, max_depenetration_velocity=0.5),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=0.03), collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, CUBE + 0.001)))
        target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target",
            spawn=sim_utils.CuboidCfg(size=(0.08, 0.08, 0.001), visual_material=vis((0.1, 0.25, 0.9)),
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                                      collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.12, 0.0006)))
        front = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam", data_types=["rgb"], width=RES[1], height=RES[0],
            offset=TiledCameraCfg.OffsetCfg(pos=FRONT_POS, rot=FRONT_ROT, convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=_aperture(58), clipping_range=(0.01, 5.0)))
        wrist = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_camera_link/wrist_cam", data_types=["rgb"], width=RES[1], height=RES[0],
            offset=TiledCameraCfg.OffsetCfg(pos=WRIST_POS, rot=WRIST_ROT, convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=_aperture(80), clipping_range=(0.005, 5.0)))

    return Cfg(num_envs=num_envs, env_spacing=spacing, replicate_physics=True)


class SO101Workspace:
    """Batched env. All positions returned are relative to each env origin (robot base frame == world frame of MuJoCo)."""

    def __init__(self, num_envs=8, seed=0, device="cuda:0"):
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        self.torch = torch
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / PHYS_HZ, render_interval=PHYS_HZ // CTRL_HZ, device=device,
                                                                       physx=sim_utils.PhysxCfg(bounce_threshold_velocity=0.2, enable_ccd=False)))
        self.sim.set_camera_view([1.2, 0.9, 0.8], [0.15, 0.0, 0.05])
        self.scene = InteractiveScene(scene_cfg(num_envs))
        self.sim.reset()
        self.n, self.dev = num_envs, device
        self.robot, self.cube, self.target = self.scene["robot"], self.scene["cube"], self.scene["target"]
        self.jidx = [self.robot.joint_names.index(j) for j in JOINTS]
        self.origins = self.scene.env_origins
        self.rng = np.random.default_rng(seed)
        self.dec = PHYS_HZ // CTRL_HZ
        self.home = None
        self.t = np.zeros(num_envs, int)
        self.target_xy = np.zeros((num_envs, 2))

    # ------------------------------------------------------------------ layout
    def sample_layout(self):
        """cube + target in the reachable half-annulus in front of the robot (same distribution as the MuJoCo bench)"""
        pts = []
        while len(pts) < 2:
            r, a = self.rng.uniform(0.14, 0.26), self.rng.uniform(-0.9, 0.9)
            p = np.array([r * np.cos(a), r * np.sin(a)])
            if all(np.linalg.norm(p - q) > 0.09 for q in pts):
                pts.append(p)
        return dict(target=pts[0], cube=pts[1], yaw=self.rng.uniform(-0.6, 0.6))

    def reset(self, env_ids=None, layouts=None, settle=10):
        torch = self.torch
        ids = list(range(self.n)) if env_ids is None else list(env_ids)
        layouts = layouts or [self.sample_layout() for _ in ids]
        idt = torch.tensor(ids, device=self.dev)
        q = self.robot.data.default_joint_pos[idt].clone()
        q[:, self.jidx] = torch.tensor(self.home, device=self.dev, dtype=q.dtype)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=idt)
        self.robot.set_joint_position_target(q, env_ids=idt)
        o = self.origins[idt]
        cp = torch.zeros(len(ids), 7, device=self.dev); tp = torch.zeros(len(ids), 7, device=self.dev)
        for k, L in enumerate(layouts):
            cp[k, :3] = torch.tensor([L["cube"][0], L["cube"][1], CUBE + 0.001]); cp[k, 3] = math.cos(L["yaw"] / 2); cp[k, 6] = math.sin(L["yaw"] / 2)
            tp[k, :3] = torch.tensor([L["target"][0], L["target"][1], 0.0006]); tp[k, 3] = 1.0
            self.target_xy[ids[k]] = L["target"]
        cp[:, :3] += o; tp[:, :3] += o
        self.cube.write_root_pose_to_sim(cp, env_ids=idt); self.cube.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.dev), env_ids=idt)
        self.target.write_root_pose_to_sim(tp, env_ids=idt)
        self.scene.write_data_to_sim()
        for _ in range(settle):
            self._phys_steps(render=False)
        self.t[ids] = 0
        self.layouts = getattr(self, "layouts", [None] * self.n)
        for i, L in zip(ids, layouts):
            self.layouts[i] = L
        return layouts

    def _phys_steps(self, render=True):
        for i in range(self.dec):
            self.scene.write_data_to_sim()
            self.sim.step(render=render and i == self.dec - 1)
            self.scene.update(1 / PHYS_HZ)

    def step(self, cmd, active=None):
        """cmd: (n,6) joint targets in JOINTS order (rad). active: bool mask of envs whose clock advances."""
        torch = self.torch
        q = self.robot.data.joint_pos_target.clone()
        q[:, self.jidx] = torch.as_tensor(np.asarray(cmd), device=self.dev, dtype=q.dtype)
        lo = self.robot.data.soft_joint_pos_limits[..., 0]; hi = self.robot.data.soft_joint_pos_limits[..., 1]
        self.robot.set_joint_position_target(torch.clamp(q, lo, hi))
        self._phys_steps(render=True)
        self.t += 1 if active is None else np.asarray(active, int)

    # ------------------------------------------------------------------ observations
    def state(self):
        return self.robot.data.joint_pos[:, self.jidx].cpu().numpy().astype(np.float64)

    def images(self):
        f = self.scene["front"].data.output["rgb"][..., :3]; w = self.scene["wrist"].data.output["rgb"][..., :3]
        return f.cpu().numpy().astype(np.uint8), w.cpu().numpy().astype(np.uint8)

    def cube_pose(self):
        p = (self.cube.data.root_pos_w - self.origins).cpu().numpy(); q = self.cube.data.root_quat_w.cpu().numpy()
        return p, q

    def success(self):
        c, _ = self.cube_pose()
        in_sq = np.all(np.abs(c[:, :2] - self.target_xy) < 0.04, 1) & (c[:, 2] < CUBE + 0.01)
        released = self.state()[:, 5] > 0.5
        still = np.linalg.norm(self.cube.data.root_lin_vel_w.cpu().numpy(), axis=1) < 0.05
        return in_sq & released & still

    def close(self):
        self.sim.app.close() if hasattr(self.sim, "app") else None
