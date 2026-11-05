"""Multi-object, multi-task SO-101 tabletop workspace (Isaac Sim 5.1 / Isaac Lab 2.3, batched envs).

Per env: SO-101 (workshop USD, tuned actuators) mounted on a table whose top texture is re-drawn every episode,
a pool of household objects. Each episode samples a task,
a random subset of objects (the rest are parked off-table) and non-overlapping positions in the reachable area.
Objects: YCB household items (banana, marker, foam brick, gelatin box, pudding box, meat can, soup can) with
physics added (assets/ycb/*_phys.usda), a bowl, a mug (distractor), the workshop tray and coloured pads.
Tasks (language instruction generated from the sampled objects):
  pick   : "pick up the {obj}"                       -> object lifted > 5 cm and held at the end
  pad    : "put the {obj} on the {colour} pad"       -> object resting on that pad, gripper open
  tray   : "put the {obj} in the tray"               -> object inside the tray, gripper open
  bowl   : "put the {obj} in the bowl"               -> object inside the bowl, gripper open
Control 30 Hz (physics 120 Hz x 4). Actions: joint position targets (rad, USD joint order), cameras 160x120 RGB.
Import only after the app has been launched.
"""
import os, sys, math, glob
import numpy as np

sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))
WS_ASSETS = os.path.expanduser("~/VLA/third_party/so101_workshop/source/sim_to_real_so101/assets")
TEX_SCALE = float(os.environ.get("TEX_SCALE", 12.0))
TEX = sorted(glob.glob(os.path.expanduser("~/VLA/isaac_bench/textures/*.png")))
CTRL_HZ, PHYS_HZ = 30, 120
BASE_X = -0.05                   # workshop SO101_CFG mounts the base at x=-0.05
RES = (120, 160)
HOME = np.array([0.0, -0.6, -0.07, 1.51, -1.57, -0.1])   # workshop rest pose with pan centred, gripper pointing down

import json as _json
YCB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ycb")   # built by assets/fetch_ycb.py
OBJ_META = _json.load(open(os.path.join(YCB_DIR, "objects.json")))
for _m in OBJ_META.values():
    _m["usd"] = os.path.join(YCB_DIR, os.path.basename(_m["usd"]))
GRASP = [k for k, m in OBJ_META.items() if m["role"] == "grasp"]          # graspable household objects (YCB)
CONTAINERS = ["bowl"]                                                     # YCB containers (+ the workshop tray)
DISTRACT = [k for k, m in OBJ_META.items() if m["role"] == "distractor"]
PADS = {"pink": (0.95, 0.45, 0.70), "white": (0.95, 0.95, 0.95), "cyan": (0.20, 0.80, 0.85)}
PAD_E = 0.08
TRAY_SCALE = 0.6                 # workshop tray 0.20 x 0.16 x 0.03 -> 0.12 x 0.096 x 0.018
TRAY_HALF = (0.10 * TRAY_SCALE, 0.08 * TRAY_SCALE)
OBJS = GRASP
TASKS = ("pick", "pad", "tray", "bowl")
PARK = lambda i: (-1.0, -0.6 + 0.15 * i)   # off-table parking spots (local xy), on the floor
FITS = {"tray": lambda m: max(m["size"][:2]) < 0.125, "bowl": lambda m: max(m["size"][:2]) < 0.15}


def scene_cfg(num_envs, spacing=3.0):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils import configclass
    from sim_to_real_so101.assets.so101 import SO101_CFG

    def mat(rgb, rough=0.5, metal=0.0):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=rough, metallic=metal)
    phys = lambda mu=1.2: sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu * 0.9, restitution=0.0, friction_combine_mode="max")
    rprops = sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=0.5)
    coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0)

    def ycb(name):
        m = OBJ_META[name]
        return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=sim_utils.UsdFileCfg(
            usd_path=m["usd"], rigid_props=rprops, collision_props=coll, mass_props=sim_utils.MassPropertiesCfg(mass=m["mass"])),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.0, 0.1)))

    def pad(name, rgb):
        return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=sim_utils.CuboidCfg(
            size=(PAD_E, PAD_E, 0.002), visual_material=mat(rgb, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)), init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.4, 0.001)))

    @configclass
    class Cfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.3, 0.3, 0.32)),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.75)))
        sky = AssetBaseCfg(prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(intensity=220.0, texture_file=f"{WS_ASSETS}/hdri/studio_small_03_1k.exr",
                                                                                visible_in_primary_ray=True, enable_color_temperature=True, color_temperature=6500.0))
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=700.0, angle=3.0),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=(0.93, 0.18, -0.30, 0.10)))
        table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/table", spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.1, 0.04), visual_material=mat((0.7, 0.6, 0.5), 0.6), physics_material=phys(0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003)), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, -0.02)))
        leg_a = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/leg_a", spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.71), visual_material=mat((0.25, 0.25, 0.27))),
                             init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, -0.5, -0.395)))
        leg_b = leg_a.replace(prim_path="{ENV_REGEX_NS}/leg_b", init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0.5, -0.395)))
        leg_c = leg_a.replace(prim_path="{ENV_REGEX_NS}/leg_c", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, -0.5, -0.395)))
        leg_d = leg_a.replace(prim_path="{ENV_REGEX_NS}/leg_d", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.5, -0.395)))
        robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        tray = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/tray", spawn=sim_utils.UsdFileCfg(
            usd_path=f"{WS_ASSETS}/usd/tray.usda", scale=(TRAY_SCALE,) * 3, mass_props=sim_utils.MassPropertiesCfg(mass=0.15)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.6, 0.02)))
        front = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/front_cam", data_types=["rgb"], width=RES[1] * 2, height=RES[0] * 2,
                               offset=TiledCameraCfg.OffsetCfg(pos=(0.50, 0.0, 0.36), rot=(0.65328, 0.27060, 0.27060, 0.65328), convention="opengl"),
                               spawn=sim_utils.PinholeCameraCfg(focal_length=13.5, horizontal_aperture=20.955, clipping_range=(0.01, 5.0)))
        wrist = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam", data_types=["rgb"], width=RES[1] * 2, height=RES[0] * 2,
                               offset=TiledCameraCfg.OffsetCfg(pos=(-0.005, 0.06, -0.062), rot=(0.92388, -0.38268, 0.0, 0.0), convention="opengl"),
                               spawn=sim_utils.PinholeCameraCfg(focal_length=13.5, horizontal_aperture=20.955, clipping_range=(0.005, 5.0)))

    cfg = Cfg(num_envs=num_envs, env_spacing=spacing, replicate_physics=False)
    # pads first so that they are created before (under) the objects; instance attributes -> picked up by InteractiveScene
    for c, rgb in PADS.items():
        setattr(cfg, f"{c}_pad", pad(f"{c}_pad", rgb))
    for name in GRASP + CONTAINERS + DISTRACT:
        setattr(cfg, name, ycb(name))
    return cfg


def make_table_materials(stage):
    """one OmniPBR material per texture, world-projected UVs (the table cuboid has no UVs)"""
    import omni.kit.commands
    from pxr import Sdf
    paths = []
    for k, tex in enumerate(TEX):
        p = f"/World/Looks/table_{k}"
        omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=p)
        sh = stage.GetPrimAtPath(p + "/Shader")
        sh.CreateAttribute("inputs:diffuse_texture", Sdf.ValueTypeNames.Asset).Set(tex)
        sh.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
        sh.CreateAttribute("inputs:texture_scale", Sdf.ValueTypeNames.Float2).Set((TEX_SCALE, TEX_SCALE))
        sh.CreateAttribute("inputs:reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.55)
        paths.append(p)
    return paths


def bind_table_material(stage, env_id, mat_path):
    from pxr import UsdShade
    prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/table")
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    for p in [prim] + list(prim.GetAllChildren()) + [c for ch in prim.GetAllChildren() for c in ch.GetAllChildren()]:
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


# ====================================================================== runtime
def describe(name):
    c, kind = name.split("_"); return f"{c} {kind}"


class Tabletop:
    """Batched multi-task tabletop env. Positions are local to each env origin (x forward from the robot, z up from the table top)."""

    def __init__(self, num_envs=4, seed=0, device="cuda:0", task_mix=TASKS):
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        self.torch, self.n, self.dev = torch, num_envs, device
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / PHYS_HZ, render_interval=PHYS_HZ // CTRL_HZ, device=device,
                                                                       physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0)))
        self.scene = InteractiveScene(scene_cfg(num_envs))
        self.stage = self.sim.stage
        self.table_mats = make_table_materials(self.stage)
        self.sim.reset()
        self._make_skins(num_envs)
        self.robot = self.scene["robot"]
        self.objs = {k: self.scene[k] for k in GRASP + CONTAINERS + DISTRACT + ["tray"] + [f"{c}_pad" for c in PADS]}
        self.origins = self.scene.env_origins
        self.rng = np.random.default_rng(seed)
        self.dec = PHYS_HZ // CTRL_HZ
        self.task_mix = task_mix
        self.ep = [None] * num_envs
        self.t = np.zeros(num_envs, int)
        self.home = torch.tensor(HOME, device=device, dtype=torch.float32)

    # ---------------------------------------------------------------- episode sampling
    def sample_episode(self):
        r = self.rng
        task = str(r.choice(self.task_mix))
        pool = list(GRASP); r.shuffle(pool)
        present = pool[: int(r.integers(2, 5))]
        if task in FITS:   # the target object must fit in the container
            ok = [o for o in present if FITS[task](OBJ_META[o])]
            if not ok:
                return self.sample_episode()
            obj = str(r.choice(ok))
        else:
            obj = str(r.choice(present))
        pads = [f"{c}_pad" for c in r.permutation(list(PADS))[: int(r.integers(1, 3))]] if (task == "pad" or r.random() < 0.5) else []
        bowl = task == "bowl" or r.random() < 0.35
        tray = task == "tray" or (not bowl and r.random() < 0.35)
        mug = r.random() < 0.3
        info = dict(task=task, obj=obj)
        d = OBJ_META[obj]["desc"]
        if task == "pad":
            info["pad"] = str(r.choice(pads)); text = f"put the {d} on the {info['pad'].split('_')[0]} pad"
        elif task == "tray":
            text = f"put the {d} in the tray"
        elif task == "bowl":
            text = f"put the {d} in the bowl"
        else:
            text = f"pick up the {d}"
        layout, placed = {}, []
        def put(name, rad, rmin=0.16, rmax=0.27, amax=1.0, yaw=None):
            for _ in range(500):
                rr, a = r.uniform(rmin, rmax), r.uniform(-amax, amax)
                p = np.array([BASE_X + rr * math.cos(a), rr * math.sin(a)])     # radius measured from the robot base
                if all(np.linalg.norm(p - q) > rad + qr + 0.01 for q, qr in placed):
                    placed.append((p, rad)); layout[name] = (p, r.uniform(-math.pi, math.pi) if yaw is None else yaw); return True
            return False
        if bowl:
            put("bowl", 0.08, 0.20, 0.27, 0.9)
        if tray:
            put("tray", 0.075, 0.19, 0.25, 0.9)
        for p in pads:
            put(p, PAD_E * 0.65)
        put(obj, 0.45 * max(OBJ_META[obj]["size"][:2]), 0.17, 0.25, 0.8)   # target object first, well inside the reach
        for o in present:
            if o != obj:
                put(o, 0.45 * max(OBJ_META[o]["size"][:2]))
        if mug:
            put("mug", 0.055)
        present = [o for o in present if o in layout]
        info.update(text=text, layout=layout, present=present, pads=[p for p in pads if p in layout], tray="tray" in layout, bowl="bowl" in layout,
                    table_mat=int(r.integers(len(self.table_mats))))
        ok = obj in layout and (task != "pad" or info["pad"] in layout) and (task != "tray" or info["tray"]) and (task != "bowl" or info["bowl"])
        return info if ok else self.sample_episode()

    def reset(self, env_ids=None, episodes=None, settle=20):
        torch = self.torch
        ids = list(range(self.n)) if env_ids is None else list(env_ids)
        episodes = episodes or [self.sample_episode() for _ in ids]
        idt = torch.tensor(ids, device=self.dev)
        q = self.home[None].repeat(len(ids), 1)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=idt)
        self.robot.set_joint_position_target(q, env_ids=idt)
        for name, obj in self.objs.items():
            pose = torch.zeros(len(ids), 7, device=self.dev)
            for k, (i, E) in enumerate(zip(ids, episodes)):
                if name in E["layout"]:
                    (x, y), yaw = E["layout"][name]
                    z = 0.0005 if name == "tray" else (0.001 if name.endswith("pad") else -OBJ_META[name]["zmin"] + 0.002)
                else:
                    (x, y), yaw, z = PARK(list(self.objs).index(name)), 0.0, (-0.74 if not name.endswith("pad") else -0.749)
                if name == "tray":   # USD origin is the tray's corner: shift so that (x, y) is the tray centre
                    hx, hy = TRAY_HALF; c, s_ = math.cos(yaw), math.sin(yaw)
                    x, y = x - (c * hx - s_ * hy), y - (s_ * hx + c * hy)
                pose[k, :3] = torch.tensor([x, y, z]) + self.origins[i].cpu(); pose[k, 3] = math.cos(yaw / 2); pose[k, 6] = math.sin(yaw / 2)
            obj.write_root_pose_to_sim(pose, env_ids=idt)
            if not name.endswith("pad"):
                obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.dev), env_ids=idt)
        import time as _t; _t0 = _t.time()
        for i, E in zip(ids, episodes):
            self._show_skin(i, E["table_mat"])
            self.ep[i] = E; self.t[i] = 0
        for _ in range(settle):
            self._phys(render=False)
        self._last_reset_s = _t.time() - _t0
        return episodes

    def _make_skins(self, num_envs):
        """one thin visual-only table-top skin per texture per env, material bound once; episodes toggle visibility"""
        from pxr import UsdGeom, UsdShade, Gf
        self.skins = []
        for i in range(num_envs):
            row = []
            for k, mp in enumerate(self.table_mats):
                path = f"/World/envs/env_{i}/table_skin_{k}"
                cube = UsdGeom.Cube.Define(self.stage, path); cube.GetSizeAttr().Set(1.0)
                x = UsdGeom.Xformable(cube); x.ClearXformOpOrder()
                x.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.0, -0.0001)); x.AddScaleOp().Set(Gf.Vec3f(0.9, 1.1, 0.0008))
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(UsdShade.Material(self.stage.GetPrimAtPath(mp)), UsdShade.Tokens.strongerThanDescendants)
                row.append(UsdGeom.Imageable(cube))
            self.skins.append(row)
        for _ in range(4):            # compile every material while all skins are visible
            self.sim.render()
        for i in range(num_envs):
            self._show_skin(i, 0)

    def _show_skin(self, i, k):
        for j, im in enumerate(self.skins[i]):
            (im.MakeVisible if j == k else im.MakeInvisible)()

    def _phys(self, render=True):
        for i in range(self.dec):
            self.scene.write_data_to_sim()
            self.sim.step(render=render and i == self.dec - 1)
            self.scene.update(1 / PHYS_HZ)

    def step(self, cmd):
        torch = self.torch
        q = torch.as_tensor(np.asarray(cmd), device=self.dev, dtype=torch.float32)
        lim = self.robot.data.soft_joint_pos_limits
        self.robot.set_joint_position_target(torch.clamp(q, lim[..., 0], lim[..., 1]))
        self._phys(render=True)
        self.t += 1

    # ---------------------------------------------------------------- observations
    def state(self):
        return self.robot.data.joint_pos.cpu().numpy().astype(np.float64)

    def images(self):
        import torch.nn.functional as F
        out = []
        for cam in ("front", "wrist"):
            x = self.scene[cam].data.output["rgb"][..., :3].permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=RES, mode="area")
            out.append(x.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy())
        return out

    def pos(self, name):
        o = self.objs[name]
        return (o.data.root_pos_w - self.origins).cpu().numpy()

    def quat(self, name):
        return self.objs[name].data.root_quat_w.cpu().numpy()

    def tray_center(self, i):
        q = self.quat("tray")[i]; yaw = 2 * math.atan2(q[3], q[0]); o = self.pos("tray")[i]
        hx, hy = TRAY_HALF; c, s_ = math.cos(yaw), math.sin(yaw)
        return np.array([o[0] + c * hx - s_ * hy, o[1] + s_ * hx + c * hy]), yaw

    def grasp_point(self, name, i):
        """world-local grasp point of a graspable object (mesh-derived offset in the object frame) and its closing-axis yaw"""
        m = OBJ_META[name]; p = self.pos(name)[i]; q = self.quat(name)[i]; yaw = 2 * math.atan2(q[3], q[0])
        c, s_ = math.cos(yaw), math.sin(yaw); gx, gy = m.get("grasp_xy") or (0.0, 0.0)
        gz = max(0.012, m.get("grasp_h") or m["height"] / 2)
        base_z = p[2] + m["zmin"]                                   # bottom of the object
        g = np.array([p[0] + c * gx - s_ * gy, p[1] + s_ * gx + c * gy, base_z + gz])
        ax = 0.0 if m.get("close_axis") == "x" else math.pi / 2
        return g, yaw + ax

    def success(self):
        out = np.zeros(self.n, bool)
        g = self.state()[:, 5]
        for i, E in enumerate(self.ep):
            if E is None:
                continue
            m = OBJ_META[E["obj"]]; p = self.pos(E["obj"])[i]; bottom = p[2] + m["zmin"]
            opened = g[i] > 0.35
            vel = np.linalg.norm(self.objs[E["obj"]].data.root_lin_vel_w[i].cpu().numpy())
            if E["task"] == "pick":
                out[i] = bottom > 0.05 and not opened
            elif E["task"] == "pad":
                d = p[:2] - self.pos(E["pad"])[i, :2]
                out[i] = np.all(np.abs(d) < PAD_E / 2) and bottom < 0.01 and opened and vel < 0.05
            elif E["task"] == "tray":
                c, yaw = self.tray_center(i)
                d = p[:2] - c; cy, sy = math.cos(yaw), math.sin(yaw)
                loc = np.array([cy * d[0] + sy * d[1], -sy * d[0] + cy * d[1]])
                out[i] = np.all(np.abs(loc) < np.array(TRAY_HALF) - 0.005) and bottom < 0.02 and opened and vel < 0.05
            elif E["task"] == "bowl":
                d = np.linalg.norm(p[:2] - self.pos("bowl")[i, :2])
                out[i] = d < 0.06 and bottom < 0.04 and opened and vel < 0.05
        return out
