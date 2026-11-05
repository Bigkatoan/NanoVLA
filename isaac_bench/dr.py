"""Domain randomisation + random robot base pose for the SO-101 tabletop (isaac_bench/tabletop.py).

    dr = DomainRandomizer(env, rng)            # after tabletop.Tabletop(...) is built (sim already reset)
    dr.randomize_global()                      # shared dome / sun / table-material / ground: only between batches
    pose = dr.sample_base_pose(rng)            # (x, y, yaw) env-local, base footprint + reach sector on the table
    E = dr.transform_episode(env.sample_episode(), pose)   # layout sampled for the nominal base -> moved base
    dr.apply_base_pose([i], [pose])            # teleport the fixed-base articulation root (before env.reset of i)
    env.reset([i], [E]);  meta = dr.randomize_episode([i], [E])     # -> E["dr"] (every sampled value)
    ...control loop: env.step(cmd); dr.step(env.t)                  # continuous light drift / moving light / flicker

Everything visual is a USD attribute write on prims that exist from __init__ (no material rebinds after init);
physics goes through the PhysX tensor views. All sampled values are returned (JSON-serialisable) per env.
Import only after the app has been launched (needs pxr / torch / isaaclab).
"""
import math, copy, time
import numpy as np

import tabletop as tt

# table top in env-local coords (tabletop.scene_cfg: 0.9 x 1.1 cuboid centred at (0.2, 0), top at z=0)
TABLE_C, TABLE_HALF = np.array([0.2, 0.0]), np.array([0.45, 0.55])
NOMINAL_BASE = (tt.BASE_X, 0.0, 0.0)
WRIST_POS0, WRIST_F0 = (-0.005, 0.06, -0.062), 13.5

DEFAULT_CFG = dict(
    # ---- base pose (env-local; yaw 0 = robot facing +x, as the nominal SO101_CFG mount at x=-0.05)
    base_x=(-0.17, 0.22), base_y=(-0.30, 0.30), base_yaw_deg=35.0,
    base_foot_r=0.07,          # footprint radius (~0.12 m square base)
    reach_r=0.34, reach_a=1.05, reach_margin=0.015,   # objects are sampled at r<=0.27 (+ their radius), |angle|<=1 rad
    # ---- per-env lights
    # intensities are NORMALIZED (inputs:normalize=1: power independent of the radius) and log-uniform.
    # calibrated on the wrist camera: lamp 8.5e3 ~ +50 grey levels, 2.8e4 ~ +90 (sun 700 ~ +35, dome 220 ~ +60)
    lamp_I=(2500.0, 20000.0), lamp_K=(2700.0, 7500.0), lamp_radius=(0.05, 0.25),
    lamp_xy=((-0.25, 0.6), (-0.55, 0.55)), lamp_z=(0.55, 1.3), lamp_tint=0.06,
    fill_p=0.85, fill_I=(1000.0, 15000.0), fill_K=(2500.0, 9000.0), fill_radius=(0.03, 0.15),
    fill_orbit_r=(0.5, 1.0), fill_z=(0.35, 1.0), fill_cone_deg=(25.0, 90.0), fill_softness=(0.0, 0.6), fill_tint=0.25,
    # ---- continuous (per control step)
    drift_amp=(0.0, 0.3), drift_hz=(0.03, 0.25), driftK_amp=(0.0, 400.0),
    orbit_w=(-0.5, 0.5),       # rad/s of the fill light around the table centre
    flicker_rate=(0.0, 0.25),  # flicker events per second (0 -> none in this episode)
    flicker_len=(2, 8), flicker_gain=(0.15, 0.7),
    # ---- wrist camera
    cam_f_rel=0.10, cam_dp=0.003, cam_rot_deg=2.0,   # mount z jitter only towards the lens side (+z clips into the housing)
    # ---- appearance
    table_rot180_p=0.5, table_mirror_p=0.5, table_tex_scale=(0.6, 1.7), leg_rgb=((0.05, 0.75),), pad_jitter=0.08,
    clutter_n=(0, 4), clutter_size=(0.03, 0.12), clutter_min_r=0.40,
    # ---- physics
    obj_mass=(0.7, 1.3), obj_mu=(0.5, 1.2), obj_rest=(0.0, 0.2),
    robot_stiff=(0.85, 1.15), robot_damp=(0.85, 1.15), robot_arm_scale=(0.85, 1.15), robot_arm_add=(0.0, 0.003),
    robot_fric_add=(0.0, 0.05), robot_mu=(0.8, 1.3),
    # ---- global (randomize_global)
    dome_I=(60.0, 320.0), dome_K=(4500.0, 8000.0), sun_I=(0.0, 900.0), sun_K=(3500.0, 7000.0), sun_elev_deg=(25.0, 80.0),
    tex_tint=0.12, ground_rgb=(0.1, 0.6), whitepoint=0.06,
)


# ------------------------------------------------------------------ frames
def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def from_base_frame(xy, pose):
    """base-frame xy (x forward from the base, y to its left; (..., 2)) -> env-local xy"""
    xy = np.asarray(xy, float)
    return xy @ _rot(pose[2]).T + np.array(pose[:2])


def to_base_frame(xy, pose):
    """env-local xy (..., 2) -> base-frame xy"""
    xy = np.asarray(xy, float)
    return (xy - np.array(pose[:2])) @ _rot(pose[2])


def nominal_to_pose(xy, pose):
    """env-local xy sampled for the nominal base (BASE_X, 0, yaw 0) -> env-local xy for the moved base"""
    return from_base_frame(np.asarray(xy, float) - np.array([tt.BASE_X, 0.0]), pose)


def on_table(xy, margin=0.0):
    d = np.abs(np.asarray(xy, float) - TABLE_C)
    return bool(np.all(d <= TABLE_HALF - margin))


def blackbody_rgb(K):
    """approximate sRGB of a black body (Tanner Helland fit), normalised to max 1"""
    t = K / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    return c / c.max()


def _quat_mul(a, b):   # (w, x, y, z)
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_rotvec(v):
    v = np.asarray(v, float); a = np.linalg.norm(v)
    if a < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.r_[math.cos(a / 2), math.sin(a / 2) * v / a]


def _quat_look(d):
    """quat (w,x,y,z) whose -Z axis points along d (lights / cameras emit along -Z)"""
    z = -np.asarray(d, float); z /= np.linalg.norm(z)
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.c_[x, y, z]
    from tabletop_expert import quat_from_axes
    return quat_from_axes(R[:, 0], R[:, 2])


def _logu(r, rg):
    return float(math.exp(r.uniform(math.log(rg[0]), math.log(rg[1]))))


def _r(v, k=3):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, k) for x in v]
    return round(float(v), k)


class DomainRandomizer:
    def __init__(self, env, rng=None, cfg=None):
        import torch
        from pxr import UsdGeom, UsdLux, UsdShade, Sdf, Gf
        self.env, self.torch = env, torch
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.stage, self.n, self.dev = env.stage, env.n, env.dev
        st = self.stage
        self.base = np.tile(np.array(NOMINAL_BASE, float), (self.n, 1))
        R = env.robot
        self._root0 = R.data.default_root_state[:, :7].clone()          # (n,7) env-local default (pos (-0.05,0,0), yaw 90 deg)
        self._q0 = self._root0[0, 3:7].cpu().numpy().astype(float)
        self._stiff0 = R.data.joint_stiffness.clone(); self._damp0 = R.data.joint_damping.clone()
        self._arm0 = R.data.joint_armature.clone()
        self._fric0 = R.data.joint_friction_coeff.clone() if hasattr(R.data, "joint_friction_coeff") else None
        self._rmat0 = R.root_physx_view.get_material_properties().clone()
        # ---- per-env prims
        self.lamp = [UsdLux.SphereLight(st.GetPrimAtPath(f"/World/envs/env_{i}/lamp")) for i in range(self.n)]
        self.fill, self.cam, self.cam0, self.clutter = [], [], [], []
        self.legs = [[self._shader_of(f"/World/envs/env_{i}/leg_{c}") for c in "abcd"] for i in range(self.n)]
        self.pads = [{p: self._shader_of(f"/World/envs/env_{i}/{p}") for p in [f"{c}_pad" for c in tt.PADS]} for i in range(self.n)]
        cl_mats = []
        for i in range(self.n):
            f = UsdLux.SphereLight.Define(st, f"/World/envs/env_{i}/dr_fill")
            f.CreateIntensityAttr(0.0); f.CreateRadiusAttr(0.08); f.CreateColorAttr(Gf.Vec3f(1, 1, 1)); f.CreateTreatAsPointAttr(False)
            for lt in (self.lamp[i], f):
                lt.CreateEnableColorTemperatureAttr(True); lt.CreateColorTemperatureAttr(6500.0); lt.CreateNormalizeAttr(True)
                lt.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
            self.lamp[i].GetIntensityAttr().Set(8000.0)
            sh = UsdLux.ShapingAPI.Apply(f.GetPrim()); sh.CreateShapingConeAngleAttr(180.0); sh.CreateShapingConeSoftnessAttr(0.0)
            xf = UsdGeom.Xformable(f); xf.ClearXformOpOrder()
            self.fill.append(dict(light=f, shaping=sh, t=xf.AddTranslateOp(), o=xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)))
            # lamp translate op
            lx = UsdGeom.Xformable(self.lamp[i]); ops = {o.GetOpName(): o for o in lx.GetOrderedXformOps()}
            self.lamp[i]._t = ops.get("xformOp:translate") or lx.AddTranslateOp()
            # wrist camera (child of the gripper link)
            cp = st.GetPrimAtPath(f"/World/envs/env_{i}/Robot/gripper/gripper_cam")
            cam = UsdGeom.Camera(cp); ops = {o.GetOpName(): o for o in UsdGeom.Xformable(cp).GetOrderedXformOps()}
            to, oo = ops["xformOp:translate"], ops["xformOp:orient"]
            q = oo.Get(); self.cam.append(dict(cam=cam, t=to, o=oo, otype=type(q)))
            self.cam0.append(dict(t=np.array(to.Get(), float), q=np.r_[q.GetReal(), list(q.GetImaginary())], f=float(cam.GetFocalLengthAttr().Get())))
            # visual-only clutter (no physics): parked under the table when unused
            row = []
            for k in range(self.cfg["clutter_n"][1]):
                kind = ("Cube", "Cylinder", "Sphere", "Cube")[k % 4]
                path = f"/World/envs/env_{i}/dr_clutter_{k}"
                g = getattr(UsdGeom, kind).Define(st, path)
                if kind == "Cube":
                    g.CreateSizeAttr(1.0)
                elif kind == "Cylinder":
                    g.CreateRadiusAttr(0.5); g.CreateHeightAttr(1.0)
                else:
                    g.CreateRadiusAttr(0.5)
                x = UsdGeom.Xformable(g); x.ClearXformOpOrder()
                ops = dict(t=x.AddTranslateOp(), r=x.AddRotateZOp(), s=x.AddScaleOp())
                ops["t"].Set(Gf.Vec3d(-1.5, 0, -0.7)); ops["s"].Set(Gf.Vec3f(0.05, 0.05, 0.05))
                mp = f"/World/Looks/dr_clutter_{i}_{k}"
                m = UsdShade.Material.Define(st, mp); s = UsdShade.Shader.Define(st, mp + "/Shader")
                s.CreateIdAttr("UsdPreviewSurface")
                col = s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f); col.Set(Gf.Vec3f(0.5, 0.5, 0.5))
                rough = s.CreateInput("roughness", Sdf.ValueTypeNames.Float); rough.Set(0.5)
                met = s.CreateInput("metallic", Sdf.ValueTypeNames.Float); met.Set(0.0)
                m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m, UsdShade.Tokens.strongerThanDescendants)
                row.append(dict(img=UsdGeom.Imageable(g), kind=kind, col=col, rough=rough, met=met, **ops))
            self.clutter.append(row)
        # ---- table skins: add a rotateZ op once (per-env 180 deg flip of the visible texture)
        self.skin_ops = []
        for i in range(self.n):
            row = []
            for im in env.skins[i]:
                x = UsdGeom.Xformable(im.GetPrim()); ops = {o.GetOpName(): o for o in x.GetOrderedXformOps()}
                if "xformOp:rotateZ" not in ops:
                    t, s = ops["xformOp:translate"], ops["xformOp:scale"]; tv, sv = t.Get(), s.Get()
                    x.ClearXformOpOrder(); t = x.AddTranslateOp(); t.Set(tv); rz = x.AddRotateZOp(); rz.Set(0.0); s = x.AddScaleOp(); s.Set(sv)
                    ops = {"xformOp:translate": t, "xformOp:rotateZ": rz, "xformOp:scale": s}
                row.append(dict(rz=ops["xformOp:rotateZ"], s=ops["xformOp:scale"], s0=np.array(ops["xformOp:scale"].Get(), float),
                                size=UsdGeom.Cube(im.GetPrim()).GetSizeAttr()))
            self.skin_ops.append(row)
        # continuous-variation state
        self.dyn = [None] * self.n
        self.focal = WRIST_F0
        self.timing = dict(reset=[], step=[])
        for _ in range(3):            # compile the new clutter materials once
            env.sim.render()

    @staticmethod
    def _shader_of(path):
        """UsdPreviewSurface shader input 'diffuseColor' bound (directly) under a spawned prim, or None"""
        import omni.usd
        from pxr import UsdShade
        st = omni.usd.get_context().get_stage()
        p = st.GetPrimAtPath(path)
        if not p.IsValid():
            return None
        from pxr import Usd
        for q in Usd.PrimRange(p):
            if q.IsA(UsdShade.Shader):
                s = UsdShade.Shader(q); inp = s.GetInput("diffuseColor") or s.GetInput("diffuse_color_constant")
                if inp:
                    return dict(inp=inp, v0=np.array(inp.Get() if inp.Get() is not None else (0.5, 0.5, 0.5), float))
        return None

    # ================================================================ base pose
    def sample_base_pose(self, rng=None):
        """random (x, y, yaw) env-local: base footprint and the whole reach sector (where objects are sampled) on the table"""
        r, c = rng if rng is not None else self.rng, self.cfg
        rr = np.linspace(0.0, c["reach_r"], 6); aa = np.linspace(-c["reach_a"], c["reach_a"], 9)
        sector = np.array([[a * math.cos(b), a * math.sin(b)] for a in rr for b in aa])
        foot = np.array([[c["base_foot_r"] * math.cos(t), c["base_foot_r"] * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
        pts = np.r_[sector, foot]
        for _ in range(1000):
            pose = (float(r.uniform(*c["base_x"])), float(r.uniform(*c["base_y"])), float(math.radians(r.uniform(-c["base_yaw_deg"], c["base_yaw_deg"]))))
            if all(on_table(p, c["reach_margin"]) for p in from_base_frame(pts, pose)):
                return pose
        return NOMINAL_BASE

    def apply_base_pose(self, env_ids, poses):
        """teleport the fixed-base root of envs `env_ids` to env-local (x, y, yaw). Call BEFORE env.reset of those envs
        (reset re-writes the joint state); body poses / Jacobians are valid after the next sim step."""
        torch = self.torch
        ids = list(env_ids)
        P = torch.zeros(len(ids), 7, device=self.dev, dtype=torch.float32)
        for k, (i, (x, y, yaw)) in enumerate(zip(ids, poses)):
            q = _quat_mul(np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), self._q0)
            P[k, :3] = torch.tensor([x, y, float(self._root0[i, 2])], dtype=torch.float32, device=self.dev) + self.env.origins[i]
            P[k, 3:] = torch.tensor(q, dtype=torch.float32, device=self.dev)
            self.base[i] = (x, y, yaw)
        idt = torch.tensor(ids, device=self.dev, dtype=torch.long)
        self.env.robot.write_root_pose_to_sim(P, env_ids=idt)
        self.env.robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.dev), env_ids=idt)

    def transform_episode(self, E, pose):
        """map an episode sampled by Tabletop._sample_once (nominal base at (BASE_X, 0), yaw 0) rigidly onto `pose`:
        layout positions + yaws, push direction, rel target. Returns a copy with E["base_pose"]; None if anything leaves the table."""
        E = copy.deepcopy(E); yaw = pose[2]
        lay = {}
        for k, (p, oy) in E["layout"].items():
            q = nominal_to_pose(p, pose)
            m = tt.OBJ_META.get(k)
            rad = 0.5 * max(m["size"][:2]) if m else (tt.PAD_E / 2 if k.endswith("pad") else 0.08)
            if not on_table(q, rad + 0.005):
                return None
            lay[k] = (q, float(oy + yaw))
        E["layout"] = lay
        if "dir" in E:
            E["dir"] = tuple(float(v) for v in _rot(yaw) @ np.asarray(E["dir"], float))
        if "rel_xy" in E:
            E["rel_xy"] = tuple(float(v) for v in nominal_to_pose(E["rel_xy"], pose))
        E["base_pose"] = [float(v) for v in pose]
        return E

    # ================================================================ per-episode
    def randomize_episode(self, env_ids, episodes=None):
        """sample + apply everything per env (call right after env.reset of these envs). Returns {env_id: dict};
        the dict is also stored in episodes[k]["dr"] when episodes are given."""
        t0 = time.time()
        from pxr import Gf
        torch, r, c = self.torch, self.rng, self.cfg
        ids = list(env_ids); out = {}
        for k, i in enumerate(ids):
            d = {"base_pose": _r(self.base[i], 4)}
            # ---- lamp (existing per-env sphere light)
            K = r.uniform(*c["lamp_K"]); I = _logu(r, c["lamp_I"]); rad = r.uniform(*c["lamp_radius"])
            lp = (r.uniform(*c["lamp_xy"][0]), r.uniform(*c["lamp_xy"][1]), r.uniform(*c["lamp_z"]))
            tint = np.clip(1 + r.uniform(-c["lamp_tint"], c["lamp_tint"], 3), 0, 1.2)
            L = self.lamp[i]
            L.GetIntensityAttr().Set(float(I)); L.GetColorTemperatureAttr().Set(float(K)); L.GetRadiusAttr().Set(float(rad))
            L.GetColorAttr().Set(Gf.Vec3f(*map(float, tint))); L._t.Set(Gf.Vec3d(*map(float, lp)))
            d["lamp"] = dict(I=_r(I, 0), K=_r(K, 0), radius=_r(rad), pos=_r(lp), tint=_r(tint))
            # ---- fill light: a second (optionally spot-shaped) light orbiting the table centre
            on = r.random() < c["fill_p"]
            fI = _logu(r, c["fill_I"]) if on else 0.0; fK = r.uniform(*c["fill_K"]); frad = r.uniform(*c["fill_radius"])
            orb_r, orb_a, fz = r.uniform(*c["fill_orbit_r"]), r.uniform(-math.pi, math.pi), r.uniform(*c["fill_z"])
            cone = r.uniform(*c["fill_cone_deg"]) if r.random() < 0.6 else 180.0; soft = r.uniform(*c["fill_softness"])
            ftint = np.clip(1 + r.uniform(-c["fill_tint"], c["fill_tint"], 3), 0, 1.3)
            aim = np.r_[self.base[i, :2] + r.normal(0, 0.08, 2) + 0.2 * np.array([math.cos(self.base[i, 2]), math.sin(self.base[i, 2])]), 0.0]
            F = self.fill[i]
            F["light"].GetIntensityAttr().Set(float(fI)); F["light"].GetColorTemperatureAttr().Set(float(fK)); F["light"].GetRadiusAttr().Set(float(frad))
            F["light"].GetColorAttr().Set(Gf.Vec3f(*map(float, ftint)))
            F["shaping"].GetShapingConeAngleAttr().Set(float(cone)); F["shaping"].GetShapingConeSoftnessAttr().Set(float(soft))
            d["fill"] = dict(on=bool(on), I=_r(fI, 0), K=_r(fK, 0), radius=_r(frad), orbit_r=_r(orb_r), orbit_a0=_r(orb_a), z=_r(fz),
                             cone_deg=_r(cone, 1), softness=_r(soft), tint=_r(ftint), aim=_r(aim))
            # ---- continuous-variation parameters (used by step())
            dyn = dict(I=I, K=K, fI=fI, fK=fK, orb_r=orb_r, orb_a=orb_a, fz=fz, aim=aim,
                       amp=r.uniform(*c["drift_amp"]), hz=r.uniform(*c["drift_hz"]), ph=r.uniform(0, 2 * math.pi),
                       ampK=r.uniform(*c["driftK_amp"]), hzK=r.uniform(*c["drift_hz"]), phK=r.uniform(0, 2 * math.pi),
                       famp=r.uniform(*c["drift_amp"]), fhz=r.uniform(*c["drift_hz"]), fph=r.uniform(0, 2 * math.pi),
                       w=r.uniform(*c["orbit_w"]), flick_rate=r.uniform(*c["flicker_rate"]) if r.random() < 0.5 else 0.0,
                       flick_until=-1, flick_gain=1.0, flicks=0)
            self.dyn[i] = dyn
            d["dyn"] = {k_: _r(dyn[k_]) for k_ in ("amp", "hz", "ph", "ampK", "hzK", "phK", "famp", "fhz", "fph", "w", "flick_rate")}
            self._apply_fill_pose(i, orb_a)
            # ---- wrist camera mount pose jitter (focal length is GLOBAL: the tiled renderer uses one focal for all tiles)
            C, C0 = self.cam[i], self.cam0[i]
            dp = r.uniform(-c["cam_dp"], c["cam_dp"], 3); dp[2] = -abs(dp[2]); rv = np.radians(r.uniform(-c["cam_rot_deg"], c["cam_rot_deg"], 3))
            q = _quat_mul(_quat_rotvec(rv), C0["q"])
            C["t"].Set(Gf.Vec3d(*map(float, C0["t"] + dp)))
            C["o"].Set(C["otype"](float(q[0]), *map(float, q[1:])))
            d["wrist_cam"] = dict(focal=_r(self.focal, 3), dpos=_r(dp, 4), drot_deg=_r(np.degrees(rv), 2))
            # ---- table skin orientation (texture index itself is E["table_mat"], shown by Tabletop.reset)
            # the OmniPBR projection is in OBJECT space: Cube.size = k with xform scale / k keeps the footprint and
            # repeats the texture k times more (per-env texture scale); rotZ 180 / mirror flip the pattern
            rot = 180.0 if r.random() < c["table_rot180_p"] else 0.0; mir = r.random() < c["table_mirror_p"]; ts = _logu(r, c["table_tex_scale"])
            tm = (episodes[k] or {}).get("table_mat") if episodes is not None else None
            for j, sk in enumerate(self.skin_ops[i]):
                if tm is not None and j != tm:
                    continue
                sk["rz"].Set(rot); s = sk["s0"] * np.array([-1.0 if mir else 1.0, 1, 1]) / ts; sk["s"].Set(Gf.Vec3f(*map(float, s))); sk["size"].Set(float(ts))
            d["table"] = dict(tex=None if tm is None else tt.TEX[tm].split("/")[-1], rot_deg=rot, mirror=bool(mir), tex_scale=_r(ts))
            # ---- leg / pad colours
            lc = r.uniform(*c["leg_rgb"][0]) * (1 + r.uniform(-0.15, 0.15, 3))
            for s in self.legs[i]:
                if s:
                    s["inp"].Set(Gf.Vec3f(*map(float, np.clip(lc, 0, 1))))
            pj = {}
            for name, s in self.pads[i].items():
                if s:
                    v = np.clip(s["v0"] * (1 + r.uniform(-c["pad_jitter"], c["pad_jitter"], 3)), 0, 1); s["inp"].Set(Gf.Vec3f(*map(float, v))); pj[name] = _r(v)
            d["legs_rgb"] = _r(np.clip(lc, 0, 1)); d["pads_rgb"] = pj
            # ---- clutter: visual-only shapes outside the reach sector
            E = episodes[k] if episodes is not None else None
            d["clutter"] = self._place_clutter(i, E)
            out[i] = d
        # ---- physics (PhysX views)
        out_phys = self._randomize_physics(ids, episodes)
        for i in ids:
            out[i].update(out_phys[i])
        if episodes is not None:
            for i, E in zip(ids, episodes):
                E["dr"] = out[i]
        self.timing["reset"].append(time.time() - t0)
        return out

    def _place_clutter(self, i, E):
        from pxr import Gf
        r, c = self.rng, self.cfg
        n = int(r.integers(c["clutter_n"][0], c["clutter_n"][1] + 1))
        bx, by, byaw = self.base[i]
        taken = [np.asarray(p, float) for p, _ in (E["layout"].values() if E else [])]
        res = []
        for k, cl in enumerate(self.clutter[i]):
            placed = False
            if k < n:
                for _ in range(200):
                    p = TABLE_C + r.uniform(-1, 1, 2) * (TABLE_HALF - 0.05)
                    pb = to_base_frame(p, self.base[i]); rr = np.linalg.norm(pb)
                    if rr < c["clutter_min_r"] and not (pb[0] < -0.12 and rr > 0.12):   # out of reach, or behind the base
                        continue
                    if any(np.linalg.norm(p - q) < 0.12 for q in taken):
                        continue
                    sz = r.uniform(*c["clutter_size"], 3); sz[2] = r.uniform(0.02, c["clutter_size"][1] * 1.3)
                    if cl["kind"] == "Sphere":
                        sz[:] = sz[0]
                    elif cl["kind"] == "Cylinder":
                        sz[1] = sz[0]
                    col = r.uniform(0.03, 0.97, 3); rough = r.uniform(0.2, 0.95); met = r.uniform(0, 0.6) if r.random() < 0.2 else 0.0
                    yaw = r.uniform(0, 360)
                    cl["t"].Set(Gf.Vec3d(float(p[0]), float(p[1]), float(sz[2] / 2))); cl["s"].Set(Gf.Vec3f(*map(float, sz))); cl["r"].Set(float(yaw))
                    cl["col"].Set(Gf.Vec3f(*map(float, col))); cl["rough"].Set(float(rough)); cl["met"].Set(float(met))
                    taken.append(p); placed = True
                    res.append(dict(kind=cl["kind"], xy=_r(p), size=_r(sz), yaw=_r(yaw, 1), rgb=_r(col), rough=_r(rough), metal=_r(met)))
                    break
            if not placed:
                cl["t"].Set(Gf.Vec3d(-1.5, 0.3 * k, -0.7))
        return res

    def _randomize_physics(self, ids, episodes):
        torch, r, c = self.torch, self.rng, self.cfg
        out = {i: dict(obj_phys={}) for i in ids}
        idx = torch.tensor(ids, dtype=torch.int32); il = idx.long(); n = len(ids)
        for name in tt.GRASP + tt.DISTRACT + tt.CONTAINERS:
            view = self.env.objs[name].root_physx_view
            m = view.get_masses().clone(); ms = torch.tensor(r.uniform(*c["obj_mass"], n), dtype=m.dtype)
            m[il] = tt.OBJ_META[name]["mass"] * ms[:, None]; view.set_masses(m, idx)
            mp = view.get_material_properties().clone()
            mu = torch.tensor(r.uniform(*c["obj_mu"], n), dtype=mp.dtype); rest = torch.tensor(r.uniform(*c["obj_rest"], n), dtype=mp.dtype)
            mp[il, :, 0] = mu[:, None]; mp[il, :, 1] = 0.9 * mu[:, None]; mp[il, :, 2] = rest[:, None]
            view.set_material_properties(mp, idx)
            for k, i in enumerate(ids):
                E = episodes[k] if episodes is not None else None
                if E is None or name in E["layout"]:
                    out[i]["obj_phys"][name] = dict(mass_scale=_r(ms[k]), mu=_r(mu[k]), restitution=_r(rest[k]))
        R = self.env.robot; nj = R.num_joints; it = torch.tensor(ids, device=self.dev, dtype=torch.long)
        U = lambda rg: torch.tensor(r.uniform(*rg, (n, nj)), device=self.dev, dtype=torch.float32)
        ks, kd, ka, aa, fa = U(c["robot_stiff"]), U(c["robot_damp"]), U(c["robot_arm_scale"]), U(c["robot_arm_add"]), U(c["robot_fric_add"])
        R.write_joint_stiffness_to_sim(self._stiff0[it] * ks, env_ids=it)
        R.write_joint_damping_to_sim(self._damp0[it] * kd, env_ids=it)
        R.write_joint_armature_to_sim(self._arm0[it] * ka + aa, env_ids=it)
        if self._fric0 is not None:
            R.write_joint_friction_coefficient_to_sim(self._fric0[it] + fa, env_ids=it)
        rm = self._rmat0.clone(); rmu = torch.tensor(r.uniform(*c["robot_mu"], n), dtype=rm.dtype)
        rm[il, :, 0] = self._rmat0[il, :, 0] * rmu[:, None]; rm[il, :, 1] = self._rmat0[il, :, 1] * rmu[:, None]
        R.root_physx_view.set_material_properties(rm, idx)
        for k, i in enumerate(ids):
            out[i]["robot"] = dict(joints=list(R.joint_names), stiff_scale=_r(ks[k].tolist()), damp_scale=_r(kd[k].tolist()),
                                   armature=_r((self._arm0[i] * ka[k] + aa[k]).tolist(), 5),
                                   friction=_r(((self._fric0[i] if self._fric0 is not None else 0) + fa[k]).tolist(), 4), mu_scale=_r(rmu[k]))
        return out

    # ================================================================ continuous
    def _apply_fill_pose(self, i, a):
        from pxr import Gf
        D = self.dyn[i]
        p = np.array([TABLE_C[0] + D["orb_r"] * math.cos(a), TABLE_C[1] + D["orb_r"] * math.sin(a), D["fz"]])
        q = _quat_look(D["aim"] - p)
        F = self.fill[i]; F["t"].Set(Gf.Vec3d(*map(float, p))); F["o"].Set(Gf.Quatd(float(q[0]), *map(float, q[1:])))

    def step(self, t=None, env_ids=None):
        """continuous variation: slow lamp intensity / colour-temperature drift, fill light orbit + drift, random flicker.
        t: control-step index per env (env.t) or a scalar. Only light attribute writes."""
        t0 = time.time()
        t = self.env.t if t is None else t
        t = np.broadcast_to(np.asarray(t, float), (self.n,))
        r = self.rng; c = self.cfg; dt = 1.0 / tt.CTRL_HZ
        for i in (range(self.n) if env_ids is None else env_ids):
            D = self.dyn[i]
            if D is None:
                continue
            s = t[i] * dt
            g = 1 + D["amp"] * math.sin(2 * math.pi * D["hz"] * s + D["ph"])
            if D["flick_rate"] > 0 and t[i] > D["flick_until"] and r.random() < D["flick_rate"] * dt:
                D["flick_until"] = t[i] + int(r.integers(*c["flicker_len"])); D["flick_gain"] = r.uniform(*c["flicker_gain"]); D["flicks"] += 1
            if t[i] <= D["flick_until"]:
                g *= D["flick_gain"] * (1 + 0.3 * r.standard_normal())
            L = self.lamp[i]
            L.GetIntensityAttr().Set(float(max(0.0, D["I"] * g)))
            L.GetColorTemperatureAttr().Set(float(np.clip(D["K"] + D["ampK"] * math.sin(2 * math.pi * D["hzK"] * s + D["phK"]), 1500, 12000)))
            if D["fI"] > 0:
                self.fill[i]["light"].GetIntensityAttr().Set(float(D["fI"] * (1 + D["famp"] * math.sin(2 * math.pi * D["fhz"] * s + D["fph"]))))
                if D["w"] != 0:
                    self._apply_fill_pose(i, D["orb_a"] + D["w"] * s)
        self.timing["step"].append(time.time() - t0)

    # ================================================================ global (shared by all envs)
    def randomize_global(self, rng=None):
        """dome HDRI / intensity / colour temperature / rotation, sun, ground colour, table-material scale / rotation / tint.
        Shared by every env: call between batches (a new HDRI loads an EXR)."""
        from pxr import UsdLux, UsdGeom, UsdShade, Sdf, Gf
        r, c, st = rng if rng is not None else self.rng, self.cfg, self.stage
        d = {}
        dome = UsdLux.DomeLight(st.GetPrimAtPath("/World/sky"))
        if tt.HDRI:
            hd = tt.HDRI[int(r.integers(len(tt.HDRI)))]; dome.GetTextureFileAttr().Set(Sdf.AssetPath(hd)); d["hdri"] = hd.split("/")[-1]
        dI, dK, drot = r.uniform(*c["dome_I"]), r.uniform(*c["dome_K"]), r.uniform(0, 360)
        dome.GetIntensityAttr().Set(float(dI)); dome.CreateEnableColorTemperatureAttr(True); dome.GetColorTemperatureAttr().Set(float(dK))
        self._set_rotz(dome.GetPrim(), drot)
        d.update(dome_I=_r(dI, 0), dome_K=_r(dK, 0), dome_rot_deg=_r(drot, 1))
        sun = UsdLux.DistantLight(st.GetPrimAtPath("/World/sun"))
        sI, sK, el, az = r.uniform(*c["sun_I"]), r.uniform(*c["sun_K"]), math.radians(r.uniform(*c["sun_elev_deg"])), r.uniform(-math.pi, math.pi)
        sun.GetIntensityAttr().Set(float(sI)); sun.CreateEnableColorTemperatureAttr(True); sun.GetColorTemperatureAttr().Set(float(sK))
        q = _quat_look(-np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]))
        xf = UsdGeom.Xformable(sun); ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
        oo = ops.get("xformOp:orient")
        if oo is not None:
            oo.Set(type(oo.Get())(float(q[0]), *map(float, q[1:])))
        d.update(sun_I=_r(sI, 0), sun_K=_r(sK, 0), sun_elev_deg=_r(math.degrees(el), 1), sun_az_deg=_r(math.degrees(az), 1))
        # ground plane colour
        gs = self._shader_of("/World/ground")
        if gs:
            g = r.uniform(*c["ground_rgb"]) * (1 + r.uniform(-0.2, 0.2, 3)); gs["inp"].Set(Gf.Vec3f(*map(float, np.clip(g, 0, 1)))); d["ground_rgb"] = _r(g)
        # table materials (shared OmniPBR per texture): scale / rotation / tint
        tm = []
        for p in self.env.table_mats:
            sh = st.GetPrimAtPath(p + "/Shader")
            rot = r.uniform(0, 360); tint = np.clip(1 + r.uniform(-c["tex_tint"], c["tex_tint"], 3), 0, 1)
            (sh.GetAttribute("inputs:texture_rotate") or sh.CreateAttribute("inputs:texture_rotate", Sdf.ValueTypeNames.Float)).Set(float(rot))
            (sh.GetAttribute("inputs:diffuse_tint") or sh.CreateAttribute("inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f)).Set(Gf.Vec3f(*map(float, tint)))
            tm.append(dict(tex=p.split("_")[-1], rot_deg=_r(rot, 1), tint=_r(tint)))
        d["table_mats"] = tm
        # wrist-camera focal length: one value for every env (TiledCamera renders all tiles with a single focal length)
        self.focal = WRIST_F0 * (1 + r.uniform(-c["cam_f_rel"], c["cam_f_rel"]))
        for C in self.cam:
            C["cam"].GetFocalLengthAttr().Set(float(self.focal))
        try:                       # keep TiledCamera.data.intrinsic_matrices consistent
            self.env.scene["wrist"]._update_intrinsic_matrices(list(range(self.n)))
        except Exception:
            pass
        d["wrist_focal"] = _r(self.focal)
        # renderer white point (global white balance; RTX tonemapper setting, applies to every camera)
        import carb
        wp = np.clip(1 + r.uniform(-c["whitepoint"], c["whitepoint"], 3), 0.5, 1.5)
        carb.settings.get_settings().set("/rtx/post/tonemap/whitepoint", [float(v) for v in wp]); d["whitepoint"] = _r(wp)
        self.global_dr = d
        return d

    @staticmethod
    def _set_rotz(prim, deg):
        from pxr import UsdGeom
        xf = UsdGeom.Xformable(prim); ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
        if "xformOp:rotateZ" in ops:
            ops["xformOp:rotateZ"].Set(float(deg)); return
        if "xformOp:orient" in ops:   # compose a z rotation into the existing orient
            from pxr import Gf
            o = ops["xformOp:orient"]; q0 = o.Get(); q = _quat_mul(np.array([math.cos(math.radians(deg) / 2), 0, 0, math.sin(math.radians(deg) / 2)]),
                                                                   np.r_[1.0, 0, 0, 0] if not hasattr(prim, "_q0") else prim._q0)
            o.Set(type(q0)(float(q[0]), *map(float, q[1:]))); return
        xf.AddRotateZOp().Set(float(deg))

    # ================================================================ misc
    def overhead(self):
        f = lambda v: (round(1e3 * float(np.mean(v)), 2), round(1e3 * float(np.max(v)), 2)) if v else None
        return dict(reset_ms_mean_max=f(self.timing["reset"]), step_ms_mean_max=f(self.timing["step"]))
