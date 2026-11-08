"""SO-101 MuJoCo pick-and-place environment (MuJoCo 3.3 from ~/VLA/third_party/sim_pkgs, EGL rendering).

Scene: SO-101 (official TheRobotStudio MJCF, STS3215-tuned position actuators) on a table, one or two coloured cubes,
a coloured target square. Cameras: 'front' (fixed, external) and 'wrist' (on the wrist camera mount).
Control at 30 Hz (physics 2 ms), actions = joint position targets (rad), state = joint positions (rad).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
if sys.version_info[:2] == (3, 10):  # MuJoCo build for the system python; other interpreters use their own pip mujoco
    sys.path.insert(0, os.path.expanduser("~/VLA/third_party/sim_pkgs"))
import numpy as np
import mujoco

XML = os.path.expanduser("~/VLA/third_party/sim_assets/SO-ARM100/Simulation/SO101/so101_new_calib_camera.xml")
COLORS = {"red": (0.85, 0.1, 0.1, 1), "blue": (0.1, 0.25, 0.9, 1), "green": (0.1, 0.7, 0.2, 1), "yellow": (0.95, 0.85, 0.1, 1)}
CUBE = 0.02          # half width (4 cm cube)
CUBE_H = 0.02        # half height
CTRL_DT = 1 / 30


def build(n_cubes=1, cam_front=((0.42, 0.0, 0.36), (0, 0, 0)), seed=0):
    spec = mujoco.MjSpec.from_file(XML)
    spec.option.timestep = 0.002
    w = spec.worldbody
    w.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], directional=True)
    w.add_light(pos=[0.6, -0.4, 0.8], dir=[-0.6, 0.4, -0.8], diffuse=[0.4, 0.4, 0.4])
    tex = spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[0.75, 0.7, 0.62], width=64, height=64)
    mat = spec.add_material(name="table_mat"); mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "table_tex"
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], material="table_mat", friction=[1.0, 0.01, 0.001])
    # target square (visual only, thin)
    tgt = w.add_body(name="target", pos=[0.2, 0.12, 0.0005], mocap=True)
    tgt.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.04, 0.0005], rgba=list(COLORS["blue"][:3]) + [1], contype=0, conaffinity=0)
    names = list(COLORS)
    for i in range(n_cubes):
        b = w.add_body(name=f"cube{i}", pos=[0.2, -0.05 + 0.1 * i, CUBE_H])
        b.add_freejoint(name=f"cube{i}_free")
        b.add_geom(name=f"cube{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE, CUBE, CUBE_H], rgba=COLORS[["red", "green", "yellow"][i]], mass=0.02,
                   friction=[1.5, 0.02, 0.002], condim=4)
    # cameras
    pos, _ = cam_front
    w.add_camera(name="front", pos=[0.55, 0.0, 0.42], xyaxes=[0.0, 1.0, 0.0, -0.7526, 0.0, 0.6585], fovy=58)
    wc = spec.body("wrist_camera")
    wc.add_camera(name="wrist", pos=[-0.0175, -0.0121, 0.0278], xyaxes=[0.0487, 0.9047, 0.4233, 0.865, -0.2501, 0.435], fovy=80)
    for g in spec.geoms:  # grippy jaws
        b = g.parent.name if hasattr(g.parent, "name") else ""
        if b in ("gripper", "moving_jaw_so101_v1") and g.contype != 0:
            g.friction = [2.0, 0.05, 0.005]; g.condim = 4
            g.solref = [0.01, 1.0]; g.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    m = spec.compile()
    return m


class SO101Env:
    JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, n_cubes=1, res=(120, 160), seed=0):
        self.m = build(n_cubes)
        self.d = mujoco.MjData(self.m)
        self.n_cubes = n_cubes
        self.rng = np.random.default_rng(seed)
        self.r = mujoco.Renderer(self.m, *res)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in self.JOINTS])
        self.nsub = int(round(CTRL_DT / self.m.opt.timestep))
        self.site = self.m.site("gripperframe").id
        q, _ = self.ik(np.array([0.18, 0.0, 0.16]), np.zeros(6), 0.0, iters=300)
        self.home = np.r_[q[:5], 0.0]  # ready pose above the workspace, tool pointing down, gripper closed

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- episode
    def reset(self, target_color="blue", cube_idx=0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = self.home
        self.d.ctrl[:] = self.home
        # sample cube + target positions in the reachable half-annulus in front of the robot (no overlap)
        def sample():
            r, a = self.rng.uniform(0.14, 0.26), self.rng.uniform(-0.9, 0.9)
            return np.array([r * np.cos(a), r * np.sin(a)])
        pts = []
        while len(pts) < self.n_cubes + 1:
            p = sample()
            if all(np.linalg.norm(p - q) > 0.09 for q in pts):
                pts.append(p)
        self.m.body("target").pos[:2] = pts[0]
        self.d.mocap_pos[0][:2] = pts[0]
        for i in range(self.n_cubes):
            ja = self.m.joint(f"cube{i}_free").qposadr[0]
            self.d.qpos[ja:ja + 3] = [pts[i + 1][0], pts[i + 1][1], CUBE_H + 0.001]
            yaw = self.rng.uniform(-0.6, 0.6)
            self.d.qpos[ja + 3:ja + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.cube_idx = cube_idx
        self.attached = None
        mujoco.mj_forward(self.m, self.d)
        for _ in range(50):
            mujoco.mj_step(self.m, self.d)
        self.t = 0
        return self.obs()

    # ---- assisted grasp: block is rigidly attached while the gripper is commanded closed and the block is between the jaws
    def _in_jaws(self):
        R = self.d.site_xmat[self.site].reshape(3, 3)
        rel = R.T @ (self.cube_pos() - self.ee_pos())   # site frame: x approach, z closing axis
        return (-0.07 < rel[0] < 0.03) and (abs(rel[1]) < 0.03) and (0.0 < rel[2] < 0.05)

    def _update_grasp(self, cmd_grip):
        ja = self.m.joint(f"cube{self.cube_idx}_free").qposadr[0]
        va = self.m.joint(f"cube{self.cube_idx}_free").dofadr[0]
        bid = self.m.body("gripper").id
        Rg, pg = self.d.xmat[bid].reshape(3, 3), self.d.xpos[bid]
        if self.attached is None and cmd_grip < 0.4 and self._in_jaws():
            qc = self.d.qpos[ja + 3:ja + 7].copy(); Rc = np.zeros(9); mujoco.mju_quat2Mat(Rc, qc)
            self.attached = (Rg.T @ (self.d.qpos[ja:ja + 3] - pg), Rg.T @ Rc.reshape(3, 3))
        elif self.attached is not None and cmd_grip > 0.8:
            self.attached = None
        if self.attached is not None:
            p_rel, R_rel = self.attached
            self.d.qpos[ja:ja + 3] = pg + Rg @ p_rel
            q = np.zeros(4); mujoco.mju_mat2Quat(q, (Rg @ R_rel).flatten()); self.d.qpos[ja + 3:ja + 7] = q
            self.d.qvel[va:va + 6] = 0

    def step(self, ctrl):
        lo, hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        self.d.ctrl[:] = np.clip(ctrl, lo, hi)
        for _ in range(self.nsub):
            mujoco.mj_step(self.m, self.d)
            self._update_grasp(self.d.ctrl[5])
        mujoco.mj_forward(self.m, self.d)
        self.t += 1
        return self.obs()

    def state(self):
        return self.d.qpos[self.qadr].copy()

    def cube_pos(self, i=None):
        i = self.cube_idx if i is None else i
        return self.d.body(f"cube{i}").xpos.copy()

    def target_pos(self):
        return self.d.mocap_pos[0].copy()

    def ee_pos(self):
        return self.d.site_xpos[self.site].copy()

    def render(self, cam):
        self.r.update_scene(self.d, camera=cam)
        return self.r.render().copy()

    def obs(self, images=False):
        return dict(state=self.state())

    def success(self):
        c, t = self.cube_pos(), self.target_pos()
        in_sq = np.all(np.abs(c[:2] - t[:2]) < 0.04) and c[2] < CUBE_H + 0.01
        gripper_open = self.state()[5] > 0.5
        return bool(in_sq and gripper_open and self.attached is None)

    # ---------------------------------------------------------------- IK
    def ik(self, target, q0, grip, iters=60, tol=2e-3, down=True):
        """damped least-squares IK on the gripper frame site position (+ keep tool pointing down)"""
        q = q0.copy()
        d2 = mujoco.MjData(self.m)
        d2.qpos[:] = self.d.qpos
        jac_p, jac_r = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        dofs = np.array([self.m.joint(j).dofadr[0] for j in self.JOINTS[:5]])
        for _ in range(iters):
            d2.qpos[self.qadr] = np.r_[q[:5], grip]
            mujoco.mj_kinematics(self.m, d2); mujoco.mj_comPos(self.m, d2)
            x = d2.site_xpos[self.site]
            err = target - x
            mujoco.mj_jacSite(self.m, d2, jac_p, jac_r, self.site)
            J = jac_p[:, dofs]
            e = err
            if down:  # approach axis (site x-axis) should point to -z world
                R = d2.site_xmat[self.site].reshape(3, 3)
                z = R[:, 0]
                e_r = np.cross(z, np.array([0, 0, -1.0]))
                J = np.vstack([J, 0.3 * jac_r[:, dofs]])
                e = np.r_[err, 0.3 * e_r]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            q[:5] += np.clip(dq, -0.2, 0.2)
            lo, hi = self.m.jnt_range[[self.m.joint(j).id for j in self.JOINTS[:5]]].T
            q[:5] = np.clip(q[:5], lo, hi)
            if np.linalg.norm(err) < tol:
                break
        return q, float(np.linalg.norm(err))
