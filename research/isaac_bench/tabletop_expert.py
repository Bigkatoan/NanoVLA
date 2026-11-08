"""Scripted expert for the multi-task tabletop (batched): top-down grasp with real PhysX friction (no attachment assist).

Task-space waypoints -> damped-least-squares IK on the PhysX Jacobian of the gripper body, evaluated at the tool point
(URDF gripper_frame: fixed-jaw tip). The IK integrates on the *command* (q_cmd += J^+ e), which removes the steady
gravity sag of the soft position drives. Orientation target: approach axis straight down, jaw closing axis aligned
with a face of the object.
"""
import math, os
DEBUG = bool(int(os.environ.get('EXPERT_DEBUG', '0')))
import numpy as np
import torch
import tabletop as tt

FIXED_TIP = (-0.0095, -0.0004, -0.095)   # inner face of the fixed-jaw tip, gripper body frame (from the USD mesh)
MOVING_TIP = (-0.0060, -0.0750, 0.0189)  # inner face of the moving-jaw tip, jaw body frame
OPEN, CLOSE = 1.0, -0.15
GRASP_SIDE = float(os.environ.get("GRASP_SIDE", 0.0))   # tool point is ~centred in the jaw gap on this USD asset
H = {"cube": 0.03, "cylinder": 0.05}   # legacy block expert (tabletop_blocks.py)


def quat_from_axes(x, z):
    """rotation whose x/z columns are the given (orthonormalised) axes, quat (w,x,y,z)"""
    x = x / np.linalg.norm(x); z = z - x * (x @ z); z /= np.linalg.norm(z); y = np.cross(z, x)
    R = np.c_[x, y, z]; t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2; return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


class Expert:
    def __init__(self, env, rng):
        from isaaclab.utils import math as mu
        self.env, self.rng, self.mu = env, rng, mu
        self.bi = env.robot.body_names.index("gripper")
        self.n = env.n
        self.tcp = torch.tensor(FIXED_TIP, device=env.dev).repeat(self.n, 1)
        self.ji = env.robot.body_names.index("jaw")
        self.mtip = torch.tensor(MOVING_TIP, device=env.dev).repeat(self.n, 1)
        self.plan = [None] * self.n
        self.q_cmd = None

    # ------------------------------------------------------------ kinematics
    def tcp_pose(self):
        """grasp centre = midpoint of the fixed-jaw tip (gripper link) and the moving-jaw tip (jaw link), from the USD meshes"""
        d = self.env.robot.data; mu = self.mu
        pb = d.body_pos_w[:, self.bi] - self.env.origins; qb = d.body_quat_w[:, self.bi]
        pj = d.body_pos_w[:, self.ji] - self.env.origins; qj = d.body_quat_w[:, self.ji]
        fixed = pb + mu.quat_apply(qb, self.tcp); moving = pj + mu.quat_apply(qj, self.mtip)
        mid = (fixed + moving) / 2
        c = torch.cat([mid[:, :2], fixed[:, 2:3]], 1)   # xy: centre of the jaw gap, z: fixed-jaw tip (where the tips meet when closed)
        self._mid_r = c - pb
        return c, qb

    def jac(self):
        J = self.env.robot.root_physx_view.get_jacobians()[:, self.bi - 1, :, :5].clone()   # fixed base -> body idx-1
        r = self._mid_r if hasattr(self, "_mid_r") else self.mu.quat_apply(self.env.robot.data.body_quat_w[:, self.bi], self.tcp)
        J[:, :3] += torch.bmm(-self.mu.skew_symmetric_matrix(r), J[:, 3:])
        return J

    def ik_step(self, pos_des, quat_des, gain=0.7, lam=0.03, wrot=0.5):
        p, q = self.tcp_pose()
        pe, re = self.mu.compute_pose_error(p, q, pos_des, quat_des, rot_error_type="axis_angle")
        e = torch.cat([pe, wrot * re], 1)
        J = self.jac(); J[:, 3:] *= wrot
        JJt = torch.bmm(J, J.transpose(1, 2)) + lam ** 2 * torch.eye(6, device=J.device)
        dq = torch.bmm(J.transpose(1, 2), torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        return torch.clamp(gain * dq, -0.06, 0.06), pe.norm(dim=1)

    # ------------------------------------------------------------ planning
    def grasp_yaw(self, name, i, azim):
        """closing-axis yaw: a face normal of the object closest to the 'natural' yaw (perpendicular to the arm plane)"""
        nat = azim - math.pi / 2          # at HOME the closing axis (gripper x) points to -y for a forward-facing arm
        if name.endswith("cylinder"):
            return nat
        q = self.env.quat(name)[i]; oy = 2 * math.atan2(q[3], q[0])
        c = [oy + k * math.pi / 2 for k in range(4)]
        return min(c, key=lambda y: abs((y - nat + math.pi) % (2 * math.pi) - math.pi))

    def start(self, i):
        env, E, r = self.env, self.env.ep[i], self.rng
        obj = E["obj"]; kind = obj.split("_")[1]; h = H[kind]
        c = env.pos(obj)[i]
        azim = math.atan2(c[1], c[0] + 0.05)
        yaw = self.grasp_yaw(obj, i, azim)
        xg = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        half = tt.CUBE_E / 2 if kind == "cube" else tt.CYL_R
        off = GRASP_SIDE * (half + 0.004) * xg
        gz = 0.014 if kind == "cube" else 0.024
        sp = r.uniform(0.85, 1.2); T = lambda k: max(4, int(k * sp))
        j = lambda s=0.003: r.normal(0, s, 3) * np.array([1, 1, 0.3])
        quat = quat_from_axes(xg, np.array([0, 0, 1.0]))
        pre = np.r_[c[:2] + off[:2], 0.10] + j(); at = np.r_[c[:2] + off[:2], gz] + j(0.0015); lift = at + [0, 0, 0.10]
        segs = [("move", pre, OPEN, T(40)), ("move", at, OPEN, T(26)), ("grip", None, CLOSE, T(14)), ("hold", None, CLOSE, 8),
                ("move", lift, CLOSE, T(30))]
        if E["task"] == "pick":
            segs += [("hold", None, CLOSE, 45)]
        else:
            segs += [("move", "HI", CLOSE, T(40)), ("move", "LO", CLOSE, T(22)), ("grip", None, OPEN, T(12)), ("move", "UP", OPEN, T(20)), ("hold", None, OPEN, 10)]
        self.plan[i] = dict(segs=segs, k=0, s=0, p0=None, g0=None, quat=quat, h=h, place=None, lift_check=c[2] + 0.05)

    def place_targets(self, i):
        env, E, P = self.env, self.env.ep[i], self.plan[i]
        p_tcp, _ = self.tcp_pose(); obj = env.pos(E["obj"])[i]
        delta = obj - p_tcp[i].cpu().numpy()                         # object centre relative to the tool point (world)
        if E["task"] == "pad":
            goal = np.r_[env.pos(E["pad"])[i, :2], P["h"] / 2 + 0.006]
        elif E["task"] == "tray":
            goal = np.r_[env.tray_center(i)[0], P["h"] / 2 + 0.012]
        else:
            b = env.pos(E["base"])[i]; goal = np.r_[b[:2], b[2] + tt.CUBE_E + 0.004]
        goal[:2] += self.rng.normal(0, 0.003, 2)
        lo = goal - delta; hi = lo.copy(); hi[2] = max(0.13, lo[2] + 0.06)
        P["place"] = dict(HI=hi, LO=lo, UP=lo + [0, 0, 0.07])

    def act(self, active):
        """one control step for all envs; returns (n,6) joint command and per-env status ('run'|'done'|'fail')"""
        env = self.env
        if self.q_cmd is None:
            self.q_cmd = env.robot.data.joint_pos.clone()
        p_now, _ = self.tcp_pose(); p_now = p_now.cpu().numpy()
        pos_des = p_now.copy(); quat_des = np.tile([1.0, 0, 0, 0], (self.n, 1)); grip = self.q_cmd[:, 5].cpu().numpy().copy()
        use_ik = np.zeros(self.n, bool); status = ["idle"] * self.n
        for i in range(self.n):
            P = self.plan[i]
            if not active[i] or P is None:
                continue
            if P["k"] >= len(P["segs"]):
                status[i] = "done"; continue
            kind, tgt, g1, T = P["segs"][P["k"]]
            if P["s"] == 0:
                if isinstance(tgt, str) and tgt == "HI" and P["place"] is None:
                    if env.pos(env.ep[i]["obj"])[i, 2] < P["lift_check"]:
                        status[i] = "fail"; continue
                    self.place_targets(i)
                P["p0"] = P.get("p_last", p_now[i]).copy() if P.get("p_last") is not None else p_now[i].copy()
                P["g0"] = grip[i]
            P["s"] += 1; s = P["s"] / T; s = 3 * s ** 2 - 2 * s ** 3
            tgt = P["place"][tgt] if isinstance(tgt, str) else tgt
            if kind == "move":
                pos_des[i] = P["p0"] + (np.asarray(tgt) - P["p0"]) * s
            else:
                pos_des[i] = P["p0"]
            grip[i] = P["g0"] + (g1 - P["g0"]) * s
            quat_des[i] = P["quat"]; use_ik[i] = True
            status[i] = "run"
            if P["s"] >= T:
                if DEBUG:
                    print("SEG", i, P["k"], kind, "tcp", np.round(p_now[i], 3).tolist(), "des", np.round(pos_des[i], 3).tolist(),
                          "obj", np.round(env.pos(env.ep[i]["obj"])[i], 3).tolist(), "q", np.round(env.state()[i], 2).tolist(), flush=True)
                P["p_last"] = pos_des[i].copy() if kind == "move" else P["p0"]
                P["k"] += 1; P["s"] = 0
        dev = env.dev
        dq, err = self.ik_step(torch.tensor(pos_des, device=dev, dtype=torch.float32), torch.tensor(quat_des, device=dev, dtype=torch.float32))
        m = torch.tensor(use_ik, device=dev)[:, None]
        self.q_cmd[:, :5] = torch.where(m, self.q_cmd[:, :5] + dq, self.q_cmd[:, :5])
        self.q_cmd[:, 5] = torch.tensor(grip, device=dev, dtype=torch.float32)
        lim = env.robot.data.soft_joint_pos_limits
        self.q_cmd = torch.clamp(self.q_cmd, lim[..., 0], lim[..., 1])
        return self.q_cmd.cpu().numpy().astype(np.float64), status, err.cpu().numpy()

    def reset_env(self, i):
        self.plan[i] = None
        if self.q_cmd is not None:
            self.q_cmd[i] = torch.tensor(tt.HOME, device=self.env.dev, dtype=torch.float32)
