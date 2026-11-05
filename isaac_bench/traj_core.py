"""Core of the node-based trajectory tool (importable without launching the app): TrajRunner + CosmosWriter.
Each trajectory step first drives the gripper to its state, then moves the grasp centre to the step's node."""
import json, math, os
import numpy as np, torch
import tabletop as tt
from tabletop_expert import Expert, quat_from_axes, OPEN, CLOSE

ROLL_SIGN = float(os.environ.get("ROLL_SIGN", 1.0))   # sign of d(gripper yaw)/d(wrist roll) for the yaw_rel "auto90" choice
GRIP_T = 18          # control steps (0.6 s) to actuate the gripper
SETTLE_T = 6         # steps to settle after a move


class CosmosWriter:
    """streams one env's episode to mp4s at 15 fps: {front,wrist} x {rgb,depth,seg}; kept only if the episode is kept"""
    STREAMS = [(c, m) for c in ("front", "wrist") for m in ("rgb", "depth", "seg")]

    def __init__(self, root, i):
        self.root, self.i, self.c = root, i, None

    def start(self):
        import av, tempfile
        self.dir = tempfile.mkdtemp(prefix=f"env{self.i}_", dir=self.root); self.c, self.s, self.idx = {}, {}, []

    def add(self, frames, fidx):
        import av
        for (cam, mod), k in zip(self.STREAMS, range(6)):
            im = frames[cam][("rgb", "depth", "seg").index(mod)]
            if (cam, mod) not in self.c:
                c = av.open(os.path.join(self.dir, f"{cam}_{mod}.mp4"), "w"); st = c.add_stream("libx264", rate=15)
                st.width, st.height, st.pix_fmt = im.shape[1], im.shape[0], "yuv420p"; st.options = {"crf": "18"}
                self.c[(cam, mod)], self.s[(cam, mod)] = c, st
            for p in self.s[(cam, mod)].encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(im), format="rgb24")):
                self.c[(cam, mod)].mux(p)
        self.idx.append(int(fidx))

    def finish(self, keep_as=None, meta=None):
        import shutil
        for k, c in self.c.items():
            for p in self.s[k].encode():
                c.mux(p)
            c.close()
        if keep_as is None:
            shutil.rmtree(self.dir, ignore_errors=True)
        else:
            dst = os.path.join(self.root, keep_as); shutil.rmtree(dst, ignore_errors=True); os.rename(self.dir, dst)
            json.dump(dict(meta or {}, fidx=self.idx, fps=15), open(os.path.join(dst, "episode.json"), "w"))


class TrajRunner:
    """Runs one spec on every env in parallel (each env resolves object-relative nodes against its own layout)."""

    def __init__(self, env, spec, rng, speed=0.18):
        self.env, self.spec, self.rng, self.speed = env, spec, rng, speed
        self.kin = Expert(env, rng)          # reuse kinematics + IK (grasp centre, Jacobian, DLS step)
        self.n = env.n

    def sub(self, i, name):
        """template placeholders: $obj -> the episode's object, $target -> its pad / tray / bowl"""
        E = self.env.ep[i]
        if name == "$obj":
            return E["obj"]
        if name == "$target":
            if E["task"] == "rel":
                return "$rel"
            return E.get("pad") or ("tray" if E["task"] == "tray" else "bowl")
        return name

    def resolve(self, i, nid, cur_yaw):
        nd = self.spec["nodes"][str(nid)]
        E = self.env.ep[i]
        if "rel_prev" in nd:                           # relative to the previous node target (e.g. rotate in place)
            p = self.p_hold[i] + np.asarray(nd["rel_prev"], float)
            yr = nd.get("yaw_rel", 0.0)
            if yr == "auto90":                          # rotate 90 deg in the direction that keeps the wrist roll inside its limits
                roll = self.env.state()[i, 4]
                yr = -90.0 * ROLL_SIGN if roll > 0 else 90.0 * ROLL_SIGN
            yaw = cur_yaw + math.radians(yr)
            return p, yaw
        if "dir_offset" in nd:                         # along / across the episode's push direction, relative to the object
            name = self.sub(i, nd.get("obj", "$obj")); u = np.asarray(E["dir"], float); v = np.array([-u[1], u[0]])
            a_, b_ = nd["dir_offset"]
            ext = 0.5 * max(tt.OBJ_META[name]["size"][:2]) + 0.012          # object radius + clearance along the push
            a_ = eval(a_, {"__builtins__": {}}, {"ext": ext, "L": E.get("push_len", 0.10)}) if isinstance(a_, str) else a_
            xy = self.env.pos(name)[i][:2] + a_ * u + b_ * v
            z = E["push_z"] if nd.get("z") == "push" else float(nd.get("z", 0.1))
            yaw = cur_yaw
            if nd.get("yaw") == "perp_dir":            # jaw closing axis across the push direction -> flat face pushes
                base = math.atan2(u[1], u[0]) + math.pi / 2
                yaw = min((base, base + math.pi), key=lambda y: abs((y - cur_yaw + math.pi) % (2 * math.pi) - math.pi))
            return np.r_[xy, z], yaw
        if nd.get("obj") == "$target" and E["task"] == "rel":
            p = np.r_[np.asarray(E["rel_xy"], float), 0.0] + np.asarray(nd.get("offset", [0, 0, 0]), float)
            if "carry" in nd:
                m = tt.OBJ_META[self.sub(i, nd["carry"])]; p[2] += max(0.012, m.get("grasp_h") or m["height"] / 2)
            return p, cur_yaw
        if "obj" in nd:
            name = self.sub(i, nd["obj"])
            off = np.asarray(nd.get("offset", [0, 0, 0]), float)
            if name in tt.GRASP:                       # graspable: offsets are relative to its grasp point
                g, gyaw = self.env.grasp_point(name, i)
                p = g + off
                if nd.get("yaw") == "auto":            # closing axis across the object; pick the sign nearest the current yaw
                    yaw = min((gyaw, gyaw + math.pi), key=lambda y: abs((y - cur_yaw + math.pi) % (2 * math.pi) - math.pi))
                else:
                    yaw = math.radians(nd["yaw"]) if isinstance(nd.get("yaw"), (int, float)) else cur_yaw
                return p, yaw
            base = self.env.tray_center(i)[0] if name == "tray" else self.env.pos(name)[i][:2]
            p = np.r_[base, 0.0] + off
            if "carry" in nd:                          # add the carried object's grasp height (its bottom ends at offset z)
                m = tt.OBJ_META[self.sub(i, nd["carry"])]; p[2] += max(0.012, m.get("grasp_h") or m["height"] / 2)
            yaw = math.radians(nd["yaw"]) if isinstance(nd.get("yaw"), (int, float)) else cur_yaw
            return p, yaw
        p = np.asarray(nd["pos"], float); yaw = math.radians(nd["yaw"]) if "yaw" in nd else cur_yaw
        return p, yaw

    def reset(self):
        self.k = np.zeros(self.n, int)              # step index
        self.phase = ["grip"] * self.n              # grip -> move -> settle -> next step
        self.s = np.zeros(self.n, int)
        self.yaw = np.zeros(self.n); self.T = np.ones(self.n, int)
        self.p_hold = np.zeros((self.n, 3)); self.p0 = np.zeros((self.n, 3)); self.p1 = np.zeros((self.n, 3))
        self.g = self.env.state()[:, 5].copy(); self.g0 = self.g.copy()
        self.done = np.zeros(self.n, bool); self.err = np.ones(self.n); self.retries = np.zeros(self.n, int)
        self.kin.q_cmd = self.env.robot.data.joint_pos.clone()
        for i in range(self.n):
            self.reset_env(i)

    def reset_env(self, i):
        """start the trajectory from step 0 in env i (after env.reset of that env)"""
        from isaaclab.utils import math as mu
        p, q = self.kin.tcp_pose(); p = p.cpu().numpy()
        xg = mu.matrix_from_quat(q)[i, :, 0].cpu().numpy()
        self.k[i], self.phase[i], self.s[i], self.T[i] = 0, "grip", 0, 1
        self.yaw[i] = math.atan2(xg[1], xg[0])
        self.p_hold[i] = self.p0[i] = self.p1[i] = p[i]
        self.g[i] = self.g0[i] = self.env.state()[i, 5]
        self.done[i], self.err[i], self.retries[i] = False, 1.0, 0
        self.kin.q_cmd[i] = self.env.robot.data.joint_pos[i]

    def _next_step(self, i, st):
        """advance after a step; optional {"retry": {"if_not_lifted": obj, "goto": k, "max": 2}} jumps back on a failed grasp"""
        rt = st.get("retry")
        if rt and self.retries[i] < rt.get("max", 2):
            name = self.sub(i, rt["if_not_lifted"])
            bottom = self.env.pos(name)[i][2] + tt.OBJ_META[name]["zmin"]
            if bottom < rt.get("min_lift", 0.03):
                self.retries[i] += 1; self.k[i] = rt["goto"]; self.phase[i], self.s[i] = "grip", 0
                return
        self.k[i] += 1; self.phase[i], self.s[i] = "grip", 0
        if self.k[i] >= len(self.spec["trajectory"]):
            self.done[i] = True

    def step(self):
        """advance the node machine one control step and return joint targets (own DLS IK on the command)"""
        pos_des, quat_des, grip = self.targets()
        dev = self.env.dev
        dq, err = self.kin.ik_step(torch.tensor(pos_des, device=dev, dtype=torch.float32), torch.tensor(quat_des, device=dev, dtype=torch.float32), gain=1.0)
        self.err = err.cpu().numpy()
        self.kin.q_cmd[:, :5] += dq
        self.kin.q_cmd[:, 5] = torch.tensor(grip, device=dev, dtype=torch.float32)
        lim = self.env.robot.data.soft_joint_pos_limits
        self.kin.q_cmd = torch.clamp(self.kin.q_cmd, lim[..., 0], lim[..., 1])
        return self.kin.q_cmd.cpu().numpy().astype(np.float64)

    def targets(self):
        """advance the node machine one control step and return task-space targets: grasp-centre position (env frame),
        gripper orientation quat (w,x,y,z) and jaw target, each (n, .). Callers that run their own controller
        (e.g. the Isaac Lab Mimic env with a differential-IK action) set self.err from their tracking error."""
        steps = self.spec["trajectory"]
        pos_des = self.p_hold.copy(); quat_des = np.zeros((self.n, 4)); grip = self.g.copy()
        for i in range(self.n):
            if self.done[i]:
                quat_des[i] = quat_from_axes(np.array([math.cos(self.yaw[i]), math.sin(self.yaw[i]), 0]), np.array([0, 0, 1.0])); continue
            st = steps[self.k[i]]
            gt = OPEN if st["gripper"] == "open" else CLOSE
            if self.phase[i] == "grip":
                if self.s[i] == 0:
                    self.g0[i] = self.g[i]
                self.s[i] += 1; u = min(1.0, self.s[i] / GRIP_T); u = 3 * u * u - 2 * u ** 3
                grip[i] = self.g0[i] + (gt - self.g0[i]) * u
                if self.s[i] >= GRIP_T + (10 if abs(gt - self.g0[i]) > 0.1 else 0):
                    self.phase[i], self.s[i] = "move", 0
                    p1, yaw = self.resolve(i, st["node"], self.yaw[i])
                    self.p0[i] = self.p_hold[i]; self.p1[i] = p1; self.yaw[i] = yaw
                    self.T[i] = max(8, int(np.linalg.norm(p1 - self.p0[i]) / self.speed * tt.CTRL_HZ))
            elif self.phase[i] == "move":
                self.s[i] += 1; u = min(1.0, self.s[i] / self.T[i]); u = 3 * u * u - 2 * u ** 3
                pos_des[i] = self.p0[i] + (self.p1[i] - self.p0[i]) * u
                if self.s[i] >= self.T[i]:
                    self.phase[i], self.s[i] = "settle", 0; self.p_hold[i] = self.p1[i]
                    if os.environ.get("TRAJ_DEBUG") and i == 0:
                        from isaaclab.utils import math as mu
                        d = self.env.robot.data; o = self.env.origins[0]
                        pb, qb = d.body_pos_w[0, self.kin.bi] - o, d.body_quat_w[0, self.kin.bi]
                        pj, qj = d.body_pos_w[0, self.kin.ji] - o, d.body_quat_w[0, self.kin.ji]
                        fx = pb + mu.quat_apply(qb[None], self.kin.tcp[:1])[0]; mv = pj + mu.quat_apply(qj[None], self.kin.mtip[:1])[0]
                        R = mu.matrix_from_quat(qb[None])[0].cpu().numpy()
                        cube = self.env.pos(self.env.ep[0]["obj"])[0]; mid = ((fx + mv) / 2).cpu().numpy()
                        print("DBG node", st["node"], "target", np.round(self.p1[i], 3).tolist(), "mid", np.round(mid, 3).tolist(), "fixed", np.round(fx.cpu().numpy(), 3).tolist(),
                              "moving", np.round(mv.cpu().numpy(), 3).tolist(), "cube", np.round(cube, 3).tolist(), "cube-mid in gripper frame", np.round(R.T @ (cube - mid), 3).tolist(),
                              "gx", np.round(R[:, 0], 2).tolist(), "gy", np.round(R[:, 1], 2).tolist(), "gz", np.round(R[:, 2], 2).tolist(), flush=True)
            elif self.phase[i] == "settle":
                self.s[i] += 1; pos_des[i] = self.p1[i]
                if (self.s[i] >= SETTLE_T and self.err[i] < 0.004) or self.s[i] >= 60:
                    self._next_step(i, st)
            if self.phase[i] in ("grip",):
                pos_des[i] = self.p_hold[i]
            quat_des[i] = quat_from_axes(np.array([math.cos(self.yaw[i]), math.sin(self.yaw[i]), 0]), np.array([0, 0, 1.0]))
        self.g = grip
        return pos_des, quat_des, grip

    def label(self, i):
        steps = self.spec["trajectory"]
        if self.done[i]:
            return f"done ({len(steps)} steps)"
        st = steps[self.k[i]]
        what = {"grip": f"{'closing' if st['gripper'] == 'close' else 'opening'} gripper", "move": f"moving to node {st['node']}", "settle": f"at node {st['node']}"}[self.phase[i]]
        rt = f"  (retry {self.retries[i]})" if self.retries[i] else ""
        return f"step {self.k[i] + 1}/{len(steps)}: [{st['gripper'].upper()}] node {st['node']} - {what}{rt}"


