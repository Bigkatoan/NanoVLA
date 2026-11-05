"""Node-based trajectory tool for data collection in the SO-101 tabletop (Isaac Lab).

A trajectory spec (JSON) has
  nodes      : id -> target. Either absolute   {"pos": [x, y, z], "yaw": deg?}
                               or object-relative {"obj": "red_cube", "offset": [dx, dy, dz], "yaw": "auto" | deg?}
               (positions in metres, robot-base frame: x forward, z up from the table top; object-relative nodes are
                resolved when the step that uses them starts, so they follow the object wherever it is)
  trajectory : list of steps {"node": id, "gripper": "open" | "close"}
               Execution of one step: FIRST drive the gripper to the requested state (and wait for it to settle),
               THEN move the grasp centre to the node in a straight line (smoothstep timing).
  text       : language instruction stored with every recorded episode
  scene      : optional {"task_hint": ..., "require": [objects that must be on the table], "seed": int}

Usage:
  python traj_tool.py --spec specs/pick_place.json --episodes 1 --video out/traj_demo.mp4   # visualise
  python traj_tool.py --spec specs/pick_place.json --episodes 50 --out ~/VLA/data/traj_pick_place  # collect
"""
import argparse, io, json, math, os, sys, time
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="", help="dataset dir (bench format); empty = do not save data")
ap.add_argument("--video", default="", help="mp4 path for an annotated visualisation of env 0")
ap.add_argument("--video_eps", type=int, default=1, help="number of episodes (env 0) in the video")
ap.add_argument("--markers", type=int, default=1, help="draw node markers in the video (never in recorded data)")
ap.add_argument("--speed", type=float, default=0.18, help="m/s of the grasp centre during moves")
a = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import tabletop as tt
from tabletop_expert import Expert, quat_from_axes, OPEN, CLOSE

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
GRIP_T = 18          # control steps (0.6 s) to actuate the gripper
SETTLE_T = 6         # steps to settle after a move


class TrajRunner:
    """Runs one spec on every env in parallel (each env resolves object-relative nodes against its own layout)."""

    def __init__(self, env, spec, rng):
        self.env, self.spec, self.rng = env, spec, rng
        self.kin = Expert(env, rng)          # reuse kinematics + IK (grasp centre, Jacobian, DLS step)
        self.n = env.n

    def sub(self, i, name):
        """template placeholders: $obj -> the episode's object, $target -> its pad / tray / bowl"""
        E = self.env.ep[i]
        if name == "$obj":
            return E["obj"]
        if name == "$target":
            return E.get("pad") or ("tray" if E["task"] == "tray" else "bowl")
        return name

    def resolve(self, i, nid, cur_yaw):
        nd = self.spec["nodes"][str(nid)]
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
        p, q = self.kin.tcp_pose(); p = p.cpu().numpy()
        from isaaclab.utils import math as mu
        xg = mu.matrix_from_quat(q)[:, :, 0].cpu().numpy()
        self.yaw = np.arctan2(xg[:, 1], xg[:, 0])
        self.p_hold = p.copy(); self.p0 = p.copy(); self.p1 = p.copy(); self.T = np.ones(self.n, int)
        self.g = self.env.state()[:, 5].copy(); self.g0 = self.g.copy()
        self.done = np.zeros(self.n, bool); self.err = np.ones(self.n)
        self.kin.q_cmd = self.env.robot.data.joint_pos.clone()

    def step(self):
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
                    self.T[i] = max(8, int(np.linalg.norm(p1 - self.p0[i]) / a.speed * tt.CTRL_HZ))
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
                        cube = self.env.pos("red_cube")[0]; mid = ((fx + mv) / 2).cpu().numpy()
                        print("DBG node", st["node"], "target", np.round(self.p1[i], 3).tolist(), "mid", np.round(mid, 3).tolist(), "fixed", np.round(fx.cpu().numpy(), 3).tolist(),
                              "moving", np.round(mv.cpu().numpy(), 3).tolist(), "cube", np.round(cube, 3).tolist(), "cube-mid in gripper frame", np.round(R.T @ (cube - mid), 3).tolist(),
                              "gx", np.round(R[:, 0], 2).tolist(), "gy", np.round(R[:, 1], 2).tolist(), "gz", np.round(R[:, 2], 2).tolist(), flush=True)
            elif self.phase[i] == "settle":
                self.s[i] += 1; pos_des[i] = self.p1[i]
                if (self.s[i] >= SETTLE_T and self.err[i] < 0.004) or self.s[i] >= 30:
                    self.k[i] += 1; self.phase[i], self.s[i] = "grip", 0
                    if self.k[i] >= len(steps):
                        self.done[i] = True
            if self.phase[i] in ("grip",):
                pos_des[i] = self.p_hold[i]
            quat_des[i] = quat_from_axes(np.array([math.cos(self.yaw[i]), math.sin(self.yaw[i]), 0]), np.array([0, 0, 1.0]))
        self.g = grip
        dev = self.env.dev
        dq, err = self.kin.ik_step(torch.tensor(pos_des, device=dev, dtype=torch.float32), torch.tensor(quat_des, device=dev, dtype=torch.float32), gain=1.0)
        self.err = err.cpu().numpy()
        self.kin.q_cmd[:, :5] += dq
        self.kin.q_cmd[:, 5] = torch.tensor(grip, device=dev, dtype=torch.float32)
        lim = self.env.robot.data.soft_joint_pos_limits
        self.kin.q_cmd = torch.clamp(self.kin.q_cmd, lim[..., 0], lim[..., 1])
        return self.kin.q_cmd.cpu().numpy().astype(np.float64)

    def label(self, i):
        steps = self.spec["trajectory"]
        if self.done[i]:
            return f"done ({len(steps)} steps)"
        st = steps[self.k[i]]
        what = {"grip": f"{'closing' if st['gripper'] == 'close' else 'opening'} gripper", "move": f"moving to node {st['node']}", "settle": f"at node {st['node']}"}[self.phase[i]]
        return f"step {self.k[i] + 1}/{len(steps)}: [{st['gripper'].upper()}] node {st['node']} - {what}"


def make_markers(spec):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    cfg = VisualizationMarkersCfg(prim_path="/Visuals/traj_nodes", markers={
        "open": sim_utils.SphereCfg(radius=0.009, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.2), emissive_color=(0.0, 0.4, 0.0))),
        "close": sim_utils.SphereCfg(radius=0.009, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.1), emissive_color=(0.4, 0.0, 0.0))),
        "active": sim_utils.SphereCfg(radius=0.014, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.1), emissive_color=(0.6, 0.5, 0.0), opacity=0.6))})
    return VisualizationMarkers(cfg)


def project(env, pts_local):
    """local 3D points -> pixel coords in env 0's high-res front image (for node id labels)"""
    cam = env.scene["front"]
    from isaaclab.utils import math as mu
    K = cam.data.intrinsic_matrices[0].cpu().numpy()
    pw = torch.tensor(pts_local, dtype=torch.float32, device=env.dev) + env.origins[0]
    # camera-frame transform (ROS convention: x right, y down, z forward)
    R = mu.matrix_from_quat(cam.data.quat_w_ros[0]).cpu().numpy(); t = cam.data.pos_w[0].cpu().numpy()
    P = (pw.cpu().numpy() - t) @ R
    uv = (P @ K.T); uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    return uv, P[:, 2] > 0


def main():
    spec = json.load(open(a.spec))
    rng = np.random.default_rng(a.seed + 11)
    req = spec.get("scene", {}).get("require", [])
    env = tt.Tabletop(num_envs=a.num_envs, seed=spec.get("scene", {}).get("seed", a.seed), task_mix=tuple(spec.get("tasks") or tt.TASKS))
    runner = TrajRunner(env, spec, rng)
    markers = make_markers(spec) if (a.video and a.markers and not a.out) else None   # markers would leak into recorded images

    def sample_ok():
        while True:
            E = env.sample_episode()
            if all(o in E["layout"] for o in req) and (not spec.get("tasks") or E["task"] in spec["tasks"]):
                E["text"] = spec["text"] if spec.get("text") and spec["text"] != "$text" else E["text"]; return E

    frames, rows, acts, meta = [], [], {}, []
    n_done = 0
    import pandas as pd
    while n_done < a.episodes:
        eps = env.reset(episodes=[sample_ok() for _ in range(env.n)])
        for _ in range(10):
            env.step(np.tile(tt.HOME, (env.n, 1)))
        env.t[:] = 0
        runner.reset()
        buf = [dict(S=[], A=[], F=[]) for _ in range(env.n)]
        node_ids = sorted(spec["nodes"], key=lambda x: int(x) if x.isdigit() else x)
        while not runner.done.all() and env.t.max() < 60 * tt.CTRL_HZ:
            cmd = runner.step()
            S = env.state()
            rec = env.t % 3 == 0
            vis_frame = bool(a.video) and n_done < a.video_eps
            if markers is not None and vis_frame:
                pts, kinds = [], []
                for st_i, st in enumerate(spec["trajectory"]):
                    p, _ = runner.resolve(0, st["node"], runner.yaw[0])
                    p = p.copy()
                    pts.append(p + env.origins[0].cpu().numpy()); kinds.append(2 if st_i == min(runner.k[0], len(spec["trajectory"]) - 1) and not runner.done[0] else (0 if st["gripper"] == "open" else 1))
                markers.visualize(translations=torch.tensor(np.array(pts), dtype=torch.float32), marker_indices=torch.tensor(kinds))
            if a.out and rec.any() or vis_frame:
                f, w = env.images()
            if a.out:
                for i in range(env.n):
                    buf[i]["S"].append(S[i].astype(np.float32)); buf[i]["A"].append(cmd[i].astype(np.float32))
                    if rec[i]:
                        buf[i]["F"].append((int(env.t[i]), f[i], w[i]))
            if vis_frame:
                hf = env.scene["front"].data.output["rgb"][0, ..., :3].cpu().numpy(); hw = env.scene["wrist"].data.output["rgb"][0, ..., :3].cpu().numpy()
                im = Image.fromarray(np.concatenate([hf, hw], 1)); W, Hh = im.size
                canvas = Image.new("RGB", (W, Hh + 46), "white"); canvas.paste(im, (0, 46)); d = ImageDraw.Draw(canvas)
                d.text((6, 3), f"\"{env.ep[0]['text']}\"", font=FONT_B, fill="black")
                d.text((6, 24), runner.label(0) + f"   t={env.t[0] / tt.CTRL_HZ:4.1f}s   jaw={S[0, 5]:+.2f} rad", font=FONT, fill=(50, 50, 50))
                if markers is not None:
                    uv, ok = project(env, [runner.resolve(0, st["node"], runner.yaw[0])[0] for st in spec["trajectory"]])
                    seen = set()
                    for (u, v), good, st in zip(uv, ok, spec["trajectory"]):
                        if good and st["node"] not in seen and 0 <= u < hf.shape[1] and 0 <= v < hf.shape[0]:
                            seen.add(st["node"]); d.text((u + 6, v + 46 - 16), str(st["node"]), font=FONT_B, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                frames.append(np.asarray(canvas))
            env.step(cmd)
        if markers is not None:
            markers.set_visibility(n_done + env.n < a.video_eps)
        succ = env.success()
        for i in range(env.n):
            if n_done >= a.episodes:
                break
            E = env.ep[i]
            meta.append(dict(ep=n_done, text=E["text"], success_check=bool(succ[i]), task=E["task"], obj=E["obj"], steps=int(env.t[i]),
                             layout={k: [float(v[0][0]), float(v[0][1]), float(v[1])] for k, v in E["layout"].items()}))
            if a.out:
                e = n_done
                acts[f"ep{e}_action"] = np.stack(buf[i]["A"]); acts[f"ep{e}_state"] = np.stack(buf[i]["S"])
                for fidx, fi, wi in buf[i]["F"]:
                    for cam, im in (("front", fi), ("wrist", wi)):
                        b = io.BytesIO(); Image.fromarray(im).save(b, format="JPEG", quality=90)
                        rows.append((e, cam, fidx, b.getvalue(), E["text"], E["text"]))
            n_done += 1
            print(json.dumps(meta[-1] | {"layout": None}), flush=True)
    if a.video and frames:
        import av
        os.makedirs(os.path.dirname(os.path.abspath(a.video)), exist_ok=True)
        c = av.open(a.video, "w"); s = c.add_stream("libx264", rate=tt.CTRL_HZ); s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close(); print("video", a.video, len(frames), "frames", flush=True)
    if a.out:
        name = "traj__" + os.path.splitext(os.path.basename(a.spec))[0]
        os.makedirs(os.path.join(a.out, "frames"), exist_ok=True); os.makedirs(os.path.join(a.out, "actions"), exist_ok=True)
        F = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig"]); F["fps"] = 30
        F.to_parquet(os.path.join(a.out, "frames", name + ".parquet"))
        np.savez_compressed(os.path.join(a.out, "actions", name + ".npz"), **acts)
        json.dump([name.replace("__", "/")], open(os.path.join(a.out, "repos.json"), "w"))
        json.dump(dict(spec=spec, episodes=meta), open(os.path.join(a.out, "episodes.json"), "w"), indent=1)
        print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)
