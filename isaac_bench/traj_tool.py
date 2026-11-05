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
ap.add_argument("--cosmos_dir", default="", help="also write high-res RGB + depth + segmentation mp4s per kept episode for Cosmos Transfer (needs TABLETOP_CAM_EXTRA=1)")
ap.add_argument("--skip_frames", type=int, default=30, help="do not record camera frames during the first control steps after a reset")
ap.add_argument("--tasks", default="", help="comma list overriding the spec's task types (e.g. hold one out for few-shot tests)")
ap.add_argument("--only_success", type=int, default=1, help="with --out: keep only episodes that pass the task's success check")
ap.add_argument("--max_steps", type=int, default=45 * 30, help="per-episode time limit in control steps")
ap.add_argument("--speed", type=float, default=0.18, help="m/s of the grasp centre during moves")
a = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import tabletop as tt
from tabletop_expert import Expert, quat_from_axes, OPEN, CLOSE
from traj_core import TrajRunner, CosmosWriter, GRIP_T, SETTLE_T

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
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
    if a.tasks:
        spec["tasks"] = a.tasks.split(",")
    rng = np.random.default_rng(a.seed + 11)
    req = spec.get("scene", {}).get("require", [])
    env = tt.Tabletop(num_envs=a.num_envs, seed=spec.get("scene", {}).get("seed", a.seed), task_mix=tuple(spec.get("tasks") or tt.TASKS))
    runner = TrajRunner(env, spec, rng, speed=a.speed)
    markers = make_markers(spec) if (a.video and a.markers and not a.out) else None   # markers would leak into recorded images

    def sample_ok():
        while True:
            E = env.sample_episode()
            if all(o in E["layout"] for o in req) and (not spec.get("tasks") or E["task"] in spec["tasks"]):
                if spec.get("force_obj") and E["obj"] != spec["force_obj"]:
                    continue
                E["text"] = spec["text"] if spec.get("text") and spec["text"] != "$text" else E["text"]; return E

    frames, rows, acts, meta = [], [], {}, []
    n_done, attempts, ep0_count = 0, 0, 0
    import pandas as pd
    env.reset(episodes=[sample_ok() for _ in range(env.n)])
    for _ in range(10):
        env.step(np.tile(tt.HOME, (env.n, 1)))
    env.t[:] = 0
    runner.reset()
    buf = [dict(S=[], A=[], F=[]) for _ in range(env.n)]
    max_bottom = np.full(env.n, -1.0)
    cw = None
    if a.cosmos_dir:
        assert tt.CAM_EXTRA, "set TABLETOP_CAM_EXTRA=1 (and TABLETOP_CAM_SCALE=4) for the Cosmos export"
        os.makedirs(a.cosmos_dir, exist_ok=True); cw = [CosmosWriter(a.cosmos_dir, i) for i in range(env.n)]
        for w_ in cw:
            w_.start()
    t0 = time.time()
    # asynchronous episodes: an env that finishes is scored, saved and reset on its own, the others keep running
    while n_done < a.episodes:
        for i in range(env.n):
            E_ = env.ep[i]
            max_bottom[i] = max(max_bottom[i], env.pos(E_["obj"])[i][2] + tt.OBJ_META[E_["obj"]]["zmin"])
        cmd = runner.step()
        S = env.state()
        rec = env.t % 3 == 0
        vis_frame = bool(a.video) and ep0_count < a.video_eps
        if markers is not None and vis_frame:
            pts, kinds = [], []
            for st_i, st in enumerate(spec["trajectory"]):
                p, _ = runner.resolve(0, st["node"], runner.yaw[0])
                pts.append(p.copy() + env.origins[0].cpu().numpy()); kinds.append(2 if st_i == min(runner.k[0], len(spec["trajectory"]) - 1) and not runner.done[0] else (0 if st["gripper"] == "open" else 1))
            markers.visualize(translations=torch.tensor(np.array(pts), dtype=torch.float32), marker_indices=torch.tensor(kinds))
        if (a.out and rec.any()) or vis_frame:
            f, w = env.images()
        if a.out:
            for i in range(env.n):
                buf[i]["S"].append(S[i].astype(np.float32)); buf[i]["A"].append(cmd[i].astype(np.float32))
                if rec[i]:
                    if env.t[i] >= a.skip_frames:       # renderer ghosting right after a reset: drop the first frames
                        buf[i]["F"].append((int(env.t[i]), None if f is None else f[i], w[i]))
        if cw is not None:
            for i in range(env.n):
                if env.t[i] % 2 == 0:
                    cw[i].add(env.cosmos_frames(i), env.t[i])
        if vis_frame:
            hw = env.scene["wrist"].data.output["rgb"][0, ..., :3].cpu().numpy()
            hf = env.scene["front"].data.output["rgb"][0, ..., :3].cpu().numpy() if tt.FRONT else np.zeros((0, hw.shape[1], 3), np.uint8)
            if not tt.FRONT:
                hf = hw; hw = np.zeros((hw.shape[0], 0, 3), np.uint8)
            im = Image.fromarray(np.concatenate([hf, hw], 1)); W, Hh = im.size
            canvas = Image.new("RGB", (W, Hh + 46), "white"); canvas.paste(im, (0, 46)); d = ImageDraw.Draw(canvas)
            d.text((6, 3), f"\"{env.ep[0]['text']}\"", font=FONT_B, fill="black")
            d.text((6, 24), runner.label(0) + f"   t={env.t[0] / tt.CTRL_HZ:4.1f}s   jaw={S[0, 5]:+.2f} rad", font=FONT, fill=(50, 50, 50))
            if markers is not None:
                uv, ok = project(env, [runner.resolve(0, st["node"], runner.yaw[0])[0] for st in spec["trajectory"]]) if tt.FRONT else (np.zeros((0, 2)), np.zeros(0, bool))
                seen = set()
                for (u, v), good, st in zip(uv, ok, spec["trajectory"]):
                    if good and st["node"] not in seen and 0 <= u < hf.shape[1] and 0 <= v < hf.shape[0]:
                        seen.add(st["node"]); d.text((u + 6, v + 46 - 16), str(st["node"]), font=FONT_B, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            frames.append(np.asarray(canvas))
        env.step(cmd)
        finished = [i for i in range(env.n) if runner.done[i] or env.t[i] >= a.max_steps]
        if not finished:
            continue
        succ = env.success()
        for i in finished:
            E = env.ep[i]
            p_end = env.pos(E["obj"])[i]; q_end = env.quat(E["obj"])[i]
            tgt = runner.sub(i, "$target") if E["task"] in ("pad", "tray", "bowl", "rel", "container") else None
            tgt_xy = (np.asarray(E["rel_xy"]) if tgt == "$rel" else env.tray_center(i)[0] if tgt == "tray" else env.pos(tgt)[i][:2]) if tgt else None
            up_z = 1 - 2 * (q_end[1] ** 2 + q_end[2] ** 2)          # z-component of the object's up axis (1 = upright)
            r_ = dict(ep=n_done, text=E["text"], success_check=bool(succ[i]), task=E["task"], obj=E["obj"], steps=int(env.t[i]),
                      retries=int(runner.retries[i]), lifted_cm=round(100 * max_bottom[i], 1), end_bottom_cm=round(100 * (p_end[2] + tt.OBJ_META[E["obj"]]["zmin"]), 1),
                      end_dist_cm=round(100 * float(np.linalg.norm(p_end[:2] - tgt_xy)), 1) if tgt else None, upright=round(float(up_z), 2),
                      jaw=round(float(S[i, 5]), 2), el=round(time.time() - t0), texture=os.path.basename(tt.TEX[E["table_mat"]]),
                      dr=E.get("dr"), hdri=getattr(env, "hdri", None),
                      layout={k: [float(v[0][0]), float(v[0][1]), float(v[1])] for k, v in E["layout"].items()})
            attempts += 1
            if i == 0:
                ep0_count += 1
            keep = n_done < a.episodes and not (a.out and a.only_success and not succ[i])
            if keep:
                meta.append(r_)
                if a.out:
                    e = n_done
                    acts[f"ep{e}_action"] = np.stack(buf[i]["A"]); acts[f"ep{e}_state"] = np.stack(buf[i]["S"])
                    for fidx, fi, wi in buf[i]["F"]:
                        for cam, im in (("front", fi), ("wrist", wi)):
                            if im is None:
                                continue
                            b = io.BytesIO(); Image.fromarray(im).save(b, format="JPEG", quality=90)
                            rows.append((e, cam, fidx, b.getvalue(), E["text"], E["text"]))
                n_done += 1
            if cw is not None:
                cw[i].finish(keep_as=f"ep{r_['ep']:05d}" if keep else None, meta={k_: v_ for k_, v_ in r_.items() if k_ != "layout"})
                cw[i].start()
            print(json.dumps(r_ | {"layout": None, "kept": keep, "attempts": attempts}), flush=True)
            buf[i] = dict(S=[], A=[], F=[]); max_bottom[i] = -1.0
        if n_done >= a.episodes:
            break
        env.reset(env_ids=finished, episodes=[sample_ok() for _ in finished], settle=0)
        for i in finished:
            runner.reset_env(i)
        if markers is not None and 0 in finished:
            markers.set_visibility(ep0_count < a.video_eps)
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
        json.dump(dict(spec=spec, episodes=meta, attempts=attempts, kept=len(meta)), open(os.path.join(a.out, "episodes.json"), "w"), indent=1)
        print("saved", a.out, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)   # Kit would hang on exit
    sys.stdout.flush(); os._exit(0)
