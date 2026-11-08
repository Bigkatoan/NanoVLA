"""Run the distilled wrist-camera student on a real SO-101 (Jetson Orin Nano or PC).

Every control tick (30 Hz) the next joint target of the current action chunk is sent; a new chunk is predicted every
--exec ticks (default 8 -> ~4 Hz re-planning, inference ~1 ms on a 3060 / a few ms on the Orin with TensorRT fp16).
Chunks overlap: the new chunk is blended into the old one over --blend ticks to avoid jumps.

Robot: lerobot SO101Follower with use_degrees=True (the units the teacher and student were trained in).
Optional --record DIR saves every tick (wrist + overview frames, joints, the action sent) as a LeRobot dataset with
the teacher's camera layout (camera1 wrist, camera2 overview) -> the teacher can label it (DAgger round).

    python deploy/run_student.py --onnx ckpt/student/task.onnx --meta ckpt/student/task.meta.json --task "put the red X in the top left cell" \
        --port /dev/ttyACM0 --robot_id my_so101 --wrist_cam 0 --front_cam 2 --seconds 40 --record data/lerobot/dagger_r1
    python deploy/run_student.py --onnx ... --meta ... --task ... --dry_run          # no robot / cameras: synthetic inputs, timing only
"""
import argparse, json, os, time
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def session(onnx_path):
    import onnxruntime as ort
    prov = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    opts = {"TensorrtExecutionProvider": {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": os.path.dirname(os.path.abspath(onnx_path))}}
    s = ort.InferenceSession(onnx_path, providers=[(p, opts[p]) if p in opts else p for p in prov])
    return s, s.get_providers()[0]


def open_cam(idx, w=640, h=480):
    import cv2
    c = cv2.VideoCapture(idx); c.set(cv2.CAP_PROP_FRAME_WIDTH, w); c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not c.isOpened():
        raise SystemExit(f"camera {idx} not available")
    return c


def grab(c):
    import cv2
    ok, f = c.read()
    if not ok:
        raise RuntimeError("camera read failed")
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--port", default="/dev/ttyACM0"); ap.add_argument("--robot_id", default="so101")
    ap.add_argument("--wrist_cam", type=int, default=0); ap.add_argument("--front_cam", type=int, default=-1, help="only needed with --record (teacher labels need it)")
    ap.add_argument("--exec", type=int, default=8); ap.add_argument("--blend", type=int, default=4); ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--max_step_deg", type=float, default=8.0, help="safety: largest joint change sent per tick")
    ap.add_argument("--record", default=""); ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    import cv2
    meta = json.load(open(a.meta)); mu, sd = np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32)
    if a.task not in meta["tasks"]:
        raise SystemExit(f"task not in the student's vocabulary: {meta['tasks']}")
    tid = np.array([meta["tasks"].index(a.task)], np.int64); H, W = meta["image_hw"]; fps = meta.get("fps", 30)
    sess, prov = session(a.onnx); print(json.dumps(dict(provider=prov, task=a.task)), flush=True)

    def infer(wrist_rgb, q):
        x = cv2.resize(wrist_rgb, (W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None].astype(np.uint8)
        out = sess.run(None, {"wrist_u8": x, "state_n": ((q - mu) / sd)[None].astype(np.float32), "task_id": tid})[0][0]
        return out * sd + mu                                          # (chunk, 6) joint targets

    robot = wcam = fcam = rec = None
    if a.dry_run:
        q = mu.copy()
        get_q = lambda: q
        get_w = lambda: np.random.randint(0, 255, (480, 640, 3), np.uint8)
        send = lambda tgt: None
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(port=a.port, id=a.robot_id, use_degrees=True)); robot.connect()
        wcam = open_cam(a.wrist_cam)
        get_q = lambda: np.array([robot.get_observation()[f"{j}.pos"] for j in JOINTS], np.float32)
        get_w = lambda: grab(wcam)
        send = lambda tgt: robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, tgt)})
        if a.record:
            fcam = open_cam(a.front_cam) if a.front_cam >= 0 else None
            if fcam is None:
                raise SystemExit("--record needs --front_cam (the teacher labels need the overview camera)")
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            feats = {c: {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]} for c in ("observation.images.camera1", "observation.images.camera2")}
            feats.update({k: {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINTS]} for k in ("observation.state", "action")})
            rec = LeRobotDataset.create("local/dagger", fps=fps, features=feats, root=a.record, robot_type="so101_follower", use_videos=True, image_writer_threads=4)
    chunk, k, ticks, lat, t_end = None, 0, 0, [], time.time() + a.seconds
    last = get_q()
    try:
        while time.time() < t_end:
            t0 = time.time(); q = get_q(); w = get_w()
            if chunk is None or k >= a.exec:
                t1 = time.time(); new = infer(w, q); lat.append(time.time() - t1)
                if chunk is not None and a.blend > 0:                # cross-fade the overlap of old and new chunk
                    rest = chunk[k:k + a.blend]; n = len(rest); wgt = np.linspace(0, 1, n + 2)[1:-1, None]
                    new[:n] = (1 - wgt) * rest + wgt * new[:n]
                chunk, k = new, 0
            tgt = np.clip(chunk[k], last - a.max_step_deg, last + a.max_step_deg); k += 1; last = tgt
            send(tgt)
            if rec is not None:
                rec.add_frame({"observation.images.camera1": w, "observation.images.camera2": grab(fcam), "observation.state": q, "action": tgt.astype(np.float32), "task": a.task})
            ticks += 1
            time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
    finally:
        if rec is not None:
            rec.save_episode(); rec.finalize()
        if robot is not None:
            robot.disconnect()
    print(json.dumps(dict(ticks=ticks, hz=round(ticks / a.seconds, 1), infer_ms_median=round(1000 * float(np.median(lat)), 2), infer_calls=len(lat))), flush=True)


if __name__ == "__main__":
    main()
