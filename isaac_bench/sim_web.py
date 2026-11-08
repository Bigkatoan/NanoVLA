"""Web control of the SO-101 in the Isaac Lab tabletop scene (the same scene the VLA is trained in).

One Isaac env runs in the main thread at 30 Hz; an aiohttp server (background thread) serves web/sim_web.html and a
WebSocket: the browser sends commands (Cartesian jog, joint targets, gripper, home, new scene), the server pushes the
front-camera and wrist-camera views as JPEG plus the robot state.

    TABLETOP_FRONT=1 TABLETOP_RES=240x320 python sim_web.py --port 8090
"""
import argparse, asyncio, json, math, os, queue, sys, threading, time
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8090); ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tasks", default="pick,pad,tray,container,push,topple,rotate,rel")
ap.add_argument("--fps", type=float, default=15.0); ap.add_argument("--jpeg", type=int, default=80)
ap.add_argument("--nano_ik", default=os.path.expanduser("~/VLA/ckpt/nano_ik/nano_ik.pt"))
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
import numpy as np, torch, cv2
from aiohttp import web, WSMsgType
import tabletop as tt
from tabletop_expert import Expert, quat_from_axes, OPEN, CLOSE

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "sim_web.html")
CMD = queue.Queue()                                   # browser -> sim
LATEST = {"front": None, "wrist": None, "state": None, "seq": 0}
LOCK = threading.Lock()
JOG = {"v": [0.0, 0.0, 0.0], "w": 0.0, "t": 0.0}       # current jog velocity (m/s, rad/s) + time of the last jog message


# ------------------------------------------------------------------ web server (background thread)
def serve():
    async def index(_):
        return web.FileResponse(HTML)

    async def ws_handler(req):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20); await ws.prepare(req)

        async def push():
            last = -1
            while not ws.closed:
                with LOCK:
                    seq, f, w, s = LATEST["seq"], LATEST["front"], LATEST["wrist"], LATEST["state"]
                if seq != last and f is not None:
                    last = seq
                    try:
                        await ws.send_bytes(b"\x00" + f); await ws.send_bytes(b"\x01" + w); await ws.send_str(json.dumps(s))
                    except Exception:
                        break
                await asyncio.sleep(1 / a.fps)
        task = asyncio.ensure_future(push())
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    m = json.loads(msg.data)
                except Exception:
                    continue
                if m.get("type") == "jog":
                    JOG["v"] = [float(x) for x in m.get("v", [0, 0, 0])][:3]; JOG["w"] = float(m.get("w", 0)); JOG["t"] = time.time()
                else:
                    CMD.put(m)
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
        task.cancel(); return ws

    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    appw = web.Application(); appw.add_routes([web.get("/", index), web.get("/ws", ws_handler)])
    runner = web.AppRunner(appw); loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, a.host, a.port).start())
    print(json.dumps(dict(event="web_ready", url=f"http://{a.host}:{a.port}")), flush=True)
    loop.run_forever()


# ------------------------------------------------------------------ simulation (main thread)
def main():
    env = tt.Tabletop(num_envs=1, seed=a.seed, task_mix=tuple(a.tasks.split(",")))
    dev = env.dev; dt = 1.0 / tt.CTRL_HZ
    kin = Expert(env, np.random.default_rng(a.seed))
    nik = None
    if a.nano_ik and os.path.exists(a.nano_ik):
        sys.path.insert(0, os.path.expanduser("~/VLA/tools")); from nano_ik import NanoIK
        nik = NanoIK().to(dev); nik.load_state_dict(torch.load(a.nano_ik, map_location=dev)["ik"]); nik.eval()
    bi = env.robot.body_names.index("gripper")
    from isaaclab.utils import math as mu
    FLIP = np.diag([1.0, -1.0, -1.0])

    def grip_yaw():
        R = mu.matrix_from_quat(env.robot.data.body_quat_w[:, bi])[0].cpu().numpy(); return math.atan2(R[1, 0], R[0, 0]), R

    def new_scene():
        env.reset()
        for _ in range(10):
            env.step(np.tile(tt.HOME, (1, 1)))
        kin.q_cmd = env.robot.data.joint_pos.clone()
        return float(env.robot.data.joint_pos[0, 5])

    g = new_scene(); g_t = g; yaw_t = grip_yaw()[0]; mode = "analytic"; jtarget = None; hold = None; steps = 0; t_last = time.time()
    print(json.dumps(dict(event="sim_ready", task=env.ep[0]["text"])), flush=True)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg]
    while True:
        t0 = time.time()
        while not CMD.empty():                                         # discrete commands
            m = CMD.get(); k = m.get("type")
            hold = None if k in ("home", "reset", "joints") else hold
            if k == "grip":
                g_t = OPEN if m.get("open", True) else CLOSE
            elif k == "home":
                jtarget = tt.HOME.copy(); g_t = jtarget[5]
            elif k == "reset":
                g = new_scene(); g_t = g; yaw_t = grip_yaw()[0]; jtarget = None
            elif k == "joints":
                jtarget = np.asarray(m["q"], float)[:6]; g_t = float(jtarget[5])
            elif k == "ik_mode":
                mode = "nano" if (m.get("mode") == "nano" and nik is not None) else "analytic"
        v = np.asarray(JOG["v"]); w = JOG["w"]
        if time.time() - JOG["t"] > 0.4:                               # browser stopped sending: stop moving
            v[:] = 0; w = 0.0
        if np.any(v != 0) or w != 0:
            jtarget = None
            cur_t, qcur = kin.tcp_pose(); cur = cur_t.cpu().numpy()
            if hold is None:                                           # jog start: keep the CURRENT orientation (only yaw changes)
                hold = dict(q=qcur.clone(), yaw=0.0, p=cur.copy())
            hold["yaw"] += float(np.clip(w, -1.5, 1.5)) * dt; yaw_t = grip_yaw()[0] + float(np.clip(w, -1.5, 1.5)) * dt
            hold["p"] = hold["p"] + np.clip(v, -0.25, 0.25)[None] * dt  # integrated set-point (no drift while the IK catches up)
            pos_des = cur + np.clip(hold["p"] - cur, -0.01, 0.01)
            qz = mu.quat_from_angle_axis(torch.tensor([hold["yaw"]], device=dev), torch.tensor([[0.0, 0.0, 1.0]], device=dev))
            quat_t = mu.quat_mul(qz, hold["q"])
            if mode == "analytic":
                dq, _ = kin.ik_step(torch.tensor(pos_des, device=dev, dtype=torch.float32), quat_t, gain=1.0, wrot=0.08)  # 5-DOF arm: position first
            else:                                                       # NanoIK: displacement in the current wrist-camera frame
                ql = env.dr.cam[0]["o"].Get() if getattr(env, "dr", None) is not None else None
                _, Rb = grip_yaw()
                qloc = np.r_[ql.GetReal(), list(ql.GetImaginary())] if ql is not None else np.array([0.92388, -0.38268, 0.0, 0.0])
                w_, x_, y_, z_ = qloc / np.linalg.norm(qloc)
                Rl = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)], [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                               [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)]])
                Rc = Rb @ Rl @ FLIP; dp = Rc.T @ (pos_des[0] - cur[0])
                dyw = np.clip((yaw_t - grip_yaw()[0] + math.pi) % (2 * math.pi) - math.pi, -0.05, 0.05)
                qa = env.robot.data.joint_pos
                with torch.no_grad():
                    dq = nik(qa.float(), torch.tensor(dp[None], device=dev, dtype=torch.float32), torch.tensor([dyw], device=dev, dtype=torch.float32))
                kin.q_cmd[:, :5] = qa[:, :5]
            kin.q_cmd[:, :5] += dq
        else:
            hold = None
        if jtarget is not None and hold is None:                                      # slew towards the joint target (<= 1.5 rad/s)
            qc = kin.q_cmd[0, :5].cpu().numpy(); st_ = np.clip(jtarget[:5] - qc, -1.5 * dt, 1.5 * dt)
            kin.q_cmd[0, :5] = torch.tensor(qc + st_, device=dev, dtype=torch.float32)
            if np.abs(jtarget[:5] - qc).max() < 1e-3:
                jtarget = None; yaw_t = grip_yaw()[0]
        g = g + float(np.clip(g_t - g, -(OPEN - CLOSE) / 18, (OPEN - CLOSE) / 18)); kin.q_cmd[:, 5] = g
        lim = env.robot.data.soft_joint_pos_limits; kin.q_cmd = torch.clamp(kin.q_cmd, lim[..., 0], lim[..., 1])
        steps += 1; frame_step = steps % max(1, int(round(tt.CTRL_HZ / a.fps))) == 0
        env.robot.set_joint_position_target(kin.q_cmd); env._phys(render=frame_step); env.t += 1   # render only when a frame is sent
        if frame_step:
            f, wr = env.images()
            fj = cv2.imencode(".jpg", cv2.cvtColor(f[0], cv2.COLOR_RGB2BGR), enc)[1].tobytes()
            wj = cv2.imencode(".jpg", cv2.cvtColor(wr[0], cv2.COLOR_RGB2BGR), enc)[1].tobytes()
            tcp = kin.tcp_pose()[0][0].cpu().numpy()
            try:
                ok = bool(np.asarray(env.success())[0])
            except Exception:
                ok = False
            now = time.time(); hz = 1.0 / max(1e-6, (now - t_last) / max(1, int(round(tt.CTRL_HZ / a.fps)))); t_last = now
            st = dict(q=env.state()[0].round(4).tolist(), q_cmd=kin.q_cmd[0].cpu().numpy().round(4).tolist(), tcp=tcp.round(4).tolist(), yaw=round(grip_yaw()[0], 3),
                      task=env.ep[0]["text"], success=ok, mode=mode, nano_ik=nik is not None, sim_hz=round(hz, 1), t=round(steps * dt, 1),
                      lim=env.robot.data.soft_joint_pos_limits[0].cpu().numpy().round(3).tolist())
            with LOCK:
                LATEST.update(front=fj, wrist=wj, state=st, seq=LATEST["seq"] + 1)
        sl = dt - (time.time() - t0)
        if sl > 0:
            time.sleep(sl)


if __name__ == "__main__":
    threading.Thread(target=serve, daemon=True).start()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback; traceback.print_exc()
    sys.stdout.flush(); os._exit(0)
