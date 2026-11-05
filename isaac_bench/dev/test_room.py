"""Visual test of isaac_bench/room.py: table + SO-101 + wrist camera + randomised room, 2 envs.

    ~/VLA/tools/gpu.sh 900 python -u isaac_bench/dev/test_room.py [--rounds 3] [--noroom] [--lineup]
Saves isaac_bench/out/room/{wrist,overview}_r<k>.png (rows = envs; columns = arm poses / viewpoints) and prints the
per-env room description and timing (seconds per control step with render, 4 physics substeps like tabletop).
--lineup: one env, all props in rows facing +x, photographed from +x (checks prop orientation / scale).
--noroom: same scene without the room (render-speed baseline).
"""
import os, sys, time, argparse, traceback
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--num_envs", type=int, default=2)
ap.add_argument("--noroom", action="store_true")
ap.add_argument("--lineup", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only", default="")
ap.add_argument("--probe", action="store_true")
ap.add_argument("--nostream", action="store_true")
args = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app
if args.nostream:
    import carb
    carb.settings.get_settings().set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/VLA/third_party/so101_workshop/source"))


def main():
    import numpy as np, torch
    from PIL import Image, ImageDraw
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils import configclass
    from sim_to_real_so101.assets.so101 import SO101_CFG
    import room
    WS = os.path.expanduser("~/VLA/third_party/so101_workshop/source/sim_to_real_so101/assets")
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "room"); os.makedirs(OUT, exist_ok=True)
    n = 1 if args.lineup else args.num_envs
    mat = lambda rgb, r=0.5: sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=r)
    W, H = 320, 240
    cam = lambda path, pos=(0, 0, 0), rot=(1, 0, 0, 0): TiledCameraCfg(prim_path=path, data_types=["rgb"], width=W, height=H,
                                                                        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
                                                                        spawn=sim_utils.PinholeCameraCfg(focal_length=13.5, horizontal_aperture=20.955, clipping_range=(0.005, 8.0)))

    @configclass
    class Cfg(InteractiveSceneCfg):   # copied from tabletop.scene_cfg (table, legs, lights, robot, wrist camera)
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.3, 0.3, 0.32)), init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.75)))
        sky = AssetBaseCfg(prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(intensity=220.0, texture_file=f"{WS}/hdri/studio_small_03_1k.exr", visible_in_primary_ray=True,
                                                                                enable_color_temperature=True, color_temperature=6500.0))
        lamp = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/lamp", spawn=sim_utils.SphereLightCfg(intensity=3000.0, radius=0.15, color=(1.0, 0.95, 0.9)), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, 0.0, 0.9)))
        sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=700.0, angle=3.0), init_state=AssetBaseCfg.InitialStateCfg(rot=(0.93, 0.18, -0.30, 0.10)))
        table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/table", spawn=sim_utils.CuboidCfg(size=(0.9, 1.1, 0.04), visual_material=mat((0.7, 0.6, 0.5), 0.6),
                             collision_props=sim_utils.CollisionPropertiesCfg()), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, -0.02)))
        leg_a = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/leg_a", spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.71), visual_material=mat((0.25, 0.25, 0.27))), init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, -0.5, -0.395)))
        leg_b = leg_a.replace(prim_path="{ENV_REGEX_NS}/leg_b", init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0.5, -0.395)))
        leg_c = leg_a.replace(prim_path="{ENV_REGEX_NS}/leg_c", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, -0.5, -0.395)))
        leg_d = leg_a.replace(prim_path="{ENV_REGEX_NS}/leg_d", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.5, -0.395)))
        robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        wrist = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam", data_types=["rgb"], width=W, height=H,
                               offset=TiledCameraCfg.OffsetCfg(pos=(-0.005, 0.06, -0.062), rot=(0.92388, -0.38268, 0.0, 0.0), convention="opengl"),
                               spawn=sim_utils.PinholeCameraCfg(focal_length=13.5, horizontal_aperture=20.955, clipping_range=(0.005, 5.0)))
        over = cam("{ENV_REGEX_NS}/over_cam")

    cfg = Cfg(num_envs=n, env_spacing=3.0, replicate_physics=False)
    if not args.noroom:
        for k, v in room.room_scene_entries(n, spacing=cfg.env_spacing).items():
            setattr(cfg, k, v)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, render_interval=4, device="cuda:0"))
    t0 = time.time()
    scene = InteractiveScene(cfg)
    sim.reset()
    print(f"scene built + reset {time.time() - t0:.1f}s", flush=True)
    stage = sim.stage
    robot, wrist, over = scene["robot"], scene["wrist"], scene["over"]
    org = scene.env_origins
    HOME = torch.tensor([0.0, -0.6, -0.07, 1.51, -1.57, -0.1], device="cuda:0")
    # arm poses for the wrist camera: home (looking down at the table), and lifted / panned so the camera sees the room
    POSES = [HOME, torch.tensor([0.0, -1.2, 0.6, 0.2, -1.57, -0.1]), torch.tensor([1.1, -1.0, 0.4, 0.5, -1.57, -0.1]),
             torch.tensor([-1.1, -1.0, 0.4, 0.5, -1.57, -0.1]), torch.tensor([0.4, -0.2, -0.2, 0.9, -1.57, -0.1])]
    # overview viewpoints (local): a person standing at the far end of the table, a person at the side, a room corner
    VIEWS = [((1.05, 0.0, 0.85), (-0.3, 0.0, -0.2)), ((0.3, -1.0, 0.9), (0.2, 0.6, -0.3)), ((1.35, -1.15, 1.45), (-0.6, 0.8, -0.4)),
             ((-0.9, 1.1, 1.2), (1.0, -0.8, -0.2)), ((0.2, 0.0, 1.5), (1.5, 0.0, 0.3))]

    def step(k=1, render=True):
        for _ in range(k):
            for j in range(4):
                scene.write_data_to_sim(); sim.step(render=render and j == 3); scene.update(1 / 120)

    def grab(c):
        return c.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)

    def label(im, txt):
        im = Image.fromarray(im); ImageDraw.Draw(im).text((3, 3), txt, fill=(255, 255, 0)); return np.asarray(im)

    def look(eye, tgt):
        """put every env's overview camera at local eye looking at local tgt (USD xform; opengl camera: -z forward, +y up)"""
        from pxr import UsdGeom, Gf
        f = np.array(tgt, float) - np.array(eye, float); f /= np.linalg.norm(f)
        rt = np.cross(f, [0, 0, 1.0]); rt /= np.linalg.norm(rt); up = np.cross(rt, f)
        from scipy.spatial.transform import Rotation as Rot
        qx, qy, qz, qw = Rot.from_matrix(np.stack([rt, up, -f], 1)).as_quat()
        for i in range(n):
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{i}/over_cam")); xf.ClearXformOpOrder()
            o = np.zeros(3)   # the camera prim lives under env_i: local coordinates
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(np.array(eye) + o)))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(qw), float(qx), float(qy), float(qz)))

    def set_pose(q):
        q = q.to("cuda:0")[None].repeat(n, 1)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q)); robot.set_joint_position_target(q)

    if args.lineup:
        from pxr import UsdGeom
        rr = room.RoomRandomizer(stage, 1, np.random.default_rng(0), sim=sim, warmup=False)
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/room_shell")).MakeInvisible()
        names = args.only.split(",") if args.only else list(room.PROPS)
        per = 8
        for b in range(0, len(names), per):
            for k in names:
                rr._set_prop(0, k, (0, 0, -10), 0.0, False)
            grp = names[b:b + per]
            ws = [room.PROPS[k]["size"][1] + 0.3 for k in grp]; cum = np.cumsum([0] + ws); cum = cum - cum[-1] / 2
            for j, k in enumerate(grp):
                y = (cum[j] + cum[j + 1]) / 2
                rr._set_prop(0, k, (2.0, y, -0.75 + (1.0 if room.PROPS[k]["cat"] in ("wall",) else 0.0)), 0.0, True)
            look((2.0 + 1.2 * max(2.0, cum[-1]), -0.6 * cum[-1], 1.0), (2.0, 0.0, -0.2))
            step(6)
            im = grab(over)[0]; Image.fromarray(im).resize((960, 720)).save(f"{OUT}/lineup_{b // per}.png")
            print("lineup", b // per, grp, flush=True)
        return

    if args.probe:   # window + ceiling light close-ups (emissive materials check)
        rr = room.RoomRandomizer(stage, n, np.random.default_rng(args.seed), sim=sim)
        from pxr import UsdGeom, UsdLux
        x0, x1, y0, y1 = rr.bounds
        ims = []
        for vw in rr.views:
            for i in range(n):
                wp = f"/World/envs/env_{i}/room_shell/window_0"; UsdGeom.Imageable(stage.GetPrimAtPath(wp)).MakeVisible()
                rr._xf(wp, (x1, 0.0, -0.75), np.pi); rr._show_view(wp, vw); rr._toggle_sub(wp + "/frame", "frame_white")
                UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}/room_shell/window_1")).MakeInvisible()
            look((0.6, 0.0, 0.8), (x1, 0.0, 0.7)); step(1); a1 = grab(over)[0]
            for _ in range(20):
                time.sleep(0.1); step(1)
            ims.append(np.concatenate([a1, grab(over)[0]], 0))
        look((0.6, 0.0, 0.3), (0.2, 0.0, 1.75)); step(4); ims.append(np.concatenate([grab(over)[0]] * 2, 0))
        Image.fromarray(np.concatenate(ims, 1)).save(f"{OUT}/probe.png"); print("probe saved", rr.views, flush=True)
        return
    rr = None
    if not args.noroom:
        t0 = time.time()
        rr = room.RoomRandomizer(stage, n, np.random.default_rng(args.seed), sim=sim)
        print(f"randomizer init {time.time() - t0:.1f}s", flush=True)
    set_pose(HOME); step(10)
    # speed: steps with the arm moving between home and pose 1
    ts = []
    for s in range(40):
        a = 0.5 + 0.5 * np.sin(s / 6)
        robot.set_joint_position_target(((1 - a) * HOME + a * POSES[4].to("cuda:0"))[None].repeat(n, 1))
        t1 = time.time(); step(1); torch.cuda.synchronize(); ts.append(time.time() - t1)
    print(f"STEP_TIME noroom={args.noroom} envs={n}: median {np.median(ts[10:]) * 1000:.1f} ms  mean {np.mean(ts[10:]) * 1000:.1f} ms", flush=True)
    for rnd in range(args.rounds):
        if rr is not None:
            t1 = time.time(); info = rr.randomize(range(n)); dt_r = time.time() - t1
            t1 = time.time(); step(1); dt_s = time.time() - t1
            print(f"round {rnd}: randomize {dt_r * 1000:.1f} ms, first step after {dt_s * 1000:.0f} ms", flush=True)
            for i in range(n):
                d = dict(info[i]); props = d.pop("props")
                print(f"  env{i}: {d}\n         props: {sorted(props)}", flush=True)
        wr = [[] for _ in range(n)]
        for q in POSES:
            set_pose(q); step(4)
            ims = grab(wrist)
            for i in range(n):
                wr[i].append(ims[i])
        ov = [[] for _ in range(n)]
        for (e, t) in VIEWS:
            look(e, t)
            set_pose(HOME); step(3)
            ims = grab(over)
            for i in range(n):
                ov[i].append(ims[i])
        Image.fromarray(np.concatenate([np.concatenate(r, 1) for r in wr], 0)).save(f"{OUT}/wrist_r{rnd}.png")
        Image.fromarray(np.concatenate([np.concatenate(r, 1) for r in ov], 0)).save(f"{OUT}/overview_r{rnd}.png")
    print("saved to", OUT, flush=True)


try:
    main()
except Exception:
    traceback.print_exc(); sys.stdout.flush(); os._exit(1)
sys.stdout.flush(); os._exit(0)
