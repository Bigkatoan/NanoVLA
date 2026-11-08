"""Physics sanity test + contact sheet for the object library built by isaac_bench/assets/fetch_objects.py.

All objects are spawned at once on a flat ground (0.6 m grid, 5 mm above their resting height, same rigid-body /
collision props as tabletop.py), simulated for 2 s (CPU PhysX so that scene-query raycasts see the settled poses):
  rest_ok      : no NaN, no explosion (peak speed < 1.5 m/s), at rest (< 3 cm/s), drift < 5 cm, upright, no sinking
  rest_upright : body z-axis within 20 deg of vertical
  rest_z       : resting height of the root (= -zmin if perfect)
  sim_grasp_width (graspable): closing width of the *collision* shape at the grasp point, measured with horizontal
                 raycasts from both jaw sides over the band the jaws span (grasp_h - 1 cm ... + 3 cm, +-1 cm across)
  container_ok (containers): a 2.5 cm probe cube dropped into the opening ends up inside
Results -> assets/objects/drop_test.json; objects.json gets the fields, failures are removed from it.
Then a camera visits every object and the renders are tiled into isaac_bench/out/objects/contact_sheet.png.
    ~/VLA/research/tools/gpu.sh 600 python -u isaac_bench/dev/test_objects.py [--no-render] [names...]
"""
import os, sys, json, math, traceback
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--no-render", action="store_true")
ap.add_argument("--no-write", action="store_true", help="do not modify objects.json / drop_test.json")
args = ap.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "assets", "objects")
OUTD = os.path.join(ROOT, "out", "objects")


def main():
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    import omni.usd

    meta = json.load(open(os.path.join(LIB, "objects.json")))
    names = args.names or list(meta)
    DT = 1 / 120
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cpu", enable_scene_query_support=True))
    stage = omni.usd.get_context().get_stage()
    sim_utils.GroundPlaneCfg(visible=False).func("/World/ground", sim_utils.GroundPlaneCfg(visible=False))
    sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0)).func("/World/dome", sim_utils.DomeLightCfg(intensity=900.0))
    dl = sim_utils.DistantLightCfg(intensity=2200.0, angle=2.0)
    dl.func("/World/sun", dl, orientation=(0.93, 0.18, -0.30, 0.10))
    # visual checker floor (2.5 cm squares) for judging scale
    tex = os.path.join(OUTD, "checker_2p5cm.png"); os.makedirs(OUTD, exist_ok=True)
    c = np.kron([[1, 0], [0, 1]], np.ones((32, 32))); Image.fromarray((150 + 60 * c).astype(np.uint8)).convert("RGB").save(tex)
    E = 6.0
    fl = UsdGeom.Mesh.Define(stage, "/World/floor_vis")
    fl.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x, y, 0.0003) for x, y in ((-1, -1), (E, -1), (E, E), (-1, E))]))
    fl.CreateFaceVertexCountsAttr([4]); fl.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    rep = (E + 1) / 0.05
    pv = UsdGeom.PrimvarsAPI(fl).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(rep, 0), Gf.Vec2f(rep, rep), Gf.Vec2f(0, rep)]))
    mat = UsdShade.Material.Define(stage, "/World/Looks/checker"); sh = UsdShade.Shader.Define(stage, "/World/Looks/checker/s"); sh.CreateIdAttr("UsdPreviewSurface")
    tx = UsdShade.Shader.Define(stage, "/World/Looks/checker/t"); tx.CreateIdAttr("UsdUVTexture"); tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat"); tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    rd = UsdShade.Shader.Define(stage, "/World/Looks/checker/r"); rd.CreateIdAttr("UsdPrimvarReader_float2"); rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(fl.GetPrim()).Bind(mat)

    rprops = sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=0.5)
    coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0)
    SP, COLS = 0.6, 8
    objs, home, probes = {}, {}, {}
    for i, n in enumerate(names):
        m = meta[n]
        x, y = (i % COLS) * SP, (i // COLS) * SP
        z = -m["zmin"] + 0.005
        home[n] = (x, y, z)
        objs[n] = RigidObject(RigidObjectCfg(prim_path=f"/World/Objects/{n}", spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(LIB, m["usd"]), rigid_props=rprops, collision_props=coll, mass_props=sim_utils.MassPropertiesCfg(mass=m["mass"])),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z))))
        if m["role"] == "container":
            probes[n] = RigidObject(RigidObjectCfg(prim_path=f"/World/Probes/{n}", spawn=sim_utils.CuboidCfg(
                size=(0.025, 0.025, 0.025), rigid_props=rprops, collision_props=coll, mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.6))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, m["height"] + 0.03))))
    cam = None
    if not args.no_render:
        cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0, height=300, width=300, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0))))
    sim.reset()
    print("[test] spawned", len(objs), "objects", flush=True)
    vmax = {n: 0.0 for n in names}
    for k in range(240):
        sim.step(render=False)
        for n, o in objs.items():
            o.update(DT)
            v = float(torch.linalg.norm(o.data.root_lin_vel_w[0]))
            vmax[n] = max(vmax[n], v if math.isfinite(v) else 1e9)
        for p in probes.values():
            p.update(DT)
    from omni.physx import get_physx_scene_query_interface
    sq = get_physx_scene_query_interface()
    res = {}
    for n, o in objs.items():
        m = meta[n]
        p = o.data.root_pos_w[0].cpu().numpy().astype(float); q = o.data.root_quat_w[0].cpu().numpy().astype(float)
        v = float(torch.linalg.norm(o.data.root_lin_vel_w[0])); w_ = float(torch.linalg.norm(o.data.root_ang_vel_w[0]))
        w, qx, qy, qz = q
        zz = 1 - 2 * (qx * qx + qy * qy); tilt = math.degrees(math.acos(max(-1.0, min(1.0, zz))))
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        drift = float(np.linalg.norm(p[:2] - np.array(home[n][:2])))
        finite = bool(np.all(np.isfinite(p)) and np.all(np.isfinite(q)))
        upright = tilt < 20.0
        zerr = p[2] - (-m["zmin"])
        why = []
        if not finite: why.append("nan")
        if vmax[n] > 1.5: why.append(f"exploded (peak {vmax[n]:.2f} m/s)")
        if v > 0.03 or w_ > 0.5: why.append(f"not at rest (v {v:.3f}, w {w_:.2f})")
        if drift > 0.05: why.append(f"drift {drift:.3f}")
        if not upright: why.append(f"tipped {tilt:.0f} deg")
        if upright and zerr < -0.006: why.append(f"sinks {zerr * 100:.1f} cm")
        r = dict(rest_ok=not why, rest_upright=upright, rest_z=round(float(p[2]), 4), rest_z_err=round(float(zerr), 4), tilt_deg=round(tilt, 1),
                 peak_speed=round(vmax[n], 3), drift=round(drift, 4), why="; ".join(why))
        if m.get("graspable") and finite:
            cy, sy = math.cos(yaw), math.sin(yaw)
            R = np.array([[cy, -sy], [sy, cy]])
            gx, gy = m["grasp_xy"]; g = p[:2] + R @ np.array([gx, gy])
            cax = np.array([1.0, 0.0]) if m["close_axis"] == "x" else np.array([0.0, 1.0]); lax = 1 - cax
            cdir, ldir = R @ cax, R @ lax
            bottom = p[2] + m["zmin"]
            lo, hi = [], []
            for t in (-0.01, 0.0, 0.01):
                for dz in (-0.008, 0.0, 0.01, 0.02, 0.03):
                    zc = bottom + m["grasp_h"] + dz
                    if zc < 0.003:
                        continue
                    for sgn, acc in ((1, hi), (-1, lo)):
                        o3 = (g + ldir * t + sgn * cdir * 0.15).tolist() + [zc]
                        d3 = (-sgn * cdir).tolist() + [0.0]
                        h = sq.raycast_closest(o3, d3, 0.15)
                        if h["hit"] and str(h.get("rigidBody", "")).startswith(f"/World/Objects/{n}"):
                            acc.append(float(np.dot(np.array(h["position"][:2]) - g, cdir)))
            r["sim_grasp_width"] = round(max(hi) - min(lo), 4) if hi and lo else None
        if n in probes:
            pp = probes[n].data.root_pos_w[0].cpu().numpy()
            cy, sy = math.cos(yaw), math.sin(yaw)
            dl_ = np.array([[cy, sy], [-sy, cy]]) @ (pp[:2] - p[:2])          # probe offset in the container frame
            op = m.get("opening") or [0, 0]
            inside = abs(dl_[0]) < op[0] / 2 + 0.003 and abs(dl_[1]) < op[1] / 2 + 0.003 and bottom_ok(pp[2], p[2], m)
            r["container_ok"] = bool(inside); r["probe_z"] = round(float(pp[2] - (p[2] + m["zmin"])), 4)
            if not inside:
                r["rest_ok"] = False; r["why"] = (r["why"] + "; " if r["why"] else "") + f"probe not inside (rel {dl_.round(3).tolist()}, z {r['probe_z']})"
        res[n] = r
        print(f"[test] {n:16s} ok={r['rest_ok']!s:5s} up={upright!s:5s} z_err={zerr * 100:+.2f}cm tilt={tilt:4.1f} peak={vmax[n]:.2f} "
              f"gw={m.get('grasp_width')}/{r.get('sim_grasp_width')} cont={r.get('container_ok')} {r['why']}", flush=True)

    if not args.no_write:
        dt_path = os.path.join(LIB, "drop_test.json")
        old = json.load(open(dt_path)) if os.path.exists(dt_path) else {"results": {}}
        old["results"].update(res)
        json.dump(old, open(dt_path, "w"), indent=1)
        keep = {}
        for n, m in meta.items():
            if n in res:
                m.update({f: res[n][f] for f in ("rest_ok", "rest_upright", "rest_z", "sim_grasp_width", "container_ok") if f in res[n]})
                if not res[n]["rest_ok"]:
                    print("[test] DROP", n, res[n]["why"]); continue
            keep[n] = m
        json.dump(keep, open(os.path.join(LIB, "objects.json"), "w"), indent=1)
        print("[test] objects.json:", len(keep), "objects", flush=True)

    if cam is not None:
        for _ in range(40):
            sim.render()
        cells = []
        dvec = np.array([0.60, -0.42, 0.68]); dvec /= np.linalg.norm(dvec)
        for n, o in objs.items():
            m = meta[n]; p = o.data.root_pos_w[0].cpu().numpy()
            tgt = np.array([p[0], p[1], max(0.01, p[2] + m["zmin"] + 0.45 * m["height"])])
            d = max(0.16, 1.9 * max(m["size"]))
            cam.set_world_poses_from_view(torch.tensor([tgt + d * dvec], dtype=torch.float32), torch.tensor([tgt], dtype=torch.float32))
            for _ in range(10):
                sim.render()
            cam.update(DT)
            im = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            dr = ImageDraw.Draw(im)
            r = res.get(n, {})
            cap = f"{n} [{m['role'][0].upper()}] {'/'.join(m['colors'])}"
            cap2 = "{:.1f}x{:.1f}x{:.1f}cm {:.0f}g".format(*[100 * s for s in m["size"]], 1000 * m["mass"]) + (f" w{100 * m['grasp_width']:.1f}" if m.get("graspable") else "")
            dr.rectangle([0, 0, 300, 26], fill=(0, 0, 0)); dr.text((3, 1), cap, fill=(255, 255, 255)); dr.text((3, 13), cap2, fill=(255, 255, 0) if r.get("rest_ok", True) else (255, 80, 80))
            cells.append(np.asarray(im))
        C = 6; Rr = (len(cells) + C - 1) // C
        sheet = np.zeros((Rr * 300, C * 300, 3), np.uint8)
        for k, c in enumerate(cells):
            sheet[(k // C) * 300:(k // C + 1) * 300, (k % C) * 300:(k % C + 1) * 300] = c
        Image.fromarray(sheet).save(os.path.join(OUTD, "contact_sheet.png"))
        print("[test] contact sheet ->", os.path.join(OUTD, "contact_sheet.png"), flush=True)


def bottom_ok(probe_z, obj_z, m):
    bottom = obj_z + m["zmin"]
    return bottom + 0.005 < probe_z < bottom + m["height"] + 0.005


try:
    main()
except Exception:
    traceback.print_exc(); sys.stdout.flush(); os._exit(1)
sys.stdout.flush(); os._exit(0)
