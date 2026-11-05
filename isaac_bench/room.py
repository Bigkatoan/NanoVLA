"""Randomizable, visual-only room around the SO-101 work table (Isaac Sim 5.1 / Isaac Lab 2.3).

Per env: a closed room (floor, 4 walls, ceiling, baseboards) whose wall / floor / ceiling finish is one of several
pre-built material variants (toggled by visibility: no runtime material rebinding -> no shader recompiles),
1-2 windows (frame + emissive outdoor view variants + a RectLight "window light" that can be switched off),
a door, an optional rug, a ceiling light panel (RectLight), and a pool of office / hospital furniture and clutter
(NVIDIA Isaac Office + Hospital props, assets/fetch_room.py) that is re-laid out every episode: tall furniture and
seating against the walls, chairs / plants / bins / boxes on the free floor, pictures / clock / boards / TV on the
walls, books / binders / vase / monitor on top of low furniture. Everything is visual only (colliders disabled) and
stays outside a keep-out around the table (table top free, nothing within ~0.45 m of the table centre) and away
from tabletop.PARK, the floor strip behind the robot where unused objects are parked.

Coordinates are local to each env origin (same frame as tabletop.py: table top z=0 centred at x=0.2, floor z=-0.75).

Integration (see the module report / dev/test_room.py):
    cfg = tabletop.scene_cfg(n)                      # or any InteractiveSceneCfg
    for k, v in room.room_scene_entries(n, spacing=cfg.env_spacing).items():
        setattr(cfg, k, v)
    scene = InteractiveScene(cfg); ...; sim.reset()
    rr = room.RoomRandomizer(stage, n, np.random.default_rng(seed), sim=sim)   # compiles all variants once
    info = rr.randomize(env_ids)                     # at every episode reset -> {env_id: dict}
    for _ in range(8): sim.render()                  # see note below

Notes
- RTX's temporal denoiser keeps a faint ghost of the previous variant for a few frames after a switch (clearly
  visible on the window pictures, which are lit by a light 1 cm in front of them). Render ~8 frames after
  randomize() before recording (tabletop.reset settles physics with render=False, so this is not automatic).
- The room is closed: the HDRI dome and the distant sun no longer reach the table. The room's own lights are a
  ceiling RectLight (1500-5000, off in 20% of episodes) and per-window RectLights (400-2500, off in ~15% or at
  night); tabletop's per-env sphere "lamp" (800-9000) still dominates the table top.
- Window pictures are diffuse textures lit by a "view light" (emissive materials rendered far too dim / grey here).
- Assets: assets/fetch_room.py (props) + assets/room_textures.py (procedural finishes). The shell USD and prop
  wrappers are (re)generated into assets/room/ by room_scene_entries().
"""
import os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOM_DIR = os.path.join(HERE, "assets", "room")
TEX_DIR = os.path.join(ROOM_DIR, "tex")
FLOOR_Z = -0.75                     # tabletop ground plane
TABLE_C = (0.2, 0.0)                # table centre (local)
TABLE_HALF = (0.45, 0.55)
HEIGHT = float(os.environ.get("ROOM_HEIGHT", 2.5))
WALL_T = 0.06
# floor-level keep-outs (x0, x1, y0, y1): table + margin, tabletop.PARK strip (x=-1.0, y=-0.6..1.35) + margin
TABLE_KEEP = (TABLE_C[0] - TABLE_HALF[0] - 0.12, TABLE_C[0] + TABLE_HALF[0] + 0.12, -TABLE_HALF[1] - 0.12, TABLE_HALF[1] + 0.12)
PARK_KEEP = (-1.20, -0.80, -0.78, 1.52)
REACH_R = 0.45

# props: front-facing correction (deg about z so that the prop's front faces +x in the wrapper) and categories.
# cat: tall / low / seat (against a wall), chair / plant / small (floor, free or against a wall), wall (hung), clutter
# (on top of 'low' furniture), plus scale overrides. Values checked visually with dev/test_room.py --lineup.
FRONT = {}                           # filled from PROP_FIX below
PROP_FIX = {   # name: (front yaw deg, scale); the source assets face +x unless listed
    "shelf_set": (90.0, 0.85), "chair_h2": (90.0, 1.0), "desk": (90.0, 1.0),
}
SKIP = {"water_cooler"}                    # renders as a black blob
RECAT = {"lamp_floor": "clutter", "frame1": "clutter", "frame5": "clutter", "frame9": "clutter"}   # desk lamp / photo frames
WALL_IDS = ("front", "back", "left", "right")


def _load_props():
    meta = {k: m for k, m in json.load(open(os.path.join(ROOM_DIR, "props.json"))).items() if m["cat"] not in ("window", "ceiling") and k not in SKIP}
    for k, m in meta.items():
        yaw, sc = PROP_FIX.get(k, (0.0, 1.0))
        m["cat"] = RECAT.get(k, m["cat"])
        m["yaw_fix"], m["scale"] = yaw, sc
        lo, hi = np.array(m["lo"]), np.array(m["hi"])
        sx, sy, sz = (hi - lo) * sc
        if round(yaw / 90) % 2:
            sx, sy = sy, sx
        m["size"] = (float(sx), float(sy), float(sz))       # after the fix: x = depth (front +x), y = width, z = height
    return meta


PROPS = _load_props() if os.path.exists(os.path.join(ROOM_DIR, "props.json")) else {}


def room_bounds(spacing=3.0):
    """interior faces of the walls (x0, x1, y0, y1), centred on the table, fitting in env_spacing with a 5 cm gap"""
    h = spacing / 2 - 0.05
    return (TABLE_C[0] - h + WALL_T, TABLE_C[0] + h - WALL_T, -h + WALL_T, h - WALL_T)


# ====================================================================== USD authoring (pxr only, runs anywhere)
def _variants():
    import importlib.util
    spec = importlib.util.spec_from_file_location("room_textures", os.path.join(HERE, "assets", "room_textures.py"))
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    return {g: list(d) for g, d in rt.ALL.items()}


def _box(stage, path, lo, hi, tile=1.0, mat=None):
    """axis-aligned box mesh with world-scaled UVs (st = metres / tile) on every face"""
    from pxr import UsdGeom, Gf, Vt, Sdf, UsdShade
    lo, hi = np.array(lo, float), np.array(hi, float)
    m = UsdGeom.Mesh.Define(stage, path)
    pts, idx, cnt, st = [], [], [], []
    # (axis normal, sign): face quad corners in the two remaining axes
    for ax in range(3):
        for sgn in (0, 1):
            a, b = [i for i in range(3) if i != ax]
            c = lo.copy(); c[ax] = hi[ax] if sgn else lo[ax]
            quad = []
            for (ua, ub) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = c.copy(); p[a] = hi[a] if ua else lo[a]; p[b] = hi[b] if ub else lo[b]
                quad.append(p)
            if (sgn == 1) != ((ax == 1)):      # outward winding (counter-clockwise seen from outside)
                quad = quad[::-1]
            n0 = len(pts)
            for p in quad:
                pts.append(Gf.Vec3f(*p))
                u, v = (p[a], p[b]) if ax != 2 else (p[0], p[1])
                if ax == 0:
                    u, v = p[1], p[2]
                elif ax == 1:
                    u, v = p[0], p[2]
                st.append(Gf.Vec2f(u / tile, v / tile))
            idx += [n0, n0 + 1, n0 + 2, n0 + 3]; cnt.append(4)
    m.CreatePointsAttr(Vt.Vec3fArray(pts)); m.CreateFaceVertexIndicesAttr(idx); m.CreateFaceVertexCountsAttr(cnt)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    m.CreateSubdivisionSchemeAttr("none"); m.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray(st))
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


EMISSIVE_NITS = float(os.environ.get("ROOM_WINDOW_NITS", 1200.0))


def make_emissive(stage, path, tex=None, rgb=(1, 1, 1), intensity=EMISSIVE_NITS):
    """OmniPBR emissive material in the root layer (Kit command, as tabletop.make_table_materials).
    UsdPreviewSurface emissiveColor is far too dim in RTX, and an OmniPBR authored inside the referenced shell renders grey."""
    import omni.kit.commands
    from pxr import Sdf, Gf, UsdShade
    if not stage.GetPrimAtPath(path):
        omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=path)
    sh = stage.GetPrimAtPath(path + "/Shader")
    sh.CreateAttribute("inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.05, 0.05))
    sh.CreateAttribute("inputs:enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateAttribute("inputs:emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(intensity))
    sh.CreateAttribute("inputs:emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    if tex:
        sh.CreateAttribute("inputs:emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
    sh.CreateAttribute("inputs:reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.1)
    return UsdShade.Material(stage.GetPrimAtPath(path))


def _material(stage, path, tex=None, rgb=(0.8, 0.8, 0.8), rough=0.8, emissive=False, metal=0.0):
    from pxr import UsdShade, Sdf, Gf
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/PBR"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    if tex:
        rd = UsdShade.Shader.Define(stage, path + "/st"); rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex"); tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat"); tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        out = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        if emissive:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(out)
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(out)
    else:
        if emissive:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    return mat


VIEW_LIGHT = float(os.environ.get("ROOM_VIEW_LIGHT", 3000.0))
WIN_W, WIN_H, WIN_SILL = 1.1, 1.2, 0.9          # window size and sill height above the floor
DOOR_W, DOOR_H = 0.9, 2.05


def build_shell(path, spacing=3.0):
    """write the room shell USD (all finish variants, windows, door, rug, ceiling light) for a given env spacing"""
    from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
    V = _variants()
    x0, x1, y0, y1 = room_bounds(spacing)
    fz, cz = FLOOR_Z, FLOOR_Z + HEIGHT
    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Shell"); st.SetDefaultPrim(root.GetPrim())
    T = lambda n: f"./tex/{n}.png"
    look = lambda n, **kw: _material(st, f"/Shell/Looks/{n}", T(n), **kw)
    T_out = WALL_T
    for g, tile0 in (("floor", 1.2), ("wall", 1.5), ("ceiling", 1.2)):
        UsdGeom.Xform.Define(st, f"/Shell/{g}")
        for n in V[g]:
            tile = {"brick_red": 0.7, "brick_white": 0.7, "tiles_white": 1.0, "wallpaper_pattern": 1.0, "wallpaper_stripe": 1.0}.get(n, tile0)
            m = look(n, rough=0.45 if (g == "floor" and "wood" in n or "tiles" in n) else 0.85)
            gp = f"/Shell/{g}/{n}"; UsdGeom.Xform.Define(st, gp)
            if g == "floor":
                _box(st, gp + "/slab", (x0 - T_out, y0 - T_out, fz - 0.02), (x1 + T_out, y1 + T_out, fz + 0.0005), tile, m)
            elif g == "ceiling":
                _box(st, gp + "/slab", (x0 - T_out, y0 - T_out, cz), (x1 + T_out, y1 + T_out, cz + 0.03), tile, m)
            else:
                _box(st, gp + "/front", (x1, y0 - T_out, fz), (x1 + T_out, y1 + T_out, cz), tile, m)
                _box(st, gp + "/back", (x0 - T_out, y0 - T_out, fz), (x0, y1 + T_out, cz), tile, m)
                _box(st, gp + "/left", (x0, y1, fz), (x1, y1 + T_out, cz), tile, m)
                _box(st, gp + "/right", (x0, y0 - T_out, fz), (x1, y0, cz), tile, m)
    # baseboards (two colours)
    UsdGeom.Xform.Define(st, "/Shell/baseboard")
    for n, rgb in (("base_white", (0.9, 0.9, 0.88)), ("base_dark", (0.22, 0.18, 0.15))):
        m = _material(st, f"/Shell/Looks/{n}", rgb=rgb, rough=0.5); gp = f"/Shell/baseboard/{n}"; UsdGeom.Xform.Define(st, gp)
        h, d = 0.09, 0.015
        _box(st, gp + "/f", (x1 - d, y0, fz), (x1, y1, fz + h), 1, m); _box(st, gp + "/b", (x0, y0, fz), (x0 + d, y1, fz + h), 1, m)
        _box(st, gp + "/l", (x0, y1 - d, fz), (x1, y1, fz + h), 1, m); _box(st, gp + "/r", (x0, y0, fz), (x1, y0 + d, fz + h), 1, m)
    # windows: local frame x = into the room (wall face at x=0), y along the wall, z up from the floor
    frame_m = {n: _material(st, f"/Shell/Looks/{n}", rgb=rgb, rough=0.4) for n, rgb in (("frame_white", (0.92, 0.92, 0.9)), ("frame_dark", (0.12, 0.12, 0.13)), ("frame_wood", (0.45, 0.30, 0.17)))}
    sill_m = _material(st, "/Shell/Looks/sill", rgb=(0.85, 0.84, 0.82), rough=0.4)
    view_m = {n: _material(st, f"/Shell/Looks/{n}", T(n), rough=0.2) for n in V["view"]}   # rebound to emissive OmniPBR at init
    night_m = _material(st, "/Shell/Looks/view_night", rgb=(0.02, 0.025, 0.05), rough=0.1)
    for w in range(2):
        wp = f"/Shell/window_{w}"; xf = UsdGeom.Xform.Define(st, wp)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, fz)); xf.AddRotateZOp().Set(0.0)
        hw, b, t = WIN_W / 2, 0.05, 0.04
        UsdGeom.Xform.Define(st, f"{wp}/frame"); UsdGeom.Xform.Define(st, f"{wp}/view")
        for n, m in frame_m.items():
            fp = f"{wp}/frame/{n}"; UsdGeom.Xform.Define(st, fp)
            z0, z1 = WIN_SILL, WIN_SILL + WIN_H
            _box(st, fp + "/l", (0, -hw - b, z0 - b), (t, -hw, z1 + b), 1, m); _box(st, fp + "/r", (0, hw, z0 - b), (t, hw + b, z1 + b), 1, m)
            _box(st, fp + "/t", (0, -hw, z1), (t, hw, z1 + b), 1, m); _box(st, fp + "/b", (0, -hw, z0 - b), (t, hw, z0), 1, m)
            _box(st, fp + "/mv", (0, -0.02, z0), (t * 0.7, 0.02, z1), 1, m); _box(st, fp + "/mh", (0, -hw, z0 + 0.62 * WIN_H - 0.02), (t * 0.7, hw, z0 + 0.62 * WIN_H + 0.02), 1, m)
        _box(st, f"{wp}/sill", (0, -hw - 0.08, WIN_SILL - b - 0.03), (0.12, hw + 0.08, WIN_SILL - b), 1, sill_m)
        for n, m in list(view_m.items()) + [("view_night", night_m)]:
            vp = f"{wp}/view/{n}"
            _box(st, vp, (0.001, -hw, WIN_SILL), (0.003, hw, WIN_SILL + WIN_H), WIN_W, m)
            # UVs of the x-faces use (y, z): shift so that the picture spans the opening exactly
            mesh = UsdGeom.Mesh(st.GetPrimAtPath(vp)); pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            mesh.AddTranslateOp().Set(Gf.Vec3d(-0.035, 0, 0))     # unused pictures are pushed into the wall (see _show_view)
            pts = mesh.GetPointsAttr().Get()
            pv.Set([Gf.Vec2f((p[1] + hw) / WIN_W, (p[2] - WIN_SILL) / WIN_H) for p in pts])
        # "view light": a RectLight 1 cm in front of the outdoor picture, facing it (lit diffuse picture = bright window;
        # emissive materials render far too dim here). It sits inside the frame recess, so it barely lights the room.
        vl = UsdLux.RectLight.Define(st, f"{wp}/viewlight"); vl.CreateWidthAttr(WIN_H); vl.CreateHeightAttr(WIN_W); vl.CreateIntensityAttr(VIEW_LIGHT)
        vx = UsdGeom.Xformable(vl); vx.AddTranslateOp().Set(Gf.Vec3d(0.012, 0, WIN_SILL + WIN_H / 2)); vx.AddRotateYOp().Set(90.0)
        lt = UsdLux.RectLight.Define(st, f"{wp}/light")
        lt.CreateWidthAttr(WIN_W); lt.CreateHeightAttr(WIN_H); lt.CreateIntensityAttr(1500.0); lt.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))
        lt.GetWidthAttr().Set(WIN_H); lt.GetHeightAttr().Set(WIN_W)       # after rotateY(-90): local x is vertical, -z faces +x (into the room)
        lx = UsdGeom.Xformable(lt); lx.AddTranslateOp().Set(Gf.Vec3d(0.02, 0, WIN_SILL + WIN_H / 2)); lx.AddRotateYOp().Set(-90.0)
    # door (same local frame as windows)
    dp = "/Shell/door"; xf = UsdGeom.Xform.Define(st, dp); xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, fz)); xf.AddRotateZOp().Set(0.0)
    for n in ("door_wood", "door_white"):
        m = look(n, rough=0.5); gp = f"{dp}/{n}"; UsdGeom.Xform.Define(st, gp)
        _box(st, gp + "/leaf", (0, -DOOR_W / 2, 0.0), (0.03, DOOR_W / 2, DOOR_H), 1.0, m)
        _box(st, gp + "/jl", (0, -DOOR_W / 2 - 0.06, 0), (0.045, -DOOR_W / 2, DOOR_H + 0.06), 1.0, frame_m["frame_white" if n == "door_wood" else "frame_wood"])
        _box(st, gp + "/jr", (0, DOOR_W / 2, 0), (0.045, DOOR_W / 2 + 0.06, DOOR_H + 0.06), 1.0, frame_m["frame_white" if n == "door_wood" else "frame_wood"])
        _box(st, gp + "/jt", (0, -DOOR_W / 2, DOOR_H), (0.045, DOOR_W / 2, DOOR_H + 0.06), 1.0, frame_m["frame_white" if n == "door_wood" else "frame_wood"])
    hm = _material(st, "/Shell/Looks/handle", rgb=(0.75, 0.75, 0.75), rough=0.25, metal=1.0)
    _box(st, f"{dp}/handle", (0.03, DOOR_W / 2 - 0.16, 1.0), (0.09, DOOR_W / 2 - 0.06, 1.03), 1, hm)
    # rugs (on the floor, under/around the table: purely visual)
    rp = "/Shell/rug"; xf = UsdGeom.Xform.Define(st, rp); xf.AddTranslateOp().Set(Gf.Vec3d(TABLE_C[0], 0, fz)); xf.AddRotateZOp().Set(0.0)
    for n in [k for k in V["misc"] if k.startswith("rug")]:
        _box(st, f"{rp}/{n}", (-1.0, -0.7, 0.001), (1.0, 0.7, 0.008), 0.7, look(n, rough=0.95))
    # ceiling light panel + RectLight
    cp = "/Shell/ceiling_light"; xf = UsdGeom.Xform.Define(st, cp); xf.AddTranslateOp().Set(Gf.Vec3d(TABLE_C[0], 0, cz))
    _box(st, f"{cp}/panel", (-0.3, -0.3, -0.02), (0.3, 0.3, 0.0), 1, _material(st, "/Shell/Looks/panel_on", rgb=(0.95, 0.95, 0.95)))
    _box(st, f"{cp}/rim", (-0.32, -0.32, -0.025), (0.32, 0.32, -0.019), 1, _material(st, "/Shell/Looks/panel_rim", rgb=(0.85, 0.85, 0.85)))
    UsdGeom.Imageable(st.GetPrimAtPath(f"{cp}/rim")).MakeInvisible()
    pl = UsdLux.RectLight.Define(st, f"{cp}/panellight"); pl.CreateWidthAttr(0.58); pl.CreateHeightAttr(0.58); pl.CreateIntensityAttr(3000.0)
    px = UsdGeom.Xformable(pl); px.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.03)); px.AddRotateXOp().Set(180.0)     # faces up: lights the panel
    lt = UsdLux.RectLight.Define(st, f"{cp}/light"); lt.CreateWidthAttr(0.6); lt.CreateHeightAttr(0.6); lt.CreateIntensityAttr(4000.0)
    UsdGeom.Xformable(lt).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.03))
    st.GetRootLayer().Save()
    return path


def build_prop_wrapper(name, m, out_dir):
    """Z-up metre wrapper: origin at the bottom centre of the bbox, front towards +x (after yaw_fix), scale"""
    from pxr import Usd, UsdGeom, Gf
    p = os.path.join(out_dir, f"{name}.usda")
    st = Usd.Stage.CreateNew(p); UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root"); st.SetDefaultPrim(root.GetPrim())
    g = UsdGeom.Xform.Define(st, "/Root/geom")
    lo, hi = np.array(m["lo"]), np.array(m["hi"]); c = (lo + hi) / 2
    g.AddRotateZOp().Set(float(m["yaw_fix"])); g.AddScaleOp().Set(Gf.Vec3f(m["scale"], m["scale"], m["scale"]))
    g.AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -lo[2]))
    if m["up"] == "Y" or m["mpu"] != 1.0:
        raise NotImplementedError(f"{name}: up={m['up']} mpu={m['mpu']}")   # all current props are Z-up metres
    g.GetPrim().GetReferences().AddReference(os.path.relpath(os.path.join(ROOM_DIR, m["usd"]), out_dir))
    st.GetRootLayer().Save()
    return p


# ====================================================================== scene entries
def room_scene_entries(num_envs, spacing=3.0, props=None):
    """{attr name: AssetBaseCfg} to setattr on an InteractiveSceneCfg (one copy per env, prims {ENV_REGEX_NS}/room_*)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    gen = os.path.join(ROOM_DIR, "gen"); os.makedirs(gen, exist_ok=True)
    shell = os.path.join(ROOM_DIR, f"shell_{int(round(spacing * 100))}.usda")
    if os.path.exists(shell):
        os.remove(shell)
    build_shell(shell, spacing)
    out = {"room_shell": AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room_shell", spawn=sim_utils.UsdFileCfg(usd_path=shell))}
    nocoll = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    for k in (props or PROPS):
        w = os.path.join(gen, f"{k}.usda")
        if os.path.exists(w):
            os.remove(w)
        build_prop_wrapper(k, PROPS[k], gen)
        out[f"room_p_{k}"] = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room_p_" + k, spawn=sim_utils.UsdFileCfg(usd_path=w, collision_props=nocoll),
                                          init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_C[0], 0.0, FLOOR_Z - 5.0)))
    return out


# ====================================================================== runtime randomisation
def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


class RoomRandomizer:
    """Per-env room randomisation by visibility toggles and xform edits only (fast; no material rebinding)."""

    def __init__(self, stage, num_envs, rng, spacing=3.0, sim=None, props=None, warmup=True):
        from pxr import UsdGeom, UsdLux
        self.stage, self.n, self.rng = stage, num_envs, rng
        self.bounds = room_bounds(spacing)
        self.V = _variants()
        self.props = list(props or PROPS)
        self.root = lambda i: f"/World/envs/env_{i}"
        self._im = lambda p: UsdGeom.Imageable(stage.GetPrimAtPath(p))
        self.wins = ["frame_white", "frame_dark", "frame_wood"]
        self.views = self.V["view"] + ["view_night"]
        self.rugs = [k for k in self.V["misc"] if k.startswith("rug")]
        self.last = {}
        if warmup and sim is not None:   # every variant visible for a few frames so that all materials compile once
            from pxr import Usd
            for i in range(num_envs):
                for k in self.props:
                    self._set_prop(i, k, (TABLE_C[0], 0.0, FLOOR_Z + 0.5), 0.0, True)
                for p in Usd.PrimRange(stage.GetPrimAtPath(f"{self.root(i)}/room_shell")):
                    if p.IsA(UsdGeom.Imageable):
                        UsdGeom.Imageable(p).MakeVisible()
            for _ in range(4):
                sim.render()
        self.randomize(range(num_envs))

    # ---------------------------------------------------------------- helpers
    def _toggle(self, i, group, keep):
        base = f"{self.root(i)}/room_shell/{group}"
        prim = self.stage.GetPrimAtPath(base)
        for c in prim.GetChildren():
            (UsdGeom_Imageable(c).MakeVisible if c.GetName() == keep else UsdGeom_Imageable(c).MakeInvisible)()

    def _xf(self, path, pos=None, yaw=None):
        from pxr import UsdGeom, Gf
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(path)); ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
        if pos is not None:
            op = ops["xformOp:translate"]
            op.Set(Gf.Vec3d(*pos) if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f(*pos))
        if yaw is not None:
            if "xformOp:rotateZ" in ops:
                ops["xformOp:rotateZ"].Set(float(math.degrees(yaw)))
            else:
                op = ops["xformOp:orient"]; q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
                op.Set(Gf.Quatd(*q) if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf(*q))

    def _set_prop(self, i, k, pos, yaw, vis):
        p = f"{self.root(i)}/room_p_{k}"
        self._xf(p, pos, yaw)
        im = self._im(p); (im.MakeVisible if vis else im.MakeInvisible)()

    def _wall_frame(self, wall):
        """(origin of the interior face at u=0, u axis, inward normal, u range) of a wall"""
        x0, x1, y0, y1 = self.bounds
        return {"front": ((x1, 0.0), (0.0, 1.0), (-1.0, 0.0), (y0, y1)), "back": ((x0, 0.0), (0.0, -1.0), (1.0, 0.0), (-y1, -y0)),
                "left": ((0.0, y1), (-1.0, 0.0), (0.0, -1.0), (-x1, -x0)), "right": ((0.0, y0), (1.0, 0.0), (0.0, 1.0), (x0, x1))}[wall]

    def _on_wall(self, wall, u, depth):
        (ox, oy), (ux, uy), (nx, ny), _ = self._wall_frame(wall)
        return ox + ux * u + nx * depth, oy + uy * u + ny * depth, math.atan2(ny, nx)

    def _floor_ok(self, rect, used):
        x0, x1, y0, y1 = self.bounds
        if rect[0] < x0 - 0.05 or rect[1] > x1 + 0.05 or rect[2] < y0 - 0.05 or rect[3] > y1 + 0.05:
            return False
        if _overlap(rect, TABLE_KEEP) or _overlap(rect, PARK_KEEP):
            return False
        cx, cy = (rect[0] + rect[1]) / 2, (rect[2] + rect[3]) / 2
        if math.hypot(cx - TABLE_C[0], cy - TABLE_C[1]) < REACH_R + 0.3:
            return False
        return not any(_overlap(rect, r) for r in used)

    @staticmethod
    def _rect(x, y, yaw, sx, sy, pad=0.02):
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        hx, hy = (c * sx + s * sy) / 2 + pad, (s * sx + c * sy) / 2 + pad
        return (x - hx, x + hx, y - hy, y + hy)

    # ---------------------------------------------------------------- main
    def randomize(self, env_ids):
        out = {}
        for i in env_ids:
            out[int(i)] = self._randomize_env(int(i))
        self.last.update(out)
        return out

    def _randomize_env(self, i):
        from pxr import UsdLux
        r = self.rng
        d = {}
        pick = lambda xs: str(xs[int(r.integers(len(xs)))])
        for g in ("wall", "floor", "ceiling"):
            d[g] = pick(self.V[g]); self._toggle(i, g, d[g])
        d["baseboard"] = pick(["base_white", "base_dark"]); self._toggle(i, "baseboard", d["baseboard"])
        d["rug"] = pick(self.rugs + ["none", "none"])
        self._toggle(i, "rug", d["rug"])
        self._xf(f"{self.root(i)}/room_shell/rug", yaw=float(r.uniform(-0.2, 0.2)) + (math.pi / 2 if r.random() < 0.3 else 0.0))
        # wall occupancy: per wall, list of (u0, u1, z0, z1)
        occ = {w: [] for w in WALL_IDS}
        x0, x1, y0, y1 = self.bounds
        # the back wall behind the robot: only the corner beyond the parking strip is usable for furniture
        occ["back"].append((-PARK_KEEP[3], -PARK_KEEP[2], 0.0, 0.9))
        for w in ("left", "right"):   # parking strip reaches the side walls at x~-1.0: keep it free near the floor
            o, (ux, uy), _, _ = self._wall_frame(w)
            us = sorted([PARK_KEEP[0] * ux, PARK_KEEP[1] * ux]) if w in ("left", "right") else None
            if (w == "left" and PARK_KEEP[3] > y1 - 0.3) or (w == "right" and PARK_KEEP[2] < y0 + 0.3):
                occ[w].append((us[0], us[1], 0.0, 0.9))
        floor_used = []
        # --- windows (1-2) and door
        d["windows"] = []
        nwin = int(r.choice([1, 1, 2]))
        walls_w = [str(w) for w in r.permutation(["front", "left", "right"])]
        for w in range(2):
            wp = f"{self.root(i)}/room_shell/window_{w}"
            if w >= nwin:
                self._im(wp).MakeInvisible(); continue
            wall = walls_w[w]; _, _, _, (u0, u1) = self._wall_frame(wall)
            for _ in range(30):
                u = float(r.uniform(u0 + WIN_W / 2 + 0.15, u1 - WIN_W / 2 - 0.15))
                iv = (u - WIN_W / 2 - 0.1, u + WIN_W / 2 + 0.1, WIN_SILL - 0.1, WIN_SILL + WIN_H + 0.1)
                if not any(_overlap(iv, o) for o in occ[wall]):
                    break
            else:
                self._im(wp).MakeInvisible(); continue
            occ[wall].append(iv)
            x, y, yaw = self._on_wall(wall, u, 0.0)
            self._im(wp).MakeVisible(); self._xf(wp, (x, y, FLOOR_Z), yaw)
            fr, vw = pick(self.wins), pick(self.views)
            self._toggle_sub(f"{wp}/frame", fr); self._show_view(wp, vw)
            lt = UsdLux.RectLight(self.stage.GetPrimAtPath(f"{wp}/light"))
            on = vw != "view_night" and r.random() < 0.85
            inten = float(r.uniform(400, 2500)) if on else 0.0
            lt.GetIntensityAttr().Set(inten); self._im(f"{wp}/light").MakeVisible() if on else self._im(f"{wp}/light").MakeInvisible()
            vli = 0.0 if vw == "view_night" else VIEW_LIGHT * float(r.uniform(0.7, 1.3)) * (0.5 if vw == "view_dusk" else 1.0)
            UsdLux.RectLight(self.stage.GetPrimAtPath(f"{wp}/viewlight")).GetIntensityAttr().Set(vli)
            (self._im(f"{wp}/viewlight").MakeVisible if vli > 0 else self._im(f"{wp}/viewlight").MakeInvisible)()
            d["windows"].append(dict(wall=wall, u=round(u, 2), frame=fr, view=vw, light=round(inten)))
        # door: any wall, avoiding windows
        dp = f"{self.root(i)}/room_shell/door"
        d["door"] = None
        for _ in range(30):
            wall = pick(list(WALL_IDS)); _, _, _, (u0, u1) = self._wall_frame(wall)
            if u1 - u0 < DOOR_W + 0.4:
                continue
            u = float(r.uniform(u0 + DOOR_W / 2 + 0.1, u1 - DOOR_W / 2 - 0.1))
            iv = (u - DOOR_W / 2 - 0.08, u + DOOR_W / 2 + 0.08, 0.0, DOOR_H + 0.1)
            if any(_overlap(iv, o) for o in occ[wall]):
                continue
            occ[wall].append(iv)
            x, y, yaw = self._on_wall(wall, u, 0.0)
            self._xf(dp, (x, y, FLOOR_Z), yaw); dn = pick(["door_wood", "door_white"]); self._toggle_sub(dp, dn, keep_extra=("handle",))
            self._im(dp).MakeVisible(); d["door"] = dict(wall=wall, u=round(u, 2), kind=dn)
            ax, ay, _ = self._on_wall(wall, u, 0.0)
            floor_used.append(self._rect(*self._on_wall(wall, u, 0.5)[:2], 0.0, DOOR_W + 0.1, DOOR_W + 0.1))  # door swing area free
            break
        else:
            self._im(dp).MakeInvisible()
        # ceiling light
        cl = UsdLux.RectLight(self.stage.GetPrimAtPath(f"{self.root(i)}/room_shell/ceiling_light/light"))
        con = r.random() < 0.8
        ci = float(r.uniform(1500, 5000)) if con else 0.0
        cl.GetIntensityAttr().Set(ci); d["ceiling_light"] = round(ci)
        pl = UsdLux.RectLight(self.stage.GetPrimAtPath(f"{self.root(i)}/room_shell/ceiling_light/panellight"))
        pl.GetIntensityAttr().Set(3000.0 if con else 0.0)
        self._im(f"{self.root(i)}/room_shell/ceiling_light/panel").MakeVisible()
        # --- furniture
        placed = {}
        by = lambda *cats: [k for k in self.props if PROPS[k]["cat"] in cats]
        order = list(r.permutation(by("tall", "low", "seat")))[: int(r.integers(4, 9))]
        for k in order:
            sx, sy, sz = PROPS[k]["size"]
            for _ in range(60):
                wall = pick(list(WALL_IDS)); _, _, _, (u0, u1) = self._wall_frame(wall)
                if u1 - u0 < sy + 0.1:
                    continue
                u = float(r.uniform(u0 + sy / 2 + 0.03, u1 - sy / 2 - 0.03))
                iv = (u - sy / 2, u + sy / 2, 0.0, sz)
                if any(_overlap(iv, o) for o in occ[wall]):
                    continue
                x, y, yaw = self._on_wall(wall, u, sx / 2 + 0.01)
                rect = self._rect(x, y, yaw, sx, sy)
                if not self._floor_ok(rect, floor_used):
                    continue
                occ[wall].append(iv); floor_used.append(rect)
                placed[k] = (x, y, yaw, wall, u)
                break
        # free-standing floor items: chairs, plants, bins, boxes (random yaw, may also back onto a wall)
        free = list(r.permutation(by("chair", "plant", "small")))[: int(r.integers(3, 8))]
        for k in free:
            sx, sy, sz = PROPS[k]["size"]
            for _ in range(40):
                if r.random() < 0.5:   # against a wall
                    wall = pick(list(WALL_IDS)); _, _, _, (u0, u1) = self._wall_frame(wall)
                    u = float(r.uniform(u0 + sy / 2 + 0.03, u1 - sy / 2 - 0.03))
                    x, y, yaw = self._on_wall(wall, u, sx / 2 + 0.03); yaw += float(r.uniform(-0.3, 0.3))
                    iv = (u - sy / 2, u + sy / 2, 0.0, sz)
                    if any(_overlap(iv, o) for o in occ[wall] if o[2] < sz):
                        continue
                else:
                    wall = None
                    x, y = float(r.uniform(x0 + 0.2, x1 - 0.2)), float(r.uniform(y0 + 0.2, y1 - 0.2))
                    yaw = math.atan2(TABLE_C[1] - y, TABLE_C[0] - x) + float(r.uniform(-0.8, 0.8)) if PROPS[k]["cat"] == "chair" else float(r.uniform(-math.pi, math.pi))
                rect = self._rect(x, y, yaw, sx, sy)
                if not self._floor_ok(rect, floor_used):
                    continue
                if wall:
                    occ[wall].append(iv)
                floor_used.append(rect); placed[k] = (x, y, yaw, wall, None)
                break
        # wall decor (hung at 1.1-1.9 m above the floor where the wall is free)
        decor = list(r.permutation(by("wall")))[: int(r.integers(1, 5))]
        for k in decor:
            sx, sy, sz = PROPS[k]["size"]
            for _ in range(25):
                wall = pick(list(WALL_IDS)); _, _, _, (u0, u1) = self._wall_frame(wall)
                if u1 - u0 < sy + 0.2:
                    continue
                u = float(r.uniform(u0 + sy / 2 + 0.1, u1 - sy / 2 - 0.1)); zc = float(r.uniform(1.15, 1.75)) if sz < 1.0 else 1.5
                iv = (u - sy / 2 - 0.05, u + sy / 2 + 0.05, zc - sz / 2 - 0.05, zc + sz / 2 + 0.05)
                if iv[3] > HEIGHT - 0.05 or any(_overlap(iv, o) for o in occ[wall]):
                    continue
                occ[wall].append(iv)
                x, y, yaw = self._on_wall(wall, u, sx / 2 + 0.002)
                placed[k] = (x, y, yaw, wall, zc - sz / 2)
                break
        # clutter on top of low furniture
        tops = [k for k in placed if PROPS[k]["cat"] == "low"]
        top_used = {k: [] for k in tops}
        for k in list(r.permutation(by("clutter"))):
            if not tops or r.random() < 0.25:
                continue
            host = str(tops[int(r.integers(len(tops)))]); hx, hy, hyaw = placed[host][:3]
            hsx, hsy, hsz = PROPS[host]["size"]; sx, sy, sz = PROPS[k]["size"]
            if sy > hsy * 0.8 or sx > hsx:
                continue
            off = float(r.uniform(-(hsy - sy) / 2 * 0.8, (hsy - sy) / 2 * 0.8))
            if any(abs(off - o) < (sy + s2) / 2 + 0.02 for o, s2 in top_used[host]):
                continue
            top_used[host].append((off, sy))
            ux, uy = -math.sin(hyaw), math.cos(hyaw)
            placed[k] = (hx + ux * off, hy + uy * off, hyaw + float(r.uniform(-0.3, 0.3)), None, hsz)
        # write
        d["props"] = {}
        for k in self.props:
            if k in placed:
                x, y, yaw, wall, z = placed[k]
                zz = FLOOR_Z + (z if (PROPS[k]["cat"] in ("wall", "clutter")) else 0.0)
                self._set_prop(i, k, (x, y, zz), yaw, True)
                d["props"][k] = [round(x, 2), round(y, 2), round(math.degrees(yaw)), wall]
            else:
                self._set_prop(i, k, (TABLE_C[0], 0.0, FLOOR_Z - 5.0), 0.0, False)
        return d

    def _show_view(self, wp, vw):
        """visibility alone leaves a faint ghost of the previous picture under the 1 cm view light: also move the
        unused pictures 3.5 cm into the wall"""
        self._toggle_sub(f"{wp}/view", vw)
        for c in self.stage.GetPrimAtPath(f"{wp}/view").GetChildren():
            self._xf(str(c.GetPath()), (0.0 if c.GetName() == vw else -0.035, 0.0, 0.0))

    def _toggle_sub(self, base, keep, keep_extra=()):
        prim = self.stage.GetPrimAtPath(base)
        for c in prim.GetChildren():
            n = c.GetName()
            im = UsdGeom_Imageable(c)
            if n in keep_extra:
                im.MakeVisible(); continue
            if c.IsA(_Light()):
                continue
            (im.MakeVisible if n == keep else im.MakeInvisible)()


def UsdGeom_Imageable(prim):
    from pxr import UsdGeom
    return UsdGeom.Imageable(prim)


def _Light():
    from pxr import UsdLux
    return UsdLux.RectLight
