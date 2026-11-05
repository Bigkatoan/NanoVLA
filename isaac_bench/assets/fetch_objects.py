"""Build the NanoVLA tabletop object library in isaac_bench/assets/objects/ (the old assets/ycb/ is left untouched).

Sources
  * all YCB "Axis_Aligned" items on NVIDIA's Isaac Sim 5.1 asset server (graspable ones possibly as a scaled small
    variant, the others as real-size obstacles / distractors),
  * other Isaac Sim props that look like household / office / lab items (mugs, blocks, Rubik's cube, beaker, food box,
    KLT bin, packing crates),
  * procedurally built toy-like items and containers (UsdGeom meshes + UsdPreviewSurface materials).
Each object becomes <name>.usda: /Root (rigid body + mass) / geom (up-axis fix, scale, centring on the bbox centre) ->
reference to the downloaded source (src/...) with mesh collision; nested rigid bodies of the source are removed.

objects.json (same schema as assets/ycb/objects.json, plus new fields)
  usd, desc (unique phrase for instructions), role (grasp | distractor | container), mass, scale, size [x,y,z], zmin,
  height, close_axis / grasp_xy / grasp_width / grasp_h (graspable only), and
  graspable (bool), category (short noun), colors (1-2 colour words computed from the textures / materials),
  aliases (list), source, opening [x,y] / floor_h / depth (containers only),
  rest_ok / rest_upright / rest_z / sim_grasp_width (filled in by isaac_bench/dev/test_objects.py; objects that fail
  the drop test are removed from objects.json and listed in drop_test.json).
Run with any python that has usd-core + Pillow + numpy (no Isaac needed):
    python3 isaac_bench/assets/fetch_objects.py
"""
import json, math, os, re, shutil, sys, urllib.error, urllib.parse, urllib.request
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "objects")
RAW = os.path.join(OUT, "src")                    # downloaded source assets, mirrored from Props/
SRC = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props"
CORE_MDL = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "OmniSurfaceLite.mdl", "OmniPBR_ClearCoat.mdl", "OmniEmissive.mdl", "OmniHair.mdl", "OmniSurfaceBlend.mdl"}
MAX_TEX = 1024
GRIP_MAX_W, GRIP_MAX_KG = 0.040, 0.150            # SO-101 top-down grasp limits (closing width, payload)

# ---------------------------------------------------------------------------------------------------------- specs
# kind "ref": (src path under Props/, up axis of the content, collision approximation)
# role is the intended role; a "grasp" object that violates the gripper limits is demoted to "distractor" (warned).
Y = "YCB/Axis_Aligned/"
SPECS = [
    # --- YCB, graspable (scaled small variants where the real one is too wide)
    dict(name="banana", src=Y + "011_banana.usd", up="y", mass=0.066, approx="convexDecomposition", role="grasp", category="banana", aliases=["fruit"]),
    dict(name="marker", src=Y + "040_large_marker.usd", up="z", mass=0.016, approx="convexHull", role="grasp", category="marker", aliases=["pen", "felt pen"]),
    dict(name="foam_brick", src=Y + "061_foam_brick.usd", up="z", mass=0.012, approx="convexHull", role="grasp", scale=0.70, category="foam brick", aliases=["foam block", "sponge brick"]),
    dict(name="gelatin_box", src=Y + "009_gelatin_box.usd", up="y", mass=0.097, approx="convexHull", role="grasp", category="gelatin box", aliases=["jello box", "small box"]),
    dict(name="pudding_box", src=Y + "008_pudding_box.usd", up="y", mass=0.120, approx="convexHull", role="grasp", category="pudding box", aliases=["chocolate box", "box"]),
    dict(name="meat_can", src=Y + "010_potted_meat_can.usd", up="y", mass=0.060, approx="convexHull", role="grasp", scale=0.65, category="meat can", aliases=["spam can", "tin"]),
    dict(name="soup_can", src=Y + "005_tomato_soup_can.usd", up="y", mass=0.060, approx="convexHull", role="grasp", scale=0.55, category="soup can", desc="small soup can", aliases=["small can", "tomato can"]),
    dict(name="scissors", src=Y + "037_scissors.usd", up="z", mass=0.082, approx="convexDecomposition", role="grasp", category="scissors", colors=["orange", "grey"], aliases=["pair of scissors"]),
    dict(name="large_clamp", src=Y + "051_large_clamp.usd", up="z", mass=0.125, approx="convexDecomposition", role="grasp", category="clamp", desc="clamp", aliases=["spring clamp"]),
    dict(name="tuna_can", src=Y + "007_tuna_fish_can.usd", up="y", mass=0.030, approx="convexHull", role="grasp", scale=0.45, category="tuna can", desc="small tuna can", aliases=["small tin", "can"]),
    dict(name="coffee_can", src=Y + "002_master_chef_can.usd", up="y", mass=0.050, approx="convexHull", role="grasp", scale=0.38, category="coffee can", desc="small coffee can", aliases=["small coffee tin", "can"]),
    # --- YCB, real size: obstacles / distractors
    dict(name="big_soup_can", src=Y + "005_tomato_soup_can.usd", up="y", mass=0.349, approx="convexHull", role="distractor", category="soup can", desc="large soup can", aliases=["tomato soup can"]),
    dict(name="master_chef_can", src=Y + "002_master_chef_can.usd", up="y", mass=0.414, approx="convexHull", role="distractor", category="coffee can", desc="large coffee can", aliases=["coffee tin"]),
    dict(name="big_tuna_can", src=Y + "007_tuna_fish_can.usd", up="y", mass=0.171, approx="convexHull", role="distractor", category="tuna can", desc="large tuna can", aliases=["tin"]),
    dict(name="cracker_box", src=Y + "003_cracker_box.usd", up="y", mass=0.411, approx="convexHull", role="distractor", category="cracker box", aliases=["cereal box", "big box"]),
    dict(name="sugar_box", src=Y + "004_sugar_box.usd", up="y", mass=0.514, approx="convexHull", role="distractor", category="sugar box", aliases=["box"]),
    dict(name="mustard", src=Y + "006_mustard_bottle.usd", up="-y", mass=0.603, approx="convexHull", role="distractor", category="mustard bottle", aliases=["mustard", "bottle"]),
    dict(name="pitcher", src=Y + "019_pitcher_base.usd", up="y", mass=0.178, approx="convexDecomposition", role="distractor", category="pitcher", aliases=["jug"]),
    dict(name="bleach", src=Y + "021_bleach_cleanser.usd", up="-y", mass=0.600, approx="convexHull", role="distractor", category="bleach bottle", aliases=["cleaner", "bottle"]),
    dict(name="drill", src=Y + "035_power_drill.usd", up="z", mass=0.895, approx="convexDecomposition", role="distractor", category="drill", aliases=["power drill"]),
    dict(name="wood_block", src=Y + "036_wood_block.usd", up="z", mass=0.729, approx="convexHull", role="distractor", category="wooden block", aliases=["wood block", "plank"]),
    dict(name="xl_clamp", src=Y + "052_extra_large_clamp.usd", up="z", mass=0.202, approx="convexDecomposition", role="distractor", category="clamp", desc="large clamp", aliases=["spring clamp"]),
    dict(name="mug", src=Y + "025_mug.usd", up="y", mass=0.118, approx="convexDecomposition", role="distractor", category="mug", aliases=["cup"]),
    dict(name="bowl", src=Y + "024_bowl.usd", up="-y", mass=0.147, approx="convexDecomposition", role="container", category="bowl", aliases=["dish"]),
    # --- other Isaac props
    dict(name="mug_a", src="Mugs/SM_Mug_A2.usd", up="z", mass=0.30, approx="convexDecomposition", role="distractor", category="mug", aliases=["cup", "coffee mug"]),
    dict(name="mug_b", src="Mugs/SM_Mug_B1.usd", up="z", mass=0.30, approx="convexDecomposition", role="distractor", category="mug", aliases=["cup", "coffee mug"]),
    dict(name="mug_c", src="Mugs/SM_Mug_C1.usd", up="z", mass=0.30, approx="convexDecomposition", role="distractor", category="mug", aliases=["cup", "coffee mug"]),
    dict(name="mug_d", src="Mugs/SM_Mug_D1.usd", up="z", mass=0.30, approx="convexDecomposition", role="distractor", category="mug", aliases=["cup", "coffee mug"]),
    dict(name="rubiks_cube", src="Rubiks_Cube/rubiks_cube.usd", up="z", mass=0.025, approx="convexHull", role="grasp", scale=0.5, category="rubik's cube", colors=["multicolored"], aliases=["puzzle cube", "cube"]),
    dict(name="dex_cube", src="Blocks/DexCube/dex_cube_instanceable.usd", up="z", mass=0.020, approx="convexHull", role="grasp", scale=0.6, category="letter cube", colors=["multicolored"], aliases=["dice", "cube"]),
    dict(name="multicolor_cube", src="Blocks/MultiColorCube/multi_color_cube_instanceable.usd", up="z", mass=0.020, approx="convexHull", role="grasp", scale=0.7, category="colorful cube", desc="multicolored cube", colors=["multicolored"], aliases=["cube"]),
    dict(name="red_block", src="Blocks/red_block.usd", up="z", mass=0.020, approx="convexHull", role="grasp", scale=0.7, category="block", colors=["red"], aliases=["cube", "brick"]),
    dict(name="green_block", src="Blocks/green_block.usd", up="z", mass=0.020, approx="convexHull", role="grasp", scale=0.7, category="block", aliases=["cube", "brick"]),
    dict(name="blue_block", src="Blocks/blue_block.usd", up="z", mass=0.020, approx="convexHull", role="grasp", scale=0.7, category="block", aliases=["cube", "brick"]),
    dict(name="yellow_block", src="Blocks/yellow_block.usd", up="z", mass=0.020, approx="convexHull", role="grasp", scale=0.7, category="block", aliases=["cube", "brick"]),
    dict(name="beaker", src="Beaker/beaker_500ml.usd", up="z", mass=0.20, approx="convexDecomposition", role="distractor", scale=0.6, category="beaker", colors=["white"], aliases=["measuring jug", "jug", "glass"]),
    dict(name="mac_cheese", src="Food/mac_n_cheese.usd", up="z", mass=0.206, approx="convexHull", role="distractor", category="mac and cheese box", colors=["white", "yellow"], aliases=["food box", "box"]),
    dict(name="klt_bin", src="KLT_Bin/small_KLT.usd", up="z", mass=0.25, approx=None, role="container", scale=0.5, category="bin", colors=["purple"], aliases=["storage bin", "crate", "box"]),
    dict(name="yellow_crate", src="PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01.usd", up="z", mass=0.15, approx="convexDecomposition", role="container", scale=0.25, category="crate", aliases=["basket", "tray"]),
    dict(name="blue_crate", src="PackingTable/props/SM_Crate_A08_Blue_01/SM_Crate_A08_Blue_01.usd", up="z", mass=0.15, approx="convexDecomposition", role="container", scale=0.25, category="crate", aliases=["basket", "tray"]),
]
# procedural items: list of parts (shape, params, sRGB colour); all parts are convex -> convexHull colliders
RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE = (0.82, 0.10, 0.10), (0.15, 0.62, 0.22), (0.12, 0.30, 0.80), (0.95, 0.80, 0.10), (0.95, 0.48, 0.08), (0.50, 0.22, 0.70)
WHITE, PINK, BROWN, BLACK, WOOD = (0.92, 0.92, 0.90), (0.95, 0.55, 0.70), (0.60, 0.42, 0.25), (0.08, 0.08, 0.08), (0.80, 0.64, 0.42)
PROC = [
    dict(name="red_peg", parts=[("cyl", dict(r=0.016, h=0.050), RED)], mass=0.030, role="grasp", category="cylinder", aliases=["peg", "red cylinder"]),
    dict(name="blue_peg", parts=[("cyl", dict(r=0.014, h=0.070), BLUE)], mass=0.030, role="grasp", category="cylinder", aliases=["peg", "tall cylinder"]),
    dict(name="green_hex", parts=[("prism", dict(r=0.019, n=6, h=0.030), GREEN)], mass=0.025, role="grasp", category="hexagon block", aliases=["hex block", "hexagon"]),
    dict(name="yellow_triangle", parts=[("prism", dict(r=0.022, n=3, h=0.030), YELLOW)], mass=0.020, role="grasp", category="triangle block", aliases=["triangle", "wedge"]),
    dict(name="orange_bar", parts=[("box", dict(s=(0.024, 0.090, 0.024)), ORANGE)], mass=0.030, role="grasp", category="bar", aliases=["stick", "long block"]),
    dict(name="purple_cube", parts=[("box", dict(s=(0.035, 0.035, 0.035)), PURPLE)], mass=0.025, role="grasp", category="cube", aliases=["block", "purple block"]),
    dict(name="bottle", parts=[("cyl", dict(r=0.018, h=0.072), BLUE), ("frustum", dict(r0=0.018, r1=0.010, z0=0.072, z1=0.084), BLUE),
                               ("cyl", dict(r=0.010, z0=0.084, h=0.014), WHITE), ("cyl", dict(r=0.0115, z0=0.098, h=0.012), WHITE)],
         mass=0.080, role="grasp", category="bottle", aliases=["small bottle", "water bottle"]),
    dict(name="glue_stick", parts=[("cyl", dict(r=0.011, h=0.065), WHITE), ("cyl", dict(r=0.0115, z0=0.065, h=0.025), ORANGE)],
         mass=0.040, role="grasp", category="glue stick", aliases=["glue"]),
    dict(name="sponge", parts=[("box", dict(s=(0.038, 0.072, 0.018)), YELLOW), ("box", dict(s=(0.038, 0.072, 0.007), c=(0, 0, 0.018)), GREEN)],
         mass=0.012, role="grasp", category="sponge", aliases=["kitchen sponge", "scrubber"]),
    dict(name="eraser", parts=[("box", dict(s=(0.022, 0.056, 0.018)), PINK)], mass=0.020, role="grasp", category="eraser", aliases=["rubber"]),
    dict(name="spool", parts=[("cyl", dict(r=0.017, h=0.005), WOOD), ("cyl", dict(r=0.013, z0=0.005, h=0.030), RED), ("cyl", dict(r=0.017, z0=0.035, h=0.005), WOOD)],
         mass=0.025, role="grasp", category="spool", aliases=["thread spool", "reel"]),
    dict(name="toy_brick", parts=[("box", dict(s=(0.032, 0.064, 0.0192)), GREEN)] +
         [("cyl", dict(r=0.0048, h=0.0018, z0=0.0192, c=(sx * 0.008, sy * 0.008)), GREEN) for sx in (-1, 1) for sy in (-3, -1, 1, 3)],
         mass=0.015, role="grasp", category="toy brick", aliases=["lego brick", "lego"]),
    # containers (open top); `opening` is the inner free size at the rim
    dict(name="cardboard_box", parts="openbox", box=dict(s=(0.14, 0.12, 0.06), t=0.005), color=BROWN, mass=0.12, role="container", category="box", aliases=["cardboard box", "open box"]),
    dict(name="blue_tray", parts="openbox", box=dict(s=(0.17, 0.12, 0.025), t=0.006), color=BLUE, mass=0.10, role="container", category="tray", aliases=["plastic tray"]),
    dict(name="plate", parts="round", round=dict(r=0.072, h=0.016, t=0.006, floor=0.006, n=24), color=WHITE, mass=0.12, role="container", category="plate", aliases=["dish", "white plate"]),
    dict(name="bucket", parts="round", round=dict(r=0.045, h=0.070, t=0.004, floor=0.005, n=20), color=RED, mass=0.08, role="container", category="bucket", aliases=["pot", "cup", "red bucket"]),
]


# ---------------------------------------------------------------------------------------------------------- download
def fetch(rel):
    path = os.path.join(RAW, rel)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            urllib.request.urlretrieve(f"{SRC}/{urllib.parse.quote(rel)}", path + ".part"); os.replace(path + ".part", path)
        except urllib.error.HTTPError as e:
            if not rel.endswith(".mdl"):
                print("  missing on server:", rel, e.code)
            return None
        if rel.lower().endswith((".png", ".jpg", ".jpeg", ".tga")):
            im = Image.open(path)
            if max(im.size) > MAX_TEX:
                fmt = im.format; im.thumbnail((MAX_TEX, MAX_TEX), Image.LANCZOS); im.save(path, format=fmt)
    return path


def fetch_tree(rel, seen=None):
    """download rel and (recursively) every relative layer / texture / MDL it references"""
    seen = set() if seen is None else seen
    rel = os.path.normpath(rel)
    if rel in seen:
        return
    seen.add(rel)
    path = fetch(rel)
    if path is None:
        return
    deps = []
    if rel.endswith((".usd", ".usda", ".usdc")):
        layers, assets, unresolved = UsdUtils.ExtractExternalReferences(path)
        deps = list(layers) + list(assets) + list(unresolved)
    elif rel.endswith(".mdl"):
        deps = re.findall(r'texture_2d\("([^"]+)"', open(path, errors="ignore").read())
    for d in deps:
        d = d.split("[")[0]                       # foo.usdz[inner.png] -> foo.usdz
        if not d or "://" in d or d.startswith(("omniverse:", "/")) or os.path.basename(d) in CORE_MDL or "<UDIM>" in d:
            continue
        fetch_tree(os.path.join(os.path.dirname(rel), d), seen)


# ---------------------------------------------------------------------------------------------------------- procedural meshes
def prism(poly0, z0, z1, poly1=None, smooth=False):
    """extruded polygon (CCW xy), optionally tapering to poly1 at z1. Returns points, counts, indices, face-varying normals."""
    poly1 = poly0 if poly1 is None else poly1
    n = len(poly0)
    pts = [(x, y, z0) for x, y in poly0] + [(x, y, z1) for x, y in poly1]
    counts, idx, nrm = [n, n], list(range(n - 1, -1, -1)) + list(range(n, 2 * n)), [(0, 0, -1)] * n + [(0, 0, 1)] * n
    for i in range(n):
        j = (i + 1) % n
        counts.append(4); idx += [i, j, n + j, n + i]
        if smooth:
            for k in (i, j, j, i):
                v = np.array(poly0[k]); v = v / (np.linalg.norm(v) + 1e-9); nrm.append((v[0], v[1], 0.0))
        else:
            a, b, c = np.array(pts[i]), np.array(pts[j]), np.array(pts[n + j])
            v = np.cross(b - a, c - a); v = v / (np.linalg.norm(v) + 1e-12); nrm += [tuple(v)] * 4
    return pts, counts, idx, nrm


def circle(r, n, phase=None, c=(0.0, 0.0)):
    phase = (math.pi / n if n <= 8 else 0.0) if phase is None else phase
    return [(c[0] + r * math.cos(phase + 2 * math.pi * k / n), c[1] + r * math.sin(phase + 2 * math.pi * k / n)) for k in range(n)]


def rect(sx, sy, c=(0.0, 0.0), yaw=0.0):
    cs, sn = math.cos(yaw), math.sin(yaw)
    return [(c[0] + cs * x - sn * y, c[1] + sn * x + cs * y) for x, y in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]


def part_mesh(shape, p):
    c = p.get("c", (0.0, 0.0, 0.0)); c = (c[0], c[1], c[2] if len(c) > 2 else 0.0)
    z0 = p.get("z0", c[2])
    if shape == "box":
        sx, sy, sz = p["s"]; return prism(rect(sx, sy, c[:2], p.get("yaw", 0.0)), z0, z0 + sz)
    if shape == "cyl":
        return prism(circle(p["r"], 32, c=c[:2]), z0, z0 + p["h"], smooth=True)
    if shape == "prism":
        return prism(circle(p["r"], p["n"], c=c[:2]), z0, z0 + p["h"])
    if shape == "frustum":
        return prism(circle(p["r0"], 32), p["z0"], p["z1"], circle(p["r1"], 32), smooth=True)
    raise ValueError(shape)


def container_parts(spec):
    col = spec["color"]
    if spec["parts"] == "openbox":
        (sx, sy, sz), t = spec["box"]["s"], spec["box"]["t"]
        return [("box", dict(s=(sx, sy, t)), col),
                ("box", dict(s=(t, sy, sz - t), c=(-(sx - t) / 2, 0, t)), col), ("box", dict(s=(t, sy, sz - t), c=((sx - t) / 2, 0, t)), col),
                ("box", dict(s=(sx - 2 * t, t, sz - t), c=(0, -(sy - t) / 2, t)), col), ("box", dict(s=(sx - 2 * t, t, sz - t), c=(0, (sy - t) / 2, t)), col)]
    q = spec["round"]; r, h, t, fl, n = q["r"], q["h"], q["t"], q["floor"], q["n"]
    parts = [("cyl", dict(r=r - t / 2, h=fl), col)]
    seg = 2 * (r - t / 2) * math.tan(math.pi / n) + 0.0015         # wall segments overlap slightly
    for k in range(n):
        a = 2 * math.pi * k / n; cc = ((r - t / 2) * math.cos(a), (r - t / 2) * math.sin(a), fl)
        parts.append(("box", dict(s=(t, seg, h - fl), c=cc, yaw=a), col))
    return parts


def srgb2lin(c):
    return tuple(float(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in c)


def build_proc(spec, out):
    st = Usd.Stage.CreateNew(out); UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root"); st.SetDefaultPrim(root.GetPrim())
    geom = UsdGeom.Xform.Define(st, "/Root/geom")
    parts = container_parts(spec) if isinstance(spec["parts"], str) else spec["parts"]
    mats = {}
    for k, (shape, p, col) in enumerate(parts):
        if col not in mats:
            mp = f"/Root/Looks/mat{len(mats)}"
            m = UsdShade.Material.Define(st, mp); sh = UsdShade.Shader.Define(st, mp + "/Shader"); sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*srgb2lin(col)))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55); sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); mats[col] = m
        pts, counts, idx, nrm = part_mesh(shape, p)
        mesh = UsdGeom.Mesh.Define(st, f"/Root/geom/part{k}")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in pts])); mesh.CreateFaceVertexCountsAttr(counts); mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in nrm])); mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none); mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[col])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return st, geom


# ---------------------------------------------------------------------------------------------------------- wrappers
def build_ref(spec, out):
    src_path = os.path.join(RAW, spec["src"])
    unit = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(src_path))
    st = Usd.Stage.CreateNew(out); UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Root"); st.SetDefaultPrim(root.GetPrim())
    geom = UsdGeom.Xform.Define(st, "/Root/geom")
    src = UsdGeom.Xform.Define(st, "/Root/geom/src")      # the source keeps its own xformOps on this child
    src.GetPrim().GetReferences().AddReference("./" + os.path.relpath(src_path, OUT))
    # un-instance (instance proxies cannot be edited), then strip the source's own rigid bodies / masses
    for _ in range(4):
        inst = [p for p in Usd.PrimRange(geom.GetPrim(), Usd.TraverseInstanceProxies()) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            st.OverridePrim(p.GetPath()).SetInstanceable(False)
    for p in Usd.PrimRange(geom.GetPrim()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            st.OverridePrim(p.GetPath()).RemoveAPI(UsdPhysics.RigidBodyAPI)
        if p.HasAPI(UsdPhysics.MassAPI):
            st.OverridePrim(p.GetPath()).RemoveAPI(UsdPhysics.MassAPI)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            st.OverridePrim(p.GetPath()).RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if spec["approx"] and p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputePurpose() in ("default", "render"):
            o = st.OverridePrim(p.GetPath())
            UsdPhysics.CollisionAPI.Apply(o); UsdPhysics.MeshCollisionAPI.Apply(o).CreateApproximationAttr(spec["approx"])
            if spec["approx"] == "convexDecomposition":
                o.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
                big = spec["role"] == "container"
                o.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(64 if big else 32)
                o.CreateAttribute("physxConvexDecompositionCollision:voxelResolution", Sdf.ValueTypeNames.Int).Set(800000 if big else 300000)
                o.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(True)
    up, s = spec.get("up", "z"), spec.get("scale", 1.0) * unit
    ops = []
    if up in ("y", "-y"):                              # Y-up content -> Z-up ("-y": the asset's opening faces -Y)
        ops.append(("r", 90.0 if up == "y" else -90.0))
    if s != 1.0:
        ops.append(("s", s))
    return st, geom, ops


def finish(st, geom, ops, mass):
    """add physics on /Root, centre the geometry on its bbox centre"""
    root = st.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root); UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    t = geom.AddTranslateOp(); order = [t]
    for kind, v in ops:
        order.append(geom.AddRotateXOp() if kind == "r" else geom.AddScaleOp())
        order[-1].Set(v if kind == "r" else Gf.Vec3f(v, v, v))
    geom.SetXformOpOrder(order)
    t.Set(Gf.Vec3d(0, 0, 0))
    bb = bbox(st)
    c = (bb.GetMin() + bb.GetMax()) / 2
    t.Set(Gf.Vec3d(-c[0], -c[1], -c[2]))
    st.GetRootLayer().Save()


def bbox(st):
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(st.GetDefaultPrim()).ComputeAlignedRange()


# ---------------------------------------------------------------------------------------------------------- materials -> colours
_IMG = {}


def _img(path):
    if path not in _IMG:
        im = None
        if path and "[" in path:                                   # texture inside a .usdz package: pkg.usdz[inner.png]
            import io, zipfile
            pkg, inner = path[:-1].split("[", 1)
            if os.path.exists(pkg):
                with zipfile.ZipFile(pkg) as z:
                    im = Image.open(io.BytesIO(z.read(inner)))
        elif path and os.path.exists(path):
            im = Image.open(path)
        _IMG[path] = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 if im is not None else None
    return _IMG[path]


def srgb2lin_np(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _asset(attr_val, base_dir=None):
    if attr_val is None:
        return None
    p = attr_val.resolvedPath if hasattr(attr_val, "resolvedPath") else None
    if p:
        return p
    raw = attr_val.path if hasattr(attr_val, "path") else str(attr_val)
    if raw and base_dir:
        cand = os.path.normpath(os.path.join(base_dir, raw))
        return cand if os.path.exists(cand) else None
    return None


def material_source(mat):
    """-> (texture path or None, constant sRGB colour, tint sRGB, glass?) for the surface of a material"""
    tex, const, tint, glass = None, None, (1.0, 1.0, 1.0), False
    if not mat:
        return None, (0.6, 0.6, 0.6), tint, False
    for p in Usd.PrimRange(mat.GetPrim()):
        if not p.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(p); sid = sh.GetIdAttr().Get()
        sa = p.GetAttribute("info:mdl:sourceAsset")
        if sid == "UsdPreviewSurface" and not (sa and sa.Get() is not None):
            inp = sh.GetInput("diffuseColor")
            if inp:
                if inp.HasConnectedSource():
                    src = inp.GetConnectedSource()[0].GetPrim()
                    f = UsdShade.Shader(src).GetInput("file")
                    if f:
                        tex = _asset(f.Get(), os.path.dirname(src.GetPrimStack()[0].layer.realPath))
                elif inp.Get() is not None:
                    const = tuple(lin2srgb(inp.Get()))
            if sh.GetInput("opacity") and (sh.GetInput("opacity").Get() or 1.0) < 0.5:
                glass = True
            continue
        if sa and sa.Get() is not None:
            def val(nm):                                           # value of an input, following interface connections
                inp = sh.GetInput(nm.split(":", 1)[1])
                if not inp:
                    return None, None
                attrs = inp.GetValueProducingAttributes() or [inp.GetAttr()]
                a = attrs[0]
                return a.Get(), (os.path.dirname(a.GetPropertyStack()[0].layer.realPath) if a.GetPropertyStack() else None)
            name = os.path.basename(sa.Get().path)
            layer_dir = os.path.dirname(p.GetAttribute("info:mdl:sourceAsset").GetPropertyStack()[0].layer.realPath)
            if "Glass" in name:
                glass = True
            v, base = val("inputs:diffuse_texture")
            if v is not None and v.path:
                tex = _asset(v, base)
            for nm in ("inputs:diffuse_color_constant", "inputs:diffuseColor"):
                v, _ = val(nm)
                if v is not None:
                    const = tuple(lin2srgb(v))
            v, _ = val("inputs:diffuse_tint")
            if v is not None:
                tint = tuple(float(c) for c in v)                  # linear multiplier
            if name not in CORE_MDL and tex is None:          # custom .mdl file (e.g. packing crates): parse it
                mdl = _asset(sa.Get(), layer_dir)
                if mdl:
                    txt = open(mdl, errors="ignore").read()
                    m = re.search(r'diffuse_texture:\s*texture_2d\("([^"]+)"', txt)
                    if m:
                        tex = _asset(m.group(1), os.path.dirname(mdl))
                    m = re.search(r'diffuse_tint:\s*color\(([^)]*)\)', txt)
                    if m:
                        tint = tuple(float(v) for v in m.group(1).split(","))
    if tex is None and const is None:
        const = (0.6, 0.6, 0.6)
    return tex, const, tint, glass


def surface_samples(st, n=40000, seed=0):
    """area-uniform samples on all render meshes: points (root frame) and sRGB colour, plus a glass flag per sample"""
    rng = np.random.default_rng(seed)
    tris, uvs, srcs = [], [], []
    for p in Usd.PrimRange(st.GetDefaultPrim()):
        if not p.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(p).ComputePurpose() not in ("default", "render") or UsdGeom.Imageable(p).ComputeVisibility() == "invisible":
            continue
        mesh = UsdGeom.Mesh(p)
        P = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        if len(P) == 0:
            continue
        M = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        P = (np.c_[P, np.ones(len(P))] @ M)[:, :3]
        cnt = np.array(mesh.GetFaceVertexCountsAttr().Get()); ind = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        st_pv = None
        for pv in UsdGeom.PrimvarsAPI(p).GetPrimvars():
            if pv.GetTypeName() in (Sdf.ValueTypeNames.TexCoord2fArray, Sdf.ValueTypeNames.Float2Array) and pv.GetPrimvarName() in ("st", "st0", "UVMap", "uv", "map1", "st_0"):
                st_pv = pv; break
        uv, interp = (np.array(st_pv.ComputeFlattened(), dtype=np.float64), st_pv.GetInterpolation()) if st_pv else (None, None)
        # per-face material (GeomSubsets) -> per-face source
        bapi = UsdShade.MaterialBindingAPI(p)
        face_src = [material_source(bapi.ComputeBoundMaterial()[0])] * len(cnt)
        for sub in bapi.GetMaterialBindSubsets():
            m = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]
            s_ = material_source(m)
            for f in sub.GetIndicesAttr().Get() or []:
                if f < len(face_src):
                    face_src[f] = s_
        starts = np.r_[0, np.cumsum(cnt)[:-1]]
        for f, (c, s0) in enumerate(zip(cnt, starts)):
            for k in range(1, c - 1):
                corners = (s0, s0 + k, s0 + k + 1)
                tris.append(P[ind[list(corners)]])
                if uv is not None and interp == "faceVarying" and len(uv) >= len(ind):
                    uvs.append(uv[list(corners)])
                elif uv is not None and interp in ("vertex", "varying") and len(uv) >= len(P):
                    uvs.append(uv[ind[list(corners)]])
                else:
                    uvs.append(np.zeros((3, 2)))
                srcs.append(face_src[f])
    T = np.array(tris); UV = np.array(uvs)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    k = rng.choice(len(T), n, p=area / area.sum())
    a, b = rng.random(n), rng.random(n); flip = a + b > 1; a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    w = np.c_[1 - a - b, a, b]
    pts = np.einsum("ni,nij->nj", w, T[k]); uvp = np.einsum("ni,nij->nj", w, UV[k])
    rgb = np.zeros((n, 3), np.float32); glass = np.zeros(n, bool)
    src_ids = np.array([id(s) for s in srcs])[k]
    for s_ in {id(s): s for s in srcs}.values():
        sel = src_ids == id(s_)
        tex, const, tint, gl = s_
        im = _img(tex) if tex else None
        if im is not None:
            u, v = np.mod(np.nan_to_num(uvp[sel, 0]), 1.0), np.mod(np.nan_to_num(uvp[sel, 1]), 1.0)
            H, W = im.shape[:2]
            rgb[sel] = lin2srgb(srgb2lin_np(im[((1 - v) * (H - 1)).astype(int), (u * (W - 1)).astype(int)]) * np.array(tint, np.float32))
        else:
            rgb[sel] = lin2srgb(srgb2lin_np(np.array(const, np.float32)) * np.array(tint, np.float32))
        glass[sel] = gl
    return pts, rgb, glass


def color_name(rgb):
    """vectorised sRGB -> basic colour word"""
    rgb = np.asarray(rgb, np.float32)
    mx, mn = rgb.max(1), rgb.min(1); v = mx; s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb.T; d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    out = np.empty(len(rgb), dtype=object)
    for i in range(len(rgb)):
        H, S, V = h[i], s[i], v[i]
        if V < 0.18:
            out[i] = "black"
        elif S < 0.18 or (S < 0.28 and V < 0.45):
            out[i] = "white" if V > 0.72 else ("grey" if V > 0.30 else "black")
        elif 8 <= H < 45 and (V < 0.62 or S < 0.55) and not (V > 0.8 and S > 0.6):
            out[i] = "brown"
        elif H < 12 or H >= 340:
            out[i] = "red"
        elif H < 40:
            out[i] = "orange"
        elif H < 68:
            out[i] = "yellow"
        elif H < 165:
            out[i] = "green"
        elif H < 255:
            out[i] = "blue"
        elif H < 295:
            out[i] = "purple"
        else:
            out[i] = "pink"
    return out


def dominant_colors(rgb, glass):
    if glass.mean() > 0.5:
        return ["clear"]
    names, counts = np.unique(color_name(rgb[~glass]), return_counts=True)
    share = counts / counts.sum()
    # chromatic colours are what people name first ("the red soup can"), neutrals (metal lids, labels) count less
    score = np.where(np.isin(names, ["white", "grey", "black"]), 1.0, 1.6) * share
    order = np.argsort(-score)
    cols = [names[order[0]]]
    if len(order) > 1 and share[order[1]] >= 0.20:
        cols.append(names[order[1]])
    return [str(c) for c in cols]


# ---------------------------------------------------------------------------------------------------------- geometry metadata
def grasp_meta(P, size, zmin):
    """search a top-down grasp: slabs (jaw width 2.4 cm) across the long axis, offset up to 30 % of the length from the
    centre; width = extent along the closing axis of everything the jaws span (grasp height - 1 cm ... + 3.5 cm)."""
    long_ax = 0 if size[0] >= size[1] else 1; close_ax = 1 - long_ax
    best = None
    for t in np.linspace(-0.3, 0.3, 25) * size[long_ax]:
        sl = P[np.abs(P[:, long_ax] - t) < 0.012]
        if len(sl) < 50:
            continue
        zt = sl[:, 2].max() - zmin
        # tall objects are gripped near the top: the wrist-camera mount sits ~3.5 cm above the jaw tips
        gh = zt - 0.025 if zt > 0.06 else max(0.012, min(zt / 2, zt - 0.012))
        band = sl[(sl[:, 2] - zmin > gh - 0.010) & (sl[:, 2] - zmin < gh + 0.035)]
        if len(band) < 20:
            continue
        lo, hi = band[:, close_ax].min(), band[:, close_ax].max(); w = hi - lo
        score = w + 0.15 * abs(t)
        if best is None or score < best[0]:
            gp = np.zeros(2); gp[long_ax] = t; gp[close_ax] = (lo + hi) / 2
            best = (score, dict(close_axis="xy"[close_ax], grasp_xy=[round(float(v), 4) for v in gp], grasp_width=round(float(w), 4), grasp_h=round(float(gh), 4)))
    return best[1] if best else None


def opening_meta(P, size, zmin):
    """inner free size at the rim along x / y through the centre, inner floor height, depth"""
    top = zmin + size[2]
    out = []
    for ax in (0, 1):
        o = 1 - ax
        line = P[(np.abs(P[:, o]) < 0.006) & (P[:, 2] > top - 0.6 * size[2])]
        neg, pos = line[line[:, ax] < 0, ax], line[line[:, ax] > 0, ax]
        out.append(float(pos.min() - neg.max()) if len(neg) and len(pos) else 0.0)
    col = P[(np.abs(P[:, 0]) < 0.01) & (np.abs(P[:, 1]) < 0.01)]
    floor = float(col[:, 2].max() - zmin) if len(col) else 0.0
    return dict(opening=[round(out[0], 4), round(out[1], 4)], floor_h=round(floor, 4), depth=round(size[2] - floor, 4))


# ---------------------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(RAW, exist_ok=True)
    only = set(sys.argv[1:])
    meta = {}
    for spec in SPECS + PROC:
        name = spec["name"]
        if only and name not in only:
            continue
        out = os.path.join(OUT, f"{name}.usda")
        if os.path.exists(out):
            os.remove(out)
        if "src" in spec:
            fetch_tree(spec["src"])
            st, geom, ops = build_ref(spec, out)
        else:
            st, geom = build_proc(spec, out); ops = []
        finish(st, geom, ops, spec["mass"])
        st = Usd.Stage.Open(out)
        bb = bbox(st); mn, mx = bb.GetMin(), bb.GetMax(); size = [round(mx[i] - mn[i], 4) for i in range(3)]
        P, rgb, glass = surface_samples(st)
        m = dict(usd=f"{name}.usda", desc=spec.get("desc"), role=spec["role"], mass=spec["mass"], scale=spec.get("scale", 1.0), size=size,
                 zmin=round(mn[2], 4), height=size[2], category=spec["category"], colors=spec.get("colors") or dominant_colors(rgb, glass),
                 aliases=list(spec.get("aliases", [])), source=("isaac:" + spec["src"]) if "src" in spec else "procedural")
        if spec["role"] == "container":
            m.update(opening_meta(P, size, mn[2]))
        else:
            g = grasp_meta(P, size, mn[2])
            ok = spec["role"] == "grasp" and g is not None and g["grasp_width"] <= GRIP_MAX_W and spec["mass"] <= GRIP_MAX_KG
            if spec["role"] == "grasp" and not ok:
                print(f"  !! {name}: not graspable (width {g and g['grasp_width']}, mass {spec['mass']}) -> distractor")
                m["role"] = "distractor"
            if ok:
                m.update(g)
        m["graspable"] = m["role"] == "grasp"
        meta[name] = m
        print(f"{name:16s} {m['role']:10s} size {size} colors {m['colors']} grasp_w {m.get('grasp_width')} opening {m.get('opening')}")
    if only:                                           # partial rebuild: merge into the existing library
        old = json.load(open(os.path.join(OUT, "objects.json"))) if os.path.exists(os.path.join(OUT, "objects.json")) else {}
        old.update(meta); meta = old
    # unique instruction phrases: "<colour> <category>" when a category occurs more than once
    cats = {}
    for k, m in meta.items():
        cats.setdefault(m["category"], []).append(k)
    for k, m in meta.items():
        if not m.get("desc"):
            m["desc"] = m["category"] if len(cats[m["category"]]) == 1 else f"{m['colors'][0]} {m['category']}"
    descs = {}
    for k, m in meta.items():
        descs.setdefault(m["desc"], []).append(k)
    for d, ks in descs.items():
        if len(ks) > 1:
            print(f"  !! duplicate desc '{d}': {ks}")
    # drop-test results (isaac_bench/dev/test_objects.py) if present: merge and drop failures
    dt = os.path.join(OUT, "drop_test.json")
    if os.path.exists(dt):
        res = json.load(open(dt))["results"]
        for k in list(meta):
            if k in res:
                meta[k].update({f: res[k][f] for f in ("rest_ok", "rest_upright", "rest_z", "sim_grasp_width") if f in res[k]})
                if not res[k]["rest_ok"]:
                    print("  dropped (drop test):", k, res[k].get("why")); del meta[k]
    json.dump(meta, open(os.path.join(OUT, "objects.json"), "w"), indent=1)
    print(len(meta), "objects ->", os.path.join(OUT, "objects.json"))


if __name__ == "__main__":
    main()
