"""Download the YCB objects used by the tabletop scene from NVIDIA's Isaac Sim asset server and add physics.

For each object: <name>_phys.usda wraps the visual YCB asset (Y-up content rotated to Z-up where needed) with a rigid
body, mass and mesh collision, and objects.json records size, grasp point (mesh slice through the middle of the long
axis), grasp width / height and role. Run with any python that has usd-core + Pillow:
    python isaac_bench/assets/fetch_ycb.py
"""
import json, os, urllib.request
import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ycb")
SRC = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned"
# Graspable = fits a top-down SO-101 grasp (closing width <= ~4 cm: the moving jaw swings on an arc and lands on wider
# objects). Wider items are scaled to small real-world variants (last field = uniform scale) or kept as distractors.
# name, YCB id, description in instructions, up-axis of the source content, mass (kg), collision, role[, scale]
OBJ = [
    ("banana", "011_banana", "banana", "y", 0.066, "convexDecomposition", "grasp"),
    ("marker", "040_large_marker", "marker", "z", 0.016, "convexHull", "grasp"),
    ("foam_brick", "061_foam_brick", "foam brick", "z", 0.012, "convexHull", "grasp", 0.70),
    ("gelatin_box", "009_gelatin_box", "gelatin box", "y", 0.097, "convexHull", "grasp"),
    ("pudding_box", "008_pudding_box", "pudding box", "y", 0.120, "convexHull", "grasp"),
    ("meat_can", "010_potted_meat_can", "meat can", "y", 0.060, "convexHull", "grasp", 0.65),
    ("soup_can", "005_tomato_soup_can", "soup can", "y", 0.060, "convexHull", "grasp", 0.55),
    ("big_soup_can", "005_tomato_soup_can", "large soup can", "y", 0.150, "convexHull", "distractor"),
    ("bowl", "024_bowl", "bowl", "-y", 0.147, "convexDecomposition", "container"),
    ("mug", "025_mug", "mug", "y", 0.118, "convexDecomposition", "distractor"),
]


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)


def main():
    os.makedirs(os.path.join(HERE, "Materials/Textures"), exist_ok=True)
    meta = {}
    for name, ycb, desc, up, mass, approx, role, *sc in OBJ:
        scale = sc[0] if sc else 1.0
        fetch(f"{SRC}/{ycb}.usd", os.path.join(HERE, f"{ycb}.usd"))
        tex = os.path.join(HERE, f"Materials/Textures/{ycb}_COLOR.png")
        fetch(f"{SRC}/Materials/Textures/{ycb}_COLOR.png", tex)
        im = Image.open(tex)
        if im.size[0] > 1024:
            im.convert("RGB").resize((1024, 1024), Image.LANCZOS).save(tex)
        src = Usd.Stage.Open(os.path.join(HERE, f"{ycb}.usd"))
        mesh = [p for p in src.Traverse() if p.IsA(UsdGeom.Mesh)][0]
        rel_mesh = str(mesh.GetPath())[len("/Root"):]
        out = os.path.join(HERE, f"{name}_phys.usda")
        st = Usd.Stage.CreateNew(out); UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
        root = UsdGeom.Xform.Define(st, "/Root"); st.SetDefaultPrim(root.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()); UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
        geom = UsdGeom.Xform.Define(st, "/Root/geom")
        if up in ("y", "-y"):                                 # Y-up content -> Z-up ("-y": the asset's opening faces -Y)
            geom.AddRotateXOp().Set(90.0 if up == "y" else -90.0)
        if scale != 1.0:
            geom.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
        geom.GetPrim().GetReferences().AddReference(f"./{ycb}.usd")
        mp = st.OverridePrim("/Root/geom" + rel_mesh)
        UsdPhysics.CollisionAPI.Apply(mp); UsdPhysics.MeshCollisionAPI.Apply(mp).CreateApproximationAttr(approx)
        st.GetRootLayer().Save()
        # geometry metadata in the wrapper's (Z-up) frame
        st = Usd.Stage.Open(out)
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(st.GetDefaultPrim()).ComputeAlignedRange()
        mn, mx = bb.GetMin(), bb.GetMax(); size = [round(mx[i] - mn[i], 4) for i in range(3)]
        m = dict(usd=f"{name}_phys.usda", desc=desc, role=role, mass=mass, scale=scale, size=size, zmin=round(mn[2], 4), height=size[2])
        if role == "grasp":
            mesh = [p for p in st.Traverse() if p.IsA(UsdGeom.Mesh)][0]
            M = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            P = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get()); P = (np.c_[P, np.ones(len(P))] @ M)[:, :3]
            long_ax = 0 if size[0] >= size[1] else 1; close_ax = 1 - long_ax
            sl = P[np.abs(P[:, long_ax]) < 0.012]            # slice through the middle of the long axis
            lo, hi = sl[:, close_ax].min(), sl[:, close_ax].max()
            gp = np.zeros(2); gp[close_ax] = (lo + hi) / 2
            zt = sl[:, 2].max() - P[:, 2].min()
            # tall objects are gripped near the top: the wrist-camera mount sits ~3.5 cm above the jaw tips
            gh = zt - 0.025 if zt > 0.06 else max(0.012, min(zt / 2, zt - 0.012))
            m.update(close_axis="xy"[close_ax], grasp_xy=[round(float(v), 4) for v in gp], grasp_width=round(float(hi - lo), 4), grasp_h=round(float(gh), 4))
        meta[name] = m
        print(name.ljust(12), m["size"], m.get("grasp_width"))
    json.dump(meta, open(os.path.join(HERE, "objects.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
