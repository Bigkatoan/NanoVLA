"""Download the room furniture / clutter used by isaac_bench/room.py from NVIDIA's public Isaac Sim 5.1 asset server.

Each prop USD is mirrored with its dependencies (referenced layers, MDL materials + their imported MDL modules,
textures) into assets/room/<server path>, so relative asset paths keep working. Textures are downscaled to at most
TEX_MAX px (the props are far from the wrist camera; keeps VRAM and disk small). Then props.json records, per prop,
the local path, its up axis / metersPerUnit and the world-space bounding box (in metres, Z-up) used for placement.
Run with any python that has usd-core + Pillow (idempotent; re-run skips existing files):
    /usr/bin/python3 isaac_bench/assets/fetch_room.py
"""
import json, os, re, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pxr import Gf, Sdf, Usd, UsdGeom

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "room")
BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/"
TEX_MAX = int(os.environ.get("ROOM_TEX_MAX", 512))
O, H = "Environments/Office/Props/", "Environments/Hospital/Props/"
# name -> (server path, category). Categories drive room.py: floor furniture along walls, wall decor, shelf clutter, ceiling.
PROPS = {
    # tall furniture against the walls
    "bookcase": (O + "SM_BookcaseA.usd", "tall"), "cupboard": (O + "SM_Cupboard.usd", "tall"),
    "file_cab1": (O + "SM_FileCabinet_01.usd", "low"), "file_cab2": (O + "SM_FileCabinet_02.usd", "low"),
    "file_cab3": (O + "SM_FileCabinet_03.usd", "tall"), "rack": (O + "SM_Rack1m.usd", "tall"),
    "water_cooler": (O + "SM_WaterCooler.usd", "tall"), "med_cabinet": (H + "SM_MedicalCabinet_01a.usd", "tall"),
    "shelf_set": (H + "SM_ShelfSet_01a.usd", "tall"), "supply_cart": (H + "SM_SupplyCart_01e.usd", "low"),
    # seating / tables / misc on the floor
    "sofa": (O + "SM_Sofa.usd", "seat"), "armchair": (O + "SM_Armchair.usd", "seat"), "recliner": (O + "SM_Recliner.usd", "seat"),
    "chair_office": (O + "SM_ChairOffice.usd", "chair"), "chair": (O + "SM_Chair.usd", "chair"),
    "chair_h1": (H + "SM_Chair_01a.usd", "chair"), "chair_h2": (H + "SM_Chair_02a.usd", "chair"),
    "table_coffee": (O + "SM_TableCoffee.usd", "low"), "desk": (H + "SM_Desk_04a.usd", "low"),
    "plant1": (O + "SM_Plant01.usd", "plant"), "plant2": (O + "SM_Plant02.usd", "plant"), "plant3": (O + "SM_Plant03.usd", "plant"),
    "trash1": (O + "SM_TrashCan.usd", "small"), "trash2": (O + "SM_TrashCan_01.usd", "small"), "trash3": (H + "SM_TrashCan.usd", "small"),
    "printer": (O + "SM_Printer.usd", "small"), "box_big": (O + "SM_BoxBig.usd", "small"), "box_open": (O + "SM_BoxBigOpen.usd", "small"),
    "box_a": (O + "SM_BoxPortableA.usd", "small"), "box_c": (O + "SM_BoxPortableC.usd", "small"), "lamp_floor": (O + "SM_Lamp.usd", "plant"),
    # wall decor
    "picture1": (O + "SM_PictureCollage_A.usd", "wall"), "picture2": (O + "SM_PictureCollage_B.usd", "wall"),
    "picture3": (O + "SM_PictureCollage_C.usd", "wall"), "frame1": (O + "SM_Frame01.usd", "wall"), "frame5": (O + "SM_Frame05.usd", "wall"),
    "frame9": (O + "SM_Frame09.usd", "wall"), "board": (O + "SM_MarkerBoard.usd", "wall"), "clock": (O + "SM_Clock_A.usd", "wall"),
    "tv": (O + "SM_TV_for_wall.usd", "wall"),
    # clutter on shelves / desks
    "books1": (O + "SM_BooksSet_01.usd", "clutter"), "books2": (O + "SM_BooksSet_02.usd", "clutter"), "books3": (O + "SM_BooksSet_03.usd", "clutter"),
    "binders": (O + "SM_RingBinderStackA.usd", "clutter"), "vase": (O + "SM_VaseDecorative.usd", "clutter"), "monitor": (O + "SM_MonitorPC.usd", "clutter"),
    # windows / blinds / ceiling
    "blinds": (O + "SM_BlindsBigOpen.usd", "window"), "ceiling_light": (O + "SM_CeilingLight.usd", "ceiling"),
}


def fetch(rel):
    """download BASE+rel to HERE/rel (once); returns local path or None"""
    dst = os.path.join(HERE, rel)
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(BASE + urllib.request.quote(rel), timeout=30) as r, open(dst + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            break
        except urllib.error.HTTPError as e:
            print("  missing", rel, e.code); return None
        except Exception as e:          # stalled / reset connection: retry
            print("  retry", rel, e, flush=True)
    else:
        return None
    os.replace(dst + ".part", dst)
    if dst.lower().endswith((".png", ".jpg", ".jpeg", ".tga")) and TEX_MAX:
        shrink(dst)
    return dst


def shrink(path):
    try:
        from PIL import Image
        im = Image.open(path)
        if max(im.size) > TEX_MAX:
            s = TEX_MAX / max(im.size)
            im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.LANCZOS).save(path)
    except Exception as e:      # keep the original if Pillow cannot handle it
        print("  (no resize)", path, e)


def norm(rel_dir, p):
    return os.path.normpath(os.path.join(rel_dir, p)).replace(os.sep, "/")


def usd_deps(local, rel):
    layer = Sdf.Layer.FindOrOpen(local)
    d = os.path.dirname(rel); out = set()
    for p in layer.GetCompositionAssetDependencies():
        out.add(norm(d, p))
    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.AttributeSpec) and spec.default is not None:
            v = spec.default
            for a in (v if isinstance(v, (list, tuple)) or type(v).__name__.endswith("Array") else [v]):
                if isinstance(a, Sdf.AssetPath) and a.path and "://" not in a.path:
                    out.add(norm(d, a.path))
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return out


def mdl_deps(local, rel):
    d = os.path.dirname(rel); txt = open(local, encoding="utf-8", errors="ignore").read(); out = set()
    for t in re.findall(r'texture_2d\("([^"]+)"', txt):
        out.add(norm(d, t.lstrip("/") if not t.startswith(".") else t))
    for m in re.findall(r'(?:using|import)\s+\.::(\w+)', txt):
        out.add(norm(d, m + ".mdl"))
    return out


def closure(rel):
    seen, todo = set(), [rel]
    while todo:
        batch = sorted(set(r for r in todo if r not in seen)); todo = []
        seen.update(batch)
        with ThreadPoolExecutor(16) as ex:
            locs = list(ex.map(fetch, batch))
        for r, loc in zip(batch, locs):
            if loc is None:
                continue
            if r.endswith((".usd", ".usda", ".usdc")):
                todo += list(usd_deps(loc, r))
            elif r.endswith(".mdl"):
                todo += list(mdl_deps(loc, r))
    return seen


def bbox(local):
    st = Usd.Stage.Open(local)
    up, mpu = UsdGeom.GetStageUpAxis(st), UsdGeom.GetStageMetersPerUnit(st)
    b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(st.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = [list(map(float, v)) for v in (b.GetMin(), b.GetMax())]
    if up == "Y":   # -> Z-up (rotate +90 deg about X: (x, y, z) -> (x, -z, y))
        lo, hi = [lo[0], -hi[2], lo[1]], [hi[0], -lo[2], hi[1]]
    return up, mpu, [v * mpu for v in lo], [v * mpu for v in hi]


def main():
    meta = {}
    for name, (rel, cat) in PROPS.items():
        files = closure(rel)
        up, mpu, lo, hi = bbox(os.path.join(HERE, rel))
        meta[name] = dict(usd=rel, cat=cat, up=up, mpu=mpu, lo=[round(v, 4) for v in lo], hi=[round(v, 4) for v in hi])
        print(f"{name:14s} {cat:8s} up={up} mpu={mpu} size={[round(h - l, 2) for l, h in zip(lo, hi)]} files={len(files)}", flush=True)
    json.dump(meta, open(os.path.join(HERE, "props.json"), "w"), indent=1)
    import importlib.util       # procedural wall / floor / ceiling / window textures
    spec = importlib.util.spec_from_file_location("room_textures", os.path.join(os.path.dirname(HERE), "room_textures.py"))
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt); rt.main()
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(HERE) for f in fs)
    print(f"total {tot / 1e6:.1f} MB in {HERE}")


if __name__ == "__main__":
    main()
