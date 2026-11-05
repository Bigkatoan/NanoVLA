"""Audit language labels against what the episode actually shows. No pretrained models.

Checks per (repo, ep):
  degenerate   : '<ALL FROZEN FRAMES>', empty, 'test', <=2 words
  orig_conflict: annotated colours/objects disjoint from the repo's original task colours/objects
  color_grasp  : colour of the object between the gripper fingers (wrist cam, at the grasp moment) contradicts the colour word
  direction    : 'from left to right' vs 'from right to left' claims inconsistent with shoulder_pan motion within the repo
  count        : text asks for N objects (two/three/both...) but the gripper grasps fewer times
Outputs: data/clean/label_audit.parquet, data/clean/samples/label_*.jpg
"""
import os, io, re, sys, json
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from clean_data import smooth

ROOT = os.path.expanduser("~/VLA/data")
OUT = os.path.join(ROOT, "clean")
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]  # hue-detectable colours only
HUE = {"red": [(0, 10), (340, 360)], "orange": [(10, 35)], "yellow": [(35, 70)], "green": [(70, 170)], "blue": [(170, 255)], "purple": [(255, 300)], "pink": [(300, 340)]}
ALLC = COLORS + ["white", "black", "gray", "grey", "brown", "beige", "silver"]
OBJ_EQ = [{"cube", "block", "blocks", "cubes", "lego", "brick"}, {"cup", "mug", "glass"}, {"bin", "box", "container", "basket", "bucket", "crate"},
          {"ball"}, {"duck"}, {"pen", "marker", "pencil", "highlighter"}, {"bowl"}, {"plate", "dish"}, {"bottle"}, {"tape"}, {"banana"}, {"sponge"}]
NUM = {"two": 2, "2": 2, "both": 2, "three": 3, "3": 3, "four": 4, "4": 4, "five": 5, "5": 5}


def words(t):
    return re.findall(r"[a-z0-9]+", str(t).lower())


def colors_in(t):
    return {w if w != "grey" else "gray" for w in words(t) if w in ALLC}


def objs_in(t):
    ws = set(words(t))
    return {i for i, g in enumerate(OBJ_EQ) if ws & g}


def object_color(t):
    """colour word directly preceding the first manipulated-object noun (before 'in/into/on/to')"""
    ws = words(t)
    stop = next((i for i, w in enumerate(ws) if w in ("in", "into", "inside", "on", "onto", "to", "towards", "near")), len(ws))
    for i, w in enumerate(ws[:stop]):
        if w in COLORS and not set(ws[i + 1:i + 3]) & {"arm", "gripper", "robot", "robotic"}:
            return w
    return None


def hue_hist(im):
    a = np.asarray(im.convert("HSV")).astype(np.float32)
    h, s, v = a[..., 0] * 360 / 255, a[..., 1] / 255, a[..., 2] / 255
    m = (s > 0.35) & (v > 0.25)
    out = {}
    for c, rs in HUE.items():
        mm = np.zeros_like(m)
        for lo, hi in rs:
            mm |= (h >= lo) & (h < hi)
        out[c] = float((mm & m).sum()) / m.size
    return out


def main():
    I = pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["repo_id", "ep", "cam", "wrist", "fidx", "off", "len", "task_id"])
    T = pd.read_parquet(os.path.join(ROOT, "packed/tasks.parquet")).set_index("task_id")
    R = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet")).set_index("repo_id")
    blob = np.memmap(os.path.join(ROOT, "packed/frames.bin"), dtype=np.uint8, mode="r")
    epi = I.drop_duplicates(["repo_id", "ep"])[["repo_id", "ep", "task_id"]]
    ro = epi.merge(T[["task_orig"]], left_on="task_id", right_index=True).merge(R[["repo"]], left_on="repo_id", right_index=True)
    ro["user"] = ro.repo.str.split("/").str[0]; ro["o"] = ro.task_orig.str.strip().str.lower()
    users_per_orig = ro.drop_duplicates(["user", "o"]).o.value_counts()
    TEMPLATE = set(users_per_orig[users_per_orig >= 5].index) | {"test run.", "test", ""}
    print("template originals:", len(TEMPLATE), list(TEMPLATE)[:8])
    rows, crops = [], []
    for rid, Er in epi.groupby("repo_id"):
        repo, fps = R.loc[rid].repo, int(R.loc[rid].fps)
        af = os.path.join(ROOT, "actions", repo.replace("/", "__") + ".npz")
        z = np.load(af) if os.path.exists(af) else None
        per = []
        for e, tid in zip(Er.ep, Er.task_id):
            task, orig = T.loc[tid].task, T.loc[tid].task_orig
            r = dict(repo_id=rid, repo=repo, ep=e, task=task, orig=orig, flags=[])
            ws = words(task)
            if "frozen" in ws or len(ws) <= 2 or re.fullmatch(r"\s*(test|none|task|pick place|demo)\s*", str(task).lower() or ""):
                r["flags"].append("degenerate")
            ca, co = colors_in(task), colors_in(orig)
            oa, oo = objs_in(task), objs_in(orig)
            r["orig_template"] = str(orig).strip().lower() in TEMPLATE
            if r["orig_template"]:
                r["flags"].append("orig_template")
            if (ca and co and not (ca & co)) or (oa and oo and not (oa & oo)):
                r["flags"].append("orig_conflict")
            r["obj_color"] = object_color(task)
            if z is None or f"ep{e}_action" not in z.files:
                r["has_actions"] = False; per.append(r); continue
            r["has_actions"] = True
            A = z[f"ep{e}_action"]
            g = A[:, -1]; gt = (g - g.min()) / (g.max() - g.min() + 1e-6)
            closed = smooth(gt, max(1, fps // 5)) < 0.35
            on = np.nonzero(np.diff(closed.astype(int)) == 1)[0]
            r["grasps"] = len(on)
            # count
            n_req = 0
            for i, w in enumerate(ws):
                if w in NUM and i + 1 < len(ws):
                    nxt = " ".join(ws[i + 1:i + 3])
                    if re.search(r"\b(blocks|cubes|balls|cups|objects|items|pieces|pens|bottles|ducks|toys|lego|legos|dice)\b", nxt) or (i > 0 and ws[i - 1] == "stack"):
                        n_req = max(n_req, NUM[w])
            if n_req >= 2 and len(on) < n_req and (g.max() - g.min()) > 1e-3:
                r["flags"].append(f"count(need{n_req},got{len(on)})")
            # direction
            tl = str(task).lower()
            claim = "LR" if re.search(r"from (the )?left\b.*\bto (the )?right", tl) else ("RL" if re.search(r"from (the )?right\b.*\bto (the )?left", tl) else None)
            r["dir_claim"] = claim
            if len(on):
                k0 = on[0]; rel = np.nonzero(np.diff(closed.astype(int)) == -1)[0]; k1 = rel[rel > k0][0] if (rel > k0).any() else len(A) - 1
                r["pan_delta"] = float(A[k1, 0] - A[k0, 0])
            # colour at grasp (wrist cam)
            oc = object_color(task)
            Iw = I[(I.repo_id == rid) & (I.ep == e) & (I.wrist == 1)]
            if oc and len(on) and len(Iw):
                t0 = on[0]
                cand = Iw[(Iw.fidx >= t0) & (Iw.fidx <= t0 + int(0.8 * fps))]
                if len(cand):
                    row = cand.iloc[0]
                    im = Image.open(io.BytesIO(bytes(blob[row.off: row.off + row.len]))).convert("RGB")
                    w_, h_ = im.size
                    crop = im.crop((int(w_ * .25), int(h_ * .3), int(w_ * .75), int(h_ * .95)))
                    hh = hue_hist(crop)
                    dom = max(hh, key=hh.get)
                    r["grasp_color"] = dom; r["grasp_color_frac"] = hh[dom]; r["expected_color_frac"] = hh[oc]
                    r["_crop"] = crop
            per.append(r)
        # label colour consistency within repo: grasped object looks the same (same dominant hue in >=80% of episodes)
        # but the annotation names >=2 different colours for it -> minority-colour episodes are inconsistent labels
        C = [p for p in per if p.get("grasp_color") and p.get("obj_color") and p.get("grasp_color_frac", 0) > 0.08]
        if len(C) >= 4:
            hv = pd.Series([p["grasp_color"] for p in C]).value_counts()
            lv = pd.Series([p["obj_color"] for p in C]).value_counts()
            if hv.iloc[0] / len(C) >= 0.8 and len(lv) >= 2:
                major = lv.index[0]
                for p in C:
                    if p["obj_color"] != major:
                        p["flags"].append(f"color_inconsistent(said {p['obj_color']}, repo mostly {major}, same-looking object)")
                        crops.append((p["_crop"], p["obj_color"], major, repo, p["ep"]))
        for p in per:
            p.pop("_crop", None)
        # direction consistency within repo
        D = [p for p in per if p.get("dir_claim") and "pan_delta" in p and abs(p["pan_delta"]) > 1e-3]
        if len(D) >= 4 and {p["dir_claim"] for p in D} == {"LR", "RL"}:
            sgn = {c: np.sign(np.median([p["pan_delta"] for p in D if p["dir_claim"] == c])) for c in ("LR", "RL")}
            if sgn["LR"] != sgn["RL"]:
                for p in D:
                    if np.sign(p["pan_delta"]) != sgn[p["dir_claim"]]:
                        p["flags"].append("direction")
        rows += per
    A_ = pd.DataFrame(rows)
    A_["flags"] = A_["flags"].map(lambda f: ",".join(f))
    A_.to_parquet(os.path.join(OUT, "label_audit.parquet"))
    h = A_[A_.has_actions.fillna(False)]
    kinds = pd.Series([f.split("(")[0] for fl in A_["flags"] for f in fl.split(",") if f]).value_counts().to_dict()
    rep = dict(episodes=len(A_), with_actions=int(len(h)), flagged_any=int((A_["flags"] != "").sum()), flagged_with_actions=int((h["flags"] != "").sum()), by_kind=kinds,
               color_checked=int(h.get("grasp_color", pd.Series(dtype=object)).notna().sum()), direction_claims=int(h.dir_claim.notna().sum()))
    json.dump(rep, open(os.path.join(OUT, "label_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 100)
    for k in ["orig_template", "orig_conflict", "count", "color_inconsistent", "degenerate"]:
        s = A_[A_["flags"].str.contains(k)]
        print(f"\n== {k}: {len(s)}"); print(s[["repo", "ep", "orig", "task", "flags"]].head(6).to_string(index=False))
    if crops:
        rs = np.random.RandomState(0)
        sel = [crops[i] for i in rs.choice(len(crops), min(24, len(crops)), replace=False)]
        sheet = Image.new("RGB", (6 * 170, 4 * 150), "white"); d = ImageDraw.Draw(sheet)
        for i, (c, oc, dom, repo, e) in enumerate(sel):
            x, y = (i % 6) * 170, (i // 6) * 150
            sheet.paste(c.resize((160, 120)), (x + 5, y)); d.text((x + 5, y + 122), f"said {oc} / repo: {dom}", fill="black"); d.text((x + 5, y + 135), f"{repo.split('/')[-1][:22]} e{e}", fill="gray")
        sheet.save(os.path.join(OUT, "samples", "label_color_inconsistent.jpg"))


if __name__ == "__main__":
    main()
