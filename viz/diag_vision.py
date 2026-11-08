"""Does NanoVLA-C use its eyes? Offline val error with (a) real images, (b) images from ANOTHER episode, (c) zero image
vector, (d) joints frozen at t0 (no proprio motion), (e) text of another episode."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/VLA/vla"))
import nanovla2 as v2, nanovla_r as R, nanovla_c as C
torch.set_num_threads(4); torch.manual_seed(0)
ck = torch.load(os.path.expanduser("~/VLA/ckpt/nanovla_c/vlac.pt"), map_location="cpu", weights_only=False)
P = v2.prepare("isaac_all_v2", 100, 0); P["train_eps"] = sorted({int(e) for e, _ in P["train"]}); R.T_MAX = 48
m, tok = C.build("cpu", ck["st"], init=""); m.load_state_dict(ck["model"]); m.eval()
eps = sorted(e for e in P["acts"] if e not in P["train_eps"])[:100]
bl = list(torch.utils.data.DataLoader(R.SeqData(P, eps, False, "own"), batch_size=10, num_workers=4, collate_fn=R.collate))
HM = C.HM
@torch.no_grad()
def run(b, mode):
    img = R.gpu_images(b["img"], v2.SIZE["own"], False); st = b["state"].clone(); B, Tn = st.shape[:2]; lab = list(b["label"])
    if mode == "other_images": img = img.roll(1, 0)
    if mode == "frozen_joints": st[:, :, :6] = st[:, :1, :6]; st[:, :, 6:] = 0
    if mode == "other_text": lab = lab[1:] + lab[:1]
    u = m.text_vec(tok(lab)); v = m.image_vec(img.flatten(0, 1)).view(B, Tn, -1)
    if mode == "zero_image": v = v * 0 + v.mean((0, 1), keepdim=True)
    mem = m.init_mem(u); plan = torch.zeros(B, HM, 6); pred = torch.zeros(B, 6 + C.PE); first = torch.ones(B, dtype=torch.bool); snp = st[:, 0, :6]; Ps = []
    for k in range(Tn):
        plan, mem, pred, e, z = m.tick(v[:, k], st[:, k], u, mem, plan, pred, snp, first); Ps.append(plan); snp = st[:, k, :6]; first = torch.zeros_like(first)
    return torch.stack(Ps, 1)
res = {}
for mode in ("real", "other_images", "zero_image", "frozen_joints", "other_text"):
    a = h = 0.0; ag = hg = 0.0
    for b in bl:
        p = run(b, mode); msk = b["mask"]; tg = b["tgt"][:, :, :HM][msk]; va = b["valid"][:, :, :HM][msk].unsqueeze(-1).float(); p = p[msk]
        z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
        a += float(((p - tg).abs() * va)[..., :5].sum()); h += float(((z - tg).abs() * va)[..., :5].sum())
        ag += float(((p - tg).abs() * va)[..., 5].sum()); hg += float(((z - tg).abs() * va)[..., 5].sum())
    res[mode] = dict(arm=round(a / h, 3), gripper=round(ag / hg, 3)); print(mode, res[mode], flush=True)
json.dump(res, open(os.path.expanduser("~/VLA/viz/diag_vision.json"), "w"), indent=1)
