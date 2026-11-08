"""Diagnostics of NanoVLA-C on 100 held-out episodes (CPU): ablations of each closed-loop mechanism, forward model vs
naive baselines, action amplitude shrinkage, gripper timing, pooling-query collapse."""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/VLA/research/vla"))
import nanovla2 as v2, nanovla_r as R, nanovla_c as C
torch.set_num_threads(6)
ck = torch.load(os.path.expanduser("~/VLA/ckpt/nanovla_c/vlac.pt"), map_location="cpu", weights_only=False)
P = v2.prepare("isaac_all_v2", 100, 0); P["train_eps"] = sorted({int(e) for e, _ in P["train"]})
R.T_MAX = 48
m, tok = C.build("cpu", ck["st"], init=""); m.load_state_dict(ck["model"]); m.eval()
eps = sorted(e for e in P["acts"] if e not in P["train_eps"])[:100]
dl = torch.utils.data.DataLoader(R.SeqData(P, eps, False, "own"), batch_size=10, num_workers=4, collate_fn=R.collate)
batches = list(dl)
HM = C.HM

@torch.no_grad()
def run(b, mode="full"):
    img = R.gpu_images(b["img"], v2.SIZE["own"], False); st = b["state"]; B, Tn = st.shape[:2]
    u = m.text_vec(tok(b["label"])); v = m.image_vec(img.flatten(0, 1)).view(B, Tn, -1)
    mem = m.init_mem(u); m0 = mem.clone(); plan = torch.zeros(B, HM, 6); pred = torch.zeros(B, 6 + C.PE); first = torch.ones(B, dtype=torch.bool); snp = st[:, 0, :6]
    out = dict(plan=[], pred=[], z=[], delta=[], sh=[], e=[])
    for k in range(Tn):
        if mode == "no_memory": mem = m0
        sn = st[:, k, :6]; z = m.zproj(v[:, k]); keep = (~first).float()[:, None]
        e = torch.cat([sn - pred[:, :6], z - pred[:, 6:]], -1) * keep
        if mode == "no_error": e = e * 0
        sh = m.shift(plan, snp, sn) * keep[:, :, None]
        if mode == "no_plan_carry": sh = sh * 0
        delta = m.mlp(torch.cat([v[:, k], st[:, k], u, mem, e, sh.flatten(1)], -1)).view(-1, HM, 6)
        plan = sh + delta; a = plan[:, 0]
        pred = m.fwd(torch.cat([v[:, k], st[:, k], a, mem], -1))
        mem = m.gru(torch.cat([v[:, k], st[:, k], a, e], -1), mem)
        for kk, x in (("plan", plan), ("pred", pred), ("z", z), ("delta", delta), ("sh", sh), ("e", e)): out[kk].append(x)
        snp = sn; first = torch.zeros_like(first)
    return {k: torch.stack(x, 1) for k, x in out.items()}

def rel(o, b):
    msk = b["mask"]; tg = b["tgt"][:, :, :HM][msk]; va = b["valid"][:, :, :HM][msk].unsqueeze(-1).float(); p = o["plan"][msk]
    z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
    return float(((p - tg).abs() * va).sum()), float(((z - tg).abs() * va).sum())

res = {}
for mode in ("full", "no_error", "no_plan_carry", "no_memory"):
    a = h = 0.0
    for b in batches:
        o = run(b, mode); x, y = rel(o, b); a += x; h += y
    res[mode] = round(a / h, 4); print(mode, res[mode], flush=True)
# forward model vs naive baselines, amplitude, gripper timing, delta vs carried plan
fs = fn = fc = 0.0; nn_ = 0; amp_p = amp_l = 0.0; dmag = smag = 0.0; lag = []; e_mag = []
for b in batches:
    o = run(b); st = b["state"]; msk = b["mask"]; nxt = msk[:, 1:] & msk[:, :-1]
    ps = o["pred"][:, :-1, :6][nxt]; real = st[:, 1:, :6][nxt]; cur = st[:, :-1, :6][nxt]
    fs += float((ps - real).abs().mean()) ; fn += float((cur - real).abs().mean())
    tg1 = b["tgt"][:, :, 0]  # label first point (normalised)
    pl1 = o["plan"][:, :, 0]
    amp_p += float(pl1[msk][:, :5].abs().mean()); amp_l += float(tg1[msk][:, :5].abs().mean())
    dmag += float(o["delta"][:, 1:][msk[:, 1:]].abs().mean()); smag += float(o["sh"][:, 1:][msk[:, 1:]].abs().mean())
    e_mag.append(float(o["e"][:, 1:, :6][msk[:, 1:]].abs().mean()))
    for i in range(len(st)):
        T0 = int(msk[i].sum()); gl = tg1[i, :T0, 5].numpy(); gp = pl1[i, :T0, 5].numpy()
        cl = np.nonzero(np.diff((gl < 0).astype(int)) == 1)[0]; cp = np.nonzero(np.diff((gp < 0).astype(int)) == 1)[0]
        if len(cl):
            lag.append(int(cp[0] - cl[0]) if len(cp) else None)
    nn_ += 1
res["forward_state_err"] = round(fs / nn_, 4); res["naive_const_err"] = round(fn / nn_, 4)
res["amp_ratio_first_step"] = round(amp_p / amp_l, 3); res["delta_vs_carried"] = round(dmag / smag, 3); res["mean_abs_e"] = round(float(np.mean(e_mag)), 4)
L = [x for x in lag if x is not None]
res["grip_close_events"] = len(lag); res["grip_close_missed"] = sum(x is None for x in lag); res["grip_close_lag_ticks_median"] = float(np.median(L)) if L else None
# pooling query collapse: cosine similarity between the 4 pooled queries (after the pooling block)
with torch.no_grad():
    b = batches[0]; img = R.gpu_images(b["img"][:, :8], v2.SIZE["own"], False).flatten(0, 1)
    tokv = m.vproj(m.vis(img)) + m.vpos; q = m.pool_q.expand(len(img), -1, -1) * 1.0
    a_, w_ = m.pool(m.pool_n1(q), tokv, tokv, need_weights=True); q = q + a_; q = q + m.pool_mlp(m.pool_n2(q))
    qn = F.normalize(q, dim=-1); cs = (qn @ qn.transpose(1, 2))[:, ~torch.eye(4, dtype=torch.bool)].mean()
    ent = float(-(w_ * (w_ + 1e-9).log()).sum(-1).mean()); res["pool_query_cos"] = round(float(cs), 3); res["pool_attn_entropy"] = round(ent, 3); res["uniform_entropy"] = round(float(np.log(80)), 3)
    wq = F.normalize(w_, dim=-1); res["pool_attnmap_cos"] = round(float((wq @ wq.transpose(1, 2))[:, ~torch.eye(4, dtype=torch.bool)].mean()), 3)
# train / val gap from the log
L_ = [json.loads(l) for l in open(os.path.expanduser("~/VLA/ckpt/nanovla_c/train.log")) if l.startswith("{")]
res["train_act_last"] = [r["act"] for r in L_ if "act" in r][-1]; res["val_l1_last"] = [r["val_l1"] for r in L_ if "val_l1" in r][-1]
json.dump(res, open(os.path.expanduser("~/VLA/research/viz/diag_c.json"), "w"), indent=1); print(json.dumps(res, indent=1))
