"""Real forward + backward of NanoVLA-C on one training batch (CPU): losses, per-module gradient norms, gradient that
reaches the memory at each frame (BPTT), which parameters are frozen. Mirrors nanovla_c.compute / run_seq."""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/VLA/research/vla"))
import nanovla2 as v2, nanovla_r as R, nanovla_c as C
torch.manual_seed(0); np.random.seed(0)
ck = torch.load("/home/bigkatoan/VLA/research/viz/vlac_snap.pt", map_location="cpu", weights_only=False)
P = v2.prepare("isaac_v2/place_c8", 4, 0); P["train_eps"] = sorted({int(e) for e, _ in P["train"]})
R.T_MAX = 32
m, tok = C.build("cpu", ck["st"], init=""); m.load_state_dict(ck["model"]); m.train()
for n, p in m.vis.v.named_parameters():
    if n.startswith(("proj.", "pos", "cls", "view_emb")) or any(n.startswith(f"blocks.{i}.") for i in range(4)):
        p.requires_grad = False
ds = R.SeqData(P, P["train_eps"], True, "own"); import random; random.seed(1)
b = R.collate([ds[i] for i in (0, 5, 9, 14)])
img = R.gpu_images(b["img"], v2.SIZE["own"], True); states = b["state"]; tgt = b["tgt"][:, :, :C.HM]; valid = b["valid"][:, :, :C.HM]; msk = b["mask"]
B, Tn = states.shape[:2]
u = m.text_vec(tok(b["label"])); v = m.image_vec(img.flatten(0, 1)).view(B, Tn, -1); v.retain_grad()
mem = m.init_mem(u); plan = torch.zeros(B, C.HM, 6); pred = torch.zeros(B, 6 + C.PE); first = torch.ones(B, dtype=torch.bool); snp = states[:, 0, :6]
Ms, Ps, Qs, Zs, Es = [], [], [], [], []
for k in range(Tn):
    mem.retain_grad(); Ms.append(mem)
    plan, mem, pred, e, z = m.tick(v[:, k], states[:, k], u, mem, plan.detach(), pred.detach(), snp, first)
    Ps.append(plan); Qs.append(pred); Zs.append(z); Es.append(e); snp = states[:, k, :6]; first = torch.zeros_like(first)
plans, preds, zs, mems = (torch.stack(x, 1) for x in (Ps, Qs, Zs, Ms))
vm = valid.unsqueeze(-1).float() * msk[:, :, None, None].float(); w = torch.ones(6); w[5] = .5
la = ((plans - tgt).abs() * w * vm).sum() / (vm.sum() * 6)
nxt = msk[:, 1:] & msk[:, :-1]
ls = F.smooth_l1_loss(preds[:, :-1, :6][nxt], states[:, 1:, :6][nxt]); lz = F.mse_loss(preds[:, :-1, 6:][nxt], zs[:, 1:][nxt].detach())
lp = F.cross_entropy(m.phase(mems[msk]), b["phase"][msk])
loss = la + .5 * (ls + lz) + .1 * lp
loss.backward()
groups = {"ViT (blocks 4-11)": "vis.", "text encoder": "t.", "pooling": ("vproj", "vpos", "pool"), "vvec": "vvec", "init m0": "init.", "decision MLP": "mlp.",
          "forward model": "fwd.", "GRU memory": "gru.", "phase head": "phase.", "task proj": "tproj."}
gn = {}
for g, pre in groups.items():
    pre = pre if isinstance(pre, tuple) else (pre,)
    sq = sum(float(p.grad.pow(2).sum()) for n, p in m.named_parameters() if p.grad is not None and n.startswith(pre))
    gn[g] = sq ** .5
tot = sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None) ** .5
frozen = round(sum(p.numel() for p in m.parameters() if not p.requires_grad) / 1e6, 2); train_ = round(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 2)
mg = [float(x.grad.norm()) if x.grad is not None else 0.0 for x in Ms]
vg = [float(v.grad[:, k].norm()) for k in range(Tn)]
out = dict(loss=float(loss), la=float(la), ls=float(ls), lz=float(lz), lp=float(lp), grad=gn, total_grad=tot, clip=1.0, frozen_M=frozen, trainable_M=train_,
           mem_grad=mg, img_grad=vg, T=Tn, step=ck["steps"], lr_head=3e-4, lr_enc=2e-5, ema=0.999,
           err=[float(Es[k][0, :6].abs().mean()) for k in range(Tn)])
json.dump(out, open("/home/bigkatoan/VLA/research/viz/bp.json", "w"), indent=1)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k not in ("mem_grad", "img_grad", "err")}))
print("mem grad first/last", np.round(mg[:3], 4), np.round(mg[-3:], 4))
