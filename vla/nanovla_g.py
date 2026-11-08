"""Grounded Closed-loop Training (GCT), stage 1: offline pre-training of the closed-loop NanoVLA with anti-copycat
measures (2026-11-06 design doc "NanoVLA: phân tích lỗi và kỹ thuật training chuyên dụng").

Model = NanoVLA-C (+ 8 pooling queries instead of 4, a localisation head, a progress head). Changes in training:
  L = L_action + 0.5 L_forward + 0.1 L_phase + 0.5 L_localize + 0.1 L_progress + 0.05 L_diversity
  L_localize  : target grasp point + destination in the wrist-camera frame, predicted from [v, u] ONLY (not memory)
  L_diversity : mean off-diagonal cosine between the pooled queries
  history dropout: per frame p=0.3 the joint change and the error signal e are zeroed; per window p=0.2 the memory
                   is reset to m0 at a random frame
  plan dropout   : per frame p=0.3 the carried plan is dropped -> re-plan from the image
  keyframe weight: x3 within +-0.5 s of a jaw open/close, x2 in the first second of an episode
  windows        : 64 frames; mid-episode windows run their first 16 frames as burn-in (no loss, no gradient)
Monitors (TensorBoard, every eval): vision reliance (val error with images from another episode / normal),
localisation error (cm), re-plan ratio |delta| / |carried plan|, error-signal reliance (e zeroed / normal).

    python nanovla_g.py --run gct_s1 --bench isaac_gct_v1 --steps 20000 --out ~/VLA/ckpt/nanovla_g/vlag.pt
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import nanovla2 as v2
import nanovla_r as R
import nanovla_c as C
from train_vla import H

CK = os.path.expanduser("~/VLA/ckpt")
TB = os.path.expanduser("~/VLA/tb")
HM, PE = C.HM, C.PE
BURN = 16


class NanoVLAG(C.NanoVLAC):
    def __init__(self, vis, tenc, n_tok, ratio, K=8, **kw):
        super().__init__(vis, tenc, n_tok, ratio, K=K, **kw)
        d, dv, mem = self.d, 512, self.mem
        self.loc = nn.Sequential(nn.LayerNorm(dv + d), nn.Linear(dv + d, 256), nn.GELU(), nn.Linear(256, 6))     # from what is SEEN + the task
        self.prog = nn.Sequential(nn.LayerNorm(mem), nn.Linear(mem, 1))
        self._q = None
        # grounding (2026-11-07): the TASK vector attends over the image tokens (the image vector v stays text-free);
        # two queries = target object, destination; soft-argmax over the token grid + attended features -> xyz (camera frame)
        self.gq = nn.Linear(d, 2 * d); self.gk = nn.Linear(d, d)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 10), indexing="ij")
        self.register_buffer("grid", torch.stack([gx.flatten(), gy.flatten()], -1))          # [80, 2] token centres
        self.gout = nn.Sequential(nn.LayerNorm(d + 2), nn.Linear(d + 2, 128), nn.GELU(), nn.Linear(128, 3))
        self.lemb = nn.Linear(6, d); nn.init.zeros_(self.lemb.weight); nn.init.zeros_(self.lemb.bias)   # loc -> decision (starts as a no-op)
        self._tok = None

    def ground(self, tok, u):
        """tok [N,80,d], u [N,d] -> loc [N,6] (metres, camera frame: object xyz, destination xyz), attention logits [N,2,80]"""
        q = self.gq(u).view(-1, 2, self.d); k = self.gk(tok)
        logit = torch.einsum("nqd,ntd->nqt", q.float(), k.float()) / self.d ** 0.5
        att = logit.softmax(-1)
        xy = att @ self.grid                                                 # [N,2,2] soft-argmax
        feat = att @ tok.float()                                             # [N,2,d]
        xyz = self.gout(torch.cat([feat, xy], -1).to(tok.dtype)).float()     # decimetres
        return (xyz * 0.1).flatten(1), logit

    def image_vec(self, img, u=None):
        tok = self.vproj(self.vis(img)) + self.vpos
        self._tok = tok
        q = self.pool_q.expand(len(img), -1, -1) * 1.0
        a, _ = self.pool(self.pool_n1(q), tok, tok, need_weights=False)
        q = q + a
        q = q + self.pool_mlp(self.pool_n2(q))
        self._q = q
        return self.vvec(q.flatten(1))

    def tick_g(self, v, s12, u, m, plan_prev, pred_prev, sn_prev, first, drop_hist=None, drop_plan=None, loc=None):
        """NanoVLAC.tick with history / plan dropout masks ([B] bool); loc [B,6] (metres) conditions the decision"""
        if loc is not None:
            u = u + self.lemb((loc / 0.3).clamp(-5, 5).to(u.dtype))
        sn = s12[:, :6]
        z = self.zproj(v).float()
        keep = (~first).float()[:, None]
        e = torch.cat([sn - pred_prev[:, :6], z - pred_prev[:, 6:]], -1) * keep
        sh = self.shift(plan_prev, sn_prev, sn) * keep[:, :, None]
        if drop_hist is not None:
            dh = (~drop_hist).float()[:, None]
            e = e * dh; s12 = torch.cat([s12[:, :6], s12[:, 6:] * dh.to(s12.dtype)], -1)
        if drop_plan is not None:
            sh = sh * (~drop_plan).float()[:, None, None]
        delta = self.mlp(torch.cat([v, s12, u, m.to(v.dtype), e.to(v.dtype), sh.flatten(1).to(v.dtype)], -1)).view(-1, HM, 6)
        plan = sh + delta.float()
        a = plan[:, 0]
        pred = self.fwd(torch.cat([v, s12, a.to(v.dtype), m.to(v.dtype)], -1)).float()
        m2 = self.gru(torch.cat([v, s12, a.to(v.dtype), e.to(v.dtype)], -1).float(), m.float())
        return plan, m2, pred, e, z, delta, sh

    @torch.no_grad()
    def step(self, img, s12, u, m, plan_prev, pred_prev, sn_prev, first):
        v = self.image_vec(img)
        loc, _ = self.ground(self._tok, u)
        plan, m2, pred, e, _, _, _ = self.tick_g(v, s12, u, m, plan_prev, pred_prev, sn_prev, first, loc=loc)
        self.last_loc = loc
        return plan, m2, pred, e


def build(dev, st, init=os.path.join(CK, "nanovla_c/vlac.pt"), v1_ckpt=os.path.join(CK, "vla_isaac_wrist_v1/vla.pt")):
    s = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    v1, tok = v2.build_model(s["n_emb"], os.path.join(CK, "align/align.pt"), "cpu")
    v1.load_state_dict(s["model"])
    model = NanoVLAG(v2.OwnVision(v1.v), v1.t, v2.n_tokens("own"), np.asarray(st["s_std"][:5]) / np.asarray(st["dstd"]))
    if init:
        src = torch.load(init, map_location="cpu", weights_only=False)["model"]; own = model.state_dict()
        ok = {k: w for k, w in src.items() if k in own and own[k].shape == w.shape}
        if "pool_q" in src and src["pool_q"].shape[1] * 2 == own["pool_q"].shape[1]:      # 4 -> 8 queries: copy + jitter
            ok["pool_q"] = torch.cat([src["pool_q"], src["pool_q"] + 0.02 * torch.randn_like(src["pool_q"])], 1)
        own.update(ok); model.load_state_dict(own)
        print(json.dumps(dict(init=init, loaded=len(ok), new=sorted({k.split(".")[0] for k in own if k not in ok}))), flush=True)
    return model.to(dev), tok


def run_window(model, b, tok, dev, train, p_hist=0.3, p_mem=0.2, p_plan=0.3, shuffle_images=False, zero_e=False, p_tf=0.5, shuffle_loc=False):
    img = R.gpu_images(b["img"].to(dev, non_blocking=True), v2.SIZE["own"], train)
    if shuffle_images:
        img = img.roll(1, 0)
    st = b["state"].to(dev); B, Tn = st.shape[:2]
    u = model.text_vec(tok(b["label"]).to(dev))
    v = model.image_vec(img.flatten(0, 1)).view(B, Tn, -1); qn = F.normalize(model._q.float(), dim=-1)
    locp, glogit = model.ground(model._tok, u.repeat_interleave(Tn, 0)); locp = locp.view(B, Tn, 6); glogit = glogit.view(B, Tn, 2, -1)
    loc_in = locp
    if train and "loc" in b and p_tf > 0:                     # teacher forcing: true location (+1 cm noise) half of the time
        tf = (torch.rand(B, Tn, 1, device=dev) < p_tf) & b["loc_ok"].to(dev)[..., None]
        loc_in = torch.where(tf, b["loc"].to(dev) + 0.01 * torch.randn_like(locp), locp)
    if shuffle_loc:
        loc_in = loc_in.roll(1, 0)
    m = model.init_mem(u); m0 = m
    plan = torch.zeros(B, HM, 6, device=dev); pred = torch.zeros(B, 6 + PE, device=dev); first = torch.ones(B, dtype=torch.bool, device=dev); snp = st[:, 0, :6]
    burn = torch.where(b["first"].to(dev), 0, BURN) if (train and "first" in b) else torch.zeros(B, dtype=torch.long, device=dev)
    reset_at = torch.where(torch.rand(B, device=dev) < p_mem, torch.randint(BURN, max(BURN + 1, Tn), (B,), device=dev), -1) if train else torch.full((B,), -1, device=dev)
    P_, Q_, Z_, E_, M_, D_, S_ = [], [], [], [], [], [], []
    for k in range(Tn):
        if train:
            rk = (reset_at == k)[:, None]
            m = torch.where(rk, m0.detach(), m)
        dh = (torch.rand(B, device=dev) < p_hist) if train else None
        dp = (torch.rand(B, device=dev) < p_plan) if train else None
        if zero_e:
            dh = torch.ones(B, dtype=torch.bool, device=dev)
        M_.append(m)
        plan, m, pred, e, z, delta, sh = model.tick_g(v[:, k], st[:, k], u, m, plan.detach(), pred.detach(), snp, first, dh, dp, loc=loc_in[:, k])
        P_.append(plan); Q_.append(pred); Z_.append(z); E_.append(e); D_.append(delta); S_.append(sh)
        snp = st[:, k, :6]; first = torch.zeros_like(first)
    stk = lambda L: torch.stack(L, 1)
    return dict(plan=stk(P_), pred=stk(Q_), z=stk(Z_), e=stk(E_), mem=stk(M_), delta=stk(D_), sh=stk(S_), v=v, u=u, qn=qn, burn=burn, locp=locp, glogit=glogit)


def losses(model, b, tok, dev, o, w):
    tgt = b["tgt"].to(dev)[:, :, :HM]; valid = b["valid"].to(dev)[:, :, :HM]; msk = b["mask"].to(dev).clone(); B, Tn = msk.shape
    ks = torch.arange(Tn, device=dev)[None]
    lm = msk & (ks >= o["burn"][:, None])                                       # loss frames (burn-in excluded)
    # keyframe weights: jaw events (phase changes) and the first second of an episode
    ph = b["phase"].to(dev); chg = torch.zeros_like(msk); chg[:, 1:] = ph[:, 1:] != ph[:, :-1]
    near = F.max_pool1d(chg.float()[:, None], 11, 1, 5)[:, 0] > 0
    fw = 1 + 2 * near.float() + (b["fidx"].to(dev) < 30).float() * b["first"].to(dev)[:, None].float()
    wj = torch.ones(6, device=dev); wj[5] = 0.5
    vm = valid.unsqueeze(-1).float() * lm[:, :, None, None].float() * fw[:, :, None, None]
    la = ((o["plan"] - tgt).abs() * wj * vm).sum() / (vm.sum() * 6 + 1e-6)
    nxt = lm[:, 1:] & msk[:, :-1]
    st = b["state"].to(dev)
    ls = F.smooth_l1_loss(o["pred"][:, :-1, :6][nxt], st[:, 1:, :6][nxt]) if nxt.any() else la * 0
    lz = F.mse_loss(o["pred"][:, :-1, 6:][nxt], o["z"][:, 1:][nxt].detach()) if nxt.any() else la * 0
    lp = F.cross_entropy(model.phase(o["mem"][lm]).float(), ph[lm]) if lm.any() else la * 0
    lg = lpr = la * 0; loc_err = None
    if "loc" in b:
        lok = b["loc_ok"].to(dev) & lm
        if lok.any():
            pl = o["locp"][lok]; tl = b["loc"].to(dev)[lok]
            lg = F.smooth_l1_loss(pl, tl, beta=0.01) * 10
            loc_err = (pl - tl)[:, :3].norm(dim=-1).mean()
            # heat-map: project each label into the wrist image (pinhole, f = 103 px at 160 px width) -> token cell
            X, Y, Z = tl[:, 0::3], tl[:, 1::3], tl[:, 2::3]
            uu = 80 + 103.1 * X / Z.clamp(min=1e-3); vv = 60 + 103.1 * Y / Z.clamp(min=1e-3)
            vis = (Z > 0.03) & (uu >= 0) & (uu < 160) & (vv >= 0) & (vv < 120)
            if vis.any():
                gx = (uu / 16 - 0.5) / 4.5 - 1; gy = (vv / 15 - 0.5) / 3.5 - 1          # cell centres on the [-1, 1] grid
                d2 = (model.grid[None, None, :, 0] - gx[..., None]) ** 2 * 20.25 + (model.grid[None, None, :, 1] - gy[..., None]) ** 2 * 12.25
                tgt_h = torch.softmax(-d2 / 2.0, -1)                                    # gaussian, sigma ~ 1 cell
                ce = -(tgt_h * o["glogit"][lok].float().log_softmax(-1)).sum(-1)
                lg = lg + (ce * vis).sum() / vis.sum().clamp(min=1) * 0.2
        lpr = F.mse_loss(torch.sigmoid(model.prog(o["mem"][lm]).float()[:, 0]), b["prog"].to(dev)[lm]) if lm.any() else la * 0
    K = o["qn"].shape[1]; G = o["qn"] @ o["qn"].transpose(1, 2)
    ldiv = (G.sum((1, 2)) - G.diagonal(dim1=1, dim2=2).sum(-1)).mean() / (K * (K - 1))
    total = la + w["fwd"] * (ls + lz) + w["phase"] * lp + w["loc"] * lg + w["prog"] * lpr + w["div"] * ldiv
    return total, dict(action=la, fwd_state=ls, fwd_img=lz, phase=lp, localize=lg, progress=lpr, diversity=ldiv), loc_err


@torch.no_grad()
def evaluate(model, tok, P, dev, max_eps=100):
    eps = sorted(e for e in P["acts"] if e not in P["train_eps"])[:max_eps]
    dl = torch.utils.data.DataLoader(R.SeqData(P, eps, False, "own"), batch_size=4, num_workers=4, collate_fn=R.collate)
    model.eval(); acc = {k: [0.0, 0.0] for k in ("normal", "shuffled_images", "zero_e", "shuffled_loc")}; locs = []; rep = []
    for b in dl:
        for mode in acc:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = run_window(model, b, tok, dev, False, shuffle_images=mode == "shuffled_images", zero_e=mode == "zero_e", shuffle_loc=mode == "shuffled_loc")
            msk = b["mask"].to(dev); tg = b["tgt"].to(dev)[:, :, :HM][msk]; va = b["valid"].to(dev)[:, :, :HM][msk].unsqueeze(-1).float(); p = o["plan"][msk].float()
            z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
            acc[mode][0] += float(((p - tg).abs() * va).sum()); acc[mode][1] += float(((z - tg).abs() * va).sum())
            if mode == "normal":
                rep.append(float(o["delta"][:, 1:][msk[:, 1:]].abs().mean() / (o["sh"][:, 1:][msk[:, 1:]].abs().mean() + 1e-6)))
                if "loc" in b and b["loc_ok"].any():
                    lok = b["loc_ok"].to(dev) & msk
                    locs.append(float((o["locp"][lok] - b["loc"].to(dev)[lok])[:, :3].norm(dim=-1).mean()))
    model.train()
    rel = {k: v[0] / v[1] for k, v in acc.items()}
    return dict(val_rel=round(rel["normal"], 4), vision_reliance=round(rel["shuffled_images"] / rel["normal"], 3), error_reliance=round(rel["zero_e"] / rel["normal"], 3),
                loc_reliance=round(rel["shuffled_loc"] / rel["normal"], 3),
                replan_ratio=round(float(np.mean(rep)), 3), loc_err_cm=round(100 * float(np.mean(locs)), 2) if locs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="gct_s1")
    ap.add_argument("--bench", default="isaac_gct_v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=3e-5)
    ap.add_argument("--freeze_blocks", type=int, default=4)
    ap.add_argument("--p_hist", type=float, default=0.6)
    ap.add_argument("--p_tf", type=float, default=0.5, help="teacher forcing of the location input to the decision")
    ap.add_argument("--p_mem", type=float, default=0.2)
    ap.add_argument("--p_plan", type=float, default=0.5)
    ap.add_argument("--w_loc", type=float, default=1.0)
    ap.add_argument("--w_prog", type=float, default=0.1)
    ap.add_argument("--w_div", type=float, default=0.05)
    ap.add_argument("--init", default=os.path.join(CK, "nanovla_c/vlac.pt"))
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(CK, "nanovla_g/vlag.pt"))
    a = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(TB, a.run)); tb.add_text("config", "\n".join(f"{k}: {v}" for k, v in vars(a).items()))
    R.T_MAX = a.window
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"; t0 = time.time()
    P = v2.prepare(a.bench, a.n_val, a.seed); P["train_eps"] = sorted({int(e) for e, _ in P["train"]})
    zf = np.load(os.path.join(v2.ROOT, a.bench, "actions", json.load(open(os.path.join(v2.ROOT, a.bench, "repos.json")))[0].replace("/", "__") + ".npz"))
    P["pol"] = {int(k[2:].split("_")[0]): zf[k].astype(bool) for k in zf.files if k.endswith("_pol")}
    info = dict(episodes=len(P["train_eps"]), val_eps=P["n_test_eps"], frames_with_labels=len(P.get("loc", {})), dagger_eps=len(P["pol"]), prep_s=round(time.time() - t0))
    print(json.dumps(info), flush=True); tb.add_text("data", json.dumps(info))
    model, tok = build(dev, P["st"], a.init)
    for n, p in model.vis.v.named_parameters():
        if n.startswith(("proj.", "pos", "cls", "view_emb")) or any(n.startswith(f"blocks.{i}.") for i in range(a.freeze_blocks)):
            p.requires_grad = False
    for p in model.zproj.parameters():
        p.requires_grad = False
    is_enc = lambda n: n.startswith("vis.") or n.startswith("t.")
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    opt = torch.optim.AdamW([{"params": head, "base": a.lr}, {"params": enc, "base": a.enc_lr}], weight_decay=0.05, fused=True)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    dl = torch.utils.data.DataLoader(R.SeqData(P, P["train_eps"], True, "own"), batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=R.collate,
                                     drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    W = dict(fwd=0.5, phase=0.1, loc=a.w_loc, prog=a.w_prog, div=a.w_div)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    step, hist, t1 = 0, [], time.time()
    while step < a.steps:
        for b in dl:
            step += 1
            warm, dec = min(1, step / 500), 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for g in opt.param_groups:
                g["lr"] = g["base"] * warm * dec
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = run_window(model, b, tok, dev, True, a.p_hist, a.p_mem, a.p_plan, p_tf=a.p_tf)
                loss, parts, le = losses(model, b, tok, dev, o, W)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(model)
            hist.append({k: float(v) for k, v in parts.items()} | dict(grad_norm=float(gn), loc_err_cm=100 * float(le) if le is not None else np.nan))
            if step % 50 == 0:
                sps = (time.time() - t1) / 50; t1 = time.time()
                for k in hist[-1]:
                    tb.add_scalar(f"train/{k}", float(np.nanmean([h[k] for h in hist[-50:]])), step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step); tb.add_scalar("speed/s_per_step", sps, step)
                if step % 200 == 0:
                    print(json.dumps(dict(step=step, **{k: round(float(np.nanmean([h[k] for h in hist[-200:]])), 4) for k in ("action", "localize", "loc_err_cm", "diversity")}, s_per_step=round(sps, 3))), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                ev = evaluate(ema.module, tok, P, dev)
                print(json.dumps(dict(step=step, **ev)), flush=True)
                for k, v in ev.items():
                    if v is not None:
                        tb.add_scalar(f"val/{k}", v, step)
                tb.flush()
                torch.save(dict(model=ema.module.state_dict(), arch="nanovla_g", st=P["st"], coef=P["coef"], step_frames=P["step"], bench=a.bench,
                                episodes=len(P["train_eps"]), steps=step, args=vars(a), **ev), a.out)
            if step >= a.steps:
                break
    tb.close(); print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
