"""NanoVLA-C: closed-loop NanoVLA-M (2026-11-06). Every 0.1 s the model
  1. compares what it EXPECTED to what happened:   e_k = [s_k - s^_k , z(v_k) - z^_k]      (z = fixed projection of the image vector)
  2. re-thinks its previous plan instead of drawing a new one:
         plan_k = shift(plan_k-1) + MLP([v_k, s_k, u, m_k-1, e_k, shift(plan_k-1)])
     shift() drops the executed first point and re-references the remaining joint offsets to the new joint state
  3. predicts what the first action will lead to:  (s^_k+1, z^_k+1) = Forward([v_k, s_k, a_k, m_k-1])
  4. remembers what was done and how it went:      m_k = GRU(m_k-1, [v_k, s_k, a_k, e_k])
Sources as agreed: v (what I see), s (joints now), u (what to do, text, once per instruction), m (what was done).
Training runs the loop on the model's own plans (detached between frames) over windows of real episodes; the
memory receives the model's own action with a probability ramping 0 -> 1. TensorBoard logs to ~/VLA/tb/<run>.

    python nanovla_c.py --run closed_v1 --steps 20000 --out ~/VLA/ckpt/nanovla_c/vlac.pt
"""
import argparse, io, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import nanovla2 as v2
import nanovla_r as R
import nanovla_m as M
from train_vla import H

CK = os.path.expanduser("~/VLA/ckpt")
TB = os.path.expanduser("~/VLA/tb")
HM = M.H_M
PE = 128


class NanoVLAC(M.NanoVLAM):
    def __init__(self, vis, tenc, n_tok, ratio, d=256, dv=512, mem=512, hidden=1024, **kw):
        super().__init__(vis, tenc, n_tok, d=d, dv=dv, mem=mem, head="l1", hidden=hidden, **kw)
        self.register_buffer("ratio", torch.as_tensor(ratio, dtype=torch.float32))      # s_std / dstd (joints 0-4)
        self.zproj = nn.Linear(dv, PE)                                                  # fixed random projection: target space for the image prediction
        for p in self.zproj.parameters():
            p.requires_grad = False
        nin = dv + 12 + d + mem + 6 + PE + HM * 6
        self.mlp = nn.Sequential(nn.LayerNorm(nin), nn.Linear(nin, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, HM * 6))
        nf = dv + 12 + 6 + mem
        self.fwd = nn.Sequential(nn.LayerNorm(nf), nn.Linear(nf, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 6 + PE))
        self.gru = nn.GRUCell(dv + 12 + 6 + 6 + PE, mem)

    def shift(self, plan, sn_prev, sn):
        """drop the executed point, repeat the last, re-reference joint offsets from the old to the new joint state"""
        sh = torch.cat([plan[:, 1:], plan[:, -1:]], 1).clone()
        sh[..., :5] = sh[..., :5] + ((sn_prev - sn)[:, :5] * self.ratio)[:, None]
        return sh

    def tick(self, v, s12, u, m, plan_prev, pred_prev, sn_prev, first):
        """one closed-loop step. first [B] bool: episode start (no previous plan / prediction)
        -> plan [B,HM,6], new memory, prediction for the next frame (s^, z^), error e, z(v)"""
        sn = s12[:, :6]
        z = self.zproj(v).float()
        keep = (~first).float()[:, None]
        e = torch.cat([sn - pred_prev[:, :6], z - pred_prev[:, 6:]], -1) * keep
        sh = self.shift(plan_prev, sn_prev, sn) * keep[:, :, None]
        delta = self.mlp(torch.cat([v, s12, u, m.to(v.dtype), e.to(v.dtype), sh.flatten(1).to(v.dtype)], -1)).view(-1, HM, 6)
        plan = sh + delta.float()
        a = plan[:, 0]
        pred = self.fwd(torch.cat([v, s12, a.to(v.dtype), m.to(v.dtype)], -1)).float()
        m2 = self.gru(torch.cat([v, s12, a.to(v.dtype), e.to(v.dtype)], -1).float(), m.float())
        return plan, m2, pred, e, z

    def run_seq(self, imgs, states, ids, a_teacher=None, p_own=1.0):
        """imgs [B,T,3,h,w], states [B,T,12] -> plans [B,T,HM,6], preds [B,T,6+PE], z [B,T,PE], errors [B,T,6+PE], memories [B,T,mem]"""
        B, Tn = states.shape[:2]
        u = self.text_vec(ids)
        v = self.image_vec(imgs.flatten(0, 1)).view(B, Tn, -1)
        m = self.init_mem(u)
        plan = torch.zeros(B, HM, 6, device=imgs.device); pred = torch.zeros(B, 6 + PE, device=imgs.device)
        first = torch.ones(B, dtype=torch.bool, device=imgs.device); sn_prev = states[:, 0, :6]
        P_, Q_, Z_, E_, Mm = [], [], [], [], []
        for k in range(Tn):
            Mm.append(m)
            plan, m_new, pred, e, z = self.tick(v[:, k], states[:, k], u, m, plan.detach(), pred.detach(), sn_prev, first)
            if a_teacher is not None and p_own < 1.0:      # memory input: own action with prob p_own, else the demo's
                own = plan[:, 0]
                a = torch.where(torch.rand(B, 1, device=imgs.device) < p_own, own.detach(), a_teacher[:, k])
                ek = e
                m_new = self.gru(torch.cat([v[:, k], states[:, k], a.to(v.dtype), ek.to(v.dtype)], -1).float(), m.float())
            m = m_new
            P_.append(plan); Q_.append(pred); Z_.append(z); E_.append(e)
            sn_prev = states[:, k, :6]; first = torch.zeros_like(first)
        st = lambda L: torch.stack(L, 1)
        return st(P_), st(Q_), st(Z_), st(E_), st(Mm)

    @torch.no_grad()
    def step(self, img, s12, u, m, plan_prev, pred_prev, sn_prev, first):
        v = self.image_vec(img)
        plan, m2, pred, e, _ = self.tick(v, s12, u, m, plan_prev, pred_prev, sn_prev, first)
        return plan, m2, pred, e


def build(dev, st, init=os.path.join(CK, "nanovla_m_l1/vlam.pt"), v1_ckpt=os.path.join(CK, "vla_isaac_wrist_v1/vla.pt")):
    s = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    v1, tok = v2.build_model(s["n_emb"], os.path.join(CK, "align/align.pt"), "cpu")
    v1.load_state_dict(s["model"])
    model = NanoVLAC(v2.OwnVision(v1.v), v1.t, v2.n_tokens("own"), np.asarray(st["s_std"][:5]) / np.asarray(st["dstd"]))
    if init:          # reuse everything whose shape matches (encoders, pooling, task vector, memory init, phase head)
        src = torch.load(init, map_location="cpu", weights_only=False)["model"]; own = model.state_dict()
        ok = {k: w for k, w in src.items() if k in own and own[k].shape == w.shape}
        own.update(ok); model.load_state_dict(own)
        print(json.dumps(dict(init=init, loaded=len(ok), new=[k for k in own if k not in ok][:12])), flush=True)
    return model.to(dev), tok


def compute(model, b, tok, dev, train, p_own):
    img = R.gpu_images(b["img"].to(dev, non_blocking=True), v2.SIZE["own"], train)
    states = b["state"].to(dev); tgt = b["tgt"].to(dev)[:, :, :HM]; valid = b["valid"].to(dev)[:, :, :HM]; msk = b["mask"].to(dev)
    plans, preds, zs, errs, mems = model.run_seq(img, states, tok(b["label"]).to(dev), tgt[:, :, 0], p_own)
    vm = valid.unsqueeze(-1).float() * msk[:, :, None, None].float()
    w = torch.ones(6, device=dev); w[5] = 0.5
    la = ((plans - tgt).abs() * w * vm).sum() / (vm.sum() * 6 + 1e-6)
    nxt = msk[:, 1:] & msk[:, :-1]                                      # forward model: frame k predicts frame k+1
    ls = F.smooth_l1_loss(preds[:, :-1, :6][nxt], states[:, 1:, :6][nxt]) if nxt.any() else la * 0
    lz = F.mse_loss(preds[:, :-1, 6:][nxt], zs[:, 1:][nxt].detach()) if nxt.any() else la * 0
    lp = F.cross_entropy(model.phase(mems[msk]).float(), b["phase"].to(dev)[msk])
    return la, ls, lz, lp, plans, tgt, valid, msk, errs


@torch.no_grad()
def evaluate(model, tok, P, dev, max_eps=100):
    eps = sorted(e for e in P["acts"] if e not in P["train_eps"])[:max_eps]
    dl = torch.utils.data.DataLoader(R.SeqData(P, eps, False, "own"), batch_size=4, num_workers=4, collate_fn=R.collate)
    model.eval(); tot = hold = n = 0.0; fs_ = fz = 0.0; nb = 0; example = None
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            la, ls, lz, lp, plans, tg, va, msk, errs = compute(model, b, tok, dev, False, 1.0)
        p, tg, va = plans[msk].float(), tg[msk], va[msk]
        vm = va.unsqueeze(-1).float(); z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
        tot += (((p - tg).abs() * vm).sum((1, 2)) / (vm.sum((1, 2)) * 6)).sum().item()
        hold += (((z - tg).abs() * vm).sum((1, 2)) / (vm.sum((1, 2)) * 6)).sum().item(); n += len(p)
        fs_ += ls.item(); fz += lz.item(); nb += 1
        if example is None:
            T0 = int(msk[0].sum()); example = (plans[0, :T0, 0].float().cpu().numpy(), b["tgt"][0, :T0, 0].numpy(), errs[0, :T0, :6].float().abs().mean(-1).cpu().numpy())
    model.train()
    return dict(val_l1=round(tot / n, 4), val_hold=round(hold / n, 4), val_rel=round(tot / hold, 4), val_fwd_state=round(fs_ / nb, 4), val_fwd_img=round(fz / nb, 4), n_val_ticks=int(n)), example


def figure(example):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pa, ta, er = example
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for j, c in enumerate(["#4cc9f0", "#80ed99", "#ffd166", "#ff8a3d", "#b388ff", "#f72585"]):
        ax[0].plot(ta[:, j], color=c, lw=4, alpha=.3); ax[0].plot(pa[:, j], color=c, lw=1.5)
    ax[0].set_title("first action per frame: label (thick) vs closed-loop prediction (thin), normalised")
    ax[1].plot(er, color="#e8590c"); ax[1].set_title("|expected - observed joints| fed back to the model"); ax[1].set_xlabel("frame (0.1 s)")
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); plt.close(fig)
    from PIL import Image
    return np.asarray(Image.open(buf).convert("RGB")).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="closed_v1")
    ap.add_argument("--bench", default="isaac_all_v2")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--fwd_w", type=float, default=0.5)
    ap.add_argument("--phase_w", type=float, default=0.1)
    ap.add_argument("--own_ramp", type=float, default=0.5, help="fraction of training over which the memory input moves from demo to own actions")
    ap.add_argument("--freeze_blocks", type=int, default=4)
    ap.add_argument("--init", default=os.path.join(CK, "nanovla_m_l1/vlam.pt"))
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(CK, "nanovla_c/vlac.pt"))
    a = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(TB, a.run))
    tb.add_text("config", "\n".join(f"{k}: {v}" for k, v in vars(a).items()))
    R.T_MAX = a.window
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"; t0 = time.time()
    P = v2.prepare(a.bench, a.n_val, a.seed); P["train_eps"] = sorted({int(e) for e, _ in P["train"]})
    zf = np.load(os.path.join(v2.ROOT, a.bench, "actions", json.load(open(os.path.join(v2.ROOT, a.bench, "repos.json")))[0].replace("/", "__") + ".npz"))
    P["pol"] = {int(k[2:].split("_")[0]): zf[k].astype(bool) for k in zf.files if k.endswith("_pol")}     # DAgger: policy-driven steps are not labels
    tb.add_text("data", f"bench {a.bench}: {len(P['train_eps'])} train episodes, {len(P['pol'])} DAgger episodes, {int(sum(v.sum() for v in P['pol'].values()))} policy-driven steps masked")
    print(json.dumps(dict(model="nanovla_c", episodes=len(P["train_eps"]), val_eps=P["n_test_eps"], prep_s=round(time.time() - t0))), flush=True)
    model, tok = build(dev, P["st"], a.init)
    for n, p in model.vis.v.named_parameters():
        if n.startswith(("proj.", "pos", "cls", "view_emb")) or any(n.startswith(f"blocks.{i}.") for i in range(a.freeze_blocks)):
            p.requires_grad = False
    is_enc = lambda n: n.startswith("vis.") or n.startswith("t.")
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    nparams = dict(params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2), trainable_M=round(sum(p.numel() for p in enc + head) / 1e6, 2))
    print(json.dumps(nparams), flush=True); tb.add_text("params", json.dumps(nparams))
    opt = torch.optim.AdamW([{"params": head, "base": a.lr}, {"params": enc, "base": a.enc_lr}], weight_decay=0.05)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    dl = torch.utils.data.DataLoader(R.SeqData(P, P["train_eps"], True, "own"), batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=R.collate,
                                     drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    step, hist, t1 = 0, [], time.time()
    while step < a.steps:
        for b in dl:
            step += 1
            warm, dec = min(1, step / 500), 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for g in opt.param_groups:
                g["lr"] = g["base"] * warm * dec
            p_own = min(1.0, step / (a.own_ramp * a.steps))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                la, ls, lz, lp, *_ = compute(model, b, tok, dev, True, p_own)
            loss = la + a.fwd_w * (ls + lz) + a.phase_w * lp
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(model)
            hist.append((la.item(), ls.item(), lz.item(), lp.item(), float(gn)))
            if step % 50 == 0:
                h = np.array(hist[-50:]); sps = (time.time() - t1) / 50; t1 = time.time()
                for i, k in enumerate(("train/action_l1", "train/forward_state", "train/forward_image", "train/phase_ce", "train/grad_norm")):
                    tb.add_scalar(k, h[:, i].mean(), step)
                tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step); tb.add_scalar("train/p_own_action_in_memory", p_own, step); tb.add_scalar("speed/s_per_step", sps, step)
                if step % 200 == 0:
                    print(json.dumps(dict(step=step, act=round(float(h[:, 0].mean()), 4), fwd_s=round(float(h[:, 1].mean()), 4), fwd_z=round(float(h[:, 2].mean()), 4),
                                          phase=round(float(h[:, 3].mean()), 4), p_own=round(p_own, 3), s_per_step=round(sps, 3))), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                ev, ex = evaluate(ema.module, tok, P, dev)
                print(json.dumps(dict(step=step, **ev)), flush=True)
                for k, v in ev.items():
                    if k != "n_val_ticks":
                        tb.add_scalar(f"val/{k}", v, step)
                tb.add_image("val/closed_loop_example", figure(ex), step); tb.flush()
                torch.save(dict(model=ema.module.state_dict(), arch="nanovla_c", st=P["st"], coef=P["coef"], step_frames=P["step"], bench=a.bench,
                                episodes=len(P["train_eps"]), steps=step, args=vars(a), **ev), a.out)
            if step >= a.steps:
                break
    tb.close()
    print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
