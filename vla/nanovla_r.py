"""NanoVLA-R: recurrent wrist-camera VLA (user's design, completed).

Per episode
    text  -> text encoder -> text vector u            (computed once, cached while the instruction is unchanged)
    (u, joints_0) -> MLP -> LSTM initial state (h_0, c_0)
Per tick k (10 Hz = every camera frame; re-plan every tick; the image is encoded ONCE and never again)
    wrist image -> ViT-Tiny (5.5M, 160x128, 10x8 tokens; edge target Jetson Orin Nano 10-30 Hz) -> text-conditioned attention pooling (K queries) -> image vector v_k
    x_k = [v_k, joints_k, joints_k - joints_{k-1}, u]  -> 2-layer LSTM -> h_k   (memory of the whole episode)
    action head: flow matching over a 16 x 6 chunk (1.6 s), cross-attending to [h_k, v_k, joints_k, u]
    aux: phase (approach / holding / released) from h_k
Additions to the original idea (why):
    * pooling queries are conditioned on u, so the single vector keeps the instructed object (and where it is:
      the tokens carry 2-D position embeddings) instead of a generic scene summary
    * joints enter every tick (not only h_0): proprioception is the cheapest, most reliable signal
    * u enters every tick and the action head: the instruction cannot fade from the LSTM over long episodes
    * flow-matching chunk head: multi-modal motions are sampled, not averaged (v1's L1 head hovered)
Training: windows of T_MAX ticks (half from the episode start as at inference, half mid-episode), BPTT over the window,
one tick per recorded frame (10 Hz).

    python nanovla_r.py --steps 15000 --out ~/VLA/ckpt/nanovla_r/vlar.pt
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from torchvision.io import decode_jpeg
from train_vla import H, MEAN, STD
from train_text import augment
import nanovla2 as v2

CK = os.path.expanduser("~/VLA/ckpt")
TICK = 1                 # dataset frames (10 Hz) per tick: the LSTM ticks at 10 Hz (= camera rate of the recorded data)
T_MAX = 32               # ticks per training window (3.2 s); half of the windows start at the episode's first frame


VARIANTS = {
    # ours   : text-conditioned pooling, LSTM init from (text, joints), transformer flow decoder over [h, v, joints, u]
    "ours": dict(pool_text=True, init_joints=True, head="dec"),
    # diagram: the user's block diagram - image encoder -> image vector (no text), Init Inform from the text encoder only,
    #          observation vector (LSTM state) <- [image vector, joint states, text vector], observation vector -> MLP -> action
    #          (the MLP is trained as a flow-matching velocity net so distinct valid motions are not averaged)
    "diagram": dict(pool_text=False, init_joints=False, head="mlp"),
}


class NanoVLAR(nn.Module):
    def __init__(self, vis, tenc, n_tok, d=256, K=4, dv=512, hid=512, lstm_layers=2, dec_layers=3, heads=8, ff=768, n_phase=3,
                 pool_text=True, init_joints=True, head="dec", mlp_hidden=1024):
        super().__init__()
        self.vis, self.t, self.d, self.K, self.hid, self.nl = vis, tenc, d, K, hid, lstm_layers
        self.pool_text, self.init_joints, self.head = pool_text, init_joints, head
        self.tproj = nn.Sequential(nn.Linear(tenc.d, d), nn.LayerNorm(d))
        self.vproj = nn.Linear(vis.d, d)
        self.vpos = nn.Parameter(torch.zeros(1, n_tok, d)); nn.init.trunc_normal_(self.vpos, std=0.02)
        self.pool_q = nn.Parameter(torch.randn(1, K, d) * 0.02)
        self.qfilm = nn.Linear(d, 2 * d)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_n1, self.pool_n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.pool_mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.vvec = nn.Sequential(nn.Linear(K * d, dv), nn.LayerNorm(dv))
        self.init = nn.Sequential(nn.Linear(d + (6 if init_joints else 0), hid), nn.GELU(), nn.Linear(hid, 2 * lstm_layers * hid))
        self.lstm = nn.LSTM(dv + 12 + d, hid, lstm_layers, batch_first=True)
        self.tmlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        if head == "dec":
            self.c_h, self.c_v = nn.Linear(hid, d), nn.Linear(dv, d)
            self.c_s = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
            self.ctype = nn.Parameter(torch.zeros(1, 4, d)); nn.init.trunc_normal_(self.ctype, std=0.02)
            self.a_in = nn.Linear(6, d)
            self.a_pos = nn.Parameter(torch.zeros(1, H, d)); nn.init.trunc_normal_(self.a_pos, std=0.02)
            self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, ff, 0.1, batch_first=True, norm_first=True), dec_layers)
            self.a_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 6))
        else:                                    # MLP head: [observation vector, noisy chunk (16x6), time] -> velocity (16x6)
            self.mlp = nn.Sequential(nn.LayerNorm(hid + H * 6 + d), nn.Linear(hid + H * 6 + d, mlp_hidden), nn.GELU(),
                                     nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
                                     nn.Linear(mlp_hidden, H * 6))
        self.phase = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, n_phase))

    # -- pieces used both in training and step-by-step inference
    def text_vec(self, ids):
        _, pooled, _ = self.t(ids)
        return self.tproj(pooled)

    def image_vec(self, img, u):
        """img [N,3,h,w], u [N,d] -> [N,dv]"""
        tok = self.vproj(self.vis(img)) + self.vpos
        if self.pool_text:
            g, b = self.qfilm(u).chunk(2, -1)
            q = self.pool_q * (1 + g[:, None]) + b[:, None]
        else:
            q = self.pool_q.expand(len(img), -1, -1) * 1.0
        a, _ = self.pool(self.pool_n1(q), tok, tok, need_weights=False)
        q = q + a
        q = q + self.pool_mlp(self.pool_n2(q))
        return self.vvec(q.flatten(1))

    def init_state(self, u, s0):
        z = self.init(torch.cat([u, s0[:, :6]], -1) if self.init_joints else u).view(-1, 2, self.nl, self.hid)
        return z[:, 0].transpose(0, 1).contiguous().tanh().float(), z[:, 1].transpose(0, 1).contiguous().float()

    def cond(self, h, v, s, u):
        if self.head == "mlp":
            return h                                                     # the observation vector alone drives the MLP
        return torch.stack([self.c_h(h), self.c_v(v), self.c_s(s), u], 1) + self.ctype

    def velocity(self, c, x, tau):
        te = self.tmlp(v2.t_embed(tau, self.d))
        if self.head == "mlp":
            return self.mlp(torch.cat([c, x.flatten(1), te], -1)).view(-1, H, 6)
        hh = self.a_in(x) + self.a_pos + te[:, None]
        return self.a_out(self.dec(hh, c))

    @torch.no_grad()
    def sample(self, c, steps=10, noise=None):
        x = torch.randn(c.shape[0], H, 6, device=c.device) if noise is None else noise
        for k in range(steps):
            x = x + self.velocity(c, x, torch.full((x.shape[0],), k / steps, device=x.device)) / steps
        return x

    def forward_seq(self, imgs, states, ids):
        """imgs [B,T,3,h,w], states [B,T,12] -> cond tokens [B,T,4,d], h [B,T,hid]"""
        B, Tn = states.shape[:2]
        u = self.text_vec(ids)
        v = self.image_vec(imgs.flatten(0, 1), u.repeat_interleave(Tn, 0)).view(B, Tn, -1)
        h0 = self.init_state(u, states[:, 0])
        h, _ = self.lstm(torch.cat([v, states, u[:, None].expand(-1, Tn, -1)], -1).float(), h0)
        c = self.cond(h.flatten(0, 1), v.flatten(0, 1), states.flatten(0, 1), u.repeat_interleave(Tn, 0))
        return c.view(B, Tn, *c.shape[1:]), h

    # -- one control tick (policy server)
    @torch.no_grad()
    def step(self, img, s, u, hc, steps=10, noise=None):
        v = self.image_vec(img, u)
        h, hc = self.lstm(torch.cat([v, s, u], -1)[:, None].float(), hc)
        h = h[:, 0]
        return self.sample(self.cond(h, v, s, u), steps, noise), hc


def build(dev, vision="own", variant="ours", v1_ckpt=os.path.join(CK, "vla_isaac_wrist_v1/vla.pt")):
    """vision='own': v1's Isaac-adapted ViT-Tiny (5.5M, 160x128 -> 10x8 tokens) - the edge (Jetson Orin Nano) target"""
    s = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    v1, tok = v2.build_model(s["n_emb"], os.path.join(CK, "align/align.pt"), "cpu")
    v1.load_state_dict(s["model"])
    vis = v2.OwnVision(v1.v) if vision == "own" else v2.DinoVision()
    return NanoVLAR(vis, v1.t, v2.n_tokens(vision), **VARIANTS[variant]).to(dev), tok


# ------------------------------------------------------------------ data: one item = one episode from its first frame
class SeqData(torch.utils.data.Dataset):
    def __init__(self, P, eps, train, vision="own"):
        self.P, self.eps, self.train = P, list(eps), train
        self.size = v2.SIZE[vision]

    def __len__(self):
        return len(self.eps)

    def ticks(self, e, off):
        A, _ = self.P["acts"][e]; step = self.P["step"]
        f = [int(x) for x in self.P["fidx"][e][off::TICK] if x + step < len(A)]
        if self.train and len(f) > T_MAX and random.random() < 0.5:      # mid-episode window (LSTM starts from init(u, joints))
            k = random.randrange(len(f) - T_MAX + 1); f = f[k:]
        return f[:T_MAX]

    def __getitem__(self, i):
        e = self.eps[i]; P, st = self.P, self.P["st"]
        A, S = P["acts"][e]
        fs_ = self.ticks(e, random.randrange(TICK) if self.train else 0)
        if "jpg" in P and (e, fs_[0]) in P["jpg"]:          # online buffer (GCT stage 2): frames kept as bytes per (episode, frame)
            ims = torch.stack([decode_jpeg(torch.frombuffer(bytearray(P["jpg"][(e, f)]), dtype=torch.uint8)) for f in fs_])
        else:
            ims = torch.stack([decode_jpeg(torch.from_numpy(P["blob"][o: o + l].copy())) for o, l in (P["where"][(e, f)] for f in fs_)])   # uint8, augmented on the GPU
        idx = np.array(fs_)
        jit = self.train and P.get("jitter", 0) > 0
        # timestamp noise (user's idea): joint reading +-jitter control steps away from the image, action labels 0..jitter late
        si = np.clip(idx + (np.random.randint(-P["jitter"], P["jitter"] + 1, len(idx)) if jit else 0), 0, len(S) - 1)
        ja = np.random.randint(0, P["jitter"] + 1, len(idx)) if jit else np.zeros(len(idx), int)
        sn = (S[si] - st["s_mean"]) / st["s_std"]
        f0 = int(P["fidx"][e][0]); prev = idx - P["step"] * TICK                # joint change over one tick; 0 at the episode start
        ds = np.where((prev >= f0)[:, None], (S[si] - S[np.clip(prev + (si - idx), 0, len(S) - 1)]) / st["s_std"] * 5, 0.0)
        fut = idx[:, None] + ja[:, None] + P["step"] * np.arange(1, H + 1)[None]
        valid = fut < len(A); fut = np.minimum(fut, len(A) - 1)
        pol = P.get("pol", {}).get(e)
        if pol is not None:                      # DAgger data: commands issued by the policy are not labels
            valid &= ~pol[fut]
        tgt = np.empty((len(idx), H, 6), np.float32)
        if P.get("act_mode") == "vel":            # velocity: increment of the command over each 0.1 s step (first one vs the current command)
            prv = np.maximum(fut - P["step"], 0)
            tgt[..., :5] = (A[fut, :5] - A[prv, :5]) / P["vstd"]
        else:
            tgt[..., :5] = (A[fut, :5] - S[si, None, :5]) / st["dstd"]
        tgt[..., 5] = (A[fut, 5] - st["g_lo"]) / (st["g_hi"] - st["g_lo"] + 1e-6) * 2 - 1
        lab = P["task"][e]
        if self.train and random.random() < 0.3:
            lab = augment(lab, random)
        out = dict(img=ims, state=torch.from_numpy(np.clip(np.c_[sn, ds], -5, 5).astype(np.float32)), tgt=torch.from_numpy(np.clip(tgt, -10, 10)),
                   valid=torch.from_numpy(valid), phase=torch.from_numpy(P["phase"][e][np.minimum(idx, len(A) - 1)]), label=lab,
                   fidx=torch.from_numpy(idx.astype(np.int64)), first=torch.tensor(int(fs_[0]) == int(P["fidx"][e][0])))
        if "loc" in P:                           # GCT privileged labels (wrist-camera frame, metres); missing -> masked
            L = [P["loc"].get((e, int(f))) for f in idx]
            out["loc"] = torch.from_numpy(np.array([l[:6] if l is not None else [0.0] * 6 for l in L], np.float32))
            out["loc_ok"] = torch.tensor([l is not None for l in L])
            out["prog"] = torch.from_numpy((idx / max(1, len(A) - 1)).astype(np.float32))
        return out


def collate(b):
    Tn = max(len(x["state"]) for x in b)
    pad = lambda t: F.pad(t, [0, 0] * (t.dim() - 1) + [0, Tn - len(t)])
    seq = [k for k in ("img", "state", "tgt", "valid", "phase", "fidx", "loc", "loc_ok", "prog") if k in b[0]]
    out = {k: torch.stack([pad(x[k]) for x in b]) for k in seq}
    if "first" in b[0]:
        out["first"] = torch.stack([x["first"] for x in b])
    out["mask"] = torch.stack([torch.arange(Tn) < len(x["state"]) for x in b])
    out["label"] = [x["label"] for x in b]
    return out


_AUG = {}


def gpu_images(x, size, train):
    """x uint8 [B,T,3,120,160] on the GPU -> normalised [B,T,3,h,w]; train: one random augmentation per sequence"""
    if size not in _AUG:
        _AUG[size] = T.Compose([T.RandomApply([T.RandomRotation(8)], p=0.5),
                                T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(1.2, 1.45), antialias=True),
                                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.8),
                                T.RandomGrayscale(p=0.05),
                                T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3)])
    if train:
        x = torch.stack([_AUG[size](tv_tensors.Image(s)).as_subclass(torch.Tensor) for s in x])
    else:
        x = T.functional.resize(x.flatten(0, 1), list(size), antialias=True).view(*x.shape[:2], 3, *size)
    x = (x.float() / 255 - MEAN.to(x.device)) / STD.to(x.device)
    if train:
        x = x + torch.randn_like(x) * torch.where(torch.rand(len(x), 1, 1, 1, 1, device=x.device) < 0.3,
                                                  torch.empty(len(x), 1, 1, 1, 1, device=x.device).uniform_(0.02, 0.1), torch.zeros(1, device=x.device))
    return x


def losses(model, b, tok, dev, train=True):
    img = gpu_images(b["img"].to(dev, non_blocking=True), v2.SIZE[model.vision_name], train)
    c, h = model.forward_seq(img, b["state"].to(dev), tok(b["label"]).to(dev))
    m = b["mask"].to(dev)
    c, h = c[m], h[m]                                            # valid ticks only
    tgt, valid = b["tgt"].to(dev)[m], b["valid"].to(dev)[m]
    tau = torch.sigmoid(torch.randn(len(tgt), device=dev)); x0 = torch.randn_like(tgt)
    xt = (1 - tau)[:, None, None] * x0 + tau[:, None, None] * tgt
    vm = valid.unsqueeze(-1).float()
    lf = (((model.velocity(c, xt, tau).float() - (tgt - x0)) ** 2) * vm).sum() / (vm.sum() * 6 + 1e-6)
    lp = F.cross_entropy(model.phase(h).float(), b["phase"].to(dev)[m])
    return lf, lp, c, h, tgt, valid


@torch.no_grad()
def evaluate(model, tok, P, dev, max_eps=100):
    eps = sorted(e for e in P["acts"] if e not in P["train_eps"])[:max_eps]
    dl = torch.utils.data.DataLoader(SeqData(P, eps, False, model.vision_name), batch_size=4, num_workers=4, collate_fn=collate)
    model.eval(); tot = hold = acc = n = 0.0
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, c, h, tg, va = losses(model, b, tok, dev, train=False)
            p = model.sample(c, noise=torch.zeros(len(c), H, 6, device=dev)).float()
            ph = model.phase(h).argmax(-1)
        vm = va.unsqueeze(-1).float()
        z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
        tot += (((p - tg).abs() * vm).sum((1, 2)) / (vm.sum((1, 2)) * 6)).sum().item()
        hold += (((z - tg).abs() * vm).sum((1, 2)) / (vm.sum((1, 2)) * 6)).sum().item()
        acc += (ph.cpu() == b["phase"][b["mask"]]).sum().item(); n += len(p)
    model.train()
    return dict(val_l1=round(tot / n, 4), val_hold=round(hold / n, 4), val_rel=round(tot / hold, 4), phase_acc=round(acc / n, 3), n_val_ticks=int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="isaac_all_v2")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--vision", default="own", choices=("own", "dino"))
    ap.add_argument("--variant", default="ours", choices=tuple(VARIANTS))
    ap.add_argument("--bs", type=int, default=12, help="windows per batch")
    ap.add_argument("--freeze_blocks", type=int, default=4, help="own ViT: first blocks (+ patch embedding) kept frozen")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=3e-5)
    ap.add_argument("--phase_w", type=float, default=0.1)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default=os.path.join(CK, "nanovla_r/vlar.pt"))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"; t0 = time.time()
    P = v2.prepare(a.bench, a.n_val, a.seed)
    P["train_eps"] = sorted({int(e) for e, _ in P["train"]})
    print(json.dumps(dict(model="nanovla_r", episodes=len(P["train_eps"]), val_eps=P["n_test_eps"], frames=len(P["where"]), prep_s=round(time.time() - t0))), flush=True)
    model, tok = build(dev, a.vision, a.variant); model.vision_name = a.vision
    if a.vision == "own":
        for n, p in model.vis.v.named_parameters():
            if n.startswith(("proj.", "pos", "cls", "view_emb")) or any(n.startswith(f"blocks.{i}.") for i in range(a.freeze_blocks)):
                p.requires_grad = False
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
    is_enc = lambda n: n.startswith("vis.") or n.startswith("t.")
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    print(json.dumps(dict(params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2), trainable_M=round(sum(p.numel() for p in enc + head) / 1e6, 2),
                          new_M=round(sum(p.numel() for p in head) / 1e6, 2))), flush=True)
    opt = torch.optim.AdamW([{"params": head, "base": a.lr}, {"params": enc, "base": a.enc_lr}], weight_decay=0.05)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    dl = torch.utils.data.DataLoader(SeqData(P, P["train_eps"], True, a.vision), batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                                     drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    step, hist, t1 = 0, [], time.time()
    while step < a.steps:
        for b in dl:
            step += 1
            warm, dec = min(1, step / 500), 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for g in opt.param_groups:
                g["lr"] = g["base"] * warm * dec
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lf, lp, *_ = losses(model, b, tok, dev)
            loss = lf + a.phase_w * lp
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(model)
            hist.append((lf.item(), lp.item()))
            if step % 100 == 0:
                h = np.array(hist[-100:])
                print(json.dumps(dict(step=step, flow=round(float(h[:, 0].mean()), 4), phase=round(float(h[:, 1].mean()), 4), s_per_step=round((time.time() - t1) / 100, 3))), flush=True)
                t1 = time.time()
            if step % a.eval_every == 0 or step == a.steps:
                ev = evaluate(ema.module, tok, P, dev)
                print(json.dumps(dict(step=step, **ev)), flush=True)
                torch.save(dict(model=ema.module.state_dict(), arch="nanovla_r", vision=a.vision, variant=a.variant, st=P["st"], coef=P["coef"], step_frames=P["step"], tick=TICK,
                                bench=a.bench, episodes=len(P["train_eps"]), steps=step, args=vars(a), **ev), a.out)
            if step >= a.steps:
                break
    print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
