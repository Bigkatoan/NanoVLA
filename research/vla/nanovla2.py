"""NanoVLA-2: wrist-camera-only VLA for the SO-101 Isaac tabletop.

Changes vs v1 (train_vla.VLA), each aimed at a closed-loop failure seen with v1:
  * history     : current wrist frame + the frame 0.4 s earlier (= one re-plan period), state + state change
                  -> the policy can tell "not yet grasped" from "carrying" and sees its own motion.
  * no state dropout (v1 zeroed the state in 50% of the samples).
  * full frame  : the 160x120 image is not centre-cropped (v1 lost 12% on each side); finer token grid.
  * fusion      : 3 transformer layers over [img_t, img_t-0.4s, text, state] (v1: 1 layer).
  * flow head   : flow-matching action decoder (rectified flow, 10 Euler steps) instead of deterministic L1
                  regression -> no averaging of distinct valid motions (the "hover" failure).
  * phase aux   : classifies approach / holding / released from the fused image tokens.
  * vision      : --vision own   = v1's ViT-Tiny (5.5M, Isaac-adapted), 160x128 input, 10x8 tokens
                  --vision dino  = DINOv2-S/14 + 4 registers (22M, timm, last 4 blocks trained), 168x126, 12x9 tokens
Targets as in v1: joints 0-4 (A - S_t) / dstd, gripper min-max to [-1, 1]; chunk H=16 at 10 Hz (1.6 s).

    python nanovla2.py --vision dino --steps 30000 --out ~/VLA/ckpt/nanovla2_dino/vla2.pt
"""
import argparse, json, math, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from torchvision.io import decode_jpeg
import fewshot as fs
from train_vla import build_model, H, MEAN, STD
from train_text import augment, clean

ROOT = os.path.expanduser("~/VLA/data")
CK = os.path.expanduser("~/VLA/ckpt")
HIST = 12            # history offset in control steps (30 Hz) = exec 4 chunk points x 3 steps = one re-plan period
SIZE = {"own": (128, 160), "dino": (126, 168)}
DINO = "vit_small_patch14_reg4_dinov2.lvd142m"


# ------------------------------------------------------------------ vision backbones
class OwnVision(nn.Module):
    def __init__(self, vit):
        super().__init__(); self.v, self.d = vit, vit.d

    def forward(self, x):
        _, p = self.v(x, torch.ones(len(x), dtype=torch.long, device=x.device))   # view 1 = wrist
        return p


class DinoVision(nn.Module):
    def __init__(self, train_blocks=4, pretrained=True):
        super().__init__()
        import timm
        self.m = timm.create_model(DINO, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.d, self.npre = self.m.embed_dim, self.m.num_prefix_tokens
        nb = len(self.m.blocks)
        for n, p in self.m.named_parameters():
            p.requires_grad = n.startswith("norm.") or any(n.startswith(f"blocks.{i}.") for i in range(nb - train_blocks, nb))

    def forward(self, x):
        return self.m.forward_features(x)[:, self.npre:]


def n_tokens(vision):
    h, w = SIZE[vision]; p = 16 if vision == "own" else 14
    return (h // p) * (w // p)


# ------------------------------------------------------------------ model
def t_embed(tau, d):
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=tau.device) / half)
    a = tau[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class NanoVLA2(nn.Module):
    def __init__(self, vis, tenc, n_tok, d=256, fuse_layers=3, dec_layers=4, heads=8, ff=1024, n_phase=3):
        super().__init__()
        self.vis, self.t, self.d = vis, tenc, d
        self.vproj, self.tproj = nn.Linear(vis.d, d), nn.Linear(tenc.d, d)
        self.sproj = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.ttype = nn.Embedding(4, d)                       # img_t, img_prev, text, state
        self.vpos = nn.Parameter(torch.zeros(1, n_tok, d)); nn.init.trunc_normal_(self.vpos, std=0.02)
        L = lambda: nn.TransformerEncoderLayer(d, heads, ff, 0.1, batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(L(), fuse_layers, enable_nested_tensor=False)
        self.a_in = nn.Linear(6, d)
        self.a_pos = nn.Parameter(torch.zeros(1, H, d)); nn.init.trunc_normal_(self.a_pos, std=0.02)
        self.tmlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, ff, 0.1, batch_first=True, norm_first=True), dec_layers)
        self.a_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 6))
        self.phase = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_phase))
        self.n_tok = n_tok

    def encode(self, img, prev, state, ids):
        """img, prev [B,3,h,w] normalised; state [B,12] (normalised state, change since t-0.4 s); ids [B,L]"""
        B = img.shape[0]
        v = self.vproj(self.vis(torch.cat([img, prev])))
        vn, vp = v[:B], v[B:]
        te, _, tpad = self.t(ids)
        mem = torch.cat([vn + self.vpos + self.ttype.weight[0], vp + self.vpos + self.ttype.weight[1],
                         self.tproj(te) + self.ttype.weight[2], (self.sproj(state) + self.ttype.weight[3]).unsqueeze(1)], 1)
        z = torch.zeros(B, self.n_tok, dtype=torch.bool, device=img.device)
        pad = torch.cat([z, z, tpad, torch.zeros(B, 1, dtype=torch.bool, device=img.device)], 1)
        mem = self.fuse(mem, src_key_padding_mask=pad)
        return mem, pad

    def velocity(self, mem, pad, x, tau):
        h = self.a_in(x) + self.a_pos + self.tmlp(t_embed(tau, self.d))[:, None]
        return self.a_out(self.dec(h, mem, memory_key_padding_mask=pad))

    def phase_logits(self, mem):
        return self.phase(mem[:, : self.n_tok].mean(1))

    @torch.no_grad()
    def sample(self, mem, pad, steps=10, noise=None):
        x = torch.randn(mem.shape[0], H, 6, device=mem.device) if noise is None else noise
        for k in range(steps):
            tau = torch.full((x.shape[0],), k / steps, device=x.device)
            x = x + self.velocity(mem, pad, x, tau) / steps
        return x


def flow_loss(model, mem, pad, tgt, valid):
    """rectified flow: x_tau = (1-tau) noise + tau action, predict (action - noise); logit-normal tau"""
    B = tgt.shape[0]
    tau = torch.sigmoid(torch.randn(B, device=tgt.device))
    x0 = torch.randn_like(tgt)
    xt = (1 - tau)[:, None, None] * x0 + tau[:, None, None] * tgt
    v = model.velocity(mem, pad, xt, tau)
    m = valid.unsqueeze(-1).float()
    return (((v.float() - (tgt - x0)) ** 2) * m).sum() / (m.sum() * 6 + 1e-6)


def build(vision, dev, v1_ckpt=os.path.join(CK, "vla_isaac_wrist_v1/vla.pt")):
    """text encoder (+ own ViT) initialised from the Isaac-adapted v1 checkpoint"""
    s = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    v1, tok = build_model(s["n_emb"], os.path.join(CK, "align/align.pt"), "cpu")
    v1.load_state_dict(s["model"])
    vis = OwnVision(v1.v) if vision == "own" else DinoVision()
    return NanoVLA2(vis, v1.t, n_tokens(vision)).to(dev), tok


# ------------------------------------------------------------------ data
def phases(A):
    """per control step, from the gripper command (the jaw starts CLOSED at home, opens to approach):
    0 = approaching (until the jaw closes on the object after its first opening), 1 = holding, 2 = released.
    Episodes whose jaw never opens (push / topple with a closed jaw) stay 0."""
    g = A[:, 5]; hi, lo = np.percentile(g, 95), np.percentile(g, 5)
    ph = np.zeros(len(g), np.int64)
    if hi - lo < 0.05:
        return ph
    mid = (hi + lo) / 2
    op = np.nonzero(g > mid)[0]
    if not len(op):
        return ph
    cl = np.nonzero(g[op[0]:] < mid)[0]
    if not len(cl):
        return ph
    t0 = op[0] + cl[0]
    ro = np.nonzero(g[t0:] > mid)[0]
    t1 = t0 + ro[0] if len(ro) else len(g)
    ph[t0:t1] = 1; ph[t1:] = 2
    return ph


def prepare(bench, n_val, seed):
    """fewshot.prepare (wrist only) for actions / stats / instructions, plus a packed buffer of ALL wrist frames for history lookup"""
    fs.ROOT_BENCH = os.path.join(ROOT, bench)
    repo = json.load(open(os.path.join(ROOT, bench, "repos.json")))[0]
    def fs_count(repo):                         # = isaac_pretrain.fs_count (that module pulls in MuJoCo, absent from the Isaac env)
        name = repo.replace("/", "__"); z = np.load(os.path.join(fs.ROOT_BENCH, "actions", name + ".npz"))
        Fr = pd.read_parquet(os.path.join(fs.ROOT_BENCH, "frames", name + ".parquet"), columns=["ep", "fps"])
        fps = int(Fr.fps.iloc[0]); have = set(Fr.ep)
        return sum(1 for k in z.files if k.endswith("_action") and len(z[k]) > 2 * fps and int(k[2:].split("_")[0]) in have)
    P = fs.prepare(repo, fs_count(repo) - n_val, n_val, seed, cams="wrist")
    fpath = os.path.join(fs.ROOT_BENCH, "frames", repo.replace("/", "__") + ".parquet")
    import pyarrow.parquet as pq
    has_loc = "loc" in pq.read_schema(fpath).names
    Fr = pd.read_parquet(fpath, columns=["ep", "cam", "fidx", "jpeg"] + (["loc"] if has_loc else []))
    Fr = Fr[(Fr.cam == "wrist") & Fr.ep.isin(set(P["acts"]))]
    if has_loc:                                  # GCT privileged labels per wrist frame (None for older data)
        P["loc"] = {(int(e), int(f)): np.asarray(l, np.float32) for e, f, l in zip(Fr.ep, Fr.fidx, Fr["loc"]) if isinstance(l, (list, np.ndarray)) and len(l) >= 6}
    lens = Fr.jpeg.map(len).to_numpy(); offs = np.r_[0, np.cumsum(lens)[:-1]]
    P["blob"] = np.frombuffer(b"".join(Fr.jpeg.tolist()), dtype=np.uint8)
    P["where"] = {(int(e), int(f)): (int(o), int(l)) for e, f, o, l in zip(Fr.ep, Fr.fidx, offs, lens)}
    P["fidx"] = {int(e): np.sort(g.fidx.to_numpy()) for e, g in Fr.groupby("ep")}
    P["phase"] = {e: phases(A) for e, (A, _) in P["acts"].items()}
    for k in ("train", "test"):
        P[k] = np.array([(e, f) for e, f, _, _ in P[k]], dtype=np.int64).reshape(-1, 2)
    return P


def history_index(fids, f):
    """dataset frame used as the 'previous' observation: latest frame <= f - HIST, else f itself (episode start)"""
    j = np.searchsorted(fids, f - HIST, side="right") - 1
    return int(fids[j]) if j >= 0 else f


class Data2(torch.utils.data.Dataset):
    def __init__(self, P, split, train, vision):
        self.S, self.P, self.train, self.size = P[split], P, train, SIZE[vision]
        h, w = self.size
        self.aug = T.Compose([T.RandomApply([T.RandomRotation(8)], p=0.5),
                              T.RandomResizedCrop((h, w), scale=(0.7, 1.0), ratio=(1.2, 1.45), antialias=True),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.8),
                              T.RandomGrayscale(p=0.05),
                              T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
                              T.RandomApply([T.JPEG((30, 90))], p=0.3)])

    def __len__(self):
        return len(self.S)

    def _jpg(self, e, f):
        o, l = self.P["where"][(e, f)]
        return decode_jpeg(torch.from_numpy(self.P["blob"][o: o + l].copy()))

    def __getitem__(self, i):
        e, f = (int(x) for x in self.S[i])
        P, st = self.P, self.P["st"]
        A, S = P["acts"][e]
        fp = history_index(P["fidx"][e], f)
        cur, prev = self._jpg(e, f), self._jpg(e, fp)
        if self.train:
            cur, prev = self.aug(tv_tensors.Image(torch.stack([cur, prev]))).as_subclass(torch.Tensor)   # same crop / colour for both frames
        else:
            cur, prev = (T.functional.resize(x, list(self.size), antialias=True) for x in (cur, prev))
        cur, prev = ((x.float() / 255 - MEAN) / STD for x in (cur, prev))
        if self.train and random.random() < 0.3:
            n = random.uniform(0.02, 0.1); cur = cur + torch.randn_like(cur) * n; prev = prev + torch.randn_like(prev) * n
        sn = (S[f] - st["s_mean"]) / st["s_std"]
        ds = (S[f] - S[fp]) / st["s_std"] * 5
        idx = f + P["step"] * np.arange(1, H + 1); valid = idx < len(A); idx = np.minimum(idx, len(A) - 1)
        tgt = np.empty((H, 6), np.float32)
        tgt[:, :5] = (A[idx, :5] - S[f, :5]) / st["dstd"]
        tgt[:, 5] = (A[idx, 5] - st["g_lo"]) / (st["g_hi"] - st["g_lo"] + 1e-6) * 2 - 1
        lab = P["task"][e]
        if self.train and random.random() < 0.3:
            lab = augment(lab, random)
        return dict(img=cur, prev=prev, state=torch.from_numpy(np.clip(np.r_[sn, ds], -5, 5).astype(np.float32)),
                    tgt=torch.from_numpy(np.clip(tgt, -10, 10)), valid=torch.from_numpy(valid), phase=torch.tensor(int(P["phase"][e][min(f, len(A) - 1)])), label=lab)


def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in b[0] if k != "label"}
    out["label"] = [x["label"] for x in b]
    return out


# ------------------------------------------------------------------ train / eval
@torch.no_grad()
def evaluate(model, tok, P, vision, dev, max_batches=60):
    dl = torch.utils.data.DataLoader(Data2(P, "test", False, vision), batch_size=128, shuffle=True, num_workers=4, collate_fn=collate,
                                     generator=torch.Generator().manual_seed(0))
    model.eval(); tot = hold = acc = n = 0.0
    for i, b in enumerate(dl):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem, pad = model.encode(b["img"].to(dev), b["prev"].to(dev), b["state"].to(dev), tok(b["label"]).to(dev))
            p = model.sample(mem, pad, noise=torch.zeros(len(b["img"]), H, 6, device=dev)).float()
            ph = model.phase_logits(mem).argmax(-1).cpu()
        tg, va = b["tgt"].to(dev), b["valid"].to(dev).unsqueeze(-1).float()
        l = ((p - tg).abs() * va).sum((1, 2)) / (va.sum((1, 2)) * 6)
        z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
        lh = ((z - tg).abs() * va).sum((1, 2)) / (va.sum((1, 2)) * 6)
        tot += l.sum().item(); hold += lh.sum().item(); acc += (ph == b["phase"]).sum().item(); n += len(p)
    model.train()
    return dict(val_l1=round(tot / n, 4), val_hold=round(hold / n, 4), val_rel=round(tot / hold, 4), phase_acc=round(acc / n, 3), n_val=int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", default="dino", choices=tuple(SIZE))
    ap.add_argument("--bench", default="isaac_all_v2")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=3e-5)
    ap.add_argument("--phase_w", type=float, default=0.1)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init", default="", help="continue from a NanoVLA-2 checkpoint (e.g. DAgger rounds)")
    ap.add_argument("--out", default=os.path.join(CK, "nanovla2/vla2.pt"))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    t0 = time.time()
    P = prepare(a.bench, a.n_val, a.seed)
    print(json.dumps(dict(vision=a.vision, episodes=P["n_train_eps"], val_eps=P["n_test_eps"], samples=len(P["train"]), frames=len(P["where"]), prep_s=round(time.time() - t0))), flush=True)
    model, tok = build(a.vision, dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
    is_enc = lambda n: n.startswith("vis.") or n.startswith("t.")
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    print(json.dumps(dict(params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2), trainable_M=round(sum(p.numel() for p in enc + head) / 1e6, 2),
                          head_M=round(sum(p.numel() for p in head) / 1e6, 2))), flush=True)
    opt = torch.optim.AdamW([{"params": head, "base": a.lr}, {"params": enc, "base": a.enc_lr}], weight_decay=0.05)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    dl = torch.utils.data.DataLoader(Data2(P, "train", True, a.vision), batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                                     drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    step, hist, t1 = 0, [], time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    while step < a.steps:
        for b in dl:
            step += 1
            warm, dec = min(1, step / 1000), 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for g in opt.param_groups:
                g["lr"] = g["base"] * warm * dec
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mem, pad = model.encode(b["img"].to(dev, non_blocking=True), b["prev"].to(dev, non_blocking=True), b["state"].to(dev), tok(b["label"]).to(dev))
                lf = flow_loss(model, mem, pad, b["tgt"].to(dev), b["valid"].to(dev))
                lp = F.cross_entropy(model.phase_logits(mem).float(), b["phase"].to(dev))
            loss = lf + a.phase_w * lp
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(model)
            hist.append((lf.item(), lp.item()))
            if step % 250 == 0:
                h = np.array(hist[-250:])
                print(json.dumps(dict(step=step, flow=round(float(h[:, 0].mean()), 4), phase=round(float(h[:, 1].mean()), 4), ips=round(250 * a.bs / (time.time() - t1), 1))), flush=True)
                t1 = time.time()
            if step % 5000 == 0 or step == a.steps:
                ev = evaluate(ema.module, tok, P, a.vision, dev)
                print(json.dumps(dict(step=step, **ev)), flush=True)
                torch.save(dict(model=ema.module.state_dict(), vision=a.vision, st=P["st"], coef=P["coef"], step_frames=P["step"], bench=a.bench,
                                episodes=P["n_train_eps"], steps=step, args=vars(a), **ev), a.out)
            if step >= a.steps:
                break
    print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
