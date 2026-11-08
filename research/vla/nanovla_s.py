"""NanoVLA-S: state-based wrist-camera VLA (2026-11-07). No time-indexed actions.

Per frame (any rate, any jitter):
    inputs : wrist image, 6 joints, task vector u (text, cached), previous task state
    outputs: task state k (the expert's step: approach / descend / close / lift / carry / place / open / done ...),
             target point of that state in the wrist-camera frame (x, y, z, metres) + yaw still to turn, jaw open/close
    a non-learned servo (IK, speed-limited, 30 Hz) moves the gripper to the target; the target is re-estimated on
    every new frame (visual servoing), the state only changes when the observation shows an event.
Model: ViT-Tiny tokens (text-free image vector v = 8-query pooling) -> state head [v, joints, u, prev state]
       grounding: queries from [u, state] attend over the 80 tokens -> soft-argmax + features -> object / destination /
       target xyz (+ yaw, jaw)
Training is per frame (no BPTT). The previous-state input is corrupted (p=0.3 a neighbour state, p=0.1 unknown) so the
model must verify the state from the image instead of copying it; frames where the state changes weigh x5.

    python nanovla_s.py --run s1 --bench isaac_s_v1 --steps 15000 --out ~/VLA/ckpt/nanovla_s/vlas.pt
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.io import decode_jpeg
import nanovla2 as v2, nanovla_r as R
from train_vla import MEAN, STD
from train_text import augment

CK = os.path.expanduser("~/VLA/ckpt"); TB = os.path.expanduser("~/VLA/tb")
K = 8                     # task states (expert steps 0..6 + done); index K = "unknown" previous state
SIZE = v2.SIZE["own"]


class NanoVLAS(nn.Module):
    def __init__(self, vis, tenc, n_tok, d=256, Kq=8, dv=512, hidden=512):
        super().__init__()
        self.vis, self.t, self.d = vis, tenc, d
        self.tproj = nn.Sequential(nn.Linear(tenc.d, d), nn.LayerNorm(d))
        self.vproj = nn.Linear(vis.d, d)
        self.vpos = nn.Parameter(torch.zeros(1, n_tok, d)); nn.init.trunc_normal_(self.vpos, std=0.02)
        self.pool_q = nn.Parameter(torch.randn(1, Kq, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 8, batch_first=True)
        self.pool_n1 = nn.LayerNorm(d)
        self.vvec = nn.Sequential(nn.Linear(Kq * d, dv), nn.LayerNorm(dv))
        self.prev_emb = nn.Embedding(K + 1, d)                 # K = unknown
        self.st_emb = nn.Embedding(K, d)
        self.state = nn.Sequential(nn.LayerNorm(dv + 6 + 2 * d), nn.Linear(dv + 6 + 2 * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, K))
        # grounding: 3 queries (object, destination, target) from [u, state]
        self.gq = nn.Linear(2 * d, 3 * d); self.gk = nn.Linear(d, d)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 10), indexing="ij")
        self.register_buffer("grid", torch.stack([gx.flatten(), gy.flatten()], -1))
        self.gout = nn.Sequential(nn.LayerNorm(d + 2), nn.Linear(d + 2, 128), nn.GELU(), nn.Linear(128, 3))
        self.yaw = nn.Sequential(nn.LayerNorm(d + 2 * d), nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, 2))       # (cos, sin) of the yaw to turn
        self.jaw = nn.Sequential(nn.LayerNorm(dv + 2 * d), nn.Linear(dv + 2 * d, 128), nn.GELU(), nn.Linear(128, 1))

    def text_vec(self, ids):
        _, pooled, _ = self.t(ids)
        return self.tproj(pooled)

    def encode(self, img):
        tok = self.vproj(self.vis(img)) + self.vpos
        q = self.pool_q.expand(len(img), -1, -1) * 1.0
        a, _ = self.pool(self.pool_n1(q), tok, tok, need_weights=False)
        return tok, self.vvec((q + a).flatten(1))

    def state_logits(self, v, s6, u, prev):
        return self.state(torch.cat([v, s6.to(v.dtype), u, self.prev_emb(prev)], -1))

    def ground(self, tok, u, st):
        """-> xyz [N,3,3] (object, destination, target; metres, camera frame), logits [N,3,80], target feature [N,d]"""
        q = self.gq(torch.cat([u, self.st_emb(st)], -1)).view(-1, 3, self.d); k = self.gk(tok)
        logit = torch.einsum("nqd,ntd->nqt", q.float(), k.float()) / self.d ** 0.5
        att = logit.softmax(-1); xy = att @ self.grid; feat = att @ tok.float()
        xyz = self.gout(torch.cat([feat, xy], -1).to(tok.dtype)).float() * 0.1
        return xyz, logit, feat[:, 2]

    def heads(self, v, u, st, tfeat):
        e = self.st_emb(st)
        ys = self.yaw(torch.cat([tfeat.to(v.dtype), u, e], -1)).float()
        return torch.atan2(ys[:, 1], ys[:, 0]), self.jaw(torch.cat([v, u, e], -1)).float()[:, 0]

    @torch.no_grad()
    def act(self, img, s6, u, prev):
        """one frame -> dict(state, target xyz (camera frame), dyaw, jaw_open_prob, object, destination)"""
        tok, v = self.encode(img)
        st = self.state_logits(v, s6, u, prev).float().argmax(-1)
        xyz, _, tf = self.ground(tok, u, st)
        dyaw, jaw = self.heads(v, u, st, tf)
        return dict(state=st, target=xyz[:, 2], dyaw=dyaw, jaw_open=torch.sigmoid(jaw), obj=xyz[:, 0], dest=xyz[:, 1])


def build(dev, v1_ckpt=os.path.join(CK, "vla_isaac_wrist_v1/vla.pt"), init_vision=os.path.join(CK, "nanovla_g/vlag.pt")):
    s = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    v1, tok = v2.build_model(s["n_emb"], os.path.join(CK, "align/align.pt"), "cpu")
    v1.load_state_dict(s["model"])
    m = NanoVLAS(v2.OwnVision(v1.v), v1.t, v2.n_tokens("own"))
    if init_vision and os.path.exists(init_vision):         # reuse the GCT-trained ViT, text encoder and token projection
        src = torch.load(init_vision, map_location="cpu", weights_only=False)["model"]; own = m.state_dict()
        ok = {k: w for k, w in src.items() if k in own and own[k].shape == w.shape and k.split(".")[0] in ("vis", "t", "tproj", "vproj", "vpos")}
        own.update(ok); m.load_state_dict(own); print(json.dumps(dict(init_vision=init_vision, loaded=len(ok))), flush=True)
    return m.to(dev), tok


# ------------------------------------------------------------------ data: single frames
class FrameData(torch.utils.data.Dataset):
    def __init__(self, P, items, train):
        self.P, self.items, self.train = P, items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, f, fp = (int(x) for x in self.items[i]); P = self.P; st = P["st"]
        A, S = P["acts"][e]
        o, l = P["where"][(e, f)]
        img = decode_jpeg(torch.from_numpy(P["blob"][o: o + l].copy()))
        s6 = np.clip((S[f] - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32)
        L = P.get("loc", {}).get((e, f)); Lp = P.get("loc", {}).get((e, fp)) if fp >= 0 else None
        full = L is not None and len(L) >= 13
        y = dict(obj=np.zeros(3, np.float32), dest=np.zeros(3, np.float32), tgt=np.zeros(3, np.float32), dyaw=0.0, jaw=0.0, k=0, prev=K, has_loc=False, has_s=False)
        if L is not None:
            y.update(obj=L[0:3], dest=L[3:6], has_loc=True)
        if full:
            y.update(tgt=L[6:9], dyaw=float(L[9]), jaw=float(L[10]), k=int(min(L[11], K - 1)), has_s=True)
            prev = int(min(Lp[11], K - 1)) if (Lp is not None and len(Lp) >= 13) else 0
            if self.train:                                   # corrupt the previous state: the model must check it against the image
                r = random.random()
                if r < 0.15:
                    prev = int(np.clip(prev + random.choice((-1, 1)), 0, K - 1))
                elif r < 0.3:
                    prev = random.randrange(K)
                elif r < 0.4:
                    prev = K
            y["prev"] = prev; y["chg"] = int(prev != y["k"] and prev != K)
        lab = P["task"][e]
        if self.train and random.random() < 0.3:
            lab = augment(lab, random)
        return dict(img=img, s6=torch.from_numpy(s6), obj=torch.as_tensor(y["obj"], dtype=torch.float32), dest=torch.as_tensor(y["dest"], dtype=torch.float32),
                    tgt=torch.as_tensor(y["tgt"], dtype=torch.float32), dyaw=torch.tensor(y["dyaw"], dtype=torch.float32), jaw=torch.tensor(y["jaw"], dtype=torch.float32),
                    k=torch.tensor(y["k"]), prev=torch.tensor(y["prev"]), has_loc=torch.tensor(y["has_loc"]), has_s=torch.tensor(y["has_s"]),
                    chg=torch.tensor(y.get("chg", 0)), label=lab)


def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in b[0] if k != "label"}; out["label"] = [x["label"] for x in b]; return out


AUG = None


def gpu_img(x, train):
    global AUG
    if AUG is None:
        AUG = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.8, 1.0), ratio=(1.25, 1.4), antialias=True), T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.8),
                         T.RandomGrayscale(p=0.05), T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3)])
    x = torch.stack([AUG(i) for i in x]) if train else T.functional.resize(x, list(SIZE), antialias=True)
    return (x.float() / 255 - MEAN.to(x.device)) / STD.to(x.device)


def heat_ce(logit, xyz, ok):
    """soft cross-entropy of the attention logits [N,80] against the label's projected token cell (visible labels only)"""
    X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = 80 + 103.1 * X / Z.clamp(min=1e-3); v = 60 + 103.1 * Y / Z.clamp(min=1e-3)
    vis = ok & (Z > 0.03) & (u >= 0) & (u < 160) & (v >= 0) & (v < 120)
    if not vis.any():
        return logit.sum() * 0
    gx = (u / 16 - 0.5) / 4.5 - 1; gy = (v / 15 - 0.5) / 3.5 - 1
    grid = torch.stack([torch.linspace(-1, 1, 10).repeat(8), torch.linspace(-1, 1, 8).repeat_interleave(10)], -1).to(logit.device)
    d2 = (grid[None, :, 0] - gx[:, None]) ** 2 * 20.25 + (grid[None, :, 1] - gy[:, None]) ** 2 * 12.25
    q = torch.softmax(-d2 / 2.0, -1)
    return (-(q * logit.float().log_softmax(-1)).sum(-1) * vis).sum() / vis.sum()


def forward_losses(model, b, tok, dev, train, teacher_state=0.5):
    img = gpu_img(b["img"].to(dev, non_blocking=True), train)
    tokn, v = model.encode(img); u = model.text_vec(tok(b["label"]).to(dev))
    s6, prev, k = b["s6"].to(dev), b["prev"].to(dev), b["k"].to(dev)
    has_s, has_l = b["has_s"].to(dev), b["has_loc"].to(dev)
    sl = model.state_logits(v, s6, u, prev).float()
    pred_st = sl.argmax(-1)
    st_in = torch.where((torch.rand(len(k), device=dev) < teacher_state) & has_s, k, pred_st) if train else pred_st
    xyz, logit, tf = model.ground(tokn, u, st_in)
    dyaw, jaw = model.heads(v, u, st_in, tf)
    w = 1 + 4 * b["chg"].to(dev).float()
    L = {}
    L["state"] = (F.cross_entropy(sl, k, reduction="none") * w * has_s).sum() / (w * has_s).sum().clamp(min=1)
    L["target"] = (F.smooth_l1_loss(xyz[:, 2], b["tgt"].to(dev), beta=0.01, reduction="none").sum(-1) * has_s).sum() / has_s.sum().clamp(min=1) * 10
    L["objdest"] = (F.smooth_l1_loss(xyz[:, :2].flatten(1), torch.cat([b["obj"], b["dest"]], -1).to(dev), beta=0.01, reduction="none").sum(-1) * has_l).sum() / has_l.sum().clamp(min=1) * 10
    L["heat"] = (heat_ce(logit[:, 0], b["obj"].to(dev), has_l) + heat_ce(logit[:, 1], b["dest"].to(dev), has_l) + heat_ce(logit[:, 2], b["tgt"].to(dev), has_s)) * 0.2
    dy = (dyaw - b["dyaw"].to(dev) + math.pi) % (2 * math.pi) - math.pi
    L["yaw"] = (dy.abs() * has_s).sum() / has_s.sum().clamp(min=1)
    L["jaw"] = (F.binary_cross_entropy_with_logits(jaw, b["jaw"].to(dev), reduction="none") * has_s).sum() / has_s.sum().clamp(min=1)
    total = L["state"] + L["target"] + 0.5 * L["objdest"] + L["heat"] + 0.3 * L["yaw"] + 0.5 * L["jaw"]
    m = has_s
    met = dict(state_acc=float(((pred_st == k) & m).sum() / m.sum().clamp(min=1)), state_acc_change=float(((pred_st == k) & m & (b["chg"].to(dev) > 0)).sum() / ((b["chg"].to(dev) > 0) & m).sum().clamp(min=1)),
               target_err_cm=float(((xyz[:, 2] - b["tgt"].to(dev)).norm(dim=-1) * m).sum() / m.sum().clamp(min=1) * 100),
               obj_err_cm=float(((xyz[:, 0] - b["obj"].to(dev)).norm(dim=-1) * has_l).sum() / has_l.sum().clamp(min=1) * 100),
               jaw_acc=float((((jaw > 0) == (b["jaw"].to(dev) > 0.5)) & m).sum() / m.sum().clamp(min=1)), yaw_err_deg=float(L["yaw"]) * 57.3)
    return total, L, met



def event_labels(P):
    """Re-label the expert's timer phases as events (state = 'what to do next from here'):
      settle frames of step k (arrived at waypoint k)   -> step k+1
      grip frames of step k (arm holds while the jaw ramps) and settle frames -> target = step's waypoint seen from
        the hold pose (= the target of the step's first move frame), jaw = the step's command
    so a frame's (state, target, jaw) depends only on where the arm is, not on how long it has waited there."""
    loc = P.get("loc", {}); n_rel = 0
    for e, fr in P["fidx"].items():
        F = [int(f) for f in fr if (e, int(f)) in loc and len(loc[(e, int(f))]) >= 13]     # GCT data: object/dest only
        if not F:
            continue
        L = {f: np.array(loc[(e, f)], np.float32).copy() for f in F}
        first_move, jaw_of = {}, {}
        for f in F:
            k, ph = int(L[f][11]), int(L[f][12])
            if ph == 1 and k not in first_move:
                first_move[k] = L[f]
            if ph in (0, 1) and k not in jaw_of:
                jaw_of[k] = float(L[f][10])
        for f in F:
            x = L[f]; k, ph = int(x[11]), int(x[12])
            kk = k + 1 if ph == 2 else k
            if ph in (0, 2) and kk in first_move:
                x[6:10] = first_move[kk][6:10]; x[11] = kk; n_rel += 1
                if kk in jaw_of:
                    x[10] = jaw_of[kk]
            elif ph == 2 and kk not in first_move:              # settled at the last waypoint: done
                x[11] = kk
            loc[(e, f)] = x
    print(json.dumps(dict(event_relabelled=n_rel)), flush=True)

def items_of(P, eps):
    out = []
    for e in eps:
        fr = P["fidx"][e]
        for j, f in enumerate(fr):
            if (e, int(f)) in P["where"] and (e, int(f)) in P.get("loc", {}):       # frames without labels teach nothing here
                out.append((e, int(f), int(fr[j - 1]) if j else -1))
    return np.array(out, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s1"); ap.add_argument("--bench", default="isaac_s_v1"); ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--enc_lr", type=float, default=5e-5)
    ap.add_argument("--freeze_blocks", type=int, default=4); ap.add_argument("--n_val", type=int, default=80); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--out", default=os.path.join(CK, "nanovla_s/vlas.pt"))
    a = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(TB, a.run)); tb.add_text("config", "\n".join(f"{k}: {v}" for k, v in vars(a).items()))
    random.seed(0); np.random.seed(0); torch.manual_seed(0); dev = "cuda"; t0 = time.time()
    P = v2.prepare(a.bench, a.n_val, 0)
    tr_eps = sorted({int(e) for e, _ in P["train"]}); va_eps = sorted(e for e in P["acts"] if e not in set(tr_eps))
    tr, va = items_of(P, tr_eps), items_of(P, va_eps)
    for k_ in ("train", "test"):
        P.pop(k_, None)
    full = sum(1 for (e, f, _) in tr if P.get("loc", {}).get((e, f)) is not None and len(P["loc"][(e, f)]) >= 13)
    info = dict(train_frames=len(tr), val_frames=len(va), frames_with_state_labels=full, prep_s=round(time.time() - t0)); print(json.dumps(info), flush=True); tb.add_text("data", json.dumps(info))
    model, tok = build(dev)
    for n, p in model.vis.v.named_parameters():
        if n.startswith(("proj.", "pos", "cls", "view_emb")) or any(n.startswith(f"blocks.{i}.") for i in range(a.freeze_blocks)):
            p.requires_grad = False
    is_enc = lambda n: n.startswith("vis.") or n.startswith("t.")
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]; head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    opt = torch.optim.AdamW([{"params": head, "base": a.lr}, {"params": enc, "base": a.enc_lr}], weight_decay=0.05, fused=True)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    dl = torch.utils.data.DataLoader(FrameData(P, tr, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    vrng = np.random.default_rng(0); vsel = va[vrng.choice(len(va), min(4000, len(va)), replace=False)]
    dlv = torch.utils.data.DataLoader(FrameData(P, vsel, False), batch_size=256, num_workers=4, collate_fn=collate)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    step, hist, t1 = 0, [], time.time()
    while step < a.steps:
        for b in dl:
            step += 1
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1, step / 500) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, L, met = forward_losses(model, b, tok, dev, True)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(model)
            hist.append({**{k: float(v) for k, v in L.items()}, **met})
            if step % 50 == 0:
                for k in hist[-1]:
                    tb.add_scalar(f"train/{k}", float(np.mean([h[k] for h in hist[-50:]])), step)
                tb.add_scalar("speed/s_per_step", (time.time() - t1) / 50, step); t1 = time.time()
            if step % 500 == 0:
                print(json.dumps(dict(step=step, **{k: round(float(np.mean([h[k] for h in hist[-500:]])), 3) for k in ("state", "target", "state_acc", "state_acc_change", "target_err_cm", "obj_err_cm", "jaw_acc")})), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                ema.module.eval(); acc = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for bv in dlv:
                        _, _, mv = forward_losses(ema.module, bv, tok, dev, False); acc.append(mv)
                ev = {k: round(float(np.mean([x[k] for x in acc])), 3) for k in acc[0]}
                print(json.dumps(dict(step=step, val=ev)), flush=True)
                for k, v in ev.items():
                    tb.add_scalar(f"val/{k}", v, step)
                tb.flush()
                torch.save(dict(model=ema.module.state_dict(), arch="nanovla_s", st=P["st"], coef=P["coef"], step_frames=P["step"], bench=a.bench, steps=step, args=vars(a), val=ev), a.out)
            if step >= a.steps:
                break
    tb.close(); print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
