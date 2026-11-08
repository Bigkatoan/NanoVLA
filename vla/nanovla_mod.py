"""NanoVLA-Mod: the state-based policy split into small single-purpose models (2026-11-07).

  1 TextEnc   instruction -> task vector u                      (pretrained, frozen; once per instruction)
  2 Backbone  wrist image -> 80 tokens                          (ViT-Tiny, shared by 3-5)
  3 Grounder  tokens, u -> object xyz, destination xyz (wrist-camera frame, metres) + attention heat-maps
  4 StateNet  pooled image, joints, u, previous state -> task state (expert step)
  5 GraspNet  tokens, u, state -> yaw still to turn (cos, sin)
  6 Planner   state, object xyz, destination xyz, joints, u -> target point xyz (camera frame) + jaw   (NO image)
  7 NanoIK    joints, camera-frame displacement -> joint change (tools/nano_ik.py, separate)
  8 Servo     speed limit + jaw ramp (not learned, in the eval client)
Each model has its own loss and metric. The Planner is trained on the TRUE object / destination (+ noise of the size of
the Grounder error) so it does not depend on the Grounder being right; at run time it gets the Grounder's output.
Any module can be replaced by an oracle at eval time (vlas_server --oracle grounder,state,...) to find the bottleneck.

    python nanovla_mod.py --run mod1 --bench isaac_s_v1 --steps 15000 --out ~/VLA/ckpt/nanovla_mod/mod.pt
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import nanovla2 as v2
import nanovla_s as S

CK = os.path.expanduser("~/VLA/ckpt"); TB = os.path.expanduser("~/VLA/tb"); K = S.K


def mlp(i, o, h=256, n=2):
    L, d = [], i
    for _ in range(n):
        L += [nn.Linear(d, h), nn.GELU()]; d = h
    return nn.Sequential(nn.LayerNorm(i), *L, nn.Linear(d, o))


class Backbone(nn.Module):
    def __init__(self, vis, n_tok, d=256):
        super().__init__(); self.vis, self.d = vis, d
        self.vproj = nn.Linear(vis.d, d); self.vpos = nn.Parameter(torch.zeros(1, n_tok, d)); nn.init.trunc_normal_(self.vpos, std=0.02)

    def forward(self, img):
        return self.vproj(self.vis(img)) + self.vpos


class TextEnc(nn.Module):
    def __init__(self, tenc, d=256):
        super().__init__(); self.t = tenc; self.tproj = nn.Sequential(nn.Linear(tenc.d, d), nn.LayerNorm(d))

    def forward(self, ids):
        _, pooled, _ = self.t(ids); return self.tproj(pooled)


def soft_point(tok, q, k_proj, grid, out):
    """attention of queries q [N,Q,d] over tokens -> (xyz via out-MLP [N,Q,3], logits [N,Q,80], features [N,Q,d])"""
    logit = torch.einsum("nqd,ntd->nqt", q.float(), k_proj(tok).float()) / q.shape[-1] ** 0.5
    att = logit.softmax(-1); xy = att @ grid; feat = att @ tok.float()
    return out(torch.cat([feat, xy], -1).to(tok.dtype)).float() * 0.1, logit, feat


class Grounder(nn.Module):
    def __init__(self, d=256):
        super().__init__(); self.d = d
        self.q = nn.Linear(d, 2 * d); self.k = nn.Linear(d, d); self.out = mlp(d + 2, 3, 128)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 10), indexing="ij")
        self.register_buffer("grid", torch.stack([gx.flatten(), gy.flatten()], -1))

    def forward(self, tok, u):
        xyz, logit, feat = soft_point(tok, self.q(u).view(-1, 2, self.d), self.k, self.grid, self.out)
        return xyz[:, 0], xyz[:, 1], logit


class StateNet(nn.Module):
    def __init__(self, d=256, Kq=4):
        super().__init__(); self.d = d
        self.pq = nn.Parameter(torch.randn(1, Kq, d) * 0.02); self.pool = nn.MultiheadAttention(d, 8, batch_first=True); self.n1 = nn.LayerNorm(d)
        self.prev = nn.Embedding(K + 1, d); self.head = mlp(Kq * d + 6 + 2 * d, K, 512)

    def forward(self, tok, s6, u, prev):
        q = self.pq.expand(len(tok), -1, -1) * 1.0
        a, _ = self.pool(self.n1(q), tok, tok, need_weights=False)
        return self.head(torch.cat([(q + a).flatten(1), s6.to(tok.dtype), u, self.prev(prev)], -1)).float()


class GraspNet(nn.Module):
    def __init__(self, d=256):
        super().__init__(); self.d = d
        self.st = nn.Embedding(K, d); self.q = nn.Linear(2 * d, d); self.k = nn.Linear(d, d); self.out = mlp(2 * d, 2, 128)

    def forward(self, tok, u, st):
        e = self.st(st); q = self.q(torch.cat([u, e], -1))[:, None]
        logit = torch.einsum("nqd,ntd->nqt", q.float(), self.k(tok).float()) / self.d ** 0.5
        feat = (logit.softmax(-1) @ tok.float())[:, 0]
        ys = self.out(torch.cat([feat.to(tok.dtype), e], -1)).float()
        return torch.atan2(ys[:, 1], ys[:, 0])


class Planner(nn.Module):
    """pure geometry: no image. target = obj/dest + learned state-dependent offset (in camera frame, needs joints for 'up')"""
    def __init__(self, d=256, h=256, n=3, use_s6=True):
        super().__init__(); self.use_s6 = use_s6
        self.st = nn.Embedding(K, 64); self.u = nn.Linear(d, 64); self.net = mlp(64 + 64 + 3 + 3 + 6, 3 + 3 + 1, h, n)

    def forward(self, st, obj, dest, s6, u):
        s6 = s6.float() if self.use_s6 else torch.zeros_like(s6, dtype=torch.float32)
        h = self.net(torch.cat([self.st(st), self.u(u.float()), obj / 0.3, dest / 0.3, s6], -1))
        w = torch.softmax(h[:, 3:5], -1)                            # which anchor (object / destination) the target hangs on
        anchor = w[:, :1] * obj + w[:, 1:2] * dest
        return anchor + h[:, :3] * 0.1, h[:, 6], w                   # target xyz, jaw logit


class NanoVLAMod(nn.Module):
    def __init__(self, vis, tenc, n_tok, d=256):
        super().__init__()
        self.backbone, self.text = Backbone(vis, n_tok, d), TextEnc(tenc, d)
        self.grounder, self.state, self.grasp, self.planner = Grounder(d), StateNet(d), GraspNet(d), Planner(d)

    def text_vec(self, ids):
        return self.text(ids)

    @torch.no_grad()
    def act(self, img, s6, u, prev, oracle=None, tcp=None, gate=0.01):
        """oracle: optional dict of tensors replacing module outputs: obj, dest, state, dyaw.
        tcp: grasp centre in the camera frame; if given, a forward step k -> k+1 is accepted only once the arm has reached
        step k's target (event = 'arrived'), so StateNet cannot skip ahead while the arm is still on its way."""
        o = oracle or {}
        tok = self.backbone(img)
        obj, dest, _ = self.grounder(tok, u)
        obj, dest = o.get("obj", obj), o.get("dest", dest)
        st = o.get("state", self.state(tok, s6, u, prev).argmax(-1))
        if tcp is not None and "state" not in o:
            pk = prev.clamp(max=K - 1)
            tp, _, _ = self.planner(pk, obj, dest, s6, u)
            far = (tp.float() - tcp).norm(dim=-1) > gate
            st = torch.where((st > pk) & far & (prev < K), pk, st)          # no forward step (of any size) before arriving
        dyaw = o.get("dyaw", self.grasp(tok, u, st))
        tgt, jaw, _ = self.planner(st, obj, dest, s6, u)
        return dict(state=st, target=tgt, dyaw=dyaw, jaw_open=torch.sigmoid(jaw), obj=obj, dest=dest)


def build(dev, v1_ckpt=os.path.join(CK, "vla_isaac_wrist_v1/vla.pt"), init=os.path.join(CK, "nanovla_g/vlag.pt")):
    s = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    v1, tok = v2.build_model(s["n_emb"], os.path.join(CK, "align/align.pt"), "cpu"); v1.load_state_dict(s["model"])
    m = NanoVLAMod(v2.OwnVision(v1.v), v1.t, v2.n_tokens("own"))
    if init and os.path.exists(init):                           # GCT-trained ViT / text encoder / projections
        src = torch.load(init, map_location="cpu", weights_only=False)["model"]; own = m.state_dict(); ok = {}
        for k, w in src.items():
            k2 = ("backbone." + k) if k.split(".")[0] in ("vis", "vproj", "vpos") else ("text." + k) if k.split(".")[0] in ("t", "tproj") else None
            if k2 in own and own[k2].shape == w.shape:
                ok[k2] = w
        own.update(ok); m.load_state_dict(own); print(json.dumps(dict(init=init, loaded=len(ok))), flush=True)
    return m.to(dev), tok


PLAN_REAL, PLAN_SHIFT = 0.0, 0.0


def losses(m, b, tok, dev, train):
    img = S.gpu_img(b["img"].to(dev, non_blocking=True), train)
    tokn = m.backbone(img); u = m.text(tok(b["label"]).to(dev))
    s6, prev, k = b["s6"].to(dev), b["prev"].to(dev), b["k"].to(dev); hs, hl = b["has_s"].to(dev), b["has_loc"].to(dev)
    obj_t, dest_t, tgt_t = b["obj"].to(dev), b["dest"].to(dev), b["tgt"].to(dev)
    obj, dest, glog = m.grounder(tokn, u)
    sl = m.state(tokn, s6, u, prev); pst = sl.argmax(-1)
    dyaw = m.grasp(tokn, u, k)                                   # each module gets the TRUE inputs of the others (decoupled)
    no = 0.02 * torch.randn_like(obj_t) if train else 0
    pobj, pdest = obj_t + no, dest_t + (0.02 * torch.randn_like(dest_t) if train else 0)
    if train and PLAN_REAL > 0:                                  # half the batch: the Grounder's own (detached) estimates, so the
        r = (torch.rand(len(obj_t), 1, device=dev) < PLAN_REAL)  # Planner learns how far each input can be trusted
        pobj, pdest = torch.where(r, obj.detach().float(), pobj), torch.where(r, dest.detach().float(), pdest)
    tgt_p = tgt_t
    if train and PLAN_SHIFT > 0:                                 # translation augmentation: every expert target hangs on the object /
        dl = PLAN_SHIFT * torch.randn_like(obj_t)                # destination, so shifting all three together is a valid scene. Kills
        pobj, pdest, tgt_p = pobj + dl, pdest + dl, tgt_t + dl   # the shortcut "target = fixed offset from the gripper" (via joints)
    tgt, jaw, _ = m.planner(k, pobj, pdest, s6, u)
    w = 1 + 4 * b["chg"].to(dev).float()
    mean = lambda x, msk: (x * msk).sum() / msk.sum().clamp(min=1)
    L = dict(grounder=mean(F.smooth_l1_loss(torch.cat([obj, dest], -1), torch.cat([obj_t, dest_t], -1), beta=0.01, reduction="none").sum(-1), hl) * 10
             + 0.2 * (S.heat_ce(glog[:, 0], obj_t, hl) + S.heat_ce(glog[:, 1], dest_t, hl)),
             state=(F.cross_entropy(sl, k, reduction="none") * w * hs).sum() / (w * hs).sum().clamp(min=1),
             grasp=mean(((dyaw - b["dyaw"].to(dev) + math.pi) % (2 * math.pi) - math.pi).abs(), hs),
             planner=mean(F.smooth_l1_loss(tgt, tgt_p, beta=0.01, reduction="none").sum(-1), hs) * 10 + 0.5 * mean(F.binary_cross_entropy_with_logits(jaw, b["jaw"].to(dev), reduction="none"), hs))
    with torch.no_grad():                                         # module metrics + the composed (run-time) target error
        tgt_run, _, _ = m.planner(pst, obj, dest, s6, u)
        g_ = torch.Generator(device=dev).manual_seed(0); dl_ = 0.05 * torch.randn(obj_t.shape, device=dev, generator=g_)
        tgt_sh, _, _ = m.planner(k, obj_t + dl_, dest_t + dl_, s6, u)   # does the Planner follow the object when it is elsewhere?
        chg = (b["chg"].to(dev) > 0) & hs
        met = dict(grounder_obj_cm=float(mean((obj - obj_t).norm(dim=-1), hl) * 100), grounder_dest_cm=float(mean((dest - dest_t).norm(dim=-1), hl) * 100),
                   state_acc=float(mean((pst == k).float(), hs)), state_acc_change=float(mean((pst == k).float(), chg)),
                   grasp_yaw_deg=float(L["grasp"]) * 57.3, planner_target_cm=float(mean((tgt - tgt_p).norm(dim=-1), hs) * 100),
                   planner_jaw_acc=float(mean(((jaw > 0) == (b["jaw"].to(dev) > 0.5)).float(), hs)),
                   planner_shift5_cm=float(mean((tgt_sh - tgt_t - dl_).norm(dim=-1), hs) * 100),
                   composed_target_cm=float(mean((tgt_run - tgt_t).norm(dim=-1), hs) * 100))
    total = L["grounder"] + L["state"] + 0.3 * L["grasp"] + L["planner"]
    return total, L, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mod1"); ap.add_argument("--bench", default="isaac_s_v1"); ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--enc_lr", type=float, default=5e-5)
    ap.add_argument("--freeze_blocks", type=int, default=4); ap.add_argument("--n_val", type=int, default=80); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--event_labels", type=int, default=1); ap.add_argument("--plan_real", type=float, default=0.0); ap.add_argument("--plan_shift", type=float, default=0.0); ap.add_argument("--init_ckpt", default="")
    ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--out", default=os.path.join(CK, "nanovla_mod/mod.pt"))
    a = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(TB, a.run)); tb.add_text("config", "\n".join(f"{k}: {v}" for k, v in vars(a).items()))
    random.seed(0); np.random.seed(0); torch.manual_seed(0); dev = "cuda"; t0 = time.time()
    P = v2.prepare(a.bench, a.n_val, 0)
    global PLAN_REAL, PLAN_SHIFT; PLAN_REAL, PLAN_SHIFT = a.plan_real, a.plan_shift
    if a.event_labels:
        S.event_labels(P)
    tr_eps = sorted({int(e) for e, _ in P["train"]}); va_eps = sorted(e for e in P["acts"] if e not in set(tr_eps))
    tr, va = S.items_of(P, tr_eps), S.items_of(P, va_eps)
    for k_ in ("train", "test"):
        P.pop(k_, None)
    print(json.dumps(dict(train_frames=len(tr), val_frames=len(va), prep_s=round(time.time() - t0))), flush=True)
    m, tok = build(dev)
    if a.init_ckpt:
        m.load_state_dict(torch.load(a.init_ckpt, map_location="cpu", weights_only=False)["model"]); print(json.dumps(dict(init_ckpt=a.init_ckpt)), flush=True)
    for p in m.text.t.parameters():
        p.requires_grad = False                                   # TextEnc frozen
    for n, p in m.backbone.vis.v.named_parameters():
        if n.startswith(("proj.", "pos", "cls", "view_emb")) or any(n.startswith(f"blocks.{i}.") for i in range(a.freeze_blocks)):
            p.requires_grad = False
    enc = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("backbone.vis.")]
    head = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("backbone.vis.")]
    sizes = {k: round(sum(p.numel() for p in getattr(m, k).parameters()) / 1e6, 3) for k in ("text", "backbone", "grounder", "state", "grasp", "planner")}
    print(json.dumps(dict(module_params_M=sizes)), flush=True); tb.add_text("modules", json.dumps(sizes))
    opt = torch.optim.AdamW([{"params": head, "base": a.lr}, {"params": enc, "base": a.enc_lr}], weight_decay=0.05, fused=True)
    ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    dl = torch.utils.data.DataLoader(S.FrameData(P, tr, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=S.collate, drop_last=True, pin_memory=True, persistent_workers=True)
    vsel = va[np.random.default_rng(0).choice(len(va), min(4000, len(va)), replace=False)]
    dlv = torch.utils.data.DataLoader(S.FrameData(P, vsel, False), batch_size=256, num_workers=4, collate_fn=S.collate)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); step, hist = 0, []
    while step < a.steps:
        for b in dl:
            step += 1
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1, step / 500) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, L, met = losses(m, b, tok, dev, True)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(m)
            hist.append({**{f"loss_{k}": float(v) for k, v in L.items()}, **met})
            if step % 50 == 0:
                for k in hist[-1]:
                    tb.add_scalar(f"train/{k}", float(np.mean([h[k] for h in hist[-50:]])), step)
            if step % 500 == 0:
                print(json.dumps(dict(step=step, **{k: round(float(np.mean([h[k] for h in hist[-500:]])), 3) for k in met})), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                ema.module.eval(); acc = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for bv in dlv:
                        acc.append(losses(ema.module, bv, tok, dev, False)[2])
                ev = {k: round(float(np.mean([x[k] for x in acc])), 3) for k in acc[0]}
                print(json.dumps(dict(step=step, val=ev)), flush=True)
                for k, v in ev.items():
                    tb.add_scalar(f"val/{k}", v, step)
                tb.flush()
                torch.save(dict(model=ema.module.state_dict(), arch="nanovla_mod", st=P["st"], coef=P["coef"], step_frames=P["step"], bench=a.bench, steps=step, args=vars(a), val=ev), a.out)
            if step >= a.steps:
                break
    tb.close(); print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
