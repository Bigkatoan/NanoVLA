"""NanoVLA-M: wrist-camera VLA with an explicit memory (design agreed with the user, 2026-11-06).

The decision at every frame uses four separate sources:
    u      what to do          text encoder(prompt), computed ONCE per instruction and cached
    v_k    what I see now       image encoder(current frame) -> one vector (no frame stacking, no text input)
    s_k    how things are now   joint states (normalised) + change since the previous frame
    m_k-1  what has been done   recurrent memory over past frames, initialised from the task: m_0 = Init(u)

    Action_k = MLP([v_k, s_k, u, m_k-1])                    one pass, no iteration
    m_k      = GRU(m_k-1, [v_k, s_k, a_k])                  updated AFTER the decision with the action actually taken
Runs at 10 Hz (one tick per camera frame). Output: chunk of H_M = 8 future joint targets (0.8 s, 0.1 s apart);
the controller executes the first point and the model decides again at the next frame.

Heads (--head):
    l1   : MLP regresses the 8x6 chunk (normalised as in v1: joints (A - S)/dstd, gripper min-max to [-1, 1]); L1 loss
    bins : MLP outputs 64-bin logits per joint and step (one pass, keeps several modes); soft cross-entropy;
           decoded as the expectation over the +-2 bins around the arg-max

    python nanovla_m.py --head l1   --out ~/VLA/ckpt/nanovla_m_l1/vlam.pt
    python nanovla_m.py --head bins --out ~/VLA/ckpt/nanovla_m_bins/vlam.pt
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import nanovla2 as v2
import nanovla_r as R

CK = os.path.expanduser("~/VLA/ckpt")
H_M = 8
NB = 64
LO = torch.tensor([-4.0] * 5 + [-1.1]); HI = torch.tensor([4.0] * 5 + [1.1])     # bin range per joint (normalised units)


class NanoVLAM(nn.Module):
    def __init__(self, vis, tenc, n_tok, d=256, K=4, dv=512, mem=512, head="l1", hidden=1024, heads=8, ff=768, n_phase=3, a_noise=0.1):
        super().__init__()
        self.vis, self.t, self.d, self.mem, self.head_type, self.a_noise = vis, tenc, d, mem, head, a_noise
        # text -> u (task)
        self.tproj = nn.Sequential(nn.Linear(tenc.d, d), nn.LayerNorm(d))
        # image -> v (learned-query attention pooling over the ViT tokens; independent of the text)
        self.vproj = nn.Linear(vis.d, d)
        self.vpos = nn.Parameter(torch.zeros(1, n_tok, d)); nn.init.trunc_normal_(self.vpos, std=0.02)
        self.pool_q = nn.Parameter(torch.randn(1, K, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_n1, self.pool_n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.pool_mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.vvec = nn.Sequential(nn.Linear(K * d, dv), nn.LayerNorm(dv))
        # memory
        self.init = nn.Sequential(nn.Linear(d, mem), nn.GELU(), nn.Linear(mem, mem), nn.Tanh())
        self.gru = nn.GRUCell(dv + 12 + 6, mem)
        # decision MLP
        n_out = H_M * 6 * (NB if head == "bins" else 1)
        self.mlp = nn.Sequential(nn.LayerNorm(dv + 12 + d + mem), nn.Linear(dv + 12 + d + mem, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))
        self.phase = nn.Sequential(nn.LayerNorm(mem), nn.Linear(mem, n_phase))     # aux: memory must know the task phase
        self.register_buffer("lo", LO.clone()); self.register_buffer("hi", HI.clone())
        self.register_buffer("centers", torch.stack([torch.linspace(float(l), float(h), NB) for l, h in zip(LO, HI)]))   # [6, NB]

    def text_vec(self, ids):
        _, pooled, _ = self.t(ids)
        return self.tproj(pooled)

    def image_vec(self, img):
        tok = self.vproj(self.vis(img)) + self.vpos
        q = self.pool_q.expand(len(img), -1, -1) * 1.0
        a, _ = self.pool(self.pool_n1(q), tok, tok, need_weights=False)
        q = q + a
        q = q + self.pool_mlp(self.pool_n2(q))
        return self.vvec(q.flatten(1))

    def init_mem(self, u):
        return self.init(u).float()

    def decide(self, v, s, u, m):
        """-> raw head output: [N, H_M, 6] (l1) or [N, H_M, 6, NB] logits (bins)"""
        o = self.mlp(torch.cat([v, s, u, m.to(v.dtype)], -1))
        return o.view(-1, H_M, 6, NB) if self.head_type == "bins" else o.view(-1, H_M, 6)

    def decode(self, o):
        """head output -> normalised chunk [N, H_M, 6]"""
        if self.head_type != "bins":
            return o.float()
        p = o.float().softmax(-1)
        k = p.argmax(-1, keepdim=True)
        idx = torch.arange(NB, device=o.device).view(1, 1, 1, NB)
        w = p * ((idx - k).abs() <= 2)
        return (w * self.centers.view(1, 1, 6, NB)).sum(-1) / w.sum(-1).clamp(min=1e-6)

    def remember(self, m, v, s, a):
        return self.gru(torch.cat([v, s, a], -1).float(), m.float())

    def forward_seq(self, imgs, states, ids, a_taken, ss_p=0.0):
        """imgs [B,T,3,h,w], states [B,T,12], a_taken [B,T,6] (action executed at each frame, teacher forcing)
        ss_p: scheduled sampling - per sequence and frame, with this probability the memory receives the model's OWN
        first action (detached) instead of the demo's, so it learns to live with its own mistakes
        -> head outputs [B,T,...], memories m_k-1 used at each frame [B,T,mem]"""
        B, Tn = states.shape[:2]
        u = self.text_vec(ids)
        v = self.image_vec(imgs.flatten(0, 1)).view(B, Tn, -1)
        m = self.init_mem(u)
        outs, mems = [], []
        for k in range(Tn):
            outs.append(self.decide(v[:, k], states[:, k], u, m)); mems.append(m)
            a = a_taken[:, k]
            if self.training and self.a_noise > 0:
                a = a + torch.randn_like(a) * self.a_noise
            if ss_p > 0:
                own = self.decode(outs[-1])[:, 0].detach().to(a.dtype)
                a = torch.where((torch.rand(B, 1, device=a.device) < ss_p), own, a)
            m = self.remember(m, v[:, k], states[:, k], a)
        return torch.stack(outs, 1), torch.stack(mems, 1), m

    @torch.no_grad()
    def step(self, img, s, u, m):
        """one frame: decide with the memory of the past, then update the memory with the chosen first action"""
        v = self.image_vec(img)
        chunk = self.decode(self.decide(v, s, u, m))
        return chunk, self.remember(m, v, s, chunk[:, 0])


def build(dev, head="l1", v1_ckpt=os.path.join(CK, "vla_isaac_wrist_v1/vla.pt")):
    s = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    v1, tok = v2.build_model(s["n_emb"], os.path.join(CK, "align/align.pt"), "cpu")
    v1.load_state_dict(s["model"])
    return NanoVLAM(v2.OwnVision(v1.v), v1.t, v2.n_tokens("own"), head=head).to(dev), tok


def soft_ce(logits, tgt, lo, hi, sigma=1.0):
    """logits [N,6,NB], tgt [N,6] -> cross-entropy against a Gaussian (sigma in bins) around the target bin"""
    pos = ((tgt.clamp(lo, hi) - lo) / (hi - lo) * (NB - 1)).unsqueeze(-1)
    idx = torch.arange(NB, device=tgt.device).float()
    q = torch.exp(-0.5 * ((idx - pos) / sigma) ** 2); q = q / q.sum(-1, keepdim=True)
    return -(q * logits.float().log_softmax(-1)).sum(-1)


def losses(model, b, tok, dev, train=True, ss_p=0.0):
    img = R.gpu_images(b["img"].to(dev, non_blocking=True), v2.SIZE["own"], train)
    tgt, valid = b["tgt"].to(dev)[:, :, :H_M], b["valid"].to(dev)[:, :, :H_M]
    o, mems, _ = model.forward_seq(img, b["state"].to(dev), tok(b["label"]).to(dev), b["a_taken"].to(dev) if "a_taken" in b else tgt[:, :, 0], ss_p)
    msk = b["mask"].to(dev)
    o, mems, tgt, valid = o[msk], mems[msk], tgt[msk], valid[msk]
    vm = valid.unsqueeze(-1).float()
    if model.head_type == "bins":
        l = soft_ce(o.flatten(0, 1), tgt.flatten(0, 1), model.lo, model.hi).view(*tgt.shape)
    else:
        l = (o.float() - tgt).abs()
    w = torch.ones(6, device=dev); w[5] = 0.5
    la = (l * w * vm).sum() / (vm.sum() * 6 + 1e-6)
    lp = F.cross_entropy(model.phase(mems).float(), b["phase"].to(dev)[msk])
    return la, lp, o, tgt, valid


@torch.no_grad()
def evaluate(model, tok, P, dev, max_eps=100):
    eps = sorted(e for e in P["acts"] if e not in P["train_eps"])[:max_eps]
    dl = torch.utils.data.DataLoader(R.SeqData(P, eps, False, "own"), batch_size=4, num_workers=4, collate_fn=R.collate)
    model.eval(); tot = hold = acc = n = 0.0
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, o, tg, va = losses(model, b, tok, dev, train=False)
        p = model.decode(o); vm = va.unsqueeze(-1).float()
        z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
        tot += (((p - tg).abs() * vm).sum((1, 2)) / (vm.sum((1, 2)) * 6)).sum().item()
        hold += (((z - tg).abs() * vm).sum((1, 2)) / (vm.sum((1, 2)) * 6)).sum().item(); n += len(p)
    model.train()
    return dict(val_l1=round(tot / n, 4), val_hold=round(hold / n, 4), val_rel=round(tot / hold, 4), n_val_ticks=int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="l1", choices=("l1", "bins"))
    ap.add_argument("--bench", default="isaac_all_v2")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8, help="windows per batch")
    ap.add_argument("--window", type=int, default=48, help="frames per training window (4.8 s)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=3e-5)
    ap.add_argument("--phase_w", type=float, default=0.1)
    ap.add_argument("--freeze_blocks", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--ss_max", type=float, default=0.0, help="scheduled sampling: probability ramps 0 -> ss_max over the first half of training")
    ap.add_argument("--init", default="", help="start from a NanoVLA-M checkpoint (fine-tuning)")
    ap.add_argument("--act_mode", default="pos", choices=("pos", "vel"), help="pos: chunk = joint offsets from the current state; vel: per-step command increments")
    ap.add_argument("--jitter", type=int, default=0, help="timestamp noise in control steps (30 Hz): joint reading +-j vs the image, labels 0..j late")
    ap.add_argument("--out", default=os.path.join(CK, "nanovla_m/vlam.pt"))
    a = ap.parse_args()
    R.T_MAX = a.window
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"; t0 = time.time()
    P = v2.prepare(a.bench, a.n_val, a.seed)
    P["train_eps"] = sorted({int(e) for e, _ in P["train"]})
    print(json.dumps(dict(model="nanovla_m", head=a.head, episodes=len(P["train_eps"]), val_eps=P["n_test_eps"], window=a.window, prep_s=round(time.time() - t0))), flush=True)
    model, tok = build(dev, a.head)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
    zf = np.load(os.path.join(v2.ROOT, a.bench, "actions", json.load(open(os.path.join(v2.ROOT, a.bench, "repos.json")))[0].replace("/", "__") + ".npz"))
    P["pol"] = {int(k[2:].split("_")[0]): zf[k].astype(bool) for k in zf.files if k.endswith("_pol")}   # DAgger: steps driven by the policy
    P["act_mode"], P["jitter"] = a.act_mode, a.jitter
    inc = np.concatenate([A[P["step"]:, :5] - A[:-P["step"], :5] for A, _ in P["acts"].values()])
    P["vstd"] = np.maximum(inc.std(0), 1e-3).astype(np.float32)
    print(json.dumps(dict(dagger_eps=len(P["pol"]), policy_steps=int(sum(v.sum() for v in P["pol"].values())))), flush=True)
    for n, p in model.vis.v.named_parameters():
        if n.startswith(("proj.", "pos", "cls", "view_emb")) or any(n.startswith(f"blocks.{i}.") for i in range(a.freeze_blocks)):
            p.requires_grad = False
    is_enc = lambda n: n.startswith("vis.") or n.startswith("t.")
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    print(json.dumps(dict(params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2), trainable_M=round(sum(p.numel() for p in enc + head) / 1e6, 2),
                          new_M=round(sum(p.numel() for p in head) / 1e6, 2))), flush=True)
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
            ss_p = a.ss_max * min(1.0, step / (0.5 * a.steps))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                la, lp, *_ = losses(model, b, tok, dev, ss_p=ss_p)
            loss = la + a.phase_w * lp
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(model)
            hist.append((la.item(), lp.item()))
            if step % 100 == 0:
                h = np.array(hist[-100:])
                print(json.dumps(dict(step=step, act=round(float(h[:, 0].mean()), 4), phase=round(float(h[:, 1].mean()), 4), ss_p=round(ss_p, 3), s_per_step=round((time.time() - t1) / 100, 3))), flush=True)
                t1 = time.time()
            if step % a.eval_every == 0 or step == a.steps:
                ev = evaluate(ema.module, tok, P, dev)
                print(json.dumps(dict(step=step, **ev)), flush=True)
                torch.save(dict(model=ema.module.state_dict(), arch="nanovla_m", head=a.head, act_mode=a.act_mode, vstd=P["vstd"], jitter=a.jitter, st=P["st"], coef=P["coef"], step_frames=P["step"],
                                bench=a.bench, episodes=len(P["train_eps"]), steps=step, args=vars(a), **ev), a.out)
            if step >= a.steps:
                break
    print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
