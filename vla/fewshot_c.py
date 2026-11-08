"""GCT stage 3: few-shot adaptation of NanoVLA-G with a small adapter (design doc, 2026-11-06).

Learned (adapter, ~0.3 M parameters): a task-vector offset du (added to u for the new instructions) + LoRA rank 8 on
the decision MLP, the forward model and the image-vector projection. Frozen: everything else (ViT, text encoder,
GRU memory, phase head). Each batch = half few-shot demos, half replay of the base data (forgetting guard). Same
losses and dropouts as stage 1; early stop on 2 held-out demos.

    python fewshot_c.py --base ~/VLA/ckpt/nanovla_g_online/vlag.pt --demos fs_bowl_gct --N 10 --out ~/VLA/ckpt/adapters/bowl_N10.pt
    vlac_server.py --ckpt <base> --adapter ~/VLA/ckpt/adapters/bowl_N10.pt
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn
import nanovla2 as v2, nanovla_r as R, nanovla_g as G

CK = os.path.expanduser("~/VLA/ckpt")


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base; self.r = r; self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01); self.B = nn.Parameter(torch.zeros(base.out_features, r))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale


def add_adapter(model, r=8):
    """wrap the target Linear layers with LoRA and give the model a task-vector offset; returns the adapter params"""
    for seq_name in ("mlp", "fwd", "vvec"):
        seq = getattr(model, seq_name)
        for i, m in enumerate(seq):
            if isinstance(m, nn.Linear):
                seq[i] = LoRA(m, r)
    model.du = nn.Parameter(torch.zeros(model.d, device=next(model.parameters()).device))
    tv = model.text_vec
    model.text_vec = lambda ids, _tv=tv: _tv(ids) + model.du
    return [p for n, p in model.named_parameters() if ".A" in n[-2:] or ".B" in n[-2:] or n == "du"] + [model.du]


def adapter_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B") or k == "du"}


def load_adapter(model, path):
    ad = torch.load(path, map_location="cpu", weights_only=False)
    add_adapter(model, ad.get("rank", 8))
    own = model.state_dict(); own.update({k: v.to(own[k].device) for k, v in ad["state"].items()}); model.load_state_dict(own)
    return ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--demos", required=True, help="data/<bench> with the new task / objects")
    ap.add_argument("--replay", default="isaac_gct_v1")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"; t0 = time.time(); R.T_MAX = a.window
    ck = torch.load(a.base, map_location="cpu", weights_only=False)
    Pd = v2.prepare(a.demos, 2, a.seed)                                     # 2 held-out demos for early stopping
    tr = sorted({int(e) for e, _ in Pd["train"]})[: a.N]; Pd["train_eps"] = tr
    val = sorted(e for e in Pd["acts"] if e not in set(sorted({int(e) for e, _ in Pd["train"]})))
    Pr = v2.prepare(a.replay, 50, 0); Pr["train_eps"] = sorted({int(e) for e, _ in Pr["train"]})
    cb = np.asarray(ck["coef"], np.float64)
    for P in (Pd, Pr):                  # one normalisation and one command calibration: the base model's
        cd = np.asarray(P["coef"], np.float64); den = np.where(np.abs(cd[:, 0]) > 1e-3, cd[:, 0], 1.0)
        P["acts"] = {e: ((A - cd[:, 1]) / den * cb[:, 0] + cb[:, 1], S) for e, (A, S) in P["acts"].items()}
        P["phase"] = {e: v2.phases(A) for e, (A, _) in P["acts"].items()}
        P["st"], P["coef"] = ck["st"], ck["coef"]
    model, tok = G.build(dev, ck["st"], init=""); model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad = False
    params = add_adapter(model, a.rank); params = list({id(p): p for p in params}.values())
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    W = dict(fwd=0.5, phase=0.1, loc=0.5, prog=0.1, div=0.0)
    ds_d, ds_r = R.SeqData(Pd, tr, True, "own"), R.SeqData(Pr, Pr["train_eps"], True, "own")
    ds_v = R.SeqData(Pd, val, False, "own") if val else None
    best, best_state, hist = 9e9, adapter_state(model), []
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        items = [ds_d[random.randrange(len(ds_d))] for _ in range(a.bs // 2)] + [ds_r[random.randrange(len(ds_r))] for _ in range(a.bs - a.bs // 2)]
        b = R.collate(items)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = G.run_window(model, b, tok, dev, True)
            loss, parts, _ = G.losses(model, b, tok, dev, o, W)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(float(parts["action"]))
        if ds_v is not None and (step % 250 == 0 or step == a.steps):
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                b = R.collate([ds_v[i] for i in range(len(ds_v))]); o = G.run_window(model, b, tok, dev, False); _, pv, _ = G.losses(model, b, tok, dev, o, W)
            vl = float(pv["action"])
            if vl < best:
                best, best_state = vl, adapter_state(model)
            print(json.dumps(dict(step=step, train_action=round(float(np.mean(hist[-250:])), 4), val_action=round(vl, 4))), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(state=best_state if ds_v is not None else adapter_state(model), rank=a.rank, base=a.base, demos=a.demos, N=a.N, val_action=best,
                    n_params=sum(p.numel() for p in params)), a.out)
    print(json.dumps(dict(saved=a.out, N=a.N, adapter_params=sum(p.numel() for p in params), best_val=round(best, 4), secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
