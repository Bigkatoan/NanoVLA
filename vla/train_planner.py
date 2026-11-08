"""Planner-only training (NanoVLA-Mod). The Planner sees no image, so it trains on the label arrays alone: thousands of
steps per minute. Tests whether its ~2 cm floor is an optimisation problem (lr / schedule / shared optimiser) or label noise.

  inputs : state k, object + destination (camera frame, with noise of the Grounder's size), joints s6, task vector u
  target : the step's target (event labels), jaw
  aug    : translation of object + destination + target together (the Planner must follow the object, not the joints)
  filter : --drop_first_grip drops the grip frames of step 0 (the arm is still settling there, the target label is stale)

    python train_planner.py --lrs 3e-4,1e-3,3e-3 --steps 20000 --init ~/VLA/ckpt/nanovla_mod/mod3.pt --out ~/VLA/ckpt/nanovla_mod/mod4.pt
Writes the best Planner into a copy of the --init checkpoint (all other modules unchanged).
"""
import argparse, copy, json, math, os, time
import numpy as np, torch, torch.nn.functional as F
import nanovla2 as v2, nanovla_s as S, nanovla_mod as MD
from torch.utils.tensorboard import SummaryWriter
from train_text import clean


def arrays(P, eps, drop_first_grip):
    st = P["st"]; rows = []
    for e in eps:
        A, Sx = P["acts"][e]
        for f in P["fidx"][e]:
            L = P["loc"].get((e, int(f)))
            if L is None or len(L) < 13:
                continue
            if drop_first_grip and int(L[12]) == 0 and int(L[11]) == 0:
                continue
            rows.append((e, int(f), L))
    k = np.array([min(int(L[11]), S.K - 1) for _, _, L in rows]); ph = np.array([int(L[12]) for _, _, L in rows])
    obj = np.stack([L[0:3] for _, _, L in rows]); dest = np.stack([L[3:6] for _, _, L in rows]); tgt = np.stack([L[6:9] for _, _, L in rows])
    jaw = np.array([float(L[10]) for _, _, L in rows])
    s6 = np.stack([np.clip((P["acts"][e][1][f] - st["s_mean"]) / st["s_std"], -5, 5) for e, f, _ in rows]).astype(np.float32)
    task = [clean(P["task"][e]) for e, _, _ in rows]
    return dict(k=k, ph=ph, obj=obj, dest=dest, tgt=tgt, jaw=jaw, s6=s6), task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="isaac_s_v1"); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3"); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--shift", type=float, default=0.05); ap.add_argument("--obj_noise", type=float, default=0.02); ap.add_argument("--dest_noise", type=float, default=0.05)
    ap.add_argument("--drop_first_grip", type=int, default=1); ap.add_argument("--fresh", type=int, default=0, help="1: re-initialise the Planner instead of fine-tuning")
    ap.add_argument("--run", default="planner")
    ap.add_argument("--variants", default="", help="name:hidden:layers:use_s6:lr,... (fresh Planners) instead of the lr sweep"); a = ap.parse_args()
    dev = "cuda"; t0 = time.time(); torch.manual_seed(0)
    P = v2.prepare(a.bench, 80, 0); S.event_labels(P)
    tr_eps = sorted({int(e) for e, _ in P["train"]}); va_eps = sorted(e for e in P["acts"] if e not in set(tr_eps))
    TR, ttr = arrays(P, tr_eps, a.drop_first_grip); VA, tva = arrays(P, va_eps, 0)
    st_norm = P["st"]; P.clear()
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    model, tok = MD.build(dev, init=""); model.load_state_dict(ck["model"]); model.eval()
    with torch.no_grad():
        U = {t: model.text_vec(tok([t]).to(dev))[0].float() for t in sorted(set(ttr) | set(tva))}
    g = lambda D, T: ({k: torch.tensor(v, device=dev, dtype=torch.long if k in ("k", "ph") else torch.float32) for k, v in D.items()}, torch.stack([U[t] for t in T]))
    TR, UTR = g(TR, ttr); VA, UVA = g(VA, tva)
    print(json.dumps(dict(train=len(UTR), val=len(UVA), prep_s=round(time.time() - t0))), flush=True)
    tb = SummaryWriter(os.path.join(MD.TB if hasattr(MD, "TB") else os.path.expanduser("~/VLA/tb"), a.run))

    def evaluate(pl):
        pl.eval(); out = {}
        with torch.no_grad():
            gen = torch.Generator(device=dev).manual_seed(0); dl = 0.05 * torch.randn(VA["obj"].shape, device=dev, generator=gen)
            t1, j1, _ = pl(VA["k"], VA["obj"], VA["dest"], VA["s6"], UVA); t2, _, _ = pl(VA["k"], VA["obj"] + dl, VA["dest"] + dl, VA["s6"], UVA)
            e1 = (t1 - VA["tgt"]).norm(dim=-1) * 100; e2 = (t2 - VA["tgt"] - dl).norm(dim=-1) * 100
            ix = torch.arange(0, len(UTR), max(1, len(UTR) // 5000), device=dev)
            t3, _, _ = pl(TR["k"][ix], TR["obj"][ix], TR["dest"][ix], TR["s6"][ix], UTR[ix])
            out = dict(train_target_cm=float(((t3 - TR["tgt"][ix]).norm(dim=-1) * 100).mean()), target_cm=float(e1.mean()), target_med_cm=float(e1.median()), shift5_cm=float(e2.mean()), jaw_acc=float(((j1 > 0) == (VA["jaw"] > 0.5)).float().mean()))
            for s in range(4):
                m = VA["k"] == s
                if m.any():
                    out[f"state{s}_cm"] = float(e1[m].mean())
            for p_ in range(4):
                m = VA["ph"] == p_
                if m.any():
                    out[f"phase{p_}_cm"] = float(e1[m].mean())
        pl.train(); return out

    base = evaluate(copy.deepcopy(model.planner)); print(json.dumps(dict(init=a.init, **{k: round(v, 3) for k, v in base.items()})), flush=True)
    best, best_err = None, base["shift5_cm"] + base["target_cm"]; res = {"init": base}
    runs = []
    if a.variants:
        for v in a.variants.split(","):
            nm, h, nl, us, lr = v.split(":"); runs.append((nm, float(lr), lambda h=int(h), nl=int(nl), us=bool(int(us)): MD.Planner(h=h, n=nl, use_s6=us).to(dev)))
    else:
        runs = [(f"lr_{float(x):g}", float(x), (lambda: MD.Planner().to(dev)) if a.fresh else (lambda: copy.deepcopy(model.planner).float())) for x in a.lrs.split(",")]
    for name, lr, mk in runs:
        pl = mk(); print(json.dumps(dict(run=name, params=sum(p.numel() for p in pl.parameters()))), flush=True)
        opt = torch.optim.AdamW(pl.parameters(), lr, weight_decay=0.01); n = len(UTR)
        for s in range(a.steps):
            for g_ in opt.param_groups:
                g_["lr"] = lr * min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * s / a.steps))
            b = torch.randint(n, (a.bs,), device=dev)
            obj = TR["obj"][b] + a.obj_noise * torch.randn(a.bs, 3, device=dev); dest = TR["dest"][b] + a.dest_noise * torch.randn(a.bs, 3, device=dev)
            tgt = TR["tgt"][b]
            if a.shift > 0:
                dl = a.shift * torch.randn(a.bs, 3, device=dev); obj, dest, tgt = obj + dl, dest + dl, tgt + dl
            t_, j_, _ = pl(TR["k"][b], obj, dest, TR["s6"][b], UTR[b])
            loss = F.smooth_l1_loss(t_, tgt, beta=0.005, reduction="none").sum(-1).mean() * 10 + 0.5 * F.binary_cross_entropy_with_logits(j_, TR["jaw"][b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if (s + 1) % 2000 == 0 or s + 1 == a.steps:
                ev = evaluate(pl)
                for k_, v_ in ev.items():
                    tb.add_scalar(f"{name}/{k_}", v_, s + 1)
                print(json.dumps(dict(run=name, step=s + 1, loss=round(float(loss), 4), **{k: round(v, 3) for k, v in ev.items()})), flush=True)
        res[name] = ev
        if ev["shift5_cm"] + ev["target_cm"] < best_err:
            best, best_err, best_lr = copy.deepcopy(pl), ev["shift5_cm"] + ev["target_cm"], name
    tb.close()
    if best is not None:
        sd = ck["model"]; sd.update({"planner." + k: v.cpu() for k, v in best.state_dict().items()})
        ck2 = dict(ck); ck2["model"] = sd; ck2["planner_sweep"] = dict(best_lr=best_lr, results=res, args=vars(a))
        torch.save(ck2, a.out); print(json.dumps(dict(saved=a.out, best_lr=best_lr)), flush=True)
    else:
        print(json.dumps(dict(saved=None, note="no lr beat the initial Planner")), flush=True)
    json.dump(res, open(a.out.replace(".pt", "_planner_sweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
