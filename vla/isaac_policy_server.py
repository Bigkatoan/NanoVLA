"""Few-shot fine-tune our VLA on N episodes of the Isaac SO-101 vials->rack dataset, then serve it to the Isaac Lab
eval client (isaac_bench/isaac_eval.py) over a local socket (stdlib multiprocessing.connection).

Units: LeRobot SO-101 'pos' units (joints -100..100, gripper 0..100), one dataset frame == one env step (60 Hz).
Request : dict(state=(n,6) float, front=(n,120,160,3) uint8, wrist=(n,120,160,3) uint8)
Reply   : (n, H, 6) command targets in the same units, chunk spacing = P['step'] frames.
"""
import argparse, os, json, random, time
import numpy as np, torch
from multiprocessing.connection import Listener
import fewshot as fs
import sim_fewshot as sf
from sim_fewshot import finetune, prep_img, CK, ROOT
from train_vla import H

TEXT = "Pick up the vial and place it in the rack"
CKPTS = {"base": ("A_v1", 0.0, 0.0), "fix": ("A_v1", 0.5, 1.0), "fixenc": ("A_v1enc", 0.5, 1.0), "scratch": ("C", 0.0, 0.0)}


def joint_goal_labels(P):
    """aux localisation target without FK: normalised (pan, lift) joint state at the grasp and at the release.
    Encoded so that FSDataAux's fixed transform (g - [0.2,0,0.2,0]) * 10 yields the normalised values."""
    st, lab = P["st"], {}
    for e, (A, S) in P["acts"].items():
        g = A[:, 5]; hi = np.percentile(g, 90); lo = np.percentile(g, 5); mid = (hi + lo) / 2
        if hi - lo < 5:
            continue
        op = np.nonzero(g > mid)[0]
        if not len(op):
            continue
        tc = next((t for t in range(op[0], len(g)) if g[t] < mid), None)
        if tc is None:
            continue
        tr = next((t for t in range(tc, len(g)) if g[t] > mid), len(g) - 1)
        v = np.r_[(S[tc, :2] - st["s_mean"][:2]) / st["s_std"][:2], (S[tr, :2] - st["s_mean"][:2]) / st["s_std"][:2]]
        lab[e] = (np.clip(v, -4, 4) / 10 + np.array([0.2, 0, 0.2, 0])).astype(np.float32)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="fixenc", choices=list(CKPTS))
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--bench", default="isaac_bench", help="data/<bench> in the few-shot bench format")
    ap.add_argument("--vla", default=os.path.join(CK, "vla/vla.pt"), help="starting checkpoint (vla/vla.pt, vla_isaac/vla.pt)")
    ap.add_argument("--save", default="", help="optional path to save the fine-tuned weights + normalisation")
    a = ap.parse_args()
    dev = "cuda"
    fs.ROOT_BENCH = os.path.join(ROOT, a.bench)
    sf.goal_labels = joint_goal_labels
    repo = json.load(open(os.path.join(ROOT, a.bench, "repos.json")))[0]
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    P = fs.prepare(repo, a.N, 0, a.seed)
    cond, sd, aux = CKPTS[a.cfg]
    vla = a.vla
    t0 = time.time()
    model, tok, row = finetune(P, cond, a.steps, 1e-4 if cond != "C" else 3e-4, 64, dev, vla, 3e-5, sd, aux)
    print(json.dumps(dict(event="trained", cfg=a.cfg, N=a.N, seed=a.seed, train_eps=P["n_train_eps"], samples=len(P["train"]), secs=round(time.time() - t0))), flush=True)
    st, coef = P["st"], P["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    if a.save:
        torch.save(dict(model=model.state_dict(), st=st, coef=coef, row=row, step=P["step"], cfg=a.cfg, N=a.N, seed=a.seed, bench=a.bench), a.save)
    from train_text import clean
    ids_default = tok([TEXT]).to(dev)
    with Listener(("localhost", a.port), authkey=b"vla-isaac") as L:
        print(json.dumps(dict(event="ready", port=a.port, step=P["step"])), flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try:
                        req = c.recv()
                    except EOFError:
                        break
                    if req == "meta":
                        c.send(dict(step=P["step"], H=H)); continue
                    if req == "quit":
                        return
                    S = np.asarray(req["state"], np.float64); n = len(S)
                    ids = tok([clean(t) for t in req["text"]]).to(dev) if "text" in req else ids_default.expand(n, -1)
                    ext = torch.stack([prep_img(f) for f in req["front"]]).to(dev)
                    wri = torch.stack([prep_img(w) for w in req["wrist"]]).to(dev)
                    s_n = torch.from_numpy(np.clip((S - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32)).to(dev)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        p = model(ext, wri, torch.ones(n, dtype=torch.bool, device=dev), s_n, ids, torch.full((n,), row, device=dev)).float().cpu().numpy()
                    tgt = np.empty((n, H, 6))
                    tgt[..., :5] = S[:, None, :5] + p[..., :5] * st["dstd"]
                    tgt[..., 5] = (p[..., 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
                    c.send((tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0))


if __name__ == "__main__":
    main()
