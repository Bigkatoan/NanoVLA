"""Few-shot benchmark on held-out repos (data/bench, never used in any training stage, duplicates removed).

For each repo and N in --Ns: N train demos (random, seed), up to --n_test other episodes for testing.
Cleaning uses ONLY the N train demos (glitch repair, leader->follower map, stats); idle edges are trimmed.
Conditions:
  A  pretrained VLA (stage 4), new embodiment embedding, fine-tune head (+embedding), encoders frozen
  B  aligned encoders (stage 3) frozen, VLA head trained from scratch on the N demos
  C  everything from scratch on the N demos (random encoders, trainable)
Metric: masked L1 on 1.6 s action chunks (normalised by the N-demo stats), vs 'hold still' baseline.
"""
import argparse, os, json, random, io, time, math
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.io import decode_jpeg
from clean_v2 import repair, fit_calib
from clean_data import frame_flags, IDLE_EDGE, IDLE_PAUSE
from train_vla import VLA, build_model, masked_l1, H, CTRL_HZ, collate, MEAN, STD
from models import VisionViT, TextEncoder, n_params
from train_text import augment, clean

ROOT = os.path.expanduser("~/VLA/data")
CK = os.path.expanduser("~/VLA/ckpt")
WRIST_KEYS = ("wrist", "gripper", "hand", "on_robot", "arm")


ROOT_BENCH = None


def prepare(repo, N, n_test, seed):
    name = repo.replace("/", "__")
    bdir = ROOT_BENCH or os.path.join(ROOT, "bench")
    Fr = pd.read_parquet(os.path.join(bdir, "frames", name + ".parquet"))
    z = np.load(os.path.join(bdir, "actions", name + ".npz"))
    fps = int(Fr.fps.iloc[0])
    eps = sorted(int(k[2:].split("_")[0]) for k in z.files if k.endswith("_action"))
    eps = [e for e in eps if len(z[f"ep{e}_action"]) > 2 * fps and e in set(Fr.ep)]
    rng = random.Random(seed); rng.shuffle(eps)
    tr, te = eps[:N], eps[N:N + n_test]
    raw = {e: (repair(z[f"ep{e}_action"])[0], repair(z[f"ep{e}_state"])[0]) for e in tr + te}
    sc, L, coef, _ = fit_calib([raw[e] for e in tr], max_lag=int(0.6 * fps))
    acts = {e: ((A * coef[:, 0] + coef[:, 1]) if sc > 0.6 else A, S) for e, (A, S) in raw.items()}
    step = max(1, round(fps / CTRL_HZ))
    D, G, SS = [], [], []
    for e in tr:
        A, S = acts[e]; idx = np.arange(0, len(A), step)
        for k in (1, 4, 8, 16):
            D.append(A[np.clip(idx + k * step, 0, len(A) - 1), :5] - S[idx, :5])
        G.append(A[:, 5]); SS.append(S)
    D, G, SS = np.concatenate(D), np.concatenate(G), np.concatenate(SS)
    st = dict(dstd=np.maximum(D.std(0), 1e-2), g_lo=np.percentile(G, 1), g_hi=np.percentile(G, 99), s_mean=SS.mean(0), s_std=np.maximum(SS.std(0), 1e-2))
    cams = list(Fr.cam.unique())
    wr = [c for c in cams if any(k in c.lower() for k in WRIST_KEYS)]
    ext = [c for c in cams if c not in wr] or cams
    wcam = wr[0] if wr else None
    task = {}
    for e in tr + te:
        t = Fr[Fr.ep == e].task.iloc[0]
        task[e] = clean(t) if t and "FROZEN" not in t else clean(Fr[Fr.ep == e].task_orig.iloc[0])

    def samples(ep_list):
        out = []
        for e in ep_list:
            A, S = acts[e]
            flags, _ = frame_flags(A, fps)
            g = Fr[Fr.ep == e]
            ge = g[g.cam == ext[0]].sort_values("fidx")
            gw = g[g.cam == wcam].set_index("fidx") if wcam else None
            for r in ge.itertuples():
                f = int(r.fidx)
                if f + step >= len(A) or flags[min(f, len(A) - 1)] & (IDLE_EDGE | IDLE_PAUSE):
                    continue
                wj = gw.loc[f, "jpeg"] if (gw is not None and f in gw.index) else None
                out.append((e, f, r.jpeg, wj))
        return out
    cf = coef if sc > 0.6 else np.c_[np.ones(6), np.zeros(6)].astype(np.float32)
    return dict(train=samples(tr), test=samples(te), acts=acts, st=st, step=step, task=task, fps=fps, n_train_eps=len(tr), n_test_eps=len(te), coef=cf)


class FSData(torch.utils.data.Dataset):
    def __init__(self, P, split, train):
        self.S, self.P, self.train = P[split], P, train
        self.aug = T.Compose([T.RandomResizedCrop(112, scale=(0.8, 1.0), ratio=(1.2, 1.4), antialias=True), T.RandomApply([T.ColorJitter(0.3, 0.3, 0.1, 0.0)], p=0.8)])

    def __len__(self):
        return len(self.S)

    def img(self, b):
        x = decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8))
        x = self.aug(x) if self.train else T.functional.resize(x, [112, 149], antialias=True)[..., 18:130]
        return (x.float() / 255 - MEAN) / STD

    def __getitem__(self, i):
        e, f, ej, wj = self.S[i]
        A, S = self.P["acts"][e]; st = self.P["st"]; step = self.P["step"]
        idx = f + step * np.arange(1, H + 1); valid = idx < len(A); idx = np.minimum(idx, len(A) - 1)
        tgt = np.empty((H, 6), np.float32)
        tgt[:, :5] = (A[idx, :5] - S[f, :5]) / st["dstd"]
        tgt[:, 5] = (A[idx, 5] - st["g_lo"]) / (st["g_hi"] - st["g_lo"] + 1e-6) * 2 - 1
        ext = self.img(ej)
        wri = self.img(wj) if wj is not None else torch.zeros_like(ext)
        lab = self.P["task"][e]
        if self.train and random.random() < 0.3:
            lab = augment(lab, random)
        return dict(ext=ext, wri=wri, has_w=torch.tensor(wj is not None), state=torch.from_numpy(np.clip((S[f] - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32)),
                    tgt=torch.from_numpy(np.clip(tgt, -10, 10)), valid=torch.from_numpy(valid), rid=torch.tensor(0), w=torch.tensor(1.0), label=lab)


def make_model(cond, dev):
    vp = os.environ.get("VLA_CKPT", os.path.join(CK, "vla/vla.pt"))
    vs = torch.load(vp, map_location=dev, weights_only=False) if cond == "A" else {"n_emb": 1300}   # own checkpoints (store numpy stats)
    model, tok = build_model(vs["n_emb"], os.path.join(CK, "align/align.pt"), dev)
    new_row = vs["n_emb"] - 1
    if cond.startswith("A"):
        model.load_state_dict(vs["model"])
        if not vs.get("keep_row"):      # checkpoints adapted to this robot (e.g. vla_isaac) keep their learned row
            with torch.no_grad():
                unk = vs.get("unk_row")
                model.emb.weight[new_row] = model.emb.weight[unk] if unk is not None else model.emb.weight[: new_row - 63].mean(0)
    elif cond == "C":
        model.v = VisionViT(drop_path=0.05).to(dev)
        model.t = TextEncoder(model.t.tok.num_embeddings, max_len=model.t.pos.shape[1]).to(dev)
    return model, tok, new_row


def run(P, cond, steps, lr, bs, dev, enc_lr=1e-5):
    model, tok, row = make_model(cond, dev)
    is_enc = lambda n: n.startswith("v.") or n.startswith("t.")
    if cond == "A_emb":   # minimal adaptation: embodiment embedding + state projection + output layer
        train_fn = lambda n: n.startswith("emb.") or n.startswith("sproj.") or n.startswith("out.")
    elif cond in ("C", "A_enc"):
        train_fn = lambda n: True
    else:
        train_fn = lambda n: not is_enc(n)
    for n, p in model.named_parameters():
        p.requires_grad = train_fn(n)
    head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
    params = head + enc
    groups = [{"params": head, "lr": lr, "base": lr}]
    if enc:
        groups.append({"params": enc, "lr": lr if cond == "C" else enc_lr, "base": lr if cond == "C" else enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    dl = torch.utils.data.DataLoader(FSData(P, "train", True), batch_size=bs, shuffle=True, num_workers=6, collate_fn=collate, drop_last=len(P["train"]) > bs)
    dlt = torch.utils.data.DataLoader(FSData(P, "test", False), batch_size=128, shuffle=False, num_workers=4, collate_fn=collate)
    step = 0
    model.train()
    while step < steps:
        for b in dl:
            step += 1
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
            rid = torch.full((len(b["ext"]),), row, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(b["ext"].to(dev), b["wri"].to(dev), b["has_w"].to(dev), b["state"].to(dev), tok(b["label"]).to(dev), rid)
            loss, _ = masked_l1(p.float(), b["tgt"].to(dev), b["valid"].to(dev), b["w"].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step >= steps:
                break
    model.eval()
    tot, hold, n, per = 0.0, 0.0, 0, torch.zeros(6, device=dev)
    with torch.no_grad():
        for b in dlt:
            rid = torch.full((len(b["ext"]),), row, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(b["ext"].to(dev), b["wri"].to(dev), b["has_w"].to(dev), b["state"].to(dev), tok(b["label"]).to(dev), rid).float()
            l, pj = masked_l1(p, b["tgt"].to(dev), b["valid"].to(dev), torch.ones(len(p), device=dev))
            z = torch.zeros_like(p); z[..., 5] = b["tgt"][:, :1, 5].to(dev)
            hold += masked_l1(z, b["tgt"].to(dev), b["valid"].to(dev), torch.ones(len(p), device=dev))[0].item() * len(p)
            tot += l.item() * len(p); n += len(p); per += pj * len(p)
    return dict(l1=tot / n, hold=hold / n, l1_joint=(per / n).cpu().numpy().round(3).tolist(), n_test=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=8)
    ap.add_argument("--Ns", default="10,50")
    ap.add_argument("--conds", default="A,B,C")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--tag", default="results")
    ap.add_argument("--repo_list", default="")
    a = ap.parse_args()
    dev = "cuda"
    bench = json.load(open(os.path.join(ROOT, "bench/repos.json")))
    info = {r: len(np.load(os.path.join(ROOT, "bench/actions", r.replace("/", "__") + ".npz")).files) // 2 for r in bench}
    Ns = [int(x) for x in a.Ns.split(",")]
    repos = [r for r in sorted(bench, key=lambda r: -info[r]) if info[r] >= max(Ns) + 10][: a.repos]
    if a.repo_list:
        repos = a.repo_list.split(",")
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
    out = os.path.join(CK, "fewshot"); os.makedirs(out, exist_ok=True)
    rows = []
    for seed in seeds:
      for repo in repos:
        for N in Ns:
            P = prepare(repo, N, a.n_test, seed)
            for cond in a.conds.split(","):
                torch.manual_seed(seed); random.seed(seed)
                t0 = time.time()
                r = run(P, cond, a.steps, a.lr if cond != "C" else 3e-4, a.bs, dev)
                r.update(repo=repo, N=N, cond=cond, seed=seed, steps=a.steps, lr=a.lr, n_train_frames=len(P["train"]), secs=round(time.time() - t0))
                rows.append(r)
                print(json.dumps(r), flush=True)
                pd.DataFrame(rows).to_parquet(os.path.join(out, a.tag + ".parquet"))
    D = pd.DataFrame(rows)
    D["rel"] = D.l1 / D.hold
    print(D.pivot_table(index="N", columns="cond", values=["l1", "rel"], aggfunc="mean").round(3))


if __name__ == "__main__":
    main()
