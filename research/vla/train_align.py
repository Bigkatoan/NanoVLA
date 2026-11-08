"""Stage 3: align the from-scratch vision and text encoders on cleaned data (clean_v2).

Episode-level pairs (clip of K kept frames, cleaned label).
  clip   : z_clip   = attn-pool over the K frames' pooled patch features      <-> text
  effect : z_effect = MLP(z_last - z_first)                                   <-> text
Loss: sigmoid (SigLIP-style) with multi-positive targets (same canonical label text = positive).
Batches are repo-grouped (R repos x M episodes) so same-scene/different-command episodes are hard negatives.
No hue jitter (colour is semantic), no flips (left/right is semantic).
Eval on held-out repos: text->clip R@1 within repo and global, attribute-swap test.
Writes ckpt_scratch/align/{align.pt,log.jsonl} and data/clean_v2/align_scores.parquet (per-episode label score).
"""
import argparse, os, json, random, re, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.io import decode_jpeg
from tokenizers import Tokenizer
from models import VisionViT, TextEncoder, n_params
from train_text import Enc, make_negative, augment

ROOT = os.path.expanduser("~/VLA/data")
CK = os.path.expanduser("~/VLA/ckpt")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def canon(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


class Episodes(torch.utils.data.Dataset):
    """item = one episode -> K frames (one camera, prefer external) + label"""

    def __init__(self, items, K=4, train=True):
        self.items, self.K, self.train = items, K, train
        self.blob = None
        self.aug = T.Compose([T.RandomResizedCrop(112, scale=(0.7, 1.0), ratio=(1.0, 1.4), antialias=True),
                              T.RandomApply([T.ColorJitter(0.3, 0.3, 0.1, 0.0)], p=0.8)]) if train else None

    def __len__(self):
        return len(self.items)

    def frame(self, off, ln):
        if self.blob is None:
            self.blob = np.memmap(os.path.join(ROOT, "packed/frames.bin"), dtype=np.uint8, mode="r")
        x = decode_jpeg(torch.from_numpy(np.array(self.blob[off: off + ln])))
        if self.train:
            x = self.aug(x)
        else:
            x = T.functional.resize(x, [112, 149], antialias=True)[..., 18:130]
        return (x.float() / 255 - MEAN) / STD

    def __getitem__(self, i):
        it = self.items[i]
        offs, lens = it["off"], it["len"]
        n = len(offs)
        if self.train:
            # first & last kept frames always included (effect), middle ones jittered
            mids = sorted(random.sample(range(1, n - 1), min(self.K - 2, max(0, n - 2)))) if n > 2 else []
            idx = [0] + mids + [n - 1]
        else:
            idx = np.linspace(0, n - 1, self.K).round().astype(int).tolist()
        while len(idx) < self.K:
            idx.append(idx[-1])
        x = torch.stack([self.frame(offs[j], lens[j]) for j in idx[: self.K]])
        return x, it["wrist"], i


def build_items(val_repos):
    I = pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["repo_id", "ep", "cam", "wrist", "fidx", "off", "len"])
    Fm = pd.read_parquet(os.path.join(ROOT, "clean_v2/frames.parquet"), columns=["keep"])
    E = pd.read_parquet(os.path.join(ROOT, "clean_v2/episodes.parquet"))
    E = E[(E["drop"].fillna("") == "") & E.label.notna()]
    I = I[Fm.keep.values]
    lab = {(int(r), int(e)): (l, float(w) if np.isfinite(w) else 1.0) for r, e, l, w in zip(E.repo_id, E.ep, E.label, E.get("weight", pd.Series(1.0, index=E.index)).fillna(1.0))}
    TC = pd.read_parquet(os.path.join(ROOT, "clean_v2/task_clusters.parquet"))
    task = {(int(r), int(e)): (t, tt) for r, e, t, tt in zip(TC.repo_id, TC.ep, TC.task, TC.task_text)}
    items = []
    for (rid, ep, cam), g in I.groupby(["repo_id", "ep", "cam"], sort=False):
        key = (int(rid), int(ep))
        if key not in lab or key not in task or len(g) < 4:
            continue
        g = g.sort_values("fidx")
        items.append(dict(repo_id=int(rid), ep=int(ep), cam=cam, wrist=int(g.wrist.iloc[0]), off=g.off.to_numpy(), len=g.len.to_numpy(),
                          label=lab[key][0], canon=task[key][0], task_text=task[key][1], weight=lab[key][1]))
    tr = [it for it in items if it["repo_id"] not in val_repos]
    va = [it for it in items if it["repo_id"] in val_repos]
    return tr, va


class RepoBatchSampler(torch.utils.data.Sampler):
    """R repos x M items per batch; prefers repos with several distinct labels"""

    def __init__(self, items, R=16, M=8, steps=1000):
        self.by_repo = {}
        for i, it in enumerate(items):
            self.by_repo.setdefault(it["repo_id"], []).append(i)
        self.repos = list(self.by_repo)
        nl = np.array([len({items[i]["canon"] for i in self.by_repo[r]}) for r in self.repos], float)
        self.p = np.sqrt(nl) / np.sqrt(nl).sum()
        self.R, self.M, self.steps = R, M, steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            rs = np.random.choice(len(self.repos), self.R, replace=False, p=self.p)
            b = []
            for r in rs:
                ids = self.by_repo[self.repos[r]]
                b += random.sample(ids, min(self.M, len(ids)))
            yield b


class Aligner(nn.Module):
    def __init__(self, venc, tenc, d=256):
        super().__init__()
        self.v, self.t = venc, tenc
        dv, dt = venc.d, tenc.d
        self.frame_proj = nn.Sequential(nn.Linear(dv * 2, 512), nn.GELU(), nn.Linear(512, d))
        self.tpos = nn.Parameter(torch.zeros(1, 8, d))
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.clip_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.eff_head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, d))
        self.txt_head = nn.Sequential(nn.LayerNorm(dt), nn.Linear(dt, d))
        self.txt_head_e = nn.Sequential(nn.LayerNorm(dt), nn.Linear(dt, d))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def video(self, x, wrist):
        B, K = x.shape[:2]
        c, p = self.v(x.flatten(0, 1), wrist.repeat_interleave(K))
        f = self.frame_proj(torch.cat([c, p.mean(1)], -1)).view(B, K, -1) + self.tpos[:, :K]
        z, _ = self.pool(self.pool_q.expand(B, -1, -1), f, f, need_weights=False)
        zc = F.normalize(self.clip_head(z[:, 0]), dim=-1)
        ze = F.normalize(self.eff_head(f[:, -1] - f[:, 0]), dim=-1)
        return zc, ze

    def text(self, ids):
        _, pooled, _ = self.t(ids)
        return F.normalize(self.txt_head(pooled), dim=-1), F.normalize(self.txt_head_e(pooled), dim=-1)


def sig_loss(zi, zt, pos, scale, bias, w=None):
    logits = zi @ zt.T * scale.exp() + bias
    y = 2 * pos.float() - 1
    l = -F.logsigmoid(y * logits)
    if w is not None:
        l = l * w.unsqueeze(1)
    return l.sum(1).mean()


@torch.no_grad()
def evaluate(model, va, enc, dev, n=600):
    model.eval()
    rng = random.Random(0)
    sub = rng.sample(va, min(n, len(va)))
    ds = Episodes(sub, train=False)
    Z, ZE = [], []
    for i in range(0, len(sub), 32):
        xs = torch.stack([ds[j][0] for j in range(i, min(i + 32, len(sub)))]).to(dev)
        w = torch.tensor([s["wrist"] for s in sub[i:i + 32]], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zc, ze = model.video(xs, w)
        Z.append(zc.float()); ZE.append(ze.float())
    Z, ZE = torch.cat(Z), torch.cat(ZE)
    labels = sorted({s["canon"] for s in sub})
    lid = torch.tensor([labels.index(s["canon"]) for s in sub], device=dev)
    raw = {s["canon"]: s["task_text"] for s in sub}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Tc, Te = model.text(enc([raw[l] for l in labels]).to(dev))
    Tc, Te = Tc.float(), Te.float()
    S = Z @ Tc.T + ZE @ Te.T
    res = {}
    res["global_v2t_r1"] = (S.argmax(1) == lid).float().mean().item()
    # within repo: rank the true label among the labels of the same repo
    rep = np.array([s["repo_id"] for s in sub])
    hits, tot, chance, major = 0, 0, 0.0, 0
    for r in np.unique(rep):
        m = np.nonzero(rep == r)[0]
        ls = sorted({int(lid[j]) for j in m})
        if len(ls) < 2:
            continue
        cnt = pd.Series([int(lid[j]) for j in m]).value_counts()
        chance += len(m) / len(ls); major += int(cnt.iloc[0])
        for j in m:
            sc = S[j, ls]
            hits += int(ls[int(sc.argmax())] == int(lid[j])); tot += 1
    res["within_repo_v2t_r1"] = hits / max(1, tot)
    res["within_repo_n"] = tot
    res["within_repo_chance"] = chance / max(1, tot); res["within_repo_majority"] = major / max(1, tot)
    # attribute swap: true text vs attribute-swapped text, same clip
    sw = [(j, make_negative(sub[j]["label"], rng)) for j in range(len(sub))]
    sw = [(j, t) for j, t in sw if t and canon(t) != canon(sub[j]["label"])]
    if sw:
        jj = torch.tensor([j for j, _ in sw], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a_c, a_e = model.text(enc([sub[j]["label"] for j, _ in sw]).to(dev))
            b_c, b_e = model.text(enc([t for _, t in sw]).to(dev))
        st = (Z[jj] * a_c.float()).sum(1) + (ZE[jj] * a_e.float()).sum(1)
        sf = (Z[jj] * b_c.float()).sum(1) + (ZE[jj] * b_e.float()).sum(1)
        res["attr_swap_acc"] = (st > sf).float().mean().item(); res["attr_swap_n"] = len(sw)
    model.train()
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}


@torch.no_grad()
def score_all(model, items, enc, dev):
    """per-episode label score: rank of own label among the repo's labels (low rank / low margin = suspicious label)"""
    model.eval()
    ds = Episodes(items, train=False)
    by_repo = {}
    for i, it in enumerate(items):
        by_repo.setdefault(it["repo_id"], []).append(i)
    rows = []
    for r, ids in by_repo.items():
        xs = torch.stack([ds[i][0] for i in ids]).to(dev)
        w = torch.tensor([items[i]["wrist"] for i in ids], device=dev)
        zc, ze = [], []
        for k in range(0, len(ids), 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a, b = model.video(xs[k:k + 32], w[k:k + 32])
            zc.append(a.float()); ze.append(b.float())
        zc, ze = torch.cat(zc), torch.cat(ze)
        labs = sorted({items[i]["canon"] for i in ids})
        raw = {items[i]["canon"]: items[i]["task_text"] for i in ids}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tc, te = model.text(enc([raw[l] for l in labs]).to(dev))
        S = zc @ tc.float().T + ze @ te.float().T
        for k, i in enumerate(ids):
            own = labs.index(items[i]["canon"])
            sc = S[k]
            others = torch.cat([sc[:own], sc[own + 1:]])
            rows.append(dict(repo_id=r, ep=items[i]["ep"], cam=items[i]["cam"], label=items[i]["label"], task=items[i]["canon"], score=float(sc[own]),
                             margin=float(sc[own] - others.max()) if len(others) else np.nan, n_labels=len(labs),
                             best_other=labs[int(sc.argmax())] if int(sc.argmax()) != own else ""))
    model.train()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--freeze_steps", type=int, default=2000)
    ap.add_argument("--vision", default=os.path.join(CK, "vision/vision_encoder.pt"))
    ap.add_argument("--text", default=os.path.join(CK, "text_v3"))
    a = ap.parse_args()
    out = os.path.join(CK, "align"); os.makedirs(out, exist_ok=True)
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    dev = "cuda"
    R_ = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet"))
    bench = set(json.load(open(os.path.join(ROOT, "bench/repos.json"))))
    vval = set(json.load(open(os.path.join(CK, "vision/val_repos.json"))))
    val_repos = {int(r) for r in vval if R_.set_index("repo_id").loc[r].repo not in bench}
    tr, va = build_items(val_repos)
    print(f"items train {len(tr)} val {len(va)} (val repos {len(val_repos)})")
    venc = VisionViT(drop_path=0.05).to(dev)
    vs = torch.load(a.vision, map_location=dev)
    venc.load_state_dict(vs.get("encoder", vs.get("teacher")))
    ts = torch.load(os.path.join(a.text, "text_encoder.pt"), map_location=dev)
    tenc = TextEncoder(ts["cfg"]["vocab"], max_len=ts["cfg"]["max_len"]).to(dev)
    tenc.load_state_dict(ts["model"])
    enc = Enc(Tokenizer.from_file(os.path.join(a.text, "tokenizer.json")), ts["cfg"]["max_len"])
    model = Aligner(venc, tenc).to(dev)
    print("params total %.2fM (vision %.2fM, text %.2fM)" % (n_params(model), n_params(venc), n_params(tenc)))
    enc_params = list(venc.parameters()) + list(tenc.parameters())
    head_params = [p for n, p in model.named_parameters() if not (n.startswith("v.") or n.startswith("t."))]
    opt = torch.optim.AdamW([{"params": head_params, "lr": a.lr}, {"params": enc_params, "lr": 0.0}], weight_decay=0.05)
    dl = torch.utils.data.DataLoader(Episodes(tr, K=a.K), batch_sampler=RepoBatchSampler(tr, a.R, a.M, a.steps), num_workers=8, pin_memory=True, prefetch_factor=4,
                                     persistent_workers=True)
    log = open(os.path.join(out, "log.jsonl"), "w")
    t0 = time.time()
    best = -1
    for step, (x, wrist, ids) in enumerate(dl, 1):
        warm = min(1, step / 500)
        dec = 0.5 * (1 + math.cos(math.pi * step / a.steps))
        opt.param_groups[0]["lr"] = a.lr * warm * dec
        opt.param_groups[1]["lr"] = 0.0 if step < a.freeze_steps else a.enc_lr * dec
        its = [tr[i] for i in ids.tolist()]
        labels = [it["label"] if random.random() < 0.7 else augment(it["label"], random) for it in its]  # paraphrase-style text aug
        cn = [it["canon"] for it in its]
        pos = torch.tensor([[ci == cj for cj in cn] for ci in cn], device=dev)
        w = torch.tensor([it["weight"] for it in its], device=dev)
        x, wrist = x.to(dev, non_blocking=True), wrist.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zc, ze = model.video(x, wrist)
            tc, te = model.text(enc(labels).to(dev))
            # hard text negatives: attribute-swapped versions of the batch labels (never positives)
            negs = [make_negative(l, random) for l in labels]
            bl = {canon(l) for l in labels}
            nv = [n_ is not None and canon(n_) not in bl for n_ in negs]
            tn_c, tn_e = model.text(enc([n_ if v else l for n_, v, l in zip(negs, nv, labels)]).to(dev))
        zc, ze, tc, te, tn_c, tn_e = [t.float() for t in (zc, ze, tc, te, tn_c, tn_e)]
        nvm = torch.tensor(nv, device=dev)
        pos_full = torch.cat([pos, torch.zeros_like(pos)], 1)
        tc_all = torch.cat([tc, tn_c], 0)
        te_all = torch.cat([te, tn_e], 0)
        mask_cols = torch.cat([torch.ones(len(tc), dtype=torch.bool, device=dev), nvm])
        l_clip = sig_loss(zc, tc_all[mask_cols], pos_full[:, mask_cols], model.logit_scale, model.logit_bias, w)
        l_eff = sig_loss(ze, te_all[mask_cols], pos_full[:, mask_cols], model.logit_scale, model.logit_bias, w)
        loss = l_clip + 0.5 * l_eff
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            r = dict(step=step, loss=round(loss.item(), 4), clip=round(l_clip.item(), 4), eff=round(l_eff.item(), 4), scale=round(model.logit_scale.exp().item(), 2),
                     ips=round(100 * len(ids) / (time.time() - t0), 1))
            t0 = time.time()
            if step % 1000 == 0 or step == a.steps:
                r.update(evaluate(model, va, enc, dev))
                if r["within_repo_v2t_r1"] + r["global_v2t_r1"] > best:
                    best = r["within_repo_v2t_r1"] + r["global_v2t_r1"]
                    torch.save(dict(model=model.state_dict(), step=step, args=vars(a), metrics=r), os.path.join(out, "align.pt"))
            print(r, flush=True)
            log.write(json.dumps(r) + "\n"); log.flush()
    s = torch.load(os.path.join(out, "align.pt"), map_location=dev)
    model.load_state_dict(s["model"])
    print("best", s["metrics"])
    sc = score_all(model, tr + va, enc, dev)
    sc.to_parquet(os.path.join(ROOT, "clean_v2/align_scores.parquet"))
    print("scored", len(sc), "episode-cams; suspicious (own label not top within repo):", int((sc.margin < 0).sum()))


if __name__ == "__main__":
    main()
