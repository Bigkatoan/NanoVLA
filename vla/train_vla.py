"""Stage 4: VLA pretraining on cleaned SO-100/101 data (clean_v2), encoders from stage 3 (align).

Sample  = one kept external-camera frame (+ wrist partner if any), its episode label, follower state at the
          lag-corrected time f*, and the future action chunk A[f* + k*fps/10], k=1..H  (10 Hz, H=16 -> 1.6 s).
Target  = joints 0-4: (A - S[f*]) / repo_delta_std ;  gripper: repo-min-max normalised absolute command.
Model   = memory [ext patch tokens, wrist patch tokens, text tokens, state token, embodiment token]
          -> ACT-style transformer decoder with H queries -> 6-dim chunk. L1 loss, episode weights.
Embodiment = one learned embedding per repo (calibration / setup); few-shot adds a new one.
"""
import argparse, os, json, random, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.io import decode_jpeg
from tokenizers import Tokenizer
from models import VisionViT, TextEncoder, n_params
from train_text import Enc, augment
from train_align import Aligner

ROOT = os.path.expanduser("~/VLA/data")
CK = os.path.expanduser("~/VLA/ckpt")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
H, CTRL_HZ = 16, 10


# ------------------------------------------------------------------ data
def load_actions(repos, adir):
    acts = {}
    for rid, repo in repos.items():
        f = os.path.join(adir, repo.replace("/", "__") + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            acts[rid] = {int(k[2:].split("_")[0]): (z[k], z[k.replace("action", "state")]) for k in z.files if k.endswith("_action")}
    return acts


def repo_stats(acts, fps):
    """per repo: joint-delta std at the chunk horizon, gripper p1/p99, state mean/std"""
    st = {}
    for rid, eps in acts.items():
        step = max(1, round(fps[rid] / CTRL_HZ))
        D, G, S = [], [], []
        for A, Sx in eps.values():
            if len(A) < 2:
                continue
            idx = np.arange(0, len(A), step)
            for k in (1, 4, 8, 16):
                j = np.clip(idx + k * step, 0, len(A) - 1)
                D.append(A[j, :5] - Sx[idx, :5])
            G.append(A[:, 5]); S.append(Sx)
        if not D:
            continue
        D, G, S = np.concatenate(D), np.concatenate(G), np.concatenate(S)
        st[rid] = dict(dstd=np.maximum(D.std(0), 1e-2).astype(np.float32), g_lo=np.float32(np.percentile(G, 1)), g_hi=np.float32(np.percentile(G, 99)),
                       s_mean=S.mean(0).astype(np.float32), s_std=np.maximum(S.std(0), 1e-2).astype(np.float32))
    return st


def build_samples(I, keep, lag, weight, labels, acts, fps, repos_ok):
    """anchor = kept frame of the first external cam of each episode (wrist partner attached if kept)"""
    I = I.assign(keep=keep, lag=lag, weight=weight)
    I = I[I.repo_id.isin(repos_ok)]
    out = []
    for (rid, ep), g in I.groupby(["repo_id", "ep"], sort=False):
        if (rid, ep) not in labels or rid not in acts or ep not in acts[rid]:
            continue
        cams = g.groupby("cam").wrist.first()
        ext = [c for c, w in cams.items() if w == 0] or list(cams.index)
        wr = [c for c, w in cams.items() if w == 1 and c not in ext[:1]]
        ge = g[(g.cam == ext[0]) & g.keep].sort_values("fidx")
        gw = g[g.cam == wr[0]].set_index("fidx") if wr else None
        step = max(1, round(fps[rid] / CTRL_HZ))
        T_ = len(acts[rid][ep][0])
        sstep = int(np.median(np.diff(g[g.cam == ext[0]].fidx.to_numpy()))) if len(g[g.cam == ext[0]]) > 1 else 1
        for r in ge.itertuples():
            fstar = int(np.clip(r.fidx + r.lag * sstep, 0, T_ - 1))
            if fstar + step >= T_:
                continue
            w_off = w_len = -1
            if gw is not None and r.fidx in gw.index and bool(gw.loc[r.fidx, "keep"]):
                w_off, w_len = int(gw.loc[r.fidx, "off"]), int(gw.loc[r.fidx, "len"])
            out.append((rid, ep, fstar, step, int(r.off), int(r.len), w_off, w_len, float(r.weight)))
    return np.array(out, dtype=np.float64)


class VLAData(torch.utils.data.Dataset):
    def __init__(self, S, acts, stats, labels, blob_path, train=True):
        self.S, self.acts, self.stats, self.labels, self.train = S, acts, stats, labels, train
        self.blob_path, self.blob = blob_path, None
        self.aug = T.Compose([T.RandomResizedCrop(112, scale=(0.8, 1.0), ratio=(1.2, 1.4), antialias=True),
                              T.RandomApply([T.ColorJitter(0.3, 0.3, 0.1, 0.0)], p=0.8)])

    def __len__(self):
        return len(self.S)

    def img(self, off, ln):
        if self.blob is None:
            self.blob = np.memmap(self.blob_path, dtype=np.uint8, mode="r")
        x = decode_jpeg(torch.from_numpy(np.array(self.blob[off: off + ln])))
        x = self.aug(x) if self.train else T.functional.resize(x, [112, 149], antialias=True)[..., 18:130]
        return (x.float() / 255 - MEAN) / STD

    def __getitem__(self, i):
        rid, ep, f, step, off, ln, woff, wln, w = self.S[i]
        rid, ep, f, step = int(rid), int(ep), int(f), int(step)
        A, Sx = self.acts[rid][ep]
        st = self.stats[rid]
        idx = f + step * np.arange(1, H + 1)
        valid = idx < len(A)
        idx = np.minimum(idx, len(A) - 1)
        tgt = np.empty((H, 6), np.float32)
        tgt[:, :5] = (A[idx, :5] - Sx[f, :5]) / st["dstd"]
        tgt[:, 5] = (A[idx, 5] - st["g_lo"]) / (st["g_hi"] - st["g_lo"] + 1e-6) * 2 - 1
        state = (Sx[f] - st["s_mean"]) / st["s_std"]
        ext = self.img(int(off), int(ln))
        has_w = woff >= 0
        wri = self.img(int(woff), int(wln)) if has_w else torch.zeros_like(ext)
        label = self.labels[(rid, ep)]
        if self.train and random.random() < 0.3:
            label = augment(label, random)
        return dict(ext=ext, wri=wri, has_w=torch.tensor(has_w), state=torch.from_numpy(np.clip(state, -5, 5).astype(np.float32)),
                    tgt=torch.from_numpy(np.clip(tgt, -10, 10)), valid=torch.from_numpy(valid), rid=torch.tensor(rid), w=torch.tensor(w, dtype=torch.float32), label=label)


def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in b[0] if k != "label"}
    out["label"] = [x["label"] for x in b]
    return out


# ------------------------------------------------------------------ model
class VLA(nn.Module):
    def __init__(self, venc, tenc, n_emb, d=256, layers=3, heads=8, ff=512):
        super().__init__()
        self.v, self.t = venc, tenc
        self.vproj = nn.Linear(venc.d, d)
        self.tproj = nn.Linear(tenc.d, d)
        self.sproj = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.emb = nn.Embedding(n_emb, d)
        self.ttype = nn.Embedding(5, d)          # ext, wrist, text, state, embodiment
        self.vpos = nn.Parameter(torch.zeros(1, 49, d)); nn.init.trunc_normal_(self.vpos, std=0.02)
        self.q = nn.Parameter(torch.randn(1, H, d) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d, heads, ff, 0.1, batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(enc_layer, 1, enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(d, heads, ff, 0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 6))

    def forward(self, ext, wri, has_w, state, ids, rid):
        B = ext.shape[0]
        _, pe = self.v(ext, torch.zeros(B, dtype=torch.long, device=ext.device))
        _, pw = self.v(wri, torch.ones(B, dtype=torch.long, device=ext.device))
        te, _, tpad = self.t(ids)
        dev = ext.device
        m_e = self.vproj(pe) + self.vpos + self.ttype.weight[0]
        m_w = self.vproj(pw) + self.vpos + self.ttype.weight[1]
        m_t = self.tproj(te) + self.ttype.weight[2]
        m_s = (self.sproj(state) + self.ttype.weight[3]).unsqueeze(1)
        m_b = (self.emb(rid) + self.ttype.weight[4]).unsqueeze(1)
        mem = torch.cat([m_e, m_w, m_t, m_s, m_b], 1)
        pad = torch.cat([torch.zeros(B, 49, dtype=torch.bool, device=dev), (~has_w).unsqueeze(1).expand(B, 49), tpad,
                         torch.zeros(B, 2, dtype=torch.bool, device=dev)], 1)
        mem = self.fuse(mem, src_key_padding_mask=pad)
        h = self.dec(self.q.expand(B, -1, -1), mem, memory_key_padding_mask=pad)
        return self.out(h)


def build_model(n_emb, align_ckpt, dev):
    s = torch.load(align_ckpt, map_location=dev)
    # checkpoints store absolute paths from training time: resolve the text encoder dir by name under the ckpt root
    tdir = s["args"]["text"]
    if not os.path.isdir(tdir):
        tdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(align_ckpt))), os.path.basename(os.path.normpath(tdir)))
    s["args"]["text"] = tdir
    ts = torch.load(os.path.join(tdir, "text_encoder.pt"), map_location=dev)
    venc, tenc = VisionViT(drop_path=0.05), TextEncoder(ts["cfg"]["vocab"], max_len=ts["cfg"]["max_len"])
    al = Aligner(venc, tenc); al.load_state_dict(s["model"])
    tok = Enc(Tokenizer.from_file(os.path.join(s["args"]["text"], "tokenizer.json")), ts["cfg"]["max_len"])
    return VLA(al.v, al.t, n_emb).to(dev), tok


def masked_l1(pred, tgt, valid, w):
    l = (pred - tgt).abs()
    l[..., 5] *= 0.5
    m = valid.unsqueeze(-1).float()
    per = (l * m).sum((1, 2)) / (m.sum((1, 2)) * 6 + 1e-6)
    return (per * w).sum() / (w.sum() + 1e-6), (l * m).sum((0, 1)) / (m.sum((0, 1)) + 1e-6)


@torch.no_grad()
def evaluate(model, dl, tok, dev, max_batches=40, unk_row=None):
    model.eval()
    tot, n, per, hold = 0.0, 0, torch.zeros(6, device=dev), 0.0
    for i, b in enumerate(dl):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rid = b["rid"].to(dev) if unk_row is None else torch.full_like(b["rid"], unk_row).to(dev)
            p = model(b["ext"].to(dev), b["wri"].to(dev), b["has_w"].to(dev), b["state"].to(dev), tok(b["label"]).to(dev), rid).float()
        l, pj = masked_l1(p, b["tgt"].to(dev), b["valid"].to(dev), torch.ones(len(p), device=dev))
        tot += l.item() * len(p); n += len(p); per += pj * len(p)
        z = torch.zeros_like(p); z[..., 5] = b["tgt"][:, :1, 5].to(dev)  # hold: joints stay, gripper keeps its next value
        hold += masked_l1(z, b["tgt"].to(dev), b["valid"].to(dev), torch.ones(len(p), device=dev))[0].item() * len(p)
    model.train()
    return dict(val_l1=round(tot / n, 4), val_l1_hold=round(hold / n, 4), val_l1_joint=[round(x, 3) for x in (per / n).tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc_lr", type=float, default=1e-5)
    ap.add_argument("--freeze_steps", type=int, default=3000)
    ap.add_argument("--align", default=os.path.join(CK, "align/align.pt"))
    ap.add_argument("--out", default=os.path.join(CK, "vla"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--emb_drop", type=float, default=0.0, help="prob. of replacing the repo embedding by the shared UNK row")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = "cuda"
    R = pd.read_parquet(os.path.join(ROOT, "packed/repos.parquet"))
    bench = set(json.load(open(os.path.join(ROOT, "bench/repos.json"))))
    repos = {int(r): p for r, p in zip(R.repo_id, R.repo) if p not in bench}
    fps = {int(r): int(f) for r, f in zip(R.repo_id, R.fps)}
    acts = load_actions(repos, os.path.join(ROOT, "clean_v2/actions_fixed"))
    stats = repo_stats(acts, fps)
    E = pd.read_parquet(os.path.join(ROOT, "clean_v2/episodes.parquet"))
    E = E[(E["drop"].fillna("") == "") & E.label.notna()]
    labels = {(int(r), int(e)): l for r, e, l in zip(E.repo_id, E.ep, E.label)}
    I = pd.read_parquet(os.path.join(ROOT, "packed/index.parquet"), columns=["repo_id", "ep", "cam", "wrist", "fidx", "off", "len"])
    Fm = pd.read_parquet(os.path.join(ROOT, "clean_v2/frames.parquet"))
    vval = set(json.load(open(os.path.join(CK, "vision/val_repos.json"))))
    ok = [r for r in stats]
    val_r = [r for r in ok if r in vval]
    tr_r = [r for r in ok if r not in vval]
    S_tr = build_samples(I, Fm.keep.values, Fm.lag.values, Fm.weight.values, labels, acts, fps, tr_r)
    S_va = build_samples(I, Fm.keep.values, Fm.lag.values, Fm.weight.values, labels, acts, fps, val_r)
    print(f"samples train {len(S_tr)} val {len(S_va)}; repos train {len(tr_r)} val {len(val_r)}", flush=True)
    json.dump(dict(stats={str(k): {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()} for k, v in stats.items()}), open(os.path.join(a.out, "repo_stats.json"), "w"))
    blob = os.path.join(ROOT, "packed/frames.bin")
    dl = torch.utils.data.DataLoader(VLAData(S_tr, acts, stats, labels, blob), batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                                     drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dlv = torch.utils.data.DataLoader(VLAData(S_va, acts, stats, labels, blob, train=False), batch_size=128, shuffle=True, num_workers=4, collate_fn=collate)
    n_emb = int(R.repo_id.max()) + 64  # spare rows for few-shot embodiments
    model, tok = build_model(n_emb, a.align, dev)
    print("VLA params %.2fM (head %.2fM)" % (n_params(model), n_params(model) - n_params(model.v) - n_params(model.t)), flush=True)
    enc_p = list(model.v.parameters()) + list(model.t.parameters())
    head_p = [p for n, p in model.named_parameters() if not (n.startswith("v.") or n.startswith("t."))]
    opt = torch.optim.AdamW([{"params": head_p, "lr": a.lr}, {"params": enc_p, "lr": 0.0}], weight_decay=0.05)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    step, t0, best = 0, time.time(), 1e9
    while step < a.steps:
        for b in dl:
            step += 1
            warm, dec = min(1, step / 1000), 0.5 * (1 + math.cos(math.pi * step / a.steps))
            opt.param_groups[0]["lr"] = a.lr * warm * dec
            opt.param_groups[1]["lr"] = 0.0 if step < a.freeze_steps else a.enc_lr * dec
            rid = b["rid"].to(dev)
            if a.emb_drop > 0:
                rid = torch.where(torch.rand(len(rid), device=dev) < a.emb_drop, torch.full_like(rid, n_emb - 2), rid)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(b["ext"].to(dev, non_blocking=True), b["wri"].to(dev, non_blocking=True), b["has_w"].to(dev), b["state"].to(dev),
                          tok(b["label"]).to(dev), rid)
            loss, _ = masked_l1(p.float(), b["tgt"].to(dev), b["valid"].to(dev), b["w"].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                r = dict(step=step, loss=round(loss.item(), 4), ips=round(100 * a.bs / (time.time() - t0), 1))
                t0 = time.time()
                if step % 2000 == 0 or step == a.steps:
                    r.update(evaluate(model, dlv, tok, dev, unk_row=(n_emb - 2) if a.emb_drop > 0 else None))
                    if r["val_l1"] < best:
                        best = r["val_l1"]
                        torch.save(dict(model=model.state_dict(), step=step, args=vars(a), metrics=r, n_emb=n_emb, unk_row=(n_emb - 2) if a.emb_drop > 0 else None), os.path.join(a.out, "vla.pt"))
                print(r, flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            if step >= a.steps:
                break
    print("done best val_l1", best)


if __name__ == "__main__":
    main()
