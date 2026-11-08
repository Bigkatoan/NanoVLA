"""From-scratch ViT-Tiny for SO-100/101 frames.

Losses:
  DINO     : student/teacher (EMA) self-distillation. Positive "second global view" is, per sample,
             another crop of the same frame / a temporal neighbour (+-1..2 steps) / the other camera
             at the same timestep. 4 local crops from the anchor.
  IDM      : inverse dynamics, predict per-repo-normalised joint delta between frame t and t+k
             (same camera, shared crop box) from the two CLS embeddings.
No horizontal flips (keeps left/right semantics for later language alignment).
"""
import argparse, math, os, json, time, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.io import decode_jpeg
from models import VisionViT, DINOHead, n_params

ROOT = os.path.expanduser("~/VLA/data/packed")
OUT = os.path.expanduser("~/VLA/ckpt/vision")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Frames(torch.utils.data.Dataset):
    def __init__(self, rows, I, repo_dstd, n_local=4, g=112, l=64):
        self.rows = rows
        self.I = I
        self.dstd = repo_dstd
        self.blob = None
        self.n_local = n_local
        cj = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.1)], p=0.8)
        self.glob1 = T.Compose([T.RandomResizedCrop(g, scale=(0.35, 1.0), ratio=(0.75, 1.33), antialias=True), cj, T.RandomGrayscale(0.2), T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=1.0)])
        self.glob2 = T.Compose([T.RandomResizedCrop(g, scale=(0.35, 1.0), ratio=(0.75, 1.33), antialias=True), cj, T.RandomGrayscale(0.2), T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.1), T.RandomSolarize(128, p=0.2)])
        self.loc = T.Compose([T.RandomResizedCrop(l, scale=(0.08, 0.35), ratio=(0.75, 1.33), antialias=True), cj, T.RandomGrayscale(0.2), T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.5)])
        self.mild_cj = T.ColorJitter(0.3, 0.3, 0.2, 0.05)
        self.g = g
        # numpy columns for speed
        self.off = I.off.to_numpy(); self.len = I.len.to_numpy()
        self.seq = I.seq.to_numpy(); self.pos = I.pos.to_numpy(); self.slen = I.seq_len.to_numpy()
        self.partner = I.partner.to_numpy(); self.wrist = I.wrist.to_numpy()
        self.repo = I.repo_id.to_numpy()
        self.state = np.stack(I.state.to_numpy()).astype(np.float32)

    def __len__(self):
        return len(self.rows)

    def img(self, r):
        if self.blob is None:
            self.blob = np.memmap(os.path.join(ROOT, "frames.bin"), dtype=np.uint8, mode="r")
        b = torch.from_numpy(np.array(self.blob[self.off[r]: self.off[r] + self.len[r]]))
        return decode_jpeg(b)  # uint8 [3,H,W]

    @staticmethod
    def norm(x):
        return (x.float() / 255 - MEAN) / STD

    def __getitem__(self, i):
        r = self.rows[i]
        a = self.img(r)
        # positive for the second global view
        u = random.random()
        pr, pview = r, self.wrist[r]
        if u < 0.35:
            k = random.choice([-2, -1, 1, 2])
            p2 = self.pos[r] + k
            if 0 <= p2 < self.slen[r]:
                pr = r + k
        elif u < 0.55 and self.partner[r] >= 0:
            pr = self.partner[r]
            pview = self.wrist[pr]
        b = a if pr == r else self.img(pr)
        g1 = self.norm(self.glob1(a))
        g2 = self.norm(self.glob2(b))
        locs = torch.stack([self.norm(self.loc(a)) for _ in range(self.n_local)])
        # IDM pair: same seq, k ahead, shared crop box
        k = random.randint(1, 3)
        j = r + k if self.pos[r] + k < self.slen[r] else r - k
        valid = 0 <= self.pos[r] + (j - r) < self.slen[r] and j != r
        if not valid:
            j = r
        c = self.img(j)
        top, left, h, w = T.RandomResizedCrop.get_params(a, scale=(0.7, 1.0), ratio=(1.0, 1.4))
        fa = self.norm(self.mild_cj(T.functional.resized_crop(a, top, left, h, w, [self.g, self.g], antialias=True)))
        fb = self.norm(self.mild_cj(T.functional.resized_crop(c, top, left, h, w, [self.g, self.g], antialias=True)))
        dstd = self.dstd[self.repo[r]]
        tgt = np.clip((self.state[j] - self.state[r]) / (dstd * abs(j - r) if j != r else 1), -5, 5)
        return dict(g1=g1, g2=g2, loc=locs, fa=fa, fb=fb, v1=int(self.wrist[r]), v2=int(pview),
                    dq=torch.from_numpy(tgt.astype(np.float32)), dq_valid=torch.tensor(float(valid)), kk=torch.tensor(float(abs(j - r))))


class DINOLoss(nn.Module):
    def __init__(self, out, ncrops, warm_tt=0.04, tt=0.06, warm_steps=10000, st=0.1, cm=0.9):
        super().__init__()
        self.st, self.cm, self.ncrops = st, cm, ncrops
        self.register_buffer("center", torch.zeros(1, out))
        self.warm_tt, self.tt, self.warm_steps = warm_tt, tt, warm_steps

    def forward(self, s_out, t_out, step):
        tt = self.warm_tt + (self.tt - self.warm_tt) * min(1, step / self.warm_steps)
        s = (s_out / self.st).chunk(self.ncrops)
        t = F.softmax((t_out - self.center) / tt, dim=-1).detach().chunk(2)
        loss, n = 0, 0
        for iq, q in enumerate(t):
            for v in range(len(s)):
                if v == iq:
                    continue
                loss = loss + torch.sum(-q * F.log_softmax(s[v], dim=-1), dim=-1).mean()
                n += 1
        with torch.no_grad():
            self.center.mul_(self.cm).add_(t_out.mean(0, keepdim=True), alpha=1 - self.cm)
        return loss / n


def spatial_pool(p, out=3):
    B, N, D = p.shape
    g = int(N ** 0.5)
    return F.adaptive_avg_pool2d(p.reshape(B, g, g, D).permute(0, 3, 1, 2), out).flatten(1)


def cosine(step, total, base, final, warm=0):
    if step < warm:
        return base * step / warm
    p = (step - warm) / max(1, total - warm)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def probe(model, ds_val, dev, n=6000):
    """Linear ridge probe: per-repo standardised joint state from frozen CLS (external cams). R^2 on held-out episodes."""
    model.eval()
    I = ds_val.I
    rows = [r for r in ds_val.rows if ds_val.wrist[r] == 0]
    rng = np.random.default_rng(0)
    rows = rng.choice(rows, min(n, len(rows)), replace=False)
    feats = []
    for i in range(0, len(rows), 256):
        x = torch.stack([ds_val.norm(T.functional.resize(ds_val.img(r), [112, 149], antialias=True)[..., 18:130]) for r in rows[i:i + 256]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, p = model(x, torch.zeros(len(x), dtype=torch.long, device=dev))
        feats.append(spatial_pool(p).float())
    X = torch.cat(feats)
    st = torch.from_numpy(ds_val.state[rows]).to(dev)
    rep = ds_val.repo[rows]
    Y = st.clone()
    for rp in np.unique(rep):
        m = torch.from_numpy(rep == rp).to(dev)
        Y[m] = (st[m] - st[m].mean(0)) / (st[m].std(0) + 1e-3)
    ep_key = I.ep.to_numpy()[rows] * 10000 + rep
    test = torch.from_numpy(np.isin(ep_key, rng.choice(np.unique(ep_key), len(np.unique(ep_key)) // 3, replace=False))).to(dev)
    Xm = X[~test].mean(0); Xs = X[~test].std(0) + 1e-5
    Xn = (X - Xm) / Xs
    Xa = torch.cat([Xn, torch.ones(len(Xn), 1, device=dev)], 1)
    A = Xa[~test]
    W = torch.linalg.solve(A.T @ A + 100.0 * torch.eye(A.shape[1], device=dev), A.T @ Y[~test])
    P = Xa[test] @ W
    r2 = 1 - ((P - Y[test]) ** 2).sum(0) / ((Y[test] - Y[test].mean(0)) ** 2).sum(0)
    model.train()
    return r2.cpu().numpy().round(3).tolist(), float(X.std(0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dim", type=int, default=4096)
    ap.add_argument("--idm_w", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    random.seed(0); torch.manual_seed(0)
    I = pd.read_parquet(os.path.join(ROOT, "index.parquet"))
    R = pd.read_parquet(os.path.join(ROOT, "repos.parquet"))
    dstd = np.stack(R.sort_values("repo_id").dstate_std.to_numpy()).astype(np.float32)
    rng = np.random.default_rng(0)
    val_repos = set(rng.choice(R.repo_id.to_numpy(), max(1, len(R) // 20), replace=False).tolist())
    is_val = I.repo_id.isin(val_repos).to_numpy()
    tr_rows = np.nonzero(~is_val)[0]
    va_rows = np.nonzero(is_val)[0]
    json.dump(sorted(int(x) for x in val_repos), open(os.path.join(OUT, "val_repos.json"), "w"))
    print(f"frames train {len(tr_rows)} val {len(va_rows)} repos {len(R)} (val {len(val_repos)})")
    ds = Frames(tr_rows, I, dstd)
    ds_val = Frames(va_rows, I, dstd)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dev = "cuda"
    student, teacher = VisionViT().to(dev), VisionViT(drop_path=0.0).to(dev)
    s_head, t_head = DINOHead(student.d, a.out_dim).to(dev), DINOHead(student.d, a.out_dim).to(dev)
    idm = nn.Sequential(nn.Linear(3 * 9 * student.d + 1, 512), nn.GELU(), nn.Linear(512, 6)).to(dev)
    teacher.load_state_dict(student.state_dict()); t_head.load_state_dict(s_head.state_dict())
    for p in list(teacher.parameters()) + list(t_head.parameters()):
        p.requires_grad = False
    print("vision encoder params %.2fM" % n_params(student))
    ncrops = 2 + ds.n_local
    crit = DINOLoss(a.out_dim, ncrops).to(dev)
    params_reg = [p for n, p in list(student.named_parameters()) + list(s_head.named_parameters()) + list(idm.named_parameters()) if p.requires_grad and p.ndim > 1]
    params_nr = [p for n, p in list(student.named_parameters()) + list(s_head.named_parameters()) + list(idm.named_parameters()) if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": params_reg}, {"params": params_nr, "weight_decay": 0.0}], lr=a.lr)
    step0 = 0
    ck = os.path.join(OUT, "last.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        student.load_state_dict(s["student"]); teacher.load_state_dict(s["teacher"]); s_head.load_state_dict(s["s_head"]); t_head.load_state_dict(s["t_head"])
        idm.load_state_dict(s["idm"]); opt.load_state_dict(s["opt"]); crit.center.copy_(s["center"]); step0 = s["step"]
        print("resumed at", step0)
    log = open(os.path.join(OUT, "log.jsonl"), "a")
    step, t0 = step0, time.time()
    warm = 3000
    while step < a.steps:
        for b in dl:
            if step >= a.steps:
                break
            lr = cosine(step, a.steps, a.lr, 1e-6, warm)
            wd = cosine(step, a.steps, 0.04, 0.4) if False else 0.04 + (0.4 - 0.04) * step / a.steps
            for i, g in enumerate(opt.param_groups):
                g["lr"] = lr
                if i == 0:
                    g["weight_decay"] = wd
            mom = 1 - (1 - 0.996) * (math.cos(math.pi * step / a.steps) + 1) / 2
            g1, g2 = b["g1"].to(dev, non_blocking=True), b["g2"].to(dev, non_blocking=True)
            loc = b["loc"].to(dev, non_blocking=True).flatten(0, 1)
            v1, v2 = b["v1"].to(dev), b["v2"].to(dev)
            vl = v1.repeat_interleave(ds.n_local)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    tc, _ = teacher(torch.cat([g1, g2]), torch.cat([v1, v2]))
                    t_out = t_head(tc).float()
                sg, _ = student(torch.cat([g1, g2]), torch.cat([v1, v2]))
                sl, _ = student(loc, vl)
                # reorder local outputs crop-major to match chunk() in the loss
                sl = sl.view(-1, ds.n_local, sl.shape[-1]).transpose(0, 1).flatten(0, 1)
                s_out = s_head(torch.cat([sg, sl])).float()
                dino = crit(s_out, t_out, step)
                fa, fb = b["fa"].to(dev, non_blocking=True), b["fb"].to(dev, non_blocking=True)
                _, pt = student(torch.cat([fa, fb]), torch.cat([v1, v1]))
                pa, pb = spatial_pool(pt).chunk(2)
                kk = b["kk"].to(dev).unsqueeze(1)
                pred = idm(torch.cat([pa, pb, pb - pa, kk / 3], 1)).float()
                m = b["dq_valid"].to(dev)
                idm_l = (F.smooth_l1_loss(pred, b["dq"].to(dev), reduction="none").mean(1) * m).sum() / m.sum().clamp(min=1)
                loss = dino + a.idm_w * idm_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_reg + params_nr, 3.0)
            if step < 1000:  # freeze last layer of the DINO head in the first steps (DINO trick)
                for p in s_head.last.parameters():
                    p.grad = None
            opt.step()
            with torch.no_grad():
                for ps, pt in zip(list(student.parameters()) + list(s_head.parameters()), list(teacher.parameters()) + list(t_head.parameters())):
                    pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
            step += 1
            if step % 100 == 0:
                ent = -(F.softmax((t_out - crit.center) / 0.04, -1) * F.log_softmax((t_out - crit.center) / 0.04, -1)).sum(-1).mean().item()
                r = dict(step=step, loss=round(loss.item(), 4), dino=round(dino.item(), 4), idm=round(idm_l.item(), 4), lr=round(lr, 7), t_ent=round(ent, 3),
                         ips=round(100 * a.bs / (time.time() - t0), 1))
                t0 = time.time()
                if step % 2500 == 0:
                    r2, fstd = probe(teacher, ds_val, dev)
                    r["probe_r2"] = r2; r["probe_mean"] = round(float(np.mean(r2)), 3); r["feat_std"] = round(fstd, 4)
                    torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(), s_head=s_head.state_dict(), t_head=t_head.state_dict(),
                                    idm=idm.state_dict(), opt=opt.state_dict(), center=crit.center, step=step, args=vars(a)), ck)
                print(r, flush=True)
                log.write(json.dumps(r) + "\n"); log.flush()
    torch.save(dict(encoder=teacher.state_dict(), args=vars(a), step=step), os.path.join(OUT, "vision_encoder.pt"))
    print("done")


if __name__ == "__main__":
    main()
