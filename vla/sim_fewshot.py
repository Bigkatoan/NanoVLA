"""Closed-loop success-rate benchmark: few-shot fine-tune on N simulated SO-101 demos, then roll the policy out.

Policy loop: every 0.1 s (3 ctrl steps @30 Hz) the policy can re-plan; it predicts a 1.6 s chunk (16 x 10 Hz),
the first EXEC chunk steps are executed with linear interpolation, then it re-plans. Episode limit 25 s.
Success = block centre inside the target square, released, gripper open (sim_env.success).
"""
import argparse, os, json, random, time, math, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, pandas as pd, torch
import torchvision.transforms.v2 as T
import fewshot as fs
from train_vla import masked_l1, H, collate, MEAN, STD
import sim_env as se

ROOT = os.path.expanduser("~/VLA/data")
CK = os.path.expanduser("~/VLA/ckpt")


def goal_labels(P):
    """per training episode: gripper-site xy at the grasp (first close after opening) and at the release, via FK"""
    import mujoco
    env = se.SO101Env(n_cubes=1, seed=0); lab = {}
    for e, (A, S) in P["acts"].items():
        opened = np.nonzero(A[:, 5] > 0.8)[0]
        if not len(opened):
            continue
        tc = next((t for t in range(opened[0], len(A)) if A[t, 5] < 0.4), None)
        if tc is None:
            continue
        tr = next((t for t in range(tc, len(A)) if A[t, 5] > 0.8), len(A) - 1)
        pts = []
        for t in (tc, tr):
            env.d.qpos[env.qadr] = S[t]; mujoco.mj_kinematics(env.m, env.d); pts.append(env.ee_pos()[:2].copy())
        lab[e] = np.r_[pts[0], pts[1]].astype(np.float32)
    env.close()
    return lab


class FSDataAux(fs.FSData):
    def __init__(self, P, split, train, goals, state_drop=0.0):
        super().__init__(P, split, train); self.goals, self.state_drop = goals, state_drop

    def __getitem__(self, i):
        d = super().__getitem__(i)
        e = self.S[i][0]
        g = self.goals.get(e)
        d["aux"] = torch.from_numpy((g - np.array([0.2, 0.0, 0.2, 0.0], np.float32)) * 10) if g is not None else torch.zeros(4)
        d["aux_ok"] = torch.tensor(g is not None)
        if self.train and random.random() < self.state_drop:
            d["state"] = torch.zeros_like(d["state"])
        return d


def finetune(P, cond, steps, lr, bs, dev, vla_ckpt, enc_lr=3e-5, state_drop=0.0, aux_w=0.0, hist=None):
    """hist: optional list, receives the action loss of every step (logging only)"""
    os.environ["VLA_CKPT"] = vla_ckpt
    base = "A" if cond.startswith("A") else ("B" if cond.startswith("B") else "C")
    model, tok, row = fs.make_model(base, dev)
    is_enc = lambda n: n.startswith("v.") or n.startswith("t.")
    enc_ft = cond == "C" or cond.endswith("enc")
    for n, p in model.named_parameters():
        p.requires_grad = enc_ft or not is_enc(n)
    head = [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)]
    params = head + enc
    groups = [{"params": head, "base": lr}] + ([{"params": enc, "base": lr if cond == "C" else enc_lr}] if enc else [])
    aux_head = None
    feats = {}
    if aux_w > 0:
        d_ = model.vproj.out_features
        aux_head = torch.nn.Sequential(torch.nn.LayerNorm(d_), torch.nn.Linear(d_, 256), torch.nn.GELU(), torch.nn.Linear(256, 4)).to(dev)
        groups[0]["params"] = list(groups[0]["params"]) + list(aux_head.parameters())
        model.fuse.register_forward_hook(lambda m, i, o: feats.__setitem__("mem", o))
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.05)
    goals = goal_labels(P) if aux_w > 0 else {}
    dl = torch.utils.data.DataLoader(FSDataAux(P, "train", True, goals, state_drop), batch_size=bs, shuffle=True, num_workers=6, collate_fn=collate, drop_last=len(P["train"]) > bs)
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
            if hist is not None:
                hist.append(loss.item())
            if aux_head is not None:
                hw = b["has_w"].to(dev)
                if hw.all():
                    vis = feats["mem"][:, :98].float().mean(1)            # visual tokens only (ext + wrist) after fusion
                else:                                                     # skip masked (absent) wrist tokens
                    wm = torch.cat([torch.ones_like(hw)[:, None].expand(-1, 49), hw[:, None].expand(-1, 49)], 1).float()
                    vis = (feats["mem"][:, :98].float() * wm[..., None]).sum(1) / wm.sum(1, keepdim=True)
                ok = b["aux_ok"].to(dev)
                if ok.any():
                    loss = loss + aux_w * torch.nn.functional.smooth_l1_loss(aux_head(vis)[ok], b["aux"].to(dev)[ok])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step >= steps:
                break
    model.eval()
    return model, tok, row


def prep_img(a):
    x = torch.from_numpy(a).permute(2, 0, 1)
    x = T.functional.resize(x, [112, 149], antialias=True)[..., 18:130]
    return (x.float() / 255 - MEAN) / STD


@torch.no_grad()
def rollout_te(model, tok, row, P, env, text, dev, max_s=25.0, video=None, m=0.1):
    """ACT-style temporal ensembling: re-plan every 0.1 s, action = exp-weighted mean of all chunk predictions for this tick"""
    st = P["st"]; coef = P["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    env.reset(); cur = env.state().copy(); ids = tok([text]).to(dev)
    buf = {}  # tick -> list of predicted command targets (oldest first)
    tick = 0
    while env.t < int(max_s * 30):
        S = env.state(); f, w = env.render("front"), env.render("wrist")
        if video is not None:
            video.append(np.concatenate([f, w], 1))
        ext, wri = prep_img(f)[None].to(dev), prep_img(w)[None].to(dev)
        s_n = torch.from_numpy(np.clip((S - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(ext, wri, torch.tensor([True], device=dev), s_n, ids, torch.tensor([row], device=dev)).float()[0].cpu().numpy()
        tgt = np.empty((H, 6)); tgt[:, :5] = S[:5] + p[:, :5] * st["dstd"]; tgt[:, 5] = (p[:, 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
        tgt = (tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)
        for k in range(H):
            buf.setdefault(tick + k + 1, []).append(tgt[k])
        preds = np.array(buf.pop(tick + 1))
        wts = np.exp(-m * np.arange(len(preds))); nxt = (preds * wts[:, None]).sum(0) / wts.sum()
        for i in range(3):
            env.step(cur + (nxt - cur) * (i + 1) / 3)
            if env.success():
                return True, env.t / 30
        cur = nxt; tick += 1
    return False, max_s


@torch.no_grad()
def rollout(model, tok, row, P, env, text, dev, exec_steps=4, max_s=25.0, video=None):
    if exec_steps == 0:
        return rollout_te(model, tok, row, P, env, text, dev, max_s, video)
    st = P["st"]; coef = P["coef"]
    a_s, b_s = coef[:, 0], coef[:, 1]
    env.reset()
    cur = env.state().copy()                     # last commanded target (rad)
    t_max = int(max_s * 30)
    ids = tok([text]).to(dev)
    while env.t < t_max:
        S = env.state()
        f, w = env.render("front"), env.render("wrist")
        if video is not None:
            video.append(np.concatenate([f, w], 1))
        ext, wri = prep_img(f)[None].to(dev), prep_img(w)[None].to(dev)
        s_n = torch.from_numpy(np.clip((S - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(ext, wri, torch.tensor([True], device=dev), s_n, ids, torch.tensor([row], device=dev)).float()[0].cpu().numpy()
        # back to follower-unit targets, then to command units (invert the leader->follower affine map)
        tgt_s = np.empty((H, 6))
        tgt_s[:, :5] = S[:5] + p[:, :5] * st["dstd"]
        tgt_s[:, 5] = (p[:, 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
        tgt_c = (tgt_s - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)
        for k in range(exec_steps):
            nxt = tgt_c[k]
            for i in range(3):                     # 10 Hz chunk -> 30 Hz control, linear interpolation
                env.step(cur + (nxt - cur) * (i + 1) / 3)
                if env.success():
                    return True, env.t / 30
            cur = nxt
    return False, max_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="10,20,50")
    ap.add_argument("--conds", default="A_v2,A_v1,B,C")
    ap.add_argument("--rollouts", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--exec_steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="sim_sr")
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--enc_lr", type=float, default=3e-5)
    ap.add_argument("--state_drop", type=float, default=0.0)
    ap.add_argument("--aux_w", type=float, default=0.0)
    ap.add_argument("--bench", default="sim_bench")
    a = ap.parse_args()
    dev = "cuda"
    fs.ROOT_BENCH = os.path.join(ROOT, a.bench)
    repo = json.load(open(os.path.join(ROOT, a.bench, "repos.json")))[0]
    text = "Pick up the red cube and place it in the blue square."
    ck = {"A_v1": os.path.join(CK, "vla/vla.pt"), "A_v2": os.path.join(CK, "vla_v2/vla.pt"), "A_v2enc": os.path.join(CK, "vla_v2/vla.pt"),
          "A_v1enc": os.path.join(CK, "vla/vla.pt"), "B": os.path.join(CK, "vla/vla.pt"), "Benc": os.path.join(CK, "vla/vla.pt"), "C": os.path.join(CK, "vla/vla.pt")}
    out = os.path.join(CK, "sim_fewshot"); os.makedirs(out, exist_ok=True)
    rows = []
    for N in [int(x) for x in a.Ns.split(",")]:
        P = fs.prepare(repo, N, 0, a.seed)
        for cond in a.conds.split(","):
            torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
            t0 = time.time()
            model, tok, row = finetune(P, cond, a.steps, a.lr if cond != "C" else 3e-4, a.bs, dev, ck[cond], a.enc_lr, a.state_drop, a.aux_w)
            t_ft = time.time() - t0
            env = se.SO101Env(n_cubes=1, seed=10_000 + a.seed)   # evaluation layouts disjoint from demo layouts
            succ, times, vids = [], [], []
            for i in range(a.rollouts):
                v = [] if i < a.videos else None
                s, tt = rollout(model, tok, row, P, env, text, dev, a.exec_steps, video=v)
                succ.append(s); times.append(tt)
                if v:
                    vids.append(np.stack(v))
            env.close()
            sr = float(np.mean(succ)); lo, hi = wilson(sum(succ), len(succ))
            r = dict(cond=cond, N=N, state_drop=a.state_drop, aux_w=a.aux_w, success_rate=sr, ci95=[lo, hi], n=len(succ), mean_time_success=float(np.mean([t for s, t in zip(succ, times) if s])) if any(succ) else None,
                     finetune_s=round(t_ft), eval_s=round(time.time() - t0 - t_ft), steps=a.steps, exec_steps=a.exec_steps)
            rows.append(r); print(json.dumps(r), flush=True)
            pd.DataFrame(rows).to_json(os.path.join(out, a.tag + ".json"), orient="records", indent=1)
            for j, v in enumerate(vids):
                np.save(os.path.join(out, f"{a.tag}_{cond}_N{N}_vid{j}.npy"), v)
            del model; torch.cuda.empty_cache()
    print(pd.DataFrame(rows).pivot_table(index="N", columns="cond", values="success_rate"))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


if __name__ == "__main__":
    main()
