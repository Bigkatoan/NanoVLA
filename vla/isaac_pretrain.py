"""Multi-task adaptation of NanoVLA to the Isaac tabletop: fine-tune VLA v1 on all pretraining demos recorded by
isaac_bench/traj_tool.py (pick / pad / tray, many objects and instructions; the bowl task is held out), with the
best few-shot recipe (state dropout, localisation loss, encoder fine-tune). Saves a checkpoint that later few-shot
runs start from (keep_row: the robot embedding learned here is kept).

    python isaac_pretrain.py --bench isaac_pre_all --steps 20000 --out ~/VLA/ckpt/vla_isaac/vla.pt
    # wrist-camera-only robot (front camera never read), strong image augmentation:
    python isaac_pretrain.py --bench isaac_pre_all --cams wrist --aug strong --steps 20000 --out ~/VLA/ckpt/vla_isaac_wrist/vla.pt

--n_val K holds out K episodes and reports the validation chunk L1 (vs the 'hold still' baseline) at the end.
"""
import argparse, json, os, random, time
import numpy as np, torch
import fewshot as fs
import sim_fewshot as sf
from sim_fewshot import finetune, CK, ROOT
from isaac_policy_server import joint_goal_labels
from train_vla import masked_l1, collate


@torch.no_grad()
def evaluate(model, tok, row, P, dev, split="test"):
    dl = torch.utils.data.DataLoader(fs.FSData(P, split, False), batch_size=128, shuffle=False, num_workers=4, collate_fn=collate)
    model.eval(); tot = hold = n = 0.0
    for b in dl:
        rid = torch.full((len(b["ext"]),), row, device=dev); one = torch.ones(len(rid), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(b["ext"].to(dev), b["wri"].to(dev), b["has_w"].to(dev), b["state"].to(dev), tok(b["label"]).to(dev), rid).float()
        tg, va = b["tgt"].to(dev), b["valid"].to(dev)
        z = torch.zeros_like(p); z[..., 5] = tg[:, :1, 5]
        tot += masked_l1(p, tg, va, one)[0].item() * len(p); hold += masked_l1(z, tg, va, one)[0].item() * len(p); n += len(p)
    return dict(val_l1=round(tot / n, 4), val_hold=round(hold / n, 4), val_rel=round(tot / hold, 4), n_val=int(n))


def fs_count(repo):
    """number of usable episodes (same filter as fewshot.prepare)"""
    import pandas as pd
    name = repo.replace("/", "__"); z = np.load(os.path.join(fs.ROOT_BENCH, "actions", name + ".npz"))
    Fr = pd.read_parquet(os.path.join(fs.ROOT_BENCH, "frames", name + ".parquet"), columns=["ep", "fps"])
    fps = int(Fr.fps.iloc[0]); have = set(Fr.ep)
    return sum(1 for k in z.files if k.endswith("_action") and len(z[k]) > 2 * fps and int(k[2:].split("_")[0]) in have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="isaac_pre_all")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--enc_lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(CK, "vla_isaac/vla.pt"))
    ap.add_argument("--cams", default="both", choices=fs.CAM_MODES, help="wrist: wrist camera only (front never read)")
    ap.add_argument("--wrist_slot", default="ext", choices=fs.WRIST_SLOTS, help="cams=wrist: model slot of the wrist image (see fewshot.wrist_views)")
    ap.add_argument("--aug", default="base", choices=("base", "strong"), help="strong: rotation/crop/colour/blur/JPEG/noise augmentation")
    ap.add_argument("--n_val", type=int, default=0, help="hold out this many episodes for a validation L1")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    fs.ROOT_BENCH = os.path.join(ROOT, a.bench); sf.goal_labels = joint_goal_labels
    repo = json.load(open(os.path.join(ROOT, a.bench, "repos.json")))[0]
    if a.n_val:
        P = fs.prepare(repo, fs_count(repo) - a.n_val, a.n_val, a.seed, cams=a.cams)
    else:
        P = fs.prepare(repo, 10 ** 6, 0, a.seed, cams=a.cams)
    P["aug"], P["wrist_slot"] = a.aug, a.wrist_slot
    print(json.dumps(dict(cams=a.cams, wrist_slot=a.wrist_slot if a.cams == "wrist" else None, aug=a.aug, episodes=P["n_train_eps"], samples=len(P["train"]), tasks=len(set(P["task"].values())))), flush=True)
    t0 = time.time()
    hist = []
    model, tok, row = finetune(P, "A_v1enc", a.steps, a.lr, 64, "cuda", os.path.join(CK, "vla/vla.pt"), a.enc_lr, 0.5, 1.0, hist=hist)
    for k in range(0, len(hist), 250):
        print(json.dumps(dict(step=k + 250, train_l1=round(float(np.mean(hist[k:k + 250])), 4))), flush=True)
    ev = evaluate(model, tok, row, P, "cuda") if a.n_val else {}
    if ev:
        print(json.dumps(ev), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(model=model.state_dict(), n_emb=model.emb.weight.shape[0], keep_row=True, unk_row=None, st=P["st"], coef=P["coef"],
                    steps=a.steps, bench=a.bench, episodes=P["n_train_eps"],
                    cams=a.cams, wrist_slot=a.wrist_slot, aug=a.aug, train_l1_last=float(np.mean(hist[-250:])), **ev), a.out)
    print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
