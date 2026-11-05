"""Multi-task adaptation of NanoVLA to the Isaac tabletop: fine-tune VLA v1 on all pretraining demos recorded by
isaac_bench/traj_tool.py (pick / pad / tray, many objects and instructions; the bowl task is held out), with the
best few-shot recipe (state dropout, localisation loss, encoder fine-tune). Saves a checkpoint that later few-shot
runs start from (keep_row: the robot embedding learned here is kept).

    python isaac_pretrain.py --bench isaac_pre_all --steps 20000 --out ~/VLA/ckpt/vla_isaac/vla.pt
"""
import argparse, json, os, random, time
import numpy as np, torch
import fewshot as fs
import sim_fewshot as sf
from sim_fewshot import finetune, CK, ROOT
from isaac_policy_server import joint_goal_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="isaac_pre_all")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--enc_lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(CK, "vla_isaac/vla.pt"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    fs.ROOT_BENCH = os.path.join(ROOT, a.bench); sf.goal_labels = joint_goal_labels
    repo = json.load(open(os.path.join(ROOT, a.bench, "repos.json")))[0]
    P = fs.prepare(repo, 10 ** 6, 0, a.seed)
    print(json.dumps(dict(episodes=P["n_train_eps"], samples=len(P["train"]), tasks=len(set(P["task"].values())))), flush=True)
    t0 = time.time()
    model, tok, row = finetune(P, "A_v1enc", a.steps, a.lr, 64, "cuda", os.path.join(CK, "vla/vla.pt"), a.enc_lr, 0.5, 1.0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(model=model.state_dict(), n_emb=model.emb.weight.shape[0], keep_row=True, unk_row=None, st=P["st"], coef=P["coef"],
                    steps=a.steps, bench=a.bench, episodes=P["n_train_eps"]), a.out)
    print(json.dumps(dict(saved=a.out, secs=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
