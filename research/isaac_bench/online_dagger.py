"""GCT stage 2: online DAgger in Isaac Lab, end-to-end on one GPU in ONE process (design doc, 2026-11-06).

Sim (8-12 envs, DR + room), the NanoVLA-G policy (10 Hz, per-env memory), the scripted expert (shadow mode) and the
optimizer all live here. Each episode:
  * the POLICY drives (start with prob 0.7); the expert's node machine waits (frozen)
  * the EXPERT takes over when the policy stalls (no progress towards the expert's current sub-goal for 1.5 s),
    after a long policy segment with prob beta (0.5 -> 0.1), and drives 1-2.5 s from the state the policy left
  * recorded: frames (JPEG q90, like the offline data), joints, executed commands, which steps the policy drove,
    privileged localisation labels; the policy-driven commands are masked out of the action loss
Training is interleaved: one gradient step every --train_every control steps on half online windows, half
offline demos (stage-1 data). The acting network is refreshed from the EMA weights every 200 updates.
TensorBoard: online success (all / without intervention), intervention rate, policy share, losses.

    python online_dagger.py --init ~/VLA/ckpt/nanovla_g/vlag.pt --bench isaac_gct_v1 --hours 10 --out ~/VLA/ckpt/nanovla_g_online/vlag.pt
"""
import argparse, io, json, math, os, random, sys, time
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--bench", default="isaac_gct_v1")
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--hours", type=float, default=10.0)
ap.add_argument("--max_episodes", type=int, default=100000)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--run", default="gct_s2")
ap.add_argument("--out", default=os.path.expanduser("~/VLA/ckpt/nanovla_g_online/vlag.pt"))
ap.add_argument("--save_dir", default=os.path.expanduser("~/VLA/data/gct_online"), help="online episodes are also written here (bench format)")
ap.add_argument("--p_policy_start", type=float, default=0.7)
ap.add_argument("--beta0", type=float, default=0.5)
ap.add_argument("--beta1", type=float, default=0.1)
ap.add_argument("--pol_long", type=int, default=120, help="control steps after which a policy segment may be cut (prob beta)")
ap.add_argument("--stall", type=int, default=45, help="no 1 cm progress towards the expert sub-goal within this many steps -> expert takes over")
ap.add_argument("--exp_steps", default="30,75")
ap.add_argument("--max_s", type=float, default=30.0)
ap.add_argument("--train_every", type=int, default=6)
ap.add_argument("--warm_episodes", type=int, default=16)
ap.add_argument("--bs", type=int, default=6)
ap.add_argument("--window", type=int, default=48)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--enc_lr", type=float, default=1e-5)
ap.add_argument("--buffer", type=int, default=3000, help="online episodes kept for training")
ap.add_argument("--save_every_s", type=int, default=1800)
ap.add_argument("--n_val", type=int, default=50)
a = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True).app

HERE = os.path.dirname(os.path.abspath(__file__)); VLA_DIR = os.path.expanduser("~/VLA/research/vla")
sys.path.insert(0, HERE); sys.path.insert(1, VLA_DIR)
import numpy as np, torch, torch.nn.functional as F
import tabletop as tt
from traj_core import TrajRunner
import nanovla_g as G, nanovla2 as v2, nanovla_r as R
from train_text import clean
import numpy.core, numpy.core.multiarray, numpy.core.numeric                 # numpy-2 pickles (checkpoints) under numpy 1.26
sys.modules.setdefault("numpy._core", numpy.core); sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray); sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import decode_jpeg

GROUPS = {"place": ("specs/place.json", ("pad", "tray", "container", "rel")), "pick": ("specs/pick.json", ("pick",)),
          "push": ("specs/push.json", ("push", "topple")), "rotate": ("specs/rotate.json", ("rotate",))}
ORDER = ["place", "pick", "place", "push", "rotate", "pick", "place", "push", "rotate", "place", "pick", "push"]


def jpeg(x, q=90):
    b = io.BytesIO(); Image.fromarray(x).save(b, format="JPEG", quality=q); return b.getvalue()


def main():
    rng = np.random.default_rng(a.seed); random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"; tb = SummaryWriter(os.path.join(os.path.expanduser("~/VLA/tb"), a.run)); tb.add_text("config", "\n".join(f"{k}: {v}" for k, v in vars(a).items()))
    # ---------------- offline data (half of every batch) + model
    R.T_MAX = a.window
    Poff = v2.prepare(a.bench, a.n_val, 0); Poff["train_eps"] = sorted({int(e) for e, _ in Poff["train"]})
    zf = np.load(os.path.join(v2.ROOT, a.bench, "actions", json.load(open(os.path.join(v2.ROOT, a.bench, "repos.json")))[0].replace("/", "__") + ".npz"))
    Poff["pol"] = {int(k[2:].split("_")[0]): zf[k].astype(bool) for k in zf.files if k.endswith("_pol")}
    st, coef = Poff["st"], Poff["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    model, tok = G.build(dev, st, init=""); model.load_state_dict(ck["model"])
    for n_, p in model.vis.v.named_parameters():
        if n_.startswith(("proj.", "pos", "cls", "view_emb")) or any(n_.startswith(f"blocks.{i}.") for i in range(4)):
            p.requires_grad = False
    for p in model.zproj.parameters():
        p.requires_grad = False
    is_enc = lambda n_: n_.startswith("vis.") or n_.startswith("t.")
    enc = [p for n_, p in model.named_parameters() if p.requires_grad and is_enc(n_)]; head = [p for n_, p in model.named_parameters() if p.requires_grad and not is_enc(n_)]
    opt = torch.optim.AdamW([{"params": head, "lr": a.lr}, {"params": enc, "lr": a.enc_lr}], weight_decay=0.05)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    actor, _ = G.build(dev, st, init=""); actor.load_state_dict(ck["model"]); actor.eval()
    W = dict(fwd=0.5, phase=0.1, loc=0.5, prog=0.1, div=0.05)
    print(json.dumps(dict(event="model", init=a.init, offline_eps=len(Poff["train_eps"]))), flush=True)
    # ---------------- sim + one expert runner per task group
    env = tt.Tabletop(num_envs=a.num_envs, seed=a.seed, task_mix=tuple(t for g in GROUPS.values() for t in g[1]))
    grp = [ORDER[i % len(ORDER)] for i in range(env.n)]
    def sample_for(i):
        while True:
            E = env.sample_episode()
            if E["task"] in GROUPS[grp[i]][1]:
                return E
    env.reset(episodes=[sample_for(i) for i in range(env.n)])
    for _ in range(10):
        env.step(np.tile(tt.HOME, (env.n, 1)))
    env.t[:] = 0
    runners = {}
    for g, (spec, _) in GROUPS.items():
        r_ = TrajRunner(env, json.load(open(os.path.join(HERE, spec))), np.random.default_rng(a.seed + len(g)))
        for st_ in r_.spec["trajectory"]:
            if st_.get("retry"):
                st_["retry"]["max"] = max(st_["retry"].get("max", 2), 5)
        r_.reset()
        for i in range(env.n):
            if grp[i] != g:
                r_.done[i] = True                       # this runner never drives env i
        runners[g] = r_
    from isaaclab.utils import math as mu
    bi_g = env.robot.body_names.index("gripper"); FLIP = np.diag([1.0, -1.0, -1.0])
    def q2m(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    def dest_point(i):
        E = env.ep[i]; t = E["task"]
        if t == "pad": return env.pos(E["pad"])[i]
        if t == "tray": c, _ = env.tray_center(i); return np.r_[c, 0.0]
        if t == "container": return env.pos(E["container"])[i]
        if t == "rel": return np.r_[np.asarray(E["rel_xy"], float), 0.0]
        if t == "push": return np.r_[np.asarray(E["p0"], float) + E.get("push_len", 0.10) * np.asarray(E["dir"], float), 0.0]
        g = env.pos(E["obj"])[i]
        return g + np.array([0, 0, 0.10]) if t == "pick" else g
    def wrist_labels():
        pb = env.robot.data.body_pos_w[:, bi_g].cpu().numpy(); Rb = mu.matrix_from_quat(env.robot.data.body_quat_w[:, bi_g]).cpu().numpy()
        o = env.origins.cpu().numpy(); out = np.zeros((env.n, 6), np.float32)
        for i in range(env.n):
            tq = env.dr.cam[i]["t"].Get(); oq = env.dr.cam[i]["o"].Get(); q_loc = np.r_[oq.GetReal(), list(oq.GetImaginary())]
            Rc = Rb[i] @ q2m(q_loc / np.linalg.norm(q_loc)) @ FLIP; tc = pb[i] + Rb[i] @ np.array(tq, float)
            E = env.ep[i]; g = env.grasp_point(E["obj"], i)[0] if E["obj"] in tt.GRASP else env.pos(E["obj"])[i]
            for k, pnt in enumerate((g, dest_point(i))):
                out[i, 3 * k: 3 * k + 3] = (pnt + o[i] - tc) @ Rc
        return out
    # ---------------- per-env state
    n = env.n; size = v2.SIZE["own"]
    drive_pol = np.zeros(n, bool); seg_t = np.zeros(n, int); exp_until = np.zeros(n, int); best_d = np.full(n, 9.0); best_t = np.zeros(n, int)
    interventions = np.zeros(n, int); pcur = env.state().copy(); pplan = [[] for _ in range(n)]
    mem = [None] * n; tcache = {}
    buf = [dict(S=[], A=[], P=[], F={}, L={}) for _ in range(n)]
    lo, hi = (int(x) for x in a.exp_steps.split(","))
    def start_episode(i):
        drive_pol[i] = rng.random() < a.p_policy_start; seg_t[i] = 0; interventions[i] = 0
        best_d[i] = 9.0; best_t[i] = 0; pcur[i] = env.state()[i]; pplan[i] = []; mem[i] = None
        r_ = runners[grp[i]]; r_.reset_env(i)
        if drive_pol[i]:
            r_.freeze(i)
        else:
            exp_until[i] = int(rng.integers(lo, hi + 1))
        buf[i] = dict(S=[], A=[], P=[], F={}, L={})
    for i in range(n):
        start_episode(i)
    # online buffer in the SeqData "P" format
    Pon = dict(acts={}, st=st, step=3, fidx={}, phase={}, task={}, pol={}, loc={}, jpg={}, coef=coef)
    on_eps, ep_id = [], 0
    hist = dict(success=[], clean_success=[], policy_share=[], interventions=[], length=[])
    upd, t0, last_save, step_i = 0, time.time(), time.time(), 0
    saved_eps = []
    def kin_tcp():
        return runners[grp[0]].kin.tcp_pose()[0].cpu().numpy()
    while time.time() - t0 < a.hours * 3600 and ep_id < a.max_episodes:
        step_i += 1
        S = env.state()
        # ---- expert commands (each runner advances only its own, non-frozen envs)
        exp_cmd = np.zeros((n, 6))
        for g, r_ in runners.items():
            c_ = r_.step()
            for i in range(n):
                if grp[i] == g:
                    exp_cmd[i] = c_[i]
        # ---- policy at 10 Hz (every frame of every episode, so its memory sees the whole episode)
        rec = env.t % 3 == 0
        if rec.any():
            _, w = env.images(); lab = wrist_labels()
            q = [i for i in range(n) if rec[i]]
            jp = [jpeg(w[i]) for i in q]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                texts = [clean(env.ep[i]["text"]) for i in q]
                for t_ in set(texts) - set(tcache):
                    tcache[t_] = actor.text_vec(tok([t_]).to(dev))[0].float()
                U = torch.stack([tcache[t_] for t_ in texts]); sn = (S[q] - st["s_mean"]) / st["s_std"]
                ms, pls, prs, snp, first, ds = [], [], [], [], [], np.zeros((len(q), 6))
                for j, i in enumerate(q):
                    e_ = mem[i]
                    if e_ is None:
                        ms.append(actor.init_mem(U[j:j + 1])); pls.append(torch.zeros(1, G.HM, 6, device=dev)); prs.append(torch.zeros(1, 6 + G.PE, device=dev)); snp.append(sn[j]); first.append(True)
                    else:
                        ms.append(e_["m"]); pls.append(e_["plan"]); prs.append(e_["pred"]); snp.append(e_["sn"]); first.append(False); ds[j] = (S[i] - e_["S"]) / st["s_std"] * 5
                s12 = torch.from_numpy(np.clip(np.c_[sn, ds], -5, 5).astype(np.float32)).to(dev)
                x = torch.stack([decode_jpeg(torch.frombuffer(bytearray(b_), dtype=torch.uint8)) for b_ in jp]).to(dev)
                img = (torch.nn.functional.interpolate(x.float(), size=size, mode="bilinear", antialias=True, align_corners=False) / 255 - R.MEAN.to(dev)) / R.STD.to(dev)
                plan, m2, pred, _ = actor.step(img, s12, U, torch.cat(ms), torch.cat(pls), torch.cat(prs), torch.from_numpy(np.clip(np.stack(snp), -5, 5).astype(np.float32)).to(dev),
                                               torch.tensor(first, device=dev))
            p_ = plan.float().cpu().numpy()
            for j, i in enumerate(q):
                mem[i] = dict(m=m2[j:j + 1].float(), plan=plan[j:j + 1].float(), pred=pred[j:j + 1].float(), sn=np.clip(sn[j], -5, 5), S=S[i].copy())
                tgt = np.r_[S[i, :5] + p_[j, 0, :5] * st["dstd"], (p_[j, 0, 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]]
                tgt = (tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)
                pplan[i] = [pcur[i] + (tgt - pcur[i]) * (k + 1) / 3 for k in range(3)]
                if env.t[i] >= 2:
                    buf[i]["F"][int(env.t[i])] = jp[j]; buf[i]["L"][int(env.t[i])] = lab[i].tolist()
        # ---- who drives: stall / long segment / end of an expert segment
        tcp = kin_tcp()
        for i in range(n):
            r_ = runners[grp[i]]
            if drive_pol[i]:
                seg_t[i] += 1
                goal = r_.p1[i] if r_.phase[i] in ("move", "settle") else r_.p_hold[i]
                d = float(np.linalg.norm(tcp[i] - goal))
                if d < best_d[i] - 0.01:
                    best_d[i], best_t[i] = d, seg_t[i]
                beta = a.beta0 + (a.beta1 - a.beta0) * min(1.0, ep_id / 1500)
                stalled = seg_t[i] - best_t[i] > a.stall
                cut = seg_t[i] > a.pol_long and rng.random() < beta / 30
                if stalled or cut:
                    drive_pol[i] = False; r_.resume(i); interventions[i] += 1; seg_t[i] = 0; exp_until[i] = int(rng.integers(lo, hi + 1))
            else:
                seg_t[i] += 1
                if seg_t[i] >= exp_until[i] and not r_.done[i]:
                    drive_pol[i] = True; r_.freeze(i); seg_t[i] = 0; best_d[i] = 9.0; best_t[i] = 0; pcur[i] = S[i].copy(); pplan[i] = []
        cmd = exp_cmd.copy()
        for i in range(n):
            if drive_pol[i]:
                if pplan[i]:
                    pcur[i] = pplan[i].pop(0)
                cmd[i] = pcur[i]
            buf[i]["S"].append(S[i].astype(np.float32)); buf[i]["A"].append(cmd[i].astype(np.float32)); buf[i]["P"].append(bool(drive_pol[i]))
        env.step(cmd)
        # ---- episode ends
        succ = env.success()
        done = [i for i in range(n) if succ[i] or env.t[i] >= a.max_s * tt.CTRL_HZ or (runners[grp[i]].done[i] and not drive_pol[i])]
        for i in done:
            B = buf[i]; A = np.stack(B["A"]); Sx = np.stack(B["S"]); Pm = np.array(B["P"], bool)
            if len(B["F"]) >= 8:
                e = ep_id; Ac = A * coef[:, 0] + coef[:, 1]
                Pon["acts"][e] = (Ac, Sx); Pon["fidx"][e] = np.array(sorted(B["F"])); Pon["phase"][e] = v2.phases(Ac); Pon["task"][e] = clean(env.ep[i]["text"]); Pon["pol"][e] = Pm
                for f, b_ in B["F"].items():
                    Pon["jpg"][(e, f)] = b_; Pon["loc"][(e, f)] = np.asarray(B["L"][f], np.float32)
                on_eps.append(e); saved_eps.append(dict(ep=e, text=env.ep[i]["text"], task=env.ep[i]["task"], success=bool(succ[i]), interventions=int(interventions[i]),
                                                        policy_share=round(float(Pm.mean()), 3), steps=len(A)))
                while len(on_eps) > a.buffer:
                    old = on_eps.pop(0)
                    for f in Pon["fidx"][old]:
                        Pon["jpg"].pop((old, int(f)), None); Pon["loc"].pop((old, int(f)), None)
                    for k_ in ("acts", "fidx", "phase", "task", "pol"):
                        Pon[k_].pop(old, None)
                hist["success"].append(float(succ[i])); hist["policy_share"].append(float(Pm.mean())); hist["interventions"].append(int(interventions[i])); hist["length"].append(len(A) / tt.CTRL_HZ)
                if interventions[i] == 0 and Pm.mean() > 0.95:
                    hist["clean_success"].append(float(succ[i]))
                ep_id += 1
                k_ = ep_id
                for key, vals in hist.items():
                    if vals:
                        tb.add_scalar(f"online/{key}_last50", float(np.mean(vals[-50:])), k_)
                print(json.dumps(saved_eps[-1] | dict(el=round(time.time() - t0), upd=upd)), flush=True)
        if done:
            env.reset(env_ids=done, episodes=[sample_for(i) for i in done], settle=0)
            for i in done:
                start_episode(i)
        # ---- interleaved training
        if len(on_eps) >= a.warm_episodes and step_i % a.train_every == 0:
            ds_on = R.SeqData(Pon, on_eps, True, "own"); ds_off = R.SeqData(Poff, Poff["train_eps"], True, "own")
            items = [ds_on[random.randrange(len(ds_on))] for _ in range(a.bs // 2)] + [ds_off[random.randrange(len(ds_off))] for _ in range(a.bs - a.bs // 2)]
            b = R.collate(items)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = G.run_window(model, b, tok, dev, True)
                loss, parts, le = G.losses(model, b, tok, dev, o, W)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(enc + head, 1.0); opt.step(); ema.update_parameters(model); upd += 1
            if upd % 50 == 0:
                for k2, v_ in parts.items():
                    tb.add_scalar(f"train/{k2}", float(v_), upd)
                if le is not None:
                    tb.add_scalar("train/loc_err_cm", 100 * float(le), upd)
                tb.add_scalar("online/episodes", ep_id, upd); tb.add_scalar("speed/wall_h", (time.time() - t0) / 3600, upd)
            if upd % 200 == 0:
                actor.load_state_dict(ema.module.state_dict()); actor.eval()
        if time.time() - last_save > a.save_every_s:
            last_save = time.time()
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            torch.save(dict(model=ema.module.state_dict(), arch="nanovla_g", st=st, coef=coef, step_frames=3, bench=a.bench, online_episodes=ep_id, updates=upd), a.out)
            tb.flush()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(model=ema.module.state_dict(), arch="nanovla_g", st=st, coef=coef, step_frames=3, bench=a.bench, online_episodes=ep_id, updates=upd), a.out)
    # online episodes -> bench format (reusable offline: merge_sim_bench, *_pol masks)
    import pandas as pd
    os.makedirs(os.path.join(a.save_dir, "frames"), exist_ok=True); os.makedirs(os.path.join(a.save_dir, "actions"), exist_ok=True)
    rows, acts = [], {}
    inv = np.where(np.abs(coef[:, 0]) > 1e-3, coef[:, 0], 1.0)
    for e in on_eps:
        Ac, Sx = Pon["acts"][e]; acts[f"ep{e}_action"] = ((Ac - coef[:, 1]) / inv).astype(np.float32); acts[f"ep{e}_state"] = Sx; acts[f"ep{e}_pol"] = Pon["pol"][e]
        for f in Pon["fidx"][e]:
            rows.append((e, "wrist", int(f), Pon["jpg"][(e, int(f))], Pon["task"][e], Pon["task"][e], Pon["loc"][(e, int(f))].tolist()))
    name = "traj__online"
    D = pd.DataFrame(rows, columns=["ep", "cam", "fidx", "jpeg", "task", "task_orig", "loc"]); D["fps"] = 30
    D.to_parquet(os.path.join(a.save_dir, "frames", name + ".parquet")); np.savez_compressed(os.path.join(a.save_dir, "actions", name + ".npz"), **acts)
    json.dump([name.replace("__", "/")], open(os.path.join(a.save_dir, "repos.json"), "w"))
    json.dump(dict(episodes=[s_ for s_ in saved_eps if s_["ep"] in set(on_eps)], kept=len(on_eps), attempts=ep_id), open(os.path.join(a.save_dir, "episodes.json"), "w"), indent=1)
    tb.close(); print(json.dumps(dict(saved=a.out, online_episodes=ep_id, updates=upd, hours=round((time.time() - t0) / 3600, 2))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    sys.stdout.flush(); os._exit(0)
