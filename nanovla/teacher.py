"""Offline tools for a SmolVLA checkpoint on a LeRobot dataset (2 cameras: camera1 wrist, camera2 overview).

eval : action-chunk error vs the recorded demo on given episodes (every --stride frames). Reports mean absolute
       error in the dataset's units (degrees; gripper 0-100) over the first 10 steps (0.33 s) and the full chunk,
       per joint, plus inference latency.
label: teacher labels for distillation: for every --stride-th frame of the given episodes and --aug augmented copies
       (colour jitter + small shift, same on both cameras), store the wrist image (resized), state, task, the
       teacher's action chunk and the demo chunk -> .npz

    python -m nanovla.teacher eval --ckpt <pretrained_model> --root data/lerobot/ttt_v1 --episodes 30-44 --out x.json
    python -m nanovla.teacher label --ckpt <pretrained_model> --root data/lerobot/ttt_v1 --episodes 0-29 --stride 3 --aug 2 --out labels.npz
"""
import argparse, json, math, os, time
import numpy as np, torch


def parse_eps(s):
    out = []
    for p in s.split(","):
        a, _, b = p.partition("-"); out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load(ckpt, steps=10):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    pol = SmolVLAPolicy.from_pretrained(ckpt).to("cuda").eval(); pol.config.num_steps = steps
    pre, post = make_pre_post_processors(pol.config, pretrained_path=ckpt)
    return pol, pre, post


@torch.no_grad()
def predict(pol, pre, post, batch, seed=0):
    pol.reset(); b = pre(batch)
    B = b["observation.state"].shape[0]
    g = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn(B, pol.config.chunk_size, pol.config.max_action_dim, device="cuda", generator=g)
    a = pol.predict_action_chunk(b, noise=noise)
    return post(a).float().cpu()


def frames(root, eps, stride, H=50):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    meta_fps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]
    ds = LeRobotDataset("local/x", root=root, episodes=eps, delta_timestamps={"action": [i / meta_fps for i in range(H)]})
    fi = np.asarray([int(x) for x in ds.hf_dataset["frame_index"]])
    idx = [int(i) for i in np.nonzero(fi % stride == 0)[0]]
    return ds, idx


def collate(items):
    b = {k: torch.stack([x[k] for x in items]) for k in ("observation.images.camera1", "observation.images.camera2", "observation.state", "action", "action_is_pad")}
    b["task"] = [x["task"] for x in items]; b["episode_index"] = torch.tensor([int(x["episode_index"]) for x in items]); return b


def augment(img1, img2, gen):
    """same colour jitter + shift on both cameras (B,3,H,W in 0..1)"""
    B = img1.shape[0]
    br = 1 + 0.25 * (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1); co = 1 + 0.25 * (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1)
    sh = (torch.rand(B, 2, generator=gen) * 2 - 1) * 0.04
    out = []
    for im in (img1, img2):
        m = im.mean((1, 2, 3), keepdim=True); im = ((im - m) * co + m) * br
        th = torch.zeros(B, 2, 3); th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, :, 2] = sh
        grid = torch.nn.functional.affine_grid(th, im.shape, align_corners=False)
        out.append(torch.nn.functional.grid_sample(im, grid, padding_mode="border", align_corners=False).clamp(0, 1))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("eval", "label"))
    ap.add_argument("--ckpt", required=True); ap.add_argument("--root", required=True); ap.add_argument("--episodes", required=True)
    ap.add_argument("--stride", type=int, default=10); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--out", required=True)
    ap.add_argument("--aug", type=int, default=0); ap.add_argument("--hw", default="128x160"); a = ap.parse_args()
    pol, pre, post = load(a.ckpt); eps = parse_eps(a.episodes)
    ds, idx = frames(a.root, eps, a.stride, pol.config.chunk_size)
    print(json.dumps(dict(mode=a.mode, frames=len(idx), episodes=len(eps))), flush=True)
    t0 = time.time(); lat = []
    if a.mode == "eval":
        E10, E50, n10, n50 = torch.zeros(6), torch.zeros(6), 0, 0
        for s in range(0, len(idx), a.bs):
            b = collate([ds[i] for i in idx[s:s + a.bs]])
            torch.cuda.synchronize(); t1 = time.time()
            p = predict(pol, pre, post, {k: v for k, v in b.items() if k not in ("action", "action_is_pad", "episode_index")})
            torch.cuda.synchronize(); lat.append((time.time() - t1) / len(p))
            ok = ~b["action_is_pad"]; err = (p - b["action"]).abs()
            m10 = ok[:, :10]; E10 += (err[:, :10] * m10[..., None]).sum((0, 1)); n10 += int(m10.sum())
            E50 += (err * ok[..., None]).sum((0, 1)); n50 += int(ok.sum())
        r = dict(ckpt=a.ckpt, episodes=a.episodes, frames=len(idx), mae10=(E10 / n10).round(decimals=2).tolist(), mae50=(E50 / n50).round(decimals=2).tolist(),
                 mae10_arm=round(float((E10 / n10)[:5].mean()), 2), mae50_arm=round(float((E50 / n50)[:5].mean()), 2),
                 mae10_grip=round(float((E10 / n10)[5]), 2), ms_per_frame=round(1000 * float(np.median(lat)), 1), s=round(time.time() - t0))
        print(json.dumps(r), flush=True); json.dump(r, open(a.out, "w"), indent=1)
        return
    H, W = (int(x) for x in a.hw.split("x")); gen = torch.Generator().manual_seed(0)
    IMG, ST, TASK, TCH, GT, PAD, AUGF, EP = [], [], [], [], [], [], [], []
    for s in range(0, len(idx), a.bs):
        b = collate([ds[i] for i in idx[s:s + a.bs]])
        for k in range(a.aug + 1):
            c1, c2 = (b["observation.images.camera1"], b["observation.images.camera2"]) if k == 0 else augment(b["observation.images.camera1"], b["observation.images.camera2"], gen)
            p = predict(pol, pre, post, {"observation.images.camera1": c1, "observation.images.camera2": c2, "observation.state": b["observation.state"], "task": b["task"]}, seed=s * 7 + k)
            IMG.append((torch.nn.functional.interpolate(c1, size=(H, W), mode="area") * 255).round().byte().numpy()); ST.append(b["observation.state"].numpy())
            TASK += b["task"]; TCH.append(p.numpy()); GT.append(b["action"].numpy()); PAD.append(b["action_is_pad"].numpy()); AUGF += [k] * len(p); EP.append(b["episode_index"].numpy())
        if (s // a.bs) % 20 == 0:
            print(json.dumps(dict(labelled=len(TASK), s=round(time.time() - t0))), flush=True)
    np.savez(a.out, img=np.concatenate(IMG), state=np.concatenate(ST), task=np.array(TASK), teacher=np.concatenate(TCH), gt=np.concatenate(GT),
             pad=np.concatenate(PAD), aug=np.array(AUGF), ep=np.concatenate(EP))
    print(json.dumps(dict(saved=a.out, n=len(TASK), s=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
