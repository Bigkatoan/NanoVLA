"""NanoVLA student: wrist camera only, distilled from the 2-camera SmolVLA teacher (tools/smol_offline.py label).

  wrist image (128x160) -> ResNet18 (ImageNet init) -> 512
  joints (6, normalised) -> MLP -> 128          task -> learned embedding (closed vocabulary of the specialised task) -> 128
  concat -> MLP -> action chunk 50 x 6 (normalised; absolute joint targets in the dataset's units)
Loss: L1 to the teacher's chunk (all frames, incl. augmented copies) and/or to the demo chunk (real frames).

    python -m nanovla.student --train labels_train.npz --test labels_test.npz --target both --out models/ttt/student.pt
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class Student(nn.Module):
    def __init__(self, n_tasks, H=50, A=6):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(); w = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
        if os.path.exists(w):
            r.load_state_dict(torch.load(w, map_location="cpu"))
        r.fc = nn.Identity(); self.vis = r; self.H, self.A = H, A
        self.st = nn.Sequential(nn.Linear(A, 128), nn.GELU(), nn.Linear(128, 128)); self.task = nn.Embedding(n_tasks, 128)
        self.head = nn.Sequential(nn.Linear(512 + 256, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, H * A))
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img_u8, state_n, task_id):
        x = (img_u8.float() / 255 - self.im_mean) / self.im_std
        h = torch.cat([self.vis(x), self.st(state_n), self.task(task_id)], -1)
        return self.head(h).view(-1, self.H, self.A)


def load_npz(p):
    z = np.load(p, allow_pickle=False); return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="both", choices=("teacher", "gt", "both")); ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tb", default="")
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
    parts = [load_npz(p) for p in a.train.split(",")]                 # several label files (demos + DAgger rounds) are concatenated
    tr = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}; te = load_npz(a.test)
    tasks = sorted(set(tr["task"].tolist()) | set(te["task"].tolist())); T = {t: i for i, t in enumerate(tasks)}
    real = tr["aug"] == 0
    mu = np.concatenate([tr["state"][real], tr["gt"][real].reshape(-1, 6)]).mean(0); sd = np.concatenate([tr["state"][real], tr["gt"][real].reshape(-1, 6)]).std(0) + 1e-3
    g = lambda x: torch.tensor(x, device=dev)
    IMG, ST, TK = g(tr["img"]), g(((tr["state"] - mu) / sd).astype(np.float32)), g(np.array([T[t] for t in tr["task"]]))
    TCH, GT, PAD, RL = g(((tr["teacher"] - mu) / sd).astype(np.float32)), g(((tr["gt"] - mu) / sd).astype(np.float32)), g(tr["pad"]), g(real)
    m = Student(len(tasks)).to(dev); opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.01)
    tb = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(a.tb)

    @torch.no_grad()
    def evaluate():
        m.eval(); P = []
        for s in range(0, len(te["state"]), 256):
            P.append(m(g(te["img"][s:s + 256]), g(((te["state"][s:s + 256] - mu) / sd).astype(np.float32)), g(np.array([T[t] for t in te["task"][s:s + 256]]))).float().cpu().numpy())
        m.train(); P = np.concatenate(P) * sd + mu; ok = ~te["pad"]; err = np.abs(P - te["gt"])
        e10 = (err[:, :10] * ok[:, :10, None]).sum((0, 1)) / ok[:, :10].sum(); e50 = (err * ok[..., None]).sum((0, 1)) / ok.sum()
        t10 = (np.abs(te["teacher"] - te["gt"])[:, :10] * ok[:, :10, None]).sum((0, 1)) / ok[:, :10].sum()
        return dict(mae10_arm=round(float(e10[:5].mean()), 2), mae50_arm=round(float(e50[:5].mean()), 2), mae10_grip=round(float(e10[5]), 2),
                    teacher_mae10_arm=round(float(t10[:5].mean()), 2), teacher_mae10_grip=round(float(t10[5]), 2))
    n = len(ST); t0 = time.time(); best = None
    for s in range(a.steps):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1, s / 300) * 0.5 * (1 + math.cos(math.pi * s / a.steps))
        b = torch.randint(n, (a.bs,), device=dev)
        img = IMG[b]
        if True:                                                         # light photometric jitter on the student side too
            img = (img.float() * (1 + 0.2 * (torch.rand(a.bs, 1, 1, 1, device=dev) - 0.5))).clamp(0, 255).byte()
        p = m(img, ST[b] + 0.02 * torch.randn_like(ST[b]), TK[b])
        okp = (~PAD[b]).float()[..., None]
        lt = ((p - TCH[b]).abs() * okp).sum() / okp.sum().clamp(min=1) / 6
        rl = RL[b].float()[:, None, None] * okp; lg = ((p - GT[b]).abs() * rl).sum() / rl.sum().clamp(min=1) / 6
        loss = {"teacher": lt, "gt": lg, "both": 0.5 * lt + 0.5 * lg}[a.target]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if tb is not None and s % 100 == 0:
            tb.add_scalar("train/loss", float(loss), s)
        if (s + 1) % 2000 == 0 or s + 1 == a.steps:
            ev = evaluate(); print(json.dumps(dict(step=s + 1, loss=round(float(loss), 4), **ev, s=round(time.time() - t0))), flush=True)
            if tb is not None:
                for k, v in ev.items():
                    tb.add_scalar(f"test/{k}", v, s + 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(model=m.state_dict(), tasks=tasks, mu=mu, sd=sd, target=a.target, test=ev), a.out)
    json.dump(dict(tasks=tasks, mu=mu.tolist(), sd=sd.tolist(), chunk=50, image_hw=[128, 160], fps=30,
                   inputs=dict(wrist_u8="(1,3,128,160) uint8 RGB", state_n="(1,6) (joints - mu) / sd", task_id="(1,) int64 index into tasks"),
                   output="chunk_n (1,50,6): denormalise with * sd + mu -> joint targets (degrees, gripper 0-100)"),
              open(a.out.replace(".pt", ".meta.json"), "w"), indent=1)
    # ONNX + latency
    m.eval().cpu()
    dummy = (torch.zeros(1, 3, 128, 160, dtype=torch.uint8), torch.zeros(1, 6), torch.zeros(1, dtype=torch.long))
    onnx = a.out.replace(".pt", ".onnx")
    try:
        torch.onnx.export(m, dummy, onnx, input_names=["wrist_u8", "state_n", "task_id"], output_names=["chunk_n"], opset_version=17, dynamo=False)
    except Exception as e:
        onnx = f"export failed: {e}"
    m.to(dev); x = (dummy[0].to(dev), dummy[1].to(dev), dummy[2].to(dev))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(20):
            m(*x)
        torch.cuda.synchronize(); t1 = time.time()
        for _ in range(200):
            m(*x)
        torch.cuda.synchronize()
    res = dict(out=a.out, onnx=onnx, params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 2), ms_fp16_3060=round((time.time() - t1) / 200 * 1000, 2), target=a.target, **ev)
    print(json.dumps(res), flush=True); json.dump(res, open(a.out.replace(".pt", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
