"""NanoIK: learned differential IK (and FK) of the SO-101 in the wrist-camera frame.

  FK net : (q[6], dq[5])        -> (dp_cam[3], dyaw)      unique, used as the checker
  IK net : (q[6], dp_cam[3], dyaw) -> dq[5]               trained on the data + a cycle loss FK(q, IK(q, dp)) = dp
Metrics (held-out): FK error, and the IK "achieved displacement" error through the FK net, in mm / degrees, for
requests up to the servo step size. Exports ik.onnx for the Jetson and nano_ik.pt for the eval client.

    python tools/nano_ik.py --data ~/VLA/data/nano_ik/ik.npz --out ~/VLA/ckpt/nano_ik
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn


def mlp(i, o, h=256, n=3):
    L, d = [], i
    for _ in range(n):
        L += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*L, nn.Linear(d, o))


class NanoIK(nn.Module):
    """inputs scaled inside: q (rad), dp (m, camera frame), dyaw (rad) -> dq (rad, arm joints 0-4)"""
    def __init__(self):
        super().__init__(); self.net = mlp(6 + 6 + 4, 5)

    def forward(self, q, dp, dyaw):
        x = torch.cat([torch.sin(q), torch.cos(q), dp / 0.01, dyaw[:, None] / 0.05], -1)
        return self.net(x) * 0.02


class NanoFK(nn.Module):
    def __init__(self):
        super().__init__(); self.net = mlp(12 + 5, 4)

    def forward(self, q, dq):
        x = torch.cat([torch.sin(q), torch.cos(q), dq / 0.02], -1); y = self.net(x)
        return y[:, :3] * 0.01, y[:, 3] * 0.05


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--tb", default=os.path.expanduser("~/VLA/tb/nano_ik")); a = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(a.tb)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    z = np.load(a.data); q, dp, dy, dq = (torch.tensor(z[k], dtype=torch.float32) for k in ("q", "dp_cam", "dyaw", "dq"))
    n = len(q); idx = torch.randperm(n); va, tr = idx[:50000], idx[50000:]
    q, dp, dy, dq = q.to(dev), dp.to(dev), dy.to(dev), dq.to(dev)
    fk, ik = NanoFK().to(dev), NanoIK().to(dev)

    @torch.no_grad()
    def val_fk():
        p_, y_ = fk(q[va], dq[va]); return float((p_ - dp[va]).norm(dim=1).mean() * 1000), float((y_ - dy[va]).abs().mean() * 57.3)

    @torch.no_grad()
    def val_ik():
        dq_ = ik(q[va], dp[va], dy[va]); p_, y_ = fk(q[va], dq_)
        return float((p_ - dp[va]).norm(dim=1).mean() * 1000), float((y_ - dy[va]).abs().mean() * 57.3)
    log = []
    # ---- FK
    opt = torch.optim.AdamW(fk.parameters(), 1e-3)
    for s in range(a.steps):
        b = tr[torch.randint(len(tr), (a.bs,))]
        for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * s / a.steps))
        p_, y_ = fk(q[b], dq[b]); loss = ((p_ - dp[b]) / 0.001).pow(2).mean() + ((y_ - dy[b]) / 0.01).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            tb.add_scalar("fk/train_loss", float(loss), s)
        if s % 1000 == 0 or s == a.steps - 1:
            m_, d_ = val_fk(); tb.add_scalar("fk/val_err_mm", m_, s); tb.add_scalar("fk/val_err_deg", d_, s)
    with torch.no_grad():
        p_, y_ = fk(q[va], dq[va]); fk_mm = float((p_ - dp[va]).norm(dim=1).mean() * 1000); fk_deg = float((y_ - dy[va]).abs().mean() * 57.3)
    print(json.dumps(dict(fk_err_mm=round(fk_mm, 3), fk_err_deg=round(fk_deg, 3), dp_mean_mm=round(float(dp[va].norm(dim=1).mean() * 1000), 2))), flush=True)
    for p in fk.parameters():
        p.requires_grad = False
    # ---- IK: data term + cycle term through the frozen FK
    opt = torch.optim.AdamW(ik.parameters(), 1e-3)
    for s in range(a.steps):
        b = tr[torch.randint(len(tr), (a.bs,))]
        for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * s / a.steps))
        dq_ = ik(q[b], dp[b], dy[b]); p_, y_ = fk(q[b], dq_)
        cyc = ((p_ - dp[b]) / 0.001).pow(2).mean() + ((y_ - dy[b]) / 0.01).pow(2).mean()
        loss = cyc + 0.1 * ((dq_ - dq[b]) / 0.01).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            tb.add_scalar("ik/train_loss", float(loss), s); tb.add_scalar("ik/train_cycle_loss", float(cyc), s)
        if s % 1000 == 0 or s == a.steps - 1:
            m_, d_ = val_ik(); tb.add_scalar("ik/val_achieved_err_mm", m_, s); tb.add_scalar("ik/val_achieved_err_deg", d_, s)
    with torch.no_grad():
        dq_ = ik(q[va], dp[va], dy[va]); p_, y_ = fk(q[va], dq_)
        ik_mm = float((p_ - dp[va]).norm(dim=1).mean() * 1000); ik_deg = float((y_ - dy[va]).abs().mean() * 57.3)
        rel = float(((p_ - dp[va]).norm(dim=1) / dp[va].norm(dim=1).clamp(min=1e-4)).median())
    res = dict(fk_err_mm=round(fk_mm, 3), fk_err_deg=round(fk_deg, 3), ik_achieved_err_mm=round(ik_mm, 3), ik_achieved_err_deg=round(ik_deg, 3), ik_rel_err_median=round(rel, 4),
               params_ik=sum(p.numel() for p in ik.parameters()), samples=n)
    print(json.dumps(res), flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(ik=ik.state_dict(), fk=fk.state_dict(), metrics=res), os.path.join(a.out, "nano_ik.pt"))
    ik.cpu().eval()
    torch.onnx.export(ik, (torch.zeros(1, 6), torch.zeros(1, 3), torch.zeros(1)), os.path.join(a.out, "ik.onnx"), input_names=["q", "dp_cam", "dyaw"], output_names=["dq"], opset_version=17, dynamo=False)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    tb.add_text("result", json.dumps(res)); tb.close()


if __name__ == "__main__":
    main()
