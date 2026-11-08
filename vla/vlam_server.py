"""Policy server for NanoVLA-M (memory model). Same socket protocol as vlar_server.py: requests carry `ids` (env index)
and `new` (episode start). Per env the server keeps the memory m, the previous joint state and the cached task vector u.
One request = one frame (10 Hz): tabletop_eval.py --exec 1.

    python vlam_server.py --ckpt ~/VLA/ckpt/nanovla_m_l1/vlam.pt --port 6000
"""
import argparse, json, os
import numpy as np, torch
from multiprocessing.connection import Listener
from train_vla import MEAN, STD
from train_text import clean
import torchvision.transforms.v2 as T
import nanovla_m as M
import nanovla2 as v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6000)
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model, tok = M.build(dev, ck["head"]); model.load_state_dict(ck["model"]); model.eval()
    st, coef = ck["st"], ck["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    vel = ck.get("act_mode", "pos") == "vel"; vstd = ck.get("vstd")
    size = v2.SIZE["own"]
    mem, tcache = {}, {}
    with Listener(("localhost", a.port), authkey=b"vla-isaac") as L:
        print(json.dumps(dict(event="ready", port=a.port, step=ck["step_frames"], cams="wrist", arch="nanovla_m", head=ck["head"])), flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try:
                        req = c.recv()
                    except EOFError:
                        break
                    if req == "meta":
                        c.send(dict(step=ck["step_frames"], H=M.H_M, cams="wrist", recurrent=True)); continue
                    if req == "quit":
                        return
                    S = np.asarray(req["state"], np.float64); n = len(S)
                    ids = req.get("ids", list(range(n))); new = req.get("new", [False] * n)
                    texts = [clean(t) for t in req["text"]]
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for t in set(texts) - set(tcache):
                            tcache[t] = model.text_vec(tok([t]).to(dev))[0].float()
                        U = torch.stack([tcache[t] for t in texts])
                        ms, ds = [], np.zeros_like(S)
                        for j, i in enumerate(ids):
                            e = mem.get(i)
                            if new[j] or e is None or e["text"] != texts[j]:
                                ms.append(model.init_mem(U[j:j + 1]))
                                e = None
                            else:
                                ms.append(e["m"]); ds[j] = (S[j] - e["S"]) / st["s_std"] * 5
                        sn = (S - st["s_mean"]) / st["s_std"]
                        s12 = torch.from_numpy(np.clip(np.c_[sn, ds], -5, 5).astype(np.float32)).to(dev)
                        x = torch.stack([torch.from_numpy(np.asarray(w)).permute(2, 0, 1) for w in req["wrist"]]).to(dev)
                        img = (T.functional.resize(x, list(size), antialias=True).float() / 255 - MEAN.to(dev)) / STD.to(dev)
                        p, m2 = model.step(img, s12, U, torch.cat(ms))
                    p = p.float().cpu().numpy()
                    tgt = np.empty((n, M.H_M, 6))
                    if vel:                                   # command = previous command + cumulative increments (calibrated units)
                        last = np.stack([mem[i]["cmd"] if (i in mem and not new[j] and mem[i]["text"] == texts[j]) else S[j, :5]
                                         for j, i in enumerate(ids)])
                        tgt[..., :5] = last[:, None] + np.cumsum(p[..., :5] * vstd, 1)
                    else:
                        tgt[..., :5] = S[:, None, :5] + p[..., :5] * st["dstd"]
                    for j, i in enumerate(ids):
                        mem[i] = dict(m=m2[j:j + 1].float(), S=S[j].copy(), text=texts[j], cmd=tgt[j, 0, :5].copy())
                    tgt[..., 5] = (p[..., 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
                    c.send(((tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)).astype(float).tolist())


if __name__ == "__main__":
    main()
