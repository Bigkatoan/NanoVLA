"""Policy server for NanoVLA-R (recurrent, wrist-only). Same socket protocol as isaac_policy_server.py, plus memory:
the request carries `ids` (env indices) and `new` (episode just started); the server keeps per env the LSTM state,
the previous joint state and the cached text vector (re-encoded only when the instruction changes).
Each request is one tick; the client must query every 0.1 s (tabletop_eval --exec 1: one chunk point = 3 control steps) as in training (TICK=1, 10 Hz).

    python vlar_server.py --ckpt ~/VLA/ckpt/nanovla_r/vlar.pt --port 6000
"""
import argparse, json, os
import numpy as np, torch
import torchvision.transforms.v2 as T
from multiprocessing.connection import Listener
from train_vla import H, MEAN, STD
from train_text import clean
import nanovla_r as R
import nanovla2 as v2


def prep(a, size):
    x = torch.from_numpy(np.asarray(a)).permute(2, 0, 1)
    x = T.functional.resize(x, list(size), antialias=True)
    return (x.float() / 255 - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/VLA/ckpt/nanovla_r/vlar.pt"))
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--flow_steps", type=int, default=5)
    ap.add_argument("--noise", default="zero", choices=("zero", "random"), help="flow start: zero = deterministic chunk")
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    vision = ck.get("vision", "own")
    model, tok = R.build(dev, vision, ck.get("variant", "ours")); model.load_state_dict(ck["model"]); model.eval()
    st, coef = ck["st"], ck["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    size = v2.SIZE[vision]
    mem = {}                        # env id -> dict(hc=(h,c), S=prev joints, text=str, u=tensor)
    tcache = {}
    with Listener(("localhost", a.port), authkey=b"vla-isaac") as L:
        print(json.dumps(dict(event="ready", port=a.port, step=ck["step_frames"], cams="wrist", arch="nanovla_r")), flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try:
                        req = c.recv()
                    except EOFError:
                        break
                    if req == "meta":
                        c.send(dict(step=ck["step_frames"], H=H, cams="wrist", recurrent=True)); continue
                    if req == "quit":
                        return
                    S = np.asarray(req["state"], np.float64); n = len(S)
                    ids = req.get("ids", list(range(n))); new = req.get("new", [False] * n)
                    texts = [clean(t) for t in req["text"]]
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for t in set(texts) - set(tcache):
                            tcache[t] = model.text_vec(tok([t]).to(dev))[0].float()
                        U = torch.stack([tcache[t] for t in texts])
                        sn = torch.from_numpy(((S - st["s_mean"]) / st["s_std"]).astype(np.float32)).to(dev)
                        hs, cs, ds = [], [], np.zeros_like(S)
                        for j, i in enumerate(ids):
                            m = mem.get(i)
                            if new[j] or m is None or m["text"] != texts[j]:
                                h0, c0 = model.init_state(U[j:j + 1], sn[j:j + 1])
                            else:
                                (h0, c0), ds[j] = m["hc"], (S[j] - m["S"]) / st["s_std"] * 5
                            hs.append(h0); cs.append(c0)
                        s12 = torch.cat([sn, torch.from_numpy(np.clip(ds, -5, 5).astype(np.float32)).to(dev)], -1).clamp(-5, 5)
                        img = torch.stack([prep(w, size) for w in req["wrist"]]).to(dev)
                        noise = torch.zeros(n, H, 6, device=dev) if a.noise == "zero" else None
                        p, (h, cc) = model.step(img, s12, U, (torch.cat(hs, 1), torch.cat(cs, 1)), a.flow_steps, noise)
                    for j, i in enumerate(ids):
                        mem[i] = dict(hc=(h[:, j:j + 1].float(), cc[:, j:j + 1].float()), S=S[j].copy(), text=texts[j])
                    p = p.float().cpu().numpy()
                    tgt = np.empty((n, H, 6))
                    tgt[..., :5] = S[:, None, :5] + p[..., :5] * st["dstd"]
                    tgt[..., 5] = (p[..., 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
                    c.send(((tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)).astype(float).tolist())


if __name__ == "__main__":
    main()
