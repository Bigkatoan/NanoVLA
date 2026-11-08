"""Policy server for NanoVLA-C (closed loop). Same protocol as vlam_server.py (ids / new per request, one request per
frame = 0.1 s, tabletop_eval.py --exec 1). Per env it keeps: memory m, the previous plan, the prediction for this frame,
the previous joint state and the cached task vector.

    python vlac_server.py --ckpt ~/VLA/ckpt/nanovla_c/vlac.pt --port 6000
"""
import argparse, json
import numpy as np, torch
import torchvision.transforms.v2 as T
from multiprocessing.connection import Listener
from train_vla import MEAN, STD
from train_text import clean
import nanovla_c as C
import nanovla2 as v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--jpeg", type=int, default=90, help="JPEG-encode incoming frames like the training data (0 = off)")
    ap.add_argument("--adapter", default="", help="few-shot adapter (fewshot_c.py) applied on top of the checkpoint")
    ap.add_argument("--elog", default="", help="jsonl: per env tick |expected - observed| (joints, image) for the closed-loop diagnosis")
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st, coef = ck["st"], ck["coef"]; a_s, b_s = coef[:, 0], coef[:, 1]
    if ck.get("arch") == "nanovla_g":
        import nanovla_g as G
        model, tok = G.build(dev, st, init="")
    else:
        model, tok = C.build(dev, st, init="")
    model.load_state_dict(ck["model"])
    if a.adapter:
        import fewshot_c
        ad = fewshot_c.load_adapter(model, a.adapter); print(json.dumps(dict(event="adapter", path=a.adapter, N=ad.get("N"), demos=ad.get("demos"))), flush=True)
    model.eval()
    import io as _io
    from PIL import Image as _I
    def jp(x):
        if not a.jpeg:
            return np.asarray(x)
        b_ = _io.BytesIO(); _I.fromarray(np.asarray(x, np.uint8)).save(b_, format="JPEG", quality=a.jpeg); return np.asarray(_I.open(b_).convert("RGB"))
    size = v2.SIZE["own"]
    mem, tcache = {}, {}
    with Listener(("localhost", a.port), authkey=b"vla-isaac") as L:
        print(json.dumps(dict(event="ready", port=a.port, step=ck["step_frames"], cams="wrist", arch=ck.get("arch", "nanovla_c"), jpeg=a.jpeg)), flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try:
                        req = c.recv()
                    except EOFError:
                        break
                    if req == "meta":
                        c.send(dict(step=ck["step_frames"], H=C.HM, cams="wrist", recurrent=True)); continue
                    if req == "quit":
                        return
                    S = np.asarray(req["state"], np.float64); n = len(S)
                    ids = req.get("ids", list(range(n))); new = req.get("new", [False] * n)
                    texts = [clean(t) for t in req["text"]]
                    sn = (S - st["s_mean"]) / st["s_std"]
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for t in set(texts) - set(tcache):
                            tcache[t] = model.text_vec(tok([t]).to(dev))[0].float()
                        U = torch.stack([tcache[t] for t in texts])
                        ms, plans, preds, snp, first, ds = [], [], [], [], [], np.zeros_like(S)
                        for j, i in enumerate(ids):
                            e = mem.get(i)
                            if new[j] or e is None or e["text"] != texts[j]:
                                ms.append(model.init_mem(U[j:j + 1])); plans.append(torch.zeros(1, C.HM, 6, device=dev))
                                preds.append(torch.zeros(1, 6 + C.PE, device=dev)); snp.append(sn[j]); first.append(True)
                            else:
                                ms.append(e["m"]); plans.append(e["plan"]); preds.append(e["pred"]); snp.append(e["sn"]); first.append(False)
                                ds[j] = (S[j] - e["S"]) / st["s_std"] * 5
                        s12 = torch.from_numpy(np.clip(np.c_[sn, ds], -5, 5).astype(np.float32)).to(dev)
                        x = torch.stack([torch.from_numpy(jp(w)).permute(2, 0, 1) for w in req["wrist"]]).to(dev)
                        img = (T.functional.resize(x, list(size), antialias=True).float() / 255 - MEAN.to(dev)) / STD.to(dev)
                        snp_t = torch.from_numpy(np.clip(np.stack(snp), -5, 5).astype(np.float32)).to(dev)
                        plan, m2, pred, err = model.step(img, s12, U, torch.cat(ms), torch.cat(plans), torch.cat(preds), snp_t,
                                                         torch.tensor(first, device=dev))
                    if a.elog:
                        ee = err.float().abs().cpu().numpy()
                        with open(a.elog, "a") as fh:
                            for j, i in enumerate(ids):
                                fh.write(json.dumps(dict(id=int(i), first=bool(first[j]), e_joint=round(float(ee[j, :6].mean()), 4), e_img=round(float(ee[j, 6:].mean()), 4),
                                                         e_joint_max=round(float(ee[j, :6].max()), 4))) + "\n")
                    for j, i in enumerate(ids):
                        mem[i] = dict(m=m2[j:j + 1].float(), plan=plan[j:j + 1].float(), pred=pred[j:j + 1].float(), sn=np.clip(sn[j], -5, 5),
                                      S=S[j].copy(), text=texts[j])
                    p = plan.float().cpu().numpy()
                    tgt = np.empty((n, C.HM, 6))
                    tgt[..., :5] = S[:, None, :5] + p[..., :5] * st["dstd"]
                    tgt[..., 5] = (p[..., 5] + 1) / 2 * (st["g_hi"] - st["g_lo"]) + st["g_lo"]
                    c.send(((tgt - b_s) / np.where(np.abs(a_s) > 1e-3, a_s, 1.0)).astype(float).tolist())


if __name__ == "__main__":
    main()
