"""Policy server for NanoVLA-S (state-based). Request: state (n,6), wrist (n,120,160,3), text, ids, new.
Reply: list of dicts {state, target_cam [x,y,z] (m, wrist-camera frame, x right / y down / z forward), dyaw (rad), jaw_open (0..1)}.
Keeps the previous task state per env (reset on `new`). Frames are JPEG-encoded like the training data.

    python vlas_server.py --ckpt ~/VLA/ckpt/nanovla_s/vlas.pt --port 6000
"""
import argparse, io, json
import numpy as np, torch
from multiprocessing.connection import Listener
from PIL import Image
import nanovla_s as S
from train_text import clean


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--port", type=int, default=6000); ap.add_argument("--jpeg", type=int, default=90)
    ap.add_argument("--log", default=""); ap.add_argument("--gate", type=float, default=0.0, help="m: forward state step only when the target is reached"); ap.add_argument("--oracle", default="", help="comma list of modules replaced by the client's oracle: grounder,state,grasp")
    a = ap.parse_args()
    dev = "cuda"; ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); st = ck["st"]
    if ck.get("arch") == "nanovla_mod":
        import nanovla_mod as MD
        model, tok = MD.build(dev, init="")
    else:
        model, tok = S.build(dev, init_vision="")
    model.load_state_dict(ck["model"]); model.eval()
    om = set(x for x in a.oracle.split(",") if x)
    prev, tcache = {}, {}
    def jp(x):
        b_ = io.BytesIO(); Image.fromarray(np.asarray(x, np.uint8)).save(b_, format="JPEG", quality=a.jpeg); return np.asarray(Image.open(b_).convert("RGB"))
    with Listener(("localhost", a.port), authkey=b"vla-isaac") as L:
        print(json.dumps(dict(event="ready", port=a.port, arch="nanovla_s", cams="wrist", step=ck["step_frames"])), flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try:
                        req = c.recv()
                    except EOFError:
                        break
                    if req == "meta":
                        c.send(dict(step=ck["step_frames"], H=1, cams="wrist", waypoint=True)); continue
                    if req == "quit":
                        return
                    S_ = np.asarray(req["state"], np.float64); n = len(S_)
                    ids = req.get("ids", list(range(n))); new = req.get("new", [False] * n); texts = [clean(t) for t in req["text"]]
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for t in set(texts) - set(tcache):
                            tcache[t] = model.text_vec(tok([t]).to(dev))[0].float()
                        U = torch.stack([tcache[t] for t in texts])
                        pv = torch.tensor([0 if (new[j] or i not in prev) else prev[i] for j, i in enumerate(ids)], device=dev)
                        x = torch.stack([torch.from_numpy(jp(w)).permute(2, 0, 1) for w in req["wrist"]]).to(dev)
                        img = S.gpu_img(x, False)
                        s6 = torch.from_numpy(np.clip((S_ - st["s_mean"]) / st["s_std"], -5, 5).astype(np.float32)).to(dev)
                        orc = None
                        if om and "oracle" in req:
                            O = req["oracle"]; orc = {}
                            if "grounder" in om:
                                orc["obj"] = torch.tensor([x["obj"] for x in O], device=dev); orc["dest"] = torch.tensor([x["dest"] for x in O], device=dev)
                            if "state" in om:
                                orc["state"] = torch.tensor([min(int(x["state"]), S.K - 1) for x in O], device=dev)
                            if "grasp" in om:
                                orc["dyaw"] = torch.tensor([x["dyaw"] for x in O], device=dev)
                        tcp = torch.tensor(req["tcp_cam"], device=dev, dtype=torch.float32) if (a.gate > 0 and "tcp_cam" in req) else None
                        o = model.act(img, s6, U.to(img.dtype), pv, oracle=orc, tcp=tcp, gate=a.gate) if ck.get("arch") == "nanovla_mod" else model.act(img, s6, U.to(img.dtype), pv)
                    out = []
                    for j, i in enumerate(ids):
                        prev[i] = int(o["state"][j])
                        out.append(dict(state=int(o["state"][j]), target_cam=o["target"][j].float().cpu().tolist(), dyaw=float(o["dyaw"][j]), jaw_open=float(o["jaw_open"][j]),
                                        obj_cam=o["obj"][j].float().cpu().tolist(), dest_cam=o["dest"][j].float().cpu().tolist()))
                    if a.log:
                        with open(a.log, "a") as fh:
                            for j, i in enumerate(ids):
                                fh.write(json.dumps(dict(id=int(i), new=bool(new[j]), tcp_cam=req.get("tcp_cam", [None] * n)[j], truth=req.get("truth", [None] * n)[j], **out[j])) + "\n")
                    c.send(out)


if __name__ == "__main__":
    main()
