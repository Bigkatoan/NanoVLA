"""Policy server that lets a finished student (models/<name>/) drive the Isaac Sim tabletop (research/isaac_bench/tabletop_eval.py).

The simulator works in radians, the models in degrees (SO-101 convention): state * 57.3 in, chunk / 57.3 out.
The student knows a closed set of instructions; an unseen instruction is matched to the closest known one.

    envs/lerobot/bin/python -m nanovla.sim_server --model models/sim_pick --port 6200
"""
import argparse, difflib, io, json, math, os
import numpy as np, torch
from multiprocessing.connection import Listener
from PIL import Image

DEG = 180 / math.pi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, default=6200); ap.add_argument("--jpeg", type=int, default=90)
    a = ap.parse_args()
    from nanovla.student import Student
    meta = json.load(open(os.path.join(a.model, "meta.json"))); ck = torch.load(os.path.join(a.model, "student.pt"), map_location="cpu", weights_only=False)
    m = Student(len(ck["tasks"])); m.load_state_dict(ck["model"]); m = m.cuda().eval()
    mu, sd = np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32); tasks = ck["tasks"]; H, W = meta["image_hw"]
    step = max(1, round(30 / meta.get("fps", 30)))                     # control steps (30 Hz) per chunk element
    tid_cache = {}

    def tid(t):
        if t not in tid_cache:
            tid_cache[t] = tasks.index(t) if t in tasks else tasks.index(difflib.get_close_matches(t, tasks, n=1, cutoff=0)[0])
        return tid_cache[t]

    def jp(x):
        b = io.BytesIO(); Image.fromarray(np.asarray(x, np.uint8)).save(b, format="JPEG", quality=a.jpeg); return np.asarray(Image.open(b).convert("RGB"))
    with Listener(("localhost", a.port), authkey=b"vla-isaac") as L:
        print(json.dumps(dict(event="ready", port=a.port, model=a.model, step=step, tasks=len(tasks))), flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try:
                        req = c.recv()
                    except EOFError:
                        break
                    if req == "meta":
                        c.send(dict(step=step, H=meta.get("chunk", 50), cams="wrist")); continue
                    if req == "quit":
                        return
                    q = np.asarray(req["state"], np.float32) * DEG
                    img = torch.stack([torch.from_numpy(jp(w).copy()).permute(2, 0, 1) for w in req["wrist"]]).float()
                    img = torch.nn.functional.interpolate(img, size=(H, W), mode="area").round().byte().cuda()
                    with torch.no_grad():
                        out = m(img, torch.from_numpy((q - mu) / sd).cuda(), torch.tensor([tid(t) for t in req["text"]]).cuda()).float().cpu().numpy()
                    c.send(((out * sd + mu) / DEG).tolist())


if __name__ == "__main__":
    main()
