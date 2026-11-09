"""SmolVLA policy server for the Isaac tabletop chunk client (isaac_bench/tabletop_eval.py --cams wrist).

Request: state (n,6) rad, wrist (n,H,W,3) uint8, text [n]. Reply: (n, chunk, 6) joint targets; chunk element k is
reached in `step` = 3 control steps (dataset fps 10, control 30 Hz). Frames are JPEG-encoded like the training data.

    ~/VLA/envs/lerobot/bin/python vla/smolvla_server.py --ckpt ~/VLA/ckpt/smolvla_sim/checkpoints/last/pretrained_model --port 6100
"""
import argparse, io, json, time
import numpy as np, torch
from multiprocessing.connection import Listener
from PIL import Image


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--port", type=int, default=6100)
    ap.add_argument("--jpeg", type=int, default=90); ap.add_argument("--steps", type=int, default=10, help="flow-matching denoising steps")
    ap.add_argument("--log", default=""); ap.add_argument("--cams", default="both", choices=("both", "wrist")); ap.add_argument("--scale", type=float, default=1.0, help="57.29578: sim radians in, degrees model")
    a = ap.parse_args()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    pol = SmolVLAPolicy.from_pretrained(a.ckpt).to("cuda").eval(); pol.config.num_steps = a.steps
    pre, post = make_pre_post_processors(pol.config, pretrained_path=a.ckpt)
    cam = "observation.images.camera1"; feats = set(pol.config.input_features)

    def jp(x):
        b_ = io.BytesIO(); Image.fromarray(np.asarray(x, np.uint8)).save(b_, format="JPEG", quality=a.jpeg); return np.asarray(Image.open(b_).convert("RGB"))
    with Listener(("localhost", a.port), authkey=b"vla-isaac") as L:
        print(json.dumps(dict(event="ready", port=a.port, arch="smolvla", cam=cam, chunk=pol.config.chunk_size)), flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try:
                        req = c.recv()
                    except EOFError:
                        break
                    if req == "meta":
                        c.send(dict(step=3, H=pol.config.chunk_size, cams=a.cams)); continue
                    if req == "quit":
                        return
                    t0 = time.time()
                    img = torch.stack([torch.from_numpy(jp(w)).permute(2, 0, 1) for w in req["wrist"]]).float() / 255
                    st = torch.tensor(np.asarray(req["state"], np.float32) * a.scale)
                    batch = {cam: img, "observation.state": st, "task": [t.strip() for t in req["text"]]}
                    if "observation.images.camera2" in feats and "front" in req:   # overview camera (2-camera teacher)
                        batch["observation.images.camera2"] = torch.stack([torch.from_numpy(jp(w).copy()).permute(2, 0, 1) for w in req["front"]]).float() / 255
                    with torch.no_grad():
                        pol.reset()
                        b = pre(batch); act = pol.predict_action_chunk(b); act = post(act)
                    out = act.float().cpu().numpy() / a.scale
                    if a.log:
                        with open(a.log, "a") as fh:
                            fh.write(json.dumps(dict(n=len(out), ms=round(1000 * (time.time() - t0), 1))) + "\n")
                    c.send(out.tolist())                       # plain lists: the Isaac client runs numpy 1.x


if __name__ == "__main__":
    main()
