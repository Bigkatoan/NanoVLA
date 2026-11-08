"""GCT stage 4: export NanoVLA-G for the Jetson Orin Nano as three ONNX graphs and measure latency here.
  vision.onnx : wrist image (1,3,128,160, normalised) -> image vector v (1,512)           every frame
  tick.onnx   : (v, joints12, u, memory, plan_prev, pred_prev, sn_prev, first) -> (plan, memory', pred)   every frame
  text.onnx   : token ids (1,L) -> task vector u (1,256)                                    once per instruction
Checks ONNX Runtime outputs against PyTorch and writes bench.json + a README with the TensorRT commands for the device.
Runs in ~/VLA/envs/deploy (torch + onnx + onnxruntime-gpu).

    ~/VLA/envs/deploy/bin/python tools/export_bench.py --ckpt ~/VLA/ckpt/nanovla_g_online/vlag.pt --out ~/VLA/deploy
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.expanduser("~/VLA/research/vla"))
import nanovla_g as G


class Vision(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, img):
        return self.m.image_vec(img)


class Tick(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, v, s12, u, mem, plan_prev, pred_prev, sn_prev, first):
        plan, m2, pred, e, z, delta, sh = self.m.tick_g(v, s12, u, mem, plan_prev, pred_prev, sn_prev, first > 0.5)
        return plan, m2, pred


class Text(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, ids):
        return self.m.text_vec(ids)


def timeit(fn, n=200, warm=20, cuda=True):
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default=os.path.expanduser("~/VLA/deploy")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m, tok = G.build("cpu", ck["st"], init=""); m.load_state_dict(ck["model"]); m.eval()
    ids = tok(["pick up the small tuna can"])
    with torch.no_grad():
        u = m.text_vec(ids); mem = m.init_mem(u)
    img = torch.randn(1, 3, 128, 160); s12 = torch.zeros(1, 12); plan = torch.zeros(1, G.HM, 6); pred = torch.zeros(1, 6 + G.PE); sn = torch.zeros(1, 6); first = torch.ones(1)
    with torch.no_grad():
        v = m.image_vec(img)
    vis, tick, txt = Vision(m), Tick(m), Text(m)
    paths = {k: os.path.join(a.out, f"{k}.onnx") for k in ("vision", "tick", "text")}
    torch.onnx.export(vis, (img,), paths["vision"], input_names=["image"], output_names=["v"], opset_version=17, dynamo=False)
    torch.onnx.export(tick, (v, s12, u, mem, plan, pred, sn, first), paths["tick"], input_names=["v", "joints12", "u", "memory", "plan_prev", "pred_prev", "sn_prev", "first"],
                      output_names=["plan", "memory_out", "pred"], opset_version=17, dynamo=False)
    torch.onnx.export(txt, (ids,), paths["text"], input_names=["ids"], output_names=["u"], dynamic_axes={"ids": {1: "L"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    res = dict(ckpt=a.ckpt, sizes_mb={k: round(os.path.getsize(p) / 1e6, 2) for k, p in paths.items()})
    feeds = dict(v=v.numpy(), joints12=s12.numpy(), u=u.numpy(), memory=mem.numpy(), plan_prev=plan.numpy(), pred_prev=pred.numpy(), sn_prev=sn.numpy(), first=first.numpy())
    with torch.no_grad():
        ref_plan = tick(v, s12, u, mem, plan, pred, sn, first)[0].numpy(); ref_v = v.numpy()
    for prov in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if prov not in ort.get_available_providers():
            continue
        sv = ort.InferenceSession(paths["vision"], providers=[prov]); stk = ort.InferenceSession(paths["tick"], providers=[prov])
        ov = sv.run(None, {"image": img.numpy()})[0]; op = stk.run(None, feeds)[0]
        key = "ort_cuda" if prov.startswith("CUDA") else "ort_cpu"
        res[key] = dict(max_abs_diff_v=float(np.abs(ov - ref_v).max()), max_abs_diff_plan=float(np.abs(op - ref_plan).max()),
                        ms_vision=round(timeit(lambda: sv.run(None, {"image": img.numpy()}), cuda=False), 2),
                        ms_tick=round(timeit(lambda: stk.run(None, feeds), cuda=False), 2))
        res[key]["ms_frame"] = round(res[key]["ms_vision"] + res[key]["ms_tick"], 2)
    if torch.cuda.is_available():
        mc = m.cuda(); ic, vc = img.cuda(), v.cuda()
        args = [x.cuda() for x in (s12, u, mem, plan, pred, sn)]; fst = torch.ones(1, dtype=torch.bool, device="cuda")
        def vis_f():
            with torch.autocast("cuda", dtype=torch.float16):
                return mc.image_vec(ic)
        def tick_f():
            with torch.autocast("cuda", dtype=torch.float16):
                return mc.tick_g(vc, args[0], args[1], args[2], args[3], args[4], args[5], fst)
        with torch.no_grad():
            res["torch_cuda_fp16"] = dict(ms_vision=round(timeit(vis_f), 2), ms_tick=round(timeit(tick_f), 2))
        res["torch_cuda_fp16"]["ms_frame"] = round(res["torch_cuda_fp16"]["ms_vision"] + res["torch_cuda_fp16"]["ms_tick"], 2)
        res["gpu"] = torch.cuda.get_device_name()
    json.dump(res, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
    open(os.path.join(a.out, "README.md"), "w").write(f"""# NanoVLA-G on Jetson Orin Nano

Three graphs (ONNX opset 17), exported from `{a.ckpt}`:

| graph | runs | inputs -> outputs |
| --- | --- | --- |
| vision.onnx | every frame (10 Hz) | image (1,3,128,160) ImageNet-normalised -> v (1,512) |
| tick.onnx | every frame | v, joints12, u, memory, plan_prev, pred_prev, sn_prev, first -> plan (1,8,6), memory_out, pred |
| text.onnx | once per instruction | token ids (1,L) -> u (1,256) |

On the Jetson (JetPack 6, TensorRT):

    /usr/src/tensorrt/bin/trtexec --onnx=vision.onnx --fp16 --saveEngine=vision.plan
    /usr/src/tensorrt/bin/trtexec --onnx=tick.onnx --fp16 --saveEngine=tick.plan

Target: vision + tick <= 33 ms per frame (30 Hz); 100 ms budget at 10 Hz. Measured on this machine: see bench.json.
The control loop interpolates the first plan point at 30 Hz; keep `memory_out`, `plan`, `pred` and the joints as the
next frame's `memory`, `plan_prev`, `pred_prev`, `sn_prev`, `first = 0`.
""")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
