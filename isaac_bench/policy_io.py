"""Client-side policy I/O for tabletop_eval.py, importable without Isaac (unit-testable / dry runs).

cams='both' : front + wrist via Tabletop.images() (unchanged behaviour)
cams='wrist': only env.scene["wrist"] is read (the scene may have no front camera); requests carry no "front".
"""
import numpy as np

import os
RES = tuple(int(x) for x in os.environ.get("TABLETOP_RES", "120x160").split("x"))   # policy / recorded resolution (H x W), e.g. 240x320
CAM_MODES = ("both", "wrist")


def cam_rgb(env, cam, res=RES):
    """(n, H, W, 3) uint8 at the policy resolution, read straight from the scene camera (same as Tabletop.images)"""
    import torch.nn.functional as F
    x = env.scene[cam].data.output["rgb"][..., :3].permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=res, mode="area")
    return x.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()


def get_images(env, cams):
    """-> dict cam -> (n,H,W,3) uint8. wrist mode never touches env.scene['front']"""
    if cams == "wrist":
        return {"wrist": cam_rgb(env, "wrist")}
    f, w = env.images()
    return {"front": f, "wrist": w}


def build_request(S, need, imgs, texts, cams, fresh=None):
    """policy-server request for the envs in `need` (indices); imgs from get_images.
    ids / new let a recurrent server (vla/vlar_server.py) keep per-env memory; stateless servers ignore them."""
    req = dict(state=np.asarray(S)[need], wrist=[imgs["wrist"][i] for i in need], text=[texts[i] for i in need],
               ids=[int(i) for i in need], new=[bool(fresh[i]) if fresh is not None else False for i in need])
    if cams != "wrist":
        req["front"] = [imgs["front"][i] for i in need]
    return req


def video_frame(imgs, cams, i=0):
    return imgs["wrist"][i] if cams == "wrist" else np.concatenate([imgs["front"][i], imgs["wrist"][i]], 1)


# ------------------------------------------------------------------ dry run (no Isaac): fake wrist-only scene + policy
class _Cam:
    def __init__(self, n, h, w, rng):
        self.rng, self.shape = rng, (n, h, w, 4)

    @property
    def data(self):
        import torch
        class D:
            pass
        d = D(); d.output = {"rgb": torch.from_numpy(self.rng.integers(0, 256, self.shape, dtype=np.uint8))}
        return d


class FakeTabletop:
    """minimal stand-in for tabletop.Tabletop with a WRIST-ONLY scene (env.scene has no 'front'; images() raises)"""
    CTRL_HZ = 30
    HOME = np.array([0.0, -0.6, -0.07, 1.51, -1.57, -0.1])

    def __init__(self, num_envs=2, seed=0, task_mix=("bowl",), ep_len=(20, 60)):
        self.n, self.rng, self.ep_len = num_envs, np.random.default_rng(seed), ep_len
        self.scene = {"wrist": _Cam(num_envs, RES[0] * 2, RES[1] * 2, self.rng)}
        self.t = np.zeros(num_envs, int); self.q = np.tile(self.HOME, (num_envs, 1)); self.ep = [None] * num_envs
        self.task_mix, self.len = task_mix, np.zeros(num_envs, int)

    def reset(self, env_ids=None, settle=0):
        for i in (range(self.n) if env_ids is None else env_ids):
            self.ep[i] = dict(task=self.task_mix[0], obj="banana", text=f"put the banana in the {self.task_mix[0]}")
            self.t[i] = 0; self.len[i] = self.rng.integers(*self.ep_len)

    def images(self):
        raise RuntimeError("FakeTabletop is wrist-only: images() would read the front camera")

    def step(self, cmd):
        assert cmd.shape == (self.n, 6) and np.isfinite(cmd).all()
        self.q = cmd.copy(); self.t += 1

    def state(self):
        return self.q.copy()

    def success(self):
        return self.t >= self.len


class FakeConn:
    """in-process policy: checks the request format, returns zero-delta chunks"""

    def __init__(self, cams, step=3, H=16):
        self.cams, self.step, self.H, self.reqs, self._out = cams, step, H, [], None

    def send(self, req):
        if req == "meta":
            self._out = dict(step=self.step, H=self.H, cams=self.cams); return
        n = len(req["state"])
        assert len(req["wrist"]) == n and len(req["text"]) == n and all(w.shape == RES + (3,) and w.dtype == np.uint8 for w in req["wrist"])
        assert ("front" in req) == (self.cams != "wrist"), sorted(req)
        self.reqs.append(req); self._out = np.repeat(np.asarray(req["state"], float)[:, None], self.H, 1)

    def recv(self):
        return self._out

    def close(self):
        pass
