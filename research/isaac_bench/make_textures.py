"""Procedural table-top textures (512x512 PNG): wood, marble, fabric, concrete, checker, rubber mat, paper."""
import numpy as np, os
from PIL import Image, ImageFilter
rng = np.random.default_rng(0); S = 512; out = os.path.expanduser("~/VLA/research/isaac_bench/textures")
def noise(scale, oct=4):
    acc = np.zeros((S, S))
    for o in range(oct):
        n = max(2, int(S / scale * 2 ** o))
        g = rng.random((n, n)); im = Image.fromarray((g * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)
        acc += np.asarray(im, float) / 255 / 2 ** o
    return (acc - acc.min()) / (acc.max() - acc.min())
def save(a, name): Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).save(f"{out}/{name}.png")
y, x = np.mgrid[0:S, 0:S] / S
def wood(c1, c2, name, freq=18):
    n = noise(64); r = np.sin((y * freq + n * 4) * np.pi * 2) * 0.5 + 0.5; r = r ** 3 * 0.6 + noise(8) * 0.4
    save(np.array(c1)[None, None] * (1 - r[..., None]) + np.array(c2)[None, None] * r[..., None], name)
wood((0.72, 0.52, 0.32), (0.50, 0.32, 0.18), "wood_oak"); wood((0.85, 0.74, 0.58), (0.70, 0.58, 0.42), "wood_birch", 12)
wood((0.35, 0.22, 0.14), (0.22, 0.13, 0.08), "wood_walnut", 22)
def marble(base, vein, name):
    n = noise(96, 5); v = np.abs(np.sin((x * 6 + y * 3 + n * 6) * np.pi)) ** 0.15; v = 1 - v
    save(np.array(base)[None, None] * (1 - v[..., None]) + np.array(vein)[None, None] * v[..., None], name)
marble((0.92, 0.91, 0.89), (0.45, 0.45, 0.48), "marble_white"); marble((0.15, 0.15, 0.17), (0.75, 0.75, 0.75), "marble_black")
for name, c in (("fabric_blue", (0.20, 0.30, 0.55)), ("fabric_green", (0.25, 0.45, 0.30))):
    w = (np.sin(x * S * 0.8) * np.sin(y * S * 0.8)) * 0.08 + noise(16) * 0.15
    save(np.array(c)[None, None] * (0.9 + w[..., None]), name)
for name, c in (("concrete", (0.6, 0.6, 0.58)), ("paper_kraft", (0.70, 0.58, 0.42))):
    save(np.array(c)[None, None] * (0.8 + 0.35 * noise(24, 6)[..., None]), name)
chk = (((x * 8).astype(int) + (y * 8).astype(int)) % 2)[..., None]; save(0.85 - 0.6 * chk * np.ones(3), "checker")
g = np.ones((S, S, 3)) * 0.12; g[(np.arange(S) % 64 < 2)[:, None] | (np.arange(S) % 64 < 2)[None, :]] = 0.4; save(g + noise(8)[..., None] * 0.05, "cutting_mat")
print(sorted(os.listdir(out)))
