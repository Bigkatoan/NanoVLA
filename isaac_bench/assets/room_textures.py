"""Procedural room textures (512x512 PNG) for isaac_bench/room.py: walls, floors, ceilings, window views, door.
Colours are baked into each variant (one material per file, pre-compiled once, toggled by visibility at runtime).
    /usr/bin/python3 isaac_bench/assets/room_textures.py     (also called by fetch_room.py)
"""
import os
import numpy as np
from PIL import Image, ImageFilter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "room", "tex")
S = 512
rng = np.random.default_rng(0)
y, x = np.mgrid[0:S, 0:S] / S


def noise(scale, octaves=4):
    acc = np.zeros((S, S))
    for o in range(octaves):
        n = max(2, int(S / scale * 2 ** o))
        g = rng.random((n, n)); im = Image.fromarray((g * 255).astype(np.uint8)).resize((S, S), Image.BICUBIC)
        acc += np.asarray(im, float) / 255 / 2 ** o
    return (acc - acc.min()) / (acc.max() - acc.min())


def save(a, name):
    Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).save(f"{OUT}/{name}.png")


def col(c, a):
    return np.array(c)[None, None] * a[..., None]


def plaster(c, name, amp=0.10):
    save(col(c, 1 - amp + amp * 2 * noise(16, 6)), name)


def stripes(c1, c2, name, n=10):
    s = (np.sin(x * n * 2 * np.pi) > 0.3)[..., None]
    save((np.array(c1) * (1 - s) + np.array(c2) * s) * (0.94 + 0.06 * noise(8)[..., None]), name)


def pattern(c1, c2, name, n=8):
    u, v = (x * n) % 1 - 0.5, (y * n) % 1 - 0.5
    d = ((np.abs(u) + np.abs(v)) < 0.22)[..., None]
    save((np.array(c1) * (1 - d) + np.array(c2) * d) * (0.95 + 0.05 * noise(8)[..., None]), name)


def bricks(c, mortar, name, rows=8, cols=4):
    r = (y * rows).astype(int); off = (r % 2) * 0.5
    bx = ((x * cols + off) % 1); by = (y * rows) % 1
    m = ((bx < 0.04) | (by < 0.08))[..., None]
    shade = rng.uniform(0.8, 1.15, (rows + 1, cols + 2))[r, ((x * cols + off).astype(int))][..., None]
    save(np.where(m, np.array(mortar), np.array(c) * shade * (0.9 + 0.2 * noise(4)[..., None])), name)


def tiles(c, grout, name, n=4, jitter=0.04):
    m = (((x * n) % 1 < 0.025) | ((y * n) % 1 < 0.025))[..., None]
    sh = 1 + rng.uniform(-jitter, jitter, (n + 1, n + 1))[(y * n).astype(int), (x * n).astype(int)][..., None]
    save(np.where(m, np.array(grout), np.array(c) * sh * (0.96 + 0.06 * noise(16)[..., None])), name)


def planks(c1, c2, name, n=6, freq=40):
    k = (x * n).astype(int); seam = ((x * n) % 1 < 0.02)[..., None]
    shift = rng.random(n + 1)[k]; endj = (((y + shift) * 1.5) % 1 < 0.008)[..., None]
    g = np.sin(((y + shift) * freq + noise(48) * 5 + k * 1.7) * np.pi * 2) * 0.5 + 0.5; g = g ** 3 * 0.6 + noise(8) * 0.4
    tone = rng.uniform(0.85, 1.1, n + 1)[k][..., None]
    a = (np.array(c1) * (1 - g[..., None]) + np.array(c2) * g[..., None]) * tone
    save(np.where(seam | endj, a * 0.55, a), name)


def carpet(c, name):
    save(col(c, 0.8 + 0.3 * noise(3, 3) * 0.6 + 0.15 * noise(32)), name)


def view(name, sky_top, sky_bot, ground, kind):
    """outdoor view seen through a window (emissive): sky gradient + buildings / trees"""
    a = np.array(sky_top)[None, None] * (1 - y[..., None]) + np.array(sky_bot)[None, None] * y[..., None]
    a = a * (0.97 + 0.06 * noise(64)[..., None])
    hor = 0.62
    if kind == "city":
        for _ in range(28):
            w = rng.uniform(0.04, 0.12); x0 = rng.uniform(-0.05, 1.0); top = rng.uniform(0.18, 0.55)
            m = (x >= x0) & (x < x0 + w) & (y > top)
            c = rng.uniform(0.3, 0.65) * np.array(rng.uniform(0.9, 1.08, 3))
            win = (((x - x0) * 60 % 1) < 0.5) & ((y * 70 % 1) < 0.5) & (rng.random() < 0.7)
            a[m] = c * 0.8; a[m & win] = c * 1.25
    elif kind == "trees":
        tl = hor - 0.18 * noise(12, 3)[0][None, :] - 0.05
        m = y > tl
        a[m] = col((0.18, 0.32, 0.15), 0.6 + 0.6 * noise(6, 5))[m]
    a[y > hor + 0.12] = col(ground, 0.8 + 0.3 * noise(8))[y > hor + 0.12]
    save(a, name)


WALLS = {
    "plaster_white": lambda n: plaster((0.90, 0.89, 0.86), n), "plaster_beige": lambda n: plaster((0.86, 0.79, 0.66), n),
    "plaster_grey": lambda n: plaster((0.68, 0.70, 0.72), n), "paint_sage": lambda n: plaster((0.60, 0.68, 0.56), n, 0.05),
    "paint_blue": lambda n: plaster((0.55, 0.65, 0.78), n, 0.05), "paint_terracotta": lambda n: plaster((0.78, 0.52, 0.40), n, 0.05),
    "wallpaper_stripe": lambda n: stripes((0.86, 0.84, 0.76), (0.72, 0.66, 0.55), n),
    "wallpaper_pattern": lambda n: pattern((0.80, 0.82, 0.84), (0.55, 0.60, 0.70), n),
    "brick_red": lambda n: bricks((0.58, 0.28, 0.20), (0.75, 0.73, 0.68), n), "brick_white": lambda n: bricks((0.88, 0.87, 0.84), (0.70, 0.70, 0.68), n),
    "wood_panel": lambda n: planks((0.66, 0.48, 0.30), (0.50, 0.34, 0.20), n, 8, 20),
    "tiles_white": lambda n: tiles((0.92, 0.92, 0.90), (0.70, 0.70, 0.70), n, 6),
    "concrete": lambda n: plaster((0.56, 0.56, 0.54), n, 0.1),
}
FLOORS = {
    "wood_oak": lambda n: planks((0.72, 0.52, 0.32), (0.50, 0.32, 0.18), n), "wood_light": lambda n: planks((0.86, 0.74, 0.56), (0.70, 0.57, 0.40), n),
    "wood_dark": lambda n: planks((0.36, 0.23, 0.14), (0.22, 0.13, 0.08), n), "tiles_grey": lambda n: tiles((0.62, 0.62, 0.60), (0.40, 0.40, 0.40), n, 2),
    "tiles_beige": lambda n: tiles((0.82, 0.76, 0.64), (0.55, 0.52, 0.46), n, 3), "checker": lambda n: save(((((x * 4).astype(int) + (y * 4).astype(int)) % 2)[..., None] * -0.6 + 0.85) * np.ones(3) * (0.95 + 0.05 * noise(16)[..., None]), n),
    "carpet_grey": lambda n: carpet((0.42, 0.43, 0.46), n), "carpet_blue": lambda n: carpet((0.22, 0.28, 0.42), n),
    "carpet_red": lambda n: carpet((0.50, 0.20, 0.18), n), "concrete_floor": lambda n: plaster((0.50, 0.50, 0.48), n, 0.25),
    "lino_green": lambda n: plaster((0.45, 0.58, 0.50), n, 0.06),
}
CEILINGS = {
    "ceil_white": lambda n: plaster((0.93, 0.93, 0.92), n, 0.04), "ceil_panels": lambda n: tiles((0.90, 0.90, 0.88), (0.60, 0.60, 0.60), n, 2, 0.01),
    "ceil_wood": lambda n: planks((0.80, 0.66, 0.48), (0.66, 0.52, 0.36), n, 10, 10),
}
VIEWS = {
    "view_city": lambda n: view(n, (0.35, 0.55, 0.85), (0.80, 0.88, 0.95), (0.35, 0.35, 0.35), "city"),
    "view_trees": lambda n: view(n, (0.40, 0.60, 0.90), (0.85, 0.90, 0.95), (0.30, 0.45, 0.22), "trees"),
    "view_overcast": lambda n: view(n, (0.70, 0.72, 0.75), (0.88, 0.88, 0.88), (0.40, 0.40, 0.38), "city"),
    "view_dusk": lambda n: view(n, (0.20, 0.22, 0.45), (0.95, 0.60, 0.35), (0.12, 0.12, 0.14), "city"),
}
MISC = {
    "door_wood": lambda n: planks((0.55, 0.38, 0.22), (0.45, 0.30, 0.17), n, 1, 12), "door_white": lambda n: plaster((0.92, 0.92, 0.90), n, 0.03),
    "rug_red": lambda n: pattern((0.55, 0.15, 0.12), (0.85, 0.70, 0.40), n, 6), "rug_blue": lambda n: stripes((0.18, 0.25, 0.45), (0.80, 0.78, 0.70), n, 6),
    "rug_beige": lambda n: carpet((0.78, 0.72, 0.60), n),
}
ALL = {"wall": WALLS, "floor": FLOORS, "ceiling": CEILINGS, "view": VIEWS, "misc": MISC}


def main():
    os.makedirs(OUT, exist_ok=True)
    for group in ALL.values():
        for name, f in group.items():
            f(name)
    print("room textures ->", OUT, len(os.listdir(OUT)))


if __name__ == "__main__":
    main()
