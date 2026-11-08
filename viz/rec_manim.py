"""NanoVLA-R (recurrent) architecture, picture-first. Real wrist frames from viz/data.npz (pick_c5 ep0).
Render: manim -qm --media_dir manim_media rec_manim.py NanoVLARArch"""
import os
import numpy as np
from manim import *

Hd = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f"{Hd}/data.npz")
BG = "#0f1115"
CV, CT, CS, CL, CQ = "#4cc9f0", "#b388ff", "#80ed99", "#ff8a3d", "#ffd166"
JC = ["#4cc9f0", "#80ed99", "#ffd166", "#ff8a3d", "#b388ff", "#f72585"]


def tag(s, fs=16, col=GREY_B):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def img(a, h):
    m = ImageMobject(np.asarray(a, dtype=np.uint8)); m.set_height(h); return m


def box(w, h, col=GREY_E, stroke=GREY_B):
    return RoundedRectangle(corner_radius=min(.1, h / 3, w / 3), width=w, height=h, stroke_color=stroke, fill_color=col, fill_opacity=1, stroke_width=1.5)


def A(a, b, c=GREY_B, w=3):
    return Arrow(a, b, buff=.06, stroke_width=w, color=c, max_tip_length_to_length_ratio=.15)


def vec(n, col, w=.11, h=.11, vertical=True):
    g = VGroup(*[Square(w, fill_color=col, fill_opacity=1, stroke_width=0) for _ in range(n)])
    return g.arrange(DOWN if vertical else RIGHT, buff=.02)


class NanoVLARArch(Scene):
    def construct(self):
        self.camera.background_color = BG
        seq = Z["seq"]
        # ---------------------------------------------------------------- 1 episode start: text -> u (cached), u + joints -> h0
        words = VGroup(*[RoundedRectangle(corner_radius=.05, width=.3, height=.2, fill_color="#3a2a4f", fill_opacity=1, stroke_color=CT, stroke_width=1.2) for _ in range(4)]).arrange(RIGHT, buff=.04).move_to([-5.9, 2.6, 0])
        txt = tag("pick up the small tuna can", 12, CT); txt.set_width(1.6); txt.next_to(words, DOWN, buff=.1)
        tenc = VGroup(*[box(.9, .12, "#3a2a4f", CT).shift(UP * i * .16) for i in range(4)]).move_to([-4.3, 2.6, 0])
        u = vec(6, CT).move_to([-3.2, 2.6, 0])
        disk = VGroup(Ellipse(width=.5, height=.18, color=CT, fill_color="#3a2a4f", fill_opacity=1), Rectangle(width=.5, height=.3, stroke_width=0, fill_color="#3a2a4f", fill_opacity=1).shift(DOWN * .15),
                      Ellipse(width=.5, height=.18, color=CT, fill_color="#3a2a4f", fill_opacity=1).shift(DOWN * .3)).move_to([-3.2, 1.55, 0])
        self.play(FadeIn(words, lag_ratio=.1), FadeIn(txt), FadeIn(tenc, lag_ratio=.1), run_time=1)
        self.play(GrowArrow(A(words.get_right(), tenc.get_left(), CT)), GrowArrow(A(tenc.get_right(), u.get_left(), CT)), FadeIn(u, lag_ratio=.1))
        self.play(FadeIn(disk), GrowArrow(A(u.get_bottom(), disk.get_top(), CT)), FadeIn(tag("cache", 12, CT).next_to(disk, LEFT, buff=.12)))
        s0 = Z["st"][0]
        jb = VGroup(*[Rectangle(width=.1, height=.1 + .25 * abs(v), fill_color=JC[j], fill_opacity=1, stroke_width=0) for j, v in enumerate(s0)]).arrange(RIGHT, buff=.03, aligned_edge=DOWN).move_to([-5.6, .9, 0])
        init = box(.7, .5, "#3d2a15", CL).move_to([-4.0, .9, 0])
        self.play(FadeIn(jb), FadeIn(init), GrowArrow(A(jb.get_right(), init.get_left(), CS)), GrowArrow(A(u.get_left() + DOWN * .25, init.get_top(), CT)))
        # ---------------------------------------------------------------- 2 unrolled LSTM over ticks (real frames, each encoded once)
        K = 5; xs = [-1.9 + i * 1.95 for i in range(K)]; yL = .9
        cells = VGroup(*[box(1.0, .7, "#3d2a15", CL).move_to([x, yL, 0]) for x in xs])
        h0 = A(init.get_right(), cells[0].get_left(), CL, 4)
        self.play(GrowArrow(h0), FadeIn(cells[0]))
        frames = Group(); vecs = VGroup(); pools = VGroup(); heads = VGroup(); chunks = VGroup()
        fi = np.linspace(0, len(seq) - 1, K).astype(int)
        for k, x in enumerate(xs):
            fr = img(seq[fi[k]], .9).move_to([x, -2.6, 0])
            dino = VGroup(*[Square(.07, fill_color=CV, fill_opacity=.85, stroke_width=0) for _ in range(54)]).arrange_in_grid(6, 9, buff=.015).move_to([x, -1.75, 0])
            pool = VGroup(*[Circle(.055, fill_color=CQ, fill_opacity=1, stroke_width=0) for _ in range(4)]).arrange(RIGHT, buff=.05).move_to([x, -1.25, 0])
            v = vec(5, CV, .1, .1).move_to([x - .3, -.45, 0])
            jt = VGroup(*[Rectangle(width=.06, height=.12, fill_color=JC[j], fill_opacity=1, stroke_width=0) for j in range(6)]).arrange(RIGHT, buff=.02).move_to([x + .25, -.45, 0])
            uu = vec(3, CT, .1, .1).move_to([x + .55, -.45, 0])
            anims = [FadeIn(fr), FadeIn(dino), GrowArrow(A(fr.get_top(), dino.get_bottom(), CV, 2)), FadeIn(pool), GrowArrow(A(dino.get_top(), pool.get_bottom(), CQ, 2)),
                     FadeIn(v), FadeIn(jt), FadeIn(uu), GrowArrow(A(VGroup(v, jt, uu).get_top(), cells[k].get_bottom(), GREY_B, 2))]
            if k:
                anims += [FadeIn(cells[k]), GrowArrow(A(cells[k - 1].get_right(), cells[k].get_left(), CL, 4))]
            # action chunk above each cell (real future joints)
            ax = Axes(x_range=[0, 15], y_range=[-2.2, 2.2], x_length=1.2, y_length=.9, tips=False, axis_config={"stroke_width": 0.8, "color": GREY_D}).move_to([x, 2.55, 0])
            ac = Z["ac"]; f0 = min(int(fi[k] * len(ac) / len(seq)), len(ac) - 1); idx = np.minimum(f0 + 3 * np.arange(16), len(ac) - 1)
            cur = VGroup(*[ax.plot_line_graph(np.arange(16), ac[idx, j], add_vertex_dots=False, line_color=JC[j], stroke_width=2).set_fill(opacity=0) for j in range(6)])
            hd = box(1.0, .35, "#3d3415", CQ).move_to([x, 1.75, 0])
            anims += [FadeIn(hd), GrowArrow(A(cells[k].get_top(), hd.get_bottom(), CQ, 2)), FadeIn(ax), Create(cur), GrowArrow(A(hd.get_top(), ax.get_bottom(), CQ, 2))]
            if k == 0:
                self.play(*anims[:9], run_time=1.4)
                self.play(*anims[9:], run_time=1.2)
                self.play(FadeIn(tag("DINOv2", 11, CV).next_to(dino, LEFT, buff=.1)), FadeIn(tag("pool ×4", 11, CQ).next_to(pool, LEFT, buff=.1)),
                          FadeIn(tag("LSTM", 12, CL).move_to(cells[0])), FadeIn(tag("flow", 11, CQ).move_to(hd)), FadeIn(tag("16×6", 11).next_to(ax, UP, buff=.05)))
                plus = lambda m: VGroup(Circle(.12, color=GREEN, stroke_width=2, fill_color=BG, fill_opacity=1), Text("+", font="DejaVu Sans", font_size=16, color=GREEN)).move_to(m)
                self.play(FadeIn(VGroup(plus(jt.get_right() + RIGHT * .05 + DOWN * .25), plus(pool.get_right() + RIGHT * .2), plus(hd.get_right() + RIGHT * .15))),
                          FadeIn(VGroup(VGroup(Circle(.12, color=GREEN, stroke_width=2), Text("+", font="DejaVu Sans", font_size=16, color=GREEN)), tag("added", 12, GREEN)).arrange(RIGHT, buff=.1).to_corner(DL, buff=.3)))
            else:
                self.play(*anims, run_time=.9)
                self.add(tag("LSTM", 12, CL).move_to(cells[k]), tag("flow", 11, CQ).move_to(hd))
            # text vector arrives from the cache (not re-encoded)
            self.play(TransformFromCopy(disk[0], uu), run_time=.35)
        self.play(FadeIn(tag("0.4 s / tick · each frame encoded once", 13).to_corner(DR, buff=.25)))
        # memory pulse along the chain
        dot = Dot(color=WHITE, radius=.09).move_to(init.get_center())
        path = VMobject().set_points_as_corners([init.get_center()] + [c.get_center() for c in cells])
        self.play(MoveAlongPath(dot, path), run_time=2.2, rate_func=linear)
        self.play(FadeOut(dot)); self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)

        # ---------------------------------------------------------------- 3 text-conditioned pooling: queries move with u
        fr = img(seq[10], 3.0).move_to([-3.2, 0, 0])
        L, T_, W, Hh = fr.get_left()[0], fr.get_top()[1], fr.width, fr.height
        grid = VGroup(*[Line([L + W * i / 12, T_, 0], [L + W * i / 12, T_ - Hh, 0], stroke_width=.6, color=WHITE, stroke_opacity=.5) for i in range(13)],
                      *[Line([L, T_ - Hh * i / 9, 0], [L + W, T_ - Hh * i / 9, 0], stroke_width=.6, color=WHITE, stroke_opacity=.5) for i in range(10)])
        self.play(FadeIn(fr), Create(grid), run_time=1)
        qs = VGroup(*[Circle(.14, fill_color=CQ, fill_opacity=1, stroke_width=0) for _ in range(4)]).arrange(DOWN, buff=.25).move_to([1.2, 0, 0])
        uvec = vec(6, CT, .14, .14).move_to([1.2, 2.4, 0])
        self.play(FadeIn(qs), FadeIn(uvec), GrowArrow(A(uvec.get_bottom(), qs.get_top(), CT)))
        cells_pos = [(4, 8), (5, 9), (3, 7), (6, 4)]
        lines = VGroup(*[Line(qs[i].get_left(), [L + W * (c + .5) / 12, T_ - Hh * (r + .5) / 9, 0], color=CQ, stroke_width=2.5) for i, (r, c) in enumerate(cells_pos)])
        self.play(Create(lines), run_time=1)
        outv = vec(8, CV, .14, .14).move_to([3.2, 0, 0])
        self.play(GrowArrow(A(qs.get_right(), outv.get_left(), CV)), FadeIn(outv, lag_ratio=.1))
        self.play(FadeIn(tag("1 vector / frame", 13).next_to(outv, DOWN, buff=.2)))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)

        # ---------------------------------------------------------------- 4 flow head: noise -> chunk in 10 steps (conditioned on h, v, q, u)
        cond = VGroup(vec(1, CL, .22), vec(1, CV, .22), vec(1, CS, .22), vec(1, CT, .22)).arrange(DOWN, buff=.12).move_to([-5.2, 0, 0])
        self.play(FadeIn(cond, lag_ratio=.2))
        ax = Axes(x_range=[0, 15], y_range=[-2.5, 2.5], x_length=7, y_length=4.5, tips=False, axis_config={"stroke_width": 1.2, "color": GREY_B}).move_to([1.5, 0, 0])
        self.play(FadeIn(ax), GrowArrow(A(cond.get_right(), ax.get_left(), CQ)))
        ac = Z["ac"]; idx = np.minimum(40 + 3 * np.arange(16), len(ac) - 1); target = ac[idx]
        rng = np.random.default_rng(0); x0 = rng.normal(0, 1, (16, 6))
        def curves(x, op=1):
            return VGroup(*[ax.plot_line_graph(np.arange(16), x[:, j], add_vertex_dots=True, line_color=JC[j], stroke_width=2.5, vertex_dot_radius=.03,
                                               vertex_dot_style={"fill_color": JC[j]}).set_fill(opacity=0).set_stroke(opacity=op) for j in range(6)])
        cur = curves(x0); self.play(FadeIn(cur))
        for k in range(1, 11):
            t = k / 10; nxt = curves((1 - t) * x0 + t * target)
            self.play(Transform(cur, nxt), run_time=.25)
        self.play(FadeIn(tag("noise → 10 steps → 1.6 s chunk", 13).next_to(ax, DOWN, buff=.2)))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)
