"""NanoVLA v1 (ckpt/vla_isaac_wrist_v1) architecture, picture-first. Real data: arch.npz / arch.json from attn.py
(real wrist frame, real forward pass, decoder cross-attention, predicted vs label chunk).
Render: manim -qm --media_dir manim_media arch_manim.py NanoVLAArch"""
import json, os
import numpy as np
from manim import *

H = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f"{H}/arch.npz"); J = json.load(open(f"{H}/arch.json"))
BG = "#0f1115"
CV, CT, CS, CE, CQ, CX = "#4cc9f0", "#b388ff", "#80ed99", "#ff8a3d", "#ffd166", "#6b7280"
JC = ["#4cc9f0", "#80ed99", "#ffd166", "#ff8a3d", "#b388ff", "#f72585"]


def tag(s, fs=18, col=GREY_B):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def img(a, h):
    m = ImageMobject(np.asarray(a, dtype=np.uint8)); m.set_height(h); return m


def box(w, h, col=GREY_E, stroke=GREY_B):
    return RoundedRectangle(corner_radius=min(.1, h / 3, w / 3), width=w, height=h, stroke_color=stroke, fill_color=col, fill_opacity=1, stroke_width=1.5)


def rgb(v):
    return ManimColor.from_rgb([float(x) / 255 for x in v])


def stack(n, w, col, stroke, gap=.2, h=.15):
    return VGroup(*[box(w, h, col, stroke).shift(UP * i * gap) for i in range(n)])


def chips(n, col, w=.13, buff=.03):
    return VGroup(*[Square(w, fill_color=col, fill_opacity=1, stroke_width=0) for _ in range(n)]).arrange(RIGHT, buff=buff)


def A(a, b, c=GREY_B, w=3):
    return Arrow(a, b, buff=.08, stroke_width=w, color=c, max_tip_length_to_length_ratio=.12)


class NanoVLAArch(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.inputs()
        self.encoders_memory()
        self.decoder_output()
        self.attention()
        self.lineage()
        self.weak_points()

    def clear(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)

    # ------------------------------------------------------------ 1 one wrist frame -> 49 tokens (+ a blank, masked wrist slot)
    def inputs(self):
        raw = img(Z["raw"], 3.0).move_to([-4.2, .6, 0])
        t1 = tag("160×120", 16).next_to(raw, DOWN, buff=.15)
        cam = VGroup(Rectangle(width=.5, height=.32, color=WHITE, fill_color=GREY_D, fill_opacity=1, stroke_width=2), Circle(.09, color=WHITE, stroke_width=2)).next_to(raw, UP, buff=.2)
        self.play(FadeIn(cam), FadeIn(raw), FadeIn(t1))
        # centre crop: 149 wide after resize, keep columns 18..130 -> 12% cut each side
        W = raw.width; side = W * 18 / 149
        cutL = Rectangle(width=side, height=raw.height, fill_color=BLACK, fill_opacity=.7, stroke_width=0).align_to(raw, LEFT).align_to(raw, UP)
        cutR = cutL.copy().align_to(raw, RIGHT)
        self.play(FadeIn(cutL), FadeIn(cutR))
        crop = img(Z["crop"], 2.6).move_to([-.6, .6, 0]); t2 = tag("112×112", 16).next_to(crop, DOWN, buff=.15)
        self.play(FadeIn(crop), FadeIn(t2), GrowArrow(A(raw.get_right(), crop.get_left())))
        L, T, S = crop.get_left()[0], crop.get_top()[1], crop.height
        grid = VGroup(*[Line([L + S * i / 7, T, 0], [L + S * i / 7, T - S, 0], stroke_width=1.2, color=WHITE) for i in range(8)],
                      *[Line([L, T - S * i / 7, 0], [L + S, T - S * i / 7, 0], stroke_width=1.2, color=WHITE) for i in range(8)])
        self.play(Create(grid), run_time=.8)
        pm = Z["crop"].astype(float).reshape(7, 16, 7, 16, 3).mean((1, 3))
        toks = VGroup(*[Square(.24, fill_color=rgb(pm[r, c]), fill_opacity=1, stroke_width=.6, stroke_color=CV) for r in range(7) for c in range(7)]).arrange_in_grid(7, 7, buff=.04).move_to([3.4, 1.5, 0])
        self.play(TransformFromCopy(VGroup(*[Square(S / 7).move_to([L + S * (c + .5) / 7, T - S * (r + .5) / 7, 0]).set_fill(rgb(pm[r, c]), 1).set_stroke(width=0) for r in range(7) for c in range(7)]), toks), run_time=1.2)
        self.play(FadeIn(tag("49 × 16×16 px", 16).next_to(toks, DOWN, buff=.12)))
        # wrist slot: blank image, 49 masked tokens
        blank = Square(1.0, fill_color=GREY_E, fill_opacity=1, stroke_color=GREY_B).move_to([1.6, -2.2, 0])
        mt = VGroup(*[Square(.12, fill_color=CX, fill_opacity=.5, stroke_width=0) for _ in range(49)]).arrange_in_grid(7, 7, buff=.03).move_to([3.4, -2.2, 0])
        x = VGroup(Line(mt.get_corner(UL), mt.get_corner(DR), color=RED, stroke_width=4), Line(mt.get_corner(UR), mt.get_corner(DL), color=RED, stroke_width=4))
        self.play(FadeIn(blank), GrowArrow(A(blank.get_right(), mt.get_left(), CX)), FadeIn(mt))
        self.play(Create(x), FadeIn(tag("wrist slot: blank, masked", 14).next_to(mt, RIGHT, buff=.2)))
        self.wait(1.2); self.clear()

    # ------------------------------------------------------------ 2 four streams -> one token memory -> 1 fusion layer
    def encoders_memory(self):
        y = [2.3, .5, -1.2, -2.6]
        cam = img(Z["crop"], 1.1).move_to([-5.7, y[0], 0])
        vit = stack(12, 1.0, "#1e3a5f", CV, .14, .1).move_to([-4.3, y[0], 0])
        words = VGroup(*[RoundedRectangle(corner_radius=.05, width=.26, height=.2, fill_color="#3a2a4f", fill_opacity=1, stroke_color=CT, stroke_width=1.2) for _ in range(5)]).arrange(RIGHT, buff=.04).move_to([-5.8, y[1], 0])
        txt = tag(J["text"], 12, CT); txt.set_width(min(txt.width, 1.9)); txt.next_to(words, DOWN, buff=.1)
        ten = stack(4, 1.0, "#3a2a4f", CT, .14, .1).move_to([-4.3, y[1], 0])
        s = Z["state"]; sb = VGroup(*[Rectangle(width=.12, height=.12 + .28 * abs(v), fill_color=JC[j], fill_opacity=1, stroke_width=0) for j, v in enumerate(s)]).arrange(RIGHT, buff=.04, aligned_edge=DOWN).move_to([-5.7, y[2], 0])
        mlp = VGroup(box(.5, .4, "#173d2a", CS), box(.5, .4, "#173d2a", CS)).arrange(RIGHT, buff=.15).move_to([-4.3, y[2], 0])
        robot = VGroup(Circle(.18, color=CE, stroke_width=3), Line(ORIGIN, UP * .35 + RIGHT * .2, color=CE, stroke_width=3)).move_to([-5.7, y[3], 0])
        lut = VGroup(*[Rectangle(width=.8, height=.05, fill_color=CE if i == 6 else GREY_D, fill_opacity=1, stroke_width=0) for i in range(10)]).arrange(DOWN, buff=.02).move_to([-4.3, y[3], 0])
        self.play(LaggedStart(*[FadeIn(m) for m in (cam, vit, words, txt, ten, sb, mlp, robot, lut)], lag_ratio=.08), run_time=1.6)
        self.play(*[GrowArrow(A(a.get_right(), b.get_left(), c)) for a, b, c in ((cam, vit, CV), (words, ten, CT), (sb, mlp, CS), (robot, lut, CE))], run_time=.6)
        tags = VGroup(tag("ViT 12×192", 13, CV).next_to(vit, DOWN, buff=.08), tag("4×192", 13, CT).next_to(ten, DOWN, buff=.08),
                      tag("MLP", 13, CS).next_to(mlp, DOWN, buff=.08), tag("1 of 1232", 13, CE).next_to(lut, DOWN, buff=.08))
        self.play(FadeIn(tags))
        # state dropout during training: bars vanish half the time
        for _ in range(2):
            self.play(*[b.animate.stretch_to_fit_height(.02, about_edge=DOWN) for b in sb], run_time=.25)
            self.play(*[b.animate.stretch_to_fit_height(.12 + .28 * abs(v), about_edge=DOWN) for b, v in zip(sb, s)], run_time=.25)
        self.play(FadeIn(tag("p=0.5 → 0", 13, RED).next_to(sb, UP, buff=.08)))
        # memory sequence (vertical strip) 49 | 49 masked | text | state | robot, d=256
        x0 = -1.6
        mv = chips(49, CV, .1, .015); mw = chips(49, CX, .1, .015).set_opacity(.35); mtx = chips(9, CT, .1, .015); ms = chips(1, CS, .16); me = chips(1, CE, .16)
        mem = VGroup(VGroup(mv[:25], mv[25:]).arrange(DOWN, buff=.03), VGroup(mw[:25], mw[25:]).arrange(DOWN, buff=.03), mtx, ms, me).arrange(DOWN, buff=.25).move_to([x0 + 1.4, 0, 0])
        self.play(*[GrowArrow(A(m.get_right(), [x0, m.get_center()[1], 0], c)) for m, c in ((vit, CV), (ten, CT), (mlp, CS), (lut, CE))], run_time=.6)
        self.play(TransformFromCopy(vit, mem[0]), FadeIn(mem[1]), TransformFromCopy(ten, mem[2]), TransformFromCopy(mlp, mem[3]), TransformFromCopy(lut, mem[4]), run_time=1.2)
        self.play(FadeIn(tag("~108 × 256", 14, WHITE).next_to(mem, UP, buff=.2)))
        fuse = box(.9, 4.6, "#2b2b2b", WHITE).move_to([2.6, 0, 0])
        arcs = VGroup(*[ArcBetweenPoints(mem[i].get_right() + RIGHT * .05, mem[j].get_right() + RIGHT * .05, angle=-PI / 2, stroke_width=1.2, color=GREY_B)
                        for i, j in ((0, 2), (2, 3), (0, 3), (0, 4), (2, 4))])
        self.play(Create(arcs), FadeIn(fuse), run_time=1)
        self.play(FadeIn(tag("1 layer", 14).next_to(fuse, DOWN, buff=.1)))
        # attention share of the decoder (real, mean of 12 frames)
        sh = J["share"]; parts = [("vision", sh["vision"], CV), ("text", sh["text"], CT), ("state", sh["state"], CS), ("robot", sh["emb"], CE)]
        bars = VGroup()
        for i, (n_, v, c) in enumerate(parts):
            b = Rectangle(width=v * 4.5, height=.3, fill_color=c, fill_opacity=1, stroke_width=0)
            bars.add(VGroup(b, tag(f"{v * 100:.0f}%", 14, WHITE)).arrange(RIGHT, buff=.12))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=.18).move_to([5.3, .2, 0])
        self.play(LaggedStart(*[GrowFromEdge(b[0], LEFT) for b in bars], lag_ratio=.15), FadeIn(VGroup(*[b[1] for b in bars])), run_time=1.2)
        self.play(FadeIn(tag("decoder attention", 13).next_to(bars, UP, buff=.2)), FadeIn(tag("1 state token ≈ 49 vision tokens / 2", 12).next_to(bars, DOWN, buff=.2)))
        self.wait(1.8); self.clear()

    # ------------------------------------------------------------ 3 16 queries, 3 decoder layers -> 16×6 -> joint targets (real forward pass)
    def decoder_output(self):
        mem = VGroup(*[Square(.12, fill_color=c, fill_opacity=1, stroke_width=0) for c in [CV] * 18 + [CT] * 4 + [CS, CE]]).arrange(DOWN, buff=.04).move_to([-6.0, 0, 0])
        q = VGroup(*[Circle(.08, fill_color=CQ, fill_opacity=1, stroke_width=0) for _ in range(16)]).arrange(DOWN, buff=.08).move_to([-4.6, 0, 0])
        layers = VGroup()
        for i in range(3):
            sa = box(.35, 3.0, "#3d3415", CQ); ca = box(.35, 3.0, "#1d2b3a", CV)
            layers.add(VGroup(sa, ca).arrange(RIGHT, buff=.08))
        layers.arrange(RIGHT, buff=.3).move_to([-2.3, 0, 0])
        self.play(FadeIn(mem), FadeIn(q, lag_ratio=.05), FadeIn(layers, lag_ratio=.2), run_time=1.2)
        cur = VGroup(*[CurvedArrow(mem.get_top() + UP * .1, l[1].get_top() + UP * .05, angle=-.6, color=CV, stroke_width=1.5) for l in layers])
        self.play(GrowArrow(A(q.get_right(), layers.get_left(), CQ)), Create(cur), run_time=1)
        self.play(FadeIn(VGroup(tag("self", 12, CQ).next_to(layers[0][0], DOWN, buff=.1), tag("cross", 12, CV).next_to(layers[0][1], DOWN, buff=.1))))
        out = VGroup(*[VGroup(*[Square(.12, fill_color=JC[j], fill_opacity=1, stroke_width=0) for j in range(6)]).arrange(RIGHT, buff=.02) for _ in range(16)]).arrange(DOWN, buff=.04).move_to([-.2, 0, 0])
        self.play(GrowArrow(A(layers.get_right(), out.get_left(), CQ)), FadeIn(out, lag_ratio=.03), run_time=1)
        self.play(FadeIn(tag("16×6", 14).next_to(out, DOWN, buff=.12)))
        # decode: joints = state + delta·σ ; gripper absolute; real pred vs label
        pred, gt, st = Z["pred"], Z["gt"], Z["state"]
        lo, hi = float(min(pred.min(), gt.min(), st.min())) - .1, float(max(pred.max(), gt.max(), st.max())) + .1
        ax = Axes(x_range=[-1, 16, 4], y_range=[lo, hi, .5], x_length=5.6, y_length=5.0, tips=False, axis_config={"stroke_width": 1.2, "color": GREY_B}).move_to([3.7, 0, 0])
        self.play(FadeIn(ax), GrowArrow(A(out.get_right(), ax.get_left(), CQ)))
        sd = VGroup(*[Dot(ax.c2p(-.5, st[j]), color=JC[j], radius=.07) for j in range(6)])
        self.play(FadeIn(sd))
        g = VGroup(*[ax.plot_line_graph(np.arange(16), gt[:, j], add_vertex_dots=False, line_color=JC[j], stroke_width=7).set_stroke(opacity=.3) for j in range(6)])
        p = VGroup(*[ax.plot_line_graph(np.r_[-.5, np.arange(16)], np.r_[st[j], pred[:, j]], add_vertex_dots=True, line_color=JC[j], stroke_width=2.5, vertex_dot_radius=.03,
                                        vertex_dot_style={"fill_color": JC[j]}) for j in range(6)])
        for m in (*g, *p):
            m.set_fill(opacity=0)
        self.play(Create(g), run_time=1)
        self.play(Create(p, lag_ratio=.1), run_time=1.6)
        leg = VGroup(VGroup(Line(ORIGIN, RIGHT * .5, stroke_width=7, color=WHITE).set_stroke(opacity=.3), tag("label", 13)).arrange(RIGHT, buff=.1),
                     VGroup(Line(ORIGIN, RIGHT * .5, stroke_width=2.5, color=WHITE), tag("pred", 13)).arrange(RIGHT, buff=.1),
                     VGroup(Dot(radius=.06), tag("state now", 13)).arrange(RIGHT, buff=.1)).arrange(RIGHT, buff=.4).next_to(ax, UP, buff=.15)
        self.play(FadeIn(leg), FadeIn(tag("1.6 s · rad · train frame", 12).next_to(ax, DOWN, buff=.1)))
        self.wait(1.8); self.clear()

    # ------------------------------------------------------------ 4 where the decoder looks (real cross-attention, last layer)
    def attention(self):
        ims = Group()
        for i in range(4):
            a = Group(img(Z["ov"][i], 2.6), tag(J["tasks"][i], 15, WHITE))
            a[1].next_to(a[0], DOWN, buff=.12); ims.add(a)
        ims.arrange(RIGHT, buff=.35).move_to(UP * .2)
        self.play(LaggedStart(*[FadeIn(m, shift=UP * .2) for m in ims], lag_ratio=.2), run_time=1.6)
        bar = VGroup(*[Rectangle(width=.3, height=.15, fill_color=c, fill_opacity=1, stroke_width=0) for c in ("#000004", "#57106e", "#bc3754", "#f98e09", "#fcffa4")]).arrange(RIGHT, buff=0).to_corner(DR, buff=.5)
        self.play(FadeIn(bar), FadeIn(tag("attention", 12).next_to(bar, LEFT, buff=.15)), FadeIn(tag("decoder L3 → 49 wrist tokens, mean of 16 queries · train frames", 12).to_corner(DL, buff=.5)))
        self.wait(2.4); self.clear()

    # ------------------------------------------------------------ 5 how the weights got here
    def lineage(self):
        st = [("text", CT, "MLM + SimCSE"), ("vision", CV, "DINO + IDM"), ("align", WHITE, "SigLIP"), ("VLA", CQ, "SO-100 real"), ("Isaac", CS, "2.5k demos")]
        g = VGroup()
        for n_, c, s in st:
            g.add(VGroup(Circle(.45, color=c, stroke_width=4, fill_color=BG, fill_opacity=1), tag(n_, 15, c), tag(s, 12)).arrange(DOWN, buff=.15))
        g.arrange(RIGHT, buff=.9)
        for m in g:
            m[1].move_to(m[0])
        arr = VGroup(*[A(g[i][0].get_right(), g[i + 1][0].get_left()) for i in range(4)])
        self.play(LaggedStart(*[FadeIn(m, scale=.8) for m in g], lag_ratio=.2), Create(arr), run_time=2)
        wrist = VGroup(Rectangle(width=.5, height=.32, color=WHITE, stroke_width=2), Circle(.09, color=WHITE, stroke_width=2)).next_to(g[4], UP, buff=.3)
        self.play(FadeIn(wrist), Indicate(g[4][0], color=CS))
        self.wait(1.2); self.clear()

    # ------------------------------------------------------------ 6 weak points (schematic)
    def weak_points(self):
        P = [[-4.6, 1.6, 0], [1.6, 1.6, 0], [-4.6, -1.9, 0], [1.6, -1.9, 0]]
        # (a) one frame, no history
        fr = VGroup(*[Square(.8, color=GREY_B if i < 2 else WHITE, stroke_width=2, fill_color=GREY_E, fill_opacity=.5 if i < 2 else 1).shift(RIGHT * i * .5 + DOWN * i * .1) for i in range(3)]).move_to(P[0])
        xs = VGroup(*[VGroup(Line(UL * .2, DR * .2, color=RED, stroke_width=4), Line(UR * .2, DL * .2, color=RED, stroke_width=4)).move_to(fr[i]) for i in range(2)])
        a_ = VGroup(fr, xs, tag("t-2 t-1 t", 12).next_to(fr, DOWN, buff=.15))
        # (b) gripper state dropped
        gb = VGroup(*[Rectangle(width=.18, height=h, fill_color=JC[j], fill_opacity=1, stroke_width=0) for j, h in enumerate((.4, .7, .5, .6, .3, .9))]).arrange(RIGHT, buff=.06, aligned_edge=DOWN).move_to(P[1])
        b_ = VGroup(gb, tag("state 50% = 0", 12).next_to(gb, DOWN, buff=.15))
        # (c) L1 regression averages two valid paths
        o = np.array(P[2]); start = o + LEFT * 1.4; goal = o + RIGHT * 1.4; obst = Circle(.3, color=GREY_B, fill_color=GREY_D, fill_opacity=1).move_to(o)
        up = VMobject(stroke_color=GREEN, stroke_width=3).set_points_smoothly([start, o + UP * .7, goal]); dn = VMobject(stroke_color=GREEN, stroke_width=3).set_points_smoothly([start, o + DOWN * .7, goal])
        mid = Line(start, goal, color=RED, stroke_width=4)
        c_ = VGroup(obst, up, dn, mid, tag("L1: mean of modes", 12).next_to(obst, DOWN, buff=.85))
        # (d) coarse 7x7 + 1 fusion layer
        gr = VGroup(*[Square(.22, stroke_width=1, color=GREY_B) for _ in range(49)]).arrange_in_grid(7, 7, buff=0).move_to(P[3] + LEFT * .7)
        obj = Square(.1, fill_color=CQ, fill_opacity=1, stroke_width=0).move_to(gr[24].get_center() + UR * .05)
        fl = box(.3, 1.4, "#2b2b2b", WHITE).move_to(P[3] + RIGHT * .7)
        d_ = VGroup(gr, obj, fl, tag("7×7 · 1 fusion", 12).next_to(VGroup(gr, fl), DOWN, buff=.15))
        for m in (a_, b_, c_, d_):
            self.play(FadeIn(m, lag_ratio=.1), run_time=.9)
        self.play(Create(mid.copy().set_color(RED)), run_time=.6)
        self.play(FadeIn(tag("schematic", 12).to_corner(DR, buff=.3)))
        self.wait(2); self.clear()
