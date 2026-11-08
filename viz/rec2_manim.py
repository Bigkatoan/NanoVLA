"""NanoVLA-R as trained now (ViT-Tiny, 10 Hz LSTM, flow head). Real internals from extract_r.py (rec.npz):
pooling attention, LSTM memory path, flow sampling path, chunk vs label; per-tick compute from FlopCounterMode.
Render: manim -qm --media_dir manim_media rec2_manim.py NanoVLARNow"""
import json, os
import numpy as np
from manim import *

Hd = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f"{Hd}/rec.npz"); J = json.load(open(f"{Hd}/rec.json"))
BG = "#0f1115"
CV, CT, CS, CL, CQ = "#4cc9f0", "#b388ff", "#80ed99", "#ff8a3d", "#ffd166"
JC = ["#4cc9f0", "#80ed99", "#ffd166", "#ff8a3d", "#b388ff", "#f72585"]
PH = ["#4cc9f0", "#ffd166", "#f72585"]          # approach, holding, released


def tag(s, fs=16, col=GREY_B):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def img(a, h):
    m = ImageMobject(np.asarray(a, dtype=np.uint8)); m.set_height(h); return m


def box(w, h, col=GREY_E, stroke=GREY_B):
    return RoundedRectangle(corner_radius=min(.1, h / 3, w / 3), width=w, height=h, stroke_color=stroke, fill_color=col, fill_opacity=1, stroke_width=1.5)


def A(a, b, c=GREY_B, w=3):
    return Arrow(a, b, buff=.06, stroke_width=w, color=c, max_tip_length_to_length_ratio=.15)


def vec(n, col, w=.11, vertical=True):
    return VGroup(*[Square(w, fill_color=col, fill_opacity=1, stroke_width=0) for _ in range(n)]).arrange(DOWN if vertical else RIGHT, buff=.02)


def heat(a):
    """attention map [8,10] -> inferno-ish RGB [8,10,3]"""
    stops = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], float)
    x = (a - a.min()) / (np.ptp(a) + 1e-9) * 4
    i = np.clip(x.astype(int), 0, 3); f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def overlay(frame, att):
    h = np.kron(heat(att), np.ones((16, 16, 1), np.uint8))
    return (0.45 * frame + 0.55 * h).astype(np.uint8)


class NanoVLARNow(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.overview()
        self.pooling()
        self.memory()
        self.flow()
        self.speed()

    def clear(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)

    # ------------------------------------------------------------ 1 one episode, unrolled (real frames)
    def overview(self):
        fr = Z["frames"]; N = len(fr)
        words = VGroup(*[RoundedRectangle(corner_radius=.05, width=.3, height=.2, fill_color="#3a2a4f", fill_opacity=1, stroke_color=CT, stroke_width=1.2) for _ in range(4)]).arrange(RIGHT, buff=.04).move_to([-5.9, 2.7, 0])
        txt = tag(J["text"], 12, CT); txt.set_width(1.7); txt.next_to(words, DOWN, buff=.1)
        tenc = VGroup(*[box(.9, .12, "#3a2a4f", CT).shift(UP * i * .16) for i in range(4)]).move_to([-4.3, 2.7, 0])
        u = vec(6, CT).move_to([-3.2, 2.7, 0])
        disk = VGroup(Ellipse(width=.5, height=.18, color=CT, fill_color="#3a2a4f", fill_opacity=1), Rectangle(width=.5, height=.3, stroke_width=0, fill_color="#3a2a4f", fill_opacity=1).shift(DOWN * .15),
                      Ellipse(width=.5, height=.18, color=CT, fill_color="#3a2a4f", fill_opacity=1).shift(DOWN * .3)).move_to([-3.2, 1.6, 0])
        st = Z["state"][0]
        jb = VGroup(*[Rectangle(width=.1, height=.1 + .25 * abs(v), fill_color=JC[j], fill_opacity=1, stroke_width=0) for j, v in enumerate(st)]).arrange(RIGHT, buff=.03, aligned_edge=DOWN).move_to([-5.6, .9, 0])
        init = box(.7, .5, "#3d2a15", CL).move_to([-4.0, .9, 0])
        self.play(FadeIn(words, lag_ratio=.1), FadeIn(txt), FadeIn(tenc, lag_ratio=.1), run_time=.9)
        self.play(GrowArrow(A(words.get_right(), tenc.get_left(), CT)), GrowArrow(A(tenc.get_right(), u.get_left(), CT)), FadeIn(u, lag_ratio=.1), run_time=.7)
        self.play(FadeIn(disk), GrowArrow(A(u.get_bottom(), disk.get_top(), CT)), FadeIn(tag("1× / episode", 12, CT).next_to(disk, LEFT, buff=.12)), run_time=.7)
        self.play(FadeIn(jb), FadeIn(init), GrowArrow(A(jb.get_right(), init.get_left(), CS)), GrowArrow(A(u.get_left() + DOWN * .25, init.get_top(), CT)), run_time=.8)
        K = 5; xs = [-1.9 + i * 1.95 for i in range(K)]; yL = .9
        cells = VGroup(*[box(1.0, .7, "#3d2a15", CL).move_to([x, yL, 0]) for x in xs])
        self.play(GrowArrow(A(init.get_right(), cells[0].get_left(), CL, 4)), FadeIn(cells[0]), run_time=.5)
        ti = np.linspace(0, N - 1, K).astype(int)
        for k, x in enumerate(xs):
            t = int(ti[k])
            f_ = img(fr[t], .85).move_to([x, -2.65, 0])
            vit = VGroup(*[Square(.06, fill_color=CV, fill_opacity=.85, stroke_width=0) for _ in range(80)]).arrange_in_grid(8, 10, buff=.012).move_to([x, -1.75, 0])
            pool = VGroup(*[Circle(.055, fill_color=CQ, fill_opacity=1, stroke_width=0) for _ in range(4)]).arrange(RIGHT, buff=.05).move_to([x, -1.2, 0])
            v = vec(5, CV, .1).move_to([x - .3, -.45, 0])
            jt = VGroup(*[Rectangle(width=.06, height=.12, fill_color=JC[j], fill_opacity=1, stroke_width=0) for j in range(6)]).arrange(RIGHT, buff=.02).move_to([x + .25, -.45, 0])
            uu = vec(3, CT, .1).move_to([x + .55, -.45, 0])
            ax = Axes(x_range=[0, 15], y_range=[-2.2, 2.2], x_length=1.2, y_length=.9, tips=False, axis_config={"stroke_width": .8, "color": GREY_D}).move_to([x, 2.6, 0])
            ch = Z["chunk"][t]
            cur = VGroup(*[ax.plot_line_graph(np.arange(16), ch[:, j], add_vertex_dots=False, line_color=JC[j], stroke_width=2).set_fill(opacity=0) for j in range(6)])
            hd = box(1.0, .35, "#3d3415", CQ).move_to([x, 1.75, 0])
            phb = Rectangle(width=1.0, height=.08, fill_color=PH[int(Z["phase"][t])], fill_opacity=1, stroke_width=0).next_to(cells[k], DOWN, buff=.05)
            anims = [FadeIn(f_), FadeIn(vit), GrowArrow(A(f_.get_top(), vit.get_bottom(), CV, 2)), FadeIn(pool), GrowArrow(A(vit.get_top(), pool.get_bottom(), CQ, 2)),
                     FadeIn(v), FadeIn(jt), FadeIn(uu), GrowArrow(A(VGroup(v, jt, uu).get_top(), phb.get_bottom(), GREY_B, 2)),
                     FadeIn(hd), GrowArrow(A(cells[k].get_top(), hd.get_bottom(), CQ, 2)), FadeIn(ax), Create(cur), GrowArrow(A(hd.get_top(), ax.get_bottom(), CQ, 2)), FadeIn(phb)]
            if k:
                anims += [FadeIn(cells[k]), GrowArrow(A(cells[k - 1].get_right(), cells[k].get_left(), CL, 4))]
            self.play(*anims, run_time=1.1 if k == 0 else .8)
            self.add(tag("LSTM", 12, CL).move_to(cells[k]), tag("flow", 11, CQ).move_to(hd))
            if k == 0:
                self.play(FadeIn(tag("ViT-Tiny", 11, CV).next_to(vit, LEFT, buff=.08)), FadeIn(tag("pool ×4", 11, CQ).next_to(pool, LEFT, buff=.08)), FadeIn(tag("16×6", 11).next_to(ax, UP, buff=.04)), run_time=.5)
        leg = VGroup(*[VGroup(Rectangle(width=.3, height=.08, fill_color=c, fill_opacity=1, stroke_width=0), tag(n, 12)).arrange(RIGHT, buff=.08)
                       for c, n in zip(PH, ("approach", "holding", "released"))]).arrange(RIGHT, buff=.3).to_corner(DL, buff=.25)
        self.play(FadeIn(leg), FadeIn(tag("10 Hz · each frame encoded once", 13).to_corner(DR, buff=.25)))
        dot = Dot(color=WHITE, radius=.09).move_to(init.get_center())
        self.play(MoveAlongPath(dot, VMobject().set_points_as_corners([init.get_center()] + [c.get_center() for c in cells])), run_time=2, rate_func=linear)
        self.wait(.8); self.clear()

    # ------------------------------------------------------------ 2 text-conditioned pooling: real attention of the 4 queries
    def pooling(self):
        t = int(np.argmax(Z["phase"] == 0) + 8)
        fr = Z["frames"][t]; at = Z["att"][t]
        base = img(fr, 2.4).move_to([-4.6, .5, 0])
        self.play(FadeIn(base))
        ims = Group(*[img(overlay(fr, at[q]), 1.55) for q in range(4)]).arrange_in_grid(2, 2, buff=.2).move_to([.2, .5, 0])
        qd = VGroup(*[Circle(.1, fill_color=CQ, fill_opacity=1, stroke_width=0).next_to(ims[q], UP, buff=.06) for q in range(4)])
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=.2), FadeIn(qd), GrowArrow(A(base.get_right(), ims.get_left(), CV)), run_time=1.4)
        uvec = vec(6, CT, .14).move_to([.2, 3.3, 0]).rotate(PI / 2)
        self.play(FadeIn(uvec), GrowArrow(A(uvec.get_bottom(), ims.get_top() + UP * .25, CT)))
        out = vec(8, CV, .16).move_to([4.6, .5, 0])
        self.play(GrowArrow(A(ims.get_right(), out.get_left(), CV)), FadeIn(out, lag_ratio=.1))
        self.play(FadeIn(tag("4 queries · u-conditioned → v (512)", 13).next_to(out, DOWN, buff=.25)), FadeIn(tag("real attention · step 2500", 12).to_corner(DR, buff=.25)))
        self.wait(1.4); self.clear()

    # ------------------------------------------------------------ 3 LSTM memory: real hidden-state path (PCA), coloured by phase
    def memory(self):
        pc = Z["pc"]; ph = Z["phase"]; fr = Z["frames"]
        lo, hi = pc.min(0) - .5, pc.max(0) + .5
        ax = Axes(x_range=[lo[0], hi[0]], y_range=[lo[1], hi[1]], x_length=6.5, y_length=5.5, tips=False, axis_config={"stroke_width": 0}).move_to([1.8, 0, 0])
        frame = Rectangle(width=6.5, height=5.5, stroke_color=GREY_D, stroke_width=1).move_to(ax)
        self.play(FadeIn(frame))
        segs = VGroup(*[Line(ax.c2p(*pc[i]), ax.c2p(*pc[i + 1]), color=PH[int(ph[i + 1])], stroke_width=3) for i in range(len(pc) - 1)])
        cam = img(fr[0], 2.2).move_to([-4.6, .8, 0]); self.add(cam)
        dot = Dot(ax.c2p(*pc[0]), color=WHITE, radius=.09); self.add(dot)
        n = len(pc); step = 4
        for i in range(0, n - 1, step):
            j = min(i + step, n - 1)
            new = img(fr[j], 2.2).move_to(cam)
            self.play(Create(segs[i:j]), dot.animate.move_to(ax.c2p(*pc[j])), FadeTransform(cam, new), run_time=.12, rate_func=linear)
            cam = new
        leg = VGroup(*[VGroup(Line(ORIGIN, RIGHT * .4, color=c, stroke_width=4), tag(n_, 13)).arrange(RIGHT, buff=.1) for c, n_ in zip(PH, ("approach", "holding", "released"))]).arrange(DOWN, aligned_edge=LEFT, buff=.15).move_to([-4.6, -2.0, 0])
        self.play(FadeIn(leg), FadeIn(tag(f"h (512) → PCA 2-D · {J['var2'] * 100:.0f}% var", 12).next_to(frame, DOWN, buff=.12)))
        self.wait(1.4); self.clear()

    # ------------------------------------------------------------ 4 flow head: the real 5-step path at one tick, then chunk vs label
    def flow(self):
        t = int(np.nonzero(Z["phase"] == 1)[0][5])
        F_ = Z["flow"][t]; ch, lb, st = Z["chunk"][t], Z["label"][t], Z["state"][t]
        ax = Axes(x_range=[0, 15], y_range=[-3, 3], x_length=5.6, y_length=4.4, tips=False, axis_config={"stroke_width": 1.2, "color": GREY_B}).move_to([-3.3, 0, 0])
        def curves(x):
            return VGroup(*[ax.plot_line_graph(np.arange(16), np.clip(x[:, j], -3, 3), add_vertex_dots=True, line_color=JC[j], stroke_width=2.5, vertex_dot_radius=.03,
                                               vertex_dot_style={"fill_color": JC[j]}).set_fill(opacity=0) for j in range(6)])
        self.play(FadeIn(ax), FadeIn(tag("normalised Δ · start x₀ = 0", 12).next_to(ax, UP, buff=.1)))
        cur = curves(F_[0]); self.play(FadeIn(cur))
        ks = VGroup()
        for k in range(1, 6):
            self.play(Transform(cur, curves(F_[k])), run_time=.5)
            ks.add(Dot(color=CQ, radius=.06))
        ks.arrange(RIGHT, buff=.15).next_to(ax, DOWN, buff=.2); self.play(FadeIn(ks, lag_ratio=.2))
        lo = float(min(ch.min(), lb.min(), st.min())) - .1; hi = float(max(ch.max(), lb.max(), st.max())) + .1
        ax2 = Axes(x_range=[-1, 15], y_range=[lo, hi], x_length=5.6, y_length=4.4, tips=False, axis_config={"stroke_width": 1.2, "color": GREY_B}).move_to([3.4, 0, 0])
        self.play(FadeIn(ax2), GrowArrow(A(ax.get_right(), ax2.get_left(), CQ)))
        g = VGroup(*[ax2.plot_line_graph(np.arange(16), lb[:, j], add_vertex_dots=False, line_color=JC[j], stroke_width=8).set_fill(opacity=0).set_stroke(opacity=.3) for j in range(6)])
        p = VGroup(*[ax2.plot_line_graph(np.r_[-.5, np.arange(16)], np.r_[st[j], ch[:, j]], add_vertex_dots=False, line_color=JC[j], stroke_width=2.5).set_fill(opacity=0) for j in range(6)])
        self.play(Create(g), run_time=.8); self.play(Create(p), run_time=1.2)
        leg = VGroup(VGroup(Line(ORIGIN, RIGHT * .5, stroke_width=8, color=WHITE).set_stroke(opacity=.3), tag("label", 13)).arrange(RIGHT, buff=.1),
                     VGroup(Line(ORIGIN, RIGHT * .5, stroke_width=2.5, color=WHITE), tag("pred", 13)).arrange(RIGHT, buff=.1)).arrange(RIGHT, buff=.4).next_to(ax2, UP, buff=.1)
        self.play(FadeIn(leg), FadeIn(tag("rad · 1.6 s · holding phase · train episode", 12).next_to(ax2, DOWN, buff=.2)))
        self.wait(1.4); self.clear()

    # ------------------------------------------------------------ 5 compute per tick (FlopCounterMode, batch 1)
    def speed(self):
        parts = [("ViT-Tiny + pool", .459, CV), ("flow ×5", .135, CQ), ("LSTM", .005, CL), ("text (1×/ep)", .076, CT)]
        x0, y0 = -4.5, -2.4; bars = VGroup()
        for i, (n_, v, c) in enumerate(parts):
            h = 4.6 * v / .5
            b = Rectangle(width=.9, height=max(h, .03), fill_color=c, fill_opacity=1 if i < 3 else .45, stroke_width=0).move_to([x0 + i * 1.4, y0 + max(h, .03) / 2, 0])
            bars.add(VGroup(b, tag(f"{v:.3f}", 13, WHITE).next_to(b, UP, buff=.06), tag(n_, 12, c).next_to(b, DOWN, buff=.12)))
        axis = Line([x0 - .7, y0, 0], [x0 + 3 * 1.4 + .7, y0, 0], color=GREY_B, stroke_width=2)
        self.play(Create(axis), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=.2), FadeIn(tag("GMAC / tick", 14).move_to([x0 + 2.1, 3.2, 0])), run_time=1.5)
        # timeline: 100 ms (10 Hz) and 33 ms (30 Hz) budgets vs measured CPU fp32 25 ms
        tl = NumberLine(x_range=[0, 100, 10], length=5.2, include_numbers=False, color=GREY_B).move_to([3.6, 0, 0])
        mk = lambda ms, c, lab, y: VGroup(Line(tl.n2p(ms) + DOWN * .15, tl.n2p(ms) + UP * .6, color=c, stroke_width=3), tag(lab, 12, c).next_to(tl.n2p(ms) + UP * .6, UP, buff=.05).shift(UP * y))
        cpu = Rectangle(width=tl.n2p(25)[0] - tl.n2p(0)[0], height=.25, fill_color=GREEN, fill_opacity=.8, stroke_width=0).next_to(tl.n2p(0), UP, buff=.05, aligned_edge=LEFT)
        self.play(Create(tl), FadeIn(tag("ms", 12).next_to(tl, RIGHT, buff=.1)))
        self.play(FadeIn(mk(100, GREY_A, "10 Hz", 0)), FadeIn(mk(33.3, YELLOW, "30 Hz", 0)), GrowFromEdge(cpu, LEFT), FadeIn(tag("CPU fp32 25 ms", 12, GREEN).next_to(cpu, DOWN, buff=.25)), run_time=1.2)
        self.play(FadeIn(tag("desktop CPU measured · Jetson not measured", 12).to_corner(DR, buff=.25)))
        self.wait(1.6); self.clear()
