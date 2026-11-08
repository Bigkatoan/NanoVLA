"""NanoVLA-M (L1 head) closed-loop results vs v1. Real rollout frames (eval A, env 0), real success counts, real offline val.
Render: manim -qm --media_dir manim_media resM_manim.py ResultsM"""
import json, os
import numpy as np
from manim import *

Hd = os.path.dirname(os.path.abspath(__file__))
RO = np.load(f"{Hd}/rollM.npz"); RJ = json.load(open(f"{Hd}/resM.json"))
BG = "#0f1115"
C1, CM = "#9aa1b2", "#ff8a3d"
TEXTS = ["put the green block in the tray", "put the eraser in the tray", "drop the triangle block into the box"]


def tag(s, fs=16, col=GREY_B):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def img(a, h):
    m = ImageMobject(np.asarray(a, dtype=np.uint8)); m.set_height(h); return m


class ResultsM(Scene):
    def construct(self):
        self.camera.background_color = BG
        # ---------------- 1 what the robot did: three rollouts (env 0), 8 frames over 25 s each
        rows = Group()
        for k in range(3):
            strip = Group(*[img(RO[f"arr_{k}"][i], .95) for i in range(8)]).arrange(RIGHT, buff=.06)
            lab = tag(TEXTS[k], 13, WHITE)
            x = VGroup(Line(UL * .13, DR * .13, color=RED, stroke_width=5), Line(UR * .13, DL * .13, color=RED, stroke_width=5))
            rows.add(Group(VGroup(lab, x).arrange(RIGHT, buff=.15), strip).arrange(DOWN, aligned_edge=LEFT, buff=.08))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=.3).move_to(DOWN * .1)
        tl = VGroup(Arrow(LEFT * 3.5, RIGHT * 3.5, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=.02), tag("0 → 25 s", 12)).arrange(RIGHT, buff=.1).next_to(rows, DOWN, buff=.15)
        for r in rows:
            self.play(FadeIn(r[0]), LaggedStart(*[FadeIn(f) for f in r[1]], lag_ratio=.12), run_time=1.3)
        self.play(FadeIn(tl))
        # mark the drift: last frames of episode 3 see the room, not the table
        box = SurroundingRectangle(Group(*rows[2][1][4:]), color=RED, buff=.04, stroke_width=3)
        self.play(Create(box), FadeIn(tag("tay trôi lên, camera chỉ thấy tường / trần", 13, RED).next_to(box, UP, buff=.05).align_to(box, RIGHT)))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)

        # ---------------- 2 success per task: v1 vs NanoVLA-M (same seed, same tasks)
        T = list(RJ["v1"]); x0, y0, Hh = -5.6, -2.4, 4.2
        axis = Line([x0 - .5, y0, 0], [x0 + len(T) * 1.6 - .6, y0, 0], color=GREY_B, stroke_width=2)
        g = VGroup()
        for i, t in enumerate(T):
            for j, (key, col) in enumerate((("v1", C1), ("m", CM))):
                s, n = RJ[key][t]; h = max(.03, Hh * s / max(n, 1)); x = x0 + i * 1.6 + j * .5
                ghost = Rectangle(width=.42, height=Hh, stroke_color=col, stroke_width=1, fill_opacity=0).move_to([x, y0 + Hh / 2, 0])
                b = Rectangle(width=.42, height=h, fill_color=col, fill_opacity=1, stroke_width=0).move_to([x, y0 + h / 2, 0])
                g.add(VGroup(ghost, b, tag(f"{s}/{n}", 12, WHITE).next_to(ghost, UP, buff=.04)))
            g.add(tag(t, 13).move_to([x0 + i * 1.6 + .25, y0 - .3, 0]))
        leg = VGroup(VGroup(Square(.2, fill_color=C1, fill_opacity=1, stroke_width=0), tag("v1  6/48", 14, WHITE)).arrange(RIGHT, buff=.1),
                     VGroup(Square(.2, fill_color=CM, fill_opacity=1, stroke_width=0), tag("NanoVLA-M (L1)  0/48", 14, WHITE)).arrange(RIGHT, buff=.1)).arrange(RIGHT, buff=.6).to_edge(UP, buff=.4)
        self.play(Create(axis), FadeIn(leg), LaggedStart(*[FadeIn(m) for m in g], lag_ratio=.05), run_time=2)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)

        # ---------------- 3 the paradox: better offline, worse closed loop
        ax = Axes(x_range=[0.2, 0.65, .1], y_range=[0, 0.2, .05], x_length=7, y_length=4.5, tips=False,
                  axis_config={"stroke_width": 1.5, "color": GREY_B}).move_to(DOWN * .2)
        xl = tag("lỗi offline (÷ mức đứng yên) — thấp hơn = tốt hơn", 13).next_to(ax, DOWN, buff=.15)
        yl = tag("thành công vòng kín (A)", 13).rotate(PI / 2).next_to(ax, LEFT, buff=.15)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        pts = [(0.577, 6 / 48, C1, "v1"), (0.27, 0.0, CM, "M-L1")]
        for x, y, c, n in pts:
            d = Dot(ax.c2p(x, y), color=c, radius=.13)
            self.play(GrowFromCenter(d), FadeIn(tag(n, 14, c).next_to(d, UP, buff=.12)), run_time=.6)
        hollow = Circle(.13, color="#80ed99", stroke_width=3).move_to(ax.c2p(0.359, 0.1))
        q = tag("R (flow) · chưa eval", 12, "#80ed99").next_to(hollow, UP, buff=.1)
        self.play(Create(hollow), FadeIn(q))
        self.play(Create(DashedLine(hollow.get_center() + DOWN * .13, ax.c2p(0.359, 0), color="#80ed99", stroke_width=2)))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=.6)

        # ---------------- 4 why (schematic): compounding drift off the demo distribution
        ax2 = Axes(x_range=[0, 10], y_range=[-3, 3], x_length=10, y_length=5, tips=False, axis_config={"stroke_width": 0})
        xs = np.linspace(0, 10, 60); base = np.sin(xs * .6) * .8
        band = Polygon(*[ax2.c2p(x, y + .5) for x, y in zip(xs, base)], *[ax2.c2p(x, y - .5) for x, y in zip(xs[::-1], base[::-1])],
                       fill_color=GREEN, fill_opacity=.18, stroke_width=0)
        demo = ax2.plot_line_graph(xs, base, add_vertex_dots=False, line_color=GREEN, stroke_width=3).set_fill(opacity=0)
        roll = base + np.r_[np.zeros(15), np.cumsum(np.linspace(.0, .12, 45))]
        rl = ax2.plot_line_graph(xs, roll, add_vertex_dots=False, line_color=RED, stroke_width=4).set_fill(opacity=0)
        self.play(FadeIn(band), Create(demo), FadeIn(tag("demo (dữ liệu train)", 14, GREEN).move_to(ax2.c2p(1.6, -1.6))))
        self.play(Create(rl), run_time=2.5)
        self.play(FadeIn(tag("rollout: lệch nhỏ → trạng thái lạ → lệch lớn hơn", 14, RED).move_to(ax2.c2p(6.8, 2.7))))
        self.play(FadeIn(tag("minh hoạ", 12).to_corner(DR, buff=.25)))
        self.wait(2)
