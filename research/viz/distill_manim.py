"""Manim (picture-first): NanoVLA teacher -> few-shot -> distill pipeline (plan, before training).
Render:  manim -qm --media_dir manim_media distill_manim.py DistillPipeline"""
import numpy as np
from manim import *
from PIL import Image

IMG = "/tmp/claude-1000/-home-bigkatoan/95ac64cd-e522-5e0a-b638-e3a5259fcf3a/scratchpad/hr_smoke.png"   # real 320x240 wrist frame (new data)
BG = "#101413"
TEA, STU, ADP, GR, ST, GS, PL = "#4cc9f0", "#80ed99", "#ff9f1c", "#f72585", "#b388ff", "#ffd166", "#06d6a0"
FROZ = "#5c6670"


def tag(s, fs=18, col=GREY_B):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def box(w, h, col, op=0.25, sw=2):
    return Rectangle(width=w, height=h, stroke_color=col, stroke_width=sw, fill_color=col, fill_opacity=op)


def heads(scale=1.0):
    """four module boxes with glyphs: grounder (crosshair), state (step bars), grasp (rotation), planner (path)"""
    out = VGroup()
    for col, kind in ((GR, "g"), (ST, "s"), (GS, "r"), (PL, "p")):
        b = box(0.9 * scale, 0.62 * scale, col, 0.18)
        if kind == "g":
            gl = VGroup(Circle(0.17 * scale, color=col, stroke_width=2), Line(LEFT * 0.25 * scale, RIGHT * 0.25 * scale, color=col, stroke_width=2),
                        Line(UP * 0.25 * scale, DOWN * 0.25 * scale, color=col, stroke_width=2))
        elif kind == "s":
            gl = VGroup(*[Rectangle(width=0.09 * scale, height=(0.1 + 0.08 * i) * scale, stroke_width=0, fill_color=col, fill_opacity=0.9) for i in range(4)]).arrange(RIGHT, buff=0.04 * scale, aligned_edge=DOWN)
        elif kind == "r":
            gl = Arc(radius=0.18 * scale, start_angle=0.3, angle=4.6, color=col, stroke_width=3).add_tip(tip_length=0.1 * scale)
        else:
            gl = VMobject(color=col, stroke_width=3).set_points_smoothly([np.array([-0.3, -0.12, 0]) * scale, np.array([-0.05, 0.15, 0]) * scale, np.array([0.3, 0.05, 0]) * scale])
        gl.move_to(b); out.add(VGroup(b, gl))
    return out.arrange(DOWN, buff=0.12 * scale)


def arm(col=GREY_B, s=1.0):
    p0, p1, p2, p3 = np.array([0, 0, 0]), np.array([0.15, 0.55, 0]), np.array([0.6, 0.75, 0]), np.array([0.85, 0.45, 0])
    seg = VGroup(*[Line(a * s, b * s, color=col, stroke_width=6) for a, b in ((p0, p1), (p1, p2), (p2, p3))])
    jt = VGroup(*[Dot(p * s, radius=0.06 * s, color=WHITE) for p in (p0, p1, p2)])
    base = Rectangle(width=0.5 * s, height=0.12 * s, stroke_width=0, fill_color=col, fill_opacity=1).move_to(p0 * s + DOWN * 0.06 * s)
    jaw = VGroup(Line(p3 * s, (p3 + np.array([0.05, -0.2, 0])) * s, color=col, stroke_width=4), Line(p3 * s, (p3 + np.array([0.18, -0.12, 0])) * s, color=col, stroke_width=4))
    return VGroup(base, seg, jt, jaw)


def chip(col, label, w=1.3):
    b = RoundedRectangle(corner_radius=0.1, width=w, height=w * 0.7, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.15)
    pins = VGroup(*[Line(ORIGIN, DOWN * 0.12, color=col, stroke_width=2).move_to(b.get_bottom() + RIGHT * (i - 2) * w * 0.18 + DOWN * 0.06) for i in range(5)],
                  *[Line(ORIGIN, UP * 0.12, color=col, stroke_width=2).move_to(b.get_top() + RIGHT * (i - 2) * w * 0.18 + UP * 0.06) for i in range(5)])
    return VGroup(b, pins, tag(label, 16, col).move_to(b))


def heatmap(img_mob, cx, cy, n=10, m=8):
    """illustrative soft-label heatmap over the frame (gaussian around the object cell)"""
    W, H = img_mob.width, img_mob.height; out = VGroup()
    for r in range(m):
        for c in range(n):
            v = np.exp(-((c - cx) ** 2 + (r - cy) ** 2) / 3.0)
            sq = Rectangle(width=W / n, height=H / m, stroke_width=0, fill_color=GR, fill_opacity=0.75 * v)
            sq.move_to(img_mob.get_corner(UL) + RIGHT * W * (c + 0.5) / n + DOWN * H * (r + 0.5) / m); out.add(sq)
    return out


class DistillPipeline(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ------------------------------------------------ 1. teacher: big backbone, all sim data
        data = VGroup(*[Square(0.11, stroke_width=0, fill_color=interpolate_color(ManimColor(TEA), ManimColor(PL), (i * 7 % 11) / 10), fill_opacity=0.85)
                        for i in range(150)]).arrange_in_grid(15, 10, buff=0.04).move_to(LEFT * 5.6)
        dtag = tag("1500 ep · sim", 16).next_to(data, DOWN, buff=0.2)
        img = ImageMobject(IMG).set_height(1.8).move_to(LEFT * 3.2 + UP * 0.2)
        itag = tag("320×240", 16).next_to(img, DOWN, buff=0.15)
        bb = box(1.3, 3.6, TEA, 0.3).move_to(LEFT * 0.9)
        btag = tag("450M", 18, TEA).next_to(bb, DOWN, buff=0.15)
        hd = heads(1.0).move_to(RIGHT * 1.3)
        servo = box(0.9, 0.9, GREY_B, 0.12).move_to(RIGHT * 3.2)
        sg = VGroup(*[Line(LEFT * 0.3, RIGHT * 0.3, color=GREY_B).shift(UP * 0.12 * k) for k in (-1, 0, 1)]).move_to(servo)
        stag = tag("IK", 16).next_to(servo, DOWN, buff=0.15)
        a1 = arm(TEA, 1.6).move_to(RIGHT * 5.4)
        title = tag("Thầy", 26, TEA).to_edge(UP, buff=0.35)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(d) for d in data], lag_ratio=0.01, run_time=1.5), FadeIn(dtag))
        self.play(FadeIn(img), FadeIn(itag), GrowFromEdge(bb, DOWN), FadeIn(btag))
        self.play(LaggedStart(*[FadeIn(h, shift=RIGHT * 0.2) for h in hd], lag_ratio=0.2), FadeIn(servo), FadeIn(sg), FadeIn(stag), FadeIn(a1))
        wires = VGroup(Arrow(data.get_right(), img.get_left(), buff=0.15, stroke_width=3, color=GREY_B),
                       Arrow(img.get_right(), bb.get_left(), buff=0.1, stroke_width=3, color=GREY_B),
                       *[Arrow(bb.get_right(), h.get_left(), buff=0.08, stroke_width=2, color=GREY_C, max_tip_length_to_length_ratio=0.12) for h in hd],
                       *[Arrow(h.get_right(), servo.get_left(), buff=0.08, stroke_width=2, color=GREY_C, max_tip_length_to_length_ratio=0.12) for h in hd],
                       Arrow(servo.get_right(), a1.get_left(), buff=0.1, stroke_width=3, color=GREY_B))
        self.play(Create(wires), run_time=1.2)
        for _ in range(2):
            dot = Dot(color=YELLOW, radius=0.09).move_to(img.get_left())
            path = VMobject().set_points_as_corners([img.get_left(), bb.get_center(), hd[0].get_center(), servo.get_center(), a1.get_center()])
            self.play(MoveAlongPath(dot, path), run_time=1.4, rate_func=linear); self.remove(dot)
        pc = chip(TEA, "PC", 1.0).next_to(a1, DOWN, buff=0.5)
        self.play(FadeIn(pc)); self.wait(0.6)
        self.play(FadeOut(data), FadeOut(dtag), FadeOut(wires), FadeOut(pc), FadeOut(itag))

        # ------------------------------------------------ 2. few-shot: frozen teacher + small adapters, 5 demos
        teacher = Group(img, bb, btag, hd, servo, sg, stag, a1)
        self.play(teacher.animate.shift(RIGHT * 1.2), title.animate.become(tag("Few-shot", 26, ADP).to_edge(UP, buff=0.35)))
        lock = VGroup(Arc(radius=0.18, start_angle=0, angle=PI, color=WHITE, stroke_width=4).shift(UP * 0.12),
                      RoundedRectangle(corner_radius=0.05, width=0.5, height=0.36, stroke_width=0, fill_color=WHITE, fill_opacity=0.9)).move_to(bb).shift(UP * 1.2)
        self.play(bb.animate.set_fill(FROZ, 0.35).set_stroke(FROZ), btag.animate.set_color(FROZ), FadeIn(lock))
        adapters = VGroup(*[Rectangle(width=0.14, height=0.5, stroke_width=0, fill_color=ADP, fill_opacity=1).next_to(hd[i], LEFT, buff=0.02) for i in (0, 1)],
                          Rectangle(width=0.14, height=0.5, stroke_width=0, fill_color=ADP, fill_opacity=1).next_to(bb, LEFT, buff=0.02).shift(DOWN * 1.2))
        objs = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.35, height=0.35, stroke_width=0, fill_color=c, fill_opacity=1) for c in ("#c0392b", "#8e7cc3", "#2a9d8f")]).arrange(DOWN, buff=0.25).move_to(LEFT * 5.6 + UP * 1.0)
        demos = VGroup(*[Dot(radius=0.12, color=ADP) for _ in range(5)]).arrange(RIGHT, buff=0.15).move_to(LEFT * 5.3 + DOWN * 1.4)
        dmt = tag("5–20 demo", 16, ADP).next_to(demos, DOWN, buff=0.2)
        self.play(FadeIn(objs, shift=RIGHT * 0.3), FadeIn(demos), FadeIn(dmt))
        for d in demos:
            c = d.copy(); self.add(c)
            self.play(c.animate.move_to(adapters[0]).scale(0.4), run_time=0.35); self.remove(c)
        self.play(LaggedStart(*[GrowFromEdge(x, DOWN) for x in adapters], lag_ratio=0.3))
        ptag = tag("~80k", 16, ADP).next_to(adapters[0], UP, buff=0.15)
        self.play(FadeIn(ptag)); self.wait(0.8)
        self.play(FadeOut(objs), FadeOut(demos), FadeOut(dmt), FadeOut(ptag))

        # ------------------------------------------------ 3. distill: student acts, teacher labels its frames, student learns
        t2 = Group(teacher, lock, adapters)
        self.play(t2.animate.scale(0.7).move_to(LEFT * 2.5 + DOWN * 0.2), title.animate.become(tag("Distill", 26, STU).to_edge(UP, buff=0.35)))
        sbb = box(0.6, 1.6, STU, 0.3); shd = heads(0.55); stu = VGroup(sbb, shd).arrange(RIGHT, buff=0.15)
        sarm = arm(STU, 1.0).next_to(stu, RIGHT, buff=0.3)
        student = VGroup(stu, sarm).move_to(RIGHT * 3.8 + DOWN * 0.2)
        st_tag = tag("Trò", 22, STU).next_to(student, UP, buff=0.3)
        self.play(FadeIn(student), FadeIn(st_tag))
        top = ArcBetweenPoints(student.get_top() + UP * 0.6, t2.get_top() + UP * 0.25 + RIGHT * 0.6, angle=0.6, color=GREY_B, stroke_width=3).add_tip(tip_length=0.2)
        bot = ArcBetweenPoints(t2.get_bottom() + DOWN * 0.25 + RIGHT * 0.6, student.get_bottom() + DOWN * 0.3, angle=0.6, color=GREY_B, stroke_width=3).add_tip(tip_length=0.2)
        self.play(Create(top), Create(bot))
        for it in range(2):
            fr = ImageMobject(IMG).set_height(0.7).move_to(student.get_top() + UP * 0.6)
            self.play(Rotate(sarm, 0.25, about_point=sarm.get_bottom()), FadeIn(fr), run_time=0.5)
            self.play(MoveAlongPath(fr, top), run_time=1.0)
            big = ImageMobject(IMG).set_height(1.5).move_to(DOWN * 2.7 + LEFT * 0.6)
            hm = heatmap(big, 6 + it, 4)
            bars = VGroup(*[Rectangle(width=0.16, height=h, stroke_width=0, fill_color=ST, fill_opacity=0.9) for h in (0.15, 0.9, 0.3, 0.1)]).arrange(RIGHT, buff=0.05, aligned_edge=DOWN).next_to(big, RIGHT, buff=0.3)
            tgt = VGroup(Circle(0.12, color=PL, stroke_width=3), Dot(radius=0.04, color=PL)).next_to(bars, RIGHT, buff=0.3)
            self.play(FadeOut(fr), FadeIn(big), FadeIn(hm), GrowFromEdge(bars, DOWN), FadeIn(tgt), run_time=0.7)
            lab = Group(big, hm, bars, tgt)
            self.play(lab.animate.scale(0.4).move_to(student.get_bottom() + DOWN * 0.3), run_time=0.9)
            self.play(FadeOut(lab), sbb.animate.set_fill(STU, 0.3 + 0.3 * (it + 1)), Rotate(sarm, -0.25, about_point=sarm.get_bottom()), run_time=0.6)
        self.wait(0.5)

        # ------------------------------------------------ 4. deploy: student only on the Jetson; teacher stays on the PC
        self.play(FadeOut(top), FadeOut(bot), title.animate.become(tag("Deploy", 26, WHITE).to_edge(UP, buff=0.35)))
        jet = chip(STU, "Jetson", 1.6).move_to(RIGHT * 3.8 + DOWN * 2.6)
        pc = chip(TEA, "PC", 1.2).next_to(t2, DOWN, buff=0.35)
        self.play(FadeIn(jet), FadeIn(pc), student.animate.scale(0.9).next_to(jet, UP, buff=0.35), st_tag.animate.next_to(jet, UP, buff=2.2))
        ticks = VGroup(*[Line(UP * 0.12, DOWN * 0.12, color=STU, stroke_width=3) for _ in range(10)]).arrange(RIGHT, buff=0.12).next_to(jet, LEFT, buff=0.4)
        self.play(LaggedStart(*[FadeIn(t) for t in ticks], lag_ratio=0.1), run_time=1.0)
        hz = tag("10–30 Hz", 16, STU).next_to(ticks, DOWN, buff=0.15)
        self.play(FadeIn(hz)); self.wait(0.8)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ------------------------------------------------ 5. how it will be judged (no results yet)
        title = tag("Kiểm chứng", 26, WHITE).to_edge(UP, buff=0.35)
        objs = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.4, height=0.4, stroke_width=0, fill_color=c, fill_opacity=1) for c in ("#c0392b", "#8e7cc3", "#2a9d8f")]).arrange(RIGHT, buff=0.2).move_to(UP * 2.2)
        cols = (STU, TEA, STU); labs = ("trò\nfew-shot", "thầy\nfew-shot", "trò\ndistill")
        axis = Line(LEFT * 3.2, RIGHT * 3.2, color=GREY_B).move_to(DOWN * 1.8)
        bars = VGroup()
        for i, (c, l) in enumerate(zip(cols, labs)):
            b = Rectangle(width=1.1, height=2.6, stroke_color=c, stroke_width=2, fill_opacity=0).move_to(RIGHT * (i - 1) * 2.1 + DOWN * 0.5)
            if i == 2:
                b.set_fill(c, 0.12)
            q = tag("?", 34, c).move_to(b)
            t = tag(l, 15).next_to(axis, DOWN, buff=0.2).set_x(b.get_x())
            bars.add(VGroup(b, q, t))
        foot = tag("glue_stick · spool · toy_brick — chưa có kết quả", 14, GREY_C).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(title), FadeIn(objs), Create(axis))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in bars], lag_ratio=0.3), FadeIn(foot))
        self.wait(2)
