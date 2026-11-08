"""NanoVLA-M (agreed design): four sources -> one-pass MLP -> action; memory updated after the decision.
Real wrist frames, joint states and demo label chunks from rec.npz.
Render: manim -qm --media_dir manim_media memory_manim.py NanoVLAM"""
import os
import numpy as np
from manim import *

Hd = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f"{Hd}/rec.npz")
BG = "#141414"
CV, CT, CS, CL, CQ = "#4cc9f0", "#b388ff", "#80ed99", "#ff8a3d", "#ffd166"
JC = ["#4cc9f0", "#80ed99", "#ffd166", "#ff8a3d", "#b388ff", "#f72585"]


def tag(s, fs=15, col=WHITE):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def blk(c, name, w=1.6, h=.8, col=WHITE, round_=False, fs=15):
    r = RoundedRectangle(corner_radius=.15, width=w, height=h) if round_ else Rectangle(width=w, height=h)
    r.set_stroke(col, 2).set_fill("#111111", 1).move_to(c)
    t = tag(name, fs, col); t.set_width(min(t.width, w - .15)); t.move_to(r)
    return VGroup(r, t)


def ar(a, b, col=GREY_B, w=2.5):
    return Arrow(a, b, buff=0, stroke_width=w, color=col, max_tip_length_to_length_ratio=.1, tip_length=.16)


class NanoVLAM(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.add(NumberPlane(x_range=[-8, 8, .5], y_range=[-5, 5, .5], background_line_style={"stroke_color": "#252525", "stroke_width": 1}, axis_config={"stroke_opacity": 0}))
        # ---------------- layout
        B = dict(
            pr=blk([-5.9, -2.9, 0], "Prompt", 1.3), te=blk([-3.9, -2.9, 0], "Text Encoder", 1.8, .7, CT, True), tv=blk([-1.9, -2.9, 0], "Task u", 1.3, .8, CT),
            init=blk([-3.9, -1.3, 0], "Init", 1.3, .7, CT, True),
            img=blk([-5.9, 1.2, 0], "", 1.3, 1.0), ie=blk([-3.9, 1.2, 0], "Image Encoder", 1.8, .7, CV, True), iv=blk([-1.9, 1.2, 0], "Image vₖ", 1.3, .8, CV),
            js=blk([-1.9, 2.9, 0], "Joints sₖ", 1.3, .8, CS),
            mp=blk([-1.9, -1.3, 0], "Memory mₖ₋₁", 1.5, .8, CL),
            mlp=blk([1.3, 0.4, 0], "MLP", 1.6, .9, CQ, True), act=blk([4.4, 0.4, 0], "", 2.3, 1.5),
            gru=blk([1.3, -1.9, 0], "GRU", 1.6, .8, CL, True), mn=blk([4.4, -1.9, 0], "Memory mₖ", 1.5, .8, CL),
        )
        cyl = VGroup(Ellipse(width=.9, height=.28).set_stroke(WHITE, 2).set_fill("#111111", 1), ).move_to([-0.2, -3.4, 0])
        cache = VGroup(Rectangle(width=.9, height=.55).set_stroke(width=0).set_fill("#111111", 1).move_to([-0.2, -3.65, 0]),
                       Line([-0.65, -3.4, 0], [-0.65, -3.9, 0], color=WHITE, stroke_width=2), Line([0.25, -3.4, 0], [0.25, -3.9, 0], color=WHITE, stroke_width=2),
                       Ellipse(width=.9, height=.28).set_stroke(WHITE, 2).move_to([-0.2, -3.9, 0]), cyl, tag("Cache", 12).move_to([-0.2, -3.7, 0]))
        e = lambda n, s: B[n][0].get_edge_center(s)
        W = dict(
            pr_te=ar(e("pr", RIGHT), e("te", LEFT), CT), te_tv=ar(e("te", RIGHT), e("tv", LEFT), CT), tv_c=ar(e("tv", RIGHT) + DOWN * .2, cyl.get_left() + LEFT * .05, CT),
            te_init=ar(e("te", UP), e("init", DOWN), CT), init_mp=ar(e("init", RIGHT), e("mp", LEFT), CT),
            img_ie=ar(e("img", RIGHT), e("ie", LEFT)), ie_iv=ar(e("ie", RIGHT), e("iv", LEFT), CV),
            iv_mlp=ar(e("iv", RIGHT), B["mlp"][0].get_left() + UP * .3, CV), js_mlp=ar(e("js", RIGHT), B["mlp"][0].get_corner(UL) + RIGHT * .3, CS),
            tv_mlp=ar(e("tv", RIGHT) + UP * .2, B["mlp"][0].get_corner(DL) + RIGHT * .25, CT), mp_mlp=ar(e("mp", RIGHT), B["mlp"][0].get_left() + DOWN * .3, CL),
            mlp_act=ar(e("mlp", RIGHT), e("act", LEFT), CQ),
            act_gru=ar(B["act"][0].get_bottom() + LEFT * .6, e("gru", RIGHT) + UP * .2, CQ), mp_gru=ar(e("mp", RIGHT) + DOWN * .25, e("gru", LEFT), CL),
            iv_gru=ar(e("iv", RIGHT) + DOWN * .3, B["gru"][0].get_corner(UL) + RIGHT * .25, CV), js_gru=ar(e("js", RIGHT) + DOWN * .25, B["gru"][0].get_top() + LEFT * .05, CS),
            gru_mn=ar(e("gru", RIGHT), e("mn", LEFT), CL),
        )
        loop = VMobject(stroke_color=CL, stroke_width=2.5).set_points_as_corners([e("mn", DOWN), e("mn", DOWN) + DOWN * .7, [-1.9, -2.6, 0] + DOWN * 0 + RIGHT * 0])
        loop_path = VMobject(stroke_color=CL, stroke_width=2.5).set_points_as_corners([e("mn", RIGHT), e("mn", RIGHT) + RIGHT * .5, [4.95 + .5, -2.45, 0], [-1.1, -2.45, 0], [-1.1, -1.75, 0]])
        loop = DashedVMobject(loop_path.copy(), num_dashes=40)
        loop_tag = tag("k → k+1", 12, CL).move_to([2.2, -2.25, 0])
        # four-source legend
        src = VGroup(*[VGroup(Square(.18, fill_color=c, fill_opacity=1, stroke_width=0), tag(t, 13)).arrange(RIGHT, buff=.1)
                       for c, t in ((CV, "đang thấy gì"), (CS, "hiện tại ra sao"), (CT, "cần làm gì"), (CL, "đã làm gì"))]).arrange(DOWN, aligned_edge=LEFT, buff=.12).to_corner(UR, buff=.3)
        # ---------------- 1 build
        self.play(LaggedStart(*[FadeIn(b) for b in B.values()], lag_ratio=.05), FadeIn(cache), run_time=1.8)
        self.play(LaggedStart(*[GrowArrow(w) for w in W.values()], lag_ratio=.04), Create(loop), FadeIn(loop_tag), run_time=2)
        self.play(FadeIn(src, lag_ratio=.15))
        self.wait(.5)

        def pulse(keys, col, rt=.7):
            ds = [Dot(W[k].get_start(), color=col, radius=.08) for k in keys]
            self.play(*[MoveAlongPath(d, Line(W[k].get_start(), W[k].get_end())) for d, k in zip(ds, keys)], run_time=rt)
            self.remove(*ds)
        # ---------------- 2 episode start (once)
        self.play(Indicate(B["pr"], color=CT))
        pulse(["pr_te"], CT); pulse(["te_tv"], CT); pulse(["tv_c", "te_init"], CT); pulse(["init_mp"], CT)
        self.play(B["mp"][0].animate.set_fill(CL, .15), FadeIn(tag("1× / lệnh", 12, CT).next_to(B["te"], DOWN, buff=.08)))
        # ---------------- 3 three frames (real)
        fr, st, lb = Z["frames"], Z["state"], Z["label"]
        ax = Axes(x_range=[0, 7], y_range=[-2.5, 2.5], x_length=2.0, y_length=1.2, tips=False, axis_config={"stroke_width": 1, "color": GREY_D}).move_to(B["act"])
        self.add(ax, tag("Action 8×6 · 0.8 s", 12, CQ).next_to(B["act"], UP, buff=.06))
        cur_img, cur_ch, bars = None, None, None
        fill = .15
        for k, t in enumerate(np.linspace(15, len(fr) - 25, 3).astype(int)):
            im = ImageMobject(fr[t]); im.set_height(.92); im.move_to(B["img"])
            jb = VGroup(*[Rectangle(width=.1, height=.06 + .14 * abs(v), fill_color=JC[j], fill_opacity=1, stroke_width=0) for j, v in enumerate(st[t])]).arrange(RIGHT, buff=.03, aligned_edge=DOWN).next_to(B["js"], LEFT, buff=.15)
            self.play(FadeIn(im) if cur_img is None else FadeTransform(cur_img, im), FadeIn(jb) if bars is None else Transform(bars, jb), run_time=.4)
            cur_img = im; bars = bars or jb
            pulse(["img_ie"], WHITE, .35); pulse(["ie_iv"], CV, .35)
            pulse(["iv_mlp", "js_mlp", "tv_mlp", "mp_mlp"], CQ, .7)
            ch = lb[t][:24:3][:8]
            cv = VGroup(*[ax.plot_line_graph(np.arange(8), ch[:, j], add_vertex_dots=True, line_color=JC[j], stroke_width=2, vertex_dot_radius=.025,
                                             vertex_dot_style={"fill_color": JC[j]}).set_fill(opacity=0) for j in range(6)])
            pulse(["mlp_act"], CQ, .35)
            self.play(FadeIn(cv) if cur_ch is None else Transform(cur_ch, cv), run_time=.35); cur_ch = cur_ch or cv
            hl = Circle(.09, color=WHITE, stroke_width=3).move_to(ax.c2p(0, ch[0, 3]))
            self.play(Create(hl), run_time=.25)
            pulse(["act_gru", "mp_gru", "iv_gru", "js_gru"], CL, .6); self.remove(hl)
            pulse(["gru_mn"], CL, .35)
            fill = min(.6, fill + .15)
            dot = Dot(e("mn", RIGHT), color=CL, radius=.08)
            self.play(MoveAlongPath(dot, loop_path), run_time=.7); self.remove(dot)
            self.play(B["mp"][0].animate.set_fill(CL, fill), B["mn"][0].animate.set_fill(CL, fill), run_time=.25)
        foot = VGroup(tag("điểm đầu của chunk = hành động thực thi aₖ → vào memory", 12, CQ), tag("ViT-Tiny 0.46 GMAC · MLP + GRU < 0.01 · 13–16M tham số · chunk minh hoạ = nhãn demo", 11, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=.06).to_corner(UL, buff=.2)
        self.play(FadeIn(foot))
        self.wait(2.5)
