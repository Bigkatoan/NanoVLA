"""The user's block diagram, animated tick by tick (same layout). Real wrist frames + real label chunk from rec.npz.
Render: manim -qm --media_dir manim_media diagram_manim.py UserDiagram"""
import os
import numpy as np
from manim import *

Hd = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f"{Hd}/rec.npz")
BG = "#141414"
CV, CT, CS, CL, CQ = "#4cc9f0", "#b388ff", "#80ed99", "#ff8a3d", "#ffd166"
JC = ["#4cc9f0", "#80ed99", "#ffd166", "#ff8a3d", "#b388ff", "#f72585"]


def P(x, y):                       # user's drawing pixels -> scene units
    return np.array([(x - 560) / 80, (330 - y) / 80, 0])


def tag(s, fs=15, col=WHITE):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def blk(x, y, name, w=100, h=100, round_=False, col=WHITE, fs=15):
    r = (RoundedRectangle(corner_radius=.15, width=w / 80, height=h / 80) if round_ else Rectangle(width=w / 80, height=h / 80))
    r.set_stroke(col, 2).set_fill("#111111", 1).move_to(P(x, y))
    t = tag(name, fs, col); t.set_width(min(t.width, w / 80 - .15)); t.move_to(r)
    return VGroup(r, t)


def ar(a, b, col=GREY_B, w=2.5):
    return Arrow(a, b, buff=0, stroke_width=w, color=col, max_tip_length_to_length_ratio=.08, tip_length=.16)


class UserDiagram(Scene):
    def construct(self):
        self.camera.background_color = BG
        B = {}
        B["img"] = blk(105, 272, "Image")
        B["enc"] = blk(280, 272, "Image Encoder", 150, 75, True, CV)
        for k, x in enumerate((455, 612, 768)):
            B[f"iv{k}"] = blk(x, 272, "Image Vector", col=CV)
            B[f"js{k}"] = blk(x, 122, "JointStates", col=CS)
        B["init"] = blk(455, 422, "Init Inform", col=CT)
        B["ob0"] = blk(612, 422, "Observation\nVector", col=CL)
        B["ob1"] = blk(768, 422, "Observation\nVector", col=CL)
        B["obf"] = blk(1005, 272, "Observation Vector", 150, 75, col=CL)
        B["mlp"] = blk(1005, 422, "MLP", 150, 75, True, CQ)
        B["act"] = blk(1005, 572, "Action", 150, 75)
        B["pr"] = blk(280, 572, "Prompt")
        B["te"] = blk(455, 572, "Text Encoder", 150, 75, True, CT)
        B["tv"] = blk(630, 572, "Text Vector", col=CT)
        cyl = VGroup(Ellipse(width=75 / 80, height=.3).set_stroke(WHITE, 2).set_fill("#111111", 1).move_to(P(768, 545)),
                     Rectangle(width=75 / 80, height=.9).set_stroke(width=0).set_fill("#111111", 1).move_to(P(768, 580)),
                     Line(P(730, 545), P(730, 615), color=WHITE, stroke_width=2), Line(P(806, 545), P(806, 615), color=WHITE, stroke_width=2),
                     Ellipse(width=75 / 80, height=.3).set_stroke(WHITE, 2).move_to(P(768, 615)).set_fill(opacity=0))
        B["cache"] = VGroup(cyl, tag("Cache", 15).move_to(P(768, 588)))
        e = lambda n, side: B[n][0].get_edge_center(side)
        W = {
            "img_enc": ar(e("img", RIGHT), e("enc", LEFT)), "enc_iv0": ar(e("enc", RIGHT), e("iv0", LEFT), CV),
            "pr_te": ar(e("pr", RIGHT), e("te", LEFT), CT), "te_tv": ar(e("te", RIGHT), e("tv", LEFT), CT), "tv_c": ar(e("tv", RIGHT), B["cache"][0][0].get_left() + DOWN * .45, CT),
            "te_init": ar(e("te", UP), e("init", DOWN), CT),
            "init_ob0": ar(e("init", RIGHT), B["ob0"][0].get_corner(DL) + UP * .35, CT),
            "iv0_ob0": ar(B["iv0"][0].get_corner(DR), B["ob0"][0].get_corner(UL) + RIGHT * .1, CV),
            "js0_ob0": ar(e("js0", DOWN) + RIGHT * .35, B["ob0"][0].get_top() + LEFT * .1, CS),
            "tv_ob0": ar(e("tv", UP) + LEFT * .1, e("ob0", DOWN), CT),
            "ob0_ob1": ar(B["ob0"][0].get_right() + DOWN * .1, B["ob1"][0].get_corner(DL) + UP * .35, CL),
            "iv1_ob1": ar(B["iv1"][0].get_corner(DR), B["ob1"][0].get_corner(UL) + RIGHT * .1, CV),
            "js1_ob1": ar(e("js1", DOWN) + RIGHT * .35, B["ob1"][0].get_top() + LEFT * .1, CS),
            "tv_ob1": ar(e("tv", UP) + RIGHT * .2, e("ob1", DOWN) + LEFT * .1, CT),
            "ob1_obf": ar(e("ob1", RIGHT), B["obf"][0].get_corner(DL) + RIGHT * .1, CL),
            "iv2_obf": ar(e("iv2", RIGHT), e("obf", LEFT) + DOWN * .1, CV),
            "js2_obf": ar(B["js2"][0].get_corner(DR), B["obf"][0].get_corner(UL) + RIGHT * .2, CS),
            "tv_obf": ar(e("tv", UP) + RIGHT * .35, B["obf"][0].get_bottom() + LEFT * .5, CT),
            "obf_mlp": ar(e("obf", DOWN), e("mlp", UP), CL), "mlp_act": ar(e("mlp", DOWN), e("act", UP), CQ),
        }
        # ---- 1 build the diagram (as drawn)
        grid = NumberPlane(x_range=[-8, 8, .5], y_range=[-5, 5, .5], background_line_style={"stroke_color": "#2a2a2a", "stroke_width": 1}, axis_config={"stroke_opacity": 0})
        self.add(grid)
        self.play(LaggedStart(*[FadeIn(b) for b in B.values()], lag_ratio=.05), run_time=2)
        self.play(LaggedStart(*[GrowArrow(w) for w in W.values()], lag_ratio=.04), run_time=2)
        self.wait(.6)
        # ---- 2 episode start: prompt -> text vector -> cache + Init Inform (once)
        def pulse(keys, col, rt=.8):
            ds = [Dot(W[k].get_start(), color=col, radius=.08) for k in keys]
            self.play(*[MoveAlongPath(d, Line(W[k].get_start(), W[k].get_end())) for d, k in zip(ds, keys)], run_time=rt)
            self.remove(*ds)
        self.play(Indicate(B["pr"], color=CT))
        pulse(["pr_te"], CT); pulse(["te_tv", "te_init"], CT); pulse(["tv_c"], CT, .5)
        self.play(FadeIn(tag("1× / episode", 13, CT).next_to(B["cache"], UP, buff=.08)))
        # ---- 3 three ticks with real frames; the image is encoded once per tick
        fr = Z["frames"]; ti = np.linspace(10, len(fr) - 20, 3).astype(int)
        cur = None
        for k in range(3):
            im = ImageMobject(fr[ti[k]]); im.set_height(.95); im.move_to(B["img"])
            self.play(FadeIn(im) if cur is None else FadeTransform(cur, im), run_time=.4); cur = im
            pulse(["img_enc"], WHITE, .4)
            # encoder output lands in this tick's Image Vector (drawn arrow only exists to the first one)
            d = Dot(e("enc", RIGHT), color=CV, radius=.08)
            self.play(MoveAlongPath(d, Line(e("enc", RIGHT), B[f"iv{k}"][0].get_center())), run_time=.5); self.remove(d)
            self.play(B[f"iv{k}"][0].animate.set_fill(CV, .35), B[f"js{k}"][0].animate.set_fill(CS, .35), run_time=.3)
            keys = [["init_ob0", "iv0_ob0", "js0_ob0", "tv_ob0"], ["ob0_ob1", "iv1_ob1", "js1_ob1", "tv_ob1"], ["ob1_obf", "iv2_obf", "js2_obf", "tv_obf"]][k]
            pulse(keys, CL, .8)
            tgt = [B["ob0"], B["ob1"], B["obf"]][k]
            self.play(tgt[0].animate.set_fill(CL, .35), run_time=.3)
        # ---- 4 observation vector -> MLP (iterated: flow, 5 steps) -> action chunk
        pulse(["obf_mlp"], CL, .5)
        loop = CurvedArrow(B["mlp"][0].get_right() + UP * .2, B["mlp"][0].get_right() + DOWN * .2, angle=-PI * 1.2, color=CQ, stroke_width=3)
        plus = VGroup(Circle(.14, color=GREEN, stroke_width=2, fill_color=BG, fill_opacity=1), tag("+", 16, GREEN)).next_to(loop, DOWN, buff=.1)
        self.play(Create(loop), FadeIn(plus), FadeIn(tag("×5", 14, CQ).next_to(loop, UP, buff=.02)))
        for _ in range(2):
            self.play(Indicate(B["mlp"], color=CQ, scale_factor=1.05), run_time=.35)
        pulse(["mlp_act"], CQ, .5)
        ax = Axes(x_range=[0, 15], y_range=[-2.5, 2.5], x_length=1.25, y_length=1.0, tips=False, axis_config={"stroke_width": 1, "color": GREY_D}).move_to([3.85, -3.0, 0])
        lb = Z["label"][ti[2]]
        cv = VGroup(*[ax.plot_line_graph(np.arange(16), lb[:, j], add_vertex_dots=False, line_color=JC[j], stroke_width=2).set_fill(opacity=0) for j in range(6)])
        self.play(FadeIn(ax), Create(cv), FadeIn(tag("16×6 · 1.6 s", 11, GREY_B).next_to(ax, DOWN, buff=.04)))
        # ---- 5 what is inside the boxes (tags)
        notes = VGroup(tag("ViT-Tiny + pool · 0.46 GMAC", 12, CV).next_to(B["enc"], DOWN, buff=.1),
                       tag("LSTM 2×512", 12, CL).next_to(B["ob0"], UP, buff=.06).shift(LEFT * .9),
                       tag("3.1M", 12, CQ).next_to(B["mlp"], LEFT, buff=.1),
                       VGroup(Circle(.12, color=GREEN, stroke_width=2), tag("+", 14, GREEN), tag("MLP = flow velocity, 5 steps", 12, GREEN)).arrange(RIGHT, buff=.08).to_corner(DL, buff=.25))
        notes[3][1].move_to(notes[3][0])
        self.play(FadeIn(notes, lag_ratio=.2))
        self.play(FadeIn(tag("action chunk shown = demo label", 11, GREY_B).to_corner(UR, buff=.2)))
        self.wait(2.5)
