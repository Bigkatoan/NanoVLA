"""NanoVLA-S: state-based policy (no time-indexed actions). Each new frame -> current task state, target point in the
camera frame, jaw; a non-learned servo (IK) moves the gripper there at its own rate. Real wrist frames from rec.npz.
Render: manim -qm --media_dir manim_media state_manim.py NanoVLAS"""
import os
import numpy as np
from manim import *

Hd = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f"{Hd}/rec.npz")
BG = "#141414"
CV, CT, CS, CL, CQ, CG = "#4cc9f0", "#b388ff", "#80ed99", "#ff8a3d", "#ffd166", "#f72585"
STATES = ["tiếp cận", "hạ xuống", "đóng kẹp", "nhấc", "mang tới", "đặt", "nhả"]


def tag(s, fs=15, col=WHITE):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def blk(c, name, w=1.6, h=.75, col=WHITE, round_=True, fs=15):
    r = RoundedRectangle(corner_radius=.14, width=w, height=h) if round_ else Rectangle(width=w, height=h)
    r.set_stroke(col, 2).set_fill("#111111", 1).move_to(c)
    t = tag(name, fs, col); t.set_width(min(t.width, w - .15)); t.move_to(r)
    return VGroup(r, t)


def ar(a, b, col=GREY_B, w=2.5):
    return Arrow(a, b, buff=0, stroke_width=w, color=col, max_tip_length_to_length_ratio=.12, tip_length=.15)


class NanoVLAS(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.add(NumberPlane(x_range=[-8, 8, .5], y_range=[-5, 5, .5], background_line_style={"stroke_color": "#232323", "stroke_width": 1}, axis_config={"stroke_opacity": 0}))
        fr = Z["frames"]
        # ---------------- layout: perception (left) -> decision (middle) -> servo (right)
        B = dict(
            img=blk([-5.8, 1.6, 0], "", 1.4, 1.1, WHITE, False), vit=blk([-3.9, 1.6, 0], "ViT-Tiny", 1.5, .7, CV), tok=blk([-1.9, 1.6, 0], "80 token", 1.4, .7, CV),
            txt=blk([-5.8, -1.0, 0], "Lệnh", 1.3, .7, CT, False), u=blk([-3.9, -1.0, 0], "Task u", 1.3, .7, CT),
            js=blk([-3.9, 0.3, 0], "Khớp", 1.3, .7, CS),
            prev=blk([-1.9, -2.4, 0], "Trạng thái trước", 1.9, .7, CL),
            st=blk([0.6, -0.6, 0], "Trạng thái", 1.6, .8, CL),
            gr=blk([0.6, 1.6, 0], "Grounding", 1.6, .8, CQ),
            way=blk([3.0, 1.6, 0], "Điểm đích (x,y,z,yaw)", 2.3, .8, CQ, fs=13), jaw=blk([3.0, -0.6, 0], "Kẹp mở / đóng", 2.0, .8, CG),
            servo=blk([5.4, 0.5, 0], "IK servo", 1.5, .9, WHITE, fs=15),
        )
        e = lambda n, s: B[n][0].get_edge_center(s)
        W = [ar(e("img", RIGHT), e("vit", LEFT), CV), ar(e("vit", RIGHT), e("tok", LEFT), CV), ar(e("tok", RIGHT), e("gr", LEFT), CV),
             ar(e("txt", RIGHT), e("u", LEFT), CT), ar(B["u"][0].get_corner(UR), B["gr"][0].get_corner(DL) + RIGHT * .2, CT),
             ar(e("u", RIGHT), B["st"][0].get_left() + DOWN * .15, CT), ar(e("js", RIGHT), B["st"][0].get_left() + UP * .2, CS),
             ar(B["tok"][0].get_bottom(), B["st"][0].get_corner(UL) + RIGHT * .2, CV),
             ar(e("prev", RIGHT), B["st"][0].get_corner(DL) + RIGHT * .3, CL), ar(e("st", UP), e("gr", DOWN), CL),
             ar(e("gr", RIGHT), e("way", LEFT), CQ), ar(e("st", RIGHT), e("jaw", LEFT), CL),
             ar(e("way", RIGHT), B["servo"][0].get_corner(UL) + DOWN * .2, CQ), ar(e("jaw", RIGHT), B["servo"][0].get_corner(DL) + UP * .2, CG)]
        loop = DashedVMobject(VMobject(stroke_color=CL, stroke_width=2.5).set_points_as_corners([e("st", DOWN), e("st", DOWN) + DOWN * 1.1, B["prev"][0].get_right() + RIGHT * .6, ]), num_dashes=16)
        self.play(LaggedStart(*[FadeIn(b) for b in B.values()], lag_ratio=.05), run_time=1.6)
        self.play(LaggedStart(*[GrowArrow(w) for w in W], lag_ratio=.05), Create(loop), run_time=1.6)
        self.play(FadeIn(tag("mỗi khi có ảnh mới (5–30 Hz, không cần đều)", 13, GREY_B).to_corner(UL, buff=.25)),
                  FadeIn(VGroup(tag("không học:", 12, GREY_B), tag("đi tới đích, tốc độ giới hạn, 30 Hz", 12, GREY_B)).arrange(DOWN, buff=.04).next_to(B["servo"], DOWN, buff=.15).to_edge(RIGHT, buff=.2)))
        # ---------------- state strip (discrete, event-driven)
        chips = VGroup(*[RoundedRectangle(corner_radius=.1, width=1.15, height=.42).set_stroke(CL, 1.5).set_fill("#111111", 1) for _ in STATES]).arrange(RIGHT, buff=.12).move_to([0.2, -3.45, 0])
        labels = VGroup(*[tag(s, 11, CL).move_to(c) for s, c in zip(STATES, chips)])
        self.play(FadeIn(chips, lag_ratio=.05), FadeIn(labels, lag_ratio=.05), run_time=.8)
        hl = chips[0].copy().set_fill(CL, .35)
        self.add(hl)
        # ---------------- run through real frames: the state advances on events, the target point moves with the view
        im = None
        events = ["tới trên vật", "chạm vị trí gắp", "kẹp đã đóng", "vật đã nhấc", "tới trên đích", "đã chạm mặt", "đã nhả"]
        ticks = np.linspace(5, len(fr) - 5, 7).astype(int)
        for s_i, t in enumerate(ticks):
            new = ImageMobject(fr[t]); new.set_height(1.0); new.move_to(B["img"])
            self.play(FadeIn(new) if im is None else FadeTransform(im, new), run_time=.35); im = new
            dot = Dot(color=CQ, radius=.07).move_to(new.get_center() + np.array([(-1) ** s_i * .18, .12 * ((s_i % 3) - 1), 0]))
            self.play(FadeIn(dot), Indicate(B["gr"], color=CQ, scale_factor=1.04), run_time=.35)
            self.play(hl.animate.move_to(chips[s_i]), FadeIn(tag(events[s_i], 12, CL).next_to(chips[s_i], UP, buff=.08)), run_time=.45)
            self.play(Indicate(B["servo"], color=WHITE, scale_factor=1.04), run_time=.3)
            self.remove(dot)
        self.play(FadeIn(VGroup(tag("trạng thái chỉ đổi khi thế giới đổi (sự kiện), không theo đồng hồ", 13, CL),
                                tag("trễ / jitter chỉ làm chậm, không làm sai · minh hoạ, ảnh thật", 11, GREY_B)).arrange(DOWN, buff=.06).to_corner(UR, buff=.25)))
        self.wait(2.5)
