"""NanoVLA-C: forward through the closed loop over frames, losses, back-propagation (BPTT through the memory only,
stop-gradient on the carried plan / prediction), gradient clipping, AdamW + EMA update. Numbers from extract_bp.py
(one real training batch, checkpoint step 4000). Frames: real wrist images (rec.npz).
Render: manim -qm --media_dir manim_media bp_manim.py ForwardBackward"""
import json, os
import numpy as np
from manim import *

Hd = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{Hd}/bp.json")); Z = np.load(f"{Hd}/rec.npz")
BG = "#0f1115"
CV, CT, CS, CL, CQ, CF = "#4cc9f0", "#b388ff", "#80ed99", "#ff8a3d", "#ffd166", "#63e6be"
FWD, BWD = "#74c0fc", "#ff6b6b"


def tag(s, fs=14, col=GREY_B):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def blk(c, name, w=1.0, h=.45, col=WHITE, fs=13):
    r = RoundedRectangle(corner_radius=.08, width=w, height=h).set_stroke(col, 2).set_fill("#15181e", 1).move_to(c)
    t = tag(name, fs, col); t.set_width(min(t.width, w - .1)); t.move_to(r)
    return VGroup(r, t)


def ar(a, b, col=GREY_B, w=2.2, dashed=False):
    if dashed:
        return DashedVMobject(Arrow(a, b, buff=.04, stroke_width=w, color=col, max_tip_length_to_length_ratio=.12, tip_length=.13), num_dashes=10)
    return Arrow(a, b, buff=.04, stroke_width=w, color=col, max_tip_length_to_length_ratio=.12, tip_length=.13)


def scissors(p):
    return VGroup(Line(UL * .11, DR * .11, color=BWD, stroke_width=3), Line(UR * .11, DL * .11, color=BWD, stroke_width=3)).move_to(p)


class ForwardBackward(Scene):
    def construct(self):
        self.camera.background_color = BG
        xs = [-4.3, -0.5, 3.3]
        T = {}
        fr = Z["frames"]; ti = [30, 31, 32]
        for k, x in enumerate(xs):
            im = ImageMobject(fr[ti[k]]); im.set_height(.75); im.move_to([x - .4, -3.3, 0])
            vit = VGroup(*[Rectangle(width=.8, height=.07).set_fill(GREY_D if i < 4 else CV, 1).set_stroke(width=0).shift(UP * i * .09) for i in range(12)]).move_to([x - .4, -2.15, 0])
            T[k] = dict(im=im, vit=vit, v=blk([x - .4, -1.25, 0], "v", .55, .4, CV), s=blk([x + .45, -1.25, 0], "s", .55, .4, CS),
                        e=blk([x + 1.25, -1.25, 0], "e", .5, .4, CF), sh=blk([x + 1.25, -.45, 0], "shift(plan)", 1.0, .4, CQ, 11),
                        mlp=blk([x, -.45, 0], "MLP", .9, .45, CQ), plan=blk([x, .45, 0], "plan", .9, .4, CQ),
                        fwd=blk([x + 1.0, .45, 0], "Forward", .9, .4, CF), gru=blk([x, -2.15 + .0, 0] if False else [x + .45, -2.15, 0], "GRU", .65, .4, CL),
                        L=blk([x, 1.45, 0], "L_action", 1.0, .4, BWD, 12))
        u = blk([-6.3, -.45, 0], "u (task)", 1.0, .45, CT, 12); m0 = blk([-6.3, -2.15, 0], "m₀", .6, .4, CL)
        Lf = [blk([(xs[k] + xs[k + 1]) / 2 + .7, 1.45, 0], "L_forward", 1.1, .4, BWD, 12) for k in range(2)]
        Lp = blk([5.6, -2.9, 0], "L_phase", 1.0, .4, BWD, 12)
        # ------------- 1 build + forward pass, frame by frame
        self.play(FadeIn(u), FadeIn(m0))
        carries = []
        prev_m = m0
        for k in range(3):
            t = T[k]
            self.play(FadeIn(t["im"]), FadeIn(t["vit"]), *[FadeIn(t[n]) for n in ("v", "s", "e", "sh", "mlp", "plan", "fwd", "gru")], run_time=.6)
            A = [ar(t["im"].get_top(), t["vit"].get_bottom(), FWD), ar(t["vit"].get_top(), t["v"].get_bottom(), FWD),
                 ar(t["v"].get_top(), t["mlp"][0].get_bottom() + LEFT * .2, FWD), ar(t["s"].get_top(), t["mlp"][0].get_bottom() + RIGHT * .1, FWD),
                 ar(t["e"].get_top(), t["sh"].get_bottom(), FWD) if False else ar(t["e"][0].get_left() + UP * .1, t["mlp"][0].get_right() + DOWN * .1, FWD),
                 ar(t["sh"][0].get_left(), t["mlp"][0].get_right() + UP * .1, FWD), ar(t["mlp"].get_top(), t["plan"].get_bottom(), FWD),
                 ar(t["plan"][0].get_right(), t["fwd"][0].get_left(), FWD), ar(prev_m[0].get_right(), t["gru"][0].get_left(), CL, 3),
                 ar(t["v"][0].get_bottom() + RIGHT * .1, t["gru"][0].get_top() + LEFT * .15, FWD), ar(u[0].get_right(), t["mlp"][0].get_left(), CT, 1.5) if k == 0 else VGroup()]
            self.play(LaggedStart(*[GrowArrow(a) if isinstance(a, Arrow) else Create(a) for a in A], lag_ratio=.06), run_time=1.0)
            t["arrows"] = A
            if k:
                pt = T[k - 1]
                c1 = ar(pt["plan"][0].get_right() + DOWN * .1, t["sh"][0].get_top() + LEFT * .2, CQ, 2, dashed=True)
                c2 = ar(pt["fwd"][0].get_bottom(), t["e"][0].get_top(), CF, 2, dashed=True)
                self.play(Create(c1), Create(c2), run_time=.6)
                carries += [c1, c2]
            prev_m = t["gru"]
        self.play(FadeIn(tag("forward: mỗi frame 0,1 s · trái → phải", 13, FWD).to_corner(UL, buff=.25)))
        # ------------- 2 losses (real values)
        self.play(*[FadeIn(T[k]["L"]) for k in range(3)], *[FadeIn(l) for l in Lf], FadeIn(Lp),
                  *[GrowArrow(ar(T[k]["plan"].get_top(), T[k]["L"].get_bottom(), BWD, 1.5)) for k in range(3)],
                  *[GrowArrow(ar(T[k]["fwd"].get_top(), Lf[k][0].get_bottom() + LEFT * .2, BWD, 1.5)) for k in range(2)],
                  *[GrowArrow(ar(T[k + 1]["s"][0].get_right() + UP * .1, Lf[k][0].get_bottom() + RIGHT * .3, BWD, 1)) for k in range(2)],
                  GrowArrow(ar(T[2]["gru"][0].get_right(), Lp[0].get_left(), BWD, 1.5)), run_time=1)
        vals = VGroup(tag(f"L_action {J['la']:.3f}", 13, WHITE), tag(f"L_forward {J['ls']:.4f} + {J['lz']:.4f}", 13, WHITE), tag(f"L_phase {J['lp']:.3f}", 13, WHITE),
                      tag(f"L = L_act + 0.5·L_fwd + 0.1·L_phase = {J['loss']:.3f}", 14, BWD)).arrange(DOWN, aligned_edge=LEFT, buff=.08).to_corner(UR, buff=.25)
        self.play(FadeIn(vals, lag_ratio=.2))
        self.wait(.6)
        # ------------- 3 backward: red flows back; memory chain carries gradient across frames (BPTT), carries are cut
        self.play(FadeOut(self.mobjects[0]) if False else Wait(0.01))
        sc = VGroup(*[scissors(c.get_center()) for c in carries])
        self.play(FadeIn(sc), FadeIn(tag("✕ detach: không có gradient qua plan / dự đoán mang sang", 12, BWD).to_corner(UL, buff=.25).shift(DOWN * .35)))
        mg = np.array(J["mem_grad"]); g3 = [mg[int(i)] for i in np.linspace(0, len(mg) - 1, 3)]; gmax = max(g3)
        for k in (2, 1, 0):
            t = T[k]
            back = [t["L"], t["plan"], t["mlp"], t["v"], t["gru"]]
            dots = [Dot(t["L"].get_center(), color=BWD, radius=.08) for _ in range(3)]
            paths = [Line(t["L"].get_center(), t["mlp"].get_center()).append_points(Line(t["mlp"].get_center(), t["v"].get_center()).points),
                     Line(t["L"].get_center(), t["mlp"].get_center()).append_points(Line(t["mlp"].get_center(), t["gru"].get_center()).points),
                     Line(t["L"].get_center(), t["mlp"].get_center()).append_points(Line(t["mlp"].get_center(), t["vit"].get_center()).points)]
            self.play(*[MoveAlongPath(d, p) for d, p in zip(dots, paths)], *[b[0].animate.set_stroke(BWD, 3) for b in back],
                      t["vit"][4:].animate.set_fill(BWD, 1), run_time=.8)
            self.remove(*dots)
            wid = 2 + 8 * g3[k] / gmax
            self.play(t["gru"][0].animate.set_fill(BWD, .25 + .5 * g3[k] / gmax), run_time=.3)
            if k:
                pm = T[k - 1]["gru"]
                ln = Line(t["gru"][0].get_left(), pm[0].get_right(), color=BWD, stroke_width=wid)
                self.play(Create(ln), run_time=.4)
        lock = VGroup(*[VGroup(RoundedRectangle(corner_radius=.03, width=.16, height=.12).set_fill(GREY_C, 1).set_stroke(width=0),
                               Arc(radius=.05, start_angle=0, angle=PI, color=GREY_C, stroke_width=2).shift(UP * .07)).next_to(T[k]["vit"][:4], LEFT, buff=.05) for k in range(3)])
        self.play(FadeIn(lock), FadeIn(tag("ViT block 0–3 + patch: đóng băng (2,0M) · đỏ = nhận gradient", 12).to_corner(UL, buff=.25).shift(DOWN * .7)))
        bp = tag(f"BPTT qua memory (độ dày = |∂L/∂m| thật): frame đầu {mg[0]:.4f} → frame cuối {mg[-1]:.4f}", 12, BWD).to_corner(UL, buff=.25).shift(DOWN * 1.05)
        self.play(FadeIn(bp))
        self.wait(1.5)
        self.play(*[FadeOut(x) for x in self.mobjects], run_time=.6)
        # ------------- 4 update: per-module gradient norms (real), clipping, AdamW, EMA
        g = J["grad"]; names = list(g); mx = max(g.values())
        bars = VGroup()
        for i, n in enumerate(names):
            w = 6.0 * g[n] / mx
            col = CQ if "MLP" in n else CL if "GRU" in n else CF if "forward" in n else CV if ("ViT" in n or "pool" in n or "vvec" in n) else CT if ("text" in n or "task" in n) else GREY_B
            bars.add(VGroup(tag(n, 13).set_width(min(tag(n, 13).width, 2.3)), Rectangle(width=max(w, .03), height=.26).set_fill(col, 1).set_stroke(width=0), tag(f"{g[n]:.3f}", 12, WHITE)))
        for b in bars:
            b[0].move_to([-4.6, 0, 0], aligned_edge=RIGHT); b[1].next_to(b[0], RIGHT, buff=.2); b[2].next_to(b[1], RIGHT, buff=.1)
        bars.arrange(DOWN, aligned_edge=LEFT, buff=.12).move_to([-1.0, .4, 0])
        hdr = tag("‖gradient‖ theo khối (1 batch thật)", 15, WHITE).next_to(bars, UP, buff=.25)
        self.play(FadeIn(hdr), LaggedStart(*[GrowFromEdge(b[1], LEFT) for b in bars], lag_ratio=.07), FadeIn(VGroup(*[VGroup(b[0], b[2]) for b in bars])), run_time=1.8)
        clip = VGroup(tag(f"tổng ‖g‖ = {J['total_grad']:.2f}  <  clip 1.0  → không cắt", 14, WHITE)).next_to(bars, DOWN, buff=.35)
        self.play(FadeIn(clip))
        opt = VGroup(blk([0, 0, 0], "AdamW", 1.6, .55, BWD, 15), VGroup(tag("lr head 3e-4 · lr encoder 2e-5", 12), tag("warmup 500 + cosine · wd 0.05", 12)).arrange(DOWN, buff=.05)).arrange(DOWN, buff=.1).move_to([4.6, 1.5, 0])
        ema = VGroup(blk([0, 0, 0], "EMA 0.999", 1.6, .55, CS, 15), tag("bản trung bình → checkpoint / eval", 12)).arrange(DOWN, buff=.1).move_to([4.6, -1.0, 0])
        w_ = blk([4.6, .1, 0], "θ ← θ − lr·Adam(g)", 2.6, .45, WHITE, 13)
        self.play(FadeIn(opt), GrowArrow(ar(bars.get_right() + RIGHT * .3, opt[0].get_left(), BWD)))
        self.play(FadeIn(w_), GrowArrow(ar(opt[0].get_bottom(), w_.get_top(), BWD)))
        self.play(FadeIn(ema), GrowArrow(ar(w_.get_bottom(), ema[0].get_top(), CS)))
        self.play(FadeIn(tag("lặp lại 20.000 lần · 8 cửa sổ × 48 frame / bước", 13).to_edge(DOWN, buff=.3)))
        self.wait(2)
