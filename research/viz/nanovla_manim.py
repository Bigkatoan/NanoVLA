"""NanoVLA wrist-only: data + model + results, picture-first (real data from viz/extract.py).
Render: manim -qm --media_dir manim_media nanovla_manim.py NanoVLAViz"""
import json, os
import numpy as np
from manim import *
from PIL import Image

H = os.path.dirname(os.path.abspath(__file__))
Z = np.load(f"{H}/data.npz"); J = json.load(open(f"{H}/data.json"))
BG = "#0f1115"
TASKS = ["pick", "pad", "tray", "container", "bowl", "rel", "push", "topple", "rotate"]
TC = {"pick": "#4cc9f0", "pad": "#f72585", "tray": "#b388ff", "container": "#ff8a3d", "bowl": "#ffd166",
      "rel": "#80ed99", "push": "#ef476f", "topple": "#06d6a0", "rotate": "#a0c4ff"}
JC = ["#4cc9f0", "#80ed99", "#ffd166", "#ff8a3d", "#b388ff", "#f72585"]


def tag(s, fs=18, col=GREY_B):
    return Text(s, font="DejaVu Sans", font_size=fs, color=col)


def img(a, h):
    m = ImageMobject(np.asarray(a, dtype=np.uint8)); m.set_height(h); return m


def box(w, h, col=GREY_E, stroke=GREY_B):
    return RoundedRectangle(corner_radius=.12, width=w, height=h, stroke_color=stroke, fill_color=col, fill_opacity=1, stroke_width=1.5)


def rgb(v):
    return ManimColor.from_rgb([float(x) / 255 for x in v])


def clear(scene):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=.6)


class NanoVLAViz(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.data_tasks()
        self.data_dr()
        self.data_episode()
        self.model()
        self.loop()
        self.results()

    # ------------------------------------------------------------------ 1 nine tasks, real wrist frames + demo counts
    def data_tasks(self):
        tiles = Group()
        for i, t in enumerate(TASKS):
            fr = Z[f"th_{t}"]
            im = img(fr[0], 1.15)
            fr_rect = SurroundingRectangle(im, buff=0, color=TC[t], stroke_width=3)
            g = Group(im, fr_rect, tag(t, 16, TC[t]).next_to(im, DOWN, buff=.06))
            tiles.add(g)
        tiles.arrange_in_grid(3, 3, buff=(.25, .3)).move_to(LEFT * 3.4)
        cam = VGroup(Rectangle(width=.5, height=.32, color=WHITE, fill_color=GREY_D, fill_opacity=1, stroke_width=2),
                     Circle(.09, color=WHITE, stroke_width=2)).move_to([-3.4, 3.55, 0])
        self.play(FadeIn(cam), LaggedStart(*[FadeIn(t, scale=.85) for t in tiles], lag_ratio=.08), run_time=1.8)
        # play each tile's 6 real frames
        for k in range(1, 6):
            new = []
            for i, t in enumerate(TASKS):
                im = img(Z[f"th_{t}"][k], 1.15).move_to(tiles[i][0]); new.append(im)
            self.play(*[FadeTransform(tiles[i][0], new[i]) for i in range(9)], run_time=.35)
            for i in range(9):
                tiles[i].submobjects[0] = new[i]
        # bars of demo counts (real)
        cnt = J["cnt"]; mx = max(cnt.values()); x0, y0 = 1.2, -2.9
        bars = VGroup(); labs = VGroup()
        for i, t in enumerate(TASKS):
            h = 5.2 * cnt[t] / mx
            b = Rectangle(width=.46, height=h, fill_color=TC[t], fill_opacity=1, stroke_width=0).move_to([x0 + i * .6, y0 + h / 2, 0])
            bars.add(b); labs.add(tag(str(cnt[t]), 14, WHITE).next_to(b, UP, buff=.05))
        axis = Line([x0 - .4, y0, 0], [x0 + 8 * .6 + .4, y0, 0], color=GREY_B, stroke_width=2)
        tot = tag(f"{sum(cnt.values())} demos", 18, WHITE).move_to([x0 + 2.4, 3.3, 0])
        self.play(Create(axis), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=.08), run_time=1.4)
        self.play(FadeIn(labs), FadeIn(tot))
        # each bar <-> its tile
        self.play(LaggedStart(*[Indicate(tiles[i][1], color=TC[t], scale_factor=1.06) for i, t in enumerate(TASKS)], lag_ratio=.1), run_time=1.6)
        self.wait(.6); clear(self)

    # ------------------------------------------------------------------ 2 domain randomization: 24 real episodes
    def data_dr(self):
        dr = Z["dr"]
        grid = Group(*[img(a, .95) for a in dr]).arrange_in_grid(4, 6, buff=.12).move_to(UP * .2)
        self.play(LaggedStart(*[FadeIn(g, shift=UP * .15) for g in grid], lag_ratio=.04), run_time=2.2)
        # icons of what changes (light / table / room / robot base)
        bulb = VGroup(Circle(.17, color=YELLOW, fill_opacity=.8), Rectangle(width=.12, height=.12, color=GREY_B, fill_opacity=1).shift(DOWN * .22))
        table = VGroup(*[Square(.14, fill_color=c, fill_opacity=1, stroke_width=0) for c in ("#8d6e63", "#bcaaa4", "#5d4037", "#d7ccc8")]).arrange_in_grid(2, 2, buff=0)
        room = VGroup(Polygon([-.25, -.2, 0], [.25, -.2, 0], [.25, .12, 0], [0, .3, 0], [-.25, .12, 0], color=GREY_A, stroke_width=2))
        base = VGroup(Circle(.16, color="#4cc9f0", stroke_width=3), *[Arrow(ORIGIN, d * .35, buff=0, stroke_width=2, color="#4cc9f0", max_tip_length_to_length_ratio=.35) for d in (LEFT, RIGHT, UP, DOWN)])
        icons = VGroup(bulb, table, room, base).arrange(RIGHT, buff=1.2).move_to(DOWN * 3.2)
        self.play(FadeIn(icons, lag_ratio=.2))
        for _ in range(2):
            perm = np.random.default_rng(_).permutation(len(dr))
            self.play(*[g.animate.move_to(grid[int(p)].get_center()) for g, p in zip(grid, perm)], run_time=.9)
        self.wait(.5); clear(self)

    # ------------------------------------------------------------------ 3 one demo: frames + real joint curves; DART: label vs executed
    def data_episode(self):
        seq = Z["seq"]; st = Z["st"]; ac = Z["ac"]
        cam = img(seq[0], 2.4).move_to([-4.3, 1.6, 0]); self.add(cam)
        ax = Axes(x_range=[0, len(st), 50], y_range=[-2.2, 2.2, 1], x_length=7.2, y_length=4.6, tips=False,
                  axis_config={"stroke_width": 1.5, "color": GREY_B}).move_to([2.6, .2, 0])
        self.play(FadeIn(ax), FadeIn(cam))
        tr = ValueTracker(1)
        def curve(j, arr, op=1., w=3):
            return always_redraw(lambda: ax.plot_line_graph(np.arange(int(tr.get_value())), arr[:int(tr.get_value()), j], add_vertex_dots=False,
                                                         line_color=JC[j], stroke_width=w, stroke_opacity=op) if tr.get_value() >= 2 else VGroup())
        lab = [curve(j, ac, 1, 3) for j in range(6)]
        exe = [curve(j, st, .45, 6) for j in range(6)]
        self.add(*exe, *lab)
        n = len(seq); T = len(st)
        for k in range(1, n):
            new = img(seq[k], 2.4).move_to(cam)
            self.play(tr.animate.set_value(max(2, T * (k + 1) / n)), FadeTransform(cam, new), run_time=.22, rate_func=linear)
            cam = new
        # zoom: thick = executed (noisy), thin = clean label
        legend = VGroup(VGroup(Line(ORIGIN, RIGHT * .6, stroke_width=7, stroke_opacity=.45, color=WHITE), tag("exec", 16)).arrange(RIGHT, buff=.15),
                        VGroup(Line(ORIGIN, RIGHT * .6, stroke_width=3, color=WHITE), tag("label", 16)).arrange(RIGHT, buff=.15)).arrange(DOWN, aligned_edge=LEFT).move_to([-4.3, -1.6, 0])
        jl = VGroup(*[VGroup(Dot(color=JC[j], radius=.07), tag(n_, 14)).arrange(RIGHT, buff=.08) for j, n_ in
                      enumerate(["pan", "lift", "elbow", "wrist", "roll", "grip"])]).arrange_in_grid(3, 2, buff=(.3, .12)).move_to([-4.3, -2.8, 0])
        self.play(FadeIn(legend), FadeIn(jl))
        foot = tag("pick_c5 ep0 · DART σ=0.03", 14).to_corner(DR, buff=.3)
        self.play(FadeIn(foot)); self.wait(1.4)
        self.remove(*lab, *exe); clear(self)

    # ------------------------------------------------------------------ 4 the model (shapes from ckpt/vla_isaac_wrist_v1)
    def model(self):
        fr = np.asarray(Image.fromarray(Z["seq"][8]).resize((112, 112)), dtype=float)
        pm = fr.reshape(7, 16, 7, 16, 3).mean((1, 3))
        im = img(Z["seq"][8], 1.9).move_to([-5.6, 2.0, 0])
        self.play(FadeIn(im))
        cells = VGroup(*[Square(.2, fill_color=rgb(pm[r, c]), fill_opacity=1, stroke_width=.5, stroke_color=GREY_B)
                         .move_to([-5.6 + (c - 3) * .21, 2.0 - 1.9 + (3 - r) * .21 - .9, 0]) for r in range(7) for c in range(7)])
        cells.move_to([-5.6, -.4, 0])
        self.play(FadeIn(cells, lag_ratio=.01), run_time=1.0)
        # ViT: 12 stacked blocks
        vit = VGroup(*[box(1.1, .16, "#1e3a5f", "#4cc9f0").shift(UP * i * .2) for i in range(12)]).move_to([-3.4, 1.0, 0])
        # text encoder: 4 blocks; words as small chips
        words = VGroup(*[RoundedRectangle(corner_radius=.05, width=.42, height=.22, fill_color="#3a2a4f", fill_opacity=1, stroke_color="#b388ff", stroke_width=1.2) for _ in range(5)]).arrange(RIGHT, buff=.06).move_to([-5.6, -2.6, 0])
        txt = tag("pick up the small tuna can", 14, "#b388ff").next_to(words, DOWN, buff=.12)
        ten = VGroup(*[box(1.1, .16, "#3a2a4f", "#b388ff").shift(UP * i * .2) for i in range(4)]).move_to([-3.4, -2.3, 0])
        # state: 6 joint bars
        s = Z["st"][60]; sb = VGroup(*[Rectangle(width=.12, height=.15 + .35 * abs(v), fill_color=JC[j], fill_opacity=1, stroke_width=0) for j, v in enumerate(s)]).arrange(RIGHT, buff=.05, aligned_edge=DOWN).move_to([-3.4, -.8, 0])
        A = lambda a, b, c=GREY_B: Arrow(a, b, buff=.08, stroke_width=3, color=c, max_tip_length_to_length_ratio=.12)
        self.play(FadeIn(vit, lag_ratio=.05), run_time=.6)
        a1 = A(cells.get_right(), vit.get_left(), "#4cc9f0")
        self.play(FadeIn(words, lag_ratio=.1), FadeIn(txt), FadeIn(ten, lag_ratio=.1), FadeIn(sb), GrowArrow(a1), run_time=1.2)
        a2 = A(words.get_right() + UP * .1, ten.get_left(), "#b388ff")
        self.play(GrowArrow(a2))
        # fusion token sequence: 50 vision + 1 state + text tokens
        seqx = -1.3
        vt = VGroup(*[Square(.13, fill_color=c.fill_color, fill_opacity=1, stroke_width=0) for c in cells]).arrange_in_grid(7, 7, buff=.03).move_to([seqx, 1.0, 0])
        stt = Square(.22, fill_color=WHITE, fill_opacity=1, stroke_width=0).move_to([seqx, -.8, 0])
        tt = VGroup(*[Square(.13, fill_color="#b388ff", fill_opacity=1, stroke_width=0) for _ in range(8)]).arrange(RIGHT, buff=.03).move_to([seqx, -2.3, 0])
        self.play(GrowArrow(A(vit.get_right(), vt.get_left(), "#4cc9f0")), GrowArrow(A(sb.get_right(), stt.get_left())), GrowArrow(A(ten.get_right(), tt.get_left(), "#b388ff")),
                  FadeIn(vt), FadeIn(stt), FadeIn(tt), run_time=1.0)
        fuse = box(.9, 4.4, "#2b2b2b", WHITE).move_to([.4, -.4, 0])
        fl = VGroup(*[Line([.0, y, 0], [.8, y, 0], stroke_width=1, color=GREY_B) for y in np.linspace(-2.4, 1.6, 9)])
        self.play(FadeIn(fuse), *[GrowArrow(A(m.get_right(), [fuse.get_left()[0], m.get_center()[1], 0])) for m in (vt, stt, tt)], run_time=.8)
        # decoder: 16 action queries attend to fused memory, 3 layers
        q = VGroup(*[Circle(.08, fill_color=YELLOW, fill_opacity=1, stroke_width=0) for _ in range(16)]).arrange(DOWN, buff=.08).move_to([1.9, -.4, 0])
        dec = VGroup(*[box(.3, 3.2, "#3d3415", YELLOW).shift(RIGHT * i * .38) for i in range(3)]).move_to([2.9, -.4, 0])
        self.play(FadeIn(q, lag_ratio=.05), FadeIn(dec, lag_ratio=.2), run_time=1.0)
        mem = VGroup(*[CurvedArrow(fuse.get_right() + UP * y, dec[i].get_top() + DOWN * .3, angle=-.6, color=GREY_B, stroke_width=1.5) for i, y in enumerate((1.2, .4, -.4))])
        self.play(Create(mem), GrowArrow(A(q.get_right(), dec.get_left(), YELLOW)), run_time=.8)
        # output: 16 x 6 chunk -> future joint curves (real label trajectory slice)
        ax = Axes(x_range=[0, 15, 5], y_range=[-2, 2, 1], x_length=2.6, y_length=3.0, tips=False, axis_config={"stroke_width": 1.2, "color": GREY_B}).move_to([5.4, -.4, 0])
        ac = Z["ac"]; sl = ac[60:60 + 16 * 3:3]
        curves = VGroup(*[ax.plot_line_graph(np.arange(16), sl[:, j], add_vertex_dots=True, line_color=JC[j], stroke_width=2.5,
                                             vertex_dot_radius=.03, vertex_dot_style={"fill_color": JC[j]}) for j in range(6)])
        self.play(GrowArrow(A(dec.get_right(), ax.get_left(), YELLOW)), FadeIn(ax), run_time=.5)
        self.play(Create(curves, lag_ratio=.1), run_time=1.4)
        self.play(FadeIn(tag("16 × 6 · 1.6 s", 14).next_to(ax, DOWN, buff=.15)))
        # token dots flowing
        dots = [Dot(color=WHITE, radius=.07).move_to(p) for p in (cells.get_center(), words.get_center(), sb.get_center())]
        paths = [[cells.get_center(), vit.get_center(), vt.get_center(), fuse.get_center(), dec.get_center(), ax.get_center()],
                 [words.get_center(), ten.get_center(), tt.get_center(), fuse.get_center(), dec.get_center(), ax.get_center()],
                 [sb.get_center(), stt.get_center(), fuse.get_center(), dec.get_center(), ax.get_center()]]
        def path(pts):
            m = VMobject(); m.set_points_as_corners([np.array(p, dtype=float) for p in pts]); return m
        self.play(*[MoveAlongPath(d, path(p)) for d, p in zip(dots, paths)], run_time=2.2, rate_func=smooth)
        self.play(*[FadeOut(d) for d in dots])
        # param bars (real counts)
        P = [("ViT", 5.50, "#4cc9f0"), ("text", 2.47, "#b388ff"), ("dec", 2.37, YELLOW), ("fuse+emb", 0.70, WHITE)]
        pb = VGroup()
        for i, (n_, v, c) in enumerate(P):
            b = Rectangle(width=v * .45, height=.22, fill_color=c, fill_opacity=1, stroke_width=0)
            g = VGroup(tag(n_, 14).set_width(.9) if len(n_) > 5 else tag(n_, 14), b, tag(f"{v:.1f}M", 14, WHITE)).arrange(RIGHT, buff=.15)
            pb.add(g)
        pb.arrange(DOWN, aligned_edge=LEFT, buff=.1).to_corner(UR, buff=.3)
        self.play(FadeIn(pb, lag_ratio=.2))
        self.play(FadeIn(tag("11.0M params · wrist cam only", 14).next_to(pb, DOWN, buff=.15, aligned_edge=RIGHT)))
        self.wait(1.5); clear(self)

    # ------------------------------------------------------------------ 5 closed loop: predict 16, execute 4, re-plan
    def loop(self):
        ac = Z["ac"]; j = 1; T = 120
        ax = Axes(x_range=[0, T, 20], y_range=[float(ac[:T, j].min()) - .2, float(ac[:T, j].max()) + .2, .5], x_length=11, y_length=4.2, tips=False,
                  axis_config={"stroke_width": 1.5, "color": GREY_B}).move_to(DOWN * .3)
        self.play(FadeIn(ax))
        done = VGroup(); t = 0
        while t + 48 <= T + 48 and t < T - 12:
            idx = np.arange(t, min(t + 48, T), 3)
            pred = ax.plot_line_graph(idx, ac[idx, j], add_vertex_dots=True, line_color=YELLOW, stroke_width=2, vertex_dot_radius=.035, vertex_dot_style={"fill_color": YELLOW})
            pred.set_fill(opacity=0).set_stroke(opacity=.6)
            self.play(Create(pred), run_time=.35)
            ex_idx = np.arange(t, min(t + 13, T))
            ex = ax.plot_line_graph(ex_idx, ac[ex_idx, j], add_vertex_dots=False, line_color="#4cc9f0", stroke_width=6); ex.set_fill(opacity=0)
            self.play(Create(ex), run_time=.35)
            self.play(FadeOut(pred), run_time=.15)
            done.add(ex); t += 12
        legend = VGroup(VGroup(Dot(color=YELLOW), tag("16 predicted", 14)).arrange(RIGHT, buff=.1),
                        VGroup(Line(ORIGIN, RIGHT * .5, color="#4cc9f0", stroke_width=6), tag("4 executed", 14)).arrange(RIGHT, buff=.1)).arrange(RIGHT, buff=.8).to_edge(UP, buff=.5)
        self.play(FadeIn(legend)); self.wait(1); clear(self)

    # ------------------------------------------------------------------ 6 honest results
    def results(self):
        A = J["res"]["A_zeroshot_multitask"]["by"]; order = [t for t in TASKS if t in A]
        x0, y0, Hh = -6.2, -2.4, 4.4
        axis = Line([x0 - .3, y0, 0], [x0 + len(order) * .7, y0, 0], color=GREY_B, stroke_width=2)
        g = VGroup()
        for i, t in enumerate(order):
            s, n = A[t]; h = max(.04, Hh * s / n)
            ghost = Rectangle(width=.5, height=Hh, stroke_color=TC[t], stroke_width=1.2, fill_opacity=0).move_to([x0 + i * .7, y0 + Hh / 2, 0])
            b = Rectangle(width=.5, height=h, fill_color=TC[t], fill_opacity=1, stroke_width=0).move_to([x0 + i * .7, y0 + h / 2, 0])
            g.add(VGroup(ghost, b, tag(f"{s}/{n}", 14, WHITE).next_to(ghost, UP, buff=.05), tag(t, 12, TC[t]).next_to(ghost, DOWN, buff=.12 + .22 * (i % 2))))
        hA = tag("A zero-shot", 16, WHITE).move_to([x0 + 2.2, 3.4, 0])
        self.play(Create(axis), FadeIn(hA), LaggedStart(*[FadeIn(x) for x in g], lag_ratio=.12), run_time=2)
        self.play(FadeIn(tag("rel: 2 spurious t<0.3 s removed", 12).next_to(axis, DOWN, buff=.75)))
        # B / C : few-shot bowl & unseen objects
        R = J["res"]
        items = [("bowl N10", R["B_bowl_isaac_N10"], "#ffd166"), ("bowl N50", R["B_bowl_isaac_N50"], "#ffd166"), ("bowl v1 N50", R["B_bowl_v1_N50"], "#ffd166"),
                 ("unseen 0-shot", R["C_unseen_zeroshot"], "#80ed99"), ("unseen ft 60", R["C_unseen_ft"], "#80ed99")]
        x1 = 1.0; g2 = VGroup()
        for i, (n_, r, c) in enumerate(items):
            s = round(r["sr"] * r["n"]); h = max(.04, Hh * r["sr"])
            ghost = Rectangle(width=.6, height=Hh, stroke_color=c, stroke_width=1.2, fill_opacity=0).move_to([x1 + i * 1.05, y0 + Hh / 2, 0])
            b = Rectangle(width=.6, height=h, fill_color=c, fill_opacity=1, stroke_width=0).move_to([x1 + i * 1.05, y0 + h / 2, 0])
            g2.add(VGroup(ghost, b, tag(f"{s}/{r['n']}", 14, WHITE).next_to(ghost, UP, buff=.05), tag(n_, 12, c).next_to(ghost, DOWN, buff=.12 + .22 * (i % 2))))
        axis2 = Line([x1 - .5, y0, 0], [x1 + 4 * 1.05 + .5, y0, 0], color=GREY_B, stroke_width=2)
        hB = tag("B few-shot · C unseen", 16, WHITE).move_to([x1 + 2.1, 3.4, 0])
        self.play(Create(axis2), FadeIn(hB), LaggedStart(*[FadeIn(x) for x in g2], lag_ratio=.12), run_time=2)
        self.wait(1.5); clear(self)
        # failure mode: two-step task -> goes to destination first (schematic)
        tbl = Rectangle(width=10, height=.15, fill_color="#5d4037", fill_opacity=1, stroke_width=0).move_to(DOWN * 2)
        obj = Square(.45, fill_color="#4cc9f0", fill_opacity=1, stroke_width=0).move_to([-3, -1.7, 0])
        dst = VGroup(Line([1.8, -1.92, 0], [3.4, -1.92, 0], color="#ffd166", stroke_width=6), Line([1.8, -1.92, 0], [1.6, -1.3, 0], color="#ffd166", stroke_width=6),
                     Line([3.4, -1.92, 0], [3.6, -1.3, 0], color="#ffd166", stroke_width=6))
        grip = VGroup(Line(LEFT * .25, RIGHT * .25, stroke_width=6), Line(LEFT * .25, LEFT * .25 + DOWN * .35, stroke_width=6), Line(RIGHT * .25, RIGHT * .25 + DOWN * .35, stroke_width=6)).move_to([0, 2, 0])
        self.play(FadeIn(tbl), FadeIn(obj), FadeIn(dst), FadeIn(grip))
        good = DashedVMobject(VMobject(stroke_color=GREEN, stroke_width=3).set_points_as_corners([[0, 1.7, 0], [-3, -1.2, 0], [-3, .4, 0], [2.6, -1.2, 0]]), num_dashes=40)
        self.play(Create(good), run_time=1.2)
        bad = VMobject(stroke_color=RED, stroke_width=3).set_points_as_corners([[0, 1.7, 0], [2.6, -1.25, 0]])
        self.play(grip.animate.move_to([2.6, -1.3, 0]), Create(bad), run_time=1.4)
        x = VGroup(Line(UL * .3, DR * .3, color=RED, stroke_width=8), Line(UR * .3, DL * .3, color=RED, stroke_width=8)).move_to([-3, -.9, 0])
        self.play(Create(x)); self.play(Indicate(obj, color=RED))
        self.play(FadeIn(tag("schematic · 2-step tasks: gripper goes to target first", 14).to_corner(DR, buff=.3)))
        self.wait(1.5); clear(self)
