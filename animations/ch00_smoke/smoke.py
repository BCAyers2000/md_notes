"""Smoke test of the Manim toolchain and the house style.

    manim -qm --media_dir ../media smoke.py Lissajous

A point traces a Lissajous figure while the curve it leaves is drawn
behind it; a mathpazo label checks the TeX template.
"""

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UP, Create, Dot, FadeIn, MoveAlongPath, Scene,
                   VGroup, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from house import ACCENT, REFERENCE, MathTex, axes  # noqa: E402


class Lissajous(Scene):
    def construct(self):
        frame = axes(x_range=[-1.2, 1.2], y_range=[-1.2, 1.2],
                     x_length=5.0, y_length=5.0)
        curve = frame.plot_parametric_curve(
            lambda t: np.array([np.sin(3 * t), np.sin(2 * t), 0.0]),
            t_range=[0, 2 * np.pi], color=ACCENT, stroke_width=4)
        label = MathTex(r"\big(\sin 3t,\ \sin 2t\big),\quad 0 \le t < 2\pi")
        label.next_to(frame, DOWN)
        title = MathTex(r"\textsc{mdbook} \enskip smoke test",
                        color=REFERENCE).scale(0.8).next_to(frame, UP)
        point = Dot(curve.get_start(), color=ACCENT, radius=0.09)

        self.play(FadeIn(VGroup(frame, label, title)), run_time=0.8)
        self.play(Create(curve), MoveAlongPath(point, curve),
                  run_time=4.0, rate_func=linear)
        self.wait(0.5)
