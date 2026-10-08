"""Figure fig:la-bead: the bead on the hill-shaped wire, against the clock.

The wire of Chapter 3, h(x) = 0.15x⁴ − 0.6x² + 0.15x + 1, with the bead
released from rest at its left turning point for E/mg = 1.3 m,
x = −2.206 m. Its motion comes from the Lagrangian equation of Section
6.4, solved by lagrangian.solve_wire. (a) The position along x against
time, with the moments at which the bead passes the bottom of the deep
valley, the top of the hill and the bottom of the shallow valley, and
turns at x = 2 m. (b) Its speed along the wire against x, on top of the
speed √(2g(1.3 − h)) that Chapter 3 found from energy alone.

Prints the times quoted in Section 6.4.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G
from scipy.optimize import brentq

from mdlab import energy, lagrangian, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, THRESHOLD_STYLE, figure_path

H_E = 1.3  # m, E/mg


def shape(x):
    h, slope = energy.hill_track(x)
    return float(h), float(slope), 1.8 * x**2 - 1.2


start = brentq(lambda x: float(energy.hill_track(x)[0]) - H_E, -2.5, -2.0)
print(f"release at x = {start:.4f} m, h = {H_E} m")
t = np.linspace(0.0, 4.2, 42001)
x, v = lagrangian.solve_wire(shape, start, 0.0, t, g=G)
h, slope = energy.hill_track(x)
speed = np.sqrt(1 + slope**2) * np.abs(v)
drift = np.ptp(0.5 * speed**2 + G * h)
print(f"largest change of E/m along the run: {drift:.1e}")

marks = {}
roots = np.sort(np.roots([0.6, 0.0, -1.2, 0.15]).real)  # h' = 0
for name, target in zip(
    ("deep valley", "hill top", "shallow valley"), roots, strict=True
):
    k = np.argmax(x >= target)  # first passage
    when = np.interp(target, x[k - 1 : k + 1], t[k - 1 : k + 1])
    marks[name] = (target, when)
    print(
        f"{name} at x = {target:.4f} m: t = {when:.4f} s, speed "
        f"{speed[k]:.3f} m/s"
    )
turn = np.argmax(v[1:] < 0) + 1  # first time it moves back
print(
    f"turns at x = {x[turn]:.4f} m after {t[turn]:.4f} s; a full swing "
    f"there and back takes {2 * t[turn]:.4f} s"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.5), gridspec_kw=dict(wspace=0.35)
)
stop = t <= 2 * t[turn]
left.plot(t[stop], x[stop], color=ACCENT)
for where, when in marks.values():
    left.plot(when, where, "o", color="black", ms=3)
left.axhline(2.0, **THRESHOLD_STYLE)
left.set_xlim(0, 2 * t[turn])
left.set_xlabel(r"time $t$ / s")
left.set_ylabel(r"position $x$ / m")
viz.panel_tag(left, "a")

grid = np.linspace(start, 2.0, 400)
hg, _ = energy.hill_track(grid)
right.plot(
    grid,
    np.sqrt(2 * G * np.clip(H_E - hg, 0, None)),
    label="from energy",
    **REFERENCE_STYLE,
)
first = t <= t[turn]
right.plot(
    x[first], speed[first], color=ACCENT, lw=0.9, label="from the motion"
)
right.set_xlim(-2.3, 2.1)
right.set_xlabel(r"position $x$ / m")
right.set_ylabel(r"speed / m\,s$^{-1}$")
right.legend(loc="lower center")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch06_lagrange", "bead.pdf")))
