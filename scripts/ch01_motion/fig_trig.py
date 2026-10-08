"""Figure fig:mo-trig: sine and cosine as coordinates on the unit circle.

(a) The point at angle theta on a circle of radius 1: its x coordinate is
cos theta and its y coordinate sin theta; the arc from (1, 0) has length
theta. (b) The same coordinates against theta over one full turn, with
the point of (a) marked.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, THRESHOLD_STYLE, figure_path

THETA = 1.1  # rad, about 63 degrees
c, s = np.cos(THETA), np.sin(THETA)
print(
    f"theta = {THETA} rad = {np.degrees(THETA):.1f} deg; "
    f"cos = {c:.4f}, sin = {s:.4f}, cos^2 + sin^2 = {c * c + s * s:.12f}"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1,
    2,
    figsize=(viz.FULL, 2.55),
    gridspec_kw=dict(width_ratios=(1.25, 1.6), wspace=0.22),
)
phi = np.linspace(0.0, 2 * np.pi, 300)
left.plot(np.cos(phi), np.sin(phi), color="black", lw=0.9)
left.axhline(0.0, color="black", lw=0.4)
left.axvline(0.0, color="black", lw=0.4)
arc = np.linspace(0.0, THETA, 50)
left.plot(
    np.cos(arc), np.sin(arc), color="black", lw=2.4, solid_capstyle="butt"
)
left.plot([0, c], [0, s], color="black", lw=0.9)
left.plot([c, c], [0, s], color=ACCENT, lw=1.6)
left.plot([0, c], [0, 0], color=OCHRE, lw=1.6)
left.plot([0, c], [s, s], **THRESHOLD_STYLE)
left.plot(c, s, "o", color="black", ms=4, zorder=4)
small = np.linspace(0.0, THETA, 30)
left.plot(0.22 * np.cos(small), 0.22 * np.sin(small), color="black", lw=0.6)
left.text(
    0.27 * np.cos(THETA / 2),
    0.27 * np.sin(THETA / 2),
    r"$\theta$",
    va="center",
    ha="left",
    fontsize=9,
)
left.text(
    c + 0.04,
    s / 2,
    r"$\sin\theta$",
    color=ACCENT,
    va="center",
    ha="left",
    fontsize=9,
)
left.text(c / 2, -0.13, r"$\cos\theta$", color=OCHRE, ha="center", fontsize=9)
left.text(
    1.06 * np.cos(THETA * 0.4),
    1.06 * np.sin(THETA * 0.4),
    r"arc $\theta$",
    fontsize=8,
    va="bottom",
    ha="left",
)
left.text(c + 0.04, s + 0.05, r"$P$", fontsize=9)
left.text(-0.08, -0.13, r"$O$", fontsize=9, ha="right")
left.text(0.45 * c - 0.05, 0.45 * s + 0.05, "1", fontsize=8, ha="right")
left.set_aspect("equal")
left.set_xlim(-1.15, 1.4)
left.set_ylim(-1.15, 1.15)
left.set_xticks([-1, 0, 1])
left.spines["bottom"].set_bounds(-1.15, 1.15)
left.set_yticks([-1, 0, 1])
left.set_xlabel(r"$x$")
left.set_ylabel(r"$y$")
viz.panel_tag(left, "a")

right.axhline(0.0, color="black", lw=0.4)
right.plot(phi, np.sin(phi), color=ACCENT, label=r"$\sin\theta$")
right.plot(phi, np.cos(phi), color=OCHRE, label=r"$\cos\theta$")
right.axvline(THETA, **THRESHOLD_STYLE)
right.plot([THETA, THETA], [s, c], "o", color="black", ms=3.5, zorder=4)
viz.pi_ticks(right, 0, 4)
right.set_xlim(0, 2 * np.pi)
right.set_ylim(-1.15, 1.15)
right.set_yticks([-1, 0, 1])
right.set_xlabel(r"angle $\theta$ / rad")
right.legend(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.0),
    ncols=2,
    handlelength=1.4,
    borderaxespad=0.1,
)
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch01_motion", "trig.pdf")))
