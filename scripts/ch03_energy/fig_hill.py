"""Figure fig:en-hill: partial derivatives and the gradient of a hill.

A hill of height h(x, y) = 300 exp(−(x² + 2y²)/2) metres, with x
pointing east and y north, both in km. (a) Its contours every 50 m, the
gradient ∇h at a grid of points (grey) and at the point P = (0.6, 0.5)
km (teal); the dotted lines run east and north through P. (b) The height
along the east-west line through P, whose slope at P is ∂h/∂x. (c) The
height along the north-south line through P, whose slope at P is ∂h/∂y.

Prints the numbers that Section 3.10 (sec:en-gradient) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE, THRESHOLD_STYLE, figure_path

PEAK = 300.0  # m
P = np.array([0.6, 0.5])  # km


def height(x, y):
    return PEAK * np.exp(-(x**2 + 2 * y**2) / 2)


def gradient(x, y):
    h = height(x, y)
    return -x * h, -2 * y * h  # m per km


hp = height(*P)
gx, gy = gradient(*P)
print(
    f"h(P) = {hp:.1f} m, dh/dx = {gx:.1f} m/km, dh/dy = {gy:.1f} m/km, "
    f"|grad h| = {np.hypot(gx, gy):.1f} m/km"
)

viz.use_style(notebook=False)
fig = plt.figure(figsize=(viz.FULL, 2.9))
grid = fig.add_gridspec(
    2, 2, width_ratios=(1.55, 1.0), wspace=0.3, hspace=0.75
)
ax_map = fig.add_subplot(grid[:, 0])
ax_east = fig.add_subplot(grid[0, 1])
ax_north = fig.add_subplot(grid[1, 1])

x = np.linspace(-2.0, 2.0, 301)
y = np.linspace(-1.5, 1.5, 241)
X, Y = np.meshgrid(x, y)
ax_map.contour(
    X,
    Y,
    height(X, Y),
    levels=np.arange(50, 300, 50),
    colors="black",
    linewidths=0.5,
)
xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 7), np.linspace(-1.0, 1.0, 5))
u, w = gradient(xs, ys)
scale = 450.0  # m/km of gradient per km of arrow
ax_map.quiver(xs, ys, u, w, color=REFERENCE, scale=scale, **viz.ARROW)
ax_map.quiver(*P, gx, gy, color=ACCENT, scale=scale, **viz.ARROW)
ax_map.plot(*P, "o", color=ACCENT, ms=3)
ax_map.text(
    P[0] + 0.1,
    P[1] + 0.08,
    r"$P$",
    ha="left",
    va="bottom",
    bbox=dict(fc="white", ec="none", pad=0.5),
)
ax_map.axhline(P[1], **THRESHOLD_STYLE)
ax_map.axvline(P[0], **THRESHOLD_STYLE)
ax_map.set_aspect("equal")
ax_map.set_xlim(-2.0, 2.0)
ax_map.set_ylim(-1.5, 1.5)
ax_map.set_xlabel(r"east $x$ / km")
ax_map.set_ylabel(r"north $y$ / km")
viz.panel_tag(ax_map, "a")

s = np.linspace(-2.0, 2.0, 200)
near = np.linspace(-0.6, 0.6, 2)  # a short stretch of each tangent
ax_east.plot(s, height(s, P[1]), color="black")
ax_east.plot(P[0] + near, hp + gx * near, color=OCHRE, lw=1.2)
ax_east.plot(P[0], hp, "o", color=ACCENT, ms=3)
ax_east.set_xlim(-2.0, 2.0)
ax_east.set_ylim(0, 320)
ax_east.set_yticks([0, 100, 200, 300])
ax_east.set_xlabel(r"$x$ / km, at $y = 0.5$")
ax_east.set_ylabel(r"height $h$ / m")
ax_east.text(
    1.05,
    230,
    r"slope $\partial h/\partial x$",
    color=OCHRE,
    ha="left",
    va="center",
)
viz.panel_tag(ax_east, "b")

ax_north.plot(s, height(P[0], s), color="black")
ax_north.plot(P[1] + near, hp + gy * near, color=OCHRE, lw=1.2)
ax_north.plot(P[1], hp, "o", color=ACCENT, ms=3)
ax_north.set_xlim(-1.5, 1.5)
ax_north.set_xlabel(r"$y$ / km, at $x = 0.6$")
ax_north.text(
    0.85,
    230,
    r"slope $\partial h/\partial y$",
    color=OCHRE,
    ha="left",
    va="center",
)
ax_north.set_ylim(0, 320)
ax_north.set_yticks([0, 100, 200, 300])
ax_north.set_ylabel(r"height $h$ / m")
viz.panel_tag(ax_north, "c")

print("wrote", viz.save(fig, figure_path("ch03_energy", "hill.pdf")))
