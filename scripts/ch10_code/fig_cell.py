"""Figure fig:md-cell: rounding and the nearest image in a skewed cell.

The hexagonal cell of graphite (a = 2.464 Å, c = 6.711 Å, Trucano and
Chen 1975), seen in the plane of its layers. (a) Separations d whose
fractional coordinates lie in [−½, ½], the parallelogram that rounding
returns, shaded where a shorter copy exists; the hexagon of separations
that are their own nearest copy; and the circle of radius w/2, half the
perpendicular width, inside both. (b) For separations that are their own
nearest copy, the fraction that rounding returns wrongly, against their
length.

Prints the numbers of Section 10.2.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import cell, viz
from mdlab.viz import (
    ACCENT,
    OXBLOOD,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
    tint,
)

A, C = 2.464, 6.711
H = np.array([[A, 0, 0], [-A / 2, A * math.sqrt(3) / 2, 0], [0, 0, C]]).T

widths = cell.perpendicular_widths(H)
half = 0.5 * widths.min()
print(
    f"widths {widths.round(3)} Å; half the smallest {half:.3f} Å; "
    f"volume {cell.cell_volume(H):.2f} Å³"
)
for r_cut in (10.0, 12.0):
    n = np.ceil(2 * r_cut / widths).astype(int)
    print(
        f"r_c = {r_cut} Å needs at least {n} cells, "
        f"{4 * int(np.prod(n))} atoms"
    )

# (a) a grid over the rounding parallelogram
s1, s2 = np.meshgrid(np.linspace(-0.5, 0.5, 401), np.linspace(-0.5, 0.5, 401))
frac = np.column_stack([s1.ravel(), s2.ravel(), np.zeros(s1.size)])
d = cell.to_cartesian(frac, H)
best = cell.nearest_image(d, H)
longer = np.linalg.norm(d, axis=1) > np.linalg.norm(best, axis=1) + 1e-9
print(
    f"fraction of the parallelogram where rounding is not nearest: "
    f"{longer.mean():.4f}"
)

# (b) separations that are their own nearest copy, by length
rng = np.random.default_rng(10)
radii = np.linspace(0.0, 0.99 * A / math.sqrt(3), 100)
missed = []
for rho in radii:
    angle = rng.uniform(0, 2 * np.pi, 20000)
    pts = np.column_stack(
        [rho * np.cos(angle), rho * np.sin(angle), np.zeros_like(angle)]
    )
    own = cell.nearest_image(pts, H)
    pts = pts[np.all(np.isclose(own, pts, atol=1e-9), axis=1)]
    rounded = cell.minimum_image(pts, H)
    missed.append(np.mean(np.linalg.norm(rounded - pts, axis=1) > 1e-9))
missed = np.array(missed)
first = radii[np.argmax(missed > 0)]
print(
    f"rounding first misses at |d| = {first:.3f} Å "
    f"(half the width {half:.3f} Å); at the hexagon's corner "
    f"{A / math.sqrt(3):.3f} Å it misses {missed[-1]:.2f} of them"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1,
    2,
    figsize=(viz.FULL, 2.8),
    gridspec_kw=dict(wspace=0.3, width_ratios=[1.3, 1]),
)
left.scatter(
    d[longer, 0],
    d[longer, 1],
    s=0.3,
    color=tint(OXBLOOD, 0.35),
    rasterized=True,
    linewidths=0,
)
corners = cell.to_cartesian(
    np.array(
        [
            [-0.5, -0.5, 0],
            [0.5, -0.5, 0],
            [0.5, 0.5, 0],
            [-0.5, 0.5, 0],
            [-0.5, -0.5, 0],
        ]
    ),
    H,
)
left.plot(corners[:, 0], corners[:, 1], color="black", lw=0.9)
left.text(1.95, -1.3, "rounding", fontsize=7, ha="center", va="top")
ang = np.radians(30 + 60 * np.arange(7))
left.plot(
    A / math.sqrt(3) * np.cos(ang),
    A / math.sqrt(3) * np.sin(ang),
    color=ACCENT,
    lw=1.2,
)
left.text(
    0.6,
    1.52,
    "nearest copy",
    color=ACCENT,
    fontsize=7,
    ha="center",
    va="bottom",
)
t = np.linspace(0, 2 * np.pi, 200)
left.plot(half * np.cos(t), half * np.sin(t), **REFERENCE_STYLE, lw=0.9)
left.annotate(
    r"$w_{\min}/2$",
    xy=(-half * 0.71, -half * 0.71),
    xytext=(-2.3, -1.45),
    fontsize=8,
    color="0.3",
    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6),
)
for vec, name in ((H[:, 0], r"$\mathbf{a}$"), (H[:, 1], r"$\mathbf{b}$")):
    left.annotate(
        "",
        xy=vec[:2],
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.8),
    )
    left.text(*(1.08 * vec[:2]), name, ha="center", va="center", fontsize=8)
left.set_aspect("equal")
left.set_xlabel(r"$x$ / Å")
left.set_ylabel(r"$y$ / Å")
left.set_xlim(-2.6, 2.9)
left.set_ylim(-1.75, 2.45)
viz.panel_tag(left, "a")

right.plot(radii, missed, color=OXBLOOD, lw=1.0)
right.axvline(half, **THRESHOLD_STYLE)
right.text(half - 0.03, 0.5, r"$w_{\min}/2$", ha="right", fontsize=8)
right.set_xlabel(r"length of the nearest copy / Å")
right.set_ylabel("fraction missed by rounding")
right.set_xlim(0, A / math.sqrt(3))
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch10_code", "cell.pdf")))
