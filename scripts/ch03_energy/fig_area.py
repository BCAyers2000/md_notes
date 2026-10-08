"""Figure fig:en-area: the work of a varying force is an area.

(a) A force that varies along a line, with the work over each short step
approximated by a thin rectangle of height F(x_k) at the middle of the
step. (b) The force kx that a hand must exert to stretch a spring, whose
work from 0 to x is the triangle under it, kx²/2.

Prints the numbers that Section 3.3 (sec:en-varying) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, figure_path

K_SPRING = 50.0  # N/m, the spring of Section 2.8
STRETCH = 0.04  # m


def force(x):
    """The force of the text, F = 3 + 1.5 sin 1.4x + 0.4x, in N."""
    return 3.0 + 1.5 * np.sin(1.4 * x) + 0.4 * x


X_A, X_B, STEPS = 0.5, 4.5, 8

# Rectangles against the exact area -----------------------------------------
fine = np.linspace(X_A, X_B, 200001)
exact = np.trapezoid(force(fine), fine)
for n in (STEPS, 4 * STEPS, 16 * STEPS):
    edges = np.linspace(X_A, X_B, n + 1)
    middles = 0.5 * (edges[1:] + edges[:-1])
    total = np.sum(force(middles) * np.diff(edges))
    print(f"{n:3d} rectangles: {total:.5f} J (area {exact:.5f} J)")
print(
    f"spring: k x²/2 = {0.5 * K_SPRING * STRETCH**2:.4f} J at x = "
    f"{STRETCH} m, force there {K_SPRING * STRETCH:.1f} N"
)

# The figure ----------------------------------------------------------------
viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.32)
)

edges = np.linspace(X_A, X_B, STEPS + 1)
middles = 0.5 * (edges[1:] + edges[:-1])
left.bar(
    middles,
    force(middles),
    width=np.diff(edges),
    color=ACCENT,
    alpha=0.18,
    edgecolor=ACCENT,
    linewidth=0.6,
)
x = np.linspace(0.0, 5.0, 400)
left.plot(x, force(x), color="black")
left.plot(middles, force(middles), "o", color=ACCENT, ms=3)
k = 5  # label one rectangle
left.annotate(
    r"$F(x_k)\,\Delta x$",
    xy=(middles[k], 0.55 * force(middles[k])),
    xytext=(middles[k] + 0.55, 0.5),
    ha="left",
    arrowprops=dict(arrowstyle="-", lw=0.5, color="black"),
)
left.set_xticks([X_A, X_B], [r"$x_A = 0.5$", r"$x_B = 4.5$"])
left.set_xlim(0.0, 5.0)
left.set_ylim(0.0, 6.2)
left.set_xlabel(r"position $x$ / m")
left.set_ylabel(r"force $F$ / N")
viz.panel_tag(left, "a")

stretch = np.linspace(0.0, 0.05, 100)
right.plot(100 * stretch, K_SPRING * stretch, color="black")
inside = stretch <= STRETCH + 1e-12
right.fill_between(
    100 * stretch[inside], K_SPRING * stretch[inside], color=OCHRE, alpha=0.3
)
right.text(2.75, 0.45, r"$kx^2/2$" "\n" r"$= 0.04$ J", ha="center")
right.plot([100 * STRETCH] * 2, [0, K_SPRING * STRETCH], color=OCHRE, lw=0.8)
right.text(5.0, 2.62, r"$kx$", ha="right", va="bottom")
right.set_xlim(0, 5)
right.set_ylim(0, 2.6)
right.set_xlabel(r"stretch $x$ / cm")
right.set_ylabel(r"force of the hand / N")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch03_energy", "area.pdf")))
