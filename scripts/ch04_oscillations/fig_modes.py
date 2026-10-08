"""Figure fig:os-modes: two coupled carts and their normal modes.

Two carts of 1 kg between two walls, joined to the walls and to each other
by springs of 100 N/m. (a) Started in the slow mode, both 2 cm to the
right, they swing together at omega = sqrt(k/m) = 10 rad/s. (b) Started in
the fast mode, 2 cm apart symmetrically, they swing against each other at
sqrt(3k/m) = 17.32 rad/s. (c) Started with only the first cart displaced,
4 cm, the motion is the sum of the two modes and the swing passes back and
forth between the carts.

Prints the numbers that Section 4.9 (sec:os-modes) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, oscillators, viz
from mdlab.viz import ACCENT, OCHRE, figure_path

K, M = 100.0, 1.0
HESSIAN = K * np.array([[2.0, -1.0], [-1.0, 2.0]])
w2, modes = oscillators.normal_modes(HESSIAN, [M, M], force_to_accel=1.0)
print("omega =", np.sqrt(w2).round(4), "rad/s; modes (columns):")
print((modes / np.abs(modes).max(axis=0)).round(4))


def force(r):
    return -(HESSIAN @ r[:, 0])[:, None]


t = np.linspace(0.0, 2.0, 2001)
starts = (
    np.array([0.02, 0.02]),
    np.array([0.02, -0.02]),
    np.array([0.04, 0.0]),
)
viz.use_style(notebook=False)
fig, axes = plt.subplots(
    3, 1, figsize=(viz.FULL, 4.2), sharex=True, gridspec_kw=dict(hspace=0.25)
)
for ax, start, tag in zip(axes, starts, "abc", strict=True):
    r, _ = dynamics.solve_newton(
        force, [M, M], start[:, None], np.zeros((2, 1)), t, force_to_accel=1.0
    )
    ax.plot(t, 100 * r[:, 0, 0], color=ACCENT, label="cart 1")
    ax.plot(t, 100 * r[:, 1, 0], color=OCHRE, label="cart 2", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylim(-4.6, 4.6)
    ax.set_ylabel(r"$x$ / cm")
    viz.panel_tag(ax, tag)
axes[0].legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 1.0))
axes[-1].set_xlabel(r"time $t$ / s")
axes[-1].set_xlim(0, 2)
beat = 2 * np.pi / (np.sqrt(3 * K / M) - np.sqrt(K / M))
print(
    f"(c): energy passes from cart 1 to cart 2 and back every "
    f"{beat:.4f} s (2 pi / difference of the two omegas)"
)

print("wrote", viz.save(fig, figure_path("ch04_oscillations", "modes.pdf")))
