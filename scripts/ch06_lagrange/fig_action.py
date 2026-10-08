"""Figure fig:la-action: the action of trial paths of a thrown ball.

The ball of Chapters 1-3, 0.145 kg thrown straight up at 12 m/s, followed
for 2 s, by which time it is 4.387 m up and falling. Trial paths keep both
ends and add ζ sin(πt/T) to the true height. (a) The true path and four
trial paths. (b) The action ∫(K − U) dt of each trial path against ζ,
computed by lagrangian.action: least on the true path, and growing as
ζ², exactly ζ² m π²/(4T) here because U is linear in the height.

Prints the numbers quoted in Section 6.2.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import lagrangian, viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE, figure_path

MASS, V0, T = 0.145, 12.0, 2.0
t = np.linspace(0.0, T, 2001)
true = V0 * t - 0.5 * G * t**2
bump = np.sin(np.pi * t / T)


def weight(x):
    return MASS * G * x


def action(eps):
    return lagrangian.action(true + eps * bump, t, MASS, weight)


s0 = action(0.0)
print(
    "by hand: 72 m T, 12 m g T², m g² T³/3 = "
    f"{72 * MASS * T:.4f}, {12 * MASS * G * T**2:.4f}, "
    f"{MASS * G**2 * T**3 / 3:.4f}"
)
print(f"end height {true[-1]:.3f} m; action of the true path {s0:.4f} J s")
for eps in (-2.0, -1.0, 1.0, 2.0):
    print(
        f"eps = {eps:+.0f} m: action {action(eps):.4f} J s, excess "
        f"{action(eps) - s0:.4f}"
    )
print(
    f"excess per m² predicted m pi²/(4T) = {MASS * math.pi**2 / (4 * T):.5f}"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.5), gridspec_kw=dict(wspace=0.35)
)
for eps, colour in (
    (-2.0, OCHRE),
    (-1.0, REFERENCE),
    (1.0, REFERENCE),
    (2.0, OCHRE),
):
    left.plot(t, true + eps * bump, color=colour, lw=0.9)
left.plot(t, true, color=ACCENT, lw=1.6, label="true path")
left.plot([0, T], [true[0], true[-1]], "o", color="black", ms=3.5)
left.set_xlim(-0.03, T + 0.03)
left.set_xlabel(r"time $t$ / s")
left.set_ylabel(r"height $x$ / m")
left.legend(loc="lower center")
viz.panel_tag(left, "a")

eps_grid = np.linspace(-2.5, 2.5, 101)
right.plot(eps_grid, [action(e) for e in eps_grid], color="black")
right.plot([0.0], [s0], "o", color=ACCENT, ms=4)
right.set_xlim(-2.5, 2.5)
right.set_xlabel(r"$\zeta$ / m")
right.set_ylabel(r"action $\mathcal{S}$ / J\,s")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch06_lagrange", "action.pdf")))
