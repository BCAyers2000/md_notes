"""Figure fig:ne-carts: two carts pushed apart by a compressed spring.

Carts of 1 kg and 3 kg rest on a level, smooth track with a spring of
stiffness 200 N/m compressed by 0.1 m between them. The spring is not
fixed to either cart, so it falls away once it reaches its natural
length. (a) Positions of the carts and of their centre of mass. (b)
Momenta of the carts and their sum, which stays zero.

Prints the numbers that Section 2.9 (sec:ne-bodies) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, viz
from mdlab.viz import (
    ACCENT,
    OCHRE,
    REFERENCE,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

MASSES = np.array([1.0, 3.0])  # kg
STIFFNESS = 200.0  # N/m
NATURAL_LENGTH = 0.30  # m, the gap at which the spring falls away
STARTING_GAP = 0.20  # m, so the spring starts compressed by 0.1 m


def spring_push(r):
    """Force of the spring on each cart; it can push but not pull."""
    gap = r[1, 0] - r[0, 0]
    push = STIFFNESS * max(NATURAL_LENGTH - gap, 0.0)
    return np.array([[-push], [+push]])  # backwards on 1, forwards on 2


t = np.linspace(0.0, 0.6, 1201)
r0 = np.array([[0.0], [STARTING_GAP]])  # one row per cart, one column (x)
r, v = dynamics.solve_newton(
    spring_push, MASSES, r0, np.zeros_like(r0), t, force_to_accel=1.0
)
x = r[:, :, 0]  # (time, cart)
p = dynamics.momentum(MASSES, v)[:, :, 0]
centre = dynamics.centre_of_mass(MASSES, r)[:, 0]
total = dynamics.total_momentum(MASSES, v)[:, 0]

print(
    f"final velocities {v[-1, 0, 0]:+.4f} and {v[-1, 1, 0]:+.4f} m/s "
    f"(ratio {v[-1, 0, 0] / v[-1, 1, 0]:.4f}); momenta "
    f"{p[-1, 0]:+.4f} and {p[-1, 1]:+.4f} kg m/s"
)
print(
    f"largest |P| = {np.max(np.abs(total)):.1e} kg m/s; centre of mass at "
    f"{centre[0]:.4f} m, moves at most {np.ptp(centre):.1e} m"
)
# The spring falls away at the first time the gap reaches its natural
# length. While it pushes, the pair oscillates with the reduced mass
# m1 m2 / (m1 + m2), and that takes a quarter of a period.
gap = x[:, 1] - x[:, 0]
released = t[np.flatnonzero(gap >= NATURAL_LENGTH - 1e-12)[0]]
reduced_mass = MASSES.prod() / MASSES.sum()
quarter_period = 0.5 * np.pi * np.sqrt(reduced_mass / STIFFNESS)
print(
    f"spring falls away at t = {released:.4f} s "
    f"(a quarter period of the pair: {quarter_period:.4f} s)"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.35)
)

left.plot(t, x[:, 0], color=OCHRE)
left.plot(t, x[:, 1], color=ACCENT)
left.plot(t, centre, **REFERENCE_STYLE)
left.text(0.59, x[-1, 1] + 0.05, "cart 2, 3 kg", color=ACCENT, ha="right")
left.text(0.2, -0.11, "cart 1, 1 kg", color=OCHRE, ha="left")
left.text(
    0.59, centre[-1] + 0.05, "centre of mass", color=REFERENCE, ha="right"
)
left.axvline(released, **THRESHOLD_STYLE)
left.set_xlim(0, 0.6)
left.set_xlabel(r"time $t$ / s")
left.set_ylabel(r"position $x$ / m")
viz.panel_tag(left, "a")

right.axhline(0.0, color="black", lw=0.4)
right.plot(t, p[:, 0], color=OCHRE, label=r"cart 1, $p_1$")
right.plot(t, p[:, 1], color=ACCENT, label=r"cart 2, $p_2$")
right.plot(t, total, label=r"$P = p_1 + p_2$", **REFERENCE_STYLE)
right.axvline(released, **THRESHOLD_STYLE)
right.set_xlim(0, 0.6)
right.set_ylim(-1.6, 1.6)
right.set_xlabel(r"time $t$ / s")
right.set_ylabel(r"momentum $p$ / kg\,m\,s$^{-1}$")
right.legend(loc="center right", bbox_to_anchor=(1.0, 0.67))
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch02_newton", "carts.pdf")))
