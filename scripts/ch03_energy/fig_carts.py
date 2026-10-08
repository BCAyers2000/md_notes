"""Figure fig:en-carts: the energies of the two carts of Chapter 2.

Carts of 1 kg and 3 kg on a level, smooth track are pushed apart by a
spring of 200 N/m compressed by 0.1 m (fig:ne-carts). The kinetic
energies of the two carts, the energy stored in the spring, ½ k ξ² for a
compression ξ, and their total, against time. The spring falls away
when it reaches its natural length.

Prints the numbers that Section 3.8 (sec:en-bodies) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import dynamics, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

MASSES = np.array([1.0, 3.0])  # kg
STIFFNESS = 200.0  # N/m
NATURAL_LENGTH = 0.30  # m
STARTING_GAP = 0.20  # m


def compression(r):
    """How far the spring is squashed; zero once it has fallen away."""
    return np.maximum(NATURAL_LENGTH - (r[..., 1, 0] - r[..., 0, 0]), 0.0)


def spring_push(r):
    push = STIFFNESS * compression(r)
    return np.array([[-push], [+push]])


t = np.linspace(0.0, 0.2, 801)
r0 = np.array([[0.0], [STARTING_GAP]])
r, v = dynamics.solve_newton(
    spring_push, MASSES, r0, np.zeros_like(r0), t, force_to_accel=1.0
)
kinetic = 0.5 * MASSES * v[:, :, 0] ** 2  # (time, cart)
stored = 0.5 * STIFFNESS * compression(r) ** 2
total = kinetic.sum(axis=1) + stored
released = t[np.flatnonzero(compression(r) == 0.0)[0]]
print(
    f"stored at start {stored[0]:.4f} J; final K1 = {kinetic[-1, 0]:.4f} J, "
    f"K2 = {kinetic[-1, 1]:.4f} J; total varies by "
    f"{np.ptp(total):.1e} J; spring falls away at {released:.4f} s; "
    f"final speeds {v[-1, 0, 0]:+.4f}, {v[-1, 1, 0]:+.4f} m/s"
)

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.FULL, 2.3))
ax.plot(t, total, color=ACCENT)
ax.plot(t, stored, color=OXBLOOD)
ax.plot(t, kinetic[:, 0], color=OCHRE)
ax.plot(t, kinetic[:, 1], color=OCHRE, ls="--")
ax.axvline(released, **THRESHOLD_STYLE)
ax.text(0.198, 1.04, "total", color=ACCENT, ha="right", va="bottom")
ax.text(0.03, 0.9, "spring", color=OXBLOOD, ha="left")
ax.text(0.198, 0.77, "cart 1, 1 kg", color=OCHRE, ha="right", va="bottom")
ax.text(0.198, 0.27, "cart 2, 3 kg", color=OCHRE, ha="right", va="bottom")
ax.set_xlim(0, 0.2)
ax.set_ylim(0, 1.2)
ax.set_xlabel(r"time $t$ / s")
ax.set_ylabel(r"energy / J")

print("wrote", viz.save(fig, figure_path("ch03_energy", "carts.pdf")))
