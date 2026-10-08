"""Figure fig:os-energy: kinetic and potential energy of the oscillator.

The block of Section 2.8 released from rest 4 cm out, so E = 0.04 J.
(a) K, U and E against time over two periods, with the mean of K and of
U, E/2 (dotted). (b) The energy diagram, U = kx²/2 with the line E; at
x = 2 cm the energy splits into U = 0.01 J and K = 0.03 J.

Prints the numbers that Section 4.2 (sec:os-energy) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import oscillators, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

MASS, K, AMP = 0.5, 50.0, 0.04
OMEGA = np.sqrt(K / MASS)
E = 0.5 * K * AMP**2

t = np.linspace(0.0, 2 * 2 * np.pi / OMEGA, 600)
x, v = oscillators.harmonic(t, AMP, 0.0, OMEGA)
kinetic, potential = 0.5 * MASS * v**2, 0.5 * K * x**2
print(
    f"E = {E:.4f} J, largest |K + U - E| = {np.ptp(kinetic + potential):.1e}"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.3)
)
left.plot(t, 1000 * kinetic, color=OCHRE, label=r"$K$")
left.plot(t, 1000 * potential, color=OXBLOOD, label=r"$U$")
left.plot(t, 1000 * (kinetic + potential), color=ACCENT, label=r"$E$")
left.axhline(1000 * E / 2, **THRESHOLD_STYLE)
left.set_xlim(0, t[-1])
left.set_ylim(0, 52)
left.set_xlabel(r"time $t$ / s")
left.set_ylabel(r"energy / mJ")
left.legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, 1.0))
viz.panel_tag(left, "a")

xs = np.linspace(-0.05, 0.05, 300)
right.plot(100 * xs, 1000 * 0.5 * K * xs**2, color="black")
right.axhline(1000 * E, color=ACCENT)
x_mark = 0.02
u_mark = 0.5 * K * x_mark**2
right.plot([100 * x_mark] * 2, [0, 1000 * u_mark], color=OXBLOOD, lw=2.2)
right.plot([100 * x_mark] * 2, [1000 * u_mark, 1000 * E], color=OCHRE, lw=2.2)
right.text(
    100 * x_mark + 0.2, 1000 * u_mark / 2, r"$U$", color=OXBLOOD, va="center"
)
right.text(
    100 * x_mark + 0.2,
    1000 * (u_mark + E) / 2,
    r"$K$",
    color=OCHRE,
    va="center",
)
right.text(-4.8, 1000 * E + 1.5, r"$E$", color=ACCENT)
right.plot([-4, 4], [1000 * E] * 2, "o", color=ACCENT, ms=3.5)
right.set_xlim(-5, 5)
right.set_ylim(0, 52)
right.set_xlabel(r"position $x$ / cm")
right.set_ylabel(r"energy / mJ")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch04_oscillations", "energy.pdf")))
