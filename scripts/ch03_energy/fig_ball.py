"""Figure fig:en-ball: the energies of a thrown ball.

A ball of 0.5 kg thrown straight up at 12 m/s (Chapter 1). (a) Kinetic
energy K, potential energy U = mgx and their sum E against time. (b) The
same against height: K and U are straight lines and E is constant.

Prints the numbers that Section 3.4 (sec:en-potential) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

MASS, V0 = 0.5, 12.0  # kg, m/s

flight = 2 * V0 / G
t = np.linspace(0.0, flight, 400)
x = V0 * t - 0.5 * G * t**2
v = V0 - G * t
kinetic = 0.5 * MASS * v**2
potential = MASS * G * x
total = kinetic + potential
top = V0**2 / (2 * G)
print(
    f"flight {flight:.4f} s, top {top:.4f} m at {flight / 2:.4f} s; "
    f"E = {total[0]:.2f} J, largest |E - E(0)| = "
    f"{np.max(np.abs(total - total[0])):.1e} J"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.32)
)

for ax, abscissa in ((left, t), (right, x)):
    ax.plot(abscissa, kinetic, color=OCHRE)
    ax.plot(abscissa, potential, color=OXBLOOD)
    ax.plot(abscissa, total, color=ACCENT)
    ax.set_ylim(0, 42)
    ax.set_ylabel(r"energy / J")

left.text(0.1, 37.5, r"$E = K + U$", color=ACCENT)
left.text(1.35, 3.0, r"$K$", color=OCHRE)
left.text(1.6, 26.0, r"$U$", color=OXBLOOD)
left.axvline(flight / 2, **THRESHOLD_STYLE)
left.set_xlim(0, flight)
left.set_xlabel(r"time $t$ / s")
viz.panel_tag(left, "a")

right.text(0.2, 37.5, r"$E$", color=ACCENT)
right.text(2.3, 29.5, r"$K = E - mgx$", color=OCHRE)
right.text(4.8, 19.5, r"$U = mgx$", color=OXBLOOD, ha="left")
right.axvline(top, **THRESHOLD_STYLE)
right.text(top - 0.1, 10.0, "top", ha="right")
right.set_xlim(0, 8)
right.set_xlabel(r"height $x$ / m")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch03_energy", "ball.pdf")))
