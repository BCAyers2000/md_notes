"""Figure fig:os-damped: the block on a spring with drag.

The block of Section 2.8 (omega0 = 10 rad/s) released from rest 4 cm out
in liquids of three drag rates: gamma = 4 per second (under-damped, with
its envelope ±A e^(-gamma t/2) dotted), 20 per second (critical) and 40
per second (over-damped).

Prints the numbers that Section 4.6 (sec:os-damped) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import oscillators, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

OMEGA0, AMP = 10.0, 0.04
CASES = (
    (4.0, ACCENT, "under-damped"),
    (20.0, OCHRE, "critical"),
    (40.0, OXBLOOD, "over-damped"),
)

t = np.linspace(0.0, 1.5, 1000)
viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.FULL, 2.3))
for gamma, colour, name in CASES:
    x, _ = oscillators.damped(t, AMP, 0.0, OMEGA0, gamma)
    ax.plot(
        t,
        100 * x,
        color=colour,
        label=rf"{name}, $\gamma = {gamma:g}$"
        r" s$^{-1}$",
    )
    first = t[np.flatnonzero(x < 0)[0]] if np.any(x < 0) else None
    at_half = 100 * float(x[np.argmin(abs(t - 0.5))])
    print(
        f"gamma = {gamma}: x(0.5 s) = {at_half:.4f} cm"
        + (f", first crosses zero at {first:.4f} s" if first else "")
    )
envelope = AMP * np.exp(-2.0 * t)
ax.plot(t, 100 * envelope, **THRESHOLD_STYLE)
ax.plot(t, -100 * envelope, **THRESHOLD_STYLE)
ax.axhline(0, color="black", lw=0.4)
ax.set_xlim(0, 1.5)
ax.set_xlabel(r"time $t$ / s")
ax.set_ylabel(r"position $x$ / cm")
ax.legend(loc="upper right")
wd = np.sqrt(OMEGA0**2 - 4.0)
print(
    f"under-damped: omega_d = {wd:.4f} rad/s, period {2 * np.pi / wd:.4f} s;"
    f" envelope halves every {np.log(2) / 2:.4f} s"
)

print("wrote", viz.save(fig, figure_path("ch04_oscillations", "damped.pdf")))
