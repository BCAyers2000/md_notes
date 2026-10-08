"""Figure fig:in-stability: the step beyond which velocity Verlet fails.

The spring with m = k = 1 started at q = 1 at rest, stepped 200 times by
velocity Verlet: the largest |q| reached, against ωδt, on a logarithmic
axis. Below ωδt = 2 it stays near 1; above, it grows without limit.

Prints the numbers of Section 9.6.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import integrators, viz
from mdlab.viz import ACCENT, THRESHOLD_STYLE, figure_path


def largest(dt, n=200):
    r, v, f = np.array([[1.0]]), np.array([[0.0]]), None
    worst = 1.0
    for _ in range(n):
        r, v, f = integrators.velocity_verlet_step(
            r, v, [1.0], lambda x: -x, dt, f, 1.0
        )
        worst = max(worst, abs(r.item()))
    return worst


steps = np.linspace(1.0, 2.2, 121)
peaks = np.array([largest(dt) for dt in steps])
for dt in (1.5, 1.9, 1.99, 2.01, 2.05, 2.2):
    print(f"omega dt = {dt}: largest |q| in 200 steps {largest(dt):.3e}")
for dt in (2.05, 2.2):
    tr = 2 - dt**2
    grow = abs(tr / 2) + math.sqrt(tr**2 / 4 - 1)
    print(f"omega dt = {dt}: trace {tr:.4f}, growth per step {grow:.4f}")
print(
    f"water O-H, period 8.88 fs: omega = {2 * math.pi / 8.88:.4f} rad/fs, "
    f"limit dt = {2 / (2 * math.pi / 8.88):.3f} fs, 1/omega = "
    f"{8.88 / (2 * math.pi):.3f} fs"
)

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF, 2.4))
ax.semilogy(steps, peaks, color=ACCENT)
ax.axvline(2.0, **THRESHOLD_STYLE)
ax.set_xlabel(r"$\omega\,\delta t$")
ax.set_ylabel(r"largest $|q|$ in 200 steps")
ax.set_xlim(1.0, 2.2)
top = np.ceil(np.log10(peaks.max()) / 10) * 10
ax.set_yticks(10.0 ** np.arange(0, top + 1, 10))

print("wrote", viz.save(fig, figure_path("ch09_integrators", "stability.pdf")))
