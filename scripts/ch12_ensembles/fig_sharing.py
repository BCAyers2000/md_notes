"""Figure fig:sm-sharing: how two systems share energy.

Two gases of N atoms each, with the number of states of each growing as
its energy to the power 3N/2 − 1 (Section 12.3), share a total energy E:
the probability that the first holds the fraction x of it, proportional
to x^{3N/2 − 1} (1 − x)^{3N/2 − 1}, for N = 1, 10 and 100, each scaled
to the same height. The most probable division gives equal energies; its
width falls as 1/√N.

Prints the numbers of Section 12.3.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, figure_path, tint

x = np.linspace(1e-6, 1 - 1e-6, 20001)
viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF, 2.3))
for n, strength, style in ((1, 0.65, "--"), (10, 0.8, "-."), (100, 1.0, "-")):
    log = (1.5 * n - 1) * (np.log(x) + np.log(1 - x))
    p = np.exp(log - log.max())
    density = p / np.trapezoid(p, x)
    sd = math.sqrt(np.trapezoid((x - 0.5) ** 2 * density, x))
    exact, curved = 1 / math.sqrt(12 * n + 4), 1 / math.sqrt(12 * n)
    print(f"N = {n}: standard deviation of x {sd:.4f}; exactly "
          f"1/sqrt(12N + 4) = {exact:.4f}; from the large-N curvature of "
          f"ln p at 1/2, 1/sqrt(12N) = {curved:.4f}")
    ax.plot(x, p, color=tint(ACCENT, strength), lw=1.1, ls=style, label=f"$N = {n}$")
for n in (1e3, 1e23):
    print(f"N = {n:.0e}: standard deviation about {1 / math.sqrt(12 * n):.1e}")
ax.set_xlabel("share of the energy, $x = E_1/E$")
ax.set_ylabel("density / peak density")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=3, columnspacing=0.8, handlelength=1.4)

print("wrote", viz.save(fig, figure_path("ch12_ensembles", "sharing.pdf")))
