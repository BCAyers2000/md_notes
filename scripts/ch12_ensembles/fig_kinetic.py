"""Figure fig:sm-kinetic: the kinetic energy at fixed energy.

(a) The total kinetic energy of the 256 atoms of liquid argon over 50 ps
at fixed total energy (runs.py), divided by its mean: its histogram
against the canonical density of N_f = 765 freedoms with the same mean
(dashed). (b) The relative standard deviation of the kinetic energy at
fixed energy for 108, 256, 500 and 864 atoms, against the canonical
√(2/N_f) (dashed).

Prints the numbers of Section 12.7.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch12 import RUNS

from mdlab import statmech, units, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, figure_path

run = np.load(RUNS / "liquid.npz")
k = run["kinetic"]
n_free = statmech.degrees_of_freedom(256)
kt = 2 * k.mean() / n_free
print(f"liquid: {len(k) - 1} steps; mean K {k.mean():.4f} eV, T "
      f"{kt / units.KB:.2f} K; relative std {k.std() / k.mean():.5f}, "
      f"canonical sqrt(2/N_f) {math.sqrt(2 / n_free):.5f}")
sizes = (108, 256, 500, 864)
rel, canon = [], []
for n in sizes:
    kk = np.load(RUNS / f"size_{n}.npz")["kinetic"]
    nf = statmech.degrees_of_freedom(n)
    rel.append(kk.std() / kk.mean())
    canon.append(math.sqrt(2 / nf))
    print(f"N = {n}: relative std {rel[-1]:.4f}, canonical {canon[-1]:.4f}, "
          f"ratio {rel[-1] / canon[-1]:.3f}")

viz.use_style(notebook=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.4),
                           gridspec_kw=dict(wspace=0.35))
x = k / k.mean()
counts, edges = np.histogram(x, bins=40, range=(0.8, 1.2), density=True)
a.stairs(counts, edges, color=ACCENT, fill=True, alpha=0.35, lw=0)
grid = np.linspace(0.8, 1.2, 400)
dens = statmech.kinetic_energy_density(grid * k.mean(), n_free,
                                       kt / units.KB) * k.mean()
a.plot(grid, dens, **REFERENCE_STYLE, lw=1.0)
a.set_xlabel(r"$K / \overline{K}$")
a.set_ylabel("density")
viz.panel_tag(a, "a")

b.loglog(sizes, rel, "o", color=ACCENT, ms=4, label="fixed energy")
nn = np.array([90, 1000])
b.loglog(nn, np.sqrt(2 / (3 * nn - 3)), **REFERENCE_STYLE, lw=1.0,
         label=r"canonical, $\sqrt{2/N_\mathrm{f}}$")
b.set_xlabel("atoms $N$")
b.set_ylabel(r"std of $K$ / mean of $K$")
b.set_xticks(sizes)
b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
b.xaxis.set_minor_formatter(plt.NullFormatter())
b.set_yticks([0.015, 0.02, 0.03, 0.05, 0.08])
b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
b.yaxis.set_minor_formatter(plt.NullFormatter())
b.legend(fontsize=7)
viz.panel_tag(b, "b")

print("wrote", viz.save(fig, figure_path("ch12_ensembles", "kinetic.pdf")))
