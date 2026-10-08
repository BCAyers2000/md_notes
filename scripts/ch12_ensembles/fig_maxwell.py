"""Figure fig:sm-maxwell: velocities and kinetic energies in liquid argon.

The 256 atoms of liquid argon of runs.py over 50 ps, every 50 fs, at their
kinetic temperature T. (a) The histogram of every velocity component,
in Å/ps, against the Gaussian of variance k_BT/m. (b) The histogram of
the speeds against the Maxwell-Boltzmann density. (c) The histogram of
single atoms' kinetic energies against the density
(2/√π)(k_BT)^{−3/2} √K e^{−K/k_BT} of three freedoms.

Prints the numbers of Section 12.5.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch12 import MASS, RUNS

from mdlab import diagnostics, statmech, units, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, figure_path

run = np.load(RUNS / "liquid.npz")
m, v = run["masses"], run["velocities"]
n_free = statmech.degrees_of_freedom(len(m))
t = statmech.kinetic_temperature(m, v, n_free).mean()
sigma = math.sqrt(units.KB * t / (MASS * units.MV2_TO_EV))
comps = v.reshape(-1)
speeds = np.linalg.norm(v, axis=-1).reshape(-1)
energy = 0.5 * units.MV2_TO_EV * MASS * speeds**2
print(f"T = {t:.2f} K; sigma = sqrt(k_B T/m) = {sigma:.6f} Å/fs; measured "
      f"standard deviation of the components {comps.std():.6f} Å/fs")
share = math.sqrt(765 / 768)
print(f"components share 765 freedoms over 768: sqrt(765/768) = "
      f"{share:.5f}; times sigma {share * sigma:.6f}")
print(f"{comps.size} components; excess kurtosis "
      f"{diagnostics.excess_kurtosis(m, v):+.4f}")
mean_speed = math.sqrt(8 / math.pi) * sigma
print(f"mean speed: Maxwell-Boltzmann {mean_speed:.6f}, measured "
      f"{speeds.mean():.6f} Å/fs; most probable sqrt(2) sigma "
      f"{math.sqrt(2) * sigma:.6f} Å/fs")
kt = units.KB * t
print(f"single-atom kinetic energy: mean {energy.mean() / kt:.4f} k_BT "
      f"(3/2 expected), variance {energy.var() / kt**2:.4f} (k_BT)^2 "
      f"(3/2 expected)")
for species, mass in (("Li", 6.94), ("Ar", 39.948)):
    s = math.sqrt(units.KB * 300 / (mass * units.MV2_TO_EV))
    print(f"{species} at 300 K: sigma {s:.5f} Å/fs, mean speed "
          f"{math.sqrt(8 / math.pi) * s:.5f} Å/fs")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.3),
                         gridspec_kw=dict(wspace=0.45))
a, b, c = axes
ps = 1000.0  # Å/fs to Å/ps, for readable axes
counts, edges = np.histogram(comps * ps, bins=40,
                             range=(-5 * sigma * ps, 5 * sigma * ps),
                             density=True)
a.stairs(counts, edges, color=ACCENT, fill=True, alpha=0.35, lw=0)
x = np.linspace(-5 * sigma, 5 * sigma, 400)
a.plot(x * ps, np.exp(-x**2 / (2 * sigma**2))
       / (sigma * ps * math.sqrt(2 * math.pi)), **REFERENCE_STYLE, lw=1.0)
a.set_xlabel(r"$v_x$ / Å ps$^{-1}$")
a.set_ylabel("density / ps Å$^{-1}$")
viz.panel_tag(a, "a")

counts, edges = np.histogram(speeds * ps, bins=40, range=(0, 5 * sigma * ps),
                             density=True)
b.stairs(counts, edges, color=ACCENT, fill=True, alpha=0.35, lw=0)
s = np.linspace(0, 5 * sigma, 400)
b.plot(s * ps, statmech.maxwell_speed_density(s, MASS, t) / ps,
       **REFERENCE_STYLE, lw=1.0)
b.set_xlabel(r"speed / Å ps$^{-1}$")
viz.panel_tag(b, "b")

counts, edges = np.histogram(energy / kt, bins=40, range=(0, 8),
                             density=True)
c.stairs(counts, edges, color=ACCENT, fill=True, alpha=0.35, lw=0)
e = np.linspace(1e-6, 8, 400)
c.plot(e, statmech.kinetic_energy_density(e, 3, 1.0, kb=1.0),
       **REFERENCE_STYLE, lw=1.0)
c.set_xlabel(r"$K_i / k_\mathrm{B}T$")
viz.panel_tag(c, "c")

print("wrote", viz.save(fig, figure_path("ch12_ensembles", "maxwell.pdf")))
