"""Figure fig:sm-healthy: a healthy run, read through its temperature.

The 256 atoms of liquid argon of runs.py at fixed energy, 50 ps with
steps of 10 fs. (a) The kinetic temperature against time, with its mean
(dotted). (b) The running averages of the temperatures of the x, y and z
components of the velocities. (c) The excess kurtosis of the velocities
every 50 fs, zero for the Maxwell-Boltzmann distribution, with twice its
standard deviation over the run either side of zero (dotted).

Prints the numbers of Section 12.9.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch12 import RUNS

from mdlab import diagnostics, statmech, viz
from mdlab.viz import ACCENT, CYCLE, THRESHOLD_STYLE, figure_path

run = np.load(RUNS / "liquid.npz")
m, v = run["masses"], run["velocities"]
t, ft = run["times"], run["frame_times"]
n_free = statmech.degrees_of_freedom(len(m))
full = 2 * run["kinetic"] / n_free / statmech.KB  # every step
print(f"run of {t[-1] / 1000:g} ps; T = {full.mean():.2f} K, standard "
      f"deviation {full.std():.2f} K ({full.std() / full.mean():.2%}), "
      f"range {full.min():.1f} to {full.max():.1f} K")
print(f"canonical spread would be sqrt(2/N_f) = {math.sqrt(2 / n_free):.2%}")
directions = [statmech.part_temperature(m, v, axis=a) for a in range(3)]
for name, series in zip("xyz", directions, strict=True):
    print(f"T_{name} over the run {series.mean():.2f} K")
mean_directions = np.mean([s.mean() for s in directions])
print(f"mean of the three {mean_directions:.2f} K; 765/768 of the kinetic "
      f"temperature of the saved frames "
      f"{765 / 768 * statmech.kinetic_temperature(m, v, n_free).mean():.2f} K")
kurt = np.array([diagnostics.excess_kurtosis(m, frame) for frame in v])
band = 2 * kurt.std()
print(f"excess kurtosis: mean {kurt.mean():+.3f}, standard deviation "
      f"{kurt.std():.3f} over {len(kurt)} frames of {v[0].size} "
      f"components; frames beyond twice it "
      f"{np.mean(np.abs(kurt) > band):.1%}")
print(f"total energy fluctuation "
      f"{diagnostics.energy_fluctuation(run['potential'], run['kinetic']):.4f}"
      f" of the kinetic energy's")

viz.use_style(notebook=False)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(viz.FULL, 2.3),
                              gridspec_kw=dict(wspace=0.55))
a.plot(t / 1000, full, color=ACCENT, lw=0.4)
a.axhline(full.mean(), **THRESHOLD_STYLE)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ / K")
viz.panel_tag(a, "a")
counts = np.arange(1, len(ft) + 1)
for name, series, colour in zip("xyz", directions, CYCLE[:3],
                                strict=True):
    b.plot(ft / 1000, np.cumsum(series) / counts, color=colour, lw=0.9,
           label=f"$T_{name}$")
b.axhline(full.mean(), **THRESHOLD_STYLE)
b.set_ylim(full.mean() - 25, full.mean() + 25)
b.set_xlabel("time / ps")
b.set_ylabel("running average / K")
b.legend(fontsize=7, ncol=3, loc="upper center", columnspacing=0.6,
         handlelength=1.2)
viz.panel_tag(b, "b")
c.plot(ft / 1000, kurt, color=ACCENT, lw=0.4)
for level in (band, -band):
    c.axhline(level, **THRESHOLD_STYLE)
c.set_ylim(-1.3, 1.3)
c.set_xlabel("time / ps")
c.set_ylabel("excess kurtosis")
viz.panel_tag(c, "c")

print("wrote", viz.save(fig, figure_path("ch12_ensembles", "healthy.pdf")))
