"""Figure fig:cv-healthy: the dashboard of a converged quantity.

The volume of the 1 ns run at 0.1 GPa that ends the protocol (runs.py,
protocol), drawn by ch15.dashboard: (a) the series with the start found
(dotted) and the mean after it (dashed); (b) the running mean from the
start with its error band; (c) the error from blocks, with √(g s²/n)
(dashed); (d) the autocorrelation function, with the lag at which the
estimate of g stops (dotted).

Prints the report.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch15 import DT, RUNS, dashboard

from mdlab import viz
from mdlab.analysis import stats

run = np.load(RUNS / "protocol.npz")
volume = run["npt_volume"]
report = stats.convergence_report(volume, DT)
print(report)

viz.use_style(notebook=False)
fig, axes = plt.subplots(2, 2, figsize=(viz.FULL, 3.4),
                         gridspec_kw=dict(wspace=0.35, hspace=0.75))
axes = axes.ravel()
dashboard(axes, volume, DT, report, unit=r"$V$ / \AA$^3$")
axes[1].set_ylabel(r"running mean of $V$ / \AA$^3$")
axes[2].set_ylabel(r"standard error / \AA$^3$")
for ax, tag in zip(axes, "abcd", strict=True):
    viz.panel_tag(ax, tag)
print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "healthy.pdf")))
