"""Figure fig:sm-probability: frequencies settle on probabilities.

(a) The total of two dice: the fraction of throws giving each total after
100 and after 10 000 throws, against the probability (6 − |s − 7|)/36.
(b) 1000 numbers drawn from the standard normal distribution: their
histogram, as a density, against the Gaussian e^{−x²/2}/√(2π).

Prints the numbers of Section 12.1.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE, REFERENCE_STYLE, figure_path

rng = np.random.default_rng(12)
totals = np.arange(2, 13)
exact = (6 - np.abs(totals - 7)) / 36
print("P(7) =", exact[5], "= 6/36")
freq = {}
for n in (100, 10000):
    throws = rng.integers(1, 7, (n, 2)).sum(1)
    freq[n] = np.array([np.mean(throws == s) for s in totals])
    print(f"{n} throws: largest gap from the probabilities "
          f"{np.abs(freq[n] - exact).max():.4f}; mean total "
          f"{throws.mean():.3f}, variance {throws.var():.3f}")
mean = float(totals @ exact)
var = float(((totals - mean) ** 2) @ exact)
print(f"exact: mean {mean:.4f}, variance {var:.4f} = 35/6 = {35 / 6:.4f}")

x = rng.standard_normal(1000)
print(f"1000 normal numbers: mean {x.mean():+.4f}, standard deviation "
      f"{x.std():.4f}")
counts, edges = np.histogram(x, bins=24, range=(-4, 4), density=True)
centres = 0.5 * (edges[1:] + edges[:-1])
gauss = np.exp(-centres**2 / 2) / math.sqrt(2 * math.pi)
print(f"largest gap of the histogram from the Gaussian "
      f"{np.abs(counts - gauss).max():.4f} per unit")

viz.use_style(notebook=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.4),
                           gridspec_kw=dict(wspace=0.35))
a.bar(totals, exact, width=0.8, color=REFERENCE, alpha=0.25, lw=0,
      label="probability")
a.plot(totals, freq[100], "o", ms=4, mfc="white", color=OCHRE,
       label="100 throws")
a.plot(totals, freq[10000], "o", ms=3, color=ACCENT, label="10 000 throws")
a.set_xlabel("total of two dice")
a.set_ylabel("fraction of throws")
a.set_xticks(totals)
a.set_ylim(0, 0.24)
a.legend(fontsize=7, loc="upper right")
viz.panel_tag(a, "a")

b.stairs(counts, edges, color=ACCENT, fill=True, alpha=0.35, lw=0)
b.stairs(counts, edges, color=ACCENT, lw=0.8)
grid = np.linspace(-4, 4, 400)
b.plot(grid, np.exp(-grid**2 / 2) / math.sqrt(2 * math.pi),
       **REFERENCE_STYLE, lw=1.0)
b.set_xlabel(r"$x$")
b.set_ylabel("density")
viz.panel_tag(b, "b")

print("wrote", viz.save(fig, figure_path("ch12_ensembles",
                                         "probability.pdf")))
