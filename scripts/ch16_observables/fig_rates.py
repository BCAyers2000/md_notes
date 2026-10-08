"""Figure fig:ob-rates: rates from counted events, with their error bars.

The A-B gas (runs.py, gas_<T>), events of the band that lived at least
0.5 ps, after the first 10 ps. (a) The rate of breaking, per bond and
picosecond, and the rate constant of forming, per picosecond and per
unit of n_A n_B/V, against 1/k_BT, each with the error √n of its count,
and straight lines fitted to their logarithms. (b) At 2000 K, the
numbers of bonds broken in successive windows of 1 ps (bars) against the
Poisson distribution of the same mean (points).

Prints the counts, the rates and their errors, the fitted slopes (the
activation energies), the ratio of the two rates against the
equilibrium constant from the counts of molecules, and the variance and
mean of the counts in 1 ps windows.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch16 import GAS_SIDE, MORSE_ON, RUNS, morse_pair
from scipy.integrate import quad

from mdlab import bonds, units, viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE

TEMPERATURES = (1500, 1750, 2000, 2250, 2500)
SETTLE, LIFE = 10_000.0, 500.0  # fs
V = GAS_SIDE**3
SIZES = {"A": 1, "B": 1}


def bonds_in(name):
    """The bonds of a species: atoms − 1 for the chains the gas forms."""
    atoms = sum(int(part[1:] or 1) for part in
                (name.replace("'", "").replace("A", " A").replace("B", " B")
                 .split()))
    return atoms - 1


rows = []
for temperature in TEMPERATURES:
    run = np.load(RUNS / f"gas_{temperature}.npz")
    t = run["times"]
    late = t > SETTLE
    names = list(run["band_species"])
    counts = run["band_counts"][late].astype(float)
    n_bonds = counts @ np.array([bonds_in(s) for s in names])
    n_a = counts[:, names.index("A")]
    n_b = counts[:, names.index("B")]
    span = (t[late][-1] - t[late][0]) / 1000  # ps
    frame = (t[1] - t[0]) / 1000
    events = [(float(a), int(i), int(j), int(s))
              for a, i, j, s in run["band_events"]]
    kept = [e for e in bonds.persistent(events, LIFE) if e[0] > SETTLE]
    broken = [e[0] for e in kept if e[3] == -1]
    formed = [e[0] for e in kept if e[3] == 1]
    exposure_b = n_bonds.sum() * frame  # bond-picoseconds
    exposure_f = (n_a * n_b / V).sum() * frame  # Å⁻³ ps
    k_break = len(broken) / exposure_b
    k_form = len(formed) / exposure_f
    ratio = np.mean(counts[:, names.index("AB")] * V / (n_a * n_b))
    print(f"{temperature} K: bond-ps over free-pair exposure "
          f"{exposure_b / exposure_f:.0f} Å³")
    rows.append((temperature, len(broken), len(formed), k_break,
                 k_break / math.sqrt(len(broken)), k_form,
                 k_form / math.sqrt(len(formed)), ratio, broken, span))
    print(f"{temperature} K over {span:.0f} ps: {len(broken)} broken over "
          f"{exposure_b:.0f} bond-ps, k_b = {k_break:.4f} ± "
          f"{k_break / math.sqrt(len(broken)):.4f} per ps; {len(formed)} "
          f"formed, k_f = {k_form:.1f} ± {k_form / math.sqrt(len(formed)):.1f}"
          f" Å³/ps; k_f/k_b = {k_form / k_break:.0f} Å³ against "
          f"n_AB V/(n_A n_B) = {ratio:.0f} Å³")

inverse = np.array([1 / (units.KB * r[0]) for r in rows])
for column, label in ((3, "breaking"), (5, "forming")):
    k = np.array([r[column] for r in rows])
    e = np.array([r[column + 1] for r in rows])
    slope, intercept = np.polyfit(inverse, np.log(k), 1, w=k / e)
    print(f"{label}: ln k against 1/k_BT has slope {slope:.3f} eV, "
          f"an activation energy of {-slope:.3f} eV")

k_int = [quad(lambda x, b=b: 4 * np.pi * x * x
              * np.exp(-b * float(morse_pair(np.array(x))[0])),
              0.5, MORSE_ON, limit=200)[0] for b in inverse]
k_pop = np.array([r[7] for r in rows])
slope_int = np.polyfit(inverse, np.log(k_int), 1)[0]
print(f"ln K against 1/k_BT: slope {slope_int:.3f}"
      f" eV for the integral, {np.polyfit(inverse, np.log(k_pop), 1)[0]:.3f} "
      f"eV for the counts of molecules")

run = rows[2]
broken = np.array(run[8])
windows = np.histogram(broken, bins=np.arange(SETTLE, SETTLE + 1000 * run[9]
                                              + 1, 1000.0))[0]
mean, var = windows.mean(), windows.var(ddof=1)
print(f"2000 K: {len(windows)} windows of 1 ps hold {mean:.2f} breaks on "
      f"average with variance {var:.2f} (ratio {var / mean:.2f}); for a "
      f"Poisson count the ratio scatters by about √(2/(n − 1)) = "
      f"{math.sqrt(2 / (len(windows) - 1)):.2f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
k_b = np.array([r[3] for r in rows])
e_b = np.array([r[4] for r in rows])
ax.errorbar(inverse, k_b, e_b, fmt="o", color=ACCENT, ms=3, lw=0.8,
            label=r"breaking / ps$^{-1}$")
k_f = np.array([r[5] for r in rows]) / 1000
e_f = np.array([r[6] for r in rows]) / 1000
ax.errorbar(inverse, k_f, e_f, fmt="s", color=OCHRE, ms=3, lw=0.8,
            label=r"forming / $10^3$ \AA$^3$ ps$^{-1}$")
grid = np.linspace(inverse.min() - 0.2, inverse.max() + 0.2, 20)
for k, e, colour in ((k_b, e_b, ACCENT), (k_f, e_f, OCHRE)):
    slope, intercept = np.polyfit(inverse, np.log(k), 1, w=k / e)
    ax.plot(grid, np.exp(intercept + slope * grid), color=colour, lw=0.6)
ax.set_yscale("log")
ax.set_xlabel(r"$1/k_{\mathrm B}T$ / eV$^{-1}$")
ax.set_ylabel("rate")
ax.legend(fontsize=7, loc="center right")
viz.panel_tag(ax, "a")

ax = axes[1]
values = np.arange(windows.max() + 1)
observed = np.bincount(windows, minlength=len(values)) / len(windows)
poisson = np.exp(-mean) * mean**values / np.array(
    [math.factorial(int(v)) for v in values])
ax.bar(values, observed, color=viz.tint(ACCENT, 0.5), lw=0)
ax.plot(values, poisson, "o", color=REFERENCE, ms=3)
ax.set_xlabel("bonds broken in 1 ps")
ax.set_ylabel("fraction of windows")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "rates.pdf")))
