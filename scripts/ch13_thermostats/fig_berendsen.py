"""Figure fig:th-berendsen: what weak coupling does to K, and how fast.

(a) The kinetic energy of the 256 atoms of liquid argon, divided by its
mean, over 95 ps of each of three runs (runs.py) under Berendsen with
τ = 100 fs and under CSVR with τ = 100 fs, against the canonical gamma
density of 765 freedoms (dashed). (b) The liquid with its velocities
scaled to 200 K, run under Berendsen and CSVR (τ = 1 ps), a Nosé-Hoover
chain (τ = 1 ps) and Langevin (γ = 1/ps), over the first 8 ps, against
the target (dotted) and the approach with the time constant τC_V/C_K
predicted for Berendsen (thin).

Prints the numbers of Sections 13.2 and 13.7, including those of the
flying ice cube (runs ice_*), which Fig. 13.8 shows.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch13 import N_FREE, RUNS, T_LIQUID
from scipy.optimize import curve_fit

from mdlab import statmech, units, viz
from mdlab.viz import (
    ACCENT,
    OCHRE,
    OXBLOOD,
    REFERENCE,
    THRESHOLD_STYLE,
    THERMOSTAT,
    figure_path,
)


def pooled(name):
    k = [np.load(RUNS / f"{name}_s{s}.npz")["kinetic"][50:] for s in range(3)]
    return np.concatenate([x / x.mean() for x in k])


spread = {}
for name in ("nve", "berendsen_100", "berendsen_1000", "rescale", "csvr_100"):
    if name == "rescale":
        k = np.load(RUNS / "rescale_s0.npz")["kinetic"][50:]
        x = k / k.mean()
    else:
        x = pooled(name)
    spread[name] = x.std()
    print(f"{name}: relative spread of K {x.std():.4f}")
print(f"canonical sqrt(2/N_f) = {math.sqrt(2 / N_FREE):.4f}")
for name in ("berendsen_100", "berendsen_1000", "nh_100", "csvr_100"):
    temps = [2 * np.load(RUNS / f"{name}_s{s}.npz")["kinetic"][50:].mean()
             / (N_FREE * units.KB) for s in range(3)]
    print(f"{name}: mean T over three runs {np.mean(temps):.2f} K, from "
          f"{min(temps):.2f} to {max(temps):.2f}")

growth = {}
for name in ("berendsen", "rescale", "csvr"):
    r = np.load(RUNS / f"ice_{name}.npz")
    m, v = r["masses"], r["velocities"]
    p = np.einsum("i,tix->tx", m, v)
    k_com = 0.5 * units.MV2_TO_EV * np.sum(p * p, 1) / m.sum()
    t_v = r["times"][::5] / 1000
    growth[name] = (t_v, k_com)
    slope = np.polyfit(t_v, np.log(k_com), 1)[0]
    print(f"ice, {name}: K of the centre of mass {k_com[0]:.4f} eV at the "
          f"start, {k_com[-1]:.4f} eV after {t_v[-1]:.0f} ps; growth rate of "
          f"its logarithm {slope:.4f}/ps")
k = np.load(RUNS / "berendsen_100_s0.npz")["kinetic"][50:]
x2 = np.mean((k / k.mean() - 1) ** 2)
predicted = x2 / 0.1
print(f"Berendsen, tau 0.1 ps: <(dT/T0)^2> = {x2:.2e}, predicted rate "
      f"<(dT/T0)^2>/tau = {predicted:.4f}/ps")
lam2 = 1 + 0.1 * (1 / (1 + (k / k.mean() - 1)) - 1)  # dt/tau = 0.1
print(f"with every term of ln(lambda^2): {np.mean(np.log(lam2)) / 0.01:.5f}"
      f"/ps; first order times (1 - dt/2tau): {predicted * 0.95:.5f}/ps")

viz.use_style(notebook=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.4),
                           gridspec_kw=dict(wspace=0.35))
bins = np.linspace(0.8, 1.2, 41)
for name, colour, label in (("berendsen_100", THERMOSTAT["Berendsen"], r"Berendsen, 100 fs"),
                            ("csvr_100", THERMOSTAT["CSVR"], r"CSVR, 100 fs")):
    a.hist(pooled(name), bins=bins, density=True, histtype="step",
           color=colour, lw=1.0, label=label)
grid = np.linspace(0.8, 1.2, 400)
a.plot(grid, 0.5 * N_FREE * statmech.kinetic_energy_density(
    grid * 0.5 * N_FREE, N_FREE, 1.0, kb=1.0), color=REFERENCE, ls="--",
    lw=1.0, label="canonical")
a.set_xlabel(r"$K/\overline{K}$")
a.set_ylabel("density")
a.set_ylim(0, 22)
a.legend(fontsize=6.5, loc="upper left")
viz.panel_tag(a, "a")

C_RATIO = 2.376 / 1.4941  # C_V/C_K of the liquid, Section 12.7
print(f"C_V/C_K = {C_RATIO:.3f}; predicted time constants: Berendsen and "
      f"CSVR (tau 1 ps) {C_RATIO:.2f} ps, Langevin (1/ps) "
      f"{C_RATIO / 2:.2f} ps")


def decay(t, amplitude, tau, final):
    return final + amplitude * np.exp(-(t - 0.5) / tau)


fitted = None
for name, label, colour in (("berendsen", "Berendsen", THERMOSTAT["Berendsen"]),
                            ("csvr", "CSVR", THERMOSTAT["CSVR"]),
                            ("nhc", "Nosé-Hoover chain", THERMOSTAT["Nosé-Hoover"]),
                            ("langevin", "Langevin", THERMOSTAT["Langevin"])):
    r = np.load(RUNS / f"hot_{name}.npz")
    t = r["times"] / 1000
    temp = 2 * r["kinetic"] / (N_FREE * units.KB)
    late = t > 0.5
    p, cov = curve_fit(decay, t[late], temp[late], p0=(30.0, 1.5, 135.0))
    print(f"hot start, {label}: T {temp[0]:.0f} K at the start, "
          f"{temp[np.searchsorted(t, 0.2)]:.1f} K at 0.2 ps; fitted from "
          f"0.5 ps, time constant {p[1]:.2f} ± {np.sqrt(cov[1, 1]):.2f} ps;"
          f" first below 140 K at "
          f"{t[np.argmax(temp < 140)]:.1f} ps; mean over the last 10 ps "
          f"{temp[t > 10].mean():.1f} K")
    if name == "berendsen":
        fitted = p
    b.plot(t, temp, color=colour, lw=0.7, label=label,
           ls={"berendsen": "-", "csvr": "--", "nhc": "-.", "langevin": ":"}[name])
t_fit = np.linspace(0.5, 8, 200)
b.plot(t_fit, decay(t_fit, fitted[0], C_RATIO, T_LIQUID), color=THERMOSTAT["Berendsen"],
       lw=0.4)
b.axhline(T_LIQUID, **THRESHOLD_STYLE)
b.set_xlabel("time / ps")
b.set_ylabel(r"$T$ / K")
b.set_xlim(0, 8)
b.set_ylim(110, 205)
legend = b.legend(fontsize=6.5, loc="upper right")
for handle in legend.legend_handles:
    handle.set_linewidth(1.5)
viz.panel_tag(b, "b")

print("wrote", viz.save(fig, figure_path("ch13_thermostats",
                                         "berendsen.pdf")))
