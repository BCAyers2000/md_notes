"""Figure fig:ob-vacf: velocities that remember, and the viscosity.

(a) The velocity autocorrelation function C_vv(t)/C_vv(0) of liquid
argon at fixed energy (runs.py, argon_nve) and of the fcc crystal at
21 K (crystal_nve). (b) The Green-Kubo integral of the shear stress,
(V/k_BT) ∫⟨P_xy(0)P_xy(t)⟩dt, from the 14 ns of Chapter 15's runs near
135 K (the two 2 ns runs and the ten 1 ns runs of five sizes, the three
off-diagonal components of each), with twice its standard error over
the 12 runs (band), against the viscosity that the Yeh-Hummer slope
implied in Section 15.7 (dashed, with its error).

Prints the first zero, the least value and its lag for the liquid; the
crystal's first zero; the plateau of the viscosity over 1-3 ps with its
error, in Pa s and reduced units, and its distance from Yeh-Hummer's.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import RUNS, ch12, ch15

from mdlab import units, viz
from mdlab.analysis import transport
from mdlab.viz import ACCENT, OCHRE, REFERENCE

DT = 10.0
PA_S = units.ELEMENTARY_CHARGE / units.ANGSTROM**3 * units.FEMTOSECOND
#: The reduced unit of viscosity √(mε)/σ², in Pa s.
REDUCED = np.sqrt(ch12.MASS * units.AMU * ch12.EPS * units.ELEMENTARY_CHARGE) \
    / (ch12.SIG * units.ANGSTROM) ** 2
YH, YH_ERR = 2.18e-4, 0.20e-4

liquid = np.load(RUNS / "argon_nve.npz")
crystal = np.load(RUNS / "crystal_nve.npz")
c_liq = transport.velocity_autocorrelation(liquid["velocities"], 200)
c_cry = transport.velocity_autocorrelation(crystal["velocities"], 200)
c_liq, c_cry = c_liq / c_liq[0], c_cry / c_cry[0]
lag = DT * np.arange(201)
zero = np.argmax(c_liq < 0)
print(f"liquid: C_vv first reaches 0 at {lag[zero]:.0f} fs; least value "
      f"{c_liq.min():.3f} at {lag[np.argmin(c_liq)]:.0f} fs")
print(f"crystal: first zero at {lag[np.argmax(c_cry < 0)]:.0f} fs, least "
      f"{c_cry.min():.3f} at {lag[np.argmin(c_cry)]:.0f} fs")

names = ["long_nve", "long_csvr"] + [f"size_{n}{t}" for n in
                                     (256, 500, 864, 1372, 2048)
                                     for t in ("", "_b")]
curves, temps, length = [], [], 0.0
for name in names:
    run = np.load(ch15.RUNS / f"{name}.npz")
    atoms = run["positions"].shape[1]
    temperature = 2 * run["kinetic"].mean() / ((3 * atoms - 3) * units.KB)
    temps.append(temperature)
    length += run["times"][-1] / 1e6
    corr = transport.correlation(run["shear"], max_lag=1000) / 3
    curves.append(transport.running_integral(corr, DT)
                  * run["volume"].mean() / (units.KB * temperature) * PA_S)
curves = np.array(curves)
print(f"{len(names)} runs, {length:.0f} ns in all, at {min(temps):.1f} to "
      f"{max(temps):.1f} K")
eta = curves.mean(0)
err = curves.std(0, ddof=1) / np.sqrt(len(curves))
lag_eta = DT * np.arange(1001)
window = (lag_eta >= 1000) & (lag_eta <= 3000)
plateau, plateau_err = eta[window].mean(), err[window].mean()
print(f"η_s over 1-3 ps: {plateau:.3e} ± {plateau_err:.1e} Pa s, "
      f"{plateau / REDUCED:.2f} ± {plateau_err / REDUCED:.2f} reduced; at "
      f"10 ps {eta[-1]:.3e} ± {err[-1]:.1e}")
print(f"at 1 ps {eta[100]:.3e}, {eta[100] / plateau:.3f} of the plateau")
print(f"Yeh-Hummer {YH:.2e} ± {YH_ERR:.1e}: "
      f"{(YH - plateau) / np.hypot(YH_ERR, plateau_err):.1f} combined errors "
      f"apart")
d_corr = 2.837297 * units.KB * 135.0 * units.ELEMENTARY_CHARGE / (
    6 * np.pi * plateau * 23.26e-10) * 1e5  # m²/s to Å²/fs
print(f"the Yeh-Hummer correction at L = 23.26 Å with this η_s: "
      f"{d_corr * 1e4:.3f}e-4 Å²/fs; 3.914 + it = {3.914 + d_corr * 1e4:.3f}"
      f"e-4 against Section 15.7's D∞ = 4.503 ± 0.032e-4")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.32))
ax = axes[0]
ax.plot(lag, c_cry, color=OCHRE, lw=0.8, label="crystal, 21 K")
ax.plot(lag, c_liq, color=ACCENT, lw=1.0, label="liquid, 132 K")
ax.axhline(0, color="black", lw=0.4)
ax.set_xlim(0, 2000)
ax.set_xlabel(r"$t$ / fs")
ax.set_ylabel(r"$C_{vv}(t)/C_{vv}(0)$")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.axhspan((YH - YH_ERR) * 1e4, (YH + YH_ERR) * 1e4,
           color=viz.tint(REFERENCE, 0.25), lw=0)
ax.axhline(YH * 1e4, color=REFERENCE, ls="--", lw=0.8)
ax.fill_between(lag_eta / 1000, (eta - 2 * err) * 1e4, (eta + 2 * err) * 1e4,
                color=viz.tint(ACCENT, 0.3), lw=0)
ax.plot(lag_eta / 1000, eta * 1e4, color=ACCENT, lw=1.0)
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.set_xlabel(r"upper limit $t$ / ps")
ax.set_ylabel(r"$\eta_{\mathrm s}$ / $10^{-4}$ Pa$\,$s")
viz.panel_tag(ax, "b")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "vacf.pdf")))
