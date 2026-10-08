"""Figure fig:cv-size: a result that depends on the size of the box.

The liquid at 0.8σ⁻³ and 135 K with 256, 500, 864, 1372 and 2048 atoms,
1 ns each under CSVR from two seeds (runs.py, size_* and size_*_b). (a)
The diffusion coefficient, each with its error from twenty blocks of
100 ps and moved to 135 K along dD/dT of Section 15.6, against 1/L,
with the straight line fitted to them (dashed) and its value at 1/L = 0.
(b) The potential energy per atom and the pressure against 1/L, each
less its value for 2048 atoms, in units of its own error.

Prints D, T, U/N and P for each size; the line's slope and intercept with
their errors; the viscosity the slope implies through the Yeh-Hummer
form, in Pa s and in reduced units; and the shortfall of D at 256 atoms.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch15 import GPA, RUNS, T_LIQUID, ch12, diffusion

from mdlab import statmech, units, viz
from mdlab.analysis import stats
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, THRESHOLD_STYLE

SIZES = (256, 500, 864, 1372, 2048)
DD_DT = 0.0294  # 1e-4 Å²/fs per K, Section 15.6
XI = 2.837297  # Yeh and Hummer's constant for a cubic box

def pooled(values_errors):
    """The mean of independent estimates with their errors, and its error."""
    values, errors = np.array(values_errors).T
    weights = 1 / errors**2
    return (np.sum(weights * values) / np.sum(weights),
            1 / math.sqrt(np.sum(weights)))


rows = []
for n in SIZES:
    names = [f"size_{n}"] + ([f"size_{n}_b"]
                             if (RUNS / f"size_{n}_b.npz").exists() else [])
    ds, temps, us, ps = [], [], [], []
    for name in names:
        r = np.load(RUNS / f"{name}.npz")
        pos = r["positions"].astype(float)
        side = r["cell"][0, 0]
        n_free = statmech.degrees_of_freedom(n)
        temp, temp_err, _ = stats.standard_error(2 * r["kinetic"]
                                                 / (n_free * units.KB))
        size = (len(pos) - 1) // 10
        ds += list(1e4 * np.array([
            diffusion(pos[j * size:(j + 1) * size + 1], 1000.0)
            + DD_DT * 1e-4 * (T_LIQUID - temp) for j in range(10)]))
        temps.append((temp, temp_err))
        us.append(stats.standard_error(r["potential"] / n)[:2])
        ps.append(stats.standard_error(r["pressure"] * GPA)[:2])
    ds = np.array(ds)
    d, d_err = ds.mean(), ds.std(ddof=1) / math.sqrt(len(ds))
    temp, temp_err = pooled(temps)
    u, u_err = pooled(us)
    p, p_err = pooled(ps)
    rows.append((n, side, temp, d, d_err, u, u_err, p, p_err))
    print(f"N = {n} ({len(names)} runs): L = {side:.2f} Å, T = {temp:.2f} ± "
          f"{temp_err:.2f} K; D at 135 K = {d:.3f} ± {d_err:.3f} (1e-4 Å²/fs,"
          f" {len(ds)} blocks); U/N = {1000 * u:.3f} ± {1000 * u_err:.3f} "
          f"meV; P = {p:.4f} ± {p_err:.4f} GPa")

rows = np.array(rows)
inv_l = 1 / rows[:, 1]
d, d_err = rows[:, 3], rows[:, 4]
weights = 1 / d_err**2
x_mean = np.sum(weights * inv_l) / np.sum(weights)
y_mean = np.sum(weights * d) / np.sum(weights)
s_xx = np.sum(weights * (inv_l - x_mean) ** 2)
slope = np.sum(weights * (inv_l - x_mean) * (d - y_mean)) / s_xx
slope_err = math.sqrt(1 / s_xx)
d_inf = y_mean - slope * x_mean
d_inf_err = math.sqrt(1 / np.sum(weights) + x_mean**2 / s_xx)
chi2 = np.sum(weights * (d - d_inf - slope * inv_l) ** 2)
print(f"line D = D∞ + s/L: D∞ = {d_inf:.3f} ± {d_inf_err:.3f}, s = "
      f"{slope:.2f} ± {slope_err:.2f} (1e-4 Å³/fs); χ² = {chi2:.2f} for "
      f"{len(d) - 2} degrees of freedom")
largest_fix = DD_DT * np.max(np.abs(T_LIQUID - rows[:, 2]))
print(f"change from 256 to 2048 atoms {d[-1] - d[0]:.3f}, "
      f"{(d[-1] - d[0]) / max(d_err[0], d_err[-1]):.1f} times the larger "
      f"error; D(256)/D(2048) = {d[0] / d[-1]:.3f}; P(256) − P(2048) = "
      f"{rows[0, 7] - rows[-1, 7]:+.4f} GPa, "
      f"{(rows[0, 7] - rows[-1, 7]) / rows[-1, 7]:+.3f} of P; largest "
      f"temperature correction {largest_fix:.4f}")
reduced_unit = math.sqrt(ch12.MASS * units.AMU * ch12.EPS
                         * units.ELEMENTARY_CHARGE) / (ch12.SIG * 1e-10) ** 2
print(f"reduced unit of viscosity √(mε)/σ² = {reduced_unit:.3e} Pa s")
print(f"at 256 atoms D falls short of D∞ by "
      f"{100 * (1 - d[0] / d_inf):.1f}%; at 2048 by "
      f"{100 * (1 - d[-1] / d_inf):.1f}%")

kt_joule = units.KB * T_LIQUID * units.ELEMENTARY_CHARGE
slope_si = -slope * 1e-4 * 1e-20 / 1e-15 * 1e-10  # m³/s
eta = XI * kt_joule / (6 * math.pi * slope_si)
eta_err = eta * slope_err / abs(slope)
mass = ch12.MASS * units.AMU
eps = ch12.EPS * units.ELEMENTARY_CHARGE
sigma = ch12.SIG * 1e-10
eta_star = eta * sigma**2 / math.sqrt(mass * eps)
print(f"viscosity from the slope, ξk_BT/(6π|s|): {eta:.3g} ± {eta_err:.2g} "
      f"Pa s, η* = ησ²/√(mε) = {eta_star:.2f} ± "
      f"{eta_star * slope_err / abs(slope):.2f}")
print(f"the correction ξk_BT/(6πηL) at 256 atoms: "
      f"{-slope / rows[0, 1]:.3f} e-4 Å²/fs, "
      f"{-slope / rows[0, 1] / d[0]:.3f} of its D")
print(f"from 500 to 4000 atoms (L doubles from {rows[1, 1]:.2f} Å): D rises "
      f"by {-slope * (1 / rows[1, 1] - 1 / (2 * rows[1, 1])):.3f} e-4 Å²/fs")
fit_no_256 = np.polyfit(inv_l[1:], d[1:], 1, w=1 / d_err[1:])
print(f"without 256 atoms: D∞ = {fit_no_256[1]:.3f}, slope "
      f"{fit_no_256[0]:.2f}")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.2),
                                 gridspec_kw=dict(wspace=0.4))
ax_a.errorbar(inv_l, d, yerr=d_err, fmt="o", ms=3, color=ACCENT, lw=0.8,
              capsize=0)
grid = np.linspace(0, inv_l.max() * 1.05, 50)
ax_a.plot(grid, d_inf + slope * grid, **REFERENCE_STYLE, lw=0.8)
ax_a.errorbar([0], [d_inf], yerr=[d_inf_err], fmt="s", ms=3, color=OCHRE,
              lw=0.8, capsize=0)
ax_a.set_xlim(-0.002, inv_l.max() * 1.08)
ax_a.set_xlabel(r"$1/L$ / \AA$^{-1}$")
ax_a.set_ylabel(r"$D$ / $10^{-4}$ \AA$^2$ fs$^{-1}$")
viz.panel_tag(ax_a, "a")

ref = rows[-1]
for k, (label, marker) in enumerate(((r"$U/N$", "o"), (r"$P$", "s"))):
    value, error = rows[:, 5 + 2 * k], rows[:, 6 + 2 * k]
    shift = (value - ref[5 + 2 * k]) / np.hypot(error, ref[6 + 2 * k])
    print(f"{label} less that of 2048 atoms, in errors: "
          + ", ".join(f"{int(n)}: {x:+.1f}"
                      for n, x in zip(rows[:-1, 0], shift[:-1], strict=True)))
    ax_b.plot(inv_l[:-1], shift[:-1], marker, ms=3,
              color=(ACCENT, OCHRE)[k], label=label)
for level in (-2, 2):
    ax_b.axhline(level, **THRESHOLD_STYLE)
ax_b.axhline(0, color="black", lw=0.4)
ax_b.set_xlabel(r"$1/L$ / \AA$^{-1}$")
ax_b.set_ylabel("difference / combined error")
ax_b.legend(fontsize=7, loc="upper left")
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, viz.figure_path("ch15_convergence",
                                             "size.pdf")))
