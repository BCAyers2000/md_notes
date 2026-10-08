"""Figure fig:pr-volume: the volume a barostat samples.

(a) 20 argon atoms that do not interact, at 135 K and P₀ = 10⁻⁴ eV/Å³
(runs.py, ideal_*): the density of the volume under Berendsen's
barostat, stochastic cell rescaling and the MTK piston, after the first
20 ps, against the exact V^N e^{−βP₀V} (dashed). (b) The liquid at 135 K
and 0.1 GPa (npt_*, three starts each, after the first 10 ps): the
density of the volume under the three, against the Gaussian of variance
k_BT⟨V⟩κ_T, κ_T = −1/(V dP/dV) from the pressures at three fixed
volumes (dashed).

Prints the numbers of Sections 14.5 and 14.6: the ideal gas's mean and
spread against the exact ones; the liquid's mean volume, pressure and
the compressibility from its fluctuations, run by run; and κ_T from the
pressures at three fixed volumes bracketing the mean volume.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch14 import (
    GPA,
    RUNS,
    T_LIQUID,
    compressibility_at,
    fixed_volume_pressures,
    isothermal_compressibility,
)

from mdlab import units, viz
from mdlab.virial import ideal_gas_volume_density
from mdlab.viz import BAROSTAT, REFERENCE_STYLE, figure_path

KINDS = (("berendsen", "Berendsen barostat"),
         ("scr", "stochastic cell rescaling"),
         ("mtk", "MTK piston"))
LABEL = {"berendsen": "Berendsen", "scr": "cell rescaling", "mtk": "MTK"}
KT = units.KB * T_LIQUID

# (a) the ideal gas
ideal = {}
for kind, _ in KINDS:
    r = np.load(RUNS / f"ideal_{kind}.npz")
    vol = r["volume"][r["times"] >= 20000]
    n, p0 = int(r["n"]), float(r["pressure"])
    ideal[kind] = vol
    exact_mean = (n + 1) * KT / p0
    exact_sd = math.sqrt(n + 1) * KT / p0
    print(f"(a) ideal gas, {LABEL[kind]}: mean {vol.mean():.0f} Å³ "
          f"({vol.mean() / exact_mean:.3f} of the exact {exact_mean:.0f}), "
          f"spread {vol.std():.0f} Å³ ({vol.std() / exact_sd:.3f} of "
          f"{exact_sd:.0f})")
print(f"    Nk_BT/P0 = {n * KT / p0:.0f} Å³")


# (b) the liquid
def late(name, after=10000):
    r = np.load(RUNS / f"{name}.npz")
    keep = r["times"] >= after
    tensor = r["pressure"][keep]
    p = (np.trace(tensor, axis1=1, axis2=2) / 3 if tensor.ndim == 3
         else tensor)
    return r["volume"][keep], p


liquid = {}
for kind, _ in KINDS:
    volumes, kappas = [], []
    for seed in range(3):
        vol, p = late(f"npt_{kind}_s{seed}")
        volumes.append(vol)
        kappas.append(isothermal_compressibility(vol, T_LIQUID))
        print(f"(b) {LABEL[kind]}, s{seed}: ⟨V⟩ {vol.mean():.0f} Å³, spread "
              f"{vol.std():.1f} Å³, ⟨P⟩ {p.mean() * GPA:.4f} GPa, κ from "
              f"fluctuations {kappas[-1] / GPA:.2f} GPa⁻¹")
    liquid[kind] = np.concatenate(volumes)
    print(f"    mean κ over the three {np.mean(kappas) / GPA:.2f} ± "
          f"{np.std(kappas, ddof=1) / math.sqrt(3) / GPA:.2f} GPa⁻¹")

mean_v = np.mean(np.concatenate(list(liquid.values())))
volumes, means, errors = fixed_volume_pressures()
for vol, p, err in zip(volumes, means, errors, strict=True):
    print(f"    fixed volume {vol:.0f} Å³: ⟨P⟩ {p * GPA:.4f} ± "
          f"{err * GPA:.4f} GPa (three starts)")
kappa_fixed = compressibility_at(mean_v)
rng = np.random.default_rng(0)
draws = []
for _ in range(20000):
    trial = means + errors * rng.standard_normal(3)
    slope = np.polyval(np.polyder(np.polyfit(volumes, trial, 2)), mean_v)
    draws.append(-1 / (mean_v * slope))
print(f"κ_T at the mean volume {mean_v:.0f} Å³ from the fixed-volume "
      f"pressures: {kappa_fixed / GPA:.2f} ± {np.std(draws) / GPA:.2f} "
      f"GPa⁻¹ (spread over the pressures' standard errors); predicted "
      f"spread {math.sqrt(KT * mean_v * kappa_fixed):.0f} Å³")
for seed in range(3):
    r = np.load(RUNS / f"npt_mtk_s{seed}.npz")
    keep = r["times"] >= 10000
    t_kin = 2 * r["kinetic"][keep].mean() / ((3 * 256 - 3) * units.KB)
    print(f"    MTK s{seed}: kinetic temperature over 3N − 3 freedoms "
          f"{t_kin:.2f} K (its chain targets 3N)")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                                 gridspec_kw=dict(wspace=0.35))
grid = np.linspace(500, 6000, 400)
for kind, name in KINDS:
    ax_a.hist(ideal[kind], bins=np.linspace(500, 6000, 56), density=True,
              histtype="step", color=BAROSTAT[name], lw=1.0, label=LABEL[kind])
ax_a.plot(grid, ideal_gas_volume_density(grid, 20, 1e-4, T_LIQUID),
          **REFERENCE_STYLE, lw=1.0)
ax_a.set_xlabel(r"$V$ / Å$^3$")
ax_a.set_ylabel(r"density / Å$^{-3}$")
ax_a.legend(fontsize=6, loc="upper right")
viz.panel_tag(ax_a, "a")

grid_b = np.linspace(mean_v - 700, mean_v + 700, 300)
sd = math.sqrt(KT * mean_v * kappa_fixed)
for kind, name in KINDS:
    ax_b.hist(liquid[kind], bins=np.linspace(mean_v - 700, mean_v + 700, 41),
              density=True, histtype="step", color=BAROSTAT[name], lw=1.0)
ax_b.plot(grid_b, np.exp(-0.5 * ((grid_b - mean_v) / sd) ** 2)
          / (sd * math.sqrt(2 * math.pi)), **REFERENCE_STYLE, lw=1.0)
ax_b.set_xlabel(r"$V$ / Å$^3$")
ax_b.set_ylabel(r"density / Å$^{-3}$")
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, figure_path("ch14_pressure", "volume.pdf")))
