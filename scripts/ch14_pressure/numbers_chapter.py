"""The numbers of Chapter 14 that no figure script prints.

Run from scripts/ch14_pressure/: python numbers_chapter.py

Unit conversions, the worked numbers of the sections and the answers to
the checks, each from the constants of mdlab.units and the cached runs.
"""

import math

import numpy as np
from ch14 import (
    BAR,
    GPA,
    KAPPA_LIQUID,
    N_ATOMS,
    RUNS,
    T_LIQUID,
    compressibility_at,
    fixed_volume_pressures,
)

from mdlab import units

KT = units.KB * T_LIQUID
print(f"1 eV/Å³ = {GPA:.3f} GPa = {GPA * BAR:.4e} bar; 1 bar = "
      f"{1 / (GPA * BAR):.4e} eV/Å³; 1 atm = 1.01325 bar")
kt_300 = units.KB * 300.0
density_air = (1 / (GPA * BAR)) / kt_300  # atoms per Å³ at 1 bar, 300 K
print(f"an ideal gas at 300 K and 1 bar: N/V = {density_air:.3e} Å⁻³ = "
      f"{density_air * 1000:.4f} nm⁻³; in a cube of 10 nm "
      f"{density_air * 1e6:.1f} molecules")

volume = 12577.28  # Chapter 12's box at 0.8σ⁻³
print(f"the liquid's box, Chapter 12: V = {volume:.0f} Å³; Nk_BT/V at "
      f"135 K = {N_ATOMS * KT / volume * GPA:.4f} GPa, (N − 1)k_BT/V = "
      f"{(N_ATOMS - 1) * KT / volume * GPA:.4f} GPa")

r = np.load(RUNS / "npt_scr_s0.npz")
mean_v = r["volume"][r["times"] >= 10000].mean()
kappa_t = compressibility_at(mean_v)
print(f"κ entered {KAPPA_LIQUID / GPA:.2f} GPa⁻¹ = "
      f"{KAPPA_LIQUID:.1f} Å³/eV; κ_T {kappa_t / GPA:.2f} GPa⁻¹; volume "
      f"relaxes under Berendsen with τ_P κ_T/κ = "
      f"{1.0 * kappa_t / KAPPA_LIQUID:.2f} ps for τ_P = 1 ps; with κ "
      f"10⁴ times too small, {1e4 * kappa_t / KAPPA_LIQUID / 1000:.1f} ns")
noise = math.sqrt(2 * KT * KAPPA_LIQUID * 0.01 / mean_v)
print(f"cell rescaling's noise in ln V per step (δt/τ_P = 0.01): "
      f"{noise:.2e}; its stationary spread √(k_BTκ_T/V) = "
      f"{math.sqrt(KT * kappa_t / mean_v):.4f}")
mass = (3 * N_ATOMS + 3) * KT * 1000.0**2
print(f"MTK piston mass for τ_P = 1 ps: {mass:.3e} eV fs²")

print(f"opening on lithiation: {3.70 / 3.34 - 1:.4f}")

# the checks
mu = 1 - (KAPPA_LIQUID * 10.0 / (3 * 1000.0)) * (0.1 / GPA - 0.08 / GPA)
print(f"Berendsen, P = 0.08 GPa, target 0.1, κ {KAPPA_LIQUID / GPA:.2f} "
      f"GPa⁻¹, δt/τ_P = 0.01: "
      f"lengths multiplied by {mu:.7f}")
n_ideal, p_ideal = 20, 1e-4
print(f"ideal gas, N = {n_ideal}, P0 = {p_ideal} eV/Å³ at 135 K: mean "
      f"{(n_ideal + 1) * KT / p_ideal:.0f} Å³, spread "
      f"{math.sqrt(n_ideal + 1) * KT / p_ideal:.0f} Å³")
print(f"kinetic part dropped: shift of volume κ_T Nk_BT/V ≈ "
      f"{kappa_t * N_ATOMS * KT / mean_v:.4f} of V")
p_check = 1e4 * kt_300 / 100.0**3
print(f"check 14.1: 10⁴ atoms at 300 K in (100 Å)³: {p_check:.4e} eV/Å³ = "
      f"{p_check * GPA * BAR:.1f} bar")
v1, v2 = 5.20**3, 5.40**3
b_est = (0.1282 + 0.1600) / GPA * 0.5 * (v1 + v2) / (v2 - v1)
print(f"check 14.3: B ≈ −V̄ ΔP/ΔV from a = 5.20 and 5.40 Å: "
      f"{b_est * GPA:.2f} GPa")
tensors = []
for seed in range(3):
    run = np.load(RUNS / f"npt_scr_s{seed}.npz")
    tensors.append(run["pressure"][run["times"] >= 10000])
tensors = np.concatenate(tensors) * GPA
mean_t = tensors.mean(0)
print(f"the liquid's mean pressure tensor (cell rescaling, three runs), GPa: "
      f"diagonal {np.diag(mean_t).round(4)}, off-diagonal "
      f"{mean_t[0, 1]:+.4f} {mean_t[0, 2]:+.4f} {mean_t[1, 2]:+.4f}; "
      f"spread of one instantaneous component {tensors[:, 0, 0].std():.4f}, "
      f"of P_xy {tensors[:, 0, 1].std():.4f}")
print(f"check 14.4: stress 0.002 eV/Å³ on the diagonal: P = "
      f"{-0.002 * GPA:.4f} GPa")
print(f"liquid: B = 1/κ_T = {1 / kappa_t * GPA:.3f} GPa; crystal 2.856 GPa, "
      f"ratio {2.856 / (1 / kappa_t * GPA):.2f}; spread/V "
      f"{math.sqrt(KT * mean_v * kappa_t) / mean_v:.4f}")
print(f"check 14.5: 1/√1001 = {1 / math.sqrt(1001):.4f}")
omega = math.sqrt((1e-3) ** 2 / (1.0 * 1e-3 * 1e-5))
print(f"check 14.7: ω = {omega:.2f} rad/s, period {2 * math.pi / omega:.3f} s")
print(f"check 14.8: volume grows by {3.70 / 3.34 - 1:.4f} if only c opens")
kin = 500 * kt_300 / 20000.0
print(f"check 14.9: Nk_BT/V for 500 atoms at 300 K in 20000 Å³ = "
      f"{kin * GPA:.4f} GPa; −0.04 + that = {-0.04 + kin * GPA:.4f} GPa")
print(f"exercise: air at 1 bar and 300 K, V/N = k_BT/P = "
      f"{1 / density_air:.0f} Å³, spacing (V/N)^(1/3) = "
      f"{(1 / density_air) ** (1 / 3):.1f} Å")
print(f"exercise: the 10.8% volume growth shared by three lengths: "
      f"{1.108 ** (1 / 3) - 1:.4f} each")
print(f"exercise: r_min = 2^(1/6) σ = {2 ** (1 / 6) * 3.4:.3f} Å")
dropped = np.load(RUNS / "fault_kinetic_s0.npz")
v_dropped = dropped["volume"][dropped["times"] >= 10000].mean()
print(f"exercise: measured shift of volume without the kinetic part "
      f"{v_dropped / mean_v - 1:.4f}; κ_T at the shrunken volume "
      f"{compressibility_at(v_dropped) / GPA:.2f} GPa⁻¹ (beyond the "
      f"fixed-volume runs)")
kappa_ex = 1.46 * GPA
factor = 2 * math.pi * math.sqrt(257 * KT * kappa_ex / (3 * 12306.0))
print(f"exercise: period factor with κ_T = 1.46 GPa⁻¹, V = 12306 Å³: "
      f"{factor:.3f}; κ_T N k_BT/V there "
      f"{kappa_ex * N_ATOMS * KT / 12306.0:.4f}; N k_BT/V "
      f"{N_ATOMS * KT / 12306.0 * GPA:.4f} GPa")
print(f"exercise: (C11 + 2 C12)/3 = {(3.920 + 2 * 2.324) / 3:.3f} GPa")
print(f"exercise: Nk_BT/V at the healthy volume {mean_v:.0f} Å³: "
      f"{N_ATOMS * KT / mean_v * GPA:.4f} GPa; κ_T times that "
      f"{kappa_t * N_ATOMS * KT / mean_v:.4f}")
volumes, means, _ = fixed_volume_pressures()
p_dropped = np.trace(dropped["pressure"][dropped["times"] >= 10000],
                     axis1=1, axis2=2).mean() / 3
stiff_low = (p_dropped - means[0]) / (volumes[0] - v_dropped)
stiff_mid = (means[0] - means[1]) / (volumes[1] - volumes[0])
print(f"exercise: pressure rise per Å³ lost between {v_dropped:.0f} and "
      f"{volumes[0]:.0f} Å³: {stiff_low * GPA:.2e} GPa; between "
      f"{volumes[0]:.0f} and {volumes[1]:.0f} Å³: {stiff_mid * GPA:.2e} GPa")
