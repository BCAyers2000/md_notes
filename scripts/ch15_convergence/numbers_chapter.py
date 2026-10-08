"""The numbers of Chapter 15 that no figure script prints.

The answers to the checks and the exercises, from the data each states.
"""

import math

from scipy import stats as st

from mdlab import units

KB = units.KB
SIGMA_DIE = math.sqrt(35 / 12)

print("checks")
print(f"15.1: 100 samples give 0.05; 0.01 needs (0.05/0.01)² × 100 = "
      f"{(0.05 / 0.01) ** 2 * 100:.0f} samples")
print(f"15.2: corr = 0.5^k: g = 1.5/0.5 = {1.5 / 0.5:.0f}; 1000 samples are "
      f"worth {1000 / 3:.0f}")
print(f"15.3: g = 100: blocks of {5 * 100} samples; 20 000 of them hold "
      f"{20000 // 500} blocks")
print(f"15.4: offset 2 K over 1 ps in 100 ps: shift Δτ/T = "
      f"{2 * 1 / 100:.2f} K")
print(f"15.5: 4 times the run: error falls by 4^1.5 = {4 ** 1.5:.0f}, to "
      f"{2 / 4 ** 1.5:.2f}; 3 is then {3 / (2 / 4 ** 1.5):.0f} errors")
print(f"15.6: T ≥ 2 × 5 ps × 2²/0.1² = {2 * 5 * 4 / 0.01:.0f} ps")
beta = 1 / (KB * 130) - 1 / (KB * 140)
beta_wrong = 1 / (KB * 132) - 1 / (KB * 140)
print(f"15.8: β130 − β140 = {beta:.3f}; a run at 132 K labelled 130 K "
      f"gives {beta_wrong:.3f}, {beta_wrong / beta:.2f} of it")
print(f"15.9: 500 ps at 10 fs with g = 300: {50000 / 300:.0f} independent "
      f"samples")

print("exercises")
n_rolls = 1.96**2 * (35 / 12) / 0.01**2
print(f"rolls: n ≥ 1.96² × (35/12)/0.01² = {n_rolls:.1f}, so "
      f"{math.ceil(n_rolls)} rolls")
steps = math.log(100) / (2 * math.log(1 / 0.95))
print(f"OU from x0 = 0: 1 − var(x_k) = c^(2k) < 0.01 needs k ≥ {steps:.1f}, "
      f"so {math.ceil(steps)} steps ({10 * math.ceil(steps)} fs)")
print(f"student: 2.776/1.96 − 1 = {2.776 / 1.96 - 1:.3f}; factor for four "
      f"degrees of freedom {st.t.ppf(0.975, 4):.3f}")
c = 0.99
g = (1 + c) / (1 - c)
print(f"g for c = 0.99: {g:.0f}; τ_int = g δt/2 = {g * 10 / 2:.0f} fs; "
      f"τ = −δt/ln c = {-10 / math.log(c):.1f} fs")
n_b = 2**16 // 300
print(f"blocks: 5g = 300 samples; 2^16 // 300 = {n_b} blocks; precision "
      f"1/√(2(n_b − 1)) = {1 / math.sqrt(2 * (n_b - 1)):.3f}; with blocks "
      f"of 512, {2**16 // 512} blocks and "
      f"{1 / math.sqrt(2 * (2**16 // 512 - 1)):.3f}")
left = 3 * 2 * (math.exp(-5.34 / 2) - math.exp(-50 / 2)) / (50 - 5.34)
print(f"offset left: 3 × 2 (e^-2.67 − e^-25)/44.66 = {left:.4f}")
print(f"drift limit for 10 ns: 20.8 × 10^-1.5 = {20.8 * 10 ** -1.5:.3f} μeV")
for name, sigma, tau in (("CSVR", 0.01328, 940.0), ("fixed energy", 0.009613,
                                                    123.0)):
    print(f"pressure to ±0.0005 GPa, {name}: 2 × {tau:g} fs × {sigma}² / "
          f"0.0005² = {2 * tau * sigma**2 / 0.0005**2 / 1e6:.3f} ns")
need = 100 * 2 * 2.0  # ps of production for 100 samples at τ_int = 2 ps
print(f"short run: 140 ps after the start hold {140 / (2 * 2.0):.0f}; 100 "
      f"need {need:.0f} ps, so a run of {need + 10:.0f} ps, "
      f"{need + 10 - 150:.0f} ps longer")
print(f"gamma ratio for 300 and 310 K: {1 / (KB * 300) - 1 / (KB * 310):.4f}"
      f" per eV")
kt_joule = KB * 135 * units.ELEMENTARY_CHARGE
yh = 2.837297 * kt_joule / (6 * math.pi * 2.18e-4 * 23.26e-10)
print(f"Yeh-Hummer correction at L = 23.26 Å with η = 2.18e-4 Pa s: "
      f"{yh:.3e} m²/s = {yh / 1e-5 * 1e4:.3f} e-4 Å²/fs, "
      f"{yh / 1e-5 * 1e4 / 3.914:.3f} of 3.914")
print(f"kT at 135 K = {kt_joule:.4e} J")
