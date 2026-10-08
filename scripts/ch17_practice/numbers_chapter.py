"""The numbers of Chapter 17's checks and exercises, from their data."""

import math

import numpy as np

from mdlab import units

KB = units.KB
AMU, EV, ANGSTROM = 1.66053906892e-27, 1.602176634e-19, 1e-10

# ASE's unit of time
t_ase = ANGSTROM * math.sqrt(AMU / EV) / 1e-15
print(f"ASE time unit {t_ase:.4f} fs; 0.5 fs is {0.5 / t_ase:.5f} units; "
      f"timestep=2 is {2 * t_ase:.2f} fs")
# a code in kcal/mol, Å and amu
kcal_mol = 4184 / 6.02214076e23
t_akma = ANGSTROM * math.sqrt(AMU / kcal_mol) / 1e-15
print(f"kcal/mol, Å, amu: time unit {t_akma:.3f} fs; 2 fs is "
      f"{2 / t_akma:.5f} units; kcal/mol = {kcal_mol:.4e} J")
# one velocity in three codes
v = 0.01
print(f"0.01 Å/fs: ASE {v * t_ase:.4f}, LAMMPS metal {v * 1000:.1f} Å/ps")
# friction in two codes
gamma_ps = 0.5
print(f"γ = 0.5 /ps: LAMMPS damp {1 / gamma_ps:.1f} ps; ASE friction "
      f"{gamma_ps / 1000 * t_ase:.5f} per ASE unit")
# two masses for one chain
n = 256
ratio = 3 * n / (3 * n - 3)
print(f"chain masses 3N / N_f for 256 atoms: {ratio:.5f}; periods "
      f"{math.sqrt(ratio):.5f}")
# entries for a precision: LAMMPS's own table misses by 7.6e-7 (fig_codes)
factor = math.sqrt(7.6e-7 / 1e-9)
print(f"table: spacing to fall {factor:.1f} times; entries for 1e-9 eV/Å: "
      f"{20000 * factor:.3g}")
# image flags
image = 540539393
print(f"image {image}: ix {image % 1024 - 512}, iy "
      f"{image // 1024 % 1024 - 512}, iz {image // 1024**2 - 512}")
# a nanosecond of a foundation model
per_call = 0.25 * 1000 / 126
print(f"1000 atoms: {per_call:.2f} s a step; 1 ns {per_call * 1e6 / 86400:.0f}"
      f" days; at the measured 2.84 s for 1008 atoms "
      f"{2.84 * 1e6 / 86400:.0f} days")
# sheets that come apart
kt300 = KB * 300
print(f"k_BT at 300 K {kt300 * 1000:.1f} meV against 5.2 and 41.2 "
      f"({41.2 / (kt300 * 1000):.1f} times); 64 atoms × 5.2 meV = "
      f"{64 * 5.2e-3:.2f} eV, {64 * 5.2e-3 / kt300:.0f} k_BT")
# the longest step
step = math.sqrt(0.05 / 0.0067)
print(f"spread 0.0067 at 1 fs reaches 0.05 at {step:.2f} fs; δt² gives "
      f"{0.0067 * 9:.3f} at 3 fs")
# check 17.4: four times the atoms
print(f"check: 64 × 192 s = {64 * 192} s, {64 * 192 / 3600:.1f} h")
# the price of a reference
core_hours = 192 * 10000 * 4 / 3600
print(f"10 ps of AIMD: {core_hours:.0f} core-hours; on 192 cores "
      f"{core_hours / 192:.1f} h")
# the √3 cell
a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, math.sqrt(3) / 2])
b1, b2 = 2 * a1 + a2, -a1 + a2
cos = b1 @ b2 / (np.linalg.norm(b1) * np.linalg.norm(b2))
area = abs(np.linalg.det([b1, b2])) / abs(np.linalg.det([a1, a2]))
print(f"√3 cell: lengths {np.linalg.norm(b1):.4f}, {np.linalg.norm(b2):.4f};"
      f" angle {math.degrees(math.acos(cos)):.1f}°; area ratio {area:.3f}")
# the mean of an unequal pressure
p = np.array([-1.00, -1.11, 1.68])
e = np.array([0.45, 0.36, 0.10])
print(f"mean pressure {p.mean():.3f} ± {np.sqrt(np.sum(e**2)) / 3:.3f} GPa")
# how long to resolve a rate
print(f"25 hops at 2 per ps: {25 / 2:.1f} ps; none in 20 ps: rate below "
      f"{-math.log(0.05) / 20:.3f} per ps; resolving that bound needs "
      f"{25 / (-math.log(0.05) / 20):.0f} ps")
# check 17.6: a rate to a tenth at 450 K
print(f"a tenth needs {1 / 0.1**2:.0f} hops; at 2.40 per ps "
      f"{1 / 0.1**2 / 2.40:.0f} ps")
# hops on a honeycomb
a = 2.4648
print(f"honeycomb neighbours a/√3 = {a / math.sqrt(3):.4f} Å")
# the kurtosis limit
limit = 3 * math.sqrt(24 / 768)
print(f"kurtosis limit for 768 components {limit:.3f}; -1.2 fails")
