"""Numbers quoted in Chapter 12 that no figure script prints.

Run from theory/: python scripts/ch12_ensembles/numbers_chapter.py
"""

import math

import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import thermalize_momenta
from ch12 import MASS, RUNS

from mdlab import chain, statmech, units


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


heading("12.1 a fair die and the Gaussian")
faces = np.arange(1, 7)
mean = faces.mean()
print(f"one die: mean {mean}, variance {np.mean((faces - mean) ** 2):.4f} = "
      f"35/12 = {35 / 12:.4f}; two dice variance {2 * 35 / 12:.4f}")
print(f"fraction of a Gaussian within 1, 2, 3 standard deviations: "
      f"{math.erf(1 / math.sqrt(2)):.4f}, {math.erf(2 / math.sqrt(2)):.4f}, "
      f"{math.erf(3 / math.sqrt(2)):.4f}")

heading("12.2 the chain's normal modes")
patterns, omega = chain.normal_modes(32)
print(f"N = 32: omega_1 = {omega[0]:.5f}, period {2 * math.pi / omega[0]:.2f};"
      f" omega_32 = {omega[-1]:.5f}, period {2 * math.pi / omega[-1]:.3f}")
print(f"amplitude 4 in mode 1: energy {0.5 * omega[0] ** 2 * 16:.5f}")

heading("12.3 the ideal gas")
print(f"N = 1000: relative width of the share 1/sqrt(12N) "
      f"{1 / math.sqrt(12000):.4f}")

heading("12.4 Boltzmann factors")
for t in (100, 300, 600):
    kt = units.KB * t
    print(f"T = {t} K: k_B T = {kt:.5f} eV; e^(-0.3 eV/k_B T) = "
          f"{statmech.boltzmann_factor(0.3, t):.3e}")
print(f"1/k_B = {1 / units.KB:.2f} K/eV")
odds = [statmech.boltzmann_factor(0.3, t) for t in (300, 600)]
print(f"ratio of the odds at 600 and 300 K: {odds[1] / odds[0]:.0f}")

heading("12.5 velocities at 300 K")
for name, mass in (("H", 1.008), ("Li", 6.94), ("O", 15.999),
                   ("Ar", 39.948)):
    sigma = math.sqrt(units.KB * 300 / (mass * units.MV2_TO_EV))
    print(f"{name}: sigma = sqrt(k_B T/m) = {sigma:.6f} Å/fs, mean speed "
          f"{math.sqrt(8 / math.pi) * sigma:.5f} Å/fs")

print(f"ratio of argon's sigma to lithium's: sqrt(39.948/6.94) = "
      f"{math.sqrt(39.948 / 6.94):.3f}")

heading("12.5 checked against ASE: the same velocities")
m = np.random.default_rng(0).uniform(1, 40, 50)
ours = statmech.thermal_velocities(m, 300.0, np.random.default_rng(7),
                                   remove_drift=False)
atoms = Atoms("H50", positions=np.zeros((50, 3)), masses=m)
thermalize_momenta(atoms, 300.0, rng=np.random.default_rng(7))
theirs = atoms.get_velocities() * math.sqrt(units.FORCE_TO_ACCEL)
print(f"largest relative difference {np.abs(ours / theirs - 1).max():.1e}")
from ase import units as ase_units  # noqa: E402

print(f"Boltzmann's constant: mdlab {units.KB:.10e}, ASE {ase_units.kB:.10e}"
      f" eV/K, relative difference {units.KB / ase_units.kB - 1:.1e}")

heading("12.6 temperature: mdlab against ASE")
run = np.load(RUNS / "liquid.npz")
v = run["velocities"][-1]
mm = run["masses"]
ours_t = statmech.kinetic_temperature(mm, v, statmech.degrees_of_freedom(256))
atoms = Atoms("Ar256", positions=np.zeros((256, 3)), masses=mm)
atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
print(f"last frame of the liquid: mdlab {ours_t:.4f} K over 765 freedoms; "
      f"ASE {atoms.get_temperature():.4f} K over 768; ratio "
      f"{atoms.get_temperature() / ours_t:.6f}, 765/768 = {765 / 768:.6f}")
print(f"half k_B T at 300 K: {500 * units.KB * 300:.3f} meV per freedom")
print(f"diatomic turning at 300 K: k_B T = {units.KB * 300:.5f} eV")

heading("12.7 the fourth moment of a Gaussian")
x = np.random.default_rng(5).standard_normal(2_000_000)
print(f"<x^4> of 2e6 normal numbers {np.mean(x ** 4):.4f} (3 exactly)")
print(f"sqrt(2/N_f) for N_f = 765: {math.sqrt(2 / 765):.4f}; for 3: "
      f"{math.sqrt(2 / 3):.4f}")
print(f"argon mass {MASS} amu")

heading("12.8 an orbital in contact with electrons")
for gap in (0.0, 0.1, -0.1):
    occupation = 1 / (math.exp(gap / (units.KB * 300)) + 1)
    print(f"epsilon - mu = {gap:+.1f} eV at 300 K: occupation "
          f"{occupation:.4f}")
print(f"largest entropy of one orbital, at occupation 1/2: k_B ln 2 = "
      f"{units.KB * math.log(2):.4e} eV/K")
