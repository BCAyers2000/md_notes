"""Numbers quoted in Chapter 8 that no figure script prints.

Run from theory/: python scripts/ch08_potentials/numbers_chapter.py
"""

import math

import numpy as np
from scipy import constants as c

from mdlab import potentials, units


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


heading("8.1 masses and speeds of response")
li = 6.94 * c.atomic_mass / c.m_e
print(
    f"proton/electron {c.m_p / c.m_e:.1f}, sqrt {math.sqrt(c.m_p / c.m_e):.1f}"
)
print(f"lithium (6.94 amu)/electron {li:.0f}, sqrt {math.sqrt(li):.1f}")
print(f"electron mass {c.m_e / c.atomic_mass:.4e} amu")
print(f"check: 0.1 s at n = 1000 -> n = 4000: {0.1 * 4**3:.1f} s")

heading("8.2 quantum steps against k_B T")
kt = units.KB * 300.0
hc = c.h * c.c / c.e * 100.0  # eV cm
print(
    f"hbar = {units.HBAR:.4f} eV fs, hc = {hc:.4e} eV cm, kT(300 K) = "
    f"{kt:.5f} eV, k_B = {units.KB:.4e} eV/K"
)
for name, nu in (
    ("O-H stretch", 3657.0),
    ("LiF", 910.34),
    ("C-H (check)", 3000.0),
):
    step = hc * nu
    print(
        f"{name}: hbar omega = {step:.4f} eV = {step / kt:.2f} kT; "
        f"equal to kT at {step / units.KB:.0f} K"
    )
step = units.HBAR * 2 * math.pi / 170.3
print(
    f"lithium hollow (170.3 fs): hbar omega = {step:.5f} eV = "
    f"{step / kt:.3f} kT at 300 K, {step / (units.KB * 100):.2f} kT at "
    f"100 K; equal to kT at {step / units.KB:.0f} K"
)

heading("8.3 reduced units")
unit_fs = math.sqrt(units.MV2_TO_EV)
print(f"1 Å sqrt(amu/eV) = {unit_fs:.3f} fs")
eps = 120.0 * units.KB
tau_reduced = 3.4 * math.sqrt(39.948 / eps)
print(
    f"argon: eps = {eps:.5f} eV, tau = {tau_reduced:.1f} Å sqrt(amu/eV) "
    f"= {tau_reduced * unit_fs:.0f} fs"
)
eps_kr = 160.0 * units.KB
ratio = (3.6 / 3.4) * math.sqrt((83.8 / 39.948) / (eps_kr / eps))
print(f"check: krypton tau / argon tau = {ratio:.3f}")
omega = 12 / 2 ** (1 / 6)
print(
    f"ex: argon pair omega tau = {omega:.3f}, period "
    f"{2 * math.pi / omega:.4f} tau = "
    f"{2 * math.pi / omega * tau_reduced * unit_fs:.0f} fs, "
    f"{2 * math.pi / omega / 0.005:.1f} steps of 0.005 tau"
)

r_pull = (26 / 7) ** (1 / 6)
pull = float(potentials.lennard_jones(r_pull)[1])
print(f"ex: strongest LJ pull {pull:.4f} eps/sigma at {r_pull:.4f} sigma")
print(
    f"LJ stiffness 57.15 x 2^(1/3) = {36 * 2 ** (2 / 3) * 2 ** (1 / 3):.3f}"
    " eps/r_m²"
)

heading("8.4 virial checks")
for r in (1.0, 1.5):
    phi, dphi = potentials.lennard_jones(r)
    print(
        f"LJ pair at r = {r} sigma: phi' = {float(dphi):.4f}, W = "
        f"{float(-r * dphi):.4f} eps"
    )
print(
    f"ex: best step (3 eps_m 10.75 / 1e4)^(1/3) = "
    f"{(3 * np.finfo(float).eps * 10.75 / 1e4) ** (1 / 3):.2e} sigma; "
    f"sqrt(eps_m) = {math.sqrt(np.finfo(float).eps):.2e}; "
    f"eps_m^(2/3) = {np.finfo(float).eps ** (2 / 3):.1e}"
)

heading("8.5 cutoffs")
phi_c = float(potentials.lennard_jones(2.5)[0])
print(f"phi(2.5 sigma) = {phi_c:.5f} eps; for argon {abs(phi_c) * eps:.3e} eV")
tail = 8 / 3 * math.pi * 0.8 * ((1 / 3) * 0.4**9 - 0.4**3)
print(f"ex: tail energy per atom at n = 0.8, r_c = 2.5: {tail:.4f} eps")
print(f"ex: 0.49 / 0.0163 = {0.49 / abs(phi_c):.1f} pairs")

heading("8.6 second moment")
for z in (8, 12):
    ratio_v = z * (1 - math.sqrt(1 - 1 / z))
    print(f"z = {z}: vacancy / cohesive energy {ratio_v:.4f}")
print(f"check: sqrt(12/2) = {math.sqrt(6):.3f}")

heading("8.7 dihedral")
phi = potentials.dihedral_angle([1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1])
print(
    f"ex: dihedral {math.degrees(phi):.1f} deg, torsion "
    f"{float(potentials.opls_torsion(phi, [0.1, 0.02, 0.05])):.4f} eV"
)

heading("8.8 charges")
print(f"k = {units.COULOMB:.4f} eV Å")
print(
    f"LiF ions at 1.5639 Å: {-units.COULOMB / 1.5639:.3f} eV; unit charges "
    f"at 10 Å: {units.COULOMB / 10:.3f} eV"
)
argon_10 = float(potentials.lennard_jones(10.0, eps, 3.4)[0])
print(f"argon pair at 10 Å: {argon_10:.2e} eV")
print(f"rock salt spacing 2.82 Å: {-1.747565 * units.COULOMB / 2.82:.3f} eV")
print(f"ex: k / kT(300 K) = {units.COULOMB / kt:.0f} Å")
for r in (1.0, 2**0.5, 3**0.5, 2.0):
    print(
        f"sphere sum to {r:.4f} d: "
        f"{potentials.madelung_partial_sum(r + 1e-9, 'sphere'):.3f}"
    )
print(f"cube of half-side 1: {potentials.madelung_partial_sum(1, 'cube'):.4f}")
