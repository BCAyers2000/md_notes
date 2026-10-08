"""Units and physical constants.

mdlab works in energy in eV, length in Å, time in fs, mass in amu and
temperature in K. This is not quite the LAMMPS ``metal`` set, whose time
unit is the ps, nor ASE's internal set, whose time unit is
Å √(amu/eV) ≈ 10.18 fs; times and velocities must be converted wherever
data pass to or from those codes. In mdlab units two of the book's laws
need a conversion factor:

- Newton's second law (Chapter 2): a force in eV/Å divided by a mass in
  amu is not an acceleration in Å/fs², so ``FORCE_TO_ACCEL`` converts it;
- the kinetic energy (Chapter 3): a mass in amu times a velocity squared
  in Å²/fs² is not an energy in eV, so ``MV2_TO_EV`` converts it.

The two factors are each other's inverse. Every factor is computed from
the CODATA values that ship with SciPy, so none is typed in by hand.

Examples
--------
The acceleration of a carbon atom (12.011 amu) under a force of 1 eV/Å:

>>> from mdlab import units
>>> print(f"{1.0 / 12.011 * units.FORCE_TO_ACCEL:.4e} Å/fs²")
8.0331e-04 Å/fs²
"""

from scipy import constants as _c

#: Elementary charge in C; also the size of 1 eV in J.
ELEMENTARY_CHARGE = _c.e
#: Atomic mass constant (1 amu) in kg.
AMU = _c.physical_constants["atomic mass constant"][0]
#: One ångström in m.
ANGSTROM = 1.0e-10
#: One femtosecond in s.
FEMTOSECOND = 1.0e-15

#: Boltzmann constant in eV/K.
KB = _c.k / _c.e

#: Kinetic-energy factor: 1 amu Å² fs⁻² expressed in eV, so that
#: K [eV] = 0.5 * m [amu] * v² [Å²/fs²] * MV2_TO_EV.
MV2_TO_EV = AMU * ANGSTROM**2 / FEMTOSECOND**2 / ELEMENTARY_CHARGE

#: Newton's-law factor: 1 eV Å⁻¹ amu⁻¹ expressed in Å/fs², so that
#: a [Å/fs²] = F [eV/Å] / m [amu] * FORCE_TO_ACCEL.
FORCE_TO_ACCEL = 1.0 / MV2_TO_EV

#: Pressure factor: 1 eV/Å³ expressed in GPa.
EV_PER_A3_TO_GPA = ELEMENTARY_CHARGE / ANGSTROM**3 / 1.0e9

#: Reduced Planck constant in eV fs.
HBAR = _c.hbar / _c.e / FEMTOSECOND

#: Coulomb constant e²/(4πε₀) in eV Å, so that two charges q₁, q₂ (in units
#: of e) at r Å apart have the energy COULOMB q₁ q₂ / r eV.
COULOMB = _c.e / (4 * _c.pi * _c.epsilon_0 * ANGSTROM)

#: Speed of light in cm/fs, for converting wavenumbers in cm⁻¹ to
#: angular frequencies: omega [rad/fs] = 2 pi * C_CM_PER_FS * wavenumber.
C_CM_PER_FS = _c.c * 100.0 * FEMTOSECOND

#: One kilocalorie per mole (the thermochemical calorie, 4.184 J, per
#: Avogadro's number of molecules) in eV, for force fields quoted in it.
KCAL_PER_MOL = 4184.0 / _c.Avogadro / _c.e
