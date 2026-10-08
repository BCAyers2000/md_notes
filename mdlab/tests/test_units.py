"""Units against ASE, which derives its own from CODATA independently.

ASE's default constants are CODATA 2014 and SciPy's are newer, so each
factor is checked tightly against ASE's set for SciPy's CODATA year, and
loosely against ASE's default.
"""

import math

import pytest
from scipy import constants

from mdlab import units

ase_units = pytest.importorskip("ase.units")


def _matching_codata():
    """ASE's constant set for the CODATA year SciPy uses, if ASE has it."""
    amu = constants.physical_constants["atomic mass constant"][0]
    for year, table in ase_units.CODATA.items():
        if table["_amu"] == amu and table["_e"] == constants.e:
            return ase_units.create_units(year)
    pytest.skip("ASE has no constant set matching SciPy's CODATA year")


def test_force_to_acceleration_is_ase_fs_squared():
    # ASE's time unit is Å sqrt(amu/eV), so fs² in it is the Newton factor
    ase = _matching_codata()
    assert math.isclose(units.FORCE_TO_ACCEL, ase.fs**2, rel_tol=1e-12)


def test_boltzmann_constant():
    ase = _matching_codata()
    assert math.isclose(units.KB, ase.kB, rel_tol=1e-12)


def test_pressure_factor():
    ase = _matching_codata()
    assert math.isclose(units.EV_PER_A3_TO_GPA, 1.0 / ase.GPa, rel_tol=1e-12)


def test_hbar_in_ev_fs():
    ase = _matching_codata()
    assert math.isclose(units.HBAR, ase._hbar / ase._e * 1e15, rel_tol=1e-12)


def test_ase_default_constants_differ_only_at_codata_revision_level():
    # ASE's default set is CODATA 2014; k_B moved by 3.4e-7 when the SI
    # fixed it exactly in 2019, the largest change among these constants
    assert math.isclose(units.KB, ase_units.kB, rel_tol=1e-6)
    assert math.isclose(units.FORCE_TO_ACCEL, ase_units.fs**2, rel_tol=1e-6)


def test_kinetic_and_force_factors_are_inverse():
    assert math.isclose(
        units.MV2_TO_EV * units.FORCE_TO_ACCEL, 1.0, rel_tol=1e-15
    )


def test_thermal_speed_of_argon():
    # sqrt(kB T / m) for Ar at 300 K is 249.9 m/s = 2.499e-3 Å/fs
    m_ar = 39.948
    v = math.sqrt(units.KB * 300.0 / (m_ar * units.MV2_TO_EV))
    v_si = math.sqrt(
        constants.k
        * 300.0
        / (m_ar * constants.physical_constants["atomic mass constant"][0])
    )
    assert math.isclose(v, v_si * 1e10 / 1e15, rel_tol=1e-12)
