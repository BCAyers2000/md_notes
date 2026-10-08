"""Tests of mdlab.statmech: temperatures, thermal velocities, densities."""

import math

import numpy as np
import pytest

from mdlab import statmech, units


def test_degrees_of_freedom():
    assert statmech.degrees_of_freedom(64 * 3, 192) == 381
    assert statmech.degrees_of_freedom(256, drift_removed=False) == 768


def test_thermal_velocities_agree_with_ase():
    ase = pytest.importorskip("ase")
    from ase.md.velocitydistribution import thermalize_momenta

    m = np.random.default_rng(0).uniform(1, 40, 50)
    ours = statmech.thermal_velocities(m, 300.0, np.random.default_rng(7),
                                       remove_drift=False)
    atoms = ase.Atoms("H50", positions=np.zeros((50, 3)), masses=m)
    thermalize_momenta(atoms, 300.0, rng=np.random.default_rng(7))
    theirs = atoms.get_velocities() * math.sqrt(units.FORCE_TO_ACCEL)
    # the same numbers, scaled by Boltzmann's constant of two CODATA years
    assert np.abs(ours / theirs - 1).max() < 1e-6


def test_kinetic_temperature_against_ase():
    ase = pytest.importorskip("ase")
    m = np.full(256, 39.948)
    v = statmech.thermal_velocities(m, 120.0, np.random.default_rng(1))
    ours = statmech.kinetic_temperature(m, v, statmech.degrees_of_freedom(256))
    atoms = ase.Atoms("Ar256", positions=np.zeros((256, 3)), masses=m)
    atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
    # ASE shares the kinetic energy among 3N freedoms, mdlab among 3N - 3
    assert math.isclose(ours * 765 / 768, atoms.get_temperature(),
                        rel_tol=1e-6)


def test_held_bonds_counted_as_ase_counts_them():
    ase = pytest.importorskip("ase")
    from ase.constraints import FixBondLengths

    from mdlab import water

    bonds, _ = water.constraint_bonds(64)
    m = np.tile([15.999, 1.008, 1.008], 64)
    v = statmech.thermal_velocities(m, 300.0, np.random.default_rng(3))
    n_free = statmech.degrees_of_freedom(192, len(bonds))
    ours = statmech.kinetic_temperature(m, v, n_free)
    atoms = ase.Atoms("OH2" * 64, positions=np.zeros((192, 3)), masses=m)
    atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
    atoms.set_constraint(FixBondLengths(bonds))
    # both remove one freedom per held distance; only mdlab removes the drift
    assert math.isclose(ours * 381 / 384, atoms.get_temperature(),
                        rel_tol=1e-6)


def test_kinetic_energy_density_is_scipys_gamma():
    stats = pytest.importorskip("scipy.stats")
    kt = units.KB * 300.0
    for n_free in (3, 30, 765):
        spread = math.sqrt(2 / n_free)  # within four spreads of the mean
        k = 0.5 * n_free * kt * np.linspace(max(1 - 4 * spread, 0.01),
                                            1 + 4 * spread, 50)
        expected = stats.gamma.pdf(k, n_free / 2, scale=kt)
        ours = statmech.kinetic_energy_density(k, n_free, 300.0)
        assert np.allclose(ours, expected, rtol=1e-9, atol=0.0)


def test_thermal_velocities_share_half_kt_per_freedom():
    m = np.repeat([1.008, 15.999, 39.948], 4000)
    v = statmech.thermal_velocities(m, 300.0, np.random.default_rng(2))
    for kind in range(3):
        part = slice(4000 * kind, 4000 * (kind + 1))
        t = statmech.part_temperature(m[part], v[part])
        assert abs(t / 300.0 - 1) < 4 * math.sqrt(2 / 12000)
    assert np.allclose(m @ v, 0.0, atol=1e-10)


def test_maxwell_speed_mean():
    s = np.linspace(0, 0.06, 60001)
    p = statmech.maxwell_speed_density(s, 39.948, 300.0)
    mean = np.trapezoid(s * p, s)
    expect = math.sqrt(8 * units.KB * 300 / (math.pi * 39.948
                                             * units.MV2_TO_EV))
    assert math.isclose(mean, expect, rel_tol=1e-6)


def test_kinetic_energy_density_moments():
    kt = units.KB * 120
    for n_free in (3, 30, 765):
        top = (n_free / 2 + 20 * math.sqrt(n_free / 2) + 30) * kt
        k = np.linspace(1e-12, top, 400001)
        p = statmech.kinetic_energy_density(k, n_free, 120.0)
        mean = np.trapezoid(k * p, k)
        var = np.trapezoid((k - mean) ** 2 * p, k)
        assert math.isclose(mean, n_free * kt / 2, rel_tol=1e-6)
        assert math.isclose(var, n_free * kt**2 / 2, rel_tol=1e-4)


def test_kinetic_energy_density_of_two_freedoms_at_zero():
    # e^{-K/k_BT}/k_BT, finite at K = 0 (Section 12.4's atom on a surface)
    kt = units.KB * 1000.0
    p = statmech.kinetic_energy_density([0.0, kt], 2, 1000.0)
    assert np.allclose(p, [1 / kt, math.exp(-1) / kt])


def test_heat_capacity_formula():
    # a kinetic energy whose variance is half the canonical one means
    # C_V = N_f k_B, the value of N_f springs
    n_free, kt = 100, 0.01
    rng = np.random.default_rng(3)
    k = 50 * kt + rng.standard_normal(200000)
    k = 50 * kt + (k - k.mean()) / k.std() * math.sqrt(n_free / 4) * kt
    cv = statmech.heat_capacity_nve(k, n_free, kb=1.0)
    assert math.isclose(cv, n_free, rel_tol=1e-9)
