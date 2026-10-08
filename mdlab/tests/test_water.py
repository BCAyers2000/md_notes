"""Tests of mdlab.water: TIP3P water in a periodic box."""

import math

import numpy as np
import pytest

from mdlab import potentials, units, water


def test_parameters_are_those_of_ase():
    tip3p = pytest.importorskip("ase.calculators.tip3p")
    ase_units = pytest.importorskip("ase.units")
    from ase.data import atomic_masses, atomic_numbers

    assert (tip3p.qH, tip3p.sigma0) == (water.CHARGE_H, water.SIGMA_OO)
    assert (tip3p.rOH, tip3p.angleHOH) == (water.R_OH, water.ANGLE_HOH)
    assert math.isclose(water.EPSILON_OO, tip3p.epsilon0, rel_tol=1e-7)
    assert math.isclose(units.KCAL_PER_MOL,
                        ase_units.kcal / ase_units.mol, rel_tol=1e-7)
    assert math.isclose(water.MASS_O, atomic_masses[atomic_numbers["O"]],
                        rel_tol=1e-4)
    assert math.isclose(water.MASS_H, atomic_masses[atomic_numbers["H"]],
                        rel_tol=1e-3)


def test_box_density_and_geometry():
    r, h, m, symbols = water.box(3, 0.997, np.random.default_rng(0))
    grams = m.sum() / 6.02214076e23
    assert math.isclose(grams / (np.linalg.det(h) * 1e-24), 0.997,
                        rel_tol=1e-12)
    shape = r.reshape(-1, 3, 3)
    oh = np.linalg.norm(shape[:, 1:] - shape[:, :1], axis=2)
    assert np.allclose(oh, water.R_OH)
    assert symbols[:3] == ["O", "H", "H"]


@pytest.mark.parametrize("flexible", [False, True])
def test_forces_are_slopes_of_the_energy(flexible):
    rng = np.random.default_rng(1)
    r, h, m, _ = water.box(2, 0.997, rng)
    r = r + rng.normal(scale=0.03, size=r.shape)
    model = water.WaterModel(h, 8, flexible=flexible, r_cut=3.0,
                             r_switch=2.5, accuracy=1e-10)
    _, f = model(r)
    pick = [0, 4, 11, 19]
    f_fd = potentials.finite_difference_forces(
        lambda x: model(x)[0], r, 1e-5
    )
    assert np.abs(f[pick] - f_fd[pick]).max() < 1e-6


def test_a_lone_molecule_feels_only_its_distant_copies():
    # with each molecule's own pairs removed, a single molecule in a large
    # box has only the small energy of its copies' dipoles
    r = water.molecule() + 5.0
    small = water.WaterModel(20 * np.eye(3), 1, accuracy=1e-10)(r)[0]
    large = water.WaterModel(40 * np.eye(3), 1, accuracy=1e-10)(r)[0]
    assert abs(small) < 2e-3 and abs(large) < 3e-4
    assert 6 < small / large < 10  # falls as the cube of the side


def test_moving_a_molecule_by_a_lattice_vector_changes_nothing():
    rng = np.random.default_rng(2)
    r, h, _, _ = water.box(2, 0.997, rng)
    model = water.WaterModel(h, 8, flexible=True, r_cut=3.0, r_switch=2.5)
    u0, f0 = model(r)
    moved = r.copy()
    moved[3:6] += h @ np.array([1.0, -2.0, 0.0])
    u1, f1 = model(moved)
    assert math.isclose(u0, u1, abs_tol=1e-10)
    assert np.abs(f0 - f1).max() < 1e-9
