"""Tests of mdlab.constraints: SHAKE, RATTLE and mass repartitioning."""

import math

import numpy as np
import pytest

from mdlab import cell, constraints, md, units, water


def _water_cluster(n, rng):
    """n rigid TIP3P molecules scattered in a cube, and their bonds."""
    shape = water.molecule()
    r = np.concatenate([shape @ _turn(rng).T + rng.uniform(0, 20, 3)
                        for _ in range(n)])
    bonds, lengths = water.constraint_bonds(n)
    m = np.tile([water.MASS_O, water.MASS_H, water.MASS_H], n)
    return r, m, bonds, lengths


def _turn(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_shake_one_bond_mass_weighted():
    r_old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    r_new = np.array([[0.1, 0.2, 0.0], [1.4, 0.1, 0.3]])
    m = np.array([2.0, 5.0])
    r, _ = constraints.shake(r_new, r_old, m, [[0, 1]], [1.0],
                             tolerance=1e-14)
    assert math.isclose(np.linalg.norm(r[1] - r[0]), 1.0, rel_tol=1e-13)
    moves = r - r_new
    # each atom moves along the old bond, the lighter one further
    assert np.allclose(moves[:, 1:], 0.0, atol=1e-15)
    assert math.isclose(moves[0, 0] / moves[1, 0], -m[1] / m[0])
    # the centre of mass does not move
    assert np.allclose(m @ moves, 0.0, atol=1e-15)


def test_shake_one_bond_quadratic_convergence():
    # for one bond the update is Newton's method: each sweep roughly
    # squares the relative error of the length
    r_old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    r = np.array([[0.0, 0.0, 0.0], [1.3, 0.4, 0.0]])
    errors = []
    for _ in range(4):
        r, _ = constraints.shake(r, r_old, [1.0, 1.0], [[0, 1]], [1.0],
                                 tolerance=0.0, max_sweeps=1, strict=False)
        errors.append(abs(np.linalg.norm(r[1] - r[0]) - 1.0))
    ratios = [errors[k + 1] / errors[k] ** 2 for k in range(3)]
    assert all(0.4 < x < 0.7 for x in ratios)
    with pytest.raises(RuntimeError):
        constraints.shake(r, r_old, [1.0, 1.0], [[0, 1]], [1.0],
                          tolerance=0.0, max_sweeps=1)


def test_shake_water_keeps_lengths_and_momentum():
    rng = np.random.default_rng(3)
    r0, m, bonds, lengths = _water_cluster(10, rng)
    r1 = r0 + rng.normal(scale=0.05, size=r0.shape)
    r, sweeps = constraints.shake(r1, r0, m, bonds, lengths,
                                  tolerance=1e-13)
    errors = constraints.bond_errors(r, bonds, lengths)
    assert np.abs(errors).max() < 1e-12
    assert np.allclose(m @ (r - r1), 0.0, atol=1e-12)
    assert 2 < sweeps < 100


def test_shake_after_a_wrap_between_start_and_step():
    # atom 1 crosses the face at x = 0 during the step and is wrapped to
    # the far side; the bond must still be corrected through its copy
    h = 5.0 * np.eye(3)
    r_old = np.array([[4.6, 0.0, 0.0], [0.2, 0.0, 0.0]])  # 0.6 apart
    r_new = np.array([[4.55, 0.0, 0.0], [4.85, 0.1, 0.0]])  # 1 wrapped
    r, _ = constraints.shake(r_new, r_old, [1.0, 1.0], [[0, 1]], [0.6],
                             cell=h, tolerance=1e-14)
    moves = cell.minimum_image(r - r_new, h)
    assert np.abs(moves).max() < 0.2  # small corrections, not a jump
    err = constraints.bond_errors(r, [[0, 1]], [0.6], cell=h)
    assert abs(err[0]) < 1e-12


def test_groups_share_no_atom_and_keep_water_order():
    groups = constraints._groups(np.array([[0, 1], [1, 2], [2, 0],
                                           [3, 4], [4, 5], [5, 3]]))
    assert [g.tolist() for g in groups] == [[0, 3], [1, 4], [2, 5]]
    chain = constraints._groups(np.array([[0, 1], [1, 2], [2, 3]]))
    assert [g.tolist() for g in chain] == [[0, 2], [1]]


def test_shake_uses_nearest_copy_in_a_cell():
    h = 5.0 * np.eye(3)
    r_old = np.array([[0.2, 0.0, 0.0], [4.6, 0.0, 0.0]])  # 0.6 apart
    r_new = np.array([[0.25, 0.0, 0.0], [4.5, 0.0, 0.0]])
    r, _ = constraints.shake(r_new, r_old, [1.0, 1.0], [[0, 1]], [0.6],
                             cell=h, tolerance=1e-14)
    err = constraints.bond_errors(r, [[0, 1]], [0.6], cell=h)
    assert abs(err[0]) < 1e-12


def test_rattle_velocities_remove_bond_rates_keep_momentum():
    rng = np.random.default_rng(4)
    r, m, bonds, lengths = _water_cluster(10, rng)
    v0 = rng.normal(size=r.shape)
    v, _ = constraints.rattle_velocities(r, v0, m, bonds, lengths,
                                         tolerance=1e-14)
    d = r[bonds[:, 1]] - r[bonds[:, 0]]
    rates = np.sum((v[bonds[:, 1]] - v[bonds[:, 0]]) * d, axis=1)
    assert np.abs(rates).max() < 1e-12
    assert np.allclose(m @ v, m @ v0, atol=1e-12)


def _box(rng, n_side=3, r_cut=4.5):
    r, h, m, symbols = water.box(n_side, 0.997, rng)
    n = len(m) // 3
    model = water.WaterModel(h, n, r_cut=r_cut, r_switch=r_cut - 0.5)
    bonds, lengths = water.constraint_bonds(n)
    v = md.starting_velocities(m, 0.05 * n, rng)
    v, _ = constraints.rattle_velocities(r, v, m, bonds, lengths, h,
                                         tolerance=1e-13)
    return r, v, h, m, symbols, model, bonds, lengths


def test_rattle_agrees_with_ase():
    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixBondLengths
    from ase.md.verlet import VelocityVerlet

    class Wrapped(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, model):
            super().__init__()
            self.model = model

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            u, f = self.model(self.atoms.positions)
            self.results = {"energy": u, "forces": f}

    rng = np.random.default_rng(5)
    r, v, h, m, symbols, model, bonds, lengths = _box(rng)
    dt, steps = 1.0, 20
    out = constraints.run(model, m, r, v, h, dt, steps, bonds, lengths,
                          every=steps, tolerance=1e-13)
    atoms = ase.Atoms(symbols, positions=r, cell=h.T, pbc=True, masses=m)
    fs = math.sqrt(units.FORCE_TO_ACCEL)  # mdlab's fs in ASE's time unit
    atoms.set_velocities(v / fs)
    atoms.constraints = FixBondLengths(bonds[:, ::-1], tolerance=1e-13,
                                       bondlengths=lengths)
    atoms.calc = Wrapped(model)
    VelocityVerlet(atoms, timestep=dt * fs).run(steps)
    assert np.abs(atoms.positions - out["positions"][-1]).max() < 1e-10
    v_ase = atoms.get_velocities() * fs
    assert np.abs(v_ase - out["velocities"][-1]).max() < 1e-11


def test_rattle_is_reversible_and_holds_bonds():
    rng = np.random.default_rng(6)
    r, v, h, m, _, model, bonds, lengths = _box(rng)
    forward = constraints.run(model, m, r, v, h, 2.0, 15, bonds, lengths,
                              every=15, tolerance=1e-14)
    back = constraints.run(model, m, forward["positions"][-1],
                           -forward["velocities"][-1], h, 2.0, 15, bonds,
                           lengths, every=15, tolerance=1e-14)
    assert np.abs(back["positions"][-1] - r).max() < 1e-9
    assert forward["bond_error"].max() < 1e-12


def test_constrained_bend_energy_scales_as_step_squared():
    # one flexible water molecule with its O–H bonds held: only the bend,
    # the H–H spring, is left to vibrate
    m = np.array([water.MASS_O, water.MASS_H, water.MASS_H])
    r0 = water.molecule()
    bonds, lengths = water.constraint_bonds(1, rigid=False)
    model = water.WaterModel(40 * np.eye(3), 1, flexible=True)
    v = np.zeros((3, 3))
    v[1] = [0.0, 0.004, 0.0]
    v[2] = [-0.003, 0.0, 0.0]
    v -= m @ v / m.sum()

    def spread(dt):
        out = constraints.run(model.intramolecular, m, r0, v, None, dt,
                              int(400 / dt), bonds, lengths, tolerance=1e-14)
        return np.ptp(out["potential"] + out["kinetic"])

    ratio = spread(0.5) / spread(0.25)
    assert 3.6 < ratio < 4.4


def test_repartition_keeps_total_mass():
    m = np.array([12.011, 1.008, 1.008, 1.008, 1.008])
    new = constraints.repartition_masses(m, [[0, k] for k in range(1, 5)],
                                         3.024)
    assert math.isclose(new.sum(), m.sum())
    assert np.allclose(new[1:], 3.024)
    with pytest.raises(ValueError):
        constraints.repartition_masses(m, [[0, k] for k in range(1, 5)],
                                       5.0)
