"""Tests of the virial and pressure: mdlab.virial and md.PairModel."""

import numpy as np
import pytest

from mdlab import md, neighbours, potentials, virial
from mdlab.cell import cell_volume

EPS, SIG, R_CUT = 0.01034, 3.4, 8.5


def lj_shifted(r):
    return potentials.lennard_jones(r, EPS, SIG)


PAIR = potentials.with_cutoff(lj_shifted, R_CUT, "shift")


def strained_crystal(seed=1):
    """256 argon atoms near fcc, in a cell stretched along x, squeezed in z."""
    a = 5.4
    s = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(4) for j in range(4)
                     for k in range(4)])
    h = np.diag([4 * a * 1.02, 4 * a, 4 * a * 0.97])
    frac = (grid[:, None, :] + s[None]).reshape(-1, 3) / 4
    rng = np.random.default_rng(seed)
    return frac @ h.T + 0.1 * rng.standard_normal((256, 3)), h


def test_virial_tensor_against_ase_stress():
    ase = pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    r, h = strained_crystal()
    model = md.PairModel(PAIR, h, R_CUT, 1.0)
    u, f = model(r)
    atoms = ase.Atoms("Ar256", positions=r, cell=h.T, pbc=True)
    atoms.calc = LennardJones(sigma=SIG, epsilon=EPS, rc=R_CUT, smooth=False)
    assert np.isclose(atoms.get_potential_energy(), u, rtol=1e-12)
    assert np.allclose(atoms.get_forces(), f, atol=1e-12)
    # ASE's stress is −W/V without the kinetic part: the sign is reversed
    stress = atoms.get_stress(voigt=False)
    assert np.allclose(stress, -model.virial / cell_volume(h), atol=1e-14)


def test_virial_tensor_is_minus_the_strain_derivative():
    r, h = strained_crystal(seed=2)
    model = md.PairModel(PAIR, h, R_CUT, 1.0)
    model(r)
    step = 1e-6

    def energy(e):
        strain = np.eye(3) + e
        return md.PairModel(PAIR, strain @ h, R_CUT, 1.0)(r @ strain.T)[0]

    for a in range(3):
        for b in range(3):
            e = np.zeros((3, 3))
            e[a, b] = step
            slope = (energy(e) - energy(-e)) / (2 * step)
            assert np.isclose(slope, -model.virial[a, b], atol=1e-7)


def test_wrapped_virial_depends_on_the_origin():
    r, h = strained_crystal(seed=3)
    model = md.PairModel(PAIR, h, R_CUT, 1.0)
    from mdlab.cell import wrap

    shifted = r + np.array([3.3, -1.7, 5.1])
    _, f0 = model(wrap(r, h))
    w0 = np.trace(model.virial)
    naive0 = virial.wrapped_virial(wrap(r, h), f0)
    _, f1 = model(wrap(shifted, h))
    naive1 = virial.wrapped_virial(wrap(shifted, h), f1)
    assert np.isclose(np.trace(model.virial), w0, rtol=1e-10)
    assert abs(naive1 - naive0) > 1e-3 * abs(w0)


def test_pressure_is_the_trace_over_three_v():
    r, h = strained_crystal(seed=4)
    model = md.PairModel(PAIR, h, R_CUT, 1.0)
    model(r)
    rng = np.random.default_rng(5)
    m = np.full(len(r), 39.948)
    v = 0.003 * rng.standard_normal(r.shape)
    kin = virial.kinetic_tensor(m, v)
    two_k = 2 * 0.5 * 103.6427 * float(np.sum(m[:, None] * v * v))
    assert np.isclose(np.trace(kin), two_k, rtol=1e-6)
    p = virial.pressure(m, v, model.virial, cell_volume(h))
    expect = (np.trace(kin) + np.trace(model.virial)) / (3 * cell_volume(h))
    assert np.isclose(p, expect)


def test_pair_types_use_their_own_functions():
    rng = np.random.default_rng(6)
    h = 20.0 * np.eye(3)
    r = rng.uniform(0, 20, (60, 3))
    types = rng.integers(0, 2, 60)
    soft = potentials.with_cutoff(
        lambda d: potentials.lennard_jones(d, 0.3 * EPS, 1.2 * SIG), R_CUT,
        "shift")
    table = {(0, 0): PAIR, (0, 1): soft, (1, 1): PAIR}
    model = md.PairModel(table, h, R_CUT, 0.5, types=types)
    u, _ = model(r)
    i, j, _, dist = neighbours.all_pairs(r, h, R_CUT)
    mixed = types[i] != types[j]
    expect = PAIR(dist[~mixed])[0].sum() + soft(dist[mixed])[0].sum()
    assert np.isclose(u, expect, rtol=1e-12)


def test_verlet_list_survives_a_changing_cell():
    rng = np.random.default_rng(7)
    h = 22.0 * np.eye(3)
    r = rng.uniform(0, 22, (300, 3))
    nl = neighbours.VerletList(h, 6.0, 1.0)
    nl.pairs(r)
    checked = 0
    for k in range(1, 40):
        strain = np.diag([1 + 0.002 * k, 1 - 0.001 * k, 1 + 0.0015 * k])
        nl.set_cell(strain @ h)
        moved = r @ strain.T + 0.01 * k * rng.standard_normal(r.shape)
        if nl.needs_rebuild(moved):
            break
        i, j, _, _ = nl.pairs(moved)
        bi, bj, _, _ = neighbours.all_pairs(moved, strain @ h, 6.0)
        assert set(zip(i, j, strict=True)) == set(zip(bi, bj, strict=True))
        checked += 1
    assert checked > 3
