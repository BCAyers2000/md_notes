"""Chapter 10: periodic cells, the nearest image, the Ewald sum,
neighbour lists, the loop and trajectory files."""

import math

import numpy as np
import pytest

from mdlab import cell, ewald, io, md, neighbours, potentials

HEXAGONAL = np.array(
    [
        [2.464, 0.0, 0.0],
        [-1.232, 2.464 * math.sqrt(3) / 2, 0.0],
        [0.0, 0.0, 6.711],
    ]
).T  # graphite's lattice vectors as columns
ROCK_SALT = 1.747565  # Madelung constant, Evjen (1932)


def rock_salt_cell():
    fcc = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    positions = np.vstack([fcc, fcc + [1.0, 0.0, 0.0]])
    charges = np.array([1.0] * 4 + [-1.0] * 4)
    return charges, positions, 2 * np.eye(3)


def test_fractional_round_trip_and_wrap():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=20, size=(50, 3))
    s = cell.to_fractional(r, HEXAGONAL)
    assert np.allclose(cell.to_cartesian(s, HEXAGONAL), r)
    w = cell.to_fractional(cell.wrap(r, HEXAGONAL), HEXAGONAL)
    assert np.all((w >= 0) & (w < 1))
    assert np.allclose(np.round(s - w), s - w)  # moved by whole cells


def test_volume_and_reciprocal_vectors():
    assert cell.cell_volume(HEXAGONAL) == pytest.approx(
        2.464**2 * math.sqrt(3) / 2 * 6.711
    )
    b = cell.reciprocal_vectors(HEXAGONAL)
    assert np.allclose(HEXAGONAL.T @ b.T, 2 * np.pi * np.eye(3))


def test_rounding_finds_the_nearest_image_within_half_the_width():
    rng = np.random.default_rng(1)
    d = cell.to_cartesian(rng.uniform(-3, 3, size=(20000, 3)), HEXAGONAL)
    rounded = cell.minimum_image(d, HEXAGONAL)
    nearest = cell.nearest_image(d, HEXAGONAL)
    short = np.linalg.norm(nearest, axis=1)
    half = 0.5 * cell.perpendicular_widths(HEXAGONAL).min()
    inside = short < half
    assert np.allclose(rounded[inside], nearest[inside])
    worse = np.linalg.norm(rounded, axis=1) > short + 1e-9
    assert worse.any()  # rounding fails in a skewed cell, but only beyond
    assert not (worse & inside).any()


def test_rounding_is_exact_in_an_orthorhombic_cell():
    h = np.diag([3.0, 4.0, 5.0])
    rng = np.random.default_rng(2)
    d = rng.uniform(-10, 10, size=(5000, 3))
    assert np.allclose(cell.minimum_image(d, h), cell.nearest_image(d, h))


@pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0])
def test_ewald_gives_the_madelung_constant_for_any_alpha(alpha):
    q, r, h = rock_salt_cell()
    energy, forces, _ = ewald.ewald_energy_forces(
        q, r, h, alpha=alpha, coulomb=1.0
    )
    assert -energy / 4 == pytest.approx(ROCK_SALT, abs=1e-6)
    assert np.abs(forces).max() < 1e-12  # every ion sits at a centre


def test_ewald_is_the_same_in_a_supercell_and_a_skewed_cell():
    q, r, h = rock_salt_cell()
    e1 = ewald.ewald_energy_forces(q, r, h, coulomb=1.0)[0]
    shifts = [
        np.array([i, j, k])
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    big = np.vstack([r + 2 * s for s in shifts])
    e8 = ewald.ewald_energy_forces(np.tile(q, 8), big, 2 * h, coulomb=1.0)[0]
    primitive = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float).T
    e_p = ewald.ewald_energy_forces(
        [1.0, -1.0], [[0, 0, 0], [1, 0, 0]], primitive, coulomb=1.0
    )[0]
    assert e8 == pytest.approx(8 * e1, rel=1e-9)
    assert e_p == pytest.approx(e1 / 4, rel=1e-9)


def test_ewald_forces_match_differences_in_a_triclinic_cell():
    rng = np.random.default_rng(3)
    r = rng.uniform(0, 5, size=(6, 3))
    q = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    h = np.array([[5, 0, 0], [1.2, 4.5, 0], [0.3, 0.7, 5.5]], float).T
    energy, forces, _ = ewald.ewald_energy_forces(q, r, h)
    assert energy == pytest.approx(
        ewald.ewald_energy_forces(q, r, h, alpha=0.6)[0], rel=1e-9
    )
    numeric = potentials.finite_difference_forces(
        lambda x: ewald.ewald_energy_forces(q, x, h)[0], r, 1e-5
    )
    assert np.allclose(forces, numeric, atol=1e-7)
    assert np.allclose(forces.sum(0), 0, atol=1e-10)


def test_charged_cell_with_background_is_independent_of_alpha():
    rng = np.random.default_rng(4)
    r = rng.uniform(0, 6, size=(5, 3))
    q = np.array([1.0, 1.0, -1.0, 0.5, 0.2])
    h = 6 * np.eye(3)
    energies = [
        ewald.ewald_energy_forces(q, r, h, alpha=a)[0] for a in (0.5, 0.8, 1.2)
    ]
    assert np.ptp(energies) < 1e-8


@pytest.mark.parametrize("h", [10 * np.eye(3), 4.5 * HEXAGONAL])
def test_cell_list_finds_the_same_pairs_as_all_pairs(h):
    rng = np.random.default_rng(5)
    s = rng.uniform(0, 1, size=(400, 3))
    r = cell.to_cartesian(s, h) + rng.normal(scale=3, size=(400, 3))
    r_cut = 0.3 * cell.perpendicular_widths(h).min()
    a = neighbours.all_pairs(r, h, r_cut)
    b = neighbours.cell_list_pairs(r, h, r_cut)
    pairs_a = set(zip(a[0].tolist(), a[1].tolist(), strict=True))
    pairs_b = set(zip(b[0].tolist(), b[1].tolist(), strict=True))
    assert pairs_a == pairs_b and len(pairs_a) > 100


def test_verlet_list_misses_nothing_before_a_rebuild():
    rng = np.random.default_rng(6)
    h = 12 * np.eye(3)
    r = rng.uniform(0, 12, size=(300, 3))
    nl = neighbours.VerletList(h, 2.5, 0.6)
    nl.build(r)
    direction = rng.normal(size=r.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    moved = r + 0.29 * direction  # each atom just under half the skin
    assert not nl.needs_rebuild(moved)
    got = nl.pairs(moved)
    want = neighbours.all_pairs(moved, h, 2.5)
    assert nl.builds == 1
    assert set(zip(got[0].tolist(), got[1].tolist(), strict=True)) == set(
        zip(want[0].tolist(), want[1].tolist(), strict=True)
    )


def test_starting_velocities_have_no_drift():
    m = np.array([39.948] * 10 + [6.94] * 5)
    v = md.starting_velocities(m, 0.5, np.random.default_rng(7))
    assert np.allclose(m @ v, 0, atol=1e-12)


def argon(n=4, a=5.26):
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    grid = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)],
        float,
    )
    s = (grid[:, None, :] + fcc[None, :, :]).reshape(-1, 3) / n
    h = n * a * np.eye(3)
    return cell.to_cartesian(s, h), h


def test_loop_matches_ase_velocity_verlet():
    ase = pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones
    from ase.md.verlet import VelocityVerlet

    from mdlab import units

    fs = math.sqrt(units.FORCE_TO_ACCEL)  # mdlab's fs in ASE's time unit

    eps, sig, rc = 0.0104, 3.4, 8.5
    r, h = argon()
    m = np.full(len(r), 39.948)
    v = md.starting_velocities(
        m, 0.03 * len(r) / 100, np.random.default_rng(8)
    )
    pair = potentials.with_cutoff(
        lambda x: potentials.lennard_jones(x, eps, sig), rc, "shift"
    )
    model = md.PairModel(pair, h, rc, 1.0)
    out = md.run(model, m, r, v, h, 2.0, 50, every=50)

    atoms = ase.Atoms("Ar" * len(r), positions=r, cell=h.T, pbc=True)
    atoms.set_masses(m)
    atoms.set_velocities(v / fs)
    atoms.calc = LennardJones(epsilon=eps, sigma=sig, rc=rc, smooth=False)
    assert atoms.get_potential_energy() == pytest.approx(
        out["potential"][0], abs=1e-10
    )
    VelocityVerlet(atoms, timestep=2.0 * fs).run(50)
    assert np.allclose(
        atoms.get_positions(), out["positions"][-1], rtol=0, atol=1e-12
    )
    assert np.allclose(
        atoms.get_velocities() * fs, out["velocities"][-1], rtol=0, atol=1e-14
    )


def test_extxyz_round_trip_and_ase_reads_it(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    r, h = argon(2)
    v = np.random.default_rng(9).normal(size=r.shape)
    path = str(tmp_path / "frames.extxyz")
    io.write_extxyz(
        path,
        ["Ar"] * len(r),
        [r, r + 0.1],
        h,
        velocities=[v, v],
        scalars={"energy": [-1.5, -1.25], "kinetic_energy": [0.5, 0.25]},
    )
    frames = io.read_extxyz(path)
    assert len(frames) == 2 and frames[1]["energy"] == -1.25
    assert np.allclose(frames[1]["positions"], r + 0.1)
    assert np.allclose(frames[0]["cell"], h)
    atoms = ase_io.read(path, index=-1)
    assert np.allclose(atoms.get_positions(), r + 0.1)
    assert np.allclose(atoms.cell.array, h.T)
    assert atoms.get_potential_energy() == -1.25  # 'energy' is potential
    assert frames[1]["kinetic_energy"] == 0.25


def kernel_case(n=300, seed=14):
    rng = np.random.default_rng(seed)
    side = (n / 0.8) ** (1 / 3)
    k = int(np.ceil(n ** (1 / 3)))
    grid = np.array(
        [[a, b, c] for a in range(k) for b in range(k) for c in range(k)],
        float,
    )[:n]
    r = grid / k * side + rng.normal(scale=0.05, size=(n, 3))
    h = side * np.eye(3)
    i, j, _, _ = neighbours.all_pairs(r, h, 0.49 * side)
    return r, np.full(3, side), i, j


@pytest.mark.parametrize("name", [
    "lj_numpy", "lj_numba",
    pytest.param("lj_fortran", marks=pytest.mark.external),
])
def test_kernels_agree_with_the_plain_loop(name):
    from mdlab import kernels

    if name == "lj_numba":
        pytest.importorskip("numba")
    if name == "lj_fortran":
        import os
        import shutil
        import sys
        from pathlib import Path

        tools = os.pathsep.join((str(Path(sys.executable).parent),
                                 os.environ.get("PATH", "")))
        missing = [tool for tool in ("gfortran", "meson", "ninja")
                   if shutil.which(tool, path=tools) is None]
        if missing:
            pytest.skip("Fortran comparison needs " + ", ".join(missing))
    r, box, i, j = kernel_case()
    want = kernels.lj_loop(r, box, i, j, 1.0, 1.0, 2.5)
    got = getattr(kernels, name)(r, box, i, j, 1.0, 1.0, 2.5)
    assert got[0] == pytest.approx(want[0], rel=1e-11)
    assert np.allclose(got[1], want[1], rtol=0, atol=1e-11)


def test_kernel_matches_the_pair_model():
    from mdlab import kernels

    r, box, i, j = kernel_case()
    pair = potentials.with_cutoff(potentials.lennard_jones, 2.5, "shift")
    energy, forces = md.PairModel(pair, np.diag(box), 2.5, 0.3)(r)
    got = kernels.lj_numpy(r, box, i, j, 1.0, 1.0, 2.5)
    assert got[0] == pytest.approx(energy, rel=1e-12)
    assert np.allclose(got[1], forces, rtol=0, atol=1e-10)


# Checked against ASE, an independent implementation of the same ideas.


def test_cell_quantities_match_ase():
    from ase.cell import Cell
    from ase.geometry import wrap_positions

    rng = np.random.default_rng(17)
    ase_cell = Cell(HEXAGONAL.T)  # ASE holds the lattice vectors as rows
    assert cell.cell_volume(HEXAGONAL) == pytest.approx(ase_cell.volume)
    assert np.allclose(
        cell.reciprocal_vectors(HEXAGONAL), 2 * np.pi * ase_cell.reciprocal()
    )  # ASE leaves out the 2π
    r = rng.normal(scale=15, size=(200, 3))
    assert np.allclose(
        cell.wrap(r, HEXAGONAL), wrap_positions(r, HEXAGONAL.T, eps=0)
    )


def test_nearest_image_matches_ase_find_mic():
    from ase.geometry import find_mic

    rng = np.random.default_rng(18)
    d = cell.to_cartesian(rng.uniform(-3, 3, size=(5000, 3)), HEXAGONAL)
    ase_d, ase_len = find_mic(d, HEXAGONAL.T, pbc=True)
    ours = cell.nearest_image(d, HEXAGONAL)
    assert np.allclose(np.linalg.norm(ours, axis=1), ase_len)
    short = ase_len < 0.5 * cell.perpendicular_widths(HEXAGONAL).min()
    assert np.allclose(cell.minimum_image(d, HEXAGONAL)[short], ase_d[short])


def test_neighbour_pairs_match_ase_neighbor_list():
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    rng = np.random.default_rng(19)
    h = 4.5 * HEXAGONAL
    r = cell.to_cartesian(rng.uniform(0, 1, size=(300, 3)), h)
    r_cut = 0.3 * cell.perpendicular_widths(h).min()
    i, j, _, dist = neighbours.cell_list_pairs(r, h, r_cut)
    atoms = Atoms("Ar" * len(r), positions=r, cell=h.T, pbc=True)
    ai, aj, ad = neighbor_list("ijd", atoms, r_cut)
    keep = ai < aj  # ASE lists every pair both ways
    ours = dict(
        zip(zip(i.tolist(), j.tolist(), strict=True), dist, strict=True)
    )
    theirs = dict(
        zip(
            zip(ai[keep].tolist(), aj[keep].tolist(), strict=True),
            ad[keep],
            strict=True,
        )
    )
    assert ours.keys() == theirs.keys()
    assert np.allclose([ours[k] for k in ours], [theirs[k] for k in ours])


def test_pair_model_matches_ase_lennard_jones():
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    eps, sig, rc = 0.01034, 3.4, 8.5
    r, h = argon()
    r = r + np.random.default_rng(20).normal(scale=0.1, size=r.shape)
    pair = potentials.with_cutoff(
        lambda x: potentials.lennard_jones(x, eps, sig), rc, "shift"
    )
    energy, forces = md.PairModel(pair, h, rc, 1.0)(r)
    atoms = Atoms("Ar" * len(r), positions=r, cell=h.T, pbc=True)
    atoms.calc = LennardJones(epsilon=eps, sigma=sig, rc=rc, smooth=False)
    assert energy == pytest.approx(atoms.get_potential_energy(), abs=1e-12)
    assert np.allclose(forces, atoms.get_forces(), rtol=0, atol=1e-13)


def test_drift_removal_matches_ase_stationary():
    from ase import Atoms
    from ase.md.velocitydistribution import Stationary

    rng = np.random.default_rng(21)
    m = rng.uniform(1, 40, size=30)
    v = rng.normal(size=(30, 3))
    atoms = Atoms("H" * 30, positions=rng.normal(size=(30, 3)))
    atoms.set_masses(m)
    atoms.set_velocities(v)
    Stationary(atoms, preserve_temperature=False)
    assert np.allclose(atoms.get_velocities(), v - (m @ v) / m.sum())


def test_ewald_is_unchanged_by_moving_an_atom_by_a_lattice_vector():
    rng = np.random.default_rng(22)
    r = rng.uniform(0, 5, size=(6, 3))
    q = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    h = np.array([[5, 0, 0], [1.2, 4.5, 0], [0.3, 0.7, 5.5]], float).T
    e0, f0, _ = ewald.ewald_energy_forces(q, r, h)
    moved = r.copy()
    moved[2] += h @ np.array([3.0, -2.0, 4.0])
    e1, f1, _ = ewald.ewald_energy_forces(q, moved, h)
    assert e1 == pytest.approx(e0, abs=1e-10)
    assert np.allclose(f1, f0, rtol=0, atol=1e-10)
