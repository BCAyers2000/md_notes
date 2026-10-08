"""Chapter 8: pair potentials and their forces, cutoffs, the virial,
many-body and bonded terms, and the slow sum of charges."""

import math

import numpy as np
import pytest

from mdlab import potentials, units


def cluster(n=12, seed=4, spacing=1.15):
    """N atoms near a loose cubic arrangement, in reduced units."""
    rng = np.random.default_rng(seed)
    grid = np.array(
        [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
        dtype=float,
    )[:n]
    return spacing * grid + 0.08 * rng.normal(size=(n, 3))


@pytest.mark.parametrize(
    "pair",
    [
        potentials.lennard_jones,
        lambda r: potentials.morse(r, 1.0, 1.7, 1.1),
        lambda r: potentials.buckingham(r, 400.0, 0.25, 2.0),
    ],
)
def test_pair_slopes_match_differences(pair):
    r = np.linspace(0.9, 3.0, 50)
    h = 1e-6
    numeric = (pair(r + h)[0] - pair(r - h)[0]) / (2 * h)
    assert np.allclose(pair(r)[1], numeric, rtol=1e-6, atol=1e-8)


def test_lennard_jones_minimum_and_stiffness():
    rmin = 2 ** (1 / 6)
    phi, dphi = potentials.lennard_jones(rmin)
    assert phi == pytest.approx(-1.0)
    assert dphi == pytest.approx(0.0, abs=1e-12)
    h = 1e-4
    k = (
        potentials.lennard_jones(rmin + h)[0]
        - 2 * phi
        + potentials.lennard_jones(rmin - h)[0]
    ) / h**2
    assert k == pytest.approx(36 * 2 ** (2 / 3), rel=1e-6)  # 57.15 ε/σ²


def test_morse_stiffness_is_2_d_a_squared():
    d, a, r0, h = 2.0, 1.8, 1.5, 1e-4
    k = (
        potentials.morse(r0 + h, d, a, r0)[0]
        - 2 * potentials.morse(r0, d, a, r0)[0]
        + potentials.morse(r0 - h, d, a, r0)[0]
    ) / h**2
    assert k == pytest.approx(2 * d * a**2, rel=1e-6)


@pytest.mark.parametrize("scheme", ["truncate", "shift", "switch"])
def test_pair_forces_match_differences(scheme):
    pair = potentials.with_cutoff(potentials.lennard_jones, 2.5, scheme)
    r = cluster()
    u, f, w = potentials.pair_energy_forces(r, pair)
    numeric = potentials.finite_difference_forces(
        lambda x: potentials.pair_energy_forces(x, pair)[0], r, 1e-5
    )
    assert np.allclose(f, numeric, atol=1e-6)
    assert f.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-10)
    assert w == pytest.approx(float(np.sum(r * f)), rel=1e-10)


@pytest.mark.parametrize(
    "pair",
    [
        lambda r: potentials.morse(r, 1.0, 1.7, 1.1),
        lambda r: potentials.buckingham(r, 400.0, 0.25, 2.0),
    ],
)
def test_morse_and_buckingham_forces_match_differences(pair):
    r = cluster()
    f = potentials.pair_energy_forces(r, pair)[1]
    numeric = potentials.finite_difference_forces(
        lambda x: potentials.pair_energy_forces(x, pair)[0], r, 1e-5
    )
    assert np.allclose(f, numeric, atol=1e-5)


def test_virial_does_not_depend_on_the_origin():
    r = cluster()
    w0 = potentials.pair_energy_forces(r, potentials.lennard_jones)[2]
    w1 = potentials.pair_energy_forces(
        r + [3.0, -2.0, 7.5], potentials.lennard_jones
    )[2]
    assert w1 == pytest.approx(w0, rel=1e-12)


def test_cutoff_schemes_continuity():
    rc = 2.5
    lj = potentials.lennard_jones
    trunc = potentials.with_cutoff(lj, rc, "truncate")
    shift = potentials.with_cutoff(lj, rc, "shift")
    switch = potentials.with_cutoff(lj, rc, "switch", r_switch=2.0)
    below, above = rc - 1e-9, rc + 1e-9
    assert trunc(below)[0] == pytest.approx(float(lj(rc)[0]), abs=1e-8)
    assert abs(float(lj(rc)[0])) > 0.016  # the jump of truncation
    assert shift(below)[0] == pytest.approx(0.0, abs=1e-8)
    assert abs(float(shift(below)[1])) > 0.03  # its slope still jumps
    assert switch(below)[0] == pytest.approx(0.0, abs=1e-8)
    assert switch(below)[1] == pytest.approx(0.0, abs=1e-7)
    assert switch(1.5)[0] == pytest.approx(float(lj(1.5)[0]))
    for f in (trunc, shift, switch):
        assert f(above)[0] == 0.0


def test_reduced_units_scale_out():
    # the same arrangement in argon units: energies scale with ε, forces
    # with ε/σ
    eps, sig = 120.0 * units.KB, 3.4
    r = cluster()

    def argon(d):
        return potentials.lennard_jones(d, eps, sig)

    u1, f1, _ = potentials.pair_energy_forces(r, potentials.lennard_jones)
    u2, f2, _ = potentials.pair_energy_forces(sig * r, argon)
    assert u2 == pytest.approx(eps * u1)
    assert np.allclose(f2, eps / sig * f1)


def test_second_moment_forces_and_sqrt_scaling():
    params = dict(repulsion=0.1, hopping=1.2, p=10.0, q=2.0, r0=1.0)
    r = cluster(10, seed=2, spacing=1.0)
    u, f = potentials.second_moment(r, **params)
    numeric = potentials.finite_difference_forces(
        lambda x: potentials.second_moment(x, **params)[0], r, 1e-5
    )
    assert np.allclose(f, numeric, atol=1e-6)
    assert f.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-10)
    # an atom with z neighbours at r0 has the density ξ²z, so its bond
    # energy −√ρ is −ξ√z: it grows as √z, not as z
    for z in (1, 4, 9):
        angles = 2 * np.pi * np.arange(z) / z
        star = np.vstack(
            [
                [0.0, 0.0, 0.0],
                np.column_stack([np.cos(angles), np.sin(angles), np.zeros(z)]),
            ]
        )
        d = np.linalg.norm(star[1:] - star[0], axis=1)
        rho = np.sum(
            params["hopping"] ** 2
            * np.exp(-2 * params["q"] * (d / params["r0"] - 1))
        )
        assert math.sqrt(rho) == pytest.approx(
            params["hopping"] * math.sqrt(z)
        )


def test_dihedral_angle_and_torsion():
    ref = [[0, 1, 0], [0, 0, 0], [1, 0, 0]]
    for deg in (-150.0, -60.0, 0.0, 45.0, 120.0, 180.0):
        t = math.radians(deg)
        d = [1, math.cos(t), math.sin(t)]
        assert math.degrees(
            potentials.dihedral_angle(*ref, d)
        ) == pytest.approx(deg if deg != -180.0 else 180.0, abs=1e-9)
    v = [1.0, -0.5, 0.3]
    phi = np.linspace(-np.pi, np.pi, 9)
    e = potentials.opls_torsion(phi, v)
    expected = (
        0.5 * v[0] * (1 + np.cos(phi))
        + 0.5 * v[1] * (1 - np.cos(2 * phi))
        + 0.5 * v[2] * (1 + np.cos(3 * phi))
    )
    assert np.allclose(e, expected)
    assert potentials.bond_angle(
        [1, 0, 0], [0, 0, 0], [0, 2, 0]
    ) == pytest.approx(math.pi / 2)


def test_coulomb_and_madelung():
    coulomb = units.COULOMB
    assert coulomb == pytest.approx(14.3996, abs=1e-4)
    cube = potentials.madelung_partial_sum(8, "cube")
    assert cube == pytest.approx(1.747565, abs=2e-5)
    spheres = [
        potentials.madelung_partial_sum(rad, "sphere")
        for rad in (5.0, 5.5, 6.0, 6.5, 7.0)
    ]
    assert max(spheres) - min(spheres) > 1.0  # the sphere does not settle


def test_pair_forces_match_ase_calculators():
    """Checked against ASE's Lennard-Jones and Morse calculators."""
    ase = pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones
    from ase.calculators.morse import MorsePotential

    r = cluster()
    atoms = ase.Atoms("Ar" * len(r), positions=r)
    shifted = potentials.with_cutoff(potentials.lennard_jones, 2.5, "shift")
    u, f, _ = potentials.pair_energy_forces(r, shifted)
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5, smooth=False)
    assert u == pytest.approx(atoms.get_potential_energy(), abs=1e-12)
    assert np.allclose(f, atoms.get_forces(), rtol=0, atol=1e-12)
    u, f, _ = potentials.pair_energy_forces(
        r, lambda x: potentials.morse(x, 1.0, 6.0, 1.1)
    )
    atoms.calc = MorsePotential(
        epsilon=1.0, r0=1.1, rho0=6.0 * 1.1, rcut1=100.0, rcut2=101.0
    )
    assert u == pytest.approx(atoms.get_potential_energy(), abs=1e-12)
    assert np.allclose(f, atoms.get_forces(), rtol=0, atol=1e-12)
