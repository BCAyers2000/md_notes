"""Chapter 5: angular momentum against the reference solver and finite
differences, and the inertia tensor and the removal of rigid motion
against ASE, an independent implementation."""

import numpy as np
import pytest
from ase import Atoms
from ase.md.velocitydistribution import Stationary, ZeroRotation

from mdlab import dynamics, oscillators, rotation, units

rng = np.random.default_rng(2026)

# water at its measured geometry (Section 5.5), O first, in Å and amu
R_OH, ANGLE = 0.958, np.radians(104.4776)
WATER = np.array(
    [
        [0.0, 0.0, 0.0],
        [R_OH * np.sin(ANGLE / 2), R_OH * np.cos(ANGLE / 2), 0.0],
        [-R_OH * np.sin(ANGLE / 2), R_OH * np.cos(ANGLE / 2), 0.0],
    ]
)
WATER_MASSES = np.array([15.999, 1.008, 1.008])


def random_cluster(n=5):
    return rng.normal(size=(n, 3)), rng.uniform(1.0, 20.0, n)


def rotation_matrix(axis, angle):
    """Rodrigues' formula, used only to turn test geometries."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return (
        np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    )


def test_central_force_conserves_angular_momentum():
    def force(r):  # a spring pulling towards the origin, plus r⁻² attraction
        d = np.linalg.norm(r, axis=-1, keepdims=True)
        return -3.0 * r - 2.0 * r / d**3

    t = np.linspace(0.0, 10.0, 401)
    r, v = dynamics.solve_newton(
        force,
        [2.0],
        [[1.0, 0.3, -0.2]],
        [[0.1, 0.8, 0.5]],
        t,
        force_to_accel=1.0,
    )
    ang = rotation.angular_momentum([2.0], r, v)
    assert np.allclose(ang, ang[0], atol=1e-8)


def test_rate_of_change_of_angular_momentum_is_the_torque():
    masses = np.array([1.0, 2.5, 0.7])
    forces = rng.normal(size=(3, 3))

    def path(t):  # constant forces: r = r0 + v0 t + ½ F t²/m
        r = r0 + v0 * t + 0.5 * forces / masses[:, None] * t**2
        v = v0 + forces / masses[:, None] * t
        return r, v

    r0, v0 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    origin = np.array([0.3, -0.2, 0.9])
    h, t = 1e-5, 0.4
    rp, vp = path(t + h)
    rm, vm = path(t - h)
    rate = (
        rotation.angular_momentum(masses, rp, vp, origin)
        - rotation.angular_momentum(masses, rm, vm, origin)
    ) / (2 * h)
    r, _ = path(t)
    assert np.allclose(rate, rotation.torque(r, forces, origin), atol=1e-8)


def test_principal_moments_match_ase():
    positions, masses = random_cluster(6)
    centre = dynamics.centre_of_mass(masses, positions)
    ours = np.linalg.eigvalsh(
        rotation.inertia_tensor(masses, positions, centre)
    )
    atoms = Atoms("X6", positions=positions, masses=masses)
    assert np.allclose(
        ours, np.sort(atoms.get_moments_of_inertia()), rtol=1e-10
    )


def test_water_principal_moments_match_measured_rotational_constants():
    # B = ħ / (4π c I): the measured A, B, C of 27.877, 14.512 and
    # 9.285 cm⁻¹ belong to the vibrating molecule, so agree to about 3%
    centre = dynamics.centre_of_mass(WATER_MASSES, WATER)
    moments = np.linalg.eigvalsh(
        rotation.inertia_tensor(WATER_MASSES, WATER, centre)
    )
    hbar_amu = units.HBAR * units.FORCE_TO_ACCEL  # ħ in amu Å² fs⁻¹
    constants = hbar_amu / (4 * np.pi * units.C_CM_PER_FS * moments)
    assert np.allclose(constants, [27.877, 14.512, 9.285], rtol=0.03)


def test_rigid_rotation_gives_angular_momentum_i_omega():
    positions, masses = random_cluster()
    omega = np.array([0.3, -1.2, 0.5])
    velocities = np.cross(omega, positions)
    ang = rotation.angular_momentum(masses, positions, velocities)
    assert np.allclose(ang, rotation.inertia_tensor(masses, positions) @ omega)
    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    inertia = rotation.inertia_tensor(masses, positions)
    assert kinetic == pytest.approx(0.5 * omega @ inertia @ omega)


def test_angular_velocity_recovers_a_rigid_rotation():
    positions, masses = random_cluster()
    centre = dynamics.centre_of_mass(masses, positions)
    omega, drift = np.array([0.4, 0.1, -0.7]), np.array([1.0, 2.0, 3.0])
    velocities = drift + np.cross(omega, positions - centre)
    found = rotation.angular_velocity(masses, positions, velocities)
    assert np.allclose(found, omega)


def test_removal_zeroes_momentum_and_angular_momentum():
    positions, masses = random_cluster(7)
    velocities = rng.normal(size=(7, 3))
    v = rotation.remove_rigid_motion(masses, positions, velocities)
    centre = dynamics.centre_of_mass(masses, positions)
    assert np.allclose(dynamics.total_momentum(masses, v), 0.0, atol=1e-12)
    assert np.allclose(
        rotation.angular_momentum(masses, positions, v, centre),
        0.0,
        atol=1e-12,
    )
    again = rotation.remove_rigid_motion(masses, positions, v)
    assert np.allclose(again, v, atol=1e-12)


def test_removal_matches_ase():
    positions, masses = random_cluster(6)
    velocities = rng.normal(size=(6, 3))
    atoms = Atoms("X6", positions=positions, masses=masses)
    atoms.set_velocities(velocities)
    Stationary(atoms, preserve_temperature=False)
    ZeroRotation(atoms, preserve_temperature=False)
    ours = rotation.remove_rigid_motion(masses, positions, velocities)
    assert np.allclose(ours, atoms.get_velocities(), atol=1e-10)


def test_removal_of_drift_only_keeps_rotation():
    positions, masses = random_cluster()
    velocities = rng.normal(size=(5, 3))
    v = rotation.remove_rigid_motion(
        masses, positions, velocities, rotation=False
    )
    assert np.allclose(v, velocities - masses @ velocities / masses.sum())


def test_kinetic_energy_splits_into_three_parts():
    positions, masses = random_cluster(6)
    velocities = rng.normal(size=(6, 3))
    total = masses.sum()
    drift = masses @ velocities / total
    centre = dynamics.centre_of_mass(masses, positions)
    omega = rotation.angular_velocity(masses, positions, velocities)
    inertia = rotation.inertia_tensor(masses, positions, centre)
    rest = rotation.remove_rigid_motion(masses, positions, velocities)

    def kinetic(v):
        return 0.5 * np.sum(masses * np.sum(v**2, axis=1))

    parts = 0.5 * total * drift @ drift + 0.5 * omega @ inertia @ omega
    assert kinetic(velocities) == pytest.approx(parts + kinetic(rest))


def test_linear_molecule_rotation_about_its_axis_is_ignored():
    positions = np.array(
        [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]
    )
    masses = np.array([15.999, 12.011, 15.999])
    velocities = rng.normal(size=(3, 3))
    omega = rotation.angular_velocity(masses, positions, velocities)
    assert omega[2] == pytest.approx(0.0, abs=1e-12)
    v = rotation.remove_rigid_motion(masses, positions, velocities)
    centre = dynamics.centre_of_mass(masses, positions)
    assert np.allclose(
        rotation.angular_momentum(masses, positions, v, centre),
        0.0,
        atol=1e-12,
    )


def spring_energy(flat, pairs, stiffnesses, lengths):
    r = flat.reshape(-1, 3)
    d = np.linalg.norm(r[pairs[:, 1]] - r[pairs[:, 0]], axis=1)
    return 0.5 * np.sum(stiffnesses * (d - lengths) ** 2)


def test_spring_hessian_matches_finite_differences():
    positions = rng.normal(size=(4, 3))
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]])
    k = rng.uniform(1.0, 5.0, len(pairs))
    lengths = np.linalg.norm(
        positions[pairs[:, 1]] - positions[pairs[:, 0]], axis=1
    )
    flat, h = positions.ravel(), 1e-4
    numeric = np.zeros((12, 12))
    for a in range(12):
        for b in range(12):
            e_a, e_b = np.eye(12)[a] * h, np.eye(12)[b] * h
            numeric[a, b] = (
                spring_energy(flat + e_a + e_b, pairs, k, lengths)
                - spring_energy(flat + e_a - e_b, pairs, k, lengths)
                - spring_energy(flat - e_a + e_b, pairs, k, lengths)
                + spring_energy(flat - e_a - e_b, pairs, k, lengths)
            ) / (4 * h * h)
    assert np.allclose(
        rotation.spring_hessian(positions, pairs, k), numeric, atol=1e-6
    )


@pytest.mark.parametrize(
    ("positions", "pairs", "zeros"),
    [
        (WATER, [[0, 1], [0, 2], [1, 2]], 6),
        ([[0, 0, 0], [0, 0, 1.564]], [[0, 1]], 5),
    ],
)
def test_rigid_motions_are_the_zero_modes(positions, pairs, zeros):
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    hessian = rotation.spring_hessian(
        positions, pairs, np.full(len(pairs), 40.0)
    )
    masses = np.repeat(rng.uniform(1.0, 20.0, n), 3)
    w2, _ = oscillators.normal_modes(hessian, masses, force_to_accel=1.0)
    assert np.sum(np.abs(w2) < 1e-9) == zeros
    assert np.all(w2[zeros:] > 1e-3)
    for axis in np.eye(3):  # each rigid displacement is turned into zero
        assert np.allclose(hessian @ np.tile(axis, n), 0.0, atol=1e-12)
        turn = np.cross(axis, positions).ravel()
        assert np.allclose(hessian @ turn, 0.0, atol=1e-12)


def test_turning_a_molecule_leaves_its_moments_unchanged():
    positions, masses = random_cluster()
    turned = positions @ rotation_matrix([1.0, 2.0, -0.5], 0.8).T
    before = np.linalg.eigvalsh(rotation.inertia_tensor(masses, positions))
    after = np.linalg.eigvalsh(rotation.inertia_tensor(masses, turned))
    assert np.allclose(before, after)
