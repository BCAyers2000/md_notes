"""Chapter 2: Newton's laws, checked against exact results and ASE."""

import numpy as np
import pytest
from scipy.constants import g as G

from mdlab import dynamics, units

# Section 2.10: Newton's law in eV, Å, fs and amu ----------------------------


def test_acceleration_is_force_over_mass_times_unit_factor():
    a = dynamics.acceleration([[1.0, 0.0, -2.0]], [6.94])
    expected = np.array([[1.0, 0.0, -2.0]]) / 6.94 * units.FORCE_TO_ACCEL
    assert np.allclose(a, expected, rtol=1e-15)


def test_acceleration_divides_rows_when_three_atoms_match_three_components():
    # With N = 3, forces / masses would broadcast over columns silently.
    forces = np.ones((3, 3))
    masses = np.array([1.0, 2.0, 4.0])
    a = dynamics.acceleration(forces, masses) / units.FORCE_TO_ACCEL
    assert np.allclose(a, np.repeat([[1.0], [0.5], [0.25]], 3, axis=1))


def test_acceleration_factor_matches_ase():
    # ASE measures time in its own unit; ase.units.fs is 1 fs in that
    # unit, so fs² converts eV/(Å amu) into Å/fs² independently of mdlab.
    ase_units = pytest.importorskip("ase.units")
    a = dynamics.acceleration([[1.0, 0.0, 0.0]], [1.0])[0, 0]
    assert np.isclose(a, ase_units.fs**2, rtol=1e-6)


# Section 2.9: momentum and the centre of mass -------------------------------


def test_momentum_total_momentum_and_centre_of_mass():
    masses = [1.0, 3.0]
    velocities = [[2.0, 0.0], [0.0, 1.0]]
    positions = [[0.0, 0.0], [4.0, 0.0]]
    assert np.array_equal(
        dynamics.momentum(masses, velocities), [[2.0, 0.0], [0.0, 3.0]]
    )
    assert np.allclose(dynamics.total_momentum(masses, velocities), [2, 3])
    assert np.allclose(dynamics.centre_of_mass(masses, positions), [3, 0])


def test_carts_keep_zero_momentum_and_a_fixed_centre_of_mass():
    masses = np.array([1.0, 3.0])
    stiffness, natural_length = 200.0, 0.3

    def spring_between(r):
        stretch = (r[1, 0] - r[0, 0]) - natural_length
        return np.array([[stiffness * stretch], [-stiffness * stretch]])

    r0 = np.array([[0.0], [0.2]])  # spring compressed by 0.1 m
    t = np.linspace(0.0, 0.5, 501)
    r, v = dynamics.solve_newton(
        spring_between, masses, r0, np.zeros_like(r0), t, force_to_accel=1.0
    )
    centre = dynamics.centre_of_mass(masses, r)
    assert np.max(np.abs(dynamics.total_momentum(masses, v))) < 1e-10
    assert np.max(np.abs(centre - centre[0])) < 1e-10
    assert np.max(np.abs(v[:, 0, 0])) > 0.1  # the carts did move


# Section 2.2: frames of reference -------------------------------------------


def test_change_of_frame_subtracts_velocity_and_keeps_acceleration():
    t = np.linspace(0.0, 2.0, 2001)
    r = np.stack([np.zeros_like(t), 12 * t - 0.5 * G * t**2], axis=-1)
    v = np.stack([np.zeros_like(t), 12 - G * t], axis=-1)
    r_new, v_new = dynamics.change_frame(t, r, v, frame_velocity=[-5.0, 0.0])
    assert np.allclose(v_new[:, 0], 5.0)
    assert np.allclose(r_new[:, 0], 5.0 * t)

    dt = t[1] - t[0]
    a = np.gradient(np.gradient(r, dt, axis=0), dt, axis=0)
    a_new = np.gradient(np.gradient(r_new, dt, axis=0), dt, axis=0)
    assert np.allclose(a[2:-2], a_new[2:-2], atol=1e-9)


# Section 2.7: drag ----------------------------------------------------------


def test_drag_motion_matches_the_reference_solver():
    mass, b = 0.01, 0.05
    t = np.linspace(0.0, 1.5, 151)

    def weight_and_drag(r, v):
        return mass * G - b * v

    r, v = dynamics.solve_newton(
        weight_and_drag,
        mass,
        [0.0],
        [0.0],
        t,
        force_to_accel=1.0,
        velocity_dependent=True,
    )
    x_exact, v_exact = dynamics.drag_motion(t, 0.0, 0.0, G, b / mass)
    assert np.allclose(r[:, 0], x_exact, atol=1e-10)
    assert np.allclose(v[:, 0], v_exact, atol=1e-10)


def test_drag_motion_obeys_its_equation_and_both_limits():
    gamma = 5.0
    t = np.linspace(0.0, 3.0, 30001)
    dt = t[1] - t[0]
    x, v = dynamics.drag_motion(t, x0=1.0, v0=4.0, g=G, gamma=gamma)
    # dx/dt = v and dv/dt = g − γ v, by finite differences
    assert np.allclose(np.gradient(x, dt)[1:-1], v[1:-1], atol=1e-6)
    assert np.allclose(
        np.gradient(v, dt)[1:-1], (G - gamma * v)[1:-1], atol=1e-5
    )
    # long after release: the terminal velocity g/γ
    assert np.isclose(v[-1], G / gamma, rtol=1e-6)
    # just after release from rest: free fall, x ≈ g t² / 2
    x_early, _ = dynamics.drag_motion(1e-4, 0.0, 0.0, G, gamma)
    assert np.isclose(x_early, 0.5 * G * 1e-8, rtol=1e-3)


# Section 2.6: the reference solver ------------------------------------------


def test_solver_reproduces_free_fall_in_si_units():
    t = np.linspace(0.0, 2.0, 21)

    def weight(r):
        return np.broadcast_to([0.0, -G * 0.5], r.shape)

    r, _ = dynamics.solve_newton(
        weight, 0.5, [0.0, 0.0], [3.0, 12.0], t, force_to_accel=1.0
    )
    assert np.allclose(r[:, 0], 3.0 * t, atol=1e-12)
    assert np.allclose(r[:, 1], 12.0 * t - 0.5 * G * t**2, atol=1e-10)


def test_solver_reproduces_a_uniform_force_in_metal_units():
    force = np.array([0.3, -0.1, 0.0])  # eV/Å
    mass = 6.94  # amu
    t = np.linspace(0.0, 50.0, 11)  # fs
    v0 = np.array([0.01, 0.0, 0.02])  # Å/fs

    def uniform(r):
        return np.broadcast_to(force, r.shape)

    r, v = dynamics.solve_newton(
        uniform, mass, np.zeros(3), v0, t, force_to_accel=units.FORCE_TO_ACCEL
    )
    a = force / mass * units.FORCE_TO_ACCEL
    tt = t[:, np.newaxis]
    assert np.allclose(r, v0 * tt + 0.5 * a * tt**2, atol=1e-12)
    assert np.allclose(v, v0 + a * tt, atol=1e-13)


# Section 2.8: the spring ----------------------------------------------------


def test_spring_follows_the_sinusoid_fixed_by_its_starting_state():
    # the worked example: 0.5 kg on 50 N/m, from 3 cm moving out at 0.4 m/s
    mass, stiffness, x0, v0 = 0.5, 50.0, 0.03, 0.4
    omega = np.sqrt(stiffness / mass)
    amplitude = np.hypot(x0, v0 / omega)
    phase = np.arcsin(-v0 / (omega * amplitude))
    assert np.isclose(amplitude, 0.05) and np.isclose(phase, -0.9273, 1e-4)

    t = np.linspace(0.0, 2.0, 201)
    r, _ = dynamics.solve_newton(
        lambda r: -stiffness * r, mass, [x0], [v0], t, force_to_accel=1.0
    )
    assert np.allclose(r[:, 0], amplitude * np.cos(omega * t + phase))


# Section 2.10: removing the drift -------------------------------------------


def test_subtracting_p_over_m_leaves_no_total_momentum():
    rng = np.random.default_rng(2026)
    masses = rng.uniform(1.0, 16.0, size=10)
    velocities = rng.normal(scale=0.01, size=(10, 3))
    drift = dynamics.total_momentum(masses, velocities) / masses.sum()
    still = dynamics.total_momentum(masses, velocities - drift)
    assert np.allclose(still, 0.0, atol=1e-15)
