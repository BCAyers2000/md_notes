"""Chapter 1: describing motion, checked against exact results."""

import numpy as np
import pytest

from mdlab import kinematics

G = 9.80665

# Section 1.4: constant acceleration ------------------------------------------


def test_constant_acceleration_in_the_plane_obeys_the_speed_identity():
    t = np.linspace(0.0, 2.0, 21)
    x0, v0, a = np.array([0.0, 1.0]), np.array([3.0, 4.0]), np.array([0, -G])
    x, v = kinematics.constant_acceleration(t, x0, v0, a)
    assert x.shape == v.shape == (21, 2)
    # v² − v0² = 2 a·(x − x0) whenever the acceleration is constant
    assert np.allclose(
        np.sum(v**2, axis=1) - v0 @ v0, 2 * (x - x0) @ a, atol=1e-12
    )


def test_constant_acceleration_refuses_several_bodies_at_once():
    starts = np.zeros((3, 2))  # three bodies in the plane
    with pytest.raises(ValueError):
        kinematics.constant_acceleration([0.0, 1.0], starts, starts, starts)


# Section 1.7: circular motion ------------------------------------------------


def test_circular_motion_is_tangent_and_points_to_the_centre():
    t = np.linspace(0.0, 10.0, 101)
    radius, omega, centre = 1.5, 0.7, np.array([0.3, -0.2])
    r, v, a = kinematics.circular_motion(t, radius, omega, centre)
    outward = r - centre
    assert np.allclose(np.linalg.norm(outward, axis=1), radius)
    assert np.allclose(np.sum(v * outward, axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(v, axis=1), radius * omega)
    assert np.allclose(a, -(omega**2) * outward)
    speed = np.linalg.norm(v, axis=1)
    assert np.allclose(np.linalg.norm(a, axis=1), speed**2 / radius)


def test_circular_velocity_is_the_derivative_of_position():
    t = np.linspace(0.0, 10.0, 101)
    tau = t[1] - t[0]
    radius, omega = 1.5, 0.7
    r, v, _ = kinematics.circular_motion(t, radius, omega)
    leading_error = radius * omega**3 * tau**2 / 6
    estimate = kinematics.central_difference(r, tau)
    assert np.allclose(estimate, v[1:-1], atol=1.01 * leading_error)


# Section 1.9: velocities from saved positions --------------------------------


def test_central_difference_is_exact_for_a_quadratic():
    tau = 0.05
    t = np.arange(0.0, 2.0, tau)
    x, v = kinematics.constant_acceleration(t, 1.0, 12.0, -G)
    estimate = kinematics.central_difference(x, tau)
    assert np.allclose(estimate, v[1:-1], atol=1e-12)


def test_forward_difference_error_is_half_tau_times_acceleration():
    tau = 0.01
    t = np.arange(0.0, 1.0, tau)
    x, v = kinematics.constant_acceleration(t, 0.0, 12.0, -G)
    error = kinematics.forward_difference(x, tau) - v[:-1]
    assert np.allclose(error, -G * tau / 2, atol=1e-10)


def largest_error(scheme, tau, omega=1.0):
    """Largest error of a difference scheme on x = sin(ω t)."""
    t = np.arange(0.0, 3.0 + 2 * tau, tau)
    estimate = scheme(np.sin(omega * t), tau)
    exact = omega * np.cos(omega * t)
    if len(estimate) == len(t) - 2:  # central: interior times
        exact = exact[1:-1]
    else:  # forward: all but the last time
        exact = exact[:-1]
    return np.max(np.abs(estimate - exact))


def test_forward_is_first_order_and_central_second_order():
    taus = (0.04, 0.02, 0.01)
    for scheme, order in (
        (kinematics.forward_difference, 1),
        (kinematics.central_difference, 2),
    ):
        errors = [largest_error(scheme, tau) for tau in taus]
        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        assert np.allclose(rates, order, atol=0.05)


def test_central_difference_of_a_sinusoid_is_scaled_by_sin_wt_over_wt():
    # (sin ω(t+τ) − sin ω(t−τ)) / 2τ = ω cos ωt · sin ωτ / ωτ, exactly
    period, tau = 20.0, 1.0
    omega = 2 * np.pi / period
    t = np.arange(0.0, 60.0, tau)
    estimate = kinematics.central_difference(np.sin(omega * t), tau)
    factor = np.sin(omega * tau) / (omega * tau)
    expected = omega * np.cos(omega * t[1:-1]) * factor
    assert np.allclose(estimate, expected, atol=1e-14)
