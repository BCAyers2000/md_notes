"""Chapter 4: oscillators against the reference solver, finite
differences and independent closed forms."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from mdlab import dynamics, energy, oscillators, units

rng = np.random.default_rng(2026)


def reference(force, x0, v0, t, mass=1.0, velocity_dependent=False):
    r, v = dynamics.solve_newton(
        force,
        mass,
        [x0],
        [v0],
        t,
        force_to_accel=1.0,
        velocity_dependent=velocity_dependent,
    )
    return r[:, 0], v[:, 0]


def test_harmonic_matches_the_reference_solver():
    omega, t = 3.0, np.linspace(0.0, 5.0, 501)
    x, v = oscillators.harmonic(t, 0.4, -1.1, omega)
    xr, vr = reference(lambda r: -(omega**2) * r, 0.4, -1.1, t)
    assert np.allclose(x, xr, atol=1e-9) and np.allclose(v, vr, atol=1e-9)


@pytest.mark.parametrize("gamma", [1.0, 6.0, 9.0])  # under, critical, over
def test_damped_matches_the_reference_solver(gamma):
    omega0, t = 3.0, np.linspace(0.0, 6.0, 601)
    x, v = oscillators.damped(t, 0.5, 0.7, omega0, gamma)

    def force(r, w):
        return -(omega0**2) * r - gamma * w

    xr, vr = reference(force, 0.5, 0.7, t, velocity_dependent=True)
    assert np.allclose(x, xr, atol=1e-9) and np.allclose(v, vr, atol=1e-9)


def test_driven_motion_settles_to_the_steady_state():
    omega0, gamma, f, omega = 2.0, 0.4, 1.0, 1.7
    t = np.linspace(0.0, 120.0, 12001)

    def rates(time, y):  # the force depends on time, so solve directly
        drive = f * np.cos(omega * time)
        return [y[1], -(omega0**2) * y[0] - gamma * y[1] + drive]

    sol = solve_ivp(
        rates,
        (0, 120),
        [0.0, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    amplitude, lag = oscillators.driven_steady_state(omega, omega0, gamma, f)
    late = t > 100.0  # transients have decayed as e^(−γt/2)
    steady = amplitude * np.cos(omega * t[late] - lag)
    assert np.allclose(sol.y[0][late], steady, atol=1e-7)


@pytest.mark.parametrize(
    "well",
    [
        lambda r: oscillators.morse(r, 4.5, 1.9, 0.74),
        lambda r: oscillators.lennard_jones(r, 0.0104, 3.40),
        lambda r: oscillators.cosine_well(r, 0.3, 2.46),
    ],
)
def test_forces_are_minus_the_slope(well):
    r = rng.uniform(2.0, 4.0, size=50)
    _, force = well(r)
    h = 1e-6
    numerical = -(well(r + h)[0] - well(r - h)[0]) / (2 * h)
    assert np.allclose(force, numerical, atol=1e-6)


def test_lennard_jones_minimum_and_curvature():
    eps, sigma = 1.0, 1.0
    r_min = 2 ** (1 / 6) * sigma
    u, f = oscillators.lennard_jones(r_min, eps, sigma)
    assert math.isclose(u, -eps) and abs(f) < 1e-12
    h = 1e-4
    curvature = (
        oscillators.lennard_jones(r_min + h, eps, sigma)[0]
        - 2 * u
        + oscillators.lennard_jones(r_min - h, eps, sigma)[0]
    ) / h**2
    assert math.isclose(
        curvature, 72 * eps / (2 ** (1 / 3) * sigma**2), rel_tol=1e-6
    )


def test_morse_curvature_is_2_d_a_squared():
    d, a, r0, h = 2.0, 1.5, 1.0, 1e-4
    curvature = (
        oscillators.morse(r0 + h, d, a, r0)[0]
        - 2 * oscillators.morse(r0, d, a, r0)[0]
        + oscillators.morse(r0 - h, d, a, r0)[0]
    ) / h**2
    assert math.isclose(curvature, 2 * d * a**2, rel_tol=1e-6)


def test_harmonic_period_does_not_depend_on_energy():
    k, m = 50.0, 0.5
    for e in (1e-4, 0.04, 3.0):
        t = oscillators.period(
            lambda x: 0.5 * k * x**2, e, m, 0.0, 0.01, force_to_accel=1.0
        )
        assert math.isclose(t, 2 * np.pi / np.sqrt(k / m), rel_tol=1e-10)


def test_morse_period_follows_its_closed_form():
    # an independent result: T = 2π / (ω₀ √(1 − E/D)), ω₀ = a √(2D/m)
    d, a, r0, m = 1.0, 1.0, 1.0, 1.0
    omega0 = a * np.sqrt(2 * d / m)
    for fraction in (0.01, 0.3, 0.6, 0.9):
        t = oscillators.period(
            lambda r: oscillators.morse(r, d, a, r0)[0],
            fraction * d,
            m,
            r0,
            0.01,
            force_to_accel=1.0,
            nodes=400,
        )
        exact = 2 * np.pi / (omega0 * np.sqrt(1 - fraction))
        assert math.isclose(t, exact, rel_tol=1e-8)


def test_pendulum_period_follows_the_elliptic_integral():
    # T = (4/ω₀) K(sin²(θ₀/2)), with K the complete elliptic integral
    g, length = 9.80665, 1.2
    omega0 = np.sqrt(g / length)

    def u(theta):  # per unit mass and length², so mass = 1
        return omega0**2 * (1 - np.cos(theta))

    for amplitude in (0.1, 1.0, 2.5):
        t = oscillators.period(
            u, u(amplitude), 1.0, 0.0, 0.01, force_to_accel=1.0, nodes=400
        )
        exact = 4 / omega0 * ellipk(np.sin(amplitude / 2) ** 2)
        assert math.isclose(t, exact, rel_tol=1e-8)


def test_period_matches_a_timed_motion():
    # Li in the model surface's hop coordinate, timed by the reference solver
    ub, a = energy.SURFACE_BARRIER, energy.SURFACE_SPACING

    def u(x):
        return oscillators.cosine_well(x, ub, a)[0]

    e = 0.2  # eV, below the 0.3 eV barrier
    t_period = oscillators.period(
        u, e, 6.94, 0.0, 0.01, force_to_accel=units.FORCE_TO_ACCEL
    )
    speed = math.sqrt(2 * e * units.FORCE_TO_ACCEL / 6.94)
    t = np.linspace(0.0, 1.5 * t_period, 30001)
    r, v = dynamics.solve_newton(
        lambda x: oscillators.cosine_well(x, ub, a)[1],
        6.94,
        [0.0],
        [speed],
        t,
        force_to_accel=units.FORCE_TO_ACCEL,
    )
    crossings = t[1:][(r[:-1, 0] < 0) & (r[1:, 0] >= 0)]  # upward zeros
    assert math.isclose(crossings[0], t_period, rel_tol=1e-4)


def test_cosine_well_matches_the_surface_along_the_hop():
    x = np.linspace(-1.2, 1.2, 50)
    on_line = np.stack([x, np.zeros_like(x)], axis=-1)
    u_surface, f_surface = energy.hexagonal_surface(on_line)
    u, f = oscillators.cosine_well(
        x, energy.SURFACE_BARRIER, energy.SURFACE_SPACING
    )
    assert np.allclose(u, u_surface) and np.allclose(f, f_surface[:, 0])
    assert np.allclose(f_surface[:, 1], 0.0, atol=1e-12)  # stays on the line


def test_reduced_mass_gives_the_carts_quarter_period():
    mu = oscillators.reduced_mass(1.0, 3.0)
    assert math.isclose(
        0.5 * np.pi * np.sqrt(mu / 200.0), 0.0962, abs_tol=1e-4
    )


def test_normal_modes_of_a_linear_triatomic():
    k, mo, mc = 1.0, 16.0, 12.0
    hessian = k * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], float)
    masses = np.array([mo, mc, mo])
    w2, modes = oscillators.normal_modes(hessian, masses, force_to_accel=1.0)
    assert np.allclose(
        w2, [0.0, k / mo, k / mo * (1 + 2 * mo / mc)], atol=1e-12
    )
    for j in range(3):  # each pattern solves Φ u = ω² M u
        assert np.allclose(
            hessian @ modes[:, j], w2[j] * masses * modes[:, j], atol=1e-12
        )
    assert np.allclose(modes[:, 0] / modes[0, 0], 1.0)  # free translation
