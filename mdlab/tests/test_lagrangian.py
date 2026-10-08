"""Chapter 6: the action is stationary on the true path, and the bead on a
wire against energy conservation and an independent pendulum."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mdlab import energy, lagrangian

G = 9.80665


def free_fall(t, x0=0.0, v0=8.0):
    return x0 + v0 * t - 0.5 * G * t**2


def test_action_converges_to_the_exact_value():
    # x = v0 t - g t²/2 from 0 to T: S = ∫ (½ m v² - m g x) dt in closed form
    m, v0, T = 0.5, 8.0, 1.2
    exact = m * (0.5 * v0**2 * T - v0 * G * T**2 + G**2 * T**3 / 3)
    t = np.linspace(0.0, T, 20001)
    s = lagrangian.action(free_fall(t), t, m, lambda x: m * G * x)
    assert s == pytest.approx(exact, rel=1e-7)


@pytest.mark.parametrize(
    "potential", ["gravity", "spring"], ids=["free fall", "spring"]
)
def test_action_is_stationary_on_the_true_path(potential):
    m, T = 0.5, 0.25
    t = np.linspace(0.0, T, 4001)
    if potential == "gravity":
        true, u = free_fall(t), (lambda x: m * G * x)
    else:  # spring of 50 N/m, omega = 10 rad/s, less than half a period
        true, u = 0.04 * np.cos(10 * t), (lambda x: 25.0 * x**2)
    bump = np.sin(np.pi * t / T)  # zero at both ends

    def s(eps):
        return lagrangian.action(true + eps * bump, t, m, u)

    h = 1e-3
    slope = (s(h) - s(-h)) / (2 * h)
    curvature = (s(h) - 2 * s(0) + s(-h)) / h**2
    assert abs(slope) < 1e-6 * curvature
    assert curvature > 0


def test_wire_acceleration_on_a_straight_slope():
    alpha = math.radians(30)
    a = lagrangian.wire_acceleration(0.0, math.tan(alpha), 0.0, g=G)
    assert float(a) / math.cos(alpha) == pytest.approx(-G * math.sin(alpha))


def hill(x):
    h, slope = energy.hill_track(x)
    curvature = 1.8 * x**2 - 1.2  # h'' of 0.15x⁴ - 0.6x² + 0.15x + 1
    return float(h), float(slope), curvature


def test_bead_on_the_hill_conserves_energy_and_matches_chapter_3():
    t = np.linspace(0.0, 6.0, 3001)
    x, v = lagrangian.solve_wire(hill, -2.0, 0.0, t, g=G)
    h, slope = energy.hill_track(x)
    per_mass = 0.5 * (1 + slope**2) * v**2 + G * h
    assert np.ptp(per_mass) < 1e-8
    h_e = float(energy.hill_track(-2.0)[0])
    speed = np.sqrt((1 + slope**2) * v**2)
    assert np.allclose(
        speed, np.sqrt(2 * G * np.clip(h_e - h, 0, None)), atol=1e-6
    )


def test_bead_in_a_circular_bowl_is_a_pendulum():
    radius, theta0 = 1.2, 0.8

    def bowl(x):
        root = math.sqrt(radius**2 - x**2)
        h = radius - root
        slope = x / root
        curvature = radius**2 / root**3
        return h, slope, curvature

    t = np.linspace(0.0, 3.0, 301)
    x, _ = lagrangian.solve_wire(bowl, radius * math.sin(theta0), 0.0, t, g=G)
    pend = solve_ivp(
        lambda _t, y: [y[1], -G / radius * math.sin(y[0])],
        (0, 3.0),
        [theta0, 0.0],
        t_eval=t,
        rtol=1e-11,
        atol=1e-13,
    )
    assert np.allclose(np.arcsin(x / radius), pend.y[0], atol=1e-8)


def test_rod_multiplier_gives_the_tension_of_the_pendulum():
    m, length, theta0 = 0.5, 1.2, 1.0
    gravity = np.array([0.0, -G])

    def rates(_t, y):
        r, v = y[:2], y[2:]
        lam = lagrangian.rod_multiplier(r, v, m, gravity)
        return [*v, *(gravity + lam * r / m)]

    start = [length * math.sin(theta0), -length * math.cos(theta0), 0, 0]
    t = np.linspace(0.0, 3.0, 301)
    sol = solve_ivp(rates, (0, 3), start, t_eval=t, rtol=1e-11, atol=1e-13)
    r, v = sol.y[:2].T, sol.y[2:].T
    assert np.allclose(np.linalg.norm(r, axis=1), length, atol=1e-7)
    theta = np.arctan2(r[:, 0], -r[:, 1])
    rate = (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]) / length**2  # dθ/dt
    tension = np.array(
        [
            -lagrangian.rod_multiplier(ri, vi, m, gravity) * length
            for ri, vi in zip(r, v, strict=True)
        ]
    )
    expected = m * (G * np.cos(theta) + length * rate**2)
    assert np.allclose(tension, expected, rtol=1e-9)
    pend = solve_ivp(
        lambda _t, y: [y[1], -G / length * math.sin(y[0])],
        (0, 3.0),
        [theta0, 0.0],
        t_eval=t,
        rtol=1e-11,
        atol=1e-13,
    )
    assert np.allclose(theta, pend.y[0], atol=1e-6)
