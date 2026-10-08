"""Tests of mdlab.respa: multiple time steps and their resonance."""

import math

import numpy as np

from mdlab import integrators, potentials, respa


def _lj(r):
    u, f, _ = potentials.pair_energy_forces(r, potentials.lennard_jones)
    return u, f


def _springs(r):
    # a chain of stiff springs along the atoms' order
    d = r[1:] - r[:-1]
    dist = np.linalg.norm(d, axis=1)
    stretch = dist - 1.2
    pull = (50.0 * stretch / dist)[:, None] * d
    f = np.zeros_like(r)
    f[:-1] += pull
    f[1:] -= pull
    return 25.0 * float(np.sum(stretch**2)), f


def _relaxed(rng):
    # a chain of 12 atoms 1.2 apart along x, a little shaken, with small
    # random velocities and no drift
    r = 1.2 * np.arange(12)[:, None] * np.array([1.0, 0, 0])
    v = rng.normal(scale=0.05, size=(12, 3))
    return r + rng.normal(scale=0.02, size=r.shape), v - v.mean(0)


def test_one_inner_step_is_velocity_verlet():
    rng = np.random.default_rng(1)
    r, v = _relaxed(rng)
    m = np.ones(len(r))

    def total(x):
        u1, f1 = _springs(x)
        u2, f2 = _lj(x)
        return u1 + u2, f1 + f2

    out = respa.run(_springs, _lj, m, r, v, 0.002, 1, 50,
                    every=50, force_to_accel=1)
    rr, vv, f = r.copy(), v.copy(), None
    for _ in range(50):
        rr, vv, f = integrators.velocity_verlet_step(
            rr, vv, m, lambda x: total(x)[1], 0.002, f, force_to_accel=1
        )
    assert np.abs(out["positions"][-1] - rr).max() < 1e-12
    assert np.abs(out["velocities"][-1] - vv).max() < 1e-12


def test_respa_is_reversible():
    rng = np.random.default_rng(2)
    r, v = _relaxed(rng)
    m = np.ones(len(r))
    go = respa.run(_springs, _lj, m, r, v, 0.01, 5, 40, every=40,
                   force_to_accel=1)
    back = respa.run(_springs, _lj, m, go["positions"][-1],
                     -go["velocities"][-1], 0.01, 5, 40, every=40,
                     force_to_accel=1)
    assert np.abs(back["positions"][-1] - r).max() < 1e-10
    assert go["fast_calls"] == 1 + 40 * 5 and go["slow_calls"] == 41


def test_outer_matrix_trace_and_determinant():
    for dt in (0.1, 0.7, 1.5, 3.0):
        m = respa.outer_step_matrix(dt, 2.0, 0.6)
        theta = 2.0 * dt
        trace = 2 * math.cos(theta) - 0.36 * dt / 2.0 * math.sin(theta)
        assert math.isclose(np.trace(m), trace, abs_tol=1e-12)
        assert math.isclose(np.linalg.det(m), 1.0, abs_tol=1e-12)
    # many inner steps approach the exact fast motion
    exact = respa.outer_step_matrix(0.7, 2.0, 0.6)
    fine = respa.outer_step_matrix(0.7, 2.0, 0.6, n_inner=400)
    assert np.abs(exact - fine).max() < 1e-4


def test_respa_steps_follow_the_outer_matrix():
    # one coordinate, fast spring ω_f = 2 and slow ω_s = 0.6, stepped by
    # respa_step: every outer step must be the matrix product
    def fast(q):
        return 2.0 * float(np.sum(q * q)), -4.0 * q

    def slow(q):
        return 0.18 * float(np.sum(q * q)), -0.36 * q

    for dt, n in ((0.7, 10), (1.5, 20)):
        out = respa.run(fast, slow, [1.0], [[1.0]], [[0.3]], dt, n, 30,
                        force_to_accel=1)
        m = respa.outer_step_matrix(dt, 2.0, 0.6, n_inner=n)
        state = np.array([1.0, 0.3])
        for k in range(31):
            q, v = out["positions"][k, 0, 0], out["velocities"][k, 0, 0]
            assert np.allclose([q, v], state, rtol=1e-10, atol=1e-12)
            state = m @ state


def test_resonance_band_lies_just_below_half_the_fast_period():
    omega_f, omega_s = 1.0, 0.3
    half_period = math.pi / omega_f
    traces = {
        x: abs(np.trace(respa.outer_step_matrix(x * half_period, omega_f,
                                                omega_s)))
        for x in (0.95, 0.99, 1.0, 1.01, 1.05, 0.5)
    }
    assert traces[0.99] > 2 and traces[0.95] > 2
    assert traces[1.01] < 2 and traces[1.05] < 2 and traces[0.5] < 2
