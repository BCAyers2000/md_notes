"""Chapter 9: orders of accuracy, reversal, area, the energy velocity
Verlet keeps, stability, and the equivalence of the Verlet forms."""

import math

import numpy as np
import pytest

from mdlab import integrators

OMEGA = 1.0  # spring with m = k = 1


def spring(r):
    return -r


def spring_energy_forces(r):
    return 0.5 * float(np.sum(r * r)), -r


def run_spring(method, dt, t_end=2.0):
    r, v = np.array([[1.0]]), np.array([[0.0]])
    n = int(round(t_end / dt))
    f = None
    for _ in range(n):
        if method == "velocity_verlet":
            r, v, f = integrators.velocity_verlet_step(
                r, v, [1.0], spring, dt, f, 1.0
            )
        elif method == "euler":
            r, v = integrators.euler_step(r, v, [1.0], spring, dt, 1.0)
        else:
            r, v = integrators.rk4_step(r, v, [1.0], spring, dt, 1.0)
    return abs(r.item() - math.cos(t_end))


@pytest.mark.parametrize(
    ("method", "order"), [("euler", 1), ("velocity_verlet", 2), ("rk4", 4)]
)
def test_global_order(method, order):
    errors = [run_spring(method, dt) for dt in (0.02, 0.01, 0.005)]
    slopes = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert slopes == pytest.approx([order, order], abs=0.1)


def test_reversing_velocity_verlet_retraces_the_path():
    rng = np.random.default_rng(1)
    r0 = rng.normal(size=(3, 2))
    v0 = rng.normal(size=(3, 2))

    def force(r):
        return -(r**3)  # quartic wells, a nonlinear force

    r, v, f = r0, v0, None
    for _ in range(200):
        r, v, f = integrators.velocity_verlet_step(
            r, v, [1, 2, 3], force, 0.05, f, 1.0
        )
    v, f = -v, None
    for _ in range(200):
        r, v, f = integrators.velocity_verlet_step(
            r, v, [1, 2, 3], force, 0.05, f, 1.0
        )
    assert np.allclose(r, r0, rtol=0, atol=1e-13)
    assert np.allclose(-v, v0, rtol=0, atol=1e-13)


def test_verlet_forms_give_the_same_positions():
    dt = 0.1
    r0, v0 = np.array([[1.0]]), np.array([[0.3]])

    def force(r):
        return -np.sin(r)

    rv, vv, f = r0, v0, None
    vv_positions = [r0]
    for _ in range(50):
        rv, vv, f = integrators.velocity_verlet_step(
            rv, vv, [1.0], force, dt, f, 1.0
        )
        vv_positions.append(rv)
    stormer = [r0, vv_positions[1]]
    for _ in range(49):
        stormer.append(
            integrators.position_verlet_step(
                stormer[-2], stormer[-1], [1.0], force, dt, 1.0
            )
        )
    v_half = v0 + 0.5 * dt * force(r0)
    rl, leap = r0 + dt * v_half, [r0]
    leap.append(rl)
    for _ in range(49):
        rl, v_half = integrators.leapfrog_step(
            rl, v_half, [1.0], force, dt, 1.0
        )
        leap.append(rl)
    assert np.allclose(np.array(stormer), np.array(vv_positions), atol=1e-12)
    assert np.allclose(np.array(leap), np.array(vv_positions), atol=1e-12)


def test_velocity_verlet_keeps_area_for_the_pendulum():
    h, dt = 1e-6, 0.4

    def step(q, p):
        r, v, _ = integrators.velocity_verlet_step(
            [[q]], [[p]], [1.0], lambda x: -np.sin(x), dt, None, 1.0
        )
        return r.item(), v.item()

    for q0, p0 in [(0.4, 0.2), (2.5, -1.0)]:
        cols = []
        for dq, dp in [(h, 0.0), (0.0, h)]:
            up, down = step(q0 + dq, p0 + dp), step(q0 - dq, p0 - dp)
            cols.append(
                [(u - d) / (2 * h) for u, d in zip(up, down, strict=True)]
            )
        det = cols[0][0] * cols[1][1] - cols[1][0] * cols[0][1]
        assert det == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.5])
def test_shadow_energy_is_conserved(dt):
    r, v, f = np.array([[1.0]]), np.array([[0.0]]), None
    shadow0 = 0.5 * v.item() ** 2 + 0.5 * r.item() ** 2 * (1 - dt**2 / 4)
    for _ in range(1000):
        r, v, f = integrators.velocity_verlet_step(
            r, v, [1.0], spring, dt, f, 1.0
        )
        shadow = 0.5 * v.item() ** 2 + 0.5 * r.item() ** 2 * (1 - dt**2 / 4)
        assert shadow == pytest.approx(shadow0, rel=1e-12)


def test_stability_boundary():
    for dt, bounded in [(1.99, True), (2.01, False)]:
        r, v, f = np.array([[1.0]]), np.array([[0.0]]), None
        for _ in range(2000):
            r, v, f = integrators.velocity_verlet_step(
                r, v, [1.0], spring, dt, f, 1.0
            )
        assert (abs(r.item()) < 100) == bounded


def test_rk4_loses_energy_on_the_spring():
    out = integrators.integrate(
        spring_energy_forces,
        [1.0],
        [[1.0]],
        [[0.0]],
        0.5,
        400,
        method="rk4",
        force_to_accel=1.0,
    )
    energy = out["potential"] + out["kinetic"]
    assert np.all(np.diff(energy) < 0)
    assert out["force_calls"] == 4 * 400


def test_integrate_velocity_verlet_counts_forces():
    out = integrators.integrate(
        spring_energy_forces,
        [1.0],
        [[1.0]],
        [[0.0]],
        0.1,
        100,
        force_to_accel=1.0,
    )
    assert out["force_calls"] == 101
    energy = out["potential"] + out["kinetic"]
    assert np.ptp(energy) < 0.01 / 4 * 1.01


def test_velocity_verlet_matches_ase():
    """Checked against ASE's VelocityVerlet, in reduced units, where ASE's
    own unit of time needs no conversion."""
    ase = pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones
    from ase.md.verlet import VelocityVerlet

    from mdlab import potentials

    grid = np.array(
        [[i, j, k] for i in range(3) for j in range(2) for k in range(2)],
        float,
    )
    start = 1.12 * grid + 0.03 * np.random.default_rng(9).normal(size=(12, 3))
    v_start = 0.1 * np.random.default_rng(10).normal(size=(12, 3))
    lj = potentials.with_cutoff(potentials.lennard_jones, 6.0, "shift")
    out = integrators.integrate(
        lambda r: potentials.pair_energy_forces(r, lj)[:2],
        np.ones(12),
        start,
        v_start,
        0.005,
        500,
        every=500,
        force_to_accel=1.0,
    )
    atoms = ase.Atoms("H12", positions=start)
    atoms.set_masses(np.ones(12))
    atoms.set_velocities(v_start)
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=6.0, smooth=False)
    VelocityVerlet(atoms, timestep=0.005).run(500)
    assert np.allclose(
        atoms.get_positions(), out["positions"][-1], rtol=0, atol=1e-11
    )
