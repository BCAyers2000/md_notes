"""Tests of mdlab.diagnostics: each check on a run with a known fault."""

import math

import numpy as np

from mdlab import diagnostics, integrators, md, potentials
from mdlab.cell import wrap


def _spring(x):
    return 0.5 * float(np.sum(x * x)), -x


def _record(dt, kinetic_factor=1.0, t_f=200.0):
    out = integrators.integrate(_spring, [1.0], [[1.0]], [[0.0]], dt,
                                int(t_f / dt), force_to_accel=1)
    return out["times"], out["potential"], kinetic_factor * out["kinetic"]


def test_fluctuation_falls_as_step_squared_unless_energies_are_wrong():
    healthy = [diagnostics.energy_fluctuation(*_record(dt)[1:])
               for dt in (0.1, 0.05)]
    assert 3.8 < healthy[0] / healthy[1] < 4.2
    # a kinetic energy computed with a factor 2% too large: the
    # fluctuation stays put when the step is halved
    broken = [diagnostics.energy_fluctuation(*_record(dt, 1.02)[1:])
              for dt in (0.1, 0.05)]
    assert broken[1] > 10 * healthy[1]
    assert 0.9 < broken[0] / broken[1] < 1.2


def test_drift_of_a_jumping_cutoff():
    # 64 Lennard-Jones atoms in a cube, reduced units, truncated (energy
    # jumps at the cutoff) against switched
    rng = np.random.default_rng(0)
    grid = np.array(np.meshgrid(*[np.arange(4)] * 3)).reshape(3, -1).T
    h = 4 * 1.55 * np.eye(3)
    r = 1.55 * grid + rng.normal(scale=0.05, size=(64, 3))
    m = np.ones(64)
    v = md.starting_velocities(m, 64 * 1.0, rng, mv2_to_energy=1.0)
    drifts = {}
    for scheme in ("truncate", "switch"):
        pair = potentials.with_cutoff(potentials.lennard_jones, 2.5, scheme,
                                      2.0)
        model = md.PairModel(pair, h, 2.5, 0.4)
        out = md.run(model, m, r, v, h, 0.005, 1000, force_to_accel=1)
        drifts[scheme] = abs(diagnostics.energy_drift(
            out["times"], out["potential"], out["kinetic"]))
    assert drifts["truncate"] > 20 * drifts["switch"]


def test_closest_approach_and_force_catch_an_overlap():
    h = 10.0 * np.eye(3)
    good = 2.0 * np.array(np.meshgrid(*[np.arange(5)] * 3)).reshape(3, -1).T
    bad = good.copy()
    bad[1] = bad[0] + [0.3, 0.0, 0.0]  # one atom placed on another
    pair = potentials.with_cutoff(potentials.lennard_jones, 3.0, "shift")
    for positions, overlapped in ((good, False), (bad, True)):
        closest = diagnostics.closest_approach(positions, h)
        _, f, _ = potentials.pair_energy_forces(positions, pair)
        if overlapped:
            assert closest < 0.5
            assert diagnostics.largest_force(f) > 1e5
        else:
            assert math.isclose(closest, 2.0)
            assert diagnostics.largest_force(f) < 1.0


def test_centre_of_mass_drift():
    rng = np.random.default_rng(2)
    m = rng.uniform(1, 3, 20)
    v = md.starting_velocities(m, 5.0, rng, mv2_to_energy=1.0)
    v_drift = v + [0.1, 0.0, 0.0]
    times = np.linspace(0, 10, 11)
    r0 = rng.uniform(0, 5, (20, 3))
    for velocities, speed in ((v, 0.0), (v_drift, 0.1)):
        path = r0 + times[:, None, None] * velocities  # free flight
        moved = diagnostics.centre_of_mass_path(m, path)
        assert np.allclose(moved, speed * times, atol=1e-12)


def test_jumps_reveal_wrapping_and_unwrap_restores():
    rng = np.random.default_rng(3)
    h = np.array([[8.0, 2.0, 0.0], [0.0, 7.0, 1.0], [0.0, 0.0, 9.0]])
    steps = rng.normal(scale=0.1, size=(300, 10, 3))
    path = rng.uniform(0, 5, (1, 10, 3)) + np.cumsum(steps, axis=0)
    wrapped = np.array([wrap(x, h) for x in path])
    assert diagnostics.largest_jump(path, h) < 0.1
    assert diagnostics.largest_jump(wrapped, h) > 0.5
    assert np.allclose(diagnostics.unwrap(wrapped, h) - wrapped[0],
                       path - path[0], atol=1e-9)
    assert diagnostics.fractional_spread(wrapped[-1], h) <= 1.0
    assert diagnostics.fractional_spread(path[-1] * 3, h) > 1.0


def test_rms_displacement_of_free_flight():
    v = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    path = np.array([t * v for t in range(4)])
    expected = np.arange(4) * math.sqrt((1 + 4) / 2)
    assert np.allclose(diagnostics.rms_displacement(path), expected)


def _healthy_walk(seed=0, frames=60, n=64):
    """Atoms of a 4×4×4 grid, 3 Å apart, wandering by small steps."""
    rng = np.random.default_rng(seed)
    grid = 3.0 * np.array([[i, j, k] for i in range(4) for j in range(4)
                           for k in range(4)], dtype=float)
    return grid[None, :n] + np.cumsum(rng.normal(0, 0.02, (frames, n, 3)),
                                      axis=0)


def test_health_report_passes_a_healthy_record():
    r = _healthy_walk()
    rng = np.random.default_rng(1)
    v = rng.normal(0, 0.01, (len(r), 64, 3))
    report = diagnostics.health_report(r, 12 * np.eye(3), np.ones(64), v,
                                       symbols=["Ar"] * 64)
    assert report.healthy, str(report)


def test_health_report_catches_wrapping_drift_and_shapes():
    from mdlab.cell import wrap as wrap_cell

    r = _healthy_walk(frames=400)
    h = 12 * np.eye(3)
    wrapped = np.array([wrap_cell(x, h) for x in r + 0.05 * np.arange(
        400)[:, None, None]])
    bad = diagnostics.health_report(wrapped, h, np.ones(64))
    names = [k for k, (_, _, ok) in bad.checks.items() if not ok]
    assert any("move between frames" in k for k in names)
    assert any("centre of mass" in k for k in names)
    flat = np.random.default_rng(2).uniform(-1, 1, (1, 2000, 3))
    shape = diagnostics.health_report(np.zeros((2, 2000, 3)), 50 * np.eye(3),
                                      np.ones(2000), flat)
    assert not shape.checks["excess kurtosis of the velocities"][2]


def test_health_report_energy_and_temperature():
    from mdlab.units import KB

    rng = np.random.default_rng(4)
    t = np.arange(2000.0)
    n = 64
    nf = 3 * n - 3
    kinetic = 0.5 * nf * KB * 300 * (1 + np.sqrt(2 / nf)
                                      * rng.normal(size=2000))
    potential = -kinetic + 1e-4 * rng.normal(size=2000)
    r = _healthy_walk(frames=2000, n=n)
    fine = diagnostics.health_report(r, 12 * np.eye(3), times=t,
                                     potential=potential, kinetic=kinetic,
                                     temperature=300, ensemble="nvt")
    assert fine.healthy, str(fine)
    leaking = diagnostics.health_report(r, 12 * np.eye(3), times=t,
                                        potential=potential + 1e-4 * t,
                                        kinetic=kinetic)
    assert not leaking.checks["drift of K + U / its error"][2]
    cold = diagnostics.health_report(r, 12 * np.eye(3), kinetic=kinetic,
                                     temperature=310, ensemble="nvt")
    assert not cold.checks["miss of the mean temperature / its error"][2]
    # at fixed energy only a start far from the target fails
    near = diagnostics.health_report(r, 12 * np.eye(3), kinetic=kinetic,
                                     temperature=310)
    assert near.checks["mean temperature / target, less 1"][2]
    lattice = diagnostics.health_report(r, 12 * np.eye(3), kinetic=kinetic,
                                        temperature=600)
    assert not lattice.checks["mean temperature / target, less 1"][2]


def test_health_report_energies_alone():
    rng = np.random.default_rng(5)
    kinetic = 1.0 + 0.1 * rng.normal(size=500)
    potential = -kinetic + 1e-4 * rng.normal(size=500)
    report = diagnostics.health_report(None, None, times=np.arange(500.0),
                                       potential=potential, kinetic=kinetic)
    assert set(report.checks) == {"spread of K + U / spread of K",
                                  "drift of K + U / its error"}
    assert report.healthy, str(report)


def test_health_report_langevin_leaves_out_the_centre():
    r = _healthy_walk()
    r = r + np.linspace(0, 5, len(r))[:, None, None] * [1.0, 0, 0]
    kept = diagnostics.health_report(r, 30 * np.eye(3), np.ones(r.shape[1]))
    assert not kept.checks["path of the centre of mass / Å"][2]
    free = diagnostics.health_report(r, 30 * np.eye(3), np.ones(r.shape[1]),
                                     momentum_kept=False)
    assert "path of the centre of mass / Å" not in free.checks
