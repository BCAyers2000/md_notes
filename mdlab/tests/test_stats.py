"""Statistics of a series: Chapter 15.

Against pymbar.timeseries, where it computes the same quantity, and
against series whose answers are known: the Ornstein-Uhlenbeck step
x' = cx + √(1 − c²)ξ, whose g is (1 + c)/(1 − c), with a transient or a
drift added.
"""

import contextlib
import io

import numpy as np
import pytest

from mdlab.analysis import stats

with contextlib.redirect_stdout(io.StringIO()):
    timeseries = pytest.importorskip("pymbar.timeseries")


def ou(n, c, rng, shape=()):
    """n steps of the OU process with unit variance, started in equilibrium."""
    x = np.empty((n, *shape))
    x[0] = rng.standard_normal(shape)
    kick = np.sqrt(1 - c**2) * rng.standard_normal((n, *shape))
    for k in range(1, n):
        x[k] = c * x[k - 1] + kick[k]
    return x


def test_autocorrelation_matches_direct_sums():
    a = ou(500, 0.7, np.random.default_rng(1))
    d = a - a.mean()
    direct = np.array([np.mean(d[: len(d) - k] * d[k:])
                       for k in range(len(d))]) / np.mean(d * d)
    assert np.allclose(stats.autocorrelation(a), direct, atol=1e-12)


@pytest.mark.parametrize("c", [0.0, 0.5, 0.9])
def test_inefficiency_matches_pymbar(c):
    a = ou(3000, c, np.random.default_rng(2))
    ours = stats.statistical_inefficiency(a)
    theirs = timeseries.statistical_inefficiency(a, fast=False)
    assert ours == pytest.approx(theirs, rel=1e-9)


def test_inefficiency_of_the_ou_step():
    c = 0.8
    a = ou(400_000, c, np.random.default_rng(3))
    assert stats.statistical_inefficiency(a) == pytest.approx(
        (1 + c) / (1 - c), rel=0.05)


def test_error_bars_cover_the_mean_as_often_as_promised():
    rng = np.random.default_rng(4)
    runs = ou(2000, 0.9, rng, shape=(400,))
    inside = 0
    for k in range(runs.shape[1]):
        mean, error, _ = stats.standard_error(runs[:, k])
        inside += abs(mean) < 1.96 * error
    assert 0.92 < inside / runs.shape[1] < 0.98


def test_blocks_reach_the_exact_error():
    c, n = 0.9, 2**17
    a = ou(n, c, np.random.default_rng(5))
    exact = np.sqrt((1 + c) / (1 - c) / n)
    blocks = stats.block_average(a)
    late = (blocks["length"] >= 200) & (blocks["blocks"] >= 32)
    assert np.all(np.abs(blocks["error"][late] - exact)
                  < 3 * blocks["error_error"][late])
    assert blocks["error"][0] < 0.5 * exact  # one sample per block


def test_equilibration_matches_pymbar():
    rng = np.random.default_rng(6)
    for _ in range(3):
        a = ou(600, 0.8, rng) + 4 * np.exp(-np.arange(600) / 40)
        ours, g, _ = stats.detect_equilibration(a)
        with contextlib.redirect_stdout(io.StringIO()):
            theirs, g_theirs, _ = timeseries.detect_equilibration(a,
                                                                  fast=False)
        assert ours == theirs
        assert g == pytest.approx(g_theirs, rel=1e-6)


def test_drift_errors_cover_the_slope():
    rng = np.random.default_rng(7)
    t = np.arange(2000.0)
    inside = 0
    for _ in range(300):
        a = 2e-4 * t + ou(len(t), 0.9, rng)
        slope, error, _ = stats.drift_test(t, a)
        inside += abs(slope - 2e-4) < 1.96 * error
    assert 0.91 < inside / 300 < 0.99


def test_bootstrap_spread_matches_the_error_of_a_mean():
    rng = np.random.default_rng(8)
    a = rng.standard_normal(4000)
    values = stats.block_bootstrap([a], np.mean, 1, 2000, rng)
    assert values.std() == pytest.approx(1 / np.sqrt(4000), rel=0.08)


def test_report_passes_a_settled_series_and_fails_bad_ones():
    rng = np.random.default_rng(9)
    good = ou(20000, 0.9, rng)
    report = stats.convergence_report(good, 10.0, target=0.05)
    assert report.converged and report.start < 2000.0

    # a large slow transient: the start found leaves part of it, which the
    # drift test catches
    transient = ou(20000, 0.9, rng) + 5 * np.exp(-np.arange(20000) / 500)
    report = stats.convergence_report(transient, 10.0)
    assert report.start > 2000.0 and not report.converged

    drifting = ou(20000, 0.9, rng) + 2e-4 * np.arange(20000)
    checks = stats.convergence_report(drifting, 10.0).checks
    assert not checks["|slope| < 2 errors"]

    short = ou(300, 0.95, rng)
    assert not stats.convergence_report(short, 10.0).converged


def test_minimise_reaches_ases_minimum():
    """Steepest descent and ASE's FIRE find the same lattice minimum."""
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.optimize import FIRE

    from mdlab import md, potentials

    eps, sig, r_cut, a = 0.01034, 3.4, 7.0, 5.26
    s = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(3) for j in range(3)
                     for k in range(3)])
    h = 3 * a * np.eye(3)
    r = ((grid[:, None] + s[None]).reshape(-1, 3) / 3) @ h
    r = r + 0.2 * np.random.default_rng(10).standard_normal(r.shape)
    pair = potentials.with_cutoff(
        lambda d: potentials.lennard_jones(d, eps, sig), r_cut, "shift")
    model = md.PairModel(pair, h, r_cut, 0.5)
    ours, energies, largest = md.minimise(model, r, force_tolerance=1e-4)
    assert np.all(np.diff(energies) < 0) and largest[-1] < 1e-4

    atoms = Atoms(f"Ar{len(r)}", positions=r, cell=h, pbc=True)
    atoms.calc = LennardJones(sigma=sig, epsilon=eps, rc=r_cut, smooth=False)
    with contextlib.redirect_stdout(io.StringIO()):
        FIRE(atoms, logfile=None).run(fmax=1e-4, steps=5000)
    # forces below 1e-4 eV/Å leave each energy within about N f²/k of the
    # minimum, 1e-6 eV here
    assert energies[-1] == pytest.approx(atoms.get_potential_energy(),
                                         abs=2e-6)
    assert np.allclose(ours, atoms.positions, atol=1e-3)
