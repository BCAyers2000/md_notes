"""Mean squared displacements and correlation functions (Chapter 16)."""

import numpy as np
import pytest
from ase import Atoms
from ase.md.analysis import DiffusionCoefficient

from mdlab import units
from mdlab.analysis import transport


def _walk(frames, atoms, seed, step=0.3):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, step, (frames, atoms, 3)), axis=0)


def test_msd_against_direct_sums():
    r = _walk(30, 5, 0)
    ours = transport.msd(r, 10, remove_drift=False)
    for k in (1, 4, 10):
        direct = np.mean([(r[t + k] - r[t]) ** 2 for t in range(30 - k)],
                         axis=(0, 1))
        assert ours[k] == pytest.approx(direct, rel=1e-12)


def test_msd_of_a_random_walk_and_a_drift_removed():
    r = _walk(2000, 400, 1, step=0.3)
    m = transport.msd(r, 20).sum(1)
    assert m[20] / 20 == pytest.approx(3 * 0.09, rel=0.03)
    shifted = r + 0.05 * np.arange(2000)[:, None, None]
    assert transport.msd(shifted, 20) == pytest.approx(transport.msd(r, 20),
                                                       rel=1e-9)


def test_diffusion_against_ase():
    r = _walk(200, 40, 2)
    dt = 5.0  # fs between frames
    ase_dt = dt / np.sqrt(1 / units.FORCE_TO_ACCEL)  # in ASE's time unit
    images = [Atoms("Ar40", positions=x, cell=100 * np.eye(3)) for x in r]
    theirs = DiffusionCoefficient(images, ase_dt)
    theirs.calculate()
    slopes = theirs.slopes[0][0] / (dt / ase_dt)  # Å² per fs, per axis
    ours = transport.msd(r, remove_drift=False, origin_step=len(r))
    t = dt * np.arange(len(r))
    for axis in range(3):
        assert np.polyfit(t, ours[:, axis] / 2, 1)[0] == pytest.approx(
            slopes[axis], rel=1e-10)
    d = transport.diffusion_coefficient(t, ours.sum(1), 0.0, t[-1])
    assert d == pytest.approx(np.mean(slopes), rel=1e-10)


def test_correlation_against_direct_sums():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(50, 4, 3)), rng.normal(size=(50, 4, 3))
    c = transport.correlation(a, b, 7)
    for k in (0, 3, 7):
        direct = np.mean([np.sum(a[t] * b[t + k], -1).mean()
                          for t in range(50 - k)])
        assert c[k] == pytest.approx(direct, rel=1e-10)
    m = np.array([1.0, 2.0, 3.0, 4.0])
    weighted = transport.velocity_autocorrelation(a, 5, m)
    direct = np.mean([np.sum(m[:, None] * a[t] * a[t + 5], -1).mean()
                      for t in range(45)])
    assert weighted[5] == pytest.approx(direct, rel=1e-10)


def test_running_integral_of_an_exponential():
    t = np.linspace(0, 20, 2001)
    total = transport.running_integral(np.exp(-t), t[1])
    assert total[-1] == pytest.approx(1 - np.exp(-20), rel=1e-5)


def test_msd_mass_weighted_drift_and_select():
    """The centre of mass is weighted by the masses before atoms are chosen."""
    rng = np.random.default_rng(5)
    r = np.cumsum(rng.normal(0, 0.3, (200, 6, 3)), axis=0)
    m = np.array([1.0, 1, 1, 50, 50, 50])
    centre = np.einsum("i,tix->tx", m / m.sum(), r)[:, None, :]
    ours = transport.msd(r, 10, masses=m, select=[0, 1, 2], lags=[0, 3, 10])
    ref = transport.msd((r - centre)[:, :3], 10, remove_drift=False)
    assert ours == pytest.approx(ref[[0, 3, 10]], rel=1e-12)
