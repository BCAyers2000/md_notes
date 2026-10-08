"""Tests of mdlab.chain: harmonic and FPUT chains."""

import numpy as np

from mdlab import chain, potentials


def test_forces_are_slopes():
    rng = np.random.default_rng(0)
    q = rng.normal(scale=0.3, size=12)
    _, f = chain.energy_forces(q, 0.25)
    fd = potentials.finite_difference_forces(
        lambda x: chain.energy_forces(x.ravel(), 0.25)[0], q[:, None], 1e-6
    ).ravel()
    assert np.abs(f - fd).max() < 1e-8


def test_modes_solve_the_chain():
    n = 9
    hessian = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    patterns, omega = chain.normal_modes(n)
    assert np.allclose(hessian @ patterns, patterns * omega**2)


def test_harmonic_modes_never_share():
    patterns, _ = chain.normal_modes(32)
    out = chain.run(3.0 * patterns[:, 0], np.zeros(32), 0.0, 0.05, 4000,
                    every=100)
    others = out["modes"][:, 1:]
    assert others.max() < 1e-20
    first = out["modes"][:, 0]
    assert np.ptp(first) / first.mean() < 1e-4


def test_fput_shares_and_keeps_energy():
    patterns, _ = chain.normal_modes(32)
    out = chain.run(4.0 * patterns[:, 0], np.zeros(32), 0.25, 0.05, 20000,
                    every=200)
    assert np.ptp(out["energy"]) / out["energy"].mean() < 1e-3
    share = out["modes"][:, 0] / out["modes"].sum(1)
    assert share.min() < 0.9  # energy has begun to leave mode 1
