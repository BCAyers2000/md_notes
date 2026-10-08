"""Bonded terms: energies and forces (Chapter 16)."""

import numpy as np
import pytest

from mdlab import potentials
from mdlab.forcefield import BondedModel

KAPPA = [0.03, -0.01, 0.06]


def _chain():
    model = BondedModel([[0, 1], [1, 2], [2, 3]], 2.0, 1.53,
                        [[0, 1, 2], [1, 2, 3]], 3.0, np.radians(112),
                        [[0, 1, 2, 3]], KAPPA)
    rng = np.random.default_rng(0)
    r = np.array([[0, 0, 0], [1.5, 0, 0], [2.0, 1.4, 0.2], [3.4, 1.6, 0.9]])
    return model, r + rng.normal(0, 0.1, (4, 3))


def test_forces_are_minus_the_gradient():
    model, r = _chain()
    _, f = model(r)
    fd = np.zeros_like(r)
    for a in range(4):
        for x in range(3):
            step = np.zeros_like(r)
            step[a, x] = 1e-6
            fd[a, x] = -(model(r + step)[0] - model(r - step)[0]) / 2e-6
    assert f == pytest.approx(fd, abs=1e-8)


def test_torsion_energy_and_copies():
    model, r = _chain()
    torsion_only = BondedModel(np.zeros((0, 2)), 0, 0, np.zeros((0, 3)), 0,
                               0, [[0, 1, 2, 3]], KAPPA)
    psi = potentials.dihedral_angle(*r)
    u, _ = torsion_only(r)
    assert u == pytest.approx(float(potentials.opls_torsion(psi, KAPPA)))
    copies = np.stack([r, r[::-1]])
    u2, f2 = model(copies)
    assert u2.shape == (2,)
    assert u2[0] == pytest.approx(model(r)[0])
    assert f2[0] == pytest.approx(model(r)[1])


def test_bond_and_angle_energies():
    """k(r − r₀)² for a bond stretched by 0.2 Å, and an angle bent by 10°."""
    model = BondedModel([[0, 1]], 2.0, 1.5, [[0, 1, 2]], 3.0,
                        np.radians(100), np.zeros((0, 4)), [0.0])
    r = np.array([[0.0, 0, 0], [1.7, 0, 0], [1.7, 1.0, 0]])  # 1.7 Å, 90°
    u, _ = model(r)
    assert u == pytest.approx(2.0 * 0.2**2 + 3.0 * np.radians(10) ** 2)
