"""The radial distribution function and the structure factor (Chapter 16)."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.geometry.rdf import get_rdf

from mdlab.analysis import structure


def _gas(n, side, seed, frames=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, side, (frames, n, 3)), side * np.eye(3)


def test_rdf_matches_ase_whole_and_partial():
    r, h = _gas(120, 12.0, 0, frames=3)
    symbols = ["Na"] * 60 + ["Cl"] * 60
    images = [Atoms(symbols, positions=x, cell=h.T, pbc=True) for x in r]
    ours = structure.rdf(r, h, 5.9, 59)[1]
    theirs = get_rdf(images, 5.9, 59, no_dists=True)
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-12)
    na, cl = np.arange(60), np.arange(60, 120)
    ours = structure.rdf(r, h, 5.9, 59, centres=na, partners=cl)[1]
    theirs = get_rdf(images, 5.9, 59, elements=("Na", "Cl"), no_dists=True)
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-12)


def test_rdf_of_an_ideal_gas_is_one_less_one_over_n():
    n = 200
    r, h = _gas(n, 15.0, 1, frames=200)
    x, g = structure.rdf(r, h, 7.0, 14)
    assert np.mean(g[4:]) == pytest.approx(1 - 1 / n, abs=0.01)


def test_coordination_of_fcc_shells():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(4)
    r, h = atoms.get_positions(), np.array(atoms.get_cell()).T
    x, g = structure.rdf(r, h, 9.0, 900)
    rho = len(r) / np.linalg.det(h)
    n = structure.running_coordination(x, g, rho)
    d = 5.26 / np.sqrt(2)  # nearest-neighbour distance
    assert n[np.searchsorted(x, 1.2 * d)] == pytest.approx(12, abs=1e-9)
    assert n[np.searchsorted(x, 1.2 * np.sqrt(2) * d)] == pytest.approx(
        18, abs=1e-9)


def test_structure_factor_direct_and_at_a_bragg_peak():
    r, h = _gas(50, 10.0, 2)
    q, s = structure.structure_factor(r, h, 1)
    smallest = 2 * np.pi / 10.0
    direct = np.mean([abs(np.sum(np.exp(1j * r[0] @ v))) ** 2 / 50
                      for v in smallest * np.eye(3)])
    assert s[0] == pytest.approx(direct, rel=1e-12)
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(2)
    q, s = structure.structure_factor(
        atoms.get_positions(), np.array(atoms.get_cell()).T, 2)
    bragg = np.argmin(abs(q - 2 * np.pi / 5.26 * np.sqrt(3)))
    assert s[bragg] == pytest.approx(len(atoms), rel=1e-12)


def test_structure_factor_from_a_flat_rdf_is_one():
    x = np.linspace(0.05, 9.95, 100)
    s = structure.structure_factor_from_rdf(x, np.ones_like(x), 0.02,
                                            [0.5, 1.0, 2.0])
    assert s == pytest.approx(1.0, abs=1e-15)


def test_structure_factor_from_a_step_rdf():
    """g = 0 inside σ and 1 beyond has S = 1 − 4πρσ³(sin a − a cos a)/a³."""
    sigma, rho = 3.0, 0.02
    x = np.arange(0.0005, 9.0, 0.001)
    g = (x >= sigma).astype(float)
    q = np.array([0.5, 1.0, 2.0, 3.0])
    s = structure.structure_factor_from_rdf(x, g, rho, q)
    a = q * sigma
    exact = 1 - 4 * np.pi * rho * sigma**3 * (np.sin(a) - a * np.cos(a)) / a**3
    assert s == pytest.approx(exact, abs=2e-3)
