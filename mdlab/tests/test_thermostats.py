"""Tests of mdlab.thermostats: the loop, the canonical distribution, ASE."""

import math

import numpy as np
import pytest

from mdlab import md, potentials, thermostats, units
from mdlab.thermostats import (
    CSVR,
    Andersen,
    Berendsen,
    Langevin,
    NoseHooverChain,
    Rescale,
)

KT = 1.0  # reduced units throughout the harmonic tests
REDUCED = dict(kb=1.0, mv2_to_energy=1.0)


def harmonic(r):
    """Independent 3-D oscillators of unit stiffness, one per replica."""
    return 0.5 * np.einsum("...ix,...ix->...", r, r), -r


def replicas(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 1, 3)), rng.standard_normal((n, 1, 3))


def argon(n_cells=2, seed=1):
    """A small argon crystal, slightly disordered, with its model."""
    a = 5.26
    s = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(n_cells) for j in range(n_cells)
                     for k in range(n_cells)])
    r = ((grid[:, None, :] + s[None]).reshape(-1, 3)) * a
    rng = np.random.default_rng(seed)
    r = r + 0.05 * rng.standard_normal(r.shape)
    h = n_cells * a * np.eye(3)
    pair = potentials.with_cutoff(
        lambda d: potentials.lennard_jones(d, 0.01034, 3.4), 5.0, "switch",
        4.5)
    model = md.PairModel(pair, h, 5.0, 0.2)
    m = np.full(len(r), 39.948)
    v = 0.003 * rng.standard_normal(r.shape)
    v -= v.mean(0)
    return model, m, r, v, h


def test_no_thermostat_is_velocity_verlet():
    model, m, r, v, h = argon()
    ours = thermostats.run(model, m, r, v, 5.0, 40)
    plain = md.run(model, m, r, v, h, 5.0, 40)
    assert np.allclose(ours["positions"], plain["positions"], atol=1e-10)
    assert np.all(ours["heat"] == 0)


@pytest.mark.parametrize("make", [
    lambda: Rescale(100.0, 93),
    lambda: Berendsen(100.0, 100.0, 93),
    lambda: Andersen(100.0, 0.01),
    lambda: Langevin(100.0, 0.01),
    lambda: NoseHooverChain(100.0, 100.0, 93),
    lambda: CSVR(100.0, 100.0, 93),
])
def test_effective_energy_is_kept(make):
    model, m, r, v, _ = argon()
    out = thermostats.run(model, m, r, v, 5.0, 400, make(),
                          np.random.default_rng(2), every=10)
    effective = out["potential"] + out["kinetic"] - out["heat"]
    # the run gains or loses heat, but E - heat moves only by the
    # integration error, far below the heat exchanged
    assert np.ptp(effective) < 0.05 * max(np.ptp(out["heat"]), 1e-3)


def test_rescale_holds_k_before_each_step():
    v = np.random.default_rng(0).standard_normal((5, 4, 3))
    out = Rescale(KT, 12, **REDUCED).before(v, np.ones(4), 0.1, None)
    assert np.allclose(thermostats.kinetic(np.ones(4), out, 1.0), 6.0)


@pytest.mark.parametrize("make, exact", [
    (lambda: Andersen(KT, 0.5, **REDUCED), True),
    (lambda: Langevin(KT, 0.5, **REDUCED), True),
    (lambda: NoseHooverChain(KT, 2.0, 3, chain=3, **REDUCED), True),
    (lambda: CSVR(KT, 2.0, 3, **REDUCED), True),
    (lambda: Berendsen(KT, 2.0, 3, **REDUCED), False),
])
def test_canonical_kinetic_energy(make, exact):
    r, v = replicas(2000)
    out = thermostats.run(harmonic, np.ones(1), r, v, 0.05, 6000, make(),
                          np.random.default_rng(1), every=50,
                          keep=(), force_to_accel=1.0)
    k = out["kinetic"][20:].ravel()  # past the first 50 units of time
    # the gamma distribution of 3 freedoms: mean 3/2, variance 3/2
    assert math.isclose(k.mean(), 1.5, rel_tol=0.02)
    if exact:
        assert math.isclose(k.var(), 1.5, rel_tol=0.05)
    else:
        assert k.var() < 1.2  # Berendsen narrows it


def test_nose_hoover_alone_fails_on_one_oscillator():
    # one-dimensional oscillators, each with its own thermostat: under
    # Nose-Hoover each trajectory keeps its own <x^2>, scattered about the
    # canonical k_BT = 1; under a chain of four every trajectory finds it
    rng = np.random.default_rng(3)
    r, v = rng.standard_normal((100, 1, 1)), np.zeros((100, 1, 1))
    spread = {}
    for chain in (1, 4):
        th = NoseHooverChain(KT, 1.0, 1, chain=chain, **REDUCED)
        out = thermostats.run(harmonic, np.ones(1), r, v, 0.05, 20000, th,
                              every=10, keep=("positions",),
                              force_to_accel=1.0)
        x = out["positions"][200:, :, 0, 0]
        spread[chain] = np.mean(x**2, 0).std()
    assert spread[1] > 0.15 > 0.08 > spread[4]


def test_nose_hoover_chain_conserves_extended_energy():
    r, v = replicas(4, seed=5)
    nhc = NoseHooverChain(KT, 1.0, 3, chain=3, **REDUCED)
    out = thermostats.run(harmonic, np.ones(1), r, v, 0.01, 3000, nhc,
                          keep=(), force_to_accel=1.0)
    effective = out["potential"] + out["kinetic"] - out["heat"]
    # velocity Verlet keeps the energy to about dt^2/8 of itself
    assert np.abs(effective / effective[0] - 1).max() < 1e-4
    extended = out["potential"][-1] + out["kinetic"][-1] + nhc.energy()
    assert np.allclose(extended, effective[0], rtol=1e-4)


# ---------------------------------------------------------------- ASE ---


def _ase_atoms(model, m, r, v, h):
    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes

    class Wrapped(Calculator):
        implemented_properties = ["energy", "forces", "free_energy"]

        def calculate(self, atoms=None, properties=None,
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            u, f = model(self.atoms.positions)
            self.results = {"energy": u, "free_energy": u, "forces": f}

    atoms = ase.Atoms(f"Ar{len(m)}", positions=r, cell=h, pbc=True, masses=m)
    atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
    atoms.calc = Wrapped()
    return atoms


ASE_FS = 1 / math.sqrt(units.FORCE_TO_ACCEL)  # one ASE time unit, in fs


def _compare(atoms, dyn, ours, steps):
    dyn.run(steps)
    v_ase = atoms.get_velocities() * math.sqrt(units.FORCE_TO_ACCEL)
    # the two libraries' values of k_B differ by 3.4e-7
    assert np.allclose(atoms.positions, ours["positions"][-1], atol=1e-6)
    assert np.allclose(v_ase, ours["velocities"][-1], rtol=1e-4, atol=1e-8)


def test_berendsen_against_ase():
    from ase.md.nvtberendsen import NVTBerendsen

    model, m, r, v, h = argon()
    atoms = _ase_atoms(model, m, r, v, h)
    dyn = NVTBerendsen(atoms, 5.0 / ASE_FS, temperature_K=120.0,
                       taut=200.0 / ASE_FS, fixcm=False)
    ours = thermostats.run(model, m, r, v, 5.0, 50,
                           Berendsen(120.0, 200.0, 3 * len(m)))
    _compare(atoms, dyn, ours, 50)


def test_nose_hoover_chain_against_ase():
    from ase.md.nose_hoover_chain import NoseHooverChainNVT

    model, m, r, v, h = argon()
    atoms = _ase_atoms(model, m, r, v, h)
    dyn = NoseHooverChainNVT(atoms, 5.0 / ASE_FS, temperature_K=120.0,
                             tdamp=100.0 / ASE_FS, tchain=3)
    ours = thermostats.run(model, m, r, v, 5.0, 50,
                           NoseHooverChain(120.0, 100.0, 3 * len(m)))
    _compare(atoms, dyn, ours, 50)


def test_langevin_baoab_against_ase():
    from ase.md.langevinbaoab import LangevinBAOAB

    model, m, r, v, h = argon()
    atoms = _ase_atoms(model, m, r, v, h)
    dyn = LangevinBAOAB(atoms, 5.0 / ASE_FS, temperature_K=120.0,
                        T_tau=200.0 / ASE_FS, rng=np.random.default_rng(4))
    ours = thermostats.run(model, m, r, v, 5.0, 50, Langevin(120.0, 1 / 200),
                           np.random.default_rng(4))
    _compare(atoms, dyn, ours, 50)


def test_csvr_against_ase_bussi():
    from ase.md.bussi import Bussi

    model, m, r, v, h = argon()
    atoms = _ase_atoms(model, m, r, v, h)
    dyn = Bussi(atoms, 5.0 / ASE_FS, temperature_K=120.0,
                taut=100.0 / ASE_FS, rng=np.random.default_rng(6))
    ours = thermostats.run(model, m, r, v, 5.0, 50,
                           CSVR(120.0, 100.0, 3 * len(m)),
                           np.random.default_rng(6))
    _compare(atoms, dyn, ours, 50)


def test_andersen_against_ase_in_distribution():
    # ASE redraws single components, mdlab whole velocities: the random
    # numbers differ, so the two are compared by the temperature they hold
    from ase.md.andersen import Andersen as AseAndersen

    model, m, r, v, h = argon()
    rate, n_steps, skip = 0.01, 8000, 1000
    ours, theirs = [], []
    for seed in range(2):
        atoms = _ase_atoms(model, m, r, v, h)
        dyn = AseAndersen(atoms, 5.0 / ASE_FS, temperature_K=120.0,
                          andersen_prob=-math.expm1(-rate * 5.0), fixcm=False,
                          rng=np.random.default_rng(10 + seed))
        k = []
        dyn.attach(lambda a=atoms, k=k: k.append(a.get_kinetic_energy()))
        dyn.run(n_steps)
        theirs.append(np.array(k[skip:]))
        out = thermostats.run(model, m, r, v, 5.0, n_steps,
                              Andersen(120.0, rate),
                              np.random.default_rng(20 + seed), keep=())
        ours.append(out["kinetic"][skip:])
    for k in (np.concatenate(ours), np.concatenate(theirs)):
        t = 2 * k / (3 * len(m) * units.KB)
        assert abs(t.mean() - 120.0) < 3.0
        assert math.isclose(t.std() / t.mean(), math.sqrt(2 / (3 * len(m))),
                            rel_tol=0.1)
