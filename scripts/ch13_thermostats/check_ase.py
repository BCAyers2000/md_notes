"""mdlab's thermostats against ASE's.

Run from scripts/ch13_thermostats/: python check_ase.py [method ...], with
the methods berendsen, andersen, langevin, chain and csvr (all if none).

A slightly disordered crystal of 32 argon atoms (two cells of ASE's
lattice constant, a = 5.26 Å, a model with the energy switched off between
4.5 and 5.0 Å so that the small cell holds it). Berendsen, the Nosé-Hoover
chain, BAOAB and CSVR are followed for 50 steps of 5 fs from the same
start by mdlab and by ASE's NVTBerendsen, NoseHooverChainNVT,
LangevinBAOAB and Bussi, the stochastic two with the same seed, and the
largest differences in position and velocity printed. ASE's Andersen
redraws single components where mdlab redraws whole velocities, so the two
draw different random numbers and are compared by distribution: four runs
of 100 ps each at 10 collisions per ps, the mean kinetic temperature over
all 96 freedoms and its relative spread against the canonical.
"""

import math
import sys

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.md.andersen import Andersen as AseAndersen
from ase.md.bussi import Bussi
from ase.md.langevinbaoab import LangevinBAOAB
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nvtberendsen import NVTBerendsen

from mdlab import md, potentials, thermostats, units
from mdlab.thermostats import (
    CSVR,
    Andersen,
    Berendsen,
    Langevin,
    NoseHooverChain,
)

ASE_FS = 1 / math.sqrt(units.FORCE_TO_ACCEL)  # one ASE time unit, in fs


def system():
    a = 5.26
    s = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(2) for j in range(2)
                     for k in range(2)])
    rng = np.random.default_rng(1)
    r = (grid[:, None, :] + s[None]).reshape(-1, 3) * a
    r = r + 0.05 * rng.standard_normal(r.shape)
    h = 2 * a * np.eye(3)
    pair = potentials.with_cutoff(
        lambda d: potentials.lennard_jones(d, 0.01034, 3.4), 5.0, "switch",
        4.5)
    v = 0.003 * rng.standard_normal(r.shape)
    return md.PairModel(pair, h, 5.0, 0.2), np.full(32, 39.948), r,\
        v - v.mean(0), h


class Wrapped(Calculator):
    """An ASE calculator that returns an mdlab model's energy and forces."""

    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """Evaluate the model at the atoms' positions."""
        super().calculate(atoms, properties, system_changes)
        u, f = self.model(self.atoms.positions)
        self.results = {"energy": u, "free_energy": u, "forces": f}


def atoms_of(model, m, r, v, h):
    atoms = Atoms("Ar32", positions=r, cell=h, pbc=True, masses=m)
    atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
    atoms.calc = Wrapped(model)
    return atoms


def compare(label, ours_thermostat, make_ase, seed=None):
    model, m, r, v, h = system()
    atoms = atoms_of(model, m, r, v, h)
    make_ase(atoms).run(50)
    ours = thermostats.run(model, m, r, v, 5.0, 50, ours_thermostat,
                           np.random.default_rng(seed))
    dr = np.abs(atoms.positions - ours["positions"][-1]).max()
    v_ase = atoms.get_velocities() * math.sqrt(units.FORCE_TO_ACCEL)
    dv = np.abs(v_ase / ours["velocities"][-1] - 1).max()
    print(f"{label}: largest difference in position {dr:.1e} Å, largest "
          f"relative difference in velocity {dv:.1e}")


def compare_andersen(rate=0.01, n_steps=20000, skip=1000, runs=4):
    model, m, r, v, h = system()
    temps = {"ASE": [], "mdlab": []}
    for seed in range(runs):
        atoms = atoms_of(model, m, r, v, h)
        dyn = AseAndersen(atoms, 5.0 / ASE_FS, temperature_K=120.0,
                          andersen_prob=-math.expm1(-rate * 5.0), fixcm=False,
                          rng=np.random.default_rng(10 + seed))
        k = []
        dyn.attach(lambda a=atoms, k=k: k.append(a.get_kinetic_energy()))
        dyn.run(n_steps)
        temps["ASE"].append(np.array(k[skip:]))
        ours = thermostats.run(model, m, r, v, 5.0, n_steps,
                               Andersen(120.0, rate),
                               np.random.default_rng(20 + seed), keep=())
        temps["mdlab"].append(ours["kinetic"][skip:])
    canonical = math.sqrt(2 / 96)
    for name, ks in temps.items():
        t = [2 * k / (96 * units.KB) for k in ks]
        means = [x.mean() for x in t]
        spread = np.mean([x.std() / x.mean() for x in t]) / canonical
        print(f"Andersen, {name}: mean T {np.mean(means):.2f} ± "
              f"{np.std(means, ddof=1) / math.sqrt(runs):.2f} K over {runs} "
              f"runs, target 120 K; spread {spread:.3f} of the canonical")


METHODS = {
    "berendsen": lambda: compare(
        "Berendsen", Berendsen(120.0, 200.0, 96),
        lambda a: NVTBerendsen(a, 5.0 / ASE_FS, temperature_K=120.0,
                               taut=200.0 / ASE_FS, fixcm=False)),
    "andersen": compare_andersen,
    "langevin": lambda: compare(
        "Langevin (BAOAB)", Langevin(120.0, 1 / 200),
        lambda a: LangevinBAOAB(a, 5.0 / ASE_FS, temperature_K=120.0,
                                T_tau=200.0 / ASE_FS,
                                rng=np.random.default_rng(4)), seed=4),
    "chain": lambda: compare(
        "Nose-Hoover chain", NoseHooverChain(120.0, 100.0, 96),
        lambda a: NoseHooverChainNVT(a, 5.0 / ASE_FS, temperature_K=120.0,
                                     tdamp=100.0 / ASE_FS, tchain=3)),
    "csvr": lambda: compare(
        "CSVR (Bussi)", CSVR(120.0, 100.0, 96),
        lambda a: Bussi(a, 5.0 / ASE_FS, temperature_K=120.0,
                        taut=100.0 / ASE_FS, rng=np.random.default_rng(6)),
        seed=6),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or METHODS:
        METHODS[name]()
