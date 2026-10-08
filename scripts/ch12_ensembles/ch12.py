"""Shared set-up of the Chapter 12 scripts: Lennard-Jones argon.

Argon is modelled as in Chapter 10: ε = 0.01034 eV, σ = 3.4 Å, the energy
switched off between 2σ and 2.5σ, mass 39.948 amu. The liquid is built as
a face-centred cubic lattice at the number density 0.8σ⁻³ and melted; the
crystal of the lattice-start run uses ASE's reference value for argon,
a = 5.26 Å.
"""

from pathlib import Path

import numpy as np

from mdlab import cell, md, potentials, statmech, viz

DATA = Path(viz.THEORY, "data", "ch12_ensembles")
RUNS = DATA / "runs"
EPS, SIG, MASS = 0.01034, 3.4, 39.948
R_CUT, R_SWITCH = 2.5 * SIG, 2.0 * SIG
DENSITY = 0.8 / SIG**3  # atoms per Å³
A_LIQUID = (4 / DENSITY) ** (1 / 3)  # the fcc spacing at that density
A_ASE = 5.26  # ASE's reference value for solid argon, Å
FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def fcc(n, a):
    """N × n × n cubic fcc cells of side a: positions and cell."""
    grid = np.array([[i, j, k] for i in range(n) for j in range(n)
                     for k in range(n)], float)
    s = (grid[:, None, :] + FCC[None]).reshape(-1, 3) / n
    h = n * a * np.eye(3)
    return cell.to_cartesian(s, h), h


def model(h, skin=1.0):
    pair = potentials.with_cutoff(
        lambda r: potentials.lennard_jones(r, EPS, SIG), R_CUT, "switch",
        R_SWITCH)
    return md.PairModel(pair, h, R_CUT, skin)


def liquid(n, temperature, rng, skin=1.0, rounds=4, steps=200, dt=10.0):
    """A liquid of 4n³ atoms prepared near ``temperature``.

    Melted at twice the temperature, then followed for ``rounds`` rounds
    each started from new Maxwell-Boltzmann velocities at the temperature.
    """
    r, h = fcc(n, A_LIQUID)
    m = np.full(len(r), MASS)
    w = model(h, skin)
    v = statmech.thermal_velocities(m, 2 * temperature, rng)
    for _ in range(rounds + 1):
        out = md.run(w, m, r, v, h, dt, steps, every=steps)
        r = out["positions"][-1]
        v = statmech.thermal_velocities(m, temperature, rng)
    return r, v, h, m, w
