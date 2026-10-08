"""Shared set-up of the Chapter 18 scripts.

- The A-B molecule of Chapter 16: a Morse bond, D = 1 eV, a = 2 Å⁻¹,
  r0 = 1.2 Å, both atoms of nitrogen's mass, whose stiffness at the bottom
  of the well is 2Da² = 8 eV/Å² (Section 4.5); runs under Langevin
  dynamics give the distances and forces that a fit learns from.
- Liquid argon's pair energy, the switched Lennard-Jones of Chapters 12
  to 17 (ε = 0.01034 eV, σ = 3.4 Å, switched off between 2σ and 2.5σ),
  as the curve to be learned; Chapter 17's runs of the liquid, a frame
  every 100 fs, as the frames of Section 18.8.
"""

import sys
from pathlib import Path

import numpy as np

from mdlab import learn, potentials, thermostats, viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch12_ensembles"))
import ch12  # noqa: E402

DATA = Path(viz.THEORY, "data", "ch18_learning")
RUNS17 = Path(viz.THEORY, "data", "ch17_practice", "runs")

# ------------------------------------------------------------ the bond ---
DEPTH, WIDTH, R0 = 1.0, 2.0, 1.2  # eV, 1/Å, Å: Chapter 16's A-B bond
MASS = 14.007  # amu
STIFFNESS = 2 * DEPTH * WIDTH**2  # eV/Å², the curvature at r0


def bond(positions):
    """The Morse energy of two atoms and the force on each."""
    d = positions[1] - positions[0]
    r = np.linalg.norm(d)
    phi, slope = potentials.morse(r, DEPTH, WIDTH, R0)
    f = -slope * d / r  # on atom 1, along the bond
    return float(phi), np.array([-f, f])


def bond_run(temperature, picoseconds=20.0, seed=0):
    """The A-B molecule under Langevin dynamics (γ = 0.01 fs⁻¹, 1 fs steps).

    Returns the stretch x = r − r0 in Å and the force along the bond,
    −dφ/dr, in eV/Å, every 10 fs.
    """
    rng = np.random.default_rng(seed)
    r = np.array([[0.0, 0.0, 0.0], [R0, 0.0, 0.0]])
    v = np.zeros((2, 3))
    out = thermostats.run(bond, [MASS, MASS], r, v, 1.0,
                          int(picoseconds * 1000),
                          thermostats.Langevin(temperature, 0.01), rng=rng,
                          every=10, keep=("positions",))
    d = out["positions"][:, 1] - out["positions"][:, 0]
    distance = np.linalg.norm(d, axis=1)
    force = -potentials.morse(distance, DEPTH, WIDTH, R0)[1]
    return distance - R0, force


# ----------------------------------------------------------- the argon ---
ARGON = potentials.with_cutoff(
    lambda r: potentials.lennard_jones(r, ch12.EPS, ch12.SIG), ch12.R_CUT,
    "switch", ch12.R_SWITCH)


def argon_pair(r):
    """Liquid argon's pair energy φ(r) in eV, switched off by 2.5σ."""
    return ARGON(np.asarray(r, dtype=float))[0]


#: The readings of Sections 18.2 to 18.4: the pair energy between 3.3 Å and
#: 2σ = 6.8 Å, where it is the plain Lennard-Jones energy, with noise.
NOISE = 2e-4  # eV
R_LOW, R_HIGH = 3.3, 2 * ch12.SIG
MID, HALF = 0.5 * (R_LOW + R_HIGH), 0.5 * (R_HIGH - R_LOW)


def readings(n, rng):
    """Distances drawn at random and their energies with noise, eV."""
    r = np.sort(rng.uniform(R_LOW, R_HIGH, n))
    return r, argon_pair(r) + rng.normal(0, NOISE, n)


def basis(r, degree):
    """Powers 0 to ``degree`` of the scaled distance (r − 5.05 Å)/1.75 Å."""
    return learn.polynomial_basis((np.asarray(r) - MID) / HALF, degree)
