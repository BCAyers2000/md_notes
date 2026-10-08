"""Shared set-up of the Chapter 16 scripts.

The liquid argon of Chapters 12 to 15 (256 atoms, 0.8σ⁻³, switched
Lennard-Jones, 135 K) and its face-centred cubic crystal; and three new
test systems:

- a model molten salt: 64 ions of charge ±e with the masses of sodium and
  chlorine, every pair repelling as A e^{−r/ρ} with ρ = 0.3 Å and A set
  so that a pair of opposite charges has its least energy at 2.8 Å, plus
  the Coulomb energy by the Ewald sum; illustrative, not fitted;
- carbon with Tersoff's 1989 parameters, as distributed with ASE, through
  ASE's Tersoff calculator;
- a model gas of two kinds of atom that bond in pairs, A-B (Morse,
  illustrative).
"""

import math
import os
import sys
from pathlib import Path

import numpy as np

from mdlab import md, potentials, units, viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch15_convergence"))
import ch15  # noqa: E402

ch12, ch13, ch14 = ch15.ch12, ch15.ch13, ch15.ch14
DATA = Path(viz.THEORY, "data", "ch16_observables")
RUNS = DATA / "runs"
T_LIQUID = 135.0
N_ATOMS = 256
DT = 10.0
KT = units.KB * T_LIQUID
GPA = units.EV_PER_A3_TO_GPA
liquid_start = ch13.liquid_start
argon_model = ch13.argon_model

# ---------------------------------------------------------------- salt ---
K_E = units.COULOMB
SALT_RHO = 0.3  # Å, the range of the repulsion
SALT_RMIN = 2.8  # Å, where a +/− pair has its least energy
#: A e^{−r/ρ} balances k_e/r² at r = 2.8 Å: A = ρ k_e e^{r/ρ}/r².
SALT_A = SALT_RHO * K_E * math.exp(SALT_RMIN / SALT_RHO) / SALT_RMIN**2
SALT_SIDE = 13.4  # Å, 64 ions
SALT_T = 1500.0
SALT_DT = 2.0
SALT_MASS = {1.0: 22.99, -1.0: 35.45}
SALT_CUT, SALT_SWITCH = 5.0, 4.0


def salt_lattice(side=SALT_SIDE, n=4):
    """Ions on a simple cubic grid, charges alternating: r, q, m, h."""
    grid = np.array([[i, j, k] for i in range(n) for j in range(n)
                     for k in range(n)], dtype=float)
    q = np.where(grid.sum(1) % 2 == 0, 1.0, -1.0)
    m = np.array([SALT_MASS[x] for x in q])
    return grid * side / n, q, m, side * np.eye(3)


def salt_model(q, h, accuracy=1e-5):
    """The repulsion through a Verlet list plus the Ewald sum."""
    rep = potentials.with_cutoff(
        lambda r: potentials.buckingham(r, SALT_A, SALT_RHO, 0.0),
        SALT_CUT, "switch", SALT_SWITCH)
    return md.sum_of([md.PairModel(rep, h, SALT_CUT, 1.0),
                      md.EwaldModel(q, h, accuracy=accuracy)])


# -------------------------------------------------------------- carbon ---
def tersoff_carbon():
    """ASE's Tersoff calculator with the carbon parameters of its data file.

    The file (``SiC.tersoff``, Tersoff 1989) ships with ASE's tests. For
    an atom with no third neighbour within the cutoff, ζ = 0 and ASE's
    calculator fails on 0 raised to a negative power in the slope of the
    bond order; that slope multiplies only terms of third neighbours,
    which are then all zero, so it is taken as zero.
    """
    import ase
    from ase.calculators.tersoff import Tersoff

    class Tersoff0(Tersoff):
        def _calc_bij_d(self, zeta, beta, n):
            if zeta == 0.0:
                return 0.0
            return super()._calc_bij_d(zeta, beta, n)

    path = Path(os.path.dirname(ase.__file__), "test", "testdata",
                "tersoff", "SiC.tersoff")
    return Tersoff0.from_lammps(path)


CARBON_DENSITY = 2.0  # g/cm³
CARBON_MASS = 12.011
CARBON_DT = 0.5


def carbon_start(seed=0):
    """216 carbon atoms on a diamond lattice expanded to 2.0 g/cm³.

    The lattice is ASE's reference diamond, scaled so that the cell holds
    the density, and each atom is displaced at random by about 0.05 Å.
    """
    from ase.build import bulk
    from ase.data import reference_states
    a = reference_states[6]["a"]
    atoms = bulk("C", "diamond", a=a, cubic=True).repeat(3)
    grams = len(atoms) * CARBON_MASS * units.AMU * 1e3
    volume = grams / CARBON_DENSITY * 1e24  # Å³
    scale = (volume / atoms.get_volume()) ** (1 / 3)
    rng = np.random.default_rng(seed)
    r = scale * atoms.get_positions() + rng.normal(0, 0.05, (len(atoms), 3))
    return r, scale * np.array(atoms.get_cell()).T


# ------------------------------------------------------------- crystal ---
def crystal_a0():
    """The model's own fcc lattice constant: the least energy per atom."""
    from scipy.optimize import minimize_scalar

    def energy(a):
        r, h = ch12.fcc(4, a)
        return argon_model(h)(r)[0] / len(r)

    return float(minimize_scalar(energy, bounds=(5.1, 5.4), method="bounded",
                                 options={"xatol": 1e-10}).x)


# ----------------------------------------------------------- morse gas ---
#: A gas of two kinds of atom, A and B (types 0 and 1), of nitrogen's
#: mass: A-B pairs bond through a Morse well; A-A and B-B pairs only
#: repel, D e^{−2a(r − r_like)}, so strongly at the distances a second
#: partner would need that each atom holds one partner at most.
MORSE_DEPTH, MORSE_WIDTH, MORSE_R0 = 1.0, 2.0, 1.2  # eV, 1/Å, Å
LIKE_R = 2.6  # Å, where two like atoms repel by D
GAS_MASS = 14.007
GAS_N, GAS_SIDE = 500, 63.0  # 0.002 atoms/Å³, half A and half B
GAS_CUT, GAS_SKIN = 5.0, 3.0
GAS_DT = 1.0


def morse_distance(fraction):
    """The distance beyond r0 at which φ = −fraction × D."""
    return MORSE_R0 - math.log(1 - math.sqrt(1 - fraction)) / MORSE_WIDTH


#: A bond forms where φ falls below −D/2 and breaks where it rises above
#: −D/8.
MORSE_ON, MORSE_OFF = morse_distance(0.5), morse_distance(0.125)


def morse_pair(r):
    return potentials.morse(r, MORSE_DEPTH, MORSE_WIDTH, MORSE_R0)


def like_pair(r):
    a = MORSE_DEPTH * math.exp(2 * MORSE_WIDTH * LIKE_R)
    return potentials.buckingham(r, a, 0.5 / MORSE_WIDTH, 0.0)


def gas_types():
    return np.repeat([0, 1], GAS_N // 2)


def gas_symbols():
    return ["A"] * (GAS_N // 2) + ["B"] * (GAS_N // 2)


def gas_model(h):
    cut = {(0, 1): morse_pair, (0, 0): like_pair, (1, 1): like_pair}
    table = {k: potentials.with_cutoff(f, GAS_CUT, "switch", 4.0)
             for k, f in cut.items()}
    return md.PairModel(table, h, GAS_CUT, GAS_SKIN, gas_types())


def gas_tracker(on=None, off=None):
    """Bonds between A and B, between r_on and r_off; like atoms never."""
    from mdlab import bonds
    far = (0.1, 0.1)
    on = MORSE_ON if on is None else on
    off = MORSE_OFF if off is None else off
    return bonds.BondTracker(gas_symbols(), pairs={
        ("A", "B"): (on, off), ("A", "A"): far, ("B", "B"): far})


def gas_molecules(rng):
    """250 AB molecules at r0, one in each of 250 sites of a 7³ grid.

    Each points in a random direction; A atoms come first, then B.
    """
    spacing = GAS_SIDE / 7
    grid = np.array([[i, j, k] for i in range(7) for j in range(7)
                     for k in range(7)], dtype=float)
    sites = (grid[rng.choice(len(grid), GAS_N // 2, replace=False)] + 0.5) \
        * spacing
    axis = rng.normal(size=(GAS_N // 2, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return np.vstack([sites - 0.5 * MORSE_R0 * axis,
                      sites + 0.5 * MORSE_R0 * axis])


# --------------------------------------------------------------- chain ---
CHAIN_MASS = 14.5  # amu, each bead standing for a CH₂ or CH₃ group
CHAIN_BOND_K, CHAIN_BOND_R = 5.0, 1.53  # eV/Å², Å
CHAIN_ANGLE_K, CHAIN_ANGLE = 2.5, math.radians(112.0)  # eV/rad², rad
CHAIN_TORSION = (0.06, -0.01, 0.12)  # κ₁, κ₂, κ₃ in eV
CHAIN_DT = 1.0


def chain_model():
    from mdlab.forcefield import BondedModel
    return BondedModel([[0, 1], [1, 2], [2, 3]], CHAIN_BOND_K, CHAIN_BOND_R,
                       [[0, 1, 2], [1, 2, 3]], CHAIN_ANGLE_K, CHAIN_ANGLE,
                       [[0, 1, 2, 3]], CHAIN_TORSION)


def chain_trans():
    """The flat zigzag, ψ = 180°, at the equilibrium bonds and angles."""
    half = 0.5 * CHAIN_ANGLE
    dx, dy = CHAIN_BOND_R * math.sin(half), CHAIN_BOND_R * math.cos(half)
    return np.array([[0, 0, 0], [dx, dy, 0], [2 * dx, 0, 0],
                     [3 * dx, dy, 0]], dtype=float)


def chain_dihedrals(r):
    """ψ of each copy, r (..., 4, 3), in rad."""
    b1, b2, b3 = r[..., 1, :] - r[..., 0, :], r[..., 2, :] - r[..., 1, :], \
        r[..., 3, :] - r[..., 2, :]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    x = np.sum(n1 * n2, -1)
    y = np.sum(np.cross(n1, n2) * b2, -1) / np.linalg.norm(b2, axis=-1)
    return np.arctan2(y, x)
