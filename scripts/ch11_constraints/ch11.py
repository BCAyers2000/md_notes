"""Shared set-up of the Chapter 11 scripts: the water box and its start.

Every water run of Chapter 11 starts from the state cached by
prepare_water.py: 64 TIP3P molecules (mdlab.water) at the density of
liquid water at 298.15 K and 0.1 MPa, 0.997 g/cm³ (NIST Chemistry
WebBook), prepared with the molecules held rigid.
"""

from pathlib import Path

import numpy as np

from mdlab import constraints, md, units, viz, water

DATA = Path(viz.THEORY, "data", "ch11_constraints")
START = DATA / "water_start.npz"
N_SIDE, DENSITY, SEED = 4, 0.997, 11
N_MOLECULES = N_SIDE**3
#: Kinetic energy per freedom, ½ k_B (300 K), in eV (Chapter 12).
PER_FREEDOM = 0.5 * units.KB * 300.0


def draw(masses, positions, rng, bonds=None, lengths=None):
    """Random velocities for the box.

    No drift, no part along a held bond, and PER_FREEDOM of kinetic
    energy for each freedom left.
    """
    m = np.asarray(masses, dtype=float)
    n_free = 3 * len(m) - 3 - (0 if bonds is None else len(bonds))
    v = md.starting_velocities(m, 1.0, rng)
    if bonds is not None:
        v, _ = constraints.rattle_velocities(
            positions, v, m, bonds, lengths, tolerance=1e-13
        )
    kinetic = 0.5 * units.MV2_TO_EV * float(m @ (v * v).sum(1))
    return v * np.sqrt(n_free * PER_FREEDOM / kinetic)


def load():
    """The prepared box.

    Positions, cell, masses, symbols, and the rigid and flexible starting
    positions and velocities.
    """
    if not START.exists():
        raise SystemExit("run scripts/ch11_constraints/prepare_water.py first")
    data = np.load(START)
    symbols = ["O", "H", "H"] * N_MOLECULES
    return {key: data[key] for key in data.files} | {"symbols": symbols}


def model(cell, flexible=False):
    return water.WaterModel(cell, N_MOLECULES, flexible=flexible)
