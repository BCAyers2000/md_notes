"""Graphite, LiC₆ and dilute lithium relaxed with MACE-MP-0 and D3.

Cell and positions together (ASE's FrechetCellFilter with FIRE) to forces
below 1 meV/Å and stresses below 0.02 GPa, in float64. The structures are
written to data/ch17_practice/structures/ as extended XYZ.

Prints the lattice constants against the measured ones, the spacing of
the sheets, and each structure's energy per atom.
"""

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import FIRE
from ch17 import (
    A_GRAPHITE,
    C_GRAPHITE,
    C_LIC6,
    DATA,
    GPA,
    dilute,
    graphite,
    lic6,
    mace,
)

OUT = DATA / "structures"
OUT.mkdir(parents=True, exist_ok=True)
calc = mace(dtype="float64")


def relax(atoms, cell=True):
    atoms.calc = calc
    target = FrechetCellFilter(atoms) if cell else atoms
    FIRE(target, logfile=None).run(fmax=1e-3, steps=5000)
    stress = np.abs(atoms.get_stress()).max() * GPA if cell else 0.0
    print(f"  largest force {np.abs(atoms.get_forces()).max() * 1000:.2f} "
          f"meV/Å, largest stress {stress:.4f} GPa")
    return atoms


g = relax(graphite())
a, c = g.cell.lengths()[0], g.cell.lengths()[2]
print(f"graphite: a = {a:.4f} Å ({100 * (a / A_GRAPHITE - 1):+.2f}%), c = "
      f"{c:.4f} Å ({100 * (c / C_GRAPHITE - 1):+.2f}%), sheets {c / 2:.4f} Å "
      f"apart; E/atom {g.get_potential_energy() / len(g):.4f} eV")
write(OUT / "graphite.extxyz", g)

l6 = relax(lic6())
a6, c6 = l6.cell.lengths()[0], l6.cell.lengths()[2]
print(f"LiC6: a = {a6:.4f} Å, a/√3 = {a6 / np.sqrt(3):.4f} Å "
      f"({100 * (a6 / np.sqrt(3) / a - 1):+.2f}% on graphite's), c = "
      f"{c6:.4f} Å ({100 * (c6 / C_LIC6 - 1):+.2f}% on 3.70); "
      f"E/atom {l6.get_potential_energy() / len(l6):.4f} eV")
write(OUT / "lic6.extxyz", l6)

d = relax(dilute(a, c))
ad, cd = d.cell.lengths()[0] / 4, d.cell.lengths()[2]
li = d.positions[-1]
print(f"dilute: a = {ad:.4f} Å, c = {cd:.4f} Å; lithium at fractional "
      f"{d.get_scaled_positions()[-1].round(4)}")
write(OUT / "dilute.extxyz", d)
