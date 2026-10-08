"""Smoke test of the render pipeline: a slab of AB graphite.

    python renders/ch00_smoke/graphite.py

Builds the structure with ASE, writes the colours table, renders it with
Blender headless, and composites the result on white.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.build import make_supercell
from ase.lattice.hexagonal import Graphite

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import atoms_csv  # noqa: E402
import composite  # noqa: E402
from blender_exec import blender_executable  # noqa: E402

DRIVER = HERE.parent / "common" / "render_atoms.py"
OUT = HERE / "out"


def main():
    """Build the graphite slab and render its stacking."""
    blender = blender_executable()
    # Lattice parameters rounded to two decimals; the render is a picture of
    # the stacking, and no number is read from it.
    cell = Graphite("C", latticeconstant={"a": 2.46, "c": 6.70})
    rectangular = make_supercell(
        cell, np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]]))
    slab = rectangular.repeat((6, 4, 2))

    table = atoms_csv.write(slab, OUT / "graphite_atoms.csv")
    raw = OUT / "graphite_raw.png"
    subprocess.run([blender, "-b", "-P", str(DRIVER), "--",
                    "--colours", str(table), "--out", str(raw),
                    "--elev", "18", "--azim", "-62",
                    "--samples", "128", "--res", "1600"], check=True)
    print(composite.on_white(raw, OUT / "graphite.png"), len(slab), "atoms")


if __name__ == "__main__":
    main()
