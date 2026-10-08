"""Write the colours table read by render_atoms.py from an ase.Atoms."""

import csv
from pathlib import Path

from matplotlib.colors import to_rgb

from mdlab.viz import ELEMENT


def write(atoms, path: str | Path) -> Path:
    """One row per atom: x, y, z in Å, sRGB colour, display radius in Å."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["x", "y", "z", "r", "g", "b", "rad"])
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(),
                                     atoms.positions):
            spec = ELEMENT[symbol]
            out.writerow([f"{x:.5f}", f"{y:.5f}", f"{z:.5f}",
                          *(f"{c:.4f}" for c in to_rgb(spec["colour"])),
                          spec["radius"]])
    return path
