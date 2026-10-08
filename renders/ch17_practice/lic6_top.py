"""LiC₆ seen from above, from MACE-MP-0's run at 300 K (Chapter 17).

    python renders/ch17_practice/lic6_top.py

The last frame of runs_mace.py's npt_aniso (63 atoms, 3×3×1 cells of the
√3×√3 R30° unit), repeated twice along each lattice vector of the plane:
one carbon sheet with the lithium atoms of the gallery above it, which sit
over every third hexagon. Rendered with Blender headless, looking down the
c axis, and saved as book/figures/ch17_practice/top.pdf.
"""

import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import to_rgb
from PIL import Image

from mdlab import viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from blender_exec import blender_executable  # noqa: E402

DRIVER = HERE.parent / "common" / "render_atoms.py"
OUT = HERE / "out"
FRAME = Path(viz.THEORY, "data", "ch17_practice", "runs",
             "npt_aniso_last.extxyz")


def main():
    """Render the frame from above and save the book figure."""
    atoms = read(FRAME).repeat((2, 2, 1))
    atoms.wrap()
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / "lic6_top_atoms.csv"
    with open(table, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["x", "y", "z", "r", "g", "b", "rad"])
        for atom in atoms:
            spec = viz.ELEMENT[atom.symbol]
            out.writerow([f"{c:.4f}" for c in atom.position]
                         + [f"{c:.4f}" for c in to_rgb(spec["colour"])]
                         + [0.6 * spec["radius"]])
    raw = OUT / "lic6_top_raw.png"
    subprocess.run([blender_executable(), "-b", "-P", str(DRIVER), "--",
                    "--colours", str(table), "--out", str(raw),
                    "--elev", "90", "--azim", "0", "--samples", "128",
                    "--res", "1400"], check=True, capture_output=True)
    image = Image.open(raw).convert("RGBA")
    solid = np.asarray(image)[..., 3] > 128
    rows, cols = np.flatnonzero(solid.any(1)), np.flatnonzero(solid.any(0))
    box = (cols[0], rows[0], cols[-1], rows[-1])
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    picture = Image.alpha_composite(white, image).convert("RGB").crop(box)
    viz.use_style(notebook=False)
    fig, ax = plt.subplots(figsize=(viz.HALF, viz.HALF * picture.height
                                    / picture.width))
    ax.imshow(picture)
    ax.set_axis_off()
    print("wrote", viz.save(fig, viz.figure_path("ch17_practice", "top.pdf")))


if __name__ == "__main__":
    main()
