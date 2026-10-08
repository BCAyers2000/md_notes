"""Graphite and fully lithiated LiC₆, rendered side by side (Chapter 14).

    python renders/ch14_pressure/graphite.py

Carbon sheets of the honeycomb lattice with the in-plane spacing a =
2.46 Å, four sheets each: graphite stacked A, B, A, B with the sheets
3.34 Å apart; LiC₆ stacked A, A with a lithium atom over every third
hexagon between each pair of sheets, 3.70 Å apart (the experimental
values quoted by Hazrati, de Wijs and Brocks 2014; the small in-plane
change on lithiation is left out). Both are rendered with Blender
headless from one camera and one width, so the spacings compare, and
assembled into book/figures/ch14_pressure/graphite.pdf.
"""

import csv
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

from mdlab import viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from blender_exec import blender_executable  # noqa: E402

DRIVER = HERE.parent / "common" / "render_atoms.py"
OUT = HERE / "out"
A = 2.46
GRAPHITE_SPACING, LIC6_SPACING = 3.34, 3.70
CELLS, SHEETS = 6, 4
A1 = A * np.array([1.0, 0.0])
A2 = A * np.array([0.5, math.sqrt(3) / 2])


def sheet(shift):
    """The carbon atoms of one honeycomb sheet.

    The hexagon centres sit at the lattice points plus ``shift``, in
    fractional coordinates.
    """
    sites = []
    for i in range(CELLS):
        for j in range(CELLS):
            for frac in ((1 / 3, 1 / 3), (2 / 3, 2 / 3)):
                f = np.array([i, j]) + np.array(frac) + np.array(shift)
                sites.append(f[0] * A1 + f[1] * A2)
    return np.array(sites)


def graphite():
    """Four sheets stacked A, B, A, B."""
    atoms = []
    for k in range(SHEETS):
        shift = (0.0, 0.0) if k % 2 == 0 else (1 / 3, 1 / 3)
        xy = sheet(shift)
        atoms += [("C", *p, k * GRAPHITE_SPACING) for p in xy]
    return atoms


def lic6():
    """Four sheets stacked A, A, lithium between each pair."""
    atoms = []
    for k in range(SHEETS):
        atoms += [("C", *p, k * LIC6_SPACING) for p in sheet((0.0, 0.0))]
        if k == SHEETS - 1:
            break
        for i in range(CELLS):
            for j in range(CELLS):
                if (i - j) % 3 == 0:
                    p = i * A1 + j * A2
                    atoms.append(("Li", *p, (k + 0.5) * LIC6_SPACING))
    return atoms


def render(atoms, stem):
    """A raw transparent render of a list of (symbol, x, y, z)."""
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / f"{stem}_atoms.csv"
    with open(table, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["x", "y", "z", "r", "g", "b", "rad"])
        for symbol, x, y, z in atoms:
            spec = viz.ELEMENT[symbol]
            out.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                          *(f"{c:.4f}" for c in to_rgb(spec["colour"])),
                          0.6 * spec["radius"]])
    raw = OUT / f"{stem}_raw.png"
    subprocess.run([blender_executable(), "-b", "-P", str(DRIVER), "--",
                    "--colours", str(table), "--out", str(raw),
                    "--elev", "8", "--azim", "75", "--ortho-scale", "24",
                    "--samples", "128", "--res", "1400"],
                   check=True, capture_output=True)
    return raw


def on_white(raws):
    """The renders flattened on white and cropped to one common box."""
    images = [Image.open(raw).convert("RGBA") for raw in raws]
    solid = np.any([np.asarray(im)[..., 3] > 128 for im in images], axis=0)
    rows, cols = np.flatnonzero(solid.any(1)), np.flatnonzero(solid.any(0))
    pad = int(0.02 * (cols[-1] - cols[0]))
    box = (cols[0] - pad, rows[0] - pad, cols[-1] + pad, rows[-1] + pad)
    pictures = []
    for im in images:
        white = Image.new("RGBA", im.size, (255, 255, 255, 255))
        pictures.append(Image.alpha_composite(white, im).convert("RGB")
                        .crop(box))
    return pictures


def main():
    """Render both and assemble the book figure."""
    pictures = on_white([render(graphite(), "graphite"),
                         render(lic6(), "lic6")])
    viz.use_style(notebook=False)
    fig, axes = plt.subplots(1, 2, figsize=(viz.FULL, 2.2),
                             gridspec_kw=dict(wspace=0.05))
    titles = (f"(a) graphite, sheets {GRAPHITE_SPACING:.2f} Å apart",
              f"(b) LiC$_6$, sheets {LIC6_SPACING:.2f} Å apart")
    for ax, picture, title in zip(axes, pictures, titles, strict=True):
        ax.imshow(picture)
        ax.set_axis_off()
        ax.set_title(title, loc="left")
    print("wrote", viz.save(fig, viz.figure_path("ch14_pressure",
                                                 "graphite.pdf")))


if __name__ == "__main__":
    main()
