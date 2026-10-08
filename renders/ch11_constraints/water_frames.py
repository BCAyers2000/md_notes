"""Rendered frames of the healthy rigid-water run of Chapter 11.

    python renders/ch11_constraints/water_frames.py [--movie]

Reads data/ch11_constraints/runs/rigid_healthy.npz (scripts/ch11_
constraints/runs.py), moves each molecule whole into the cell by its
oxygen, and renders frames with Blender headless: the O–H bonds are drawn,
since holding them is the point of the chapter. Without --movie, three
frames (0, 5 and 10 ps) at full quality, assembled into the book figure
book/figures/ch11_constraints/frames.pdf; with --movie, every fourth saved
frame (200 fs apart) at low quality, written as an animated GIF to
data/ch11_constraints/frames.gif for Notebook 11. The oxygen atoms of
three molecules are drawn in the accent colour so that their motion can
be followed; all frames share one camera and one crop, so distances
compare across them.
"""

import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

from mdlab import cell, viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from blender_exec import blender_executable  # noqa: E402

DRIVER = HERE.parent / "common" / "render_atoms.py"
OUT = HERE / "out"
DATA = viz.THEORY / "data" / "ch11_constraints"
SYMBOLS = ("O", "H", "H")
TRACKED = (5, 27, 50)  # molecules whose oxygen is drawn in the accent


def whole(frame, h):
    """Each molecule moved by a lattice vector so its oxygen is inside."""
    molecules = frame.reshape(-1, 3, 3)
    oxygen = molecules[:, 0]
    shift = cell.wrap(oxygen, h) - oxygen
    return (molecules + shift[:, None, :]).reshape(-1, 3)


def tables(frame, stem):
    """Write the atoms and bonds tables the render driver reads."""
    atoms, bonds = OUT / f"{stem}_atoms.csv", OUT / f"{stem}_bonds.csv"
    with open(atoms, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["x", "y", "z", "r", "g", "b", "rad"])
        for k, position in enumerate(frame):
            spec = viz.ELEMENT[SYMBOLS[k % 3]]
            colour = spec["colour"]
            if k % 3 == 0 and k // 3 in TRACKED:
                colour = viz.ACCENT
            out.writerow([*(f"{c:.4f}" for c in position),
                          *(f"{c:.4f}" for c in to_rgb(colour)),
                          spec["radius"]])
    with open(bonds, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["x1", "y1", "z1", "x2", "y2", "z2", "r1", "g1", "b1",
                      "r2", "g2", "b2", "rad"])
        o_colour = to_rgb(viz.ELEMENT["O"]["colour"])
        h_colour = to_rgb(viz.ELEMENT["H"]["colour"])
        for first in range(0, len(frame), 3):
            for hydrogen in (first + 1, first + 2):
                out.writerow([*(f"{c:.4f}" for c in frame[first]),
                              *(f"{c:.4f}" for c in frame[hydrogen]),
                              *o_colour, *h_colour, 0.16])
    return atoms, bonds


def render(frame, stem, samples, res):
    """A raw transparent render of one frame; the camera is fixed."""
    atoms, bonds = tables(frame, stem)
    raw = OUT / f"{stem}_raw.png"
    subprocess.run([blender_executable(), "-b", "-P", str(DRIVER), "--",
                    "--colours", str(atoms), "--bonds", str(bonds),
                    "--out", str(raw), "--elev", "22", "--azim", "35",
                    "--ortho-scale", "24", "--samples", str(samples),
                    "--res", str(res)], check=True, capture_output=True)
    return raw


def on_white(raws):
    """The renders flattened on white and cropped to one common box."""
    images = [Image.open(raw).convert("RGBA") for raw in raws]
    solid = np.any([np.asarray(im)[..., 3] > 128 for im in images], axis=0)
    rows, cols = np.flatnonzero(solid.any(1)), np.flatnonzero(solid.any(0))
    pad = int(0.02 * (cols[-1] - cols[0]))
    box = (cols[0] - pad, rows[0] - pad, cols[-1] + pad, rows[-1] + pad)
    out = []
    for im in images:
        white = Image.new("RGBA", im.size, (255, 255, 255, 255))
        out.append(Image.alpha_composite(white, im).convert("RGB").crop(box))
    return out


def main():
    """Render the book frames, or the notebook's animation."""
    OUT.mkdir(parents=True, exist_ok=True)
    run = np.load(DATA / "runs" / "rigid_healthy.npz")
    start = np.load(DATA / "water_start.npz")
    h = start["cell"]
    frames, times = run["frames"], run["frame_times"]
    if "--movie" in sys.argv:
        raws = [render(whole(frames[k], h), f"movie_{k:03d}", 24, 600)
                for k in range(0, len(frames), 4)]
        pictures = on_white(raws)
        target = DATA / "frames.gif"
        pictures[0].save(target, save_all=True, append_images=pictures[1:],
                         duration=150, loop=0)
        print("wrote", target, len(pictures), "frames")
        return
    picks = [int(np.argmin(np.abs(times - t))) for t in (0, 5000, 10000)]
    pictures = on_white([render(whole(frames[k], h), f"frame_{k:03d}", 128,
                                1400) for k in picks])
    viz.use_style(notebook=False)
    fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.1),
                             gridspec_kw=dict(wspace=0.05))
    for ax, picture, k, letter in zip(axes, pictures, picks, "abc",
                                      strict=True):
        ax.imshow(picture)
        ax.set_axis_off()
        ax.set_title(f"({letter}) {times[k] / 1000:g} ps", loc="left")
    print("wrote", viz.save(fig, viz.figure_path("ch11_constraints",
                                                 "frames.pdf")))


if __name__ == "__main__":
    main()
