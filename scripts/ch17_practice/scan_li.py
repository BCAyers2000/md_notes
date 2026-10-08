"""Lithium's paths between sites in dilute graphite, with MACE-MP-0.

From the relaxed dilute cell (relax.py), where lithium sits over a hollow
of the sheet below and under an atom of the sheet above, lithium is moved
in eight equal steps along two straight lines: to the nearest site of the
other kind, over an atom of the sheet below and under a hollow of the
sheet above, a/√3 = 1.42 Å away (the "hollow" path; of the six atoms
round lithium's hollow, the three with no atom of the sheet above over
them); and to the next site of its own kind, a = 2.46 Å away along a₁,
across the middle of a C-C bond of the sheet below (the "bridge" path).
At each step lithium's position in the plane is held while its height
and every carbon relax (FIRE, forces below 0.02 eV/Å), in float64 on the
CPU with D3; one carbon of each sheet, the farthest from lithium, is held
in the plane too, so that neither sheet can slide to carry its sites
back under lithium. At each relaxed point the energy without D3 is kept
too.

Writes data/ch17_practice/runs/li_scan.npz ("fraction"; for each path
"<path>_energy" and "<path>_learned" in eV and "<path>_length" in Å).
Prints both profiles and their highest points.
"""

import numpy as np
from ase.constraints import FixCartesian
from ase.optimize import FIRE
from ch17 import RUNS, mace, threads
from runs_mace import load

from mdlab.cell import minimum_image

if __name__ == "__main__":
    threads(4)
    start = load("dilute")
    calc = mace(dtype="float64")
    learned = mace(dtype="float64", dispersion=False)
    h = np.array(start.get_cell()).T
    li = start.positions[-1].copy()
    carbon = start.positions[:-1]
    d = minimum_image(carbon - li, h)
    c = np.linalg.norm(h[:, 2])
    below = (d[:, 2] < 0) & (d[:, 2] > -c / 2)
    flat = np.linalg.norm(d[:, :2], axis=1)
    pins = [int(np.flatnonzero(m)[np.argmax(flat[m])])
            for m in (below, ~below)]
    # of the six atoms round lithium's hollow, three lie under atoms of the
    # sheet above and three under its hollows: the other family's sites
    corners = np.flatnonzero(below & (flat < 1.1 * flat[below].min()))
    upper = d[~below] * [1, 1, 0]
    cover = [np.linalg.norm(minimum_image(upper - d[k] * [1, 1, 0], h),
                            axis=1).min() for k in corners]
    nearest = corners[np.argmax(cover)]
    steps = {"hollow": np.append(d[nearest, :2], 0.0),
             "bridge": h[:, 0] / 4}
    fractions = np.linspace(0, 1, 9)
    out = {"fraction": fractions}
    for name, step in steps.items():
        energy, learned_energy = [], []
        for f in fractions:
            atoms = start.copy()
            atoms.positions[-1] = li + f * step
            atoms.set_constraint(FixCartesian([*pins, len(atoms) - 1],
                                              mask=(True, True, False)))
            atoms.calc = calc
            FIRE(atoms, logfile=None).run(fmax=0.02, steps=1000)
            energy.append(atoms.get_potential_energy())
            atoms.calc = learned
            learned_energy.append(atoms.get_potential_energy())
        energy = np.array(energy) - energy[0]
        learned_energy = np.array(learned_energy) - learned_energy[0]
        out |= {f"{name}_energy": energy, f"{name}_learned": learned_energy,
                f"{name}_length": np.linalg.norm(step)}
        print(f"{name} path, {np.linalg.norm(step):.4f} Å; energy, meV: "
              + ", ".join(f"{1000 * e:.0f}" for e in energy))
        print(f"   highest {1000 * energy.max():.1f} meV with D3, "
              f"{1000 * learned_energy.max():.1f} meV without")
    np.savez(RUNS / "li_scan.npz", **out)
