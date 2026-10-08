"""Lithium's energy over the gallery of dilute graphite, with MACE-MP-0.

The relaxed dilute cell (relax.py) with its carbons held; lithium is set at
each point of a 12×12 grid of fractional positions over one primitive cell
of the sheet (a = 2.46 Å) and only its height relaxed (FIRE, force along z
below 0.01 eV/Å), in float64 on the CPU with D3. The map shows where the
sites are and which way between them is lowest.

Writes data/ch17_practice/runs/li_map.npz: "fraction" (12, the grid along
each primitive vector), "energy" (12, 12) in eV less its least, "height"
(12, 12) in Å. Prints the minima, their energies and the lowest saddle.
"""

import numpy as np
from ase.constraints import FixAtoms, FixCartesian
from ase.optimize import FIRE
from ch17 import RUNS, mace, threads
from runs_mace import load

N = 12

if __name__ == "__main__":
    threads(4)
    start = load("dilute")
    calc = mace(dtype="float64")
    cell = np.array(start.get_cell())
    prim = cell[:2] / 4  # the sheet's primitive vectors, as rows
    li0 = start.positions[-1].copy()
    origin = li0[:2] - np.array([1 / 3, 2 / 3]) @ prim[:, :2]
    fraction = np.arange(N) / N
    energy = np.zeros((N, N))
    height = np.zeros((N, N))
    for i, u in enumerate(fraction):
        for j, w in enumerate(fraction):
            atoms = start.copy()
            atoms.positions[-1, :2] = origin + np.array([u, w]) @ prim[:, :2]
            atoms.set_constraint([
                FixAtoms(indices=range(len(atoms) - 1)),
                FixCartesian([len(atoms) - 1], mask=(True, True, False))])
            atoms.calc = calc
            FIRE(atoms, logfile=None).run(fmax=0.01, steps=300)
            energy[i, j] = atoms.get_potential_energy()
            height[i, j] = atoms.positions[-1, 2]
    energy -= energy.min()
    np.savez(RUNS / "li_map.npz", fraction=fraction, energy=energy,
             height=height)
    for k in np.argsort(energy.ravel())[:6]:
        i, j = divmod(k, N)
        print(f"({fraction[i]:.3f}, {fraction[j]:.3f}): "
              f"{1000 * energy[i, j]:.1f} meV")
