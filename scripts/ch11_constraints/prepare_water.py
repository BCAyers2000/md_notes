"""Prepare the water box of Chapter 11 and cache its starting state.

64 TIP3P molecules on a 4 x 4 x 4 grid at 0.997 g/cm³, each turned at
random, are held rigid and followed with RATTLE (δt = 2 fs) in ten
rounds of 1 ps; each round starts from freshly drawn velocities with
½ k_B (300 K) of kinetic energy per freedom, which carries away the
energy released as the molecules leave the grid. The flexible model then
starts from the final positions with velocities drawn for all nine
freedoms per molecule, and is prepared the same way in two rounds of
0.5 ps with δt = 0.25 fs.

Writes data/ch11_constraints/water_start.npz (about 4 minutes).
Prints the potential energy after each round.
"""

import numpy as np
from ch11 import DATA, DENSITY, N_MOLECULES, N_SIDE, SEED, START, draw, model

from mdlab import constraints, md, water

rng = np.random.default_rng(SEED)
r, h, m, symbols = water.box(N_SIDE, DENSITY, rng)
bonds, lengths = water.constraint_bonds(N_MOLECULES)
rigid = model(h)
print(f"cell side {h[0, 0]:.4f} Å, {N_MOLECULES} molecules")
print(f"on the grid: U = {rigid(r)[0]:.3f} eV")
for k in range(10):
    v = draw(m, r, rng, bonds, lengths)
    out = constraints.run(rigid, m, r, v, h, 2.0, 500, bonds, lengths,
                          every=50)
    r = out["positions"][-1]
    print(f"rigid round {k + 1}: U = {out['potential'][-1]:.3f} eV, "
          f"K = {out['kinetic'][-1]:.3f} eV")
v_rigid = draw(m, r, rng, bonds, lengths)

flexible = model(h, flexible=True)
r_flex = r.copy()
for k in range(2):
    v = draw(m, r_flex, rng)
    out = md.run(flexible, m, r_flex, v, h, 0.25, 2000, every=200)
    r_flex = out["positions"][-1]
    print(f"flexible round {k + 1}: U = {out['potential'][-1]:.3f} eV, "
          f"K = {out['kinetic'][-1]:.3f} eV")
v_flex = draw(m, r_flex, rng)

DATA.mkdir(parents=True, exist_ok=True)
np.savez(START, positions=r, velocities=v_rigid, flexible_positions=r_flex,
         flexible_velocities=v_flex, cell=h, masses=m)
print("wrote", START)
