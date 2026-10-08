"""RATTLE in mdlab against ASE's VelocityVerlet with FixBondLengths.

The prepared box of 64 rigid TIP3P molecules, the same forces (mdlab's
WaterModel, wrapped as an ASE calculator) and the same start, followed
for 50 steps of 1 fs by both codes. ASE's time unit is Å √(amu/eV); the
mdlab femtosecond is √FORCE_TO_ACCEL of it, so both use one femtosecond.

Prints the agreement quoted in Section 11.2.
"""

import math

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixBondLengths
from ase.md.verlet import VelocityVerlet
from ch11 import N_MOLECULES, load, model

from mdlab import constraints, units, water


class Wrapped(Calculator):
    """An mdlab model presented to ASE as a calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, energy_forces):
        super().__init__()
        self.energy_forces = energy_forces

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """Energy and forces of the wrapped model."""
        super().calculate(atoms, properties, system_changes)
        u, f = self.energy_forces(self.atoms.positions)
        self.results = {"energy": u, "forces": f}


start = load()
h, m = start["cell"], start["masses"]
r, v = start["positions"], start["velocities"]
bonds, lengths = water.constraint_bonds(N_MOLECULES)
w = model(h)
dt, steps = 1.0, 50
out = constraints.run(w, m, r, v, h, dt, steps, bonds, lengths,
                      every=steps, tolerance=1e-13)

atoms = Atoms(start["symbols"], positions=r, cell=h.T, pbc=True, masses=m)
fs = math.sqrt(units.FORCE_TO_ACCEL)  # mdlab's fs in ASE's time unit
atoms.set_velocities(v / fs)
atoms.constraints = FixBondLengths(bonds[:, ::-1], tolerance=1e-13,
                                   bondlengths=lengths)
atoms.calc = Wrapped(w)
VelocityVerlet(atoms, timestep=dt * fs).run(steps)
dr = np.abs(atoms.positions - out["positions"][-1]).max()
dv = np.abs(atoms.get_velocities() * fs - out["velocities"][-1]).max()
print(f"after {steps} steps of {dt} fs: positions differ by {dr:.1e} Å, "
      f"velocities by {dv:.1e} Å/fs")
d = atoms.positions[bonds[:, 1]] - atoms.positions[bonds[:, 0]]
error = np.abs(np.linalg.norm(d, axis=1) - lengths).max()
print(f"largest bond error at the end: ASE {error:.1e} Å, mdlab "
      f"{out['bond_error'][-1]:.1e} Å (the start, prepared with a "
      f"tolerance of 1e-10, {out['bond_error'][0]:.1e} Å)")
