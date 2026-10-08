"""mdlab: the molecular dynamics and machine-learning code of the book.

The package grows with the book, one module per idea, in reading order:

- ``units``: constants and the conversion factors of eV, Å, fs and amu.
- ``kinematics``: describing motion (Chapter 1).
- ``dynamics``: forces, Newton's laws and a reference solver (Chapter 2).
- ``energy``: work, kinetic energy and model surfaces (Chapter 3).
- ``oscillators``: springs, wells, damping, resonance and normal modes
  (Chapter 4).
- ``rotation``: angular momentum, the inertia tensor and the removal of
  rigid motion (Chapter 5).
- ``lagrangian``: the action and constrained motion (Chapter 6).
- ``hamiltonian``: phase-space flow, areas and Jacobians (Chapter 7).
- ``potentials``: pair, many-body and bonded energies with their forces
  (Chapter 8).
- ``integrators``: Euler, velocity Verlet and its forms, RK4, and a driver
  (Chapter 9).
- ``cell``, ``neighbours``, ``ewald``, ``io`` and ``md``: periodic cells,
  neighbour lists, the Ewald sum, trajectory files and the loop that puts
  them together; ``kernels``: one force loop in Python, NumPy, Numba and
  Fortran (Chapter 10).
- ``constraints``, ``respa``, ``water`` and ``diagnostics``: bonds held by
  SHAKE and RATTLE, multiple time steps, a model of water, and the first
  checks of a trajectory's health (Chapter 11).
- ``statmech``: degrees of freedom, temperature, Maxwell-Boltzmann
  velocities and the ensembles' densities (Chapter 12).
- ``thermostats``: from rescaling to CSVR, with one loop (Chapter 13).
- ``virial`` and ``barostats``: pressure, its tensor, and the barostats
  (Chapter 14).
- ``analysis``: statistics of a recorded series and the convergence
  criterion (Chapter 15); structure, transport, spectra and free energy
  from a trajectory (Chapter 16).
- ``bonds`` and ``forcefield``: bonds, molecules and reactions found in a
  trajectory, and the bonded terms of a force field (Chapter 16).
- ``learn``: least squares, ridge, Gaussian processes, gradient descent,
  momentum and Adam, and training and validation splits (Chapter 18).
- ``viz``: the house style for figures and notebooks.
- ``exercise``: checks for the notebook exercises.
"""

from . import (
    cell,
    constraints,
    diagnostics,
    dynamics,
    energy,
    ewald,
    exercise,
    hamiltonian,
    integrators,
    io,
    kernels,
    kinematics,
    lagrangian,
    learn,
    md,
    neighbours,
    oscillators,
    potentials,
    respa,
    rotation,
    units,
    viz,
    water,
)

__all__ = [
    "cell",
    "constraints",
    "diagnostics",
    "dynamics",
    "energy",
    "ewald",
    "exercise",
    "hamiltonian",
    "integrators",
    "io",
    "kernels",
    "kinematics",
    "lagrangian",
    "learn",
    "md",
    "neighbours",
    "oscillators",
    "potentials",
    "respa",
    "rotation",
    "units",
    "viz",
    "water",
]
__version__ = "0.0.3"
