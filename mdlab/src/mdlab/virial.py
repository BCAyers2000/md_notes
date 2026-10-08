"""Pressure from a trajectory: Chapter 14.

- ``harmonic_walls``: soft walls round a box, with the push of the atoms
  on each pair of walls, the pressure as force per unit area (Section
  14.1);
- ``kinetic_tensor``: Σ m v_α v_β, twice the kinetic energy tensor;
- ``pressure_tensor`` and ``pressure``: P_αβ = (Σ m v_α v_β + W_αβ)/V and
  its mean diagonal, from the virial tensor W that ``md.PairModel``
  stores (Sections 14.3 and 14.4);
- ``wrapped_virial``: Σ rᵢ·Fᵢ from the positions as they stand, which in
  a periodic cell depends on where the origin is (Section 14.3);
- ``ideal_gas_volume_density``: the exact density of the volume of N
  atoms that do not interact, held at the pressure P and temperature T,
  V^N e^{−PV/k_BT} normalised (Section 14.5).

Units
-----
Positions in Å, velocities in Å/fs, masses in amu, energies in eV,
pressures in eV/Å³ (``units.EV_PER_A3_TO_GPA`` converts to GPa).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from mdlab.units import KB, MV2_TO_EV


def harmonic_walls(
    positions: ArrayLike, lengths: ArrayLike, stiffness: float
) -> tuple[float, NDArray, NDArray]:
    """Walls at 0 and L_α along each axis, each a spring of the stiffness k.

    An atom beyond a wall by the distance δ has the energy ½kδ² and is
    pushed back by kδ. Returns the energy, the forces on the atoms, and,
    for each axis, the total force the atoms exert on its two walls,
    outwards.

    >>> u, f, push = harmonic_walls([[2.1, 0.5, -0.2]], [2.0, 2.0, 2.0], 10)
    >>> print(round(u, 6), f.round(6) + 0.0, push.round(6))
    0.25 [[-1.  0.  2.]] [1. 0. 2.]
    """
    r = np.asarray(positions, dtype=float)
    length = np.asarray(lengths, dtype=float)
    below = np.minimum(r, 0.0)  # negative depth past the near wall
    above = np.maximum(r - length, 0.0)  # depth past the far wall
    energy = 0.5 * stiffness * float(np.sum(below**2) + np.sum(above**2))
    forces = -stiffness * (below + above)
    push = stiffness * (np.sum(above, axis=0) - np.sum(below, axis=0))
    return energy, forces, push


def kinetic_tensor(
    masses: ArrayLike, velocities: ArrayLike, mv2_to_energy: float = MV2_TO_EV
) -> NDArray:
    """Σᵢ mᵢ v_iα v_iβ, in eV; its trace is twice the kinetic energy.

    >>> kinetic_tensor([2.0], [[1.0, 0.0, 3.0]], 1.0)
    array([[ 2.,  0.,  6.],
           [ 0.,  0.,  0.],
           [ 6.,  0., 18.]])
    """
    m = np.asarray(masses, dtype=float)
    v = np.asarray(velocities, dtype=float)
    return mv2_to_energy * np.einsum("i,ia,ib->ab", m, v, v)


def pressure_tensor(
    masses: ArrayLike,
    velocities: ArrayLike,
    virial: ArrayLike,
    volume: float,
    mv2_to_energy: float = MV2_TO_EV,
) -> NDArray:
    """P_αβ = (Σ m v_α v_β + W_αβ)/V, in eV/Å³."""
    kin = kinetic_tensor(masses, velocities, mv2_to_energy)
    return (kin + np.asarray(virial, dtype=float)) / volume


def pressure(
    masses: ArrayLike,
    velocities: ArrayLike,
    virial: ArrayLike,
    volume: float,
    mv2_to_energy: float = MV2_TO_EV,
) -> float:
    """P = (2K + W)/3V, the mean of the diagonal of the pressure tensor."""
    tensor = pressure_tensor(masses, velocities, virial, volume, mv2_to_energy)
    return float(np.trace(tensor) / 3)


def wrapped_virial(positions: ArrayLike, forces: ArrayLike) -> float:
    """Σᵢ rᵢ·Fᵢ from the positions as given, wrapped or not."""
    return float(np.sum(np.asarray(positions) * np.asarray(forces)))


def ideal_gas_volume_density(
    volume: ArrayLike,
    n_atoms: int,
    pressure: float,
    temperature: float,
    kb: float = KB,
) -> NDArray:
    """The density of V at fixed N, P and T for atoms that do not interact.

    V^N e^{−βPV} divided by its integral N!/(βP)^{N+1}: the gamma density
    of shape N + 1 and scale k_BT/P, whose mean is (N + 1)k_BT/P.

    >>> v = np.linspace(1e-9, 400.0, 400001)
    >>> p = ideal_gas_volume_density(v, 10, 0.5, 1.0, kb=1.0)
    >>> print(round(float(np.trapezoid(p, v)), 6),
    ...       round(float(np.trapezoid(v * p, v)), 4))
    1.0 22.0
    """
    v = np.asarray(volume, dtype=float)
    beta_p = pressure / (kb * temperature)
    log = (n_atoms * np.log(v) - beta_p * v + (n_atoms + 1) * np.log(beta_p)
           - gammaln(n_atoms + 1))
    return np.exp(log)
