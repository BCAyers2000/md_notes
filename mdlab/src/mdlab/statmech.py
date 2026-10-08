"""Statistical mechanics of a simulation: Chapter 12.

- ``degrees_of_freedom``: 3N, less the held distances and, if the drift of
  the centre of mass is removed, three more (Sections 11.2 and 12.6);
- ``kinetic_temperature``: T = 2K/(N_f k_B) (Section 12.6);
- ``part_temperature``: the temperature of some of the atoms, or of one
  direction, counting every component of their velocities;
- ``thermal_velocities``: velocities drawn from the Maxwell-Boltzmann
  distribution, each component Gaussian with variance k_BT/m (Section
  12.5), in the order ASE draws them;
- ``maxwell_speed_density`` and ``kinetic_energy_density``: the
  probability densities of an atom's speed and of the kinetic energy of
  N_f freedoms in the canonical ensemble (Sections 12.5 and 12.7);
- ``boltzmann_factor``: e^{−ΔE/k_BT} (Section 12.4);
- ``heat_capacity_nve``: the heat capacity from the spread of the kinetic
  energy at fixed energy (Section 12.7).

Units
-----
Masses in amu, velocities in Å/fs, energies in eV, temperatures in K,
with k_B = ``units.KB``; or reduced units with ``kb = mv2_to_energy = 1``.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, xlogy

from mdlab.units import KB, MV2_TO_EV


def degrees_of_freedom(
    n_atoms: int, n_constraints: int = 0, drift_removed: bool = True
) -> int:
    """The number of freedoms among which the kinetic energy is shared.

    >>> degrees_of_freedom(192, 192), degrees_of_freedom(256)
    (381, 765)
    """
    return 3 * n_atoms - n_constraints - (3 if drift_removed else 0)


def kinetic_energy(
    masses: ArrayLike, velocities: ArrayLike, mv2_to_energy: float = MV2_TO_EV
) -> NDArray | float:
    """K = ½ Σ mᵢ|vᵢ|², for one frame (N, 3) or many (frames, N, 3), in eV."""
    m = np.asarray(masses, dtype=float)
    v = np.asarray(velocities, dtype=float)
    k = 0.5 * mv2_to_energy * np.einsum("i,...ix,...ix->...", m, v, v)
    return float(k) if np.ndim(k) == 0 else k


def kinetic_temperature(
    masses: ArrayLike,
    velocities: ArrayLike,
    n_free: int,
    kb: float = KB,
    mv2_to_energy: float = MV2_TO_EV,
) -> NDArray | float:
    """The kinetic temperature 2K/(N_f k_B), in K, for one frame or many.

    >>> m = np.ones(2)
    >>> v = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    >>> print(kinetic_temperature(m, v, 3, kb=1.0, mv2_to_energy=1.0))
    0.6666666666666666
    """
    k = kinetic_energy(masses, velocities, mv2_to_energy)
    return 2 * k / (n_free * kb)


def part_temperature(
    masses: ArrayLike,
    velocities: ArrayLike,
    atoms: ArrayLike | None = None,
    axis: int | None = None,
    kb: float = KB,
    mv2_to_energy: float = MV2_TO_EV,
) -> NDArray | float:
    """The temperature of some atoms, or of one direction, in K.

    Every velocity component of the chosen atoms counts as one freedom,
    with nothing taken off for the drift or for held distances, so the
    temperatures of the parts of a large system can be compared with one
    another.

    >>> v = np.array([[1.0, 0, 0], [0.0, 2.0, 0]])
    >>> print(part_temperature([1, 1], v, axis=1, kb=1, mv2_to_energy=1))
    2.0
    """
    m = np.asarray(masses, dtype=float)
    v = np.asarray(velocities, dtype=float)
    chosen = np.arange(len(m)) if atoms is None else np.asarray(atoms)
    v = v[..., chosen, :]
    if axis is not None:
        v = v[..., axis : axis + 1]
    k = 0.5 * mv2_to_energy * np.einsum("i,...ix,...ix->...", m[chosen], v, v)
    return 2 * k / (v.shape[-1] * len(chosen) * kb)


def thermal_velocities(
    masses: ArrayLike,
    temperature: float,
    rng: np.random.Generator,
    remove_drift: bool = True,
    kb: float = KB,
    mv2_to_energy: float = MV2_TO_EV,
) -> NDArray:
    """Velocities from the Maxwell-Boltzmann distribution at ``temperature``.

    Each component is k_BT/m to the power ½ times a number from the
    standard normal distribution, drawn in the same order as ASE's
    ``thermalize_momenta``; the drift of the centre of mass is then
    removed if asked. The temperature of the result scatters about the one
    asked for by √(2/N_f) of it (Section 12.7).

    >>> m = np.full(1000, 39.948)
    >>> v = thermal_velocities(m, 300.0, np.random.default_rng(0))
    >>> sigma = np.sqrt(KB * 300 / (39.948 * MV2_TO_EV))
    >>> print(round(float(np.std(v) / sigma), 2))
    0.99
    """
    m = np.asarray(masses, dtype=float)
    xi = rng.standard_normal((len(m), 3))
    v = xi * np.sqrt(kb * temperature / (m * mv2_to_energy))[:, None]
    if remove_drift:
        v = v - (m @ v) / m.sum()
    return v


def maxwell_speed_density(
    speed: ArrayLike,
    mass: float,
    temperature: float,
    kb: float = KB,
    mv2_to_energy: float = MV2_TO_EV,
) -> NDArray:
    """The probability density of an atom's speed, in fs/Å.

    4π v² (m/2πk_BT)^{3/2} e^{−mv²/2k_BT}, with m v² in eV through
    ``mv2_to_energy``.

    >>> s = np.linspace(0, 0.05, 20001)
    >>> p = maxwell_speed_density(s, 39.948, 300.0)
    >>> print(round(float(np.trapezoid(p, s)), 6))
    1.0
    """
    v = np.asarray(speed, dtype=float)
    a = mass * mv2_to_energy / (kb * temperature)  # m/k_BT in (fs/Å)²
    return 4 * math.pi * v * v * (a / (2 * math.pi)) ** 1.5 * np.exp(
        -0.5 * a * v * v
    )


def kinetic_energy_density(
    energy: ArrayLike, n_free: int, temperature: float, kb: float = KB
) -> NDArray:
    """The canonical probability density of the kinetic energy, in 1/eV.

    K^{N_f/2 − 1} e^{−K/k_BT}, divided by its integral, (k_BT)^{N_f/2}
    Γ(N_f/2), computed through the logarithm of the gamma function.

    >>> k = np.linspace(1e-9, 2.0, 200001)
    >>> p = kinetic_energy_density(k, 30, 300.0)
    >>> print(round(float(np.trapezoid(p, k)), 6),
    ...       round(float(np.trapezoid(k * p, k) / (15 * KB * 300)), 6))
    1.0 1.0
    """
    k = np.asarray(energy, dtype=float)
    kt = kb * temperature
    half = 0.5 * n_free
    # xlogy gives 0 at K = 0 for two freedoms, where (half - 1) log K is 0·(−∞)
    log = xlogy(half - 1, k) - k / kt - half * math.log(kt) - gammaln(half)
    return np.exp(log)


def boltzmann_factor(energy: ArrayLike, temperature: float, kb: float = KB):
    """e^{−E/k_BT}: the odds of a state E above another, at ``temperature``.

    >>> print(f"{boltzmann_factor(0.3, 300.0):.2e}")
    9.12e-06
    """
    return np.exp(-np.asarray(energy, dtype=float) / (kb * temperature))


def heat_capacity_nve(
    kinetic: ArrayLike, n_free: int, kb: float = KB
) -> float:
    """The heat capacity from the kinetic energy of a run at fixed energy.

    At fixed total energy the kinetic energy fluctuates less than in the
    canonical ensemble, by
    ⟨δK²⟩ = (N_f/2)(k_BT)² (1 − N_f k_B/(2C_V))
    (Lebowitz, Percus and Verlet 1967), so
    C_V = N_f k_B / (2 (1 − 2⟨δK²⟩/(N_f (k_BT)²))), in eV/K, with T the
    kinetic temperature of the run.
    """
    k = np.asarray(kinetic, dtype=float)
    kt = 2 * k.mean() / n_free  # k_B T
    ratio = 2 * k.var() / (n_free * kt * kt)
    return n_free * kb / (2 * (1 - ratio))
