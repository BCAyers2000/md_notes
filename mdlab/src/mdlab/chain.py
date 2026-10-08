"""A chain of atoms joined by springs: Chapter 12.

N atoms of unit mass move along a line between two fixed walls, each joined
to its neighbours by a bond of rest length 1 whose energy, stretched by x,
is ½x² + (α/3)x³. With α = 0 the chain is harmonic, and its normal modes
(Section 4.9) never exchange energy; with α > 0 it is the chain of Fermi,
Pasta, Ulam and Tsingou, whose modes do, slowly (Section 12.2).

- ``energy_forces``: the energy and the forces of displacements q;
- ``normal_modes``: the patterns and angular frequencies of the harmonic
  chain, sin(jkπ/(N + 1)) and 2 sin(kπ/(2(N + 1)));
- ``mode_energies``: the energy in each mode, ½(ȧₖ² + ωₖ² aₖ²);
- ``run``: velocity Verlet, recording the mode energies.

Units
-----
Dimensionless: masses, spring constants and the rest length are 1.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def energy_forces(q: ArrayLike, alpha: float = 0.0) -> tuple[float, NDArray]:
    """The energy and forces of displacements ``q`` from rest.

    >>> u, f = energy_forces([0.1, 0.0])
    >>> print(round(u, 6), f.round(6).tolist())
    0.01 [-0.2, 0.1]
    """
    q = np.asarray(q, dtype=float)
    x = np.diff(np.concatenate([[0.0], q, [0.0]]))  # bond stretches
    tension = x + alpha * x * x
    u = float(np.sum(0.5 * x * x + alpha / 3 * x**3))
    return u, tension[1:] - tension[:-1]


def normal_modes(n: int) -> tuple[NDArray, NDArray]:
    """Mode patterns (columns, each of length 1) and angular frequencies.

    >>> s, w = normal_modes(3)
    >>> print(np.allclose(s.T @ s, np.eye(3)), w.round(4).tolist())
    True [0.7654, 1.4142, 1.8478]
    """
    j = np.arange(1, n + 1)
    patterns = np.sqrt(2 / (n + 1)) * np.sin(np.outer(j, j) * np.pi / (n + 1))
    omega = 2 * np.sin(j * np.pi / (2 * (n + 1)))
    return patterns, omega


def mode_energies(q: ArrayLike, p: ArrayLike) -> NDArray:
    """The harmonic energy ½(ȧₖ² + ωₖ²aₖ²) in each mode.

    For one state (N,) or many (frames, N).
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    patterns, omega = normal_modes(q.shape[-1])
    a, rate = q @ patterns, p @ patterns
    return 0.5 * (rate * rate + omega**2 * a * a)


def run(
    q: ArrayLike,
    p: ArrayLike,
    alpha: float,
    dt: float,
    n_steps: int,
    every: int = 1,
) -> dict[str, NDArray]:
    """Velocity Verlet for the chain, recording every ``every`` steps.

    Returns "times", "q", "p", "modes" (the energy in each mode) and
    "energy" (the total).
    """
    q = np.array(q, dtype=float)
    p = np.array(p, dtype=float)
    u, f = energy_forces(q, alpha)
    out: dict[str, list] = {"times": [], "q": [], "p": [], "energy": []}
    for step in range(n_steps + 1):
        if step > 0:
            p += 0.5 * dt * f
            q += dt * p
            u, f = energy_forces(q, alpha)
            p += 0.5 * dt * f
        if step % every == 0:
            out["times"].append(step * dt)
            out["q"].append(q.copy())
            out["p"].append(p.copy())
            out["energy"].append(u + 0.5 * float(p @ p))
    result = {key: np.array(value) for key, value in out.items()}
    result["modes"] = mode_energies(result["q"], result["p"])
    return result
