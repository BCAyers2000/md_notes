"""The Ewald sum of point charges in a periodic cell: Chapter 10.

The energy of charges qᵢ at rᵢ in a cell h, with all their copies, is

    E = ½ Σ' qᵢ qⱼ k_e / |rⱼ − rᵢ + h n|,

summed over the pairs and the whole-number vectors n, leaving out i = j
at n = 0. Splitting 1/r = erfc(αr)/r + erf(αr)/r (Section 10.3) gives

- a real-space sum of qᵢqⱼ erfc(αr)/r over the copies within r_cut;
- a reciprocal-space sum (2π k_e/V) Σ_{G≠0} e^{−G²/4α²} |S(G)|²/G², with
  the structure factor |S(G)|² = (Σ q cos G·r)² + (Σ q sin G·r)²;
- the self term −k_e α/√π Σ qᵢ²;
- for a cell that is not neutral, the energy −π k_e Q²/(2Vα²) of the
  uniform background that neutralises it.

The total does not depend on α, which only moves work between the two
sums.

Units
-----
Energies in eV, lengths in Å, charges in units of the elementary charge,
with k_e = ``units.COULOMB``; or k_e = 1 for the dimensionless sums of
Section 8.8.
"""

import itertools
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erfc

from mdlab.cell import (
    cell_volume,
    perpendicular_widths,
    reciprocal_vectors,
    wrap,
)
from mdlab.units import COULOMB


def ewald_energy_forces(
    charges: ArrayLike,
    positions: ArrayLike,
    cell: ArrayLike,
    alpha: float | None = None,
    accuracy: float = 1e-10,
    coulomb: float = COULOMB,
) -> tuple[float, NDArray, dict[str, float]]:
    """The Ewald energy, the forces and the parts of the energy.

    Parameters
    ----------
    charges : array_like, shape (N,)
        Charges, in units of e.
    positions : array_like, shape (N, 3)
        Positions, in Å.
    cell : array_like, shape (3, 3)
        Lattice vectors as columns, in Å.
    alpha : float, optional
        The splitting parameter, in 1/Å. By default √π (N/V²)^(1/6),
        which balances the cost of the two sums.
    accuracy : float
        The size of the terms left out: both sums stop where their
        Gaussian factor falls below it, at r_cut = √(−ln ε)/α and
        G_cut = 2α√(−ln ε).
    coulomb : float
        The constant k_e.

    Returns
    -------
    energy : float
    forces : ndarray, shape (N, 3)
    parts : dict
        'real', 'reciprocal', 'self' and 'background'.

    Examples
    --------
    Rock salt in its cubic cell of side 2, with d = 1 and k_e = 1: the
    energy per ion pair is −𝓜, the Madelung constant.

    >>> fcc = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> pos = np.vstack([fcc, fcc + [1, 0, 0]])
    >>> q = np.array([1] * 4 + [-1] * 4)
    >>> e, f, _ = ewald_energy_forces(q, pos, 2 * np.eye(3), coulomb=1.0)
    >>> print(f"{-e / 4:.6f}")
    1.747565
    """
    q = np.asarray(charges, dtype=float)
    h = np.asarray(cell, dtype=float)
    # The copies searched below reach every partner only from inside the
    # cell, and the loop of Section 10.6 does not wrap.
    r = wrap(np.asarray(positions, dtype=float), h)
    n = len(q)
    volume = cell_volume(h)
    if alpha is None:
        alpha = math.sqrt(math.pi) * (n / volume**2) ** (1 / 6)
    reach = math.sqrt(-math.log(accuracy))
    r_cut, g_cut = reach / alpha, 2 * alpha * reach

    # Real space: every copy within r_cut, leaving out an atom with itself.
    e_real, f = 0.0, np.zeros_like(r)
    counts = np.ceil(r_cut / perpendicular_widths(h)).astype(int)
    for shift in itertools.product(*(range(-c, c + 1) for c in counts)):
        d = r[None, :, :] - r[:, None, :] + h @ np.array(shift, dtype=float)
        dist = np.linalg.norm(d, axis=2)
        keep = dist < r_cut
        if not any(shift):
            np.fill_diagonal(keep, False)
        if not keep.any():
            continue
        i, j = np.nonzero(keep)
        x, qq = dist[i, j], q[i] * q[j]
        screened = erfc(alpha * x) / x
        e_real += 0.5 * np.sum(qq * screened)
        gauss = 2 * alpha / math.sqrt(math.pi) * np.exp(-((alpha * x) ** 2))
        slope = -(screened + gauss) / x  # the derivative of erfc(αx)/x
        np.add.at(f, i, (qq * slope / x)[:, None] * d[i, j])

    # Reciprocal space: the waves G = m₁b₁ + m₂b₂ + m₃b₃ with 0 < |G| ≤ G_cut.
    b = reciprocal_vectors(h)
    m_max = np.ceil(g_cut * np.linalg.norm(h, axis=0) / (2 * np.pi))
    grid = np.array(
        list(itertools.product(*(range(-int(k), int(k) + 1) for k in m_max))),
        dtype=float,
    )
    g = grid @ b
    g2 = np.einsum("kx,kx->k", g, g)
    keep = (g2 > 0) & (g2 <= g_cut**2)
    g, g2 = g[keep], g2[keep]
    weight = np.exp(-g2 / (4 * alpha**2)) / g2
    phase = r @ g.T
    cos, sin = np.cos(phase), np.sin(phase)
    c_sum, s_sum = q @ cos, q @ sin
    e_rec = 2 * np.pi / volume * np.sum(weight * (c_sum**2 + s_sum**2))
    push = (sin * c_sum - cos * s_sum) * weight
    f += 4 * np.pi / volume * q[:, None] * (push @ g)

    e_self = -alpha / math.sqrt(math.pi) * np.sum(q * q)
    e_background = -np.pi * np.sum(q) ** 2 / (2 * volume * alpha**2)
    parts = {
        "real": coulomb * e_real,
        "reciprocal": coulomb * e_rec,
        "self": coulomb * e_self,
        "background": coulomb * e_background,
    }
    return float(sum(parts.values())), coulomb * f, parts
