"""The bonded terms of a force field, with forces: Chapter 16.

The energy of Section 8.7 without the pair terms,

    U = Σ_bonds κ_r (r − r_eq)² + Σ_angles κ_θ (θ − θ_eq)²
        + Σ_dihedrals Σ_n (κ_n/2) [1 + (−1)^{n+1} cos nψ],

for molecules held together only by these terms (Section 16.11). The
gradient of the dihedral angle is that of Blondel and Karplus (1996).
Positions may carry leading axes, (..., N, 3), for many independent
copies of the same molecule at once.

Units
-----
Lengths in Å, angles in rad, energies in eV.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _add(forces: NDArray, index: tuple, values: NDArray) -> None:
    """Add values (copies, terms, atoms of a term, 3) to their atoms."""
    for column, atom in enumerate(index):
        np.add.at(forces, (slice(None), atom), values[:, :, column])


class BondedModel:
    """Bonds, angles and dihedrals, each a list of atom indices.

    ``bonds`` rows (i, j) with ``bond_k`` κ_r and ``bond_r`` r_eq;
    ``angles`` rows (i, j, k), j at the vertex, with ``angle_k`` κ_θ and
    ``angle_theta`` θ_eq; ``dihedrals`` rows (a, b, c, d) with
    ``torsion`` the constants κ₁, κ₂, κ₃ shared by all of them.
    """

    def __init__(self, bonds, bond_k, bond_r, angles, angle_k, angle_theta,
                 dihedrals, torsion):
        self.bonds = np.asarray(bonds, dtype=int).reshape(-1, 2)
        self.angles = np.asarray(angles, dtype=int).reshape(-1, 3)
        self.dihedrals = np.asarray(dihedrals, dtype=int).reshape(-1, 4)
        self.bond_k, self.bond_r = bond_k, bond_r
        self.angle_k, self.angle_theta = angle_k, angle_theta
        self.torsion = np.asarray(torsion, dtype=float)

    def __call__(self, positions: ArrayLike) -> tuple[NDArray, NDArray]:
        """The energy of each copy and the forces on its atoms."""
        r = np.asarray(positions, dtype=float)
        lead = r.shape[:-2]
        r = r.reshape(-1, *r.shape[-2:])
        forces = np.zeros_like(r)
        energy = np.zeros(len(r))

        i, j = self.bonds.T
        d = r[:, j] - r[:, i]
        length = np.linalg.norm(d, axis=-1)
        stretch = length - self.bond_r
        energy += np.sum(self.bond_k * stretch**2, axis=-1)
        push = (2 * self.bond_k * stretch / length)[..., None] * d
        _add(forces, (i, j), np.stack([push, -push], axis=2))

        i, j, k = self.angles.T
        u, w = r[:, i] - r[:, j], r[:, k] - r[:, j]
        lu = np.linalg.norm(u, axis=-1)[..., None]
        lw = np.linalg.norm(w, axis=-1)[..., None]
        cos = np.clip(np.sum(u * w, -1) / (lu * lw)[..., 0], -1, 1)
        theta = np.arccos(cos)
        bend = theta - self.angle_theta
        energy += np.sum(self.angle_k * bend**2, axis=-1)
        # dθ/du = −(ŵ − cos θ û)/(|u| sin θ), and likewise for w; singular
        # for a straight angle, θ = 0 or π
        scale = (2 * self.angle_k * bend / np.sin(theta))[..., None]
        f_i = scale * (w / lw - cos[..., None] * u / lu) / lu
        f_k = scale * (u / lu - cos[..., None] * w / lw) / lw
        _add(forces, (i, j, k), np.stack([f_i, -f_i - f_k, f_k], axis=2))

        a, b, c, e = self.dihedrals.T
        b1, b2, b3 = r[:, b] - r[:, a], r[:, c] - r[:, b], r[:, e] - r[:, c]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        l2 = np.linalg.norm(b2, axis=-1)[..., None]
        x = np.sum(n1 * n2, -1)
        y = np.sum(np.cross(n1, n2) * b2, -1) / l2[..., 0]
        psi = np.arctan2(y, x)
        n = np.arange(1, len(self.torsion) + 1)
        sign = (-1.0) ** (n + 1)
        energy += np.sum(0.5 * self.torsion * (1 + sign * np.cos(
            n * psi[..., None])), axis=(-1, -2))
        du = -np.sum(0.5 * self.torsion * sign * n * np.sin(
            n * psi[..., None]), axis=-1)[..., None]
        a1 = np.sum(n1 * n1, -1)[..., None]
        a2 = np.sum(n2 * n2, -1)[..., None]
        g_a = -l2 / a1 * n1
        g_d = l2 / a2 * n2
        p = np.sum(b1 * b2, -1)[..., None] / (l2 * l2)
        s = np.sum(b3 * b2, -1)[..., None] / (l2 * l2)
        g_b = -(1 + p) * g_a + s * g_d
        g_c = p * g_a - (1 + s) * g_d
        _add(forces, (a, b, c, e),
             -du[..., None] * np.stack([g_a, g_b, g_c, g_d], axis=2))
        return energy.reshape(lead), forces.reshape(*lead, *r.shape[-2:])
