"""Model water in a periodic box: Chapter 11.

The TIP3P model of Jorgensen and co-workers (1983), with the parameters
ASE uses: charges −0.834 e on oxygen and 0.417 e on each hydrogen, a
Lennard-Jones energy between oxygen atoms only (σ = 3.15061 Å,
ε = 0.1521 kcal/mol), and a rigid molecule with O–H = 0.9572 Å and
H–O–H = 104.52°.

- ``molecule``: the three atoms of one molecule, centre of mass at the
  origin;
- ``box``: n³ molecules on a cubic grid at a given density, each turned
  at random;
- ``constraint_bonds``: the pairs held fixed, all three distances of
  each molecule (rigid) or the two O–H bonds only;
- ``WaterModel``: the energy and forces. Between molecules, the
  Lennard-Jones energy switched off between ``r_switch`` and ``r_cut``,
  and the Coulomb energy of all the charges by the Ewald sum, less that of
  the pairs within each molecule; inside each molecule, if ``flexible``,
  the three springs of Section 5.7 about the TIP3P geometry.

Atoms are ordered O, H, H for each molecule in turn.

Units
-----
Positions in Å, energies in eV, masses in amu, charges in e.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from mdlab.cell import minimum_image
from mdlab.ewald import ewald_energy_forces
from mdlab.neighbours import all_pairs
from mdlab.potentials import lennard_jones, with_cutoff
from mdlab.units import COULOMB, KCAL_PER_MOL

#: Masses of O and H in amu, the standard atomic weights ASE uses.
MASS_O, MASS_H = 15.999, 1.008
#: TIP3P charges in e.
CHARGE_O, CHARGE_H = -0.834, 0.417
#: TIP3P Lennard-Jones parameters between oxygen atoms, in Å and eV.
SIGMA_OO, EPSILON_OO = 3.15061, 0.1521 * KCAL_PER_MOL
#: TIP3P geometry: the O–H length in Å and the H–O–H angle in degrees.
R_OH, ANGLE_HOH = 0.9572, 104.52
#: The H–H distance of that geometry, in Å.
R_HH = 2 * R_OH * math.sin(math.radians(ANGLE_HOH) / 2)
#: Spring stiffnesses of Section 5.7, O–H and H–H, in eV/Å².
K_OH, K_HH = 48.48, 17.66
#: Molar mass of water in g/mol.
MOLAR_MASS = MASS_O + 2 * MASS_H


def molecule() -> NDArray:
    """The positions of O, H and H, in Å, with the centre of mass at 0.

    >>> r = molecule()
    >>> print(np.linalg.norm(r[1] - r[0]).round(4),
    ...       np.linalg.norm(r[2] - r[1]).round(4))
    0.9572 1.5139
    """
    angle = math.radians(ANGLE_HOH)
    r = np.array(
        [
            [0.0, 0.0, 0.0],
            [R_OH, 0.0, 0.0],
            [R_OH * math.cos(angle), R_OH * math.sin(angle), 0.0],
        ]
    )
    m = np.array([MASS_O, MASS_H, MASS_H])
    return r - m @ r / m.sum()


def box(
    n: int, density: float, rng: np.random.Generator
) -> tuple[NDArray, NDArray, NDArray, list[str]]:
    """n³ water molecules on a cubic grid, each turned at random.

    Parameters
    ----------
    n : int
        Molecules along each side.
    density : float
        Mass density, in g/cm³.
    rng : numpy.random.Generator

    Returns
    -------
    positions : ndarray, shape (3n³, 3)
        In Å, ordered O, H, H for each molecule.
    cell : ndarray, shape (3, 3)
        A cube, lattice vectors as columns, in Å.
    masses : ndarray, shape (3n³,)
        In amu.
    symbols : list of str
    """
    per_cubic_angstrom = density / MOLAR_MASS * 6.02214076e23 * 1e-24
    spacing = per_cubic_angstrom ** (-1 / 3)
    grid = spacing * (
        np.array(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"))
        .reshape(3, -1)
        .T
        + 0.5
    )
    turns = Rotation.random(len(grid), random_state=rng)
    shape = molecule()
    r = np.concatenate([turns[k].apply(shape) + grid[k]
                        for k in range(len(grid))])
    masses = np.tile([MASS_O, MASS_H, MASS_H], len(grid))
    return r, n * spacing * np.eye(3), masses, ["O", "H", "H"] * len(grid)


def constraint_bonds(
    n_molecules: int, rigid: bool = True
) -> tuple[NDArray, NDArray]:
    """The pairs held fixed and their lengths, in Å.

    ``rigid`` holds O–H, H–H and H–O of each molecule, which fixes its
    shape; otherwise only the two O–H bonds, leaving the angle free.

    >>> bonds, lengths = constraint_bonds(2)
    >>> print(bonds.tolist(), lengths.round(4).tolist()[:3])
    [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]] [0.9572, 1.5139, 0.9572]
    """
    first = 3 * np.arange(n_molecules)[:, None]
    if rigid:
        pattern, lengths = [[0, 1], [1, 2], [2, 0]], [R_OH, R_HH, R_OH]
    else:
        pattern, lengths = [[0, 1], [0, 2]], [R_OH, R_OH]
    bonds = (first[:, :, None] + np.array(pattern)[None]).reshape(-1, 2)
    return bonds, np.tile(lengths, n_molecules)


class WaterModel:
    """TIP3P water in a periodic cell, rigid or with springs.

    Calling the model returns the total potential energy and forces;
    ``intermolecular`` and ``intramolecular`` return the two parts
    separately, for the multiple-time-step method of Section 11.4.
    """

    def __init__(
        self,
        cell: ArrayLike,
        n_molecules: int,
        flexible: bool = False,
        r_cut: float = 6.0,
        r_switch: float = 5.0,
        accuracy: float = 1e-6,
    ):
        self.cell = np.asarray(cell, dtype=float)
        self.n = n_molecules
        self.flexible = flexible
        self.r_cut = r_cut
        self.lj = with_cutoff(
            lambda r: lennard_jones(r, EPSILON_OO, SIGMA_OO),
            r_cut, "switch", r_switch,
        )
        self.charges = np.tile([CHARGE_O, CHARGE_H, CHARGE_H], n_molecules)
        self.accuracy = accuracy
        first = 3 * np.arange(n_molecules)
        self.inner = [(first, first + 1), (first, first + 2),
                      (first + 1, first + 2)]

    def lennard_jones(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """The Lennard-Jones energy between oxygen atoms, and its forces."""
        r = np.asarray(positions, dtype=float)
        f = np.zeros_like(r)
        oxygen = np.arange(0, 3 * self.n, 3)
        i, j, d, dist = all_pairs(r[oxygen], self.cell, self.r_cut)
        phi, dphi = self.lj(dist)
        push = (dphi / dist)[:, None] * d
        np.add.at(f, oxygen[i], push)
        np.add.at(f, oxygen[j], -push)
        return float(np.sum(phi)), f

    def coulomb(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """The Coulomb energy between molecules, by the Ewald sum."""
        r = np.asarray(positions, dtype=float)
        u, f, _ = ewald_energy_forces(
            self.charges, r, self.cell, accuracy=self.accuracy
        )
        for a, b in self.inner:  # remove each molecule's own pairs
            d = minimum_image(r[b] - r[a], self.cell)
            dist = np.linalg.norm(d, axis=1)
            qq = COULOMB * self.charges[a] * self.charges[b]
            u -= float(np.sum(qq / dist))
            push = (qq / dist**3)[:, None] * d  # the Coulomb force on b
            f[b] -= push
            f[a] += push
        return u, f

    def intermolecular(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """Lennard-Jones between oxygens plus Coulomb between molecules."""
        u_lj, f_lj = self.lennard_jones(positions)
        u_coulomb, f_coulomb = self.coulomb(positions)
        return u_lj + u_coulomb, f_lj + f_coulomb

    def intramolecular(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """The springs inside each molecule; zero for rigid water."""
        r = np.asarray(positions, dtype=float)
        f = np.zeros_like(r)
        if not self.flexible:
            return 0.0, f
        u = 0.0
        for (a, b), k, length in zip(
            self.inner, (K_OH, K_OH, K_HH), (R_OH, R_OH, R_HH), strict=True
        ):
            d = minimum_image(r[b] - r[a], self.cell)
            dist = np.linalg.norm(d, axis=1)
            stretch = dist - length
            u += 0.5 * k * float(np.sum(stretch**2))
            pull = (k * stretch / dist)[:, None] * d  # the force on a
            f[a] += pull
            f[b] -= pull
        return u, f

    def __call__(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """The total potential energy and forces."""
        u_inter, f_inter = self.intermolecular(positions)
        u_intra, f_intra = self.intramolecular(positions)
        return u_inter + u_intra, f_inter + f_intra
