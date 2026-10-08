"""Trajectory files: Chapter 10.

The extended XYZ format, which ASE, OVITO and most learned potentials
read: for each frame, the number of atoms; a line of key=value pairs with
the lattice vectors, the columns that follow and any scalar such as the
energy; then one line per atom.

- ``write_extxyz``: frames of positions, with velocities if given;
- ``read_extxyz``: the frames back, as a list of dicts;
- ``read_frames``: the frames of any trajectory file ASE can read, extxyz
  or ASE's own, with velocities in Å/fs (Section 16.4);
- ``AseModel``: an ASE calculator as an ``mdlab`` model, called with the
  positions and returning the energy and the forces (Section 16.4);
- ``mdlab_calculator``: the reverse, an ``mdlab`` model as an ASE
  calculator, with the stress from its virial (Section 17.1);
- ``lammps_table``: a pair potential written as a LAMMPS table
  (Section 17.2);
- ``read_run``: a trajectory that ASE can read, as the arrays that
  ``diagnostics.health_report`` takes (Section 17.8).

Units
-----
Whatever the caller uses; ``mdlab`` writes Å, Å/fs and eV. ASE's unit of
time is 1/√``units.FORCE_TO_ACCEL`` fs, about 10.18 fs, so its velocities
are those in Å/fs divided by √``units.FORCE_TO_ACCEL``.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mdlab.units import FORCE_TO_ACCEL


def write_extxyz(
    path: str,
    symbols: Sequence[str],
    frames: Sequence[ArrayLike],
    cell: ArrayLike,
    velocities: Sequence[ArrayLike] | None = None,
    scalars: dict[str, Sequence[float]] | None = None,
) -> None:
    """Write ``frames`` of positions to ``path`` in extended XYZ.

    The lattice vectors, the columns of ``cell``, are written in the order
    a, b, c, each as three numbers. ``scalars`` maps names to one number
    per frame; by the convention of the format, and as ASE reads it,
    "energy" is the potential energy.
    """
    h = np.asarray(cell, dtype=float)
    lattice = " ".join(f"{x:.10f}" for x in h.T.ravel())
    columns = "species:S:1:pos:R:3" + (
        ":vel:R:3" if velocities is not None else ""
    )
    scalars = scalars or {}
    with open(path, "w") as out:
        for k, frame in enumerate(frames):
            r = np.asarray(frame, dtype=float)
            extra = "".join(
                f" {name}={values[k]:.10f}" for name, values in scalars.items()
            )
            out.write(f"{len(r)}\n")
            out.write(
                f'Lattice="{lattice}" Properties={columns}{extra} '
                f'pbc="T T T"\n'
            )
            rows = (
                r
                if velocities is None
                else np.hstack([r, np.asarray(velocities[k], dtype=float)])
            )
            for symbol, row in zip(symbols, rows, strict=True):
                out.write(symbol + "".join(f" {x:.10f}" for x in row) + "\n")


def read_extxyz(path: str) -> list[dict]:
    """Read the frames of an extended XYZ file written by ``write_extxyz``.

    Each frame is a dict with "symbols", "positions", "cell" (lattice
    vectors as columns), "velocities" when present, and each scalar of
    the header under its own name.
    """
    frames = []
    with open(path) as source:
        lines = source.read().splitlines()
    k = 0
    while k < len(lines) and lines[k].strip():
        n = int(lines[k])
        header = lines[k + 1]
        lattice = header.split('Lattice="')[1].split('"')[0]
        cell = np.array(lattice.split(), dtype=float).reshape(3, 3).T
        rows = [lines[k + 2 + a].split() for a in range(n)]
        data: NDArray = np.array([row[1:] for row in rows], dtype=float)
        frame = {
            "symbols": [row[0] for row in rows],
            "positions": data[:, :3],
            "cell": cell,
        }
        if data.shape[1] >= 6:
            frame["velocities"] = data[:, 3:6]
        rest = header.split('"', 2)[2]  # after the lattice
        for item in rest.split():
            name, _, value = item.partition("=")
            if name not in ("Properties", "pbc") and value:
                frame[name] = float(value)
        frames.append(frame)
        k += n + 2
    return frames


def read_frames(path: str, index: str = ":") -> list[dict]:
    """The frames of a trajectory file that ASE can read, as dicts.

    Each frame has "symbols", "positions" (Å), "cell" (lattice vectors as
    columns, Å) and, when the file holds them, "velocities" in Å/fs.
    """
    from ase.io import read

    images = read(path, index=index)
    images = images if isinstance(images, list) else [images]
    frames = []
    for atoms in images:
        frame = {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions(),
            "cell": np.array(atoms.get_cell()).T,
        }
        if atoms.has("momenta"):
            frame["velocities"] = atoms.get_velocities() * np.sqrt(
                FORCE_TO_ACCEL
            )
        elif "vel" in atoms.arrays:  # the column of write_extxyz, in Å/fs
            frame["velocities"] = atoms.arrays["vel"].copy()
        frames.append(frame)
    return frames


class AseModel:
    """An ASE calculator called as an ``mdlab`` model.

    ``model(positions)`` returns the potential energy in eV and the forces
    in eV/Å, as every model of ``mdlab.md`` does, so ``md.run`` and
    ``thermostats.run`` can drive a potential that only ASE provides.
    """

    def __init__(self, calculator, symbols: Sequence[str], cell: ArrayLike):
        from ase import Atoms

        h = np.asarray(cell, dtype=float)
        self.atoms = Atoms(symbols, cell=h.T, pbc=True)
        self.atoms.calc = calculator

    @property
    def cell(self) -> NDArray:
        """The cell, lattice vectors as columns."""
        return np.array(self.atoms.get_cell()).T

    def set_cell(self, cell: ArrayLike) -> None:
        """Change the cell, keeping the positions."""
        self.atoms.set_cell(np.asarray(cell, dtype=float).T)

    def __call__(self, positions: ArrayLike) -> tuple[float, NDArray]:
        """The potential energy and the forces at ``positions``."""
        self.atoms.set_positions(np.asarray(positions, dtype=float))
        return (
            float(self.atoms.get_potential_energy()),
            self.atoms.get_forces(),
        )


def mdlab_calculator(model):
    """An ``mdlab`` model as an ASE calculator.

    ``model(positions)`` must return the energy in eV and the forces in
    eV/Å. If the model keeps a virial tensor W_αβ = Σ r_α F_β of its last
    call, as ``md.PairModel`` does, the calculator also gives ASE's stress,
    −W/V in Voigt order (xx, yy, zz, yz, xz, xy), in eV/Å³. A change of
    the cell is passed on through ``model.set_cell``.
    """
    from ase.calculators.calculator import Calculator, all_changes
    class MdlabCalculator(Calculator):
        implemented_properties = ["energy", "free_energy", "forces",
                                  "stress"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            h = np.array(self.atoms.get_cell()).T
            # any change at all, since a strain of 1e-5 must be seen
            if hasattr(model, "set_cell") and not np.array_equal(
                    h, model.cell):
                model.set_cell(h)
            energy, forces = model(self.atoms.get_positions())
            self.results = {"energy": float(energy),
                            "free_energy": float(energy),
                            "forces": np.array(forces)}
            if hasattr(model, "virial"):
                w = -np.asarray(model.virial) / self.atoms.get_volume()
                # Voigt order: xx, yy, zz, yz, xz, xy
                self.results["stress"] = w[[0, 1, 2, 1, 0, 0],
                                           [0, 1, 2, 2, 2, 1]]

    return MdlabCalculator()


def lammps_table(
    pair,
    r_min: float,
    r_max: float,
    n: int,
    path: str,
    keyword: str = "PAIR",
) -> None:
    """Write the pair potential ``pair`` as a LAMMPS table.

    ``pair(r)`` returns φ(r) in eV and dφ/dr in eV/Å for an array of
    distances in Å. The file holds n equally spaced distances from
    ``r_min`` to ``r_max``, each with φ and the force −dφ/dr, under
    ``keyword``, for ``pair_style table linear n``.
    """
    r = np.linspace(r_min, r_max, n)
    phi, dphi = pair(r)
    with open(path, "w") as out:
        out.write(f"# pair potential written by mdlab\n\n{keyword}\n")
        out.write(f"N {n} R {r_min:.10g} {r_max:.10g}\n\n")
        for k, (x, e, f) in enumerate(zip(r, phi, -dphi, strict=True), 1):
            out.write(f"{k} {x:.12g} {e:.12e} {f:.12e}\n")


def read_run(path_or_images, index: str = ":") -> dict:
    """A trajectory that ASE can read, as arrays for ``health_report``.

    Returns "positions" (frames, N, 3) in Å, "cell" (lattice vectors as
    columns, of the first frame), "symbols", "masses" in amu, and, when
    the file holds them, "velocities" in Å/fs, "kinetic" in eV and
    "potential" in eV.
    """
    from ase.io import read

    from mdlab.units import MV2_TO_EV

    if isinstance(path_or_images, (str, bytes)) or hasattr(
            path_or_images, "__fspath__"):
        images = read(path_or_images, index=index)
    else:
        images = list(path_or_images)
    images = images if isinstance(images, list) else [images]
    first = images[0]
    run = {
        "positions": np.array([a.get_positions() for a in images]),
        "cell": np.array(first.get_cell()).T,
        "symbols": first.get_chemical_symbols(),
        "masses": first.get_masses(),
    }
    if first.has("momenta"):
        v = np.array([a.get_velocities() for a in images]) * np.sqrt(
            FORCE_TO_ACCEL)
        run["velocities"] = v
        run["kinetic"] = 0.5 * MV2_TO_EV * np.einsum(
            "i,tij->t", run["masses"], v * v)
    if first.calc is not None and "energy" in first.calc.results:
        run["potential"] = np.array(
            [a.get_potential_energy() for a in images])
    return run
