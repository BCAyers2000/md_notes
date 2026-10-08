"""One force kernel written four ways: Chapter 10.

The Lennard-Jones energy and forces over a list of pairs in a rectangular
box, with the energy shifted to zero at the cutoff, each version giving
the same numbers to round-off (Section 10.7):

- ``lj_loop``: a plain Python loop over the pairs, the slowest;
- ``lj_numpy``: the same arithmetic on whole arrays at once;
- ``lj_numba``: the plain loop compiled by Numba on first use;
- ``lj_fortran``: the loop of ``fortran/pair_kernels.f90``, compiled by
  f2py on first use into ``mdlab/build/``.

Each takes positions (N, 3), the box lengths (3,), the pair indices i and
j, and ε, σ and r_c, and returns the energy and the forces.

Units
-----
Any consistent set, as for ``potentials.lennard_jones``.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

SOURCE = Path(__file__).with_name("fortran") / "pair_kernels.f90"
BUILD = Path(__file__).resolve().parents[2] / "build" / "fortran"


def lj_loop(
    positions: ArrayLike,
    box: ArrayLike,
    pair_i: ArrayLike,
    pair_j: ArrayLike,
    epsilon: float,
    sigma: float,
    r_cut: float,
) -> tuple[float, NDArray]:
    """The pair sum as a plain loop, one pair at a time."""
    r = np.asarray(positions, dtype=float)
    lengths = np.asarray(box, dtype=float)
    forces = np.zeros_like(r)
    s6 = (sigma / r_cut) ** 6
    phi_cut = 4 * epsilon * (s6 * s6 - s6)
    energy = 0.0
    for i, j in zip(pair_i, pair_j, strict=True):
        sep = r[j] - r[i]
        sep -= lengths * np.round(sep / lengths)
        r2 = sep @ sep
        if r2 >= r_cut * r_cut:
            continue
        s6 = (sigma * sigma / r2) ** 3
        push = 24 * epsilon * (2 * s6 * s6 - s6) / r2
        energy += 4 * epsilon * (s6 * s6 - s6) - phi_cut
        forces[i] -= push * sep
        forces[j] += push * sep
    return energy, forces


def lj_numpy(
    positions: ArrayLike,
    box: ArrayLike,
    pair_i: ArrayLike,
    pair_j: ArrayLike,
    epsilon: float,
    sigma: float,
    r_cut: float,
) -> tuple[float, NDArray]:
    """The pair sum on whole arrays: every pair in each operation."""
    r = np.asarray(positions, dtype=float)
    lengths = np.asarray(box, dtype=float)
    i, j = np.asarray(pair_i), np.asarray(pair_j)
    sep = r[j] - r[i]
    sep -= lengths * np.round(sep / lengths)
    r2 = np.einsum("px,px->p", sep, sep)
    inside = r2 < r_cut * r_cut
    sep, r2, i, j = sep[inside], r2[inside], i[inside], j[inside]
    s6 = (sigma * sigma / r2) ** 3
    s6_cut = (sigma / r_cut) ** 6
    energy = np.sum(4 * epsilon * (s6 * s6 - s6)) - len(r2) * 4 * epsilon * (
        s6_cut * s6_cut - s6_cut
    )
    push = (24 * epsilon * (2 * s6 * s6 - s6) / r2)[:, None] * sep
    forces = np.zeros_like(r)
    np.add.at(forces, i, -push)
    np.add.at(forces, j, push)
    return float(energy), forces


_compiled: dict[str, object] = {}


def lj_numba(
    positions: ArrayLike,
    box: ArrayLike,
    pair_i: ArrayLike,
    pair_j: ArrayLike,
    epsilon: float,
    sigma: float,
    r_cut: float,
) -> tuple[float, NDArray]:
    """The plain loop, compiled to machine code by Numba on first call."""
    if "numba" not in _compiled:
        import numba

        _compiled["numba"] = numba.njit(cache=False)(_loop_body)
    energy, forces = _compiled["numba"](
        np.ascontiguousarray(positions, dtype=float),
        np.asarray(box, dtype=float),
        np.asarray(pair_i, dtype=np.int64),
        np.asarray(pair_j, dtype=np.int64),
        float(epsilon),
        float(sigma),
        float(r_cut),
    )
    return float(energy), forces


def _loop_body(r, lengths, pair_i, pair_j, epsilon, sigma, r_cut):
    forces = np.zeros_like(r)
    s6 = (sigma / r_cut) ** 6
    phi_cut = 4 * epsilon * (s6 * s6 - s6)
    energy = 0.0
    sep = np.empty(3)
    for p in range(len(pair_i)):
        i, j = pair_i[p], pair_j[p]
        r2 = 0.0
        for a in range(3):
            sep[a] = r[j, a] - r[i, a]
            sep[a] -= lengths[a] * np.round(sep[a] / lengths[a])
            r2 += sep[a] * sep[a]
        if r2 >= r_cut * r_cut:
            continue
        s6 = (sigma * sigma / r2) ** 3
        push = 24 * epsilon * (2 * s6 * s6 - s6) / r2
        energy += 4 * epsilon * (s6 * s6 - s6) - phi_cut
        for a in range(3):
            forces[i, a] -= push * sep[a]
            forces[j, a] += push * sep[a]
    return energy, forces


def build_fortran(force: bool = False) -> Path:
    """Compile ``fortran/pair_kernels.f90`` with f2py into ``BUILD``."""
    BUILD.mkdir(parents=True, exist_ok=True)
    built = list(BUILD.glob("pair_kernels_f*.so"))
    fresh = built and built[0].stat().st_mtime > SOURCE.stat().st_mtime
    if fresh and not force:
        return built[0]
    # f2py cannot evaluate the kind parameter dp; tell it dp is a double.
    (BUILD / ".f2py_f2cmap").write_text("dict(real=dict(dp='double'))\n")
    env = dict(os.environ)
    tools = str(Path(sys.executable).parent)  # where meson and ninja live
    env["PATH"] = tools + os.pathsep + env.get("PATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            str(SOURCE),
            "-m",
            "pair_kernels_f",
        ],
        cwd=BUILD,
        env=env,
        check=True,
        capture_output=True,
    )
    return next(BUILD.glob("pair_kernels_f*.so"))


def lj_fortran(
    positions: ArrayLike,
    box: ArrayLike,
    pair_i: ArrayLike,
    pair_j: ArrayLike,
    epsilon: float,
    sigma: float,
    r_cut: float,
) -> tuple[float, NDArray]:
    """The loop of the Fortran kernel, compiled by f2py on first call."""
    if "fortran" not in _compiled:
        build_fortran()
        if str(BUILD) not in sys.path:
            sys.path.insert(0, str(BUILD))
        module = importlib.import_module("pair_kernels_f")
        kernel = module.pair_kernels.lj_energy_forces
        if "array('d')" not in kernel.__doc__:
            raise RuntimeError("the Fortran kernel was not built for doubles")
        _compiled["fortran"] = kernel
    r = np.asarray(positions, dtype=float)
    energy, forces = _compiled["fortran"](
        r.T,
        np.asarray(box, dtype=float),
        np.asarray(pair_i, dtype=np.int32),
        np.asarray(pair_j, dtype=np.int32),
        epsilon,
        sigma,
        r_cut,
    )
    return float(energy), np.asarray(forces).T
