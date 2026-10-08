"""LiC₆ with FHI-aims: the reference that MACE-MP-0 is checked against.

PBE with the 'light' species defaults of 2020, scalar-relativistic
atomic ZORA, Gaussian occupations of 0.05 eV, the 28-atom LiC₆ cell
(2×2×1) of runs_mace.py's aims_frames. Each rank runs one thread; the
stack limit is raised as FHI-aims requires. Jobs:

- kgrid_<n>: the first frame with k-point grids n×n×(2.5n), n = 2, 3, 4,
  6 and 8, for the grid
  at which forces settle;
- frames: the twenty frames at the chosen grid, energies and forces;
- aimd: 200 steps of 1 fs at fixed energy from the 300 K frame and its
  MACE velocities, FHI-aims's own molecular dynamics, with the density
  matrix extrapolated from step to step; the wall time of each step.

Writes data/ch17_practice/runs/aims_<job>.npz and keeps each folder of
FHI-aims files under data/ch17_practice/aims/.

Set AIMS_ROOT to an installation containing build/aims.x and
species_defaults/defaults_2020/light, or set AIMS_BINARY and AIMS_SPECIES
separately. MPIRUN names the MPI launcher (default: mpirun on PATH);
AIMS_RANKS sets the number of ranks (default: 6). These are checked only
when a new calculation is requested; cached results need no installation.
"""

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from ch17 import DATA, RUNS

FOLDER = DATA / "aims"
KGRID = (3, 3, 7)  # chosen from the kgrid jobs


def species_directory():
    """Find the requested light species defaults, including Li and C."""
    root = os.environ.get("AIMS_ROOT")
    species = os.environ.get("AIMS_SPECIES")
    if not species and root:
        species = (
            Path(root).expanduser() / "species_defaults/defaults_2020/light"
        )
    if not species:
        raise ValueError(
            "set AIMS_ROOT or AIMS_SPECIES to the light species defaults"
        )
    folder = Path(species).expanduser().resolve()
    for name in ("03_Li_default", "06_C_default"):
        if not (folder / name).is_file():
            raise ValueError(f"AIMS_SPECIES is missing {name}: {folder}")
    return folder


def configuration():
    """Validate local executables and return the settings for a new run."""
    root = os.environ.get("AIMS_ROOT")
    binary = os.environ.get("AIMS_BINARY")
    if not binary and root:
        binary = str(Path(root).expanduser() / "build/aims.x")
    if not binary:
        raise ValueError(
            "set AIMS_ROOT or AIMS_BINARY to your FHI-aims executable"
        )
    binary = shutil.which(str(Path(binary).expanduser()))
    if binary is None:
        raise ValueError(
            "AIMS_BINARY must name an executable file or a command on PATH"
        )
    launcher = os.environ.get("MPIRUN", "mpirun")
    mpirun = shutil.which(str(Path(launcher).expanduser()))
    if mpirun is None:
        raise ValueError(
            "set MPIRUN to an MPI launcher, or put mpirun on PATH"
        )
    try:
        ranks = int(os.environ.get("AIMS_RANKS", "6"))
    except ValueError as exc:
        raise ValueError("AIMS_RANKS must be a positive integer") from exc
    if ranks < 1:
        raise ValueError("AIMS_RANKS must be a positive integer")
    return (
        Path(binary).resolve(),
        species_directory(),
        Path(mpirun).resolve(),
        ranks,
    )


def control(kgrid, md_steps=None, *, species=None):
    """control.in: the settings above, with an MD block if asked."""
    species = species_directory() if species is None else Path(species)
    lines = f"""xc                 pbe
relativistic       atomic_zora scalar
occupation_type    gaussian 0.05
k_grid             {kgrid[0]} {kgrid[1]} {kgrid[2]}
compute_forces     .true.
sc_accuracy_rho    1E-5
sc_accuracy_etot   1E-6
sc_accuracy_forces 1E-4
mixer              pulay
n_max_pulay        8
charge_mix_param   0.2
"""
    if md_steps:
        lines += f"""MD_run             {md_steps * 0.001:.3f} NVE
MD_time_step       0.001
MD_clean_rotations .false.
wf_extrapolation   polynomial 3 1
output_level       MD_light
"""
    for name in ("03_Li_default", "06_C_default"):
        lines += (species / name).read_text()
    return lines


def run(atoms, folder, kgrid, md_steps=None, velocities=None):
    """Run FHI-aims in ``folder``; return its output and the wall time, s."""
    from ase.io import write

    binary, species, mpirun, ranks = configuration()
    folder.mkdir(parents=True, exist_ok=True)
    write(
        folder / "geometry.in",
        atoms,
        format="aims",
        velocities=velocities is not None,
    )
    (folder / "control.in").write_text(
        control(kgrid, md_steps, species=species)
    )
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    t0 = time.perf_counter()
    with open(folder / "aims.out", "w") as out:
        # Raise the stack in the child shell, as required on macOS.
        # Positional arguments preserve spaces and shell characters in paths.
        command = [
            "sh",
            "-c",
            'ulimit -s 65520 && exec "$@"',
            "aims-launch",
            str(mpirun),
            "-np",
            str(ranks),
            str(binary),
        ]
        subprocess.run(
            command,
            cwd=folder,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    return (folder / "aims.out").read_text(), time.perf_counter() - t0


def energy_forces(folder):
    from ase.io import read

    frames = read(folder / "aims.out", index=":")
    return (
        np.array([f.get_potential_energy() for f in frames]),
        np.array([f.get_forces() for f in frames]),
    )


def frames():
    from ase.io import read

    return read(RUNS / "aims_frames.extxyz", index=":")


def kgrid_job(n):
    k = (n, n, int(round(2.5 * n)))
    folder = FOLDER / f"kgrid_{n}"
    text, wall = run(frames()[0], folder, k)
    e, f = energy_forces(folder)
    cycles = int(
        re.findall(r"Number of self-consistency cycles\s*:\s*(\d+)", text)[-1]
    )
    return {
        "kgrid": np.array(k),
        "energy": e[-1],
        "forces": f[-1],
        "wall": wall,
        "cycles": cycles,
    }


def frames_job():
    energies, forces, walls = [], [], []
    for k, atoms in enumerate(frames()):
        folder = FOLDER / f"frame_{k:02d}"
        _, wall = run(atoms, folder, KGRID)
        e, f = energy_forces(folder)
        energies.append(e[-1])
        forces.append(f[-1])
        walls.append(wall)
    return {
        "energy": np.array(energies),
        "forces": np.array(forces),
        "wall": np.array(walls),
        "kgrid": np.array(KGRID),
    }


def aimd_job(steps=200):
    atoms = frames()[9]  # the last 300 K frame, with MACE's velocities
    folder = FOLDER / "aimd"
    text, wall = run(atoms, folder, KGRID, md_steps=steps, velocities=True)
    e, f = energy_forces(folder)
    from ase.io import read

    traj = read(folder / "aims.out", index=":")
    return {
        "energy": e,
        "forces": f,
        "positions": np.array([a.get_positions() for a in traj]),
        "kinetic": np.array([a.get_kinetic_energy() for a in traj]),
        "wall": wall,
        "steps": len(traj),
    }


JOBS = {
    **{f"kgrid_{n}": (kgrid_job, (n,)) for n in (2, 3, 4, 6, 8)},
    "frames": (frames_job, ()),
    "aimd": (aimd_job, ()),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jobs",
        nargs="*",
        metavar="JOB",
        help="one or more of: " + ", ".join(JOBS),
    )
    names = parser.parse_args().jobs or list(JOBS)
    for name in names:
        if name not in JOBS:
            parser.error(
                f"unknown job {name!r}; choose from {', '.join(JOBS)}"
            )
    if any(not (RUNS / f"aims_{name}.npz").exists() for name in names):
        try:
            configuration()
        except ValueError as exc:
            parser.error(str(exc))
    RUNS.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = RUNS / f"aims_{name}.npz"
        if path.exists():
            print(f"{name} cached", flush=True)
            continue
        function, args = JOBS[name]
        out = function(*args)
        np.savez_compressed(path, **out)
        print(f"{name}: {np.sum(out['wall']):.0f} s", flush=True)
