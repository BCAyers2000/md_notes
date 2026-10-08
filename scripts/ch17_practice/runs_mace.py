"""Lithium in graphite with MACE-MP-0 and D3, run by ASE.

Starts from the relaxed structures of relax.py. Jobs:

- nve_<dt>_<dtype>: LiC₆, 2×2×2 cells (56 atoms), velocities drawn at
  600 K and the drift removed, 2 ps at fixed energy with steps of 0.5,
  1, 2 and 3 fs in float64 on the CPU and of 1 and 2 fs in float32 on
  the GPU; every step's energies kept. Started from the relaxed cell, half
  the kinetic energy passes into the potential and the runs settle near
  250 K.
- nve_600_<dt>_float64: the same with the velocities drawn at 1200 K and
  scaled to that temperature exactly, settling near 600 K, with steps of
  0.5, 1 and 2 fs; nve_600_1_seed2_float64 repeats 1 fs from other
  velocities.
- npt_aniso and npt_iso: LiC₆, 3×3×1 cells (63 atoms), 300 K and zero
  pressure, ASE's MaskedMTKNPT with a, b and c free (τ_T = 100 fs, τ_P =
  1 ps) for 20 ps, and IsotropicMTKNPT for 10 ps; cell, U and K every
  10 fs, positions every 100 fs.
- npt_long: npt_aniso continued from its last state for 80 ps in steps
  of 2 fs; cell, U and K every 10 fs, positions every 100 fs.
- dilute_<T>: one lithium in 64 carbon atoms, the relaxed cell, CSVR
  (ASE's Bussi) with τ_T = 1 ps at 300, 450 and 600 K, 20 ps each;
  lithium's position every 10 fs, all positions and U and K every 100 fs.
- train_<name>: 5 ps at fixed energy from the end of dilute_600 and of
  npt_aniso (at its mean cell), every 10th frame written with its forces
  and energy as extended XYZ to data/ch17_practice/training/, for
  Chapter 28.
- aims_frames: the 28-atom LiC₆ cell (2×2×1) under CSVR at 300 and 600 K
  for 2 ps each; ten frames 100 fs apart from the last 1 ps of each,
  for FHI-aims.

Steps of 1 fs unless a job says otherwise, float32 on the GPU unless it
says otherwise. Writes data/ch17_practice/runs/<job>.npz with "wall",
the seconds per step. Run with a list of jobs, or none for all.
"""

import sys
import time

import numpy as np
from ch17 import ASE_FS, DATA, RUNS, mace, threads

STRUCTURES = DATA / "structures"
#: Where float32 runs go: 'mps' (the GPU), or 'cpu' with --device=cpu.
DEVICE = next((a.split("=")[1] for a in sys.argv[1:]
               if a.startswith("--device=")), "mps")
TRAINING = DATA / "training"


def load(name):
    from ase.io import read
    return read(STRUCTURES / f"{name}.extxyz")


def start(atoms, temperature, seed, exact=False):
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                 rng=np.random.default_rng(seed),
                                 force_temp=exact)
    Stationary(atoms)


def record(dyn, atoms, every, positions_every=None, li=False):
    """Attach a recorder; returns the dict it fills."""
    out = {"times": [], "potential": [], "kinetic": [], "cell": []}
    if positions_every:
        out["frames"] = []
    if li:
        out["li"] = []
    count = {"k": 0}

    def keep():
        k = count["k"]
        out["times"].append(dyn.get_time() * ASE_FS)
        out["potential"].append(atoms.get_potential_energy())
        out["kinetic"].append(atoms.get_kinetic_energy())
        out["cell"].append(np.array(atoms.get_cell()))
        if li:
            out["li"].append(atoms.positions[-1].copy())
        if positions_every and k % (positions_every // every) == 0:
            out["frames"].append(atoms.get_positions())
        count["k"] += 1

    dyn.attach(keep, interval=every)
    return out


def finish(out, wall):
    return {k: np.array(v) for k, v in out.items()} | {"wall": wall}


def nve(dt, dtype, drawn=600, exact=False, seed=1):
    from ase.md.verlet import VelocityVerlet
    atoms = load("lic6").repeat((2, 2, 2))
    device = "cpu" if dtype == "float64" else DEVICE
    atoms.calc = mace(dtype=dtype, device=device)
    start(atoms, drawn, seed, exact)  # from the relaxed cell: half is lost
    dyn = VelocityVerlet(atoms, timestep=dt / ASE_FS)
    out = record(dyn, atoms, 1)
    n = int(round(2000 / dt))
    t0 = time.perf_counter()
    dyn.run(n)
    return finish(out, (time.perf_counter() - t0) / n)


def npt(kind):
    from ase.md.nose_hoover_chain import IsotropicMTKNPT, MaskedMTKNPT
    atoms = load("lic6").repeat((3, 3, 1))
    atoms.calc = mace(device=DEVICE)
    start(atoms, 300, 2)
    common = dict(timestep=1.0 / ASE_FS, temperature_K=300.0,
                  pressure_au=0.0, tdamp=100 / ASE_FS, pdamp=1000 / ASE_FS)
    if kind == "aniso":
        dyn, n = MaskedMTKNPT(atoms, mask=(True, True, True), **common), 20000
    else:
        dyn, n = IsotropicMTKNPT(atoms, **common), 10000
    out = record(dyn, atoms, 10, positions_every=100)
    t0 = time.perf_counter()
    dyn.run(n)
    result = finish(out, (time.perf_counter() - t0) / n)
    if kind == "aniso":
        from ase.io import write
        write(RUNS / "npt_aniso_last.extxyz", atoms)
    return result


def npt_long():
    """npt_aniso continued from its last state for 80 ps in steps of 2 fs.

    Two femtoseconds is the longest step that the test at fixed energy
    passed.
    """
    from ase.io import read
    from ase.md.nose_hoover_chain import MaskedMTKNPT
    atoms = read(RUNS / "npt_aniso_last.extxyz")
    atoms.calc = mace(device=DEVICE)
    dyn = MaskedMTKNPT(atoms, mask=(True, True, True), timestep=2.0 / ASE_FS,
                       temperature_K=300.0, pressure_au=0.0,
                       tdamp=100 / ASE_FS, pdamp=1000 / ASE_FS)
    out = record(dyn, atoms, 5, positions_every=50)
    n = 40000
    t0 = time.perf_counter()
    dyn.run(n)
    return finish(out, (time.perf_counter() - t0) / n)


def dilute(temperature):
    from ase.md.bussi import Bussi
    atoms = load("dilute")
    atoms.calc = mace(device=DEVICE)
    start(atoms, temperature, 3)
    dyn = Bussi(atoms, timestep=1.0 / ASE_FS, temperature_K=temperature,
                taut=1000 / ASE_FS, rng=np.random.default_rng(4))
    out = record(dyn, atoms, 10, positions_every=100, li=True)
    n = 20000
    t0 = time.perf_counter()
    dyn.run(n)
    result = finish(out, (time.perf_counter() - t0) / n)
    from ase.io import write
    write(RUNS / f"dilute_{temperature}_last.extxyz", atoms)
    return result


def train(name):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
    from ase.md.verlet import VelocityVerlet
    source = {"dilute": RUNS / "dilute_600_last.extxyz",
              "lic6": RUNS / "npt_aniso_last.extxyz"}[name]
    atoms = read(source)
    if name == "lic6":
        run = np.load(RUNS / "npt_aniso.npz")
        late = run["times"] > 10000
        atoms.set_cell(run["cell"][late].mean(axis=0), scale_atoms=True)
    atoms.calc = mace(device=DEVICE)
    dyn = VelocityVerlet(atoms, timestep=1.0 / ASE_FS)
    TRAINING.mkdir(parents=True, exist_ok=True)
    frames = []

    def keep():
        frame = atoms.copy()
        frame.calc = SinglePointCalculator(
            frame, energy=atoms.get_potential_energy(),
            forces=atoms.get_forces())
        frames.append(frame)

    dyn.attach(keep, interval=10)
    out = record(dyn, atoms, 10)
    t0 = time.perf_counter()
    dyn.run(5000)
    write(TRAINING / f"{name}_nve.extxyz", frames)
    return finish(out, (time.perf_counter() - t0) / 5000)


def aims_frames():
    from ase.io import write
    from ase.md.bussi import Bussi
    frames, wall = [], []
    for temperature, seed in ((300, 5), (600, 6)):
        atoms = load("lic6").repeat((2, 2, 1))
        atoms.calc = mace(device=DEVICE)
        start(atoms, temperature, seed)
        dyn = Bussi(atoms, timestep=1.0 / ASE_FS, temperature_K=temperature,
                    taut=100 / ASE_FS, rng=np.random.default_rng(seed))
        t0 = time.perf_counter()
        dyn.run(1000)
        for _ in range(10):
            dyn.run(100)
            frames.append(atoms.copy())
        wall.append((time.perf_counter() - t0) / 2000)
    write(RUNS / "aims_frames.extxyz", frames)
    return {"temperatures": np.repeat([300, 600], 10),
            "wall": np.mean(wall)}


JOBS = {**{f"nve_{dt:g}_float64": (nve, (dt, "float64"))
           for dt in (0.5, 1.0, 2.0, 3.0)},
        **{f"nve_{dt:g}_float32": (nve, (dt, "float32"))
           for dt in (1.0, 2.0)},
        **{f"nve_600_{dt:g}_float64": (nve, (dt, "float64", 1200, True))
           for dt in (0.5, 1.0, 2.0)},
        "nve_600_1_seed2_float64": (nve, (1.0, "float64", 1200, True, 2)),
        "npt_aniso": (npt, ("aniso",)), "npt_iso": (npt, ("iso",)),
        "npt_long": (npt_long, ()),
        **{f"dilute_{t}": (dilute, (t,)) for t in (300, 450, 600)},
        "train_dilute": (train, ("dilute",)),
        "train_lic6": (train, ("lic6",)),
        "aims_frames": (aims_frames, ())}

if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    threads(int(next((a.split("=")[1] for a in sys.argv[1:]
                      if a.startswith("--threads=")), 5)))
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(JOBS)
    for name in names:
        path = RUNS / f"{name}.npz"
        if path.exists():
            print(f"{name} cached", flush=True)
            continue
        function, args = JOBS[name]
        out = function(*args)
        np.savez_compressed(path, **out)
        print(f"{name}: {out['wall'] * 1000:.1f} ms per step", flush=True)
