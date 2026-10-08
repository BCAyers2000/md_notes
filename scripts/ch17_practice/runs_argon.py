"""Liquid argon run by mdlab, by ASE and by LAMMPS with one model.

The switched Lennard-Jones model of Chapter 12 (256 atoms, 0.8σ⁻³) from
Chapter 13's equilibrated liquid: in ``mdlab`` directly; in ASE through
``io.mdlab_calculator``; in LAMMPS through a table of φ and −dφ/dr on
20 001 points (``io.lammps_table``). For each code:

- nve: 5 ps at fixed energy, steps of 10 fs, every step kept;
- langevin, csvr, nhc: 200 ps under Langevin friction of 1 ps⁻¹, CSVR
  (ASE's Bussi, LAMMPS's temp/csvr) with τ_T = 1 ps, and a Nosé-Hoover
  chain of three with τ_T = 1 ps (ASE's NoseHooverChainNVT, LAMMPS's
  fix nvt), at 135 K, positions and kinetic energies every 100 fs;
- langevin_long: 1 ns under the same friction, for mdlab and for LAMMPS,
  and langevin_zero, LAMMPS's with ``zero yes``, which removes the net
  random force at each step.

Writes data/ch17_practice/runs/argon_<code>_<run>.npz with "times" (fs),
"positions" (unwrapped, Å), "kinetic" and "potential" (eV), and "wall",
the seconds per step. Run with --workers=N to run jobs side by side.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ch17 import ASE_FS, RUNS, T_LIQUID, argon_model, ch12, liquid_start

from mdlab import io, md, statmech, thermostats, units

DT = 10.0  # fs
TAU = 1000.0  # fs, of CSVR and the chain
GAMMA = 1e-3  # 1/fs, Langevin's friction
EVERY = 10  # steps between kept frames under a thermostat
NVE_STEPS, LONG_STEPS = 500, 20000
TABLE_POINTS = 20001


def start():
    r, v, h, m = liquid_start(0)
    return r, v - np.average(v, axis=0, weights=m), h, m


# --------------------------------------------------------------- mdlab ---
def steps_of(kind):
    """Steps of a run, and steps between kept frames."""
    if kind == "nve":
        return NVE_STEPS, 1
    return (LONG_STEPS * 5 if kind.startswith("langevin_") else LONG_STEPS,
            EVERY)


def run_mdlab(kind):
    r, v, h, m = start()
    model = argon_model(h)
    nf = statmech.degrees_of_freedom(len(m))
    if kind == "nve":
        t0 = time.perf_counter()
        out = md.run(model, m, r, v, h, DT, NVE_STEPS, every=1)
        wall = (time.perf_counter() - t0) / NVE_STEPS
        return dict(times=out["times"], positions=out["positions"],
                    kinetic=out["kinetic"], potential=out["potential"],
                    wall=wall)
    th = {"langevin": thermostats.Langevin(T_LIQUID, GAMMA),
          "csvr": thermostats.CSVR(T_LIQUID, TAU, nf),
          "nhc": thermostats.NoseHooverChain(T_LIQUID, TAU, nf)}[
              kind.split("_")[0]]
    n, every = steps_of(kind)
    t0 = time.perf_counter()
    out = thermostats.run(model, m, r, v, DT, n, th,
                          rng=np.random.default_rng(17), every=every,
                          keep=("positions",))
    wall = (time.perf_counter() - t0) / n
    return dict(times=out["times"], positions=out["positions"],
                kinetic=out["kinetic"], potential=out["potential"], wall=wall)


# ----------------------------------------------------------------- ASE ---
def run_ase(kind):
    from ase import Atoms
    from ase.constraints import FixCom
    from ase.md.bussi import Bussi
    from ase.md.langevin import Langevin
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
    from ase.md.verlet import VelocityVerlet

    r, v, h, m = start()
    atoms = Atoms(["Ar"] * len(m), positions=r, cell=h.T, pbc=True,
                  masses=m)
    atoms.set_velocities(v / np.sqrt(units.FORCE_TO_ACCEL))
    atoms.calc = io.mdlab_calculator(argon_model(h))
    step = DT / ASE_FS
    if kind == "nve":
        dyn, n, every = VelocityVerlet(atoms, timestep=step), NVE_STEPS, 1
    else:
        n, every = LONG_STEPS, EVERY
        if kind == "langevin":
            # from ASE 3.28 the centre of mass is held by a constraint, as
            # fixcm=True does not sample the canonical ensemble exactly
            atoms.set_constraint(FixCom())
        dyn = {"langevin": lambda: Langevin(
                   atoms, timestep=step, temperature_K=T_LIQUID,
                   friction=GAMMA * ASE_FS, fixcm=False,
                   rng=np.random.default_rng(17)),
               "csvr": lambda: Bussi(
                   atoms, timestep=step, temperature_K=T_LIQUID,
                   taut=TAU / ASE_FS, rng=np.random.default_rng(17)),
               "nhc": lambda: NoseHooverChainNVT(
                   atoms, timestep=step, temperature_K=T_LIQUID,
                   tdamp=TAU / ASE_FS)}[kind]()
    record = {"times": [], "positions": [], "kinetic": [], "potential": []}

    def keep():
        record["times"].append(dyn.get_time() * ASE_FS)
        record["positions"].append(atoms.get_positions())
        record["kinetic"].append(atoms.get_kinetic_energy())
        record["potential"].append(atoms.get_potential_energy())

    dyn.attach(keep, interval=every)
    t0 = time.perf_counter()
    dyn.run(n)
    wall = (time.perf_counter() - t0) / n
    return {k: np.array(x) for k, x in record.items()} | {"wall": wall}


# -------------------------------------------------------------- LAMMPS ---
def run_lammps(kind):
    from ase import Atoms
    from ase.io import write
    from lammps import lammps

    r, v, h, m = start()
    folder = RUNS / "lammps"
    folder.mkdir(parents=True, exist_ok=True)
    table = folder / "argon.table"
    model = argon_model(h)
    io.lammps_table(model.pair, 0.5, ch12.R_CUT, TABLE_POINTS, str(table),
                    "ARGON")
    data = folder / f"argon_{kind}.data"
    atoms = Atoms(["Ar"] * len(m), positions=r, cell=h.T, pbc=True,
                  masses=m)
    atoms.set_velocities(v / np.sqrt(units.FORCE_TO_ACCEL))
    write(str(data), atoms, format="lammps-data", masses=True,
          velocities=True, units="metal")
    n, every = steps_of(kind)
    langevin = (f"fix 1 all nve\nfix 2 all langevin {T_LIQUID} {T_LIQUID} "
                f"{1 / GAMMA / 1000} 17 zero ")
    fix = {"nve": "fix 1 all nve",
           "langevin": langevin + "no",
           "langevin_long": langevin + "no",
           "langevin_zero": langevin + "yes",
           "csvr": f"fix 1 all nve\nfix 2 all temp/csvr {T_LIQUID} "
                   f"{T_LIQUID} {TAU / 1000} 17",
           "nhc": f"fix 1 all nvt temp {T_LIQUID} {T_LIQUID} "
                  f"{TAU / 1000}"}[kind]
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_string(f"""
units metal
atom_style atomic
atom_modify map array
read_data {data}
pair_style table linear {TABLE_POINTS}
pair_coeff 1 1 {table} ARGON {ch12.R_CUT}
timestep {DT / 1000}
{fix}
thermo_style custom step pe ke
""")
    record = {"times": [], "positions": [], "kinetic": [], "potential": []}
    t0 = time.perf_counter()
    for k in range(0, n + 1, every):
        lmp.command("run 0 post no" if k == 0 else f"run {every} pre no "
                    "post no")
        # LAMMPS re-sorts its atoms in memory every 1000 steps
        order = np.argsort(lmp.numpy.extract_atom("id")[:len(m)])
        x = lmp.numpy.extract_atom("x")[:len(m)][order]
        image = lmp.numpy.extract_atom("image")[:len(m)][order]
        # LAMMPS packs the three image counts into one integer
        ix = (image & 1023) - 512
        iy = ((image >> 10) & 1023) - 512
        iz = (image >> 20) - 512
        shifts = np.stack([ix, iy, iz], axis=1) @ h.T
        record["times"].append(k * DT)
        record["positions"].append(x + shifts)
        record["kinetic"].append(lmp.get_thermo("ke"))
        record["potential"].append(lmp.get_thermo("pe"))
    wall = (time.perf_counter() - t0) / n
    lmp.close()
    return {k: np.array(x) for k, x in record.items()} | {"wall": wall}


JOBS = [(code, kind) for code in ("mdlab", "ase", "lammps")
        for kind in ("nve", "langevin", "csvr", "nhc")] + [
    ("mdlab", "langevin_long"), ("lammps", "langevin_long"),
    ("lammps", "langevin_zero")]


def job(spec):
    code, kind = spec
    path = RUNS / f"argon_{code}_{kind}.npz"
    if path.exists():
        return f"{path.name} cached"
    out = {"mdlab": run_mdlab, "ase": run_ase, "lammps": run_lammps}[code](
        kind)
    np.savez_compressed(path, **out)
    return f"{path.name}: {out['wall'] * 1000:.2f} ms per step"


if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    workers = next((int(a.split("=")[1]) for a in sys.argv[1:]
                    if a.startswith("--workers=")), 1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(workers) as pool:
        for line in pool.map(job, JOBS):
            print(line, flush=True)
