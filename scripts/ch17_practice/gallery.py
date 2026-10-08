"""Chapter 17's faults, and health_report on healthy and broken records.

Liquid argon (runs_argon.py) and LiC₆ (runs_mace.py):

- LAMMPS's arrays read in the order of the first frame, not by identity:
  the Langevin run of runs_argon.py repeated, every 100 fs, its positions
  kept both ways, and D from the MSD over 2 to 20 ps, five blocks after
  the first 10 ps;
- health_report on mdlab's run at fixed energy, on the same positions
  wrapped into the cell, on them with the centre of mass carried at
  √(3k_BT/Nm), on mdlab's CSVR run against its target, on its Langevin
  run with the centre-of-mass check and without it, on the LAMMPS
  record read in the wrong order, and on LiC₆ at fixed energy with steps
  of 1 and 3 fs.

Prints D both ways and each report.
"""

import os

import numpy as np
from ase import Atoms
from ase.io import write
from ch17 import RUNS, T_LIQUID, argon_model, ch12, liquid_start
from runs_argon import DT, GAMMA, TABLE_POINTS

from mdlab import diagnostics, io, statmech, units
from mdlab.analysis import transport
from mdlab.cell import wrap


def diffusion(times, positions, blocks=5):
    late = times > 10000
    lag = times[1] - times[0]
    n = int(round(20000 / lag))
    ds = []
    for block in np.array_split(positions[late], blocks):
        msd = transport.msd(block, n, remove_drift=True).sum(1)
        ds.append(transport.diffusion_coefficient(lag * np.arange(n + 1),
                                                  msd, 2000, 20000))
    ds = np.array(ds)
    return ds.mean(), ds.std(ddof=1) / np.sqrt(blocks)


def lammps_both_orders(steps=20000, every=10):
    """The Langevin run in LAMMPS, positions read by identity and not."""
    os.environ.setdefault("FI_PROVIDER", "tcp")
    from lammps import lammps
    r, v, h, m = liquid_start(0)
    v = v - np.average(v, axis=0, weights=m)
    folder = RUNS / "lammps"
    folder.mkdir(parents=True, exist_ok=True)
    table = folder / "argon.table"
    io.lammps_table(argon_model(h).pair, 0.5, ch12.R_CUT, TABLE_POINTS,
                    str(table), "ARGON")
    atoms = Atoms(["Ar"] * len(m), positions=r, cell=h.T, pbc=True,
                  masses=m)
    atoms.set_velocities(v / np.sqrt(units.FORCE_TO_ACCEL))
    data = folder / "argon_gallery.data"
    write(str(data), atoms, format="lammps-data", masses=True,
          velocities=True, units="metal")
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_string(f"""
units metal
atom_style atomic
atom_modify map array
read_data {data}
pair_style table linear {TABLE_POINTS}
pair_coeff 1 1 {table} ARGON {ch12.R_CUT}
timestep {DT / 1000}
fix 1 all nve
fix 2 all langevin {T_LIQUID} {T_LIQUID} {1 / GAMMA / 1000} 17 zero no
""")
    n = len(m)
    first, right, wrong = None, [], []
    for k in range(0, steps + 1, every):
        lmp.command("run 0 post no" if k == 0 else
                    f"run {every} pre no post no")
        ids = lmp.numpy.extract_atom("id")[:n]
        x = lmp.numpy.extract_atom("x")[:n]
        image = lmp.numpy.extract_atom("image")[:n]
        shifts = np.stack([(image & 1023) - 512, ((image >> 10) & 1023) - 512,
                           (image >> 20) - 512], axis=1) @ h.T
        first = np.argsort(ids) if first is None else first
        right.append((x + shifts)[np.argsort(ids)])
        wrong.append((x + shifts)[first])
    lmp.close()
    times = DT * np.arange(0, steps + 1, every)
    return times, np.array(right), np.array(wrong), h


def show(label, report):
    print(f"--- {label}")
    print(report)


if __name__ == "__main__":
    times, right, wrong, h = lammps_both_orders()
    d_right, e_right = diffusion(times, right)
    d_wrong, e_wrong = diffusion(times, wrong)
    print(f"LAMMPS Langevin: D read by identity ({d_right * 1e4:.3f} ± "
          f"{e_right * 1e4:.3f})e-4, read in the first frame's order "
          f"({d_wrong * 1e4:.2f} ± {e_wrong * 1e4:.2f})e-4 Å²/fs, "
          f"{d_wrong / d_right:.1f} times")
    np.savez(RUNS / "gallery_lammps.npz", times=times,
             msd_right=transport.msd(right, 1000).sum(1),
             msd_wrong=transport.msd(wrong, 1000).sum(1))

    r, v, h0, m = liquid_start(0)
    nve = np.load(RUNS / "argon_mdlab_nve.npz")
    symbols = ["Ar"] * len(m)
    show("argon at fixed energy, as recorded",
         diagnostics.health_report(nve["positions"], h0, m,
                                   times=nve["times"],
                                   potential=nve["potential"],
                                   kinetic=nve["kinetic"], symbols=symbols))
    wrapped = np.array([wrap(x, h0) for x in nve["positions"][::10]])
    show("the same, wrapped into the cell every 100 fs",
         diagnostics.health_report(wrapped, h0, m))
    kt_m = units.KB * T_LIQUID / (39.948 * units.MV2_TO_EV)
    drift = np.sqrt(3 * kt_m / len(m))
    carried = nve["positions"] + drift * nve["times"][:, None, None] * \
        np.array([1.0, 0, 0])
    show("the same, the centre of mass carried at √(3kT/Nm)",
         diagnostics.health_report(carried, h0, m))
    csvr = np.load(RUNS / "argon_mdlab_csvr.npz")
    late = csvr["times"] > 10000
    show("argon under CSVR, against 135 K",
         diagnostics.health_report(csvr["positions"][late], h0, m,
                                   kinetic=csvr["kinetic"][late],
                                   temperature=T_LIQUID, ensemble="nvt",
                                   n_free=statmech.degrees_of_freedom(
                                       len(m))))
    langevin = np.load(RUNS / "argon_mdlab_langevin.npz")
    late = langevin["times"] > 10000
    for kept in (True, False):
        show("argon under Langevin friction, "
             + ("as if the momentum were kept" if kept
                else "the momentum not kept"),
             diagnostics.health_report(langevin["positions"][late], h0, m,
                                       momentum_kept=kept))
    show("LAMMPS's record read in the first frame's order",
         diagnostics.health_report(wrong, h, m))
    for dt in (1.0, 3.0):
        run = np.load(RUNS / f"nve_{dt:g}_float64.npz")
        # positions were not kept, so only the energy is checked
        show(f"LiC6 at fixed energy, steps of {dt:g} fs",
             diagnostics.health_report(None, None, times=run["times"],
                                       potential=run["potential"],
                                       kinetic=run["kinetic"]))
