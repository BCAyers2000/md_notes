"""The stress of LiC₆ along its NPT runs, with MACE-MP-0 and D3.

runs_mace.py's npt_aniso and npt_iso keep positions every 100 fs; every
second frame after the first 5 ps (76 of each run's frames), with the cell
of that moment, is given to the model in float64 on the CPU for its
stress, the potential part, to which the kinetic part 2K/3V on the
diagonal is added from the run's kinetic energy at that frame.

Writes data/ch17_practice/runs/stress_<run>.npz with "times" (fs) and
"pressure" (frames, 3), the diagonal of the pressure tensor in GPa (xx,
yy, zz). Prints the means and standard errors.
"""

import numpy as np
from ase import Atoms
from ch17 import GPA, RUNS, mace, threads
from runs_mace import load

from mdlab.analysis import stats

if __name__ == "__main__":
    threads(4)
    calc = mace(dtype="float64")
    symbols = load("lic6").repeat((3, 3, 1)).get_chemical_symbols()
    for name in ("npt_aniso", "npt_iso"):
        run = np.load(RUNS / f"{name}.npz")
        frame_times = run["times"][::10]  # frames were kept every 100 fs
        keep = np.flatnonzero(frame_times > 5000)[::2]
        pressure, times = [], []
        for k in keep:
            cell = run["cell"][10 * k]
            atoms = Atoms(symbols, positions=run["frames"][k], cell=cell,
                          pbc=True)
            atoms.calc = calc
            volume = atoms.get_volume()
            kinetic = run["kinetic"][10 * k]
            stress = atoms.get_stress()[:3]  # xx, yy, zz in eV/Å³
            pressure.append((-stress + 2 * kinetic / (3 * volume)) * GPA)
            times.append(frame_times[k])
        pressure = np.array(pressure)
        np.savez(RUNS / f"stress_{name}.npz", times=np.array(times),
                 pressure=pressure)
        for axis, label in enumerate("xyz"):
            mean, err, _ = stats.standard_error(pressure[:, axis])
            print(f"{name}: P_{label}{label} = {mean:.3f} ± {err:.3f} GPa "
                  f"over {len(pressure)} frames")
