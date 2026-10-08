"""What first-principles forces cost, from FHI-aims single points.

runs_aims.py's kgrid_2, kgrid_3 and kgrid_4: the first of the 28-atom
LiC₆ frames, PBE with the light defaults, on four MPI ranks of an M1 Max,
with k-point grids 2×2×5, 3×3×8 and 4×4×10. Prints the wall time, the
cycles of the self-consistent field and the time per cycle of each, how
far the forces of the two coarser grids lie from those of the finest, and
the time a picosecond of molecular dynamics would take at the finest grid
if each step needed half the time of a start from scratch, against
MACE-MP-0's time per step for the same cell (runs_mace.py's aims_frames,
on the GPU of the same machine, which other runs shared), and its time
per call alone (timing_mace.py, on the GPU of the faster machine of
fig:ip-learned). Also prints each grid's time per ordinary cycle, the two last
cycles, which find the forces too, and how much of the finest grid's
time went on finding the eigenvectors.
"""

import re

import numpy as np
from ase.io import read
from ch17 import DATA, RUNS

frames = read(RUNS / "aims_frames.extxyz", index=":")
symbols = np.array(frames[0].get_chemical_symbols())
runs = {n: np.load(RUNS / f"aims_kgrid_{n}.npz") for n in (2, 3, 4)}
finest = runs[4]["forces"]
for m, run in runs.items():
    wall, cycles = float(run["wall"]), int(run["cycles"])
    text = (DATA / "aims" / f"kgrid_{m}" / "aims.out").read_text()
    points = re.findall(r"\| Number of k-points\s*:\s*(\d+)", text)[0]
    gap = np.linalg.norm(run["forces"] - finest, axis=1)
    # the wall-clock column of each cycle; the last two also find forces
    times = np.array([float(t) for t in re.findall(
        r"\| Time for this iteration\s*:\s*[\d.]+ s\s*([\d.]+) s", text)])
    print(f"k {'×'.join(str(int(k)) for k in run['kgrid'])} ({points} "
          f"points): {wall:.0f} s, {cycles} cycles, ordinary cycles "
          f"{times[:-2].mean():.1f} s each, the two with forces "
          f"{times[-2]:.0f} and {times[-1]:.0f} s, the rest "
          f"{wall - times.sum():.0f} s; forces from the finest grid: "
          f"Li {gap[symbols == 'Li'].max():.3f}, C "
          f"{gap[symbols == 'C'].max():.3f} eV/Å; largest force "
          f"{np.linalg.norm(run['forces'], axis=1).max():.2f} eV/Å")
per_cycle = float(runs[4]["wall"]) / int(runs[4]["cycles"])
step = per_cycle * int(runs[4]["cycles"]) / 2
print(f"a step at half the time from scratch: {step:.0f} s; a picosecond of "
      f"1 fs steps "
      f"{1000 * step / 3600:.0f} h on four ranks")
text = (DATA / "aims" / "kgrid_4" / "aims.out").read_text()
solve = float(re.findall(r"Total time for solution of K.-S. equations\s*:"
                         r"\s*[\d.]+ s\s*([\d.]+) s", text)[-1])
print(f"finding the eigenvectors: {solve:.0f} s of "
      f"{float(runs[4]['wall']):.0f}")
mace_step = float(np.load(RUNS / "aims_frames.npz")["wall"])
print(f"MACE-MP-0 with D3, a step of molecular dynamics of the same cell "
      f"on the same machine's GPU, other runs sharing the machine: "
      f"{mace_step * 1000:.0f} ms, {step / mace_step:.0f} times less")
timing = np.load(RUNS / "mace_timing.npz")
k = list(timing["atoms"]).index(28)
call = float(timing["medium_mps_float32"][k] + timing["d3"][k])
print(f"one call alone on the GPU of timing_mace.py's machine: "
      f"{call * 1000:.0f} ms for the same 28 atoms")
