"""Figure fig:md-neighbours: the cost of finding the pairs within r_c.

N atoms placed at random in a cubic cell at the number density
ρ = 0.8σ⁻³ of a dense liquid, with r_c = 2.5σ: the time to find every
pair within r_c by testing all pairs and through a cell list, against N
on logarithmic axes, with lines of slope 2 and 1. Each time is the best
of three, on one thread.

Prints the numbers of Section 10.4.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdlab import neighbours, viz  # noqa: E402
from mdlab.viz import (  # noqa: E402
    ACCENT,
    OXBLOOD,
    REFERENCE_STYLE,
    figure_path,
)

RHO, R_CUT = 0.8, 2.5


def best_time(function, *args, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        times.append(time.perf_counter() - start)
    return min(times), result


rng = np.random.default_rng(13)
sizes_all = np.array([500, 1000, 2000, 4000])
sizes_cells = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000])
t_all, t_cells = [], []
for n in sizes_cells:
    side = (n / RHO) ** (1 / 3)
    h = side * np.eye(3)
    r = rng.uniform(0, side, size=(n, 3))
    t, pairs = best_time(neighbours.cell_list_pairs, r, h, R_CUT)
    t_cells.append(t)
    per_atom = len(pairs[0]) / n
    line = f"N = {n}: cell list {t:.3f} s, {per_atom:.1f} pairs per atom"
    if n in sizes_all:
        t, check = best_time(neighbours.all_pairs, r, h, R_CUT)
        t_all.append(t)
        assert len(check[0]) == len(pairs[0])
        line += f", all pairs {t:.3f} s"
    print(line)
t_all, t_cells = np.array(t_all), np.array(t_cells)
slope_all = np.polyfit(np.log(sizes_all), np.log(t_all), 1)[0]
slope_cells = np.polyfit(np.log(sizes_cells[2:]), np.log(t_cells[2:]), 1)[0]
print(
    f"slopes: all pairs {slope_all:.2f}, cell list {slope_cells:.2f} "
    f"(from N = 2000)"
)
print(
    f"at N = 4000 the cell list is {t_all[-1] / t_cells[3]:.0f} times faster"
)

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF, 2.5))
ax.loglog(
    sizes_all, t_all, "o-", color=OXBLOOD, ms=3.5, lw=0.8, label="all pairs"
)
ax.loglog(
    sizes_cells, t_cells, "o-", color=ACCENT, ms=3.5, lw=0.8, label="cell list"
)
ax.loglog(
    sizes_all,
    t_all[-1] * (sizes_all / sizes_all[-1]) ** 2,
    **REFERENCE_STYLE,
    lw=0.8,
)
ax.loglog(
    sizes_cells,
    t_cells[-1] * (sizes_cells / sizes_cells[-1]),
    **REFERENCE_STYLE,
    lw=0.8,
)
ax.set_xlabel(r"number of atoms $N$")
ax.set_ylabel("time / s")
ax.legend(loc="upper left", fontsize=7)

print("wrote", viz.save(fig, figure_path("ch10_code", "neighbours.pdf")))
