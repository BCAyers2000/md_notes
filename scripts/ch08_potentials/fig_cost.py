"""Figure fig:pe-cost: what computing the energy costs.

(a) The time NumPy takes to find every eigenvalue and eigenvector of a
dense symmetric matrix of size n, the step at the heart of computing the
electrons' energy (Section 8.1), against n on logarithmic axes. (b) The
time `pair_energy_forces` takes for N atoms interacting in pairs, every
pair computed, against N. Each time is the shortest of several repeats;
the times depend on the computer, the slopes do not.

Prints the fitted slopes quoted in Section 8.1. The linear algebra runs
on one thread, so that the slope measures the work, not how it is shared
among the cores.
"""

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import time  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdlab import potentials, viz  # noqa: E402
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, figure_path  # noqa: E402

RNG = np.random.default_rng(8)


def best_time(task, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        task()
        times.append(time.perf_counter() - start)
    return min(times)


sizes = np.array([150, 300, 600, 1200, 2400])
eig_times = []
for n in sizes:
    a = RNG.normal(size=(n, n))
    a = a + a.T
    eig_times.append(best_time(lambda a=a: np.linalg.eigh(a), 3))
eig_times = np.array(eig_times)
slope_eig = np.polyfit(np.log(sizes[2:]), np.log(eig_times[2:]), 1)[0]

atoms = np.array([250, 500, 1000, 2000, 4000])
pair_times = []
for n in atoms:
    r = RNG.uniform(0, (n / 0.8) ** (1 / 3), size=(n, 3))
    pair_times.append(
        best_time(
            lambda r=r: potentials.pair_energy_forces(
                r, potentials.lennard_jones
            ),
            3,
        )
    )
pair_times = np.array(pair_times)
slope_pair = np.polyfit(np.log(atoms[2:]), np.log(pair_times[2:]), 1)[0]
for n, t in zip(sizes, eig_times, strict=True):
    print(f"eigh n = {n}: {t:.4f} s")
for n, t in zip(atoms, pair_times, strict=True):
    print(f"pairs N = {n}: {t:.4f} s")
print(f"slope of the eigenproblem (largest three sizes) {slope_eig:.2f}")
print(f"slope of the pair sum (largest three sizes) {slope_pair:.2f}")
print(
    f"doubling n multiplies the eigen time by "
    f"{eig_times[-1] / eig_times[-2]:.1f}; doubling N the pair time by "
    f"{pair_times[-1] / pair_times[-2]:.1f}"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.5), gridspec_kw=dict(wspace=0.38)
)
left.loglog(sizes, eig_times, "o", color=ACCENT, ms=4)
left.loglog(sizes, eig_times[-1] * (sizes / sizes[-1]) ** 3, **REFERENCE_STYLE)
left.set_xlabel(r"matrix size $n$")
left.set_ylabel(r"time / s")
left.text(
    800,
    eig_times[-1] * (800 / sizes[-1]) ** 3 / 8,
    r"$\propto n^3$",
    color="0.4",
)
viz.panel_tag(left, "a")
right.loglog(atoms, pair_times, "o", color=OCHRE, ms=4)
right.loglog(
    atoms, pair_times[-1] * (atoms / atoms[-1]) ** 2, **REFERENCE_STYLE
)
right.set_xlabel(r"atoms $N$")
right.set_ylabel(r"time / s")
right.text(
    1300,
    pair_times[-1] * (1300 / atoms[-1]) ** 2 / 6,
    r"$\propto N^2$",
    color="0.4",
)
viz.panel_tag(right, "b")
for ax, ticks in ((left, sizes), (right, atoms)):
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

print("wrote", viz.save(fig, figure_path("ch08_potentials", "cost.pdf")))
