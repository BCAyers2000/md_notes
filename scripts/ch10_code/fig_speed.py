"""Figure fig:md-speed: one force loop written four ways.

Lennard-Jones atoms (σ = ε = 1, r_c = 2.5σ, energy shifted at r_c) on a
slightly disordered simple-cubic arrangement at ρ = 0.8σ⁻³, with the pairs
within r_c + 0.3σ from a cell list. The time of one call of each kernel
of ``mdlab.kernels`` against N, best of five, on one thread: a plain
Python loop, NumPy, Numba and Fortran through f2py. The profile of a
step of ``md.run`` is in numbers_chapter.py.

Prints the numbers of Section 10.7.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdlab import kernels, neighbours, viz  # noqa: E402
from mdlab.viz import (  # noqa: E402
    ACCENT,
    OCHRE,
    OXBLOOD,
    REFERENCE_STYLE,
    figure_path,
    tint,
)

RHO, R_CUT, SKIN = 0.8, 2.5, 0.3


def system(n, seed=15):
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n ** (1 / 3)))
    grid = np.array(
        [[a, b, c] for a in range(k) for b in range(k) for c in range(k)],
        float,
    )[:n]
    side = (n / RHO) ** (1 / 3)
    r = grid / k * side + rng.normal(scale=0.05, size=(n, 3))
    return r, side


def best_time(function, args, repeats=5):
    function(*args)  # compile or warm up first
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return min(times)


kernels.build_fortran()
names = ("lj_loop", "lj_numpy", "lj_numba", "lj_fortran")
sizes = np.array([500, 1000, 2000, 4000, 8000, 16000])
times = {name: [] for name in names}
for n in sizes:
    r, side = system(n)
    i, j, _, _ = neighbours.cell_list_pairs(r, side * np.eye(3), R_CUT + SKIN)
    args = (r, np.full(3, side), i, j, 1.0, 1.0, R_CUT)
    reference = kernels.lj_numpy(*args)
    if n == 2000:
        plain = kernels.lj_loop(*args)
        for name in names[1:]:
            e_k, f_k = getattr(kernels, name)(*args)
            print(
                f"{name} against the loop at N = 2000: energy "
                f"{abs(e_k - plain[0]) / abs(plain[0]):.1e} relative, "
                f"forces {np.abs(f_k - plain[1]).max():.1e} "
                f"(largest force {np.abs(plain[1]).max():.0f})"
            )
    for name in names:
        if name == "lj_loop" and n > 2000:
            continue
        energy, forces = getattr(kernels, name)(*args)
        assert abs(energy - reference[0]) < 1e-9 * abs(reference[0])
        assert np.abs(forces - reference[1]).max() < 1e-9
        times[name].append(best_time(getattr(kernels, name), args))
    line = ", ".join(
        f"{name[3:]} {1e3 * times[name][-1]:.2f} ms"
        for name in names
        if len(times[name]) == list(sizes).index(n) + 1
    )
    print(f"N = {n}, {len(i)} listed pairs: {line}")
for name in names:
    t = np.array(times[name])
    print(
        f"{name}: time per atom per call {1e6 * t[-1] / sizes[len(t) - 1]:.2f}"
        f" µs at N = {sizes[len(t) - 1]}"
    )
at = list(sizes).index(2000)
base = times["lj_numpy"][at]
print(
    "at N = 2000, against NumPy: loop "
    f"{times['lj_loop'][at] / base:.0f} times slower, Numba "
    f"{base / times['lj_numba'][at]:.1f} and Fortran "
    f"{base / times['lj_fortran'][at]:.1f} times faster"
)

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF, 2.6))
styles = {
    "lj_loop": (OXBLOOD, "Python loop"),
    "lj_numpy": (ACCENT, "NumPy"),
    "lj_numba": (OCHRE, "Numba"),
    "lj_fortran": (tint(ACCENT, 0.5), "Fortran"),
}
for name in names:
    t = np.array(times[name])
    colour, label = styles[name]
    ax.loglog(
        sizes[: len(t)], 1e3 * t, "o-", color=colour, ms=3, lw=0.8, label=label
    )
ax.loglog(
    sizes,
    1e3 * times["lj_numpy"][-1] * sizes / sizes[-1],
    **REFERENCE_STYLE,
    lw=0.8,
)
ax.set_xlabel(r"number of atoms $N$")
ax.set_ylabel("time per call / ms")
ax.legend(loc="lower right", fontsize=6.5)

print("wrote", viz.save(fig, figure_path("ch10_code", "speed.pdf")))
