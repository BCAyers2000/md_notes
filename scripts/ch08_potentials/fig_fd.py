"""Figure fig:pe-fd: how accurate a finite-difference force can be.

Twelve Lennard-Jones atoms near a small cubic arrangement, in reduced
units. For each step h, the forces from central differences of the energy
are compared with those of `pair_energy_forces`, and the largest
difference is plotted against h on logarithmic axes. For large h the
error falls as h², from the Taylor series; for small h it grows as 1/h,
from rounding the energies; it is least between.

Prints the numbers of Section 8.4.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import potentials, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, figure_path

RNG = np.random.default_rng(4)
grid = np.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
    dtype=float,
)[:12]
POSITIONS = 1.15 * grid + 0.08 * RNG.normal(size=grid.shape)


def energy(r):
    return potentials.pair_energy_forces(r, potentials.lennard_jones)[0]


u, forces, _ = potentials.pair_energy_forces(
    POSITIONS, potentials.lennard_jones
)
steps = np.logspace(-10, -1, 37)
errors = np.array(
    [
        np.abs(
            potentials.finite_difference_forces(energy, POSITIONS, h) - forces
        ).max()
        for h in steps
    ]
)
best = int(np.argmin(errors))
print(f"U = {u:.6f} eps, largest |F| = {np.abs(forces).max():.4f} eps/sigma")
print(f"machine epsilon {np.finfo(float).eps:.3e}")
for h, e in zip(steps[::6], errors[::6], strict=True):
    print(f"h = {h:.1e}: largest error {e:.2e}")
print(f"least error {errors[best]:.2e} at h = {steps[best]:.1e}")
big = steps > 1e-3
slope_big = np.polyfit(np.log(steps[big]), np.log(errors[big]), 1)[0]
small = steps < 1e-7
slope_small = np.polyfit(np.log(steps[small]), np.log(errors[small]), 1)[0]
print(f"slope for h > 1e-3: {slope_big:.2f}; for h < 1e-7: {slope_small:.2f}")
print(f"cube root of machine epsilon: {np.finfo(float).eps ** (1 / 3):.1e}")

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF, 2.4))
ax.loglog(steps, errors, "o", color=ACCENT, ms=3)
ax.loglog(
    steps[big],
    errors[big][-1] * (steps[big] / steps[big][-1]) ** 2,
    **REFERENCE_STYLE,
)
ax.loglog(
    steps[small],
    errors[small][0] * (steps[small] / steps[small][0]) ** -1,
    **REFERENCE_STYLE,
)
ax.text(2e-3, 3e-4, r"$\propto h^2$", color="0.4")
ax.text(1.5e-10, 1.5e-8, r"$\propto 1/h$", color="0.4")
ax.set_xlabel(r"step $h$ / $\sigma$")
ax.set_ylabel(r"largest error / $(\varepsilon/\sigma)$")

print("wrote", viz.save(fig, figure_path("ch08_potentials", "fd.pdf")))
