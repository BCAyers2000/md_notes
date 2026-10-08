"""Figure fig:pe-madelung: a sum of charges that depends on its order.

The Madelung sum of rock salt, the energy of one ion with all the others
in units of k q²/d with d the nearest-neighbour distance, taken over the
ions inside a growing cube (with the ions on its surface counted in the
fraction that lies inside, dots) and inside a growing sphere (line),
against the half-side or radius in units of d.

Prints the numbers of Section 8.8.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import potentials, viz
from mdlab.viz import ACCENT, OCHRE, THRESHOLD_STYLE, figure_path

cubes = np.arange(1, 13)
cube_sums = np.array(
    [potentials.madelung_partial_sum(m, "cube") for m in cubes]
)
radii = np.linspace(1.0, 12.0, 881)
sphere_sums = np.array(
    [potentials.madelung_partial_sum(r, "sphere") for r in radii]
)
for m, s in zip(cubes, cube_sums, strict=True):
    print(f"cube of half-side {m}: {s:.6f}")
late = sphere_sums[radii >= 6]
print(
    f"spheres between radius 6 and 12 range from {late.min():.2f} to "
    f"{late.max():.2f}"
)
first = potentials.madelung_partial_sum(1.0, "sphere")
print(
    f"sphere of radius 1 (the six nearest): {first:.1f}; radius sqrt2: "
    f"{potentials.madelung_partial_sum(2**0.5, 'sphere'):.3f}"
)
shell = [(r, int(round(r * r))) for r in (1, 2**0.5, 3**0.5, 2.0)]
for r, r2 in shell:
    n = 0
    m = int(np.ceil(r))
    for a in range(-m, m + 1):
        for b in range(-m, m + 1):
            for c in range(-m, m + 1):
                n += a * a + b * b + c * c == r2
    print(f"shell at distance sqrt({r2}): {n} ions")

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.FULL, 2.4))
ax.plot(radii, sphere_sums, color=OCHRE, lw=0.8, label="spheres")
ax.plot(cubes, cube_sums, "o", color=ACCENT, ms=4, label="cubes")
ax.axhline(1.747565, **THRESHOLD_STYLE)
ax.set_xlim(1, 12)
ax.set_ylim(-12, 16)
ax.set_xlabel(r"half-side or radius / $d$")
ax.set_ylabel(r"partial sum")
ax.legend(loc="lower left", fontsize=7, ncol=2)

print("wrote", viz.save(fig, figure_path("ch08_potentials", "madelung.pdf")))
