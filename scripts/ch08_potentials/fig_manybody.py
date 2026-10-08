"""Figure fig:pe-manybody: bonds that weaken as they multiply.

The bond energy of an atom with z neighbours at the same distance, as a
fraction of its value with z = 12 (the close-packed metals), for a pair
potential, where it grows as z, and for the second-moment model of
Section 8.6, where it grows as √z. The marked structures are a pair of
atoms (z = 1), a chain (2), a graphite layer (3), diamond (4), simple
cubic (6), body-centred cubic, the structure of lithium (8), and
close-packed (12).

Prints the numbers of Section 8.6.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, figure_path

STRUCTURES = {
    1: "pair",
    2: "chain",
    3: "graphite layer",
    4: "diamond",
    6: "simple cubic",
    8: "bcc (lithium)",
    12: "close-packed",
}
z = np.array(sorted(STRUCTURES))
for zz in z:
    print(
        f"z = {zz:2d} ({STRUCTURES[zz]}): pair {zz / 12:.3f}, "
        f"second moment {math.sqrt(zz / 12):.3f}, per bond "
        f"{1 / math.sqrt(zz):.3f} of a lone bond"
    )
print(
    f"one bond of a close-packed atom is {1 / math.sqrt(12):.3f} of a "
    "lone bond in the second-moment model"
)

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.HALF, 2.4))
zs = np.linspace(0, 12.5, 200)
ax.plot(zs, zs / 12, label="pairs", **REFERENCE_STYLE)
ax.plot(zs, np.sqrt(zs / 12), color=ACCENT, label="second moment")
ax.plot(z, np.sqrt(z / 12), "o", color=ACCENT, ms=3.5)
ax.plot(z, z / 12, "o", color="0.45", ms=3)
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 1.08)
ax.set_xticks(z)
ax.set_xlabel(r"neighbours $z$")
ax.set_ylabel(r"bond energy / value at $z = 12$")
ax.legend(loc="lower right", fontsize=7)

print("wrote", viz.save(fig, figure_path("ch08_potentials", "manybody.pdf")))
