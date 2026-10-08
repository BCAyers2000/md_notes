"""Figure fig:en-drift: a force that is not −∇U changes the energy.

The lithium atom of fig:en-li, launched the same way, once on the model
surface alone and once with the swirl c ẑ × r (c = 0.005 eV/Å²) added
to the force. The change of the total energy E = K + U from its starting
value, against the work done by the swirl, ∫ F_sw · v dt (dashed).

Prints the numbers that Section 3.14 (sec:en-drift) quotes.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from mdlab import dynamics, energy, units, viz
from mdlab.viz import ACCENT, OXBLOOD, REFERENCE_STYLE, figure_path

MASS_LI = 6.94  # amu
K0 = 0.33  # eV
ANGLE = np.radians(37.5)
SWIRL = 0.005  # eV/Å²
DURATION = 1500.0  # fs

speed = math.sqrt(2 * K0 / (MASS_LI * units.MV2_TO_EV))
v0 = speed * np.array([np.cos(ANGLE), np.sin(ANGLE)])
t = np.linspace(0.0, DURATION, 15001)


def surface_force(r):
    return energy.hexagonal_surface(r)[1]


def swirled_force(r):
    return surface_force(r) + energy.swirl(r, SWIRL)


changes = {}
for name, force in (("surface", surface_force), ("swirled", swirled_force)):
    r, v = dynamics.solve_newton(
        force, MASS_LI, [0.0, 0.0], v0, t, force_to_accel=units.FORCE_TO_ACCEL
    )
    kinetic = 0.5 * MASS_LI * np.sum(v**2, axis=-1) * units.MV2_TO_EV
    total = kinetic + energy.hexagonal_surface(r)[0]
    changes[name] = total - total[0]
    if name == "swirled":
        power = np.sum(energy.swirl(r, SWIRL) * v, axis=-1)
        supplied = cumulative_trapezoid(power, t, initial=0.0)
        print(
            f"swirled: E - E(0) from {changes[name].min():+.4f} to "
            f"{changes[name].max():+.4f} eV; largest |E - E(0) - W_sw| = "
            f"{np.max(np.abs(changes[name] - supplied)):.1e} eV; "
            f"farthest from the centre "
            f"{np.linalg.norm(r, axis=-1).max():.2f} Å"
        )
    else:
        largest = np.max(np.abs(changes[name]))
        print(f"surface: largest |E - E(0)| = {largest:.1e} eV")

viz.use_style(notebook=False)
fig, ax = plt.subplots(figsize=(viz.FULL, 2.2))
ax.axhline(0.0, color="black", lw=0.5)
ax.plot(t, changes["surface"], color=ACCENT)
ax.plot(t, changes["swirled"], color=OXBLOOD)
ax.plot(t, supplied, lw=1.0, **REFERENCE_STYLE)
ax.text(20, -0.0015, "surface alone", color=ACCENT, ha="left", va="top")
ax.set_xlim(0, DURATION)
ax.set_xlabel(r"time $t$ / fs")
ax.set_ylabel(r"$E - E(0)$ / eV")
ax.legend(
    ax.get_lines()[2:4],
    [
        "surface and swirl",
        "work done by the swirl",
    ],
    loc="lower center",
    ncols=2,
    bbox_to_anchor=(0.5, 1.0),
)

print("wrote", viz.save(fig, figure_path("ch03_energy", "drift.pdf")))
