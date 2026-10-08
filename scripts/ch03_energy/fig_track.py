"""Figure fig:en-track: the track as an energy diagram.

A bead slides without friction on a wire bent into the shape
h(x) = 0.15 x⁴ − 0.6 x² + 0.15 x + 1.0 (metres). Its potential energy is
U = mgh, so U/mg is the track itself. (a) The track with three total
energies E/mg = 0.4, 0.8 and 1.3 m: each line is drawn solid where the
bead may be (U ≤ E) and dotted where it may not; dots mark the turning
points; the equilibria are marked on the track. (b) The speed along the
track at E/mg = 1.3 m, √(2g(E/mg − h)), which is zero at the turning
points and largest in the deep valley.

Prints the numbers that Sections 3.5 and 3.6 quote.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import energy, viz
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE, figure_path

LEVELS = (0.4, 0.8, 1.3)  # E/mg, m
COLOURS = (OXBLOOD, OCHRE, ACCENT)
X_END = 2.4  # m

x = np.linspace(-X_END, X_END, 48001)
h, _ = energy.hill_track(x)

roots = np.sort(np.roots([0.6, 0.0, -1.2, 0.15]).real)  # dh/dx = 0
peaks, _ = energy.hill_track(roots)
for level in LEVELS:
    turns = energy.turning_points(x, h, level)
    print(f"E/mg = {level} m: turning points {turns.round(3)} m")
speed = np.sqrt(2 * G * np.clip(LEVELS[-1] - h, 0.0, None))
at_hill, at_right = np.sqrt(2 * G * (LEVELS[-1] - peaks[1:]))
print(
    f"E/mg = 1.3 m: fastest {speed.max():.3f} m/s at x = "
    f"{x[np.argmax(speed)]:.3f} m; at the hill {at_hill:.3f} m/s; "
    f"in the right valley {at_right:.3f} m/s"
)

viz.use_style(notebook=False)
fig, (top, bottom) = plt.subplots(
    2,
    1,
    figsize=(viz.FULL, 4.3),
    sharex=True,
    gridspec_kw=dict(height_ratios=(1.7, 1.0), hspace=0.18),
)

fig.subplots_adjust(right=0.82)  # room for the energy labels
top.plot(x, h, color="black", lw=1.6)
for level, colour in zip(LEVELS, COLOURS, strict=True):
    allowed = np.where(h <= level, level, np.nan)
    top.plot(x, np.full_like(x, level), color=colour, lw=0.7, ls=":")
    top.plot(x, allowed, color=colour, lw=1.8)
    turns = energy.turning_points(x, h, level)
    top.plot(turns, np.full_like(turns, level), "o", color=colour, ms=3.5)
    top.text(
        X_END + 0.05, level, rf"$E/mg = {level}$ m", color=colour, va="center"
    )
stable = (roots[0], roots[2])
top.plot(
    stable, energy.hill_track(np.array(stable))[0], "o", color="black", ms=4.5
)
top.plot(roots[1], peaks[1], "o", mfc="white", mec="black", ms=4.5)
top.text(roots[0], 0.075, "stable", ha="center", va="center")
top.text(roots[2], 0.27, "stable", ha="center", va="center")
top.text(roots[1], peaks[1] + 0.09, "unstable", ha="center", va="bottom")
top.set_ylim(0.0, 2.0)
top.set_ylabel(r"$U/mg = h$ / m")
viz.panel_tag(top, "a")

bottom.plot(x, np.where(h <= LEVELS[-1], speed, np.nan), color=ACCENT)
for xt in energy.turning_points(x, h, LEVELS[-1]):
    bottom.axvline(xt, **THRESHOLD_STYLE)
bottom.set_xlim(-X_END, X_END)
bottom.set_ylim(0, 5.2)
bottom.set_xlabel(r"horizontal position $x$ / m")
bottom.set_ylabel(r"speed / m\,s$^{-1}$")
viz.panel_tag(bottom, "b")

print("wrote", viz.save(fig, figure_path("ch03_energy", "track.pdf")))
