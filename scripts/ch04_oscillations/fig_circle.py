"""Figure fig:os-circle: the oscillator as the shadow of a circle.

The block of Section 2.8 (0.5 kg on 50 N/m, so omega = 10 rad/s) starts
3 cm out and moving outwards at 0.4 m/s. (a) A point going round a circle
of radius A = 5 cm at omega, starting at the angle phi = -0.927 rad,
shown at four times, with its shadow on the x axis. (b) The shadow's
position against time, with time running downwards so that each square
lies directly below the square of the same colour in (a).

Prints the numbers that Section 4.1 (sec:os-circle) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import oscillators, viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE, THRESHOLD_STYLE, figure_path

OMEGA, X0, V0 = 10.0, 0.03, 0.4
AMP = np.hypot(X0, V0 / OMEGA)
PHI = np.arctan2(-V0 / OMEGA, X0)
TIMES = (0.0, 0.1, 0.2, 0.3)  # s, the moments marked on both panels
print(f"A = {100 * AMP:.1f} cm, phi = {PHI:.4f} rad")
for t in TIMES:
    x, v = oscillators.harmonic(t, X0, V0, OMEGA)
    print(
        f"t = {t:.2f} s: angle {OMEGA * t + PHI:+.3f} rad, x = "
        f"{100 * float(x):+.2f} cm, v = {float(v):+.3f} m/s"
    )

viz.use_style(notebook=False)
fig, (top, bottom) = plt.subplots(
    2,
    1,
    figsize=(viz.HALF * 1.25, 4.6),
    sharex=True,
    gridspec_kw=dict(height_ratios=(1.0, 1.25), hspace=0.15),
)
theta = np.linspace(0.0, 2 * np.pi, 400)
top.plot(5 * np.cos(theta), 5 * np.sin(theta), color=REFERENCE, lw=0.8)
top.axhline(0.0, color="black", lw=0.5)
colours = (ACCENT, OCHRE, "#7C2529", "#5B7553")
for t, colour in zip(TIMES, colours, strict=True):
    angle = OMEGA * t + PHI
    px, py = 5 * np.cos(angle), 5 * np.sin(angle)
    top.plot([0, px], [0, py], color=colour, lw=0.6)
    top.plot(px, py, "o", color=colour, ms=4)
    top.plot([px, px], [py, 0], **THRESHOLD_STYLE)
    top.plot(px, 0, "s", color=colour, ms=3.5)
top.set_aspect("equal")
top.set_ylim(-5.6, 5.6)
top.set_ylabel(r"$y$ / cm")
viz.panel_tag(top, "a")

t = np.linspace(0.0, 0.7, 400)
x, _ = oscillators.harmonic(t, X0, V0, OMEGA)
bottom.plot(100 * x, t, color="black", lw=1.0)
for tm, colour in zip(TIMES, colours, strict=True):
    xm, _ = oscillators.harmonic(tm, X0, V0, OMEGA)
    bottom.plot(100 * xm, tm, "s", color=colour, ms=3.5)
bottom.set_ylim(0.7, -0.01)
bottom.set_xlim(-5.6, 5.6)
bottom.set_xlabel(r"position $x$ / cm")
bottom.set_ylabel(r"time $t$ / s")
viz.panel_tag(bottom, "b")
top.apply_aspect()  # shrink (a) to its equal-aspect box, then match (b)
above, below = top.get_position(), bottom.get_position()
bottom.set_position([above.x0, below.y0, above.width, below.height])

print("wrote", viz.save(fig, figure_path("ch04_oscillations", "circle.pdf")))
