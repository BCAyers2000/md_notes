"""Figure fig:ro-rolling: a block, a disc and a hoop race down a slope.

All three start at rest on a slope of 30°. The block slides without
friction; the disc and the hoop roll without slipping. Rolling with
ω = v/R, the kinetic energy is ½(M + I/R²)v², so with κ = I/(MR²) the
acceleration down the slope is g sin 30° / (1 + κ), whatever the mass and
the radius: κ = 0 for the block, ½ for the disc, 1 for the hoop.
(a) Distance down the slope against time. (b) How the kinetic energy at
the bottom is shared between moving along and turning.

Prints the numbers of Section 5.4.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import viz
from mdlab.viz import (
    ACCENT,
    GREEN,
    OCHRE,
    REFERENCE,
    REFERENCE_STYLE,
    figure_path,
)

SLOPE, LENGTH = math.radians(30.0), 2.0
BODIES = (
    ("sliding block", 0.0, REFERENCE_STYLE),
    ("rolling disc", 0.5, {"color": ACCENT}),
    ("rolling hoop", 1.0, {"color": OCHRE}),
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1,
    2,
    figsize=(viz.FULL, 2.4),
    gridspec_kw=dict(wspace=0.35, width_ratios=(1.4, 1.0)),
)
t = np.linspace(0.0, 1.4, 400)
for name, kappa, style in BODIES:
    accel = G * math.sin(SLOPE) / (1 + kappa)
    down = 0.5 * accel * t**2
    left.plot(t[down <= LENGTH], down[down <= LENGTH], label=name, **style)
    t_end = math.sqrt(2 * LENGTH / accel)
    print(f"{name}: a = {accel:.3f} m/s², 2 m in {t_end:.3f} s")
left.axhline(LENGTH, color="black", lw=0.4)
left.set_xlim(0, 1.4)
left.set_ylim(0, 2.1)
left.set_xlabel(r"time $t$ / s")
left.set_ylabel("distance down the slope / m")
left.legend(loc="lower right")
viz.panel_tag(left, "a")

names = ["block", "disc", "hoop"]
along = np.array([1 / (1 + kappa) for _, kappa, _ in BODIES])
turning = 1 - along
right.bar(names, along, color=REFERENCE, label="moving along")
right.bar(names, turning, bottom=along, color=GREEN, label="turning")
right.set_ylim(0, 1.3)
right.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
right.set_ylabel("share of $K$")
right.legend(loc="upper center", ncols=2, fontsize="small")
viz.panel_tag(right, "b")
print("shares in turning:", turning.round(3))

print("wrote", viz.save(fig, figure_path("ch05_rotation", "rolling.pdf")))
