"""Figure fig:ne-spring: a block on a spring, released from rest.

A 0.5 kg block on a spring of 50 N/m is pulled 4 cm from its natural
position and released, so x = A cos(omega t) with omega = 10 rad/s.
(a) Position, (b) velocity and (c) acceleration over two periods. Dotted
lines mark the turning points, where the velocity is zero and the
acceleration largest; dashed lines mark the passes through the centre,
where the speed is largest and the acceleration zero.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, REFERENCE, THRESHOLD_STYLE, figure_path

MASS, STIFFNESS, AMPLITUDE = 0.5, 50.0, 0.04  # kg, N/m, m
OMEGA = np.sqrt(STIFFNESS / MASS)
PERIOD = 2 * np.pi / OMEGA

t = np.linspace(0.0, 2 * PERIOD, 600)
x = AMPLITUDE * np.cos(OMEGA * t)
v = -AMPLITUDE * OMEGA * np.sin(OMEGA * t)
a = -(OMEGA**2) * x
turning = np.arange(0, 5) * PERIOD / 2  # cos = ±1
centre = PERIOD / 4 + np.arange(0, 4) * PERIOD / 2  # cos = 0
print(
    f"omega = {OMEGA:.1f} rad/s, period {PERIOD:.4f} s; largest speed "
    f"{AMPLITUDE * OMEGA:.2f} m/s, largest acceleration "
    f"{AMPLITUDE * OMEGA**2:.2f} m/s²"
)

viz.use_style(notebook=False)
fig, axes = plt.subplots(
    3, 1, figsize=(viz.FULL, 3.9), sharex=True, gridspec_kw=dict(hspace=0.38)
)
rows = (
    (100 * x, r"$x$ / cm"),
    (v, r"$v$ / m\,s$^{-1}$"),
    (a, r"$a$ / m\,s$^{-2}$"),
)
for ax, (curve, label), tag in zip(axes, rows, "abc", strict=True):
    for moment in turning:
        ax.axvline(moment, **THRESHOLD_STYLE)
    for moment in centre:
        ax.axvline(moment, color=REFERENCE, lw=0.6, ls="--", alpha=0.6)
    ax.axhline(0.0, color="black", lw=0.4)
    ax.plot(t, curve, color=ACCENT)
    ax.set_ylabel(label)
    viz.panel_tag(ax, tag)
axes[0].set_ylim(-5, 5)
axes[1].set_ylim(-0.5, 0.5)
axes[2].set_ylim(-5, 5)
axes[-1].set_xlim(0, 2 * PERIOD)
axes[-1].set_xlabel(r"time $t$ / s")

print("wrote", viz.save(fig, figure_path("ch02_newton", "spring.pdf")))
