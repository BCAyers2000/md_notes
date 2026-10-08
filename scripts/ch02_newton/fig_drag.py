"""Figure fig:ne-drag: falling from rest with a drag proportional to v.

Measuring depth downwards, dv/dt = g - gamma v. (a) The velocity for
three drag rates gamma, approaching the terminal velocity
v_inf = g / gamma (dotted), against free fall, v = g t (dashed); dots
mark t = 1/gamma, where 63% of v_inf is reached. (b) The distance fallen
for gamma = 5 per second, against free fall and against the line
v_inf (t - 1/gamma) that it approaches.

Prints the numbers that Section 2.7 (sec:ne-drag) quotes.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import dynamics, viz
from mdlab.viz import (
    ACCENT,
    OCHRE,
    OXBLOOD,
    REFERENCE,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

GAMMAS = (2.5, 5.0, 10.0)  # drag rates b/m, in 1/s
COLOURS = (OCHRE, ACCENT, OXBLOOD)
MASS, DRAG_COEFFICIENT = 0.01, 0.05  # the worked example: kg, kg/s
T_END = 1.6  # s

# The worked example --------------------------------------------------------
gamma = DRAG_COEFFICIENT / MASS
v_inf = G / gamma
print(
    f"m = {MASS} kg, b = {DRAG_COEFFICIENT} kg/s: gamma = {gamma} /s, "
    f"1/gamma = {1 / gamma} s, v_inf = {v_inf:.4f} m/s"
)
for n in (1, 2, 3, 5):
    print(
        f"  t = {n / gamma:.1f} s (gamma t = {n}): "
        f"e^-gamma t = {np.exp(-n):.4f}, v / v_inf = {1 - np.exp(-n):.4f}"
    )
print(f"  90% of v_inf after ln 10 / gamma = {np.log(10) / gamma:.4f} s")
print(f"  lag behind v_inf t: v_inf / gamma = {v_inf / gamma:.4f} m")
for t_check in (0.02, 0.1, 1.0):
    x_drag, _ = dynamics.drag_motion(t_check, 0.0, 0.0, G, gamma)
    print(
        f"  fallen after {t_check} s: {1000 * x_drag:.3f} mm with drag, "
        f"{1000 * 0.5 * G * t_check**2:.3f} mm in free fall"
    )

# The figure ----------------------------------------------------------------
t = np.linspace(0.0, T_END, 400)
viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.45), gridspec_kw=dict(wspace=0.32)
)

left.plot(t, G * t, lw=1.0, **REFERENCE_STYLE)
left.text(0.50, 4.5, r"free fall, $gt$", color=REFERENCE, ha="left")
for gam, colour in zip(GAMMAS, COLOURS, strict=True):
    _, v = dynamics.drag_motion(t, 0.0, 0.0, G, gam)
    left.plot(t, v, color=colour)
    left.axhline(G / gam, **THRESHOLD_STYLE)
    left.plot(1 / gam, (1 - np.exp(-1)) * G / gam, "o", color=colour, ms=3.5)
    left.text(
        T_END - 0.02,
        G / gam + 0.12,
        rf"$\gamma = {gam:g}$ s$^{{-1}}$",
        color=colour,
        ha="right",
        va="bottom",
    )
left.set_xlim(0, T_END)
left.set_ylim(0, 4.7)
left.set_xlabel(r"time $t$ / s")
left.set_ylabel(r"velocity $v$ / m\,s$^{-1}$")
viz.panel_tag(left, "a")

x, _ = dynamics.drag_motion(t, 0.0, 0.0, G, gamma)
right.plot(t, 0.5 * G * t**2, lw=1.0, **REFERENCE_STYLE)
right.text(0.86, 2.95, r"free fall, $gt^2/2$", color=REFERENCE, ha="left")
right.plot(t, x, color=ACCENT, label=r"$\gamma = 5$ s$^{-1}$")
right.plot(
    t,
    v_inf * (t - 1 / gamma),
    label=r"$v_\infty(t - 1/\gamma)$",
    **THRESHOLD_STYLE,
)
right.set_xlim(0, T_END)
right.set_ylim(0, 3.2)
right.set_xlabel(r"time $t$ / s")
right.set_ylabel(r"distance fallen $x$ / m")
right.legend(loc="lower right", handlelength=1.4)
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch02_newton", "drag.pdf")))
