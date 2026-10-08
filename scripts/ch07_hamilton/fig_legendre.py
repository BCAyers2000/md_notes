"""Figure fig:ha-legendre: the Legendre transform of ½mv².

f(v) = ½mv² for a body of 0.5 kg. (a) The line pv of slope p = 1 kg m/s
through the origin lies above the curve f(v) by pv − f(v), which is
greatest where the slope of f equals p, at v = p/m = 2 m/s; there the
tangent of slope p meets the vertical axis at −f*(p). (b) The transform
f*(p) = p²/(2m), whose slope at p is v.

Prints the numbers of Section 7.1.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, THRESHOLD_STYLE, figure_path

MASS, P0 = 0.5, 1.0  # kg, kg m/s
V0 = P0 / MASS


def f(v):
    return 0.5 * MASS * v**2


v = np.linspace(-0.5, 3.6, 400)
gap = P0 * v - f(v)
best = v[np.argmax(gap)]
f_star = P0 * V0 - f(V0)
print(f"v at the largest gap {best:.3f} m/s (p/m = {V0:.3f})")
print(f"f(v) = {f(V0):.3f} J, f*(p) = pv - f(v) = {f_star:.3f} J")
print(f"p²/(2m) = {P0**2 / (2 * MASS):.3f} J")
for vv in (1.0, 3.0):
    print(f"gap at v = {vv} m/s: {P0 * vv - f(vv):.3f} J")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.38)
)
left.plot(v, f(v), color="black", label=r"$f(v)$")
left.plot(v, P0 * v, label=r"$pv$", **REFERENCE_STYLE)
left.plot(v, P0 * v - f_star, color=ACCENT, lw=0.8)
left.annotate(
    "",
    xy=(V0, f(V0)),
    xytext=(V0, P0 * V0),
    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0),
)
left.text(V0 - 0.12, 2.02, r"$f^*(p)$", color=ACCENT, ha="right")
left.annotate(
    "",
    xy=(0.0, -f_star),
    xytext=(0.0, 0.0),
    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0),
)
left.text(0.65, -0.85, r"$f^*(p)$", color=ACCENT, va="center")
left.plot([V0], [f(V0)], "o", color=ACCENT, ms=3.5)
left.axhline(0, **THRESHOLD_STYLE)
left.axvline(0, **THRESHOLD_STYLE)
left.text(-0.38, 0.22, r"$f$", ha="left")
left.text(2.75, 3.0, r"$pv$", color="0.4", ha="right")
left.set_xlim(-0.5, 3.6)
left.set_ylim(-1.4, 3.4)
left.set_xlabel(r"velocity $v$ / m s$^{-1}$")
left.set_ylabel(r"energy / J")
viz.panel_tag(left, "a")

p = np.linspace(-0.3, 1.8, 400)
right.plot(p, p**2 / (2 * MASS), color="black")
right.plot(p, f_star + V0 * (p - P0), color=ACCENT, lw=0.8)
right.plot([P0], [f_star], "o", color=ACCENT, ms=3.5)
right.axhline(0, **THRESHOLD_STYLE)
right.axvline(0, **THRESHOLD_STYLE)
right.text(1.25, 0.45, r"slope $v$", color=ACCENT)
right.text(1.55, 2.75, r"$f^*$", ha="right")
right.set_xlim(-0.3, 1.8)
right.set_ylim(-1.4, 3.4)
right.set_xlabel(r"momentum $p$ / kg m s$^{-1}$")
right.set_ylabel(r"$f^*(p)$ / J")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch07_hamilton", "legendre.pdf")))
