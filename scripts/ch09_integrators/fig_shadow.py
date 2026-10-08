"""Figure fig:in-shadow: the energy velocity Verlet keeps.

A spring with m = k = 1 (ω = 1), started at q = 1 at rest. (a) Forty steps
of velocity Verlet with ωδt = 0.8 (dots) lie on the ellipse of the
shadow energy H̃ = p²/2 + (1 − ω²δt²/4) q²/2 (teal), slightly lower than
the true ellipse (dashed), a circle in these units.
(b) With ωδt = 0.5, the energy H of the velocity Verlet states as a
multiple of its start, oscillating in a band, and H̃, constant.

Prints the numbers of Section 9.5.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import integrators, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, figure_path


def spring(r):
    return -r


def march(method, dt, n):
    r, v, f = np.array([[1.0]]), np.array([[0.0]]), None
    qs, ps = [1.0], [0.0]
    for _ in range(n):
        if method == "vv":
            r, v, f = integrators.velocity_verlet_step(
                r, v, [1.0], spring, dt, f, 1.0
            )
        else:
            r, v = integrators.euler_step(r, v, [1.0], spring, dt, 1.0)
        qs.append(r.item())
        ps.append(v.item())
    return np.array(qs), np.array(ps)


def shadow(q, p, dt):
    return 0.5 * p**2 + 0.5 * (1 - dt**2 / 4) * q**2


dt_a, dt_b = 0.8, 0.5
qv, pv = march("vv", dt_a, 40)
spread = np.ptp(shadow(qv, pv, dt_a)) / shadow(1.0, 0.0, dt_a)
print(
    f"omega dt = {dt_a}: shadow energy constant to {spread:.1e} over 40 "
    f"steps; semi-axis in q {1.0:.3f}, in p {math.sqrt(1 - dt_a**2 / 4):.4f}"
)
qb, pb = march("vv", dt_b, 400)
h = 0.5 * pb**2 + 0.5 * qb**2
print(
    f"omega dt = {dt_b}: H/H0 between {h.min() / h[0]:.4f} and "
    f"{h.max() / h[0]:.4f}; 1 - (omega dt)²/4 = {1 - dt_b**2 / 4:.4f}"
)
print(
    f"shadow energy constant to "
    f"{np.ptp(shadow(qb, pb, dt_b)) / shadow(1.0, 0.0, dt_b):.1e}"
)
for dt in (0.5, 0.8, 0.1):
    w = math.acos(1 - dt**2 / 2) / dt
    print(
        f"omega dt = {dt}: numerical angular frequency {w:.5f} "
        f"(shift {100 * (w - 1):.3f}%)"
    )

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.7), gridspec_kw=dict(wspace=0.35)
)
s = np.linspace(0, 2 * np.pi, 300)
left.plot(np.cos(s), np.sin(s), **REFERENCE_STYLE)
left.plot(
    np.cos(s), math.sqrt(1 - dt_a**2 / 4) * np.sin(s), color=ACCENT, lw=0.8
)
left.plot(qv, pv, "o", color=ACCENT, ms=2.5)
left.set_aspect("equal")
left.set_xlim(-1.25, 1.25)
left.set_ylim(-1.25, 1.25)
left.set_xlabel(r"$q$")
left.set_ylabel(r"$p$")
viz.panel_tag(left, "a")
t = dt_b * np.arange(len(h))
right.plot(t, h / h[0], color=ACCENT, lw=0.8, label=r"$H$")
right.plot(
    t,
    shadow(qb, pb, dt_b) / shadow(1.0, 0.0, dt_b),
    color="black",
    lw=1.2,
    label=r"$\tilde H$",
)
right.set_ylim(0.88, 1.02)
right.set_xlim(0, 60)
right.set_xlabel(r"time $t$ / $(1/\omega)$")
right.set_ylabel("energy / its start")
right.legend(loc="lower right", fontsize=7)
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch09_integrators", "shadow.pdf")))
