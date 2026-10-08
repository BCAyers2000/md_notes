"""Figure fig:ha-portrait: the pendulum's phase portrait from H.

A bob of 0.5 kg on a light rod of 1.2 m, with p = mL²θ̇ and
H = p²/(2mL²) + mgL(1 − cos θ). (a) Contours of H in the plane of θ and
p, with the flow (∂H/∂p, −∂H/∂q) as arrows: swinging below the upright
energy 2mgL (grey), the separatrix at 2mgL (teal), going over the top
above it (ochre). (b) Started at the bottom with the separatrix momentum
2mL²ω₀, the bob creeps towards the top: π − θ against time on a
logarithmic axis, from Hamilton's equations (dots) and the straight line
of the exponential approach e^{−ω₀t} (dashed).

Prints the numbers of Section 7.2.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import g as G

from mdlab import hamiltonian, viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, figure_path

MASS, LENGTH = 0.5, 1.2
INERTIA = MASS * LENGTH**2  # mL², kg m²
MGL = MASS * G * LENGTH
OMEGA0 = math.sqrt(G / LENGTH)


def dh_dq(q, p):
    return MGL * np.sin(q)


def dh_dp(q, p):
    return p / INERTIA


def energy(q, p):
    return p**2 / (2 * INERTIA) + MGL * (1 - np.cos(q))


p_sep = 2 * INERTIA * OMEGA0
print(f"mL² = {INERTIA:.3f} kg m², mgL = {MGL:.5f} J, 2mgL = {2 * MGL:.4f} J")
print(f"omega0 = {OMEGA0:.4f} rad/s, 1/omega0 = {1 / OMEGA0:.4f} s")
print(f"separatrix momentum at the bottom 2 mL² omega0 = {p_sep:.4f} kg m²/s")
print(f"  check sqrt(2 mL² 2mgL) = {math.sqrt(2 * INERTIA * 2 * MGL):.4f}")
print(f"halving time ln2/omega0 = {math.log(2) / OMEGA0:.4f} s")
print(f"10 deg to 1 deg: ln10/omega0 = {math.log(10) / OMEGA0:.4f} s")

times = np.linspace(0.0, 2.5, 26)
q, _ = hamiltonian.flow(dh_dq, dh_dp, [0.0], [p_sep], times)
gap = np.pi - q[:, 0]
exact = 4 * np.arctan(np.exp(-OMEGA0 * times))
print(
    "largest |computed − 4 arctan(e^(−ω0 t))| = "
    f"{np.abs(gap - exact).max():.1e}"
)
for t_mark in (0.5, 1.0, 1.5, 2.0, 2.5):
    i = int(round(t_mark / 0.1))
    print(
        f"t = {t_mark} s: pi − theta = {gap[i]:.6f} rad = "
        f"{math.degrees(gap[i]):.3f} deg"
    )
late = (
    np.pi
    - hamiltonian.flow(
        dh_dq, dh_dp, [0.0], [p_sep], [0.0, 2.0, 2.0 + math.log(2) / OMEGA0]
    )[0][:, 0]
)
print(
    f"from t = 2 s, after ln2/omega0 the gap falls by {late[1] / late[2]:.4f}"
)

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1,
    2,
    figsize=(viz.FULL, 2.7),
    gridspec_kw=dict(wspace=0.36, width_ratios=[1.35, 1]),
)
theta = np.linspace(-2 * np.pi, 2 * np.pi, 600)
p = np.linspace(-7.0, 7.0, 500)
TH, PP = np.meshgrid(theta, p)
H = energy(TH, PP) / MGL
left.contour(TH, PP, H, levels=[0.3, 0.9, 1.5], colors="0.6", linewidths=0.7)
left.contour(TH, PP, H, levels=[2.0], colors=ACCENT, linewidths=1.3)
left.contour(TH, PP, H, levels=[2.8, 4.2], colors=OCHRE, linewidths=0.8)


def arrow(x, y, sign, colour):
    """An arrowhead on a contour at (x, y), pointing along ±θ."""
    left.annotate(
        "",
        xy=(x + 0.25 * sign, y),
        xytext=(x, y),
        arrowprops=dict(
            arrowstyle="-|>", color=colour, lw=0.8, mutation_scale=7
        ),
    )


for level, colour, centres in [
    (0.3, "0.6", (-2 * np.pi, 0.0, 2 * np.pi)),
    (0.9, "0.6", (-2 * np.pi, 0.0, 2 * np.pi)),
    (1.5, "0.6", (-2 * np.pi, 0.0, 2 * np.pi)),
    (2.0, ACCENT, (-2 * np.pi, 0.0, 2 * np.pi)),
    (2.8, OCHRE, (-np.pi, np.pi)),
    (4.2, OCHRE, (-np.pi, np.pi)),
]:
    for x in centres:
        top = math.sqrt(2 * INERTIA * MGL * (level - (1 - math.cos(x))))
        for sign in (1, -1):
            if abs(x) == 2 * np.pi:
                continue
            arrow(x - 0.125 * sign, sign * top, sign, colour)
left.set_xticks(
    [k * np.pi for k in range(-2, 3)],
    [r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"],
)
left.set_xlim(-2 * np.pi, 2 * np.pi)
left.set_ylim(-7, 7)
left.set_xlabel(r"angle $\theta$ / rad")
left.set_ylabel(r"$p$ / kg m$^2$ s$^{-1}$")
viz.panel_tag(left, "a")

right.semilogy(times, gap, "o", color=ACCENT, ms=3)
right.semilogy(times, 4 * np.exp(-OMEGA0 * times), **REFERENCE_STYLE)
right.set_xlim(0, 2.5)
right.set_ylim(1e-3, 5)
right.set_xlabel(r"time $t$ / s")
right.set_ylabel(r"$\pi - \theta$ / rad")
right.text(1.25, 0.3, r"$\propto \mathrm{e}^{-\omega_0 t}$", color="0.4")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch07_hamilton", "portrait.pdf")))
