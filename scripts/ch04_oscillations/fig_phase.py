"""Figure fig:os-phase: phase portraits.

(a) The block of Section 2.8 at three energies, 0.01, 0.04 and 0.09 J:
ellipses in the plane of position and velocity, traversed clockwise.
(b) The pendulum, U = mgL(1 - cos theta), in units of its small-swing
angular frequency: closed curves while it swings, the separatrix (teal)
through the upright position, and open curves when it goes over the top.

Prints the semi-axes quoted in Section 4.3 and the energies of panel (b).
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE, figure_path

MASS, K = 0.5, 50.0

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.32)
)
s = np.linspace(0.0, 2 * np.pi, 400)
for e in (0.01, 0.04, 0.09):
    a, b = np.sqrt(2 * e / K), np.sqrt(2 * e / MASS)
    print(f"E = {e} J: semi-axes {100 * a:.1f} cm, {b:.2f} m/s")
    colour = ACCENT if e == 0.04 else REFERENCE
    left.plot(100 * a * np.cos(s), b * np.sin(s), color=colour)
    # clockwise: at the top (v > 0) the block moves to larger x
    left.annotate(
        "",
        xy=(100 * a * 0.25, b * np.sqrt(1 - 0.0625)),
        xytext=(-100 * a * 0.25, b * np.sqrt(1 - 0.0625)),
        arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.8),
    )
left.axhline(0, color="black", lw=0.4)
left.axvline(0, color="black", lw=0.4)
left.set_xlabel(r"position $x$ / cm")
left.set_ylabel(r"velocity $v$ / m\,s$^{-1}$")
left.set_aspect(10)  # 0.1 m/s drawn as long as 1 cm
viz.panel_tag(left, "a")

theta = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 800)
for e in (0.3, 1.0, 1.7, 2.0, 2.5):  # E / (m g L)
    grid = theta
    if e < 2.0:  # sample the turning points so that each swing closes
        turn = np.arccos(1.0 - e)
        ends = [
            s * turn + c for s in (-1, 1) for c in (-2 * np.pi, 0, 2 * np.pi)
        ]
        grid = np.sort(np.concatenate([theta, ends]))
    rate2 = 2 * (e - (1 - np.cos(grid)))  # (dθ/dt / ω₀)²
    rate2[np.abs(rate2) < 1e-12] = 0.0
    speed = np.sqrt(np.where(rate2 >= 0, rate2, np.nan))
    colour = ACCENT if e == 2.0 else (OCHRE if e > 2.0 else REFERENCE)
    right.plot(grid / np.pi, speed, color=colour)
    right.plot(grid / np.pi, -speed, color=colour)
right.axhline(0, color="black", lw=0.4)
right.set_xticks(
    [-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
    [
        r"$-\frac{3\pi}{2}$",
        r"$-\pi$",
        r"$-\frac{\pi}{2}$",
        "0",
        r"$\frac{\pi}{2}$",
        r"$\pi$",
        r"$\frac{3\pi}{2}$",
    ],
)
right.set_xlim(-1.5, 1.5)
right.set_xlabel(r"angle $\theta$")
right.set_ylabel(r"$\dot\theta/\omega_0$")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch04_oscillations", "phase.pdf")))
