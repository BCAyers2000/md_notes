"""Figure fig:os-wells: three anharmonic wells and their parabolas.

Each well (black) with the parabola of the same curvature at its bottom
(dashed), in units natural to it. (a) The cosine well, U = (D/2)(1 - cos
2 pi x / Lambda): the pendulum, and the hop coordinate of the model
surface. (b) The Morse well, U = D(1 - e^(-a(r - r0)))^2. (c) The
Lennard-Jones well, U = 4 eps((sigma/r)^12 - (sigma/r)^6), drawn from its
bottom. Dotted lines mark half the depth.

Prints the curvatures quoted in Section 4.5 (sec:os-anharmonic).
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import oscillators, viz
from mdlab.viz import REFERENCE_STYLE, THRESHOLD_STYLE, figure_path

viz.use_style(notebook=False)
fig, axes = plt.subplots(
    1, 3, figsize=(viz.FULL, 2.3), gridspec_kw=dict(wspace=0.38)
)

x = np.linspace(-1.0, 1.0, 400)  # in units of Lambda
u, _ = oscillators.cosine_well(x, 1.0, 1.0)
k_cos = 0.5 * (2 * np.pi) ** 2
axes[0].plot(x, u, color="black")
axes[0].plot(x, 0.5 * k_cos * x**2, lw=1.0, **REFERENCE_STYLE)
axes[0].set_xlabel(r"$x/\Lambda$")
axes[0].set_ylabel(r"$U/D$")
print(f"cosine well: curvature (D/2)(2 pi/Lambda)^2 = {k_cos:.4f} D/Lambda^2")

y = np.linspace(-0.8, 4.0, 400)  # a (r - r0)
u, _ = oscillators.morse(y, 1.0, 1.0, 0.0)
axes[1].plot(y, u, color="black")
axes[1].plot(y, y**2, lw=1.0, **REFERENCE_STYLE)  # 2 D a² y²/2
axes[1].set_xlabel(r"$a(r - r_0)$")
axes[1].set_ylabel(r"$U/D$")

r = np.linspace(0.95, 2.6, 400)  # r / sigma
u, _ = oscillators.lennard_jones(r, 1.0, 1.0)
r_min = 2 ** (1 / 6)
k_lj = 72 / 2 ** (1 / 3)
axes[2].plot(r, u + 1.0, color="black")
axes[2].plot(r, 0.5 * k_lj * (r - r_min) ** 2, lw=1.0, **REFERENCE_STYLE)
axes[2].set_xlabel(r"$r/\sigma$")
axes[2].set_ylabel(r"$(U + \varepsilon)/\varepsilon$")
print(f"LJ: curvature {k_lj:.3f} eps/sigma^2 at r = {r_min:.4f} sigma")

for ax, tag in zip(axes, "abc", strict=True):
    ax.axhline(0.5, **THRESHOLD_STYLE)
    ax.set_ylim(0, 1.6)
    viz.panel_tag(ax, tag)

print("wrote", viz.save(fig, figure_path("ch04_oscillations", "wells.pdf")))
