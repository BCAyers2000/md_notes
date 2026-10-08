"""Figure fig:pr-piston: a piston with mass rings.

The liquid at 135 K and 0.1 GPa under the MTK piston with chains
(runs.py, npt_mtk_*_s0). (a) The volume's departure from its mean over
the first 30 ps for the piston time constants τ_P = 1, 3 and 10 ps, each
trace offset by 1200 Å³, with a bar of 500 Å³. (b) The period of the
volume's swing, from its crossings of its mean after the first 10 ps,
against τ_P (points), and the period 2πτ_P √((N + 1)k_BTκ_T/3V) of the
linearised piston (dashed), with κ_T from the fixed-volume runs.

Prints the measured and predicted periods.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch14 import N_ATOMS, RUNS, T_LIQUID, compressibility_at, crossings_period

from mdlab import units, viz
from mdlab.viz import BAROSTAT, REFERENCE_STYLE, figure_path

NAMES = {0.3: "npt_mtk_pd0.3_s0", 1.0: "npt_mtk_s0", 3.0: "npt_mtk_pd3_s0",
         10.0: "npt_mtk_pd10_s0"}
COLOUR = BAROSTAT["MTK piston"]

runs = {tau: np.load(RUNS / f"{name}.npz") for tau, name in NAMES.items()}
volume = np.mean([r["volume"][r["times"] >= 10000].mean()
                  for r in runs.values()])
kappa = compressibility_at(volume)
factor = 2 * math.pi * math.sqrt((N_ATOMS + 1) * units.KB * T_LIQUID * kappa
                                 / (3 * volume))
print(f"mean volume {volume:.0f} Å³, κ_T there {kappa / 160.21766:.3f} GPa⁻¹;"
      f" predicted period {factor:.3f} τ_P")
measured = {}
for tau, r in runs.items():
    keep = r["times"] >= 10000
    period = crossings_period(r["times"][keep], r["volume"][keep])
    measured[tau] = period / 1000
    print(f"τ_P = {tau:g} ps: period {period / 1000:.3f} ps, predicted "
          f"{factor * tau:.3f} ps; volume spread {r['volume'][keep].std():.0f}"
          f" Å³")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                                 gridspec_kw=dict(wspace=0.35,
                                                  width_ratios=(1.6, 1)))
for row, tau in enumerate((1.0, 3.0, 10.0)):
    r = runs[tau]
    early = r["times"] <= 30000
    offset = 1.2 * (2 - row)
    swing = (r["volume"][early] - r["volume"][r["times"] >= 10000].mean())
    ax_a.plot(r["times"][early] / 1000, swing / 1000 + offset, color=COLOUR,
              lw=0.7)
    ax_a.text(30.5, offset, rf"$\tau_{{\mathrm P}}$ = {tau:g} ps",
              fontsize=6, va="center")
ax_a.plot([34.5, 34.5], [-0.75, -0.25], color="black", lw=0.8)
ax_a.text(34.0, -0.5, "500 Å$^3$", fontsize=6, va="center", ha="right")
ax_a.set_xlim(0, 36)
ax_a.set_yticks([])
ax_a.set_xlabel("time / ps")
ax_a.set_ylabel(r"$V - \langle V\rangle$, offset")
viz.panel_tag(ax_a, "a")

taus = np.array(sorted(measured))
ax_b.plot(taus, [measured[t] for t in taus], "o", color=COLOUR, ms=4)
grid = np.linspace(0, 11, 50)
ax_b.plot(grid, factor * grid, **REFERENCE_STYLE, lw=1.0)
ax_b.set_xlabel(r"$\tau_{\mathrm P}$ / ps")
ax_b.set_ylabel("period / ps")
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, figure_path("ch14_pressure", "piston.pdf")))
