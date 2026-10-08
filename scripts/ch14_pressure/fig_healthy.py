"""Figure fig:pr-healthy: a healthy run at constant pressure.

The liquid at 135 K and 0.1 GPa under stochastic cell rescaling (τ_P =
1 ps) and CSVR (τ_T = 1 ps), 300 ps (runs.py, npt_scr_s0). (a) The volume
every 100 fs, with its mean and the spread √(k_BT⟨V⟩κ_T) predicted from
the fixed-volume κ_T either side (dotted). (b) The instantaneous pressure
every 100 fs (thin) and its running mean (thick), against the target
(dotted). (c) The enthalpy-like K + U + P₀V per atom and the effective
energy, that less the heat and the barostat's work, both from their
starting values.

Prints the numbers of Section 14.9's healthy run.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch14 import (
    GPA,
    N_ATOMS,
    P_LIQUID,
    RUNS,
    SIG,
    T_LIQUID,
    compressibility_at,
)

from mdlab import units, viz
from mdlab.viz import ACCENT, OCHRE, THRESHOLD_STYLE, figure_path

r = np.load(RUNS / "npt_scr_s0.npz")
t = r["times"] / 1000
vol = r["volume"]
pressure = np.trace(r["pressure"], axis1=1, axis2=2) / 3
late = t >= 10
mean_v = vol[late].mean()
kappa = compressibility_at(mean_v)
spread = math.sqrt(units.KB * T_LIQUID * mean_v * kappa)
inside = np.mean(np.abs(vol[late] - mean_v) < spread)
density = N_ATOMS / mean_v * SIG**3
print(f"volume after 10 ps: mean {mean_v:.0f} Å³ ({density:.4f} σ⁻³), "
      f"spread {vol[late].std():.0f} Å³ against √(k_BT⟨V⟩κ_T) = "
      f"{spread:.0f} Å³; within it {inside:.3f} of the time")
print(f"pressure after 10 ps: mean {pressure[late].mean() * GPA:.4f} GPa, "
      f"instantaneous spread {pressure[late].std() * GPA:.4f} GPa "
      f"(target {P_LIQUID * GPA:.2f} GPa)")
enthalpy = r["potential"] + r["kinetic"] + P_LIQUID * vol
effective = enthalpy - r["heat"] - r["work"]
per_atom = 1000 / N_ATOMS
print(f"K + U + P0V: range {np.ptp(enthalpy) * per_atom:.2f} meV per atom; "
      f"effective energy: range {np.ptp(effective) * per_atom:.3f} meV per "
      f"atom, drift {np.polyfit(r['times'], effective, 1)[0] * 1000:.1e} "
      f"eV/ps")
temp = 2 * r["kinetic"][late].mean() / ((3 * N_ATOMS - 3) * units.KB)
print(f"kinetic temperature {temp:.2f} K")

viz.use_style(notebook=False)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(viz.FULL, 2.3),
                              gridspec_kw=dict(wspace=0.6))
a.plot(t, vol / 1000, color=ACCENT, lw=0.3)
for level in (-1, 0, 1):
    a.axhline((mean_v + level * spread) / 1000, **THRESHOLD_STYLE)
a.set_xlabel("time / ps")
a.set_ylabel(r"$V$ / 1000 Å$^3$")
viz.panel_tag(a, "a")

b.plot(t, pressure * GPA, color=ACCENT, lw=0.2, alpha=0.5)
b.plot(t, np.cumsum(pressure) / np.arange(1, len(pressure) + 1) * GPA,
       color=ACCENT, lw=1.2)
b.axhline(P_LIQUID * GPA, **THRESHOLD_STYLE)
b.set_xlabel("time / ps")
b.set_ylabel(r"$P$ / GPa")
viz.panel_tag(b, "b")

c.plot(t, (enthalpy - enthalpy[0]) * per_atom, color=OCHRE, lw=0.5,
       label=r"$K + U + P_0V$")
c.plot(t, (effective - effective[0]) * per_atom, color=ACCENT, lw=0.9,
       label="effective")
c.set_xlabel("time / ps")
c.set_ylabel("change / meV per atom")
c.set_ylim(-7, 8)
c.legend(fontsize=6, loc="upper left", ncol=2, handlelength=1.5)
viz.panel_tag(c, "c")

print("wrote", viz.save(fig, figure_path("ch14_pressure", "healthy.pdf")))
