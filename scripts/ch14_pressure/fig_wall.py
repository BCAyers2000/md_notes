"""Figure fig:pr-wall: pressure as the push on the walls.

(a) 200 argon atoms that do not interact, in a cube of 40 Å between soft
walls (runs.py, wall_ideal): the push per unit area on the six walls,
averaged over each picosecond (thin) and over the run so far (thick),
against Nk_BT/V from the run's own kinetic energy (dashed). (b) 256
Lennard-Jones atoms in a cube of 23.26 Å at 135 K (wall_lj, after 20 ps):
the running means of the push per unit area and of the virial pressure
(2K + W)/3V, against Nk_BT/V (dashed).

Prints the numbers of Sections 14.1 and 14.2.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch14 import GPA, RUNS

from mdlab import units, viz
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, figure_path


def wall_pressure(run):
    """The push on all six walls over their area, at every step."""
    side = float(run["length"])
    return run["push"].sum(1) / (6 * side**2)


def running(series):
    return np.cumsum(series) / np.arange(1, len(series) + 1)


def halves(series):
    """The means of the two halves, and of the whole."""
    half = len(series) // 2
    return series[:half].mean(), series[half:].mean(), series.mean()


ideal = np.load(RUNS / "wall_ideal.npz")
side = float(ideal["length"])
n = int(ideal["n"])
volume = side**3
p_wall = wall_pressure(ideal)
temperature = 2 * ideal["kinetic"].mean() / (3 * n * units.KB)
nkt_v = n * units.KB * temperature / volume
first, second, whole = halves(p_wall)
print(f"(a) ideal gas, {n} atoms, L = {side} Å: kinetic temperature "
      f"{temperature:.1f} K; push per area {whole * GPA * 1e4:.2f} bar "
      f"(halves {first * GPA * 1e4:.2f}, {second * GPA * 1e4:.2f}) against "
      f"Nk_BT/V {nkt_v * GPA * 1e4:.2f} bar; ratio {whole / nkt_v:.4f}")
contact = np.mean(ideal["push"] > 0)
print(f"    a given wall is being pushed at {contact:.3f} of the steps")

lj = np.load(RUNS / "wall_lj.npz")
side_lj = float(lj["length"])
n_lj = int(lj["n"])
v_lj = side_lj**3
keep = lj["times"] >= 20000
p_wall_lj = wall_pressure(lj)[keep]
p_vir = (2 * lj["kinetic"][keep] + lj["virial"][keep]) / (3 * v_lj)
t_lj = 2 * lj["kinetic"][keep].mean() / (3 * n_lj * units.KB)
ideal_lj = n_lj * units.KB * t_lj / v_lj
for label, series in (("push per area", p_wall_lj),
                      ("virial pressure", p_vir)):
    a, b, c = halves(series)
    print(f"(b) LJ fluid, {label}: {c * GPA:.4f} GPa (halves {a * GPA:.4f},"
          f" {b * GPA:.4f})")
print(f"    temperature {t_lj:.1f} K; Nk_BT/V {ideal_lj * GPA:.4f} GPa; "
      f"virial part ⟨W⟩/3V {lj['virial'][keep].mean() / (3 * v_lj) * GPA:.4f}"
      f" GPa")

viz.use_style(notebook=False)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(viz.FULL, 2.3),
                                 gridspec_kw=dict(wspace=0.35))
t = ideal["times"] / 1000
per_ps = int(round(1000 / (ideal["times"][1] - ideal["times"][0])))
blocks = len(p_wall) // per_ps
block_mean = p_wall[:blocks * per_ps].reshape(blocks, per_ps).mean(1)
ax_a.plot(np.arange(blocks) + 0.5, block_mean * GPA * 1e4, color=ACCENT,
          lw=0.4, alpha=0.6)
ax_a.plot(t, running(p_wall) * GPA * 1e4, color=ACCENT, lw=1.2)
ax_a.axhline(nkt_v * GPA * 1e4, **REFERENCE_STYLE, lw=1.0)
ax_a.set_xlabel("time / ps")
ax_a.set_ylabel("pressure / bar")
viz.panel_tag(ax_a, "a")

t_b = lj["times"][keep] / 1000
ax_b.plot(t_b, running(p_wall_lj) * GPA, color=ACCENT, lw=1.2,
          label="push per area")
ax_b.plot(t_b, running(p_vir) * GPA, color=OCHRE, lw=1.0,
          label="virial")
ax_b.axhline(ideal_lj * GPA, **REFERENCE_STYLE, lw=1.0)
ax_b.set_xlabel("time / ps")
ax_b.set_ylabel("running mean / GPa")
ax_b.legend(fontsize=6, loc="lower right")
viz.panel_tag(ax_b, "b")

print("wrote", viz.save(fig, figure_path("ch14_pressure", "wall.pdf")))
