"""Figure fig:pr-faults: barostat and pressure faults, and their checks.

(a) The pressure of Chapter 13's liquid at fixed volume (csvr_1000_s0,
every 2 ps) from the virial written with pair separations (dashed) and
from Σ rᵢ·Fᵢ with the positions wrapped into the cell (teal). (b) The
volume under cell rescaling handed the virial's pressure without the
kinetic part (teal; runs.py, fault_kinetic_s0), against the healthy run
(dashed, npt_scr_s0). (c) The density of the volume under Berendsen's
barostat (teal, npt_berendsen) against cell rescaling (dashed). (d) The
volume under a piston with τ_P = 10 ps (teal, npt_mtk_pd10_s0) against
τ_P = 1 ps (dashed). (e) The layered solid after its guests went in:
the pressure in the plane and across it under isotropic coupling (teal)
and semi-isotropic coupling (grey). (f) The running mean of the pressure
under Berendsen's barostat with κ entered as 2.5 × 10⁻⁴, the value in
bar⁻¹, where GPa⁻¹ was meant (teal, fault_kappa_s0), against the right
κ (dashed, npt_berendsen_s0).

Prints the check values of Table tab:pr-faults.
"""



import matplotlib.pyplot as plt
import numpy as np
from ch14 import (
    GPA,
    RUNS,
    T_LIQUID,
    argon_model,
    ch13,
    compressibility_at,
    crossings_period,
    isothermal_compressibility,
)

from mdlab import cell, units, viz
from mdlab.virial import kinetic_tensor, wrapped_virial
from mdlab.viz import ACCENT, REFERENCE, REFERENCE_STYLE, figure_path

BAD = dict(color=ACCENT, lw=0.7)
GOOD = dict(lw=0.7, **REFERENCE_STYLE)


def scalar(r, after=10000):
    keep = r["times"] >= after
    return r["volume"][keep], np.trace(r["pressure"][keep], axis1=1,
                                       axis2=2) / 3


# (a) the virial from wrapped positions
run = np.load(ch13.RUNS / "csvr_1000_s0.npz")
h, m = run["cell"], run["masses"]
vol_a = cell.cell_volume(h)
model = argon_model(h)
t_a, pair_p, wrapped_p, shifted_p = [], [], [], []
shift = np.array([0.3, 0.55, 0.8]) * h[0, 0]
for k in range(0, len(run["positions"]), 4):
    x, v = run["positions"][k], run["velocities"][k]
    two_k = np.trace(kinetic_tensor(m, v))
    xw = cell.wrap(x, h)
    _, f = model(xw)
    pair_p.append((two_k + np.trace(model.virial)) / (3 * vol_a))
    wrapped_p.append((two_k + wrapped_virial(xw, f)) / (3 * vol_a))
    xs = cell.wrap(x + shift, h)
    _, fs = model(xs)
    shifted_p.append((two_k + wrapped_virial(xs, fs)) / (3 * vol_a))
    t_a.append(k * 0.5)
pair_p, wrapped_p, shifted_p = (np.array(x) * GPA for x in
                                (pair_p, wrapped_p, shifted_p))
kinetic_part = np.mean([np.trace(kinetic_tensor(m, run["velocities"][k]))
                        for k in range(0, len(run["positions"]), 4)])
print(f"(a) the kinetic part alone, 2K/3V: "
      f"{kinetic_part / (3 * vol_a) * GPA:.4f} GPa")
print(f"(a) pressure from pair separations {pair_p.mean():.4f} GPa; from "
      f"wrapped positions {wrapped_p.mean():.4f} GPa, and with the origin "
      f"moved {shifted_p.mean():.4f} GPa; largest change of one frame's "
      f"value with the origin {np.abs(wrapped_p - shifted_p).max():.3f} GPa")

# (b) the kinetic part dropped
healthy = np.load(RUNS / "npt_scr_s0.npz")
fault_k = np.load(RUNS / "fault_kinetic_s0.npz")
vh, ph = scalar(healthy)
vk, pk = scalar(fault_k)
print(f"(b) without the kinetic part: ⟨V⟩ {vk.mean():.0f} Å³ against "
      f"{vh.mean():.0f}, {vk.mean() / vh.mean() - 1:+.4f}; the full "
      f"pressure there {pk.mean() * GPA:.4f} GPa "
      f"against the target 0.1; Nk_BT/V "
      f"{256 * units.KB * T_LIQUID / vk.mean() * GPA:.4f} GPa")

# (c) Berendsen's narrow volume
kappa_true = compressibility_at(vh.mean())
narrow = [scalar(np.load(RUNS / f"npt_berendsen_s{s}.npz"))[0]
          for s in range(3)]
wide = [scalar(np.load(RUNS / f"npt_scr_s{s}.npz"))[0] for s in range(3)]
k_narrow = np.mean([isothermal_compressibility(v, T_LIQUID) for v in narrow])
k_wide = np.mean([isothermal_compressibility(v, T_LIQUID) for v in wide])
print(f"(c) κ_T from the volume's fluctuations: Berendsen "
      f"{k_narrow / GPA:.2f}, cell rescaling {k_wide / GPA:.2f}, from fixed "
      f"volumes {kappa_true / GPA:.2f} GPa⁻¹")

# (d) a heavy piston
heavy = np.load(RUNS / "npt_mtk_pd10_s0.npz")
light = np.load(RUNS / "npt_mtk_s0.npz")


for label, r in (("τ_P = 10 ps", heavy), ("τ_P = 1 ps", light)):
    keep = r["times"] >= 10000
    period = crossings_period(r["times"][keep], r["volume"][keep])
    print(f"(d) {label}: period {period / 1000:.2f} ps, "
          f"{90000 / period:.0f} swings in 90 ps")

# (e) isotropic coupling of the layered solid
layered = {}
for couple in ("isotropic", "semi-isotropic"):
    r = np.load(RUNS / f"layered_{couple}.npz")
    tensors = r["tensors"][-20:]
    plane = 0.5 * (tensors[:, 0, 0] + tensors[:, 1, 1]).mean() * GPA
    across = tensors[:, 2, 2].mean() * GPA
    layered[couple] = (plane, across)
    print(f"(e) {couple}: in the plane {plane:+.3f} GPa, across "
          f"{across:+.3f} GPa")

# (f) κ in the wrong unit
fault_kappa = np.load(RUNS / "fault_kappa_s0.npz")
right = np.load(RUNS / "npt_berendsen_s0.npz")
for label, r in (("κ in bar⁻¹", fault_kappa), ("κ right", right)):
    v, p = scalar(r, after=0)
    late = r["times"] >= 50000
    p_late = np.trace(r["pressure"][late], axis1=1, axis2=2).mean() / 3
    print(f"(f) {label}: V from {r['volume'][0]:.0f} to "
          f"{r['volume'][-1]:.0f} Å³; mean P after the first 50 ps "
          f"{p_late * GPA:.4f} GPa")

viz.use_style(notebook=False)
fig, axes = plt.subplots(3, 2, figsize=(viz.FULL, 6.3),
                         gridspec_kw=dict(hspace=0.75, wspace=0.4))
ax = axes.ravel()


def title(a, letter, text):
    a.set_title(f"({letter}) {text}", loc="left")


a = ax[0]
title(a, "a", "the virial from wrapped positions")
a.plot(t_a, pair_p, **GOOD)
a.plot(t_a, wrapped_p, **BAD)
a.set_xlabel("time / ps")
a.set_ylabel(r"$P$ / GPa")

a = ax[1]
title(a, "b", "the kinetic part dropped")
a.plot(healthy["times"] / 1000, healthy["volume"] / 1000, **GOOD)
a.plot(fault_k["times"] / 1000, fault_k["volume"] / 1000, **BAD)
a.set_xlim(0, 100)
a.set_xlabel("time / ps")
a.set_ylabel(r"$V$ / 1000 Å$^3$")

a = ax[2]
title(a, "c", "a narrow volume")
bins = np.linspace(11600, 13000, 36)
a.hist(np.concatenate(wide), bins=bins, density=True, histtype="step",
       color=REFERENCE, ls="--", lw=0.8)
a.hist(np.concatenate(narrow), bins=bins, density=True, histtype="step",
       color=ACCENT, lw=0.9)
a.set_xlabel(r"$V$ / Å$^3$")
a.set_ylabel(r"density / Å$^{-3}$")

a = ax[3]
title(a, "d", "a heavy piston")
early = light["times"] <= 100000
a.plot(light["times"][early] / 1000, light["volume"][early] / 1000, **GOOD)
a.plot(heavy["times"] / 1000, heavy["volume"] / 1000, color=ACCENT, lw=1.0)
a.set_xlabel("time / ps")
a.set_ylabel(r"$V$ / 1000 Å$^3$")

a = ax[4]
title(a, "e", "isotropic coupling of layers")
x = np.arange(2)
a.bar(x - 0.18, layered["isotropic"], width=0.34, color=ACCENT)
a.bar(x + 0.18, layered["semi-isotropic"], width=0.34, color=REFERENCE)
a.axhline(0, color="black", lw=0.5)
a.set_xticks(x, ["in the plane", "across"])
a.set_ylabel("pressure / GPa")

a = ax[5]
title(a, "f", r"$\kappa$ in the wrong unit")
for r, style in ((right, GOOD), (fault_kappa, BAD)):
    p = np.trace(r["pressure"], axis1=1, axis2=2) / 3 * GPA
    a.plot(r["times"] / 1000, np.cumsum(p) / np.arange(1, len(p) + 1),
           **style)
a.axhline(0.1, color=REFERENCE, ls=":", lw=0.8)
a.set_xlim(0, 100)
a.set_xlabel("time / ps")
a.set_ylabel("running mean of $P$ / GPa")

print("wrote", viz.save(fig, figure_path("ch14_pressure", "faults.pdf")))
