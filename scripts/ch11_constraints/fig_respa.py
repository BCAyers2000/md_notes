"""Figure fig:cs-respa: what multiple time steps buy for flexible water.

Flexible water of runs.py: the springs inside each molecule are the fast
forces, the Lennard-Jones and Ewald forces between molecules the slow
ones. RESPA takes inner steps of 0.25 fs and outer steps δt; velocity
Verlet (grey dashed) takes every force at every step δt. (a) The standard
deviation of the total energy over that of the kinetic energy against δt,
with the fastest model vibration's period 𝒯 = 7.33 fs marked at 𝒯/4,
𝒯/3 and 𝒯/2, and half the next, 𝒯′ = 8.88 fs (dotted). (b) The same
fluctuation against the calls of the slow forces per picosecond, which
set the cost.

Also measures the time of one call of each part, so that the cost can be
quoted. Prints the numbers of Sections 11.4 and 11.5.
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from ch11 import DATA, load, model

from mdlab import diagnostics, viz
from mdlab.viz import ACCENT, REFERENCE_STYLE, THRESHOLD_STYLE, figure_path

RUNS = DATA / "runs"
FASTEST = 2 * math.pi / 0.8567  # symmetric stretch, numbers_chapter.py
SECOND = 2 * math.pi / 0.7075  # antisymmetric stretch
OUTER = np.arange(0.25, 4.51, 0.25)
VV = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def fluctuation(name):
    run = np.load(RUNS / f"{name}.npz")
    return diagnostics.energy_fluctuation(run["potential"], run["kinetic"])


respa = np.array([fluctuation(f"respa_{dt:g}") for dt in OUTER])
verlet = np.array([fluctuation(f"flexible_{dt:g}") for dt in VV])
print(f"fastest model period {FASTEST:.3f} fs: T/4 {FASTEST / 4:.2f}, "
      f"T/3 {FASTEST / 3:.2f}, T/2 {FASTEST / 2:.2f} fs; next "
      f"{SECOND:.3f} fs, half {SECOND / 2:.2f} fs")
for dt, value in zip(OUTER, respa, strict=True):
    run = np.load(RUNS / f"respa_{dt:g}.npz")
    drift = diagnostics.energy_drift(run["times"], run["potential"],
                                     run["kinetic"])
    print(f"RESPA outer {dt:g} fs: std(E)/std(K) {value:.4f}, drift "
          f"{1000 * drift:+.2f} eV/ps, slow calls per ps {1000 / dt:.0f}")
for dt, value in zip(VV, verlet, strict=True):
    print(f"velocity Verlet {dt:g} fs: std(E)/std(K) {value:.4f}")
def at_level(steps, values, level=0.01):
    """The step at which the fluctuation reaches ``level``.

    Found by straight lines between measured points on logarithmic axes.
    """
    for k in range(len(steps) - 1):
        if values[k] <= level < values[k + 1]:
            w = math.log(level / values[k]) / math.log(values[k + 1]
                                                        / values[k])
            return steps[k] * (steps[k + 1] / steps[k]) ** w
    return float("nan")


for level in (0.01, 0.02):
    a, b = at_level(OUTER, respa, level), at_level(np.array(VV), verlet,
                                                    level)
    print(f"at {level:.0%}: RESPA outer step {a:.2f} fs, velocity Verlet "
          f"{b:.2f} fs; slow calls per ps {1000 / a:.0f} against "
          f"{1000 / b:.0f}, a factor {a / b:.2f}")
growth = dict(zip(OUTER, respa, strict=True))
for a, b in ((1.0, 2.0), (2.0, 3.0)):
    print(f"RESPA fluctuation from {a:g} to {b:g} fs grows by "
          f"{growth[b] / growth[a]:.2f}; as dt^2 it would by "
          f"{(b / a) ** 2:.2f}")
print(f"RESPA at 2 fs with inner steps of 0.125 fs: "
      f"{fluctuation('respa_2_inner0.125'):.4f}")

# the time of one call of each part, in this process alone
start = load()
w = model(start["cell"], flexible=True)
r = start["flexible_positions"]
for part in (w.intramolecular, w.intermolecular):
    part(r)
    t0 = time.perf_counter()
    for _ in range(20):
        part(r)
    print(f"{part.__name__}: {(time.perf_counter() - t0) / 20 * 1e3:.2f} "
          f"ms per call")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.38)
)
left.semilogy(OUTER, respa, "o-", color=ACCENT, ms=3, lw=0.9,
              label="RESPA, inner 0.25 fs")
left.semilogy(VV, verlet, "o", ms=3, lw=0.9, label="velocity Verlet",
              mfc="white", **REFERENCE_STYLE)
for fraction in (4, 3, 2):
    left.axvline(FASTEST / fraction, **THRESHOLD_STYLE)
    left.text(FASTEST / fraction, 7.5, rf"$\mathcal{{T}}/{fraction}$",
              ha="center", fontsize=7)
left.axvline(SECOND / 2, **THRESHOLD_STYLE)
left.text(SECOND / 2 + 0.05, 7.5, r"$\mathcal{T}'/2$", ha="left",
          fontsize=7)
left.set_xlabel(r"step $\delta t$ / fs")
left.set_ylabel(r"std of $E$ / std of $K$")
left.set_xlim(0, 4.6)
left.set_ylim(1e-3, 5)
left.legend(fontsize=7, loc="lower right")
viz.panel_tag(left, "a")

right.loglog(1000 / OUTER, respa, "o-", color=ACCENT, ms=3, lw=0.9)
right.loglog(1000 / np.array(VV), verlet, "o", ms=3, lw=0.9, mfc="white",
             **REFERENCE_STYLE)
right.set_xticks([250, 500, 1000, 2000, 4000])
right.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
right.xaxis.set_minor_formatter(plt.NullFormatter())
right.set_xlabel("slow-force calls per ps")
right.set_ylabel(r"std of $E$ / std of $K$")
right.set_ylim(1e-3, 5)
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch11_constraints", "respa.pdf")))
