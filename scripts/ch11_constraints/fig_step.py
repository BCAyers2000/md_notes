"""Figure fig:cs-step: how long a step each model of water allows.

The 64-molecule box of runs.py: flexible water by velocity Verlet (the
reference, grey dashed); the O–H bonds held, the angle free; every
distance held (rigid water, RATTLE); and the last two again with the
hydrogen atoms made three times heavier at the oxygen's expense (open
markers). (a) The standard deviation of the total energy over that of the
kinetic energy against the step, with a line of slope 2 and the 1% level
(dotted). (b) The total energy less its starting value for rigid water at
2 and 7 fs and flexible water at 0.5 fs.

Prints the numbers of Sections 11.3 and 11.6.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch11 import DATA

from mdlab import diagnostics, viz
from mdlab.viz import (
    ACCENT,
    OCHRE,
    REFERENCE,
    THRESHOLD_STYLE,
    figure_path,
    tint,
)

RUNS = DATA / "runs"
SERIES = {
    "flexible": ("flexible", (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)),
    "O–H held": ("bonds", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)),
    "O–H held, heavy H": ("bonds_hmr", (1.0, 2.0, 3.0, 4.0, 5.0)),
    "rigid": ("rigid", (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
    "rigid, heavy H": ("rigid_hmr", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                     8.0)),
}
LEVEL = 0.01


def load(prefix, dt):
    return np.load(RUNS / f"{prefix}_{dt:g}.npz")


def fluctuation(run):
    return diagnostics.energy_fluctuation(run["potential"], run["kinetic"])


def at_level(steps, values):
    """The step at which the fluctuation reaches LEVEL.

    Found by straight lines between the measured points on logarithmic
    axes.
    """
    logs, logv = np.log(steps), np.log(values)
    for k in range(len(steps) - 1):
        if values[k] <= LEVEL < values[k + 1]:
            w = (np.log(LEVEL) - logv[k]) / (logv[k + 1] - logv[k])
            return float(np.exp(logs[k] + w * (logs[k + 1] - logs[k])))
    return float("nan")


results = {}
for label, (prefix, steps) in SERIES.items():
    values = np.array([fluctuation(load(prefix, dt)) for dt in steps])
    results[label] = (np.array(steps), values)
    print(f"{label}:")
    for dt, value in zip(steps, values, strict=True):
        run = load(prefix, dt)
        e = run["potential"] + run["kinetic"]
        drift = diagnostics.energy_drift(run["times"], run["potential"],
                                         run["kinetic"])
        line = (f"   dt = {dt:g} fs: std(E)/std(K) {value:.4f}, spread "
                f"{np.ptp(e):.4f} eV ({1000 * np.ptp(e) / 64:.3f} meV per "
                f"molecule), drift {1000 * drift:+.3f} eV/ps")
        if "bond_error" in run.files:
            line += f", bond error {run['bond_error'].max():.1e} Å"
        print(line)
    print(f"   step at {LEVEL:.0%}: {at_level(np.array(steps), values):.2f} "
          f"fs")
    print(f"   mean kinetic energy at the shortest step "
          f"{np.mean(load(prefix, steps[0])['kinetic']):.3f} eV")
base = at_level(*results["flexible"])
for label in SERIES:
    step = at_level(*results[label])
    print(f"gain over flexible at {LEVEL:.0%}: {label} {step / base:.2f}, "
          f"calls of the forces between molecules per ps {1000 / step:.0f}")
for label, (steps, values) in results.items():
    slope = np.polyfit(np.log(steps[:3]), np.log(values[:3]), 1)[0]
    print(f"slope of the fluctuation, {label}, steps {steps[0]:g} to "
          f"{steps[2]:g} fs: {slope:.2f}")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.7), gridspec_kw=dict(wspace=0.38)
)
styles = {
    "flexible": dict(color=REFERENCE, ls="--", marker="o", mfc=REFERENCE),
    "O–H held": dict(color=OCHRE, ls="-", marker="o", mfc=OCHRE),
    "O–H held, heavy H": dict(color=OCHRE, ls=":", marker="o", mfc="white"),
    "rigid": dict(color=ACCENT, ls="-", marker="s", mfc=ACCENT),
    "rigid, heavy H": dict(color=ACCENT, ls=":", marker="s", mfc="white"),
}
for label, (steps, values) in results.items():
    left.loglog(steps, values, ms=3.5, lw=0.9, label=label, **styles[label])
guide = np.array([3.0, 8.0])
left.loglog(guide, 2e-3 * (guide / 3.0) ** 2, color=REFERENCE, lw=0.6)
left.text(5.5, 1.6e-3, "slope 2", fontsize=7, color=REFERENCE)
left.axhline(LEVEL, **THRESHOLD_STYLE)
left.set_xlabel(r"step $\delta t$ / fs")
left.set_ylabel(r"std of $E$ / std of $K$")
left.set_ylim(4e-4, 2)
left.set_xticks([0.25, 0.5, 1, 2, 4, 8])
left.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
left.xaxis.set_minor_formatter(plt.NullFormatter())
left.legend(fontsize=7, loc="upper left", handlelength=2.6)
viz.panel_tag(left, "a")

for prefix, dt, colour, style in (
    ("rigid", 7.0, tint(ACCENT, 0.45), "-"),
    ("flexible", 0.5, REFERENCE, "--"),
    ("rigid", 2.0, ACCENT, "-"),
):
    run = load(prefix, dt)
    e = run["potential"] + run["kinetic"]
    right.plot(run["times"] / 1000, e - e[0], color=colour, ls=style,
               lw=0.7, label=f"{prefix}, {dt:g} fs")
right.set_xlabel("time / ps")
right.set_ylabel(r"$E - E(0)$ / eV")
right.set_xlim(0, 2)
right.legend(fontsize=7, loc="upper left")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch11_constraints", "step.pdf")))
