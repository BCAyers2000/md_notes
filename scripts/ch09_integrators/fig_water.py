"""Figure fig:in-water: sizing the step for the spring model of water.

The model of Section 5.7: springs of 48.48 eV/Å² between the oxygen and
each hydrogen and 17.66 eV/Å² between the hydrogens, at the measured
geometry, whose fastest vibration (the symmetric stretch) has
ω² = 0.7338 fs⁻². Started at the equilibrium geometry with 0.1 eV of
kinetic energy in random velocities, without drift or turning, and
followed by velocity Verlet for 2000 fs. (a) The total energy over the
first 100 fs for three steps. (b) The spread of the total energy, largest
minus smallest, over the run, against δt, with a line of slope 2 and the
stability limit 2/ω (dotted).

Prints the numbers of Section 9.6.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mdlab import integrators, rotation, units, viz
from mdlab.viz import (
    ACCENT,
    REFERENCE_STYLE,
    THRESHOLD_STYLE,
    figure_path,
)

R_OH, ANGLE = 0.958, math.radians(104.4776)
MASSES = np.array([15.999, 1.008, 1.008])
EQUILIBRIUM = np.array(
    [
        [0.0, 0.0, 0.0],
        [R_OH, 0.0, 0.0],
        [R_OH * math.cos(ANGLE), R_OH * math.sin(ANGLE), 0.0],
    ]
)
SPRINGS = [(0, 1, 48.48), (0, 2, 48.48), (1, 2, 17.66)]
NATURAL = {
    (i, j): float(np.linalg.norm(EQUILIBRIUM[j] - EQUILIBRIUM[i]))
    for i, j, _ in SPRINGS
}
OMEGA_MAX = math.sqrt(0.7338)  # rad/fs, from Section 5.7


def energy_forces(r):
    u, f = 0.0, np.zeros_like(r)
    if not np.all(np.isfinite(r)):
        return np.nan, np.full_like(r, np.nan)
    for i, j, k in SPRINGS:
        d = r[j] - r[i]
        dist = np.linalg.norm(d)
        stretch = dist - NATURAL[(i, j)]
        u += 0.5 * k * stretch**2
        pull = k * stretch * d / dist
        f[i] += pull
        f[j] -= pull
    return u, f


rng = np.random.default_rng(5)
v0 = rotation.remove_rigid_motion(MASSES, EQUILIBRIUM, rng.normal(size=(3, 3)))
ke = 0.5 * units.MV2_TO_EV * float(MASSES @ (v0 * v0).sum(1))
v0 = v0 * math.sqrt(0.1 / ke)

limit = 2 / OMEGA_MAX
print(
    f"fastest omega {OMEGA_MAX:.4f} rad/fs, period "
    f"{2 * math.pi / OMEGA_MAX:.3f} fs, stability limit {limit:.3f} fs"
)
steps = np.array([0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 2.2, 2.3])
spreads, runs = [], {}
for dt in steps:
    out = integrators.integrate(
        energy_forces, MASSES, EQUILIBRIUM, v0, dt, int(round(2000 / dt))
    )
    e = out["potential"] + out["kinetic"]
    if dt in (0.25, 0.5, 1.0):
        print(
            f"dt = {dt} fs: std of total / std of kinetic energy "
            f"{np.std(e) / np.std(out['kinetic']):.3f}"
        )
    spreads.append(np.ptp(e))
    runs[dt] = (out["times"], e)
    print(
        f"dt = {dt} fs: energy spread {np.ptp(e):.3e} eV "
        f"({100 * np.ptp(e) / 0.1:.3f}% of 0.1 eV), drift "
        f"{e[-1] - e[0]:+.1e} eV"
    )
spreads = np.array(spreads)
finite = np.isfinite(spreads)
print(f"runs that blew up: {steps[~finite]} fs")
steps_ok, spreads_ok = steps[finite], spreads[finite]
small = steps <= 1.0
slope = np.polyfit(np.log(steps[small]), np.log(spreads[small]), 1)[0]
print(f"slope for dt <= 1 fs: {slope:.2f}")
short = steps <= 0.5
slope = np.polyfit(np.log(steps[short]), np.log(spreads[short]), 1)[0]
print(f"slope for dt <= 0.5 fs: {slope:.2f}")
ratio = spreads[steps == 1.0][0] / spreads[steps == 0.5][0]
print(f"spread at 1 fs / spread at 0.5 fs: {ratio:.2f}")
for energy in (1e-6, 1e-4, 1e-2):
    out = integrators.integrate(
        energy_forces,
        MASSES,
        EQUILIBRIUM,
        v0 * math.sqrt(energy / 0.1),
        2.3,
        int(round(2000 / 2.3)),
    )
    e = out["potential"] + out["kinetic"]
    state = "bounded" if np.isfinite(np.ptp(e)) else "blew up"
    print(f"dt = 2.3 fs with {energy:g} eV: {state}")
for dt in (2.4, 2.5):
    out = integrators.integrate(
        energy_forces, MASSES, EQUILIBRIUM, v0, dt, 400
    )
    e = out["potential"] + out["kinetic"]
    print(f"dt = {dt} fs: energy after {400 * dt:.0f} fs {e[-1]:.3e} eV")

viz.use_style(notebook=False)
fig, (left, right) = plt.subplots(
    1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(wspace=0.4)
)
for dt, colour, style in ((0.25, ACCENT, "-"),
                          (1.0, viz.OCHRE, "--"),
                          (2.0, viz.OXBLOOD, "-.")):
    if dt not in runs:
        out = integrators.integrate(
            energy_forces, MASSES, EQUILIBRIUM, v0, dt, int(round(100 / dt))
        )
        runs[dt] = (out["times"], out["potential"] + out["kinetic"])
    t, e = runs[dt]
    keep = t <= 100
    left.plot(
        t[keep],
        1000 * (e[keep] - 0.1),
        color=colour,
        linestyle=style,
        lw=0.8,
        label=f"{dt:g} fs",
    )
left.set_xlabel("time / fs")
left.set_ylabel(r"$(E - E_0)$ / meV")
left.set_ylim(-3, 48)
left.legend(loc="upper center", fontsize=7, ncol=3)
viz.panel_tag(left, "a")
right.loglog(steps_ok, spreads_ok, "o", color=ACCENT, ms=3.5)
level = np.exp(np.mean(np.log(spreads[small]) - 2 * np.log(steps[small])))
right.loglog(steps_ok, level * steps_ok**2, **REFERENCE_STYLE)
right.axvline(limit, **THRESHOLD_STYLE)
right.set_xlim(0.035, 4.0)
right.set_xlabel(r"step $\delta t$ / fs")
right.set_ylabel("energy spread / eV")
viz.panel_tag(right, "b")

print("wrote", viz.save(fig, figure_path("ch09_integrators", "water.pdf")))
