"""Figure fig:cs-faults: the same water run broken in eight ways.

Each panel sets a broken run (teal) beside the healthy one (grey dashed),
through the quantity that catches the fault (runs.py):

(a) too long a step: flexible water at 2 fs against 0.5 fs, E − E(0);
(b) a cutoff that jumps: rigid water with the Coulomb energy cut off atom
    by atom at 6 Å against the Ewald sum, E − E(0);
(c) wrong units: velocities in Å/fs read as Å per ASE time unit, the
    kinetic energy per freedom against the 12.9 meV asked for (dotted);
(d) wrapped where it must not be: positions wrapped into the cell at every
    step while the bonds are taken without their nearest copies, the
    largest bond error;
(e) overlapping atoms at the start: one molecule placed with its oxygen
    1 Å from another's, the forces on the atoms at the start, largest
    first;
(f) constraints drifting: SHAKE and RATTLE stopped at a tolerance of
    10⁻³ against 10⁻¹⁰, E − E(0);
(g) drift of the centre of mass: starting velocities whose total momentum
    was not removed, the distance the centre of mass has moved;
(h) resonance: RESPA with outer steps of 4 fs against 1 fs, E − E(0).

Prints the check values of Table tab:cs-faults.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch11 import DATA, N_MOLECULES, PER_FREEDOM, load

from mdlab import diagnostics, units, viz, water
from mdlab.viz import ACCENT, REFERENCE_STYLE, THRESHOLD_STYLE, figure_path

RUNS = DATA / "runs"
start = load()
m, h = start["masses"], start["cell"]
bonds, _ = water.constraint_bonds(N_MOLECULES)
RIGID_FREEDOMS = 6 * N_MOLECULES - 3


def run(name):
    return np.load(RUNS / f"{name}.npz")


def energy(r):
    e = r["potential"] + r["kinetic"]
    return e - e[0]


def report(label, name):
    r = run(name)
    t, u, k = r["times"], r["potential"], r["kinetic"]
    line = f"{label} ({name}): {len(t) - 1} steps"
    if len(t) > 3:
        line += (f", std(E)/std(K) {diagnostics.energy_fluctuation(u, k):.4f}"
                 f", drift {1000 * diagnostics.energy_drift(t, u, k):+.3e} "
                 f"eV/ps")
    if "bond_error" in r.files:
        line += f", largest bond error {r['bond_error'].max():.2e} Å"
    print(line)
    return r


healthy = report("healthy rigid 2 fs", "rigid_2")
pairs = [("too long a step", "flexible_0.5", "fault_step"),
         ("jumping cutoff", "rigid_2", "fault_cutoff"),
         ("wrong units", "rigid_2", "fault_units"),
         ("wrapped", "rigid_2", "fault_wrapped"),
         ("overlap", "rigid_2", "fault_overlap"),
         ("tolerance", "rigid_2", "fault_tolerance"),
         ("drift", "rigid_2", "fault_drift"),
         ("resonance", "respa_1", "respa_4")]
for label, good, bad in pairs:
    report(label + ", healthy", good)
    report(label + ", broken", bad)

short = diagnostics.centre_of_mass_path(m, healthy["frames"])
print(f"healthy 2-ps run: centre of mass moves at most {short.max():.1e} Å")
cut = run("fault_cutoff")
leap = energy(cut)[cut["times"] <= 18.0]
print(f"jumping cutoff: E - E(0) reaches {leap.max():.1f} eV by 18 fs; range "
      f"over the run {energy(cut).min():+.1f} to {energy(cut).max():+.1f} "
      f"eV ({np.ptp(energy(cut)):.0f} eV)")
units_run = run("fault_units")
print(f"wrong units: K per freedom at the start "
      f"{1000 * units_run['kinetic'][0] / RIGID_FREEDOMS:.3f} meV against "
      f"{1000 * PER_FREEDOM:.2f} meV asked for "
      f"({units_run['kinetic'][0] / RIGID_FREEDOMS / PER_FREEDOM:.4f}); "
      f"mean over the run "
      f"{1000 * units_run['kinetic'].mean() / RIGID_FREEDOMS:.2f} meV")
print(f"healthy: K per freedom at the start "
      f"{1000 * healthy['kinetic'][0] / RIGID_FREEDOMS:.2f} meV, mean "
      f"{1000 * healthy['kinetic'].mean() / RIGID_FREEDOMS:.2f} meV")
wrapped = run("fault_wrapped")
spread = diagnostics.fractional_spread(start["positions"], h)
print(f"wrapped: bond errors {wrapped['bond_error']} Å at steps 0, 1, 2; "
      f"the start reaches {spread:.3f} cell widths, so molecules lie "
      f"across faces once wrapped")
overlap = run("fault_overlap")
f_bad = overlap["first_forces"]
f_good = run("fault_tolerance")["first_forces"]  # the healthy start's forces
for label, r0, f, u0 in (
    ("healthy", start["positions"], f_good, healthy["potential"][0]),
    ("overlap", overlap["frames"][0], f_bad, overlap["potential"][0]),
):
    print(f"{label} start: closest approach of atoms not bonded "
          f"{diagnostics.closest_approach(r0, h, skip=bonds):.3f} Å, "
          f"largest force {diagnostics.largest_force(f):.3e} eV/Å, "
          f"U {u0:.1f} eV")
loose = run("fault_tolerance")
print(f"loose constraints: total energy changes by "
      f"{energy(loose)[1]:+.4f} eV in the first step")
drift_run = run("fault_drift")
p = drift_run["momentum"]
com = diagnostics.centre_of_mass_path(m, drift_run["frames"])
print(f"drift: total momentum {p.round(4)} amu Å/fs, |P|/M = "
      f"{1000 * np.linalg.norm(p) / m.sum():.4f} Å/ps; centre of mass moved "
      f"{com[-1]:.3f} Å in {drift_run['frame_times'][-1] / 1000:g} ps; "
      f"kinetic energy of the drift "
      f"{0.5 * units.MV2_TO_EV * np.dot(p, p) / m.sum():.4f} eV")

viz.use_style(notebook=False)
fig, axes = plt.subplots(4, 2, figsize=(viz.FULL, 7.4),
                         gridspec_kw=dict(hspace=0.75, wspace=0.42))
ax = axes.ravel()
BAD = dict(color=ACCENT, lw=0.7)
GOOD = dict(lw=0.8, **REFERENCE_STYLE)


def title(a, letter, text):
    a.set_title(f"({letter}) {text}", loc="left")


for a, (good, bad) in zip(ax[[0, 1, 5, 7]], (("flexible_0.5", "fault_step"),
                                              ("rigid_2", "fault_cutoff"),
                                              ("rigid_2", "fault_tolerance"),
                                              ("respa_1", "respa_4")),
                          strict=True):
    g, b = run(good), run(bad)
    a.plot(b["times"] / 1000, energy(b), **BAD)
    a.plot(g["times"] / 1000, energy(g), **GOOD)
    a.set_xlabel("time / ps")
    a.set_ylabel(r"$E - E(0)$ / eV")
title(ax[0], "a", "too long a step")
title(ax[1], "b", "a cutoff that jumps")
title(ax[5], "f", "constraints drifting")
title(ax[7], "h", "resonance")
ax[5].set_ylim(-0.14, 0.03)

a = ax[2]
for r, style in ((units_run, BAD), (healthy, GOOD)):
    a.plot(r["times"] / 1000, 1000 * r["kinetic"] / RIGID_FREEDOMS, **style)
a.axhline(1000 * PER_FREEDOM, **THRESHOLD_STYLE)
a.set_xlabel("time / ps")
a.set_ylabel("$K$ per freedom / meV")
a.set_ylim(0, 16)
title(a, "c", "wrong units")

a = ax[3]
a.semilogy(wrapped["times"], wrapped["bond_error"], "o-", ms=3, **BAD)
a.semilogy(healthy["times"][:6], healthy["bond_error"][:6], "o", ms=3,
           mfc="white", **GOOD)
a.set_xlabel("time / fs")
a.set_ylabel("largest bond error / Å")
a.set_xlim(-0.5, 10)
a.set_ylim(1e-11, 1e3)
title(a, "d", "wrapped bonds")

a = ax[4]
for f, style in ((f_bad, BAD), (f_good, GOOD)):
    sizes = np.sort(np.linalg.norm(f, axis=1))[::-1][:12]
    a.semilogy(np.arange(1, 13), sizes, "o", ms=3, **style)
a.set_xlabel("atom, largest force first")
a.set_ylabel("force / eV Å$^{-1}$")
a.set_ylim(1e-1, 1e7)
title(a, "e", "overlapping atoms")

a = ax[6]
a.plot(drift_run["frame_times"] / 1000, com, **BAD)
a.plot(healthy["frame_times"] / 1000,
       diagnostics.centre_of_mass_path(m, healthy["frames"]), **GOOD)
a.set_xlabel("time / ps")
a.set_ylabel("centre of mass moved / Å")
title(a, "g", "drift of the centre of mass")

print("wrote", viz.save(fig, figure_path("ch11_constraints", "faults.pdf")))
