"""Figure fig:th-nose: Nosé-Hoover on one oscillator, and a chain.

One oscillator in reduced units (m = k = k_BT = 1, period 2π), steps of
0.05. (a) One trajectory under Nosé-Hoover (τ = 1) over 2000 units of
time, in the plane of x and p: it fills a band, not the plane. (b) The
same under a chain of four (τ = 1). (c) The time average of x² along
each of 100 trajectories, started from random x and zero velocity, under
Nosé-Hoover, the chain and Langevin (γ = 0.5), against the canonical
k_BT = 1 (dotted).

Prints the numbers of Section 13.5.
"""

import matplotlib.pyplot as plt
import numpy as np

from mdlab import thermostats, viz
from mdlab.thermostats import Langevin, NoseHooverChain
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THERMOSTAT, THRESHOLD_STYLE, figure_path

REDUCED = dict(kb=1.0, mv2_to_energy=1.0)


def oscillator(r):
    return 0.5 * np.einsum("...ix,...ix->...", r, r), -r


def follow(thermostat, r, v, steps=40000, seed=1):
    out = thermostats.run(oscillator, np.ones(1), r, v, 0.05, steps,
                          thermostat, np.random.default_rng(seed), every=10,
                          keep=("positions", "velocities"),
                          force_to_accel=1.0)
    return out["positions"][..., 0, 0], out["velocities"][..., 0, 0]


one = np.array([[1.0]]), np.array([[0.0]])
x_nh, p_nh = follow(NoseHooverChain(1.0, 1.0, 1, chain=1, **REDUCED), *one)
x_nhc, p_nhc = follow(NoseHooverChain(1.0, 1.0, 1, chain=4, **REDUCED), *one)
for name, x, p in (("Nose-Hoover", x_nh, p_nh), ("chain of 4", x_nhc, p_nhc)):
    print(f"{name}, one trajectory: <x^2> {np.mean(x**2):.3f}, <p^2> "
          f"{np.mean(p**2):.3f}, <x^4>/<x^2>^2 - 3 "
          f"{np.mean(x**4) / np.mean(x**2) ** 2 - 3:+.3f} (0 for the "
          f"canonical Gaussian)")

rng = np.random.default_rng(3)
r0, v0 = rng.standard_normal((100, 1, 1)), np.zeros((100, 1, 1))
averages = {}
for name, th in (
    ("Nosé-Hoover", NoseHooverChain(1.0, 1.0, 1, chain=1, **REDUCED)),
    ("chain of four", NoseHooverChain(1.0, 1.0, 1, chain=4, **REDUCED)),
    ("Langevin", Langevin(1.0, 0.5, **REDUCED)),
):
    x, _ = follow(th, r0, v0)
    averages[name] = np.mean(x[200:] ** 2, axis=0)
    a = averages[name]
    print(f"{name}: per-trajectory <x^2> mean {a.mean():.3f}, standard "
          f"deviation {a.std():.3f}, range {a.min():.2f} to {a.max():.2f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.3),
                         gridspec_kw=dict(wspace=0.5,
                                          width_ratios=(1, 1, 1.3)))
for a, x, p, tag in ((axes[0], x_nh, p_nh, "a"), (axes[1], x_nhc, p_nhc, "b")):
    a.plot(x, p, ",", color=ACCENT, alpha=0.5)
    a.set_xlim(-4, 4)
    a.set_ylim(-4, 4)
    a.set_aspect("equal")
    a.set_xlabel("$x$")
    a.set_ylabel("$p$")
    viz.panel_tag(a, tag)
c = axes[2]
bins = np.linspace(0.6, 1.8, 25)
for name, values in averages.items():
    colour = THERMOSTAT["Langevin"] if name == "Langevin" else THERMOSTAT["Nosé-Hoover"]
    style = "--" if name == "chain of four" else "-"
    c.hist(values, bins=bins, histtype="step", color=colour, lw=1.0,
           ls=style, label=name)
c.axvline(1.0, **THRESHOLD_STYLE)
c.set_ylim(0, 80)
c.set_xlabel(r"time average of $x^2$")
c.set_ylabel("trajectories")
c.legend(fontsize=6, loc="upper right")
viz.panel_tag(c, "c")

print("wrote", viz.save(fig, figure_path("ch13_thermostats", "nose.pdf")))
