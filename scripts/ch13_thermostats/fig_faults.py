"""Figure fig:th-faults: thermostat faults and the checks that catch them.

(a) Coupling too strong: the kinetic temperature of liquid argon under
Berendsen with τ = 100 fs, against CSVR with τ = 100 fs (dashed). (b)
Coupling too weak: the running mean of the temperature in three runs
from different starts under Andersen at 0.1 collisions per atom per ps,
against three under CSVR with τ = 100 fs (dashed). (c) The flying ice
cube: the kinetic energy of the centre of mass, started moving,
under Berendsen (τ = 100 fs), against CSVR (dashed), on a logarithmic
axis. (d) One atom, one thermostat: the density of the potential energy
of lithium atoms on the model surface at 1000 K, each with its own
Nosé-Hoover thermostat (teal) or Berendsen thermostat (ochre), against
the exact canonical density (dashed); a chain of three matches it. (e)
The wrong number of freedoms in the target: lithium atoms on the surface
under CSVR told N_f = 3, their kinetic temperature over the two freedoms
they have, against the same with N_f = 2 (dashed). (f) The motion of the
centre of mass counted as diffusion: the mean squared displacement in
liquid argon under Langevin (γ = 0.1/ps), measured in the cell and from
the centre of mass (dashed).

Prints the check values of Table tab:th-faults.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from ch13 import (
    BARRIER,
    DT_LI,
    LI_MASS,
    N_FREE,
    RUNS,
    T_LI,
    canonical_positions,
    diffusion,
    exact_energy_density,
    surface,
)

from mdlab import statmech, thermostats, units, viz
from mdlab.thermostats import CSVR
from mdlab.viz import ACCENT, OCHRE, REFERENCE_STYLE, figure_path

BAD = dict(color=ACCENT, lw=0.6)
GOOD = dict(lw=0.6, **REFERENCE_STYLE)


def temperature(name):
    # Andersen and Langevin move the centre of mass: all 3N freedoms count
    n_free = 768 if name.startswith(("andersen", "langevin")) else N_FREE
    r = np.load(RUNS / f"{name}.npz")
    return r["times"] / 1000, 2 * r["kinetic"] / (n_free * units.KB)


# (a) too strong
ta, berendsen = temperature("berendsen_100_s0")
_, csvr = temperature("csvr_100_s0")
canon = math.sqrt(2 / N_FREE)
for name, series in (("Berendsen 100 fs", berendsen), ("CSVR 100 fs", csvr)):
    rel = series[50:].std() / series[50:].mean()
    print(f"(a) {name}: relative spread of T {rel:.4f}, {rel / canon:.2f} of "
          f"the canonical {canon:.4f}")

# (b) too weak
means = {}
for label, name in (("Andersen 0.1/ps", "andersen_0.1"),
                    ("CSVR 100 fs", "csvr_100")):
    runs = [temperature(f"{name}_s{s}") for s in range(3)]
    values = [series[50:].mean() for _, series in runs]
    means[label] = runs
    print(f"(b) {label}: run means {', '.join(f'{v:.2f}' for v in values)}"
          f" K; mean {np.mean(values):.2f} ± "
          f"{np.std(values, ddof=1) / math.sqrt(3):.2f} K")

# (c) flying ice cube
ice = {}
for name in ("berendsen", "csvr"):
    r = np.load(RUNS / f"ice_{name}.npz")
    p = np.einsum("i,tix->tx", r["masses"], r["velocities"])
    k_com = 0.5 * units.MV2_TO_EV * np.sum(p * p, 1) / r["masses"].sum()
    t_v = r["times"][::5] / 1000
    ice[name] = (t_v, k_com)
    print(f"(c) {name}: growth rate of ln K_com "
          f"{np.polyfit(t_v, np.log(k_com), 1)[0]:+.4f}/ps")

# (d) one atom, one thermostat
bins = np.linspace(0, 1.125 * BARRIER, 46)
exact = exact_energy_density(T_LI, bins)
density = {}
for name in ("li_nh", "li_berendsen", "li_nhc", "li_csvr"):
    u = np.load(RUNS / f"{name}.npz")["potential"]
    hist, _ = np.histogram(u.ravel(), bins=bins, density=True)
    density[name] = hist
    print(f"(d) {name}: largest gap from the exact density "
          f"{np.abs(hist - exact).max() / exact.max():.3f} of its peak")

# (e) the wrong number of freedoms: computed here, 1000 atoms for 20 ps
rng = np.random.default_rng(7)
r0 = canonical_positions(1000, T_LI, rng)
v0 = statmech.thermal_velocities(np.full(1000, LI_MASS), T_LI, rng,
                                 remove_drift=False)[:, None, :2]
wrong_n = {}
for n_free in (3, 2):
    out = thermostats.run(surface, np.array([LI_MASS]), r0, v0, DT_LI, 4000,
                          CSVR(T_LI, 100.0, n_free), np.random.default_rng(8),
                          every=20, keep=())
    t_e = out["times"] / 1000
    temp = out["kinetic"].mean(1) / units.KB  # two freedoms: K = k_BT
    wrong_n[n_free] = (t_e, temp)
    print(f"(e) CSVR told N_f = {n_free}: kinetic temperature over the 2 "
          f"freedoms, last 10 ps, {temp[t_e > 10].mean():.0f} K")

# (f) the centre of mass counted as diffusion
pos = np.load(RUNS / "langevin_0.1_s0.npz")["positions"]
lags = np.arange(0, 41)  # frames every 500 fs
msd = {}
for relative in (False, True):
    x = pos - pos.mean(1, keepdims=True) if relative else pos
    msd[relative] = np.array([np.mean(np.sum((x[k:] - x[:len(x) - k]) ** 2,
                                             -1)) for k in lags])
d_raw = [diffusion(np.load(RUNS / f"langevin_0.1_s{s}.npz")["positions"],
                   500.0, relative=False) * 1e4 for s in range(3)]
d_rel = [diffusion(np.load(RUNS / f"langevin_0.1_s{s}.npz")["positions"],
                   500.0) * 1e4 for s in range(3)]
print(f"(f) D in the cell {np.mean(d_raw):.2f}, from the centre of mass "
      f"{np.mean(d_rel):.2f} × 1e-4 Å²/fs, ratio "
      f"{np.mean(d_raw) / np.mean(d_rel):.2f}")
kt = units.KB * 135.0
mass = 256 * 39.948 * units.MV2_TO_EV
print(f"    the centre of mass diffuses at k_BT/(Mγ) = "
      f"{kt / (mass * 1e-4) * 1e4:.2f} × 1e-4 Å²/fs")
gamma = 1e-4
lag = np.linspace(2000.0, 20000.0, 200)
curve = (lag - (1 - np.exp(-gamma * lag)) / gamma)  # in units of 2k_BT/Mγ
fraction = np.polyfit(lag, curve, 1)[0]
print(f"    over lags of 2 to 20 ps the exact curve rises with {fraction:.2f} "
      f"of its long-time slope: an excess of "
      f"{fraction * kt / (mass * gamma) * 1e4:.2f} × 1e-4 Å²/fs expected; "
      f"measured per run: "
      + ", ".join(f"{a - b:.2f}" for a, b in zip(d_raw, d_rel, strict=True)))

viz.use_style(notebook=False)
fig, axes = plt.subplots(3, 2, figsize=(viz.FULL, 6.3),
                         gridspec_kw=dict(hspace=0.75, wspace=0.4))
ax = axes.ravel()


def title(a, letter, text):
    a.set_title(f"({letter}) {text}", loc="left")


a = ax[0]
title(a, "a", "coupling too strong")
a.plot(ta, csvr, **GOOD)
a.plot(ta, berendsen, **BAD)
a.set_xlim(0, 20)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ / K")

a = ax[1]
title(a, "b", "coupling too weak")
for t_b, series in means["CSVR 100 fs"]:
    a.plot(t_b, np.cumsum(series) / np.arange(1, len(series) + 1), **GOOD)
for t_b, series in means["Andersen 0.1/ps"]:
    a.plot(t_b, np.cumsum(series) / np.arange(1, len(series) + 1), **BAD)
a.set_ylim(125, 145)
a.set_xlabel("time / ps")
a.set_ylabel("running mean of $T$ / K")

a = ax[2]
title(a, "c", "the flying ice cube")
a.semilogy(*ice["csvr"], **GOOD)
a.semilogy(*ice["berendsen"], **BAD)
a.set_xlabel("time / ps")
a.set_ylabel(r"$K$ of the centre / eV")

a = ax[3]
title(a, "d", "one atom, one thermostat")
centres = 0.5 * (bins[1:] + bins[:-1])
a.plot(centres, exact, **GOOD)
a.plot(centres, density["li_nh"], **BAD)
a.plot(centres, np.minimum(density["li_berendsen"], 40), color=OCHRE, lw=0.6)
a.set_ylim(0, 25)
a.set_xlabel(r"$U$ / eV")
a.set_ylabel("density / eV$^{-1}$")

a = ax[4]
title(a, "e", "wrong degrees of freedom")
a.plot(*wrong_n[2], **GOOD)
a.plot(*wrong_n[3], **BAD)
a.set_xlabel("time / ps")
a.set_ylabel(r"$T$ / K")

a = ax[5]
title(a, "f", "centre-of-mass diffusion")
a.plot(lags * 0.5, msd[True], **GOOD)
a.plot(lags * 0.5, msd[False], **BAD)
a.set_xlabel("lag / ps")
a.set_ylabel(r"mean squared displacement / Å$^2$")

print("wrote", viz.save(fig, figure_path("ch13_thermostats", "faults.pdf")))
