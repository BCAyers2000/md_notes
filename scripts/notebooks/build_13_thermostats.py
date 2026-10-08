"""Write notebooks/13_thermostats.ipynb, the companion to Chapter 13.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_13_thermostats.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/13_thermostats.ipynb

The liquid and lithium runs are read from data/ch13_thermostats/runs/,
written by scripts/ch13_thermostats/runs.py (about 10 minutes on 8 cores).
Sections 13.2 to 13.6 each end with a check of that section's thermostat
against ASE's; the notebook also checks against exact results, SymPy, and
the cached runs.
"""

from nbtools import code, hidden, md, solution, write

STEP_BY_STEP = ("the same 32 argon atoms, 50 steps of 5 fs, from the\n"
                "    # same start")
ASE_WHAT = {
    "berendsen": STEP_BY_STEP,
    "andersen": ("32 argon atoms at 120 K, four runs of 100 ps under each\n"
                 "    # code at 10 collisions per ps, compared by the mean\n"
                 "    # temperature and its spread"),
    "langevin": STEP_BY_STEP + " and random numbers",
    "chain": STEP_BY_STEP,
    "csvr": STEP_BY_STEP + " and random numbers",
}


def ase_check(method):
    """The closing cell of a section: its thermostat against ASE's."""
    return code(rf"""
    # Checked against ASE: {ASE_WHAT[method]}
    # (scripts/ch13_thermostats/check_ase.py).
    check_ase.METHODS["{method}"]()
    """)


SETUP = [
    md(r"""
    # Notebook 13: Thermostats

    A thermostat can hold the mean temperature and still sample the wrong
    fluctuations. We compare those two tests first, then examine what
    coupling to a thermostat does to diffusion and barrier crossing.
    The section numbers follow Chapter 13; run the cells in order.

    The longer trajectories are saved in `data/ch13_thermostats/`.
    Sections 13.2–13.6 also compare the implementations with ASE. Matching
    random-number draws allows a step-by-step comparison; Andersen's
    different draws are compared through the temperature distribution.
    """),
    code(r"""
    %matplotlib inline
    import importlib.util
    import math
    import sys
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp

    from mdlab import statmech, thermostats, units, viz
    from mdlab.exercise import check
    from mdlab.thermostats import (
        CSVR,
        Andersen,
        Berendsen,
        Langevin,
        NoseHooverChain,
        Rescale,
        Thermostat,
    )

    sys.path.insert(0, str(Path("..") / "scripts" / "ch13_thermostats"))
    import ch13  # the shared set-up of the chapter's scripts

    # The checks against ASE, loaded from their file: the chapters'
    # script folders each have a check_ase.py.
    _spec = importlib.util.spec_from_file_location(
        "check_ase_13",
        Path("..") / "scripts" / "ch13_thermostats" / "check_ase.py")
    check_ase = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(check_ase)

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    KB = units.KB
    REDUCED = dict(kb=1.0, mv2_to_energy=1.0)
    R0, V0, CELL, MASSES = ch13.liquid_start()
    LIQUID = ch13.argon_model(CELL)
    N_FREE = ch13.N_FREE  # 765
    N_ALL = 3 * len(MASSES)  # collisions move the centre of mass: 768
    CANON = math.sqrt(2 / N_FREE)


    def run(name):
        # Load a saved trajectory from scripts/ch13_thermostats/runs.py.
        return np.load(ch13.RUNS / f"{name}.npz")


    def oscillators(r):
        # Independent oscillators, m = k = 1, one per replica.
        return 0.5 * np.einsum("...ix,...ix->...", r, r), -r
    """),
]

TESTS = [
    md(r"""
    ## 13.1 What a thermostat must do

    2000 independent three-dimensional oscillators in reduced units
    ($k_\mathrm{B}T_0 = 1$), each with its own thermostat, followed for
    300 units of time: the kinetic energy against the canonical gamma
    density of three freedoms, and the effective energy, $K + U$ less the
    heat the thermostat has added.
    """),
    code(r"""
    MAKERS = {
        "rescaling": lambda: Rescale(1.0, 3, **REDUCED),
        "Berendsen, tau 2": lambda: Berendsen(1.0, 2.0, 3, **REDUCED),
        "Andersen, rate 0.5": lambda: Andersen(1.0, 0.5, **REDUCED),
        "Langevin, gamma 0.5": lambda: Langevin(1.0, 0.5, **REDUCED),
        "Nose-Hoover chain, tau 2": lambda: NoseHooverChain(
            1.0, 2.0, 3, **REDUCED),
        "CSVR, tau 2": lambda: CSVR(1.0, 2.0, 3, **REDUCED),
    }


    def canonical_test(method="CSVR, tau 2"):
        rng = np.random.default_rng(0)
        r, v = rng.standard_normal((2000, 1, 3)), rng.standard_normal(
            (2000, 1, 3))
        out = thermostats.run(oscillators, np.ones(1), r, v, 0.05, 6000,
                              MAKERS[method](), np.random.default_rng(1),
                              every=50, keep=(), force_to_accel=1.0)
        k = out["kinetic"][20:].ravel()
        eff = out["potential"] + out["kinetic"] - out["heat"]
        fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), dpi=80)
        a.hist(k, bins=60, range=(0, 8), density=True, alpha=0.4)
        grid = np.linspace(0.01, 8, 300)
        a.plot(grid, statmech.kinetic_energy_density(grid, 3, 1.0, kb=1.0),
               "k--", lw=1)
        a.set_xlabel(r"$K / (k_\mathrm{B}T_0)$")
        a.set_ylabel("probability density")
        b.plot(out["times"], (eff - eff[0])[:, :5], lw=0.8,
               color=viz.ACCENT, alpha=0.65)
        b.set_title("five independent replicas", fontsize=9)
        b.set_xlabel("time")
        b.set_ylabel(r"$[E_\mathrm{eff}(t)-E_\mathrm{eff}(0)]/(k_\mathrm{B}T_0)$")
        fig.subplots_adjust(wspace=0.45)
        plt.show()
        print(f"mean K {k.mean():.3f} (canonical 1.5), variance "
              f"{k.var():.3f} (canonical 1.5); largest change of the "
              f"effective energy {np.abs(eff - eff[0]).max():.1e}")


    widgets.interact(canonical_test,
                     method=widgets.Dropdown(options=list(MAKERS),
                                             value="CSVR, tau 2"));
    """),
    md(r"""
    Every method holds the mean near 1.5. Andersen,
    Langevin, the chain and CSVR give the canonical variance, 1.5, and
    their histograms follow the dashed curve. Rescaling and Berendsen
    coupling squeeze $K$ to a spike: an oscillator with nothing else to
    share energy with is driven into a circular orbit of constant kinetic
    energy, as the lithium atom of Section 13.2 is pinned. The effective
    energy changes by at most about 0.01 in every case, the integration
    error, while the thermostat exchanges heat of order 1.
    """),
]

BERENDSEN = [
    md(r"""
    ## 13.2 Rescaling and weak coupling

    The liquid argon of Chapter 12 under Berendsen coupling for 20 ps with
    the coupling time you choose: the spread of $K$ against the canonical
    spread and against the run at fixed energy. Each run takes about 2 s.
    """),
    code(r"""
    def berendsen(tau_fs=100.0):
        out = thermostats.run(LIQUID, MASSES, R0, V0, 10.0, 2000,
                              Berendsen(135.0, tau_fs, N_FREE), every=10,
                              keep=())
        k = out["kinetic"][20:]
        temp = 2 * out["kinetic"] / (N_FREE * KB)
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        ax.plot(out["times"] / 1000, temp, lw=0.6,
                color=viz.THERMOSTAT["Berendsen"])
        ax.axhline(135 * (1 + CANON), ls=":", color="0.4")
        ax.axhline(135 * (1 - CANON), ls=":", color="0.4")
        ax.set_xlabel("time / ps")
        ax.set_ylabel("T / K")
        plt.show()
        print(f"spread of K {k.std() / k.mean():.4f}; canonical "
              f"{CANON:.4f}; at fixed energy 0.0311; lambda for T = 150 K: "
              f"{math.sqrt(1 + 10 / tau_fs * (135 / 150 - 1)):.5f}")


    widgets.interact(berendsen, tau_fs=widgets.SelectionSlider(
        options=[20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0],
        value=100.0, **SLOW));
    """),
    md(r"""
    Whatever the coupling time, the spread stays
    below the canonical 0.0511 (dotted band): about 0.01 at 20 fs, 0.024
    at 100 fs, rising towards the 0.031 of a run at fixed energy as the
    coupling weakens. Berendsen coupling never reaches the canonical
    spread.
    """),
    code(r"""
    # SymPy: the flying ice cube. With T = T0 + dT, y = dT/T0 and <y> = 0,
    # the mean of lambda^2 - 1 is (dt/tau) <y^2> to second order in y.
    y, a = sp.symbols("y a")  # a = dt/tau
    lam2 = 1 + a * (1 / (1 + y) - 1)
    second = sp.series(lam2 - 1, y, 0, 3).removeO()
    print(sp.expand(second))
    assert sp.expand(second - a * (y**2 - y)) == 0
    for name in ("berendsen", "csvr"):
        r = run(f"ice_{name}")
        p = np.einsum("i,tix->tx", r["masses"], r["velocities"])
        k_com = 0.5 * units.MV2_TO_EV * (p**2).sum(1) / r["masses"].sum()
        slope = np.polyfit(r["times"][::5] / 1000, np.log(k_com), 1)[0]
        print(f"{name}: growth rate of ln K_com {slope:+.4f}/ps")
    k = run("berendsen_100_s0")["kinetic"][50:]
    print(f"predicted for Berendsen: <(dT/T0)^2>/tau = "
          f"{np.mean((k / k.mean() - 1) ** 2) / 0.1:.4f}/ps")
    """),
    ase_check("berendsen"),
]

ANDERSEN = [
    md(r"""
    ## 13.3 Collisions with a bath

    Andersen's thermostat on the liquid, from the cached runs at three
    collision rates: the temperature of each run and its diffusion
    coefficient, measured from the centre of mass.
    """),
    code(r"""
    def andersen(rate="1"):
        temps, ds = [], []
        for s in range(3):
            r = run(f"andersen_{rate}_s{s}")
            temps.append(2 * r["kinetic"][50:].mean() / (N_ALL * KB))
            ds.append(ch13.diffusion(r["positions"], 500.0) * 1e4)
        fixed = np.mean([ch13.diffusion(run(f"nve_s{s}")["positions"],
                                        500.0) * 1e4 for s in range(3)])
        print(f"rate {rate}/ps: run temperatures "
              + ", ".join(f"{t:.2f}" for t in temps)
              + f" K; D {np.mean(ds):.2f} e-4 Å²/fs, {np.mean(ds) / fixed:.2f}"
              f" of fixed energy")


    widgets.interact(andersen, rate=widgets.ToggleButtons(
        options=["0.1", "1", "10"], value="1"));
    """),
    md(r"""
    At 0.1 per ps the motion is barely disturbed
    (0.96 of the diffusion at fixed energy) but the three runs average
    between 130.1 and 133.7 K: the temperature is held loosely. At 1 per
    ps the runs sit near 135 K and diffusion falls to 0.72; at 10 per ps,
    to 0.17.
    """),
    ase_check("andersen"),
]

LANGEVIN = [
    md(r"""
    ## 13.4 Friction and noise

    One velocity under friction and noise, by the exact step
    $v \leftarrow cv + \sqrt{1 - c^2}\,g$ in reduced units
    ($k_\mathrm{B}T/m = 1$): 5000 velocities started at rest.
    """),
    code(r"""
    def ou(gamma=0.5, dt=0.1):
        c = math.exp(-gamma * dt)
        rng = np.random.default_rng(0)
        v = np.zeros(5000)
        times, var = [0.0], [0.0]
        for step in range(1, 401):
            v = c * v + math.sqrt(1 - c * c) * rng.standard_normal(v.size)
            times.append(step * dt)
            var.append(v.var())
        fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), dpi=80)
        a.plot(times, var)
        a.plot(times, 1 - np.exp(-2 * gamma * np.array(times)), "--")
        a.set_xlabel("time")
        a.set_ylabel("variance of v")
        b.hist(v, bins=50, density=True, alpha=0.4)
        u = np.linspace(-4, 4, 200)
        b.plot(u, np.exp(-u * u / 2) / math.sqrt(2 * math.pi), "k--")
        b.set_xlabel("v")
        b.set_ylabel("probability density")
        fig.subplots_adjust(wspace=0.4)
        plt.show()
        print(f"c = {c:.4f}; variance after {times[-1]:.0f} units "
              f"{v.var():.4f} (1 in the long run)")


    widgets.interact(
        ou,
        gamma=widgets.FloatLogSlider(0.5, base=10, min=-2, max=1, **SLOW),
        dt=widgets.FloatSlider(0.1, min=0.01, max=1.0, step=0.01, **SLOW),
    );
    """),
    md(r"""
    From rest the variance climbs as
    $1 - \mathrm{e}^{-2\gamma t}$ (dashed) whatever the step, because the
    O step is exact; it reaches 1, the canonical value, after a few times
    $1/2\gamma$, and the histogram is then the Gaussian. The 400 steps
    cover 400 $\delta t$, from 4 to 400 units of time; when that is
    shorter than $1/2\gamma$, at small friction and short steps, the
    variance has not yet reached 1.
    """),
    code(r"""
    # SymPy: the geometric sum of the kicks gives the exact variance, and
    # the fixed point of the variance is sigma^2/(2 gamma).
    g, h, n, sig, V = sp.symbols("gamma h n sigma V", positive=True)
    dt = sp.Symbol("dt", positive=True)
    c2 = (1 - g * h) ** 2
    total = sig**2 * h * (1 - c2**n) / (1 - c2)
    limit = sp.limit(total.subs(h, dt / n), n, sp.oo)
    print(sp.simplify(limit))
    assert sp.simplify(limit - sig**2 * (1 - sp.exp(-2 * g * dt))
                       / (2 * g)) == 0
    c = sp.exp(-g * dt)
    fixed = sp.solve(sp.Eq(V, c**2 * V + sig**2 / (2 * g) * (1 - c**2)), V)
    print(fixed)
    assert sp.simplify(fixed[0] - sig**2 / (2 * g)) == 0
    """),
    md(r"""
    Lithium atoms on the model surface at 1000 K, each with its own
    Langevin thermostat: 1000 atoms from the canonical distribution, 20 ps,
    counting hops between hollows, against the transition-state rate.
    """),
    code(r"""
    LI0 = ch13.canonical_positions(1000, ch13.T_LI, np.random.default_rng(0))
    LIV = statmech.thermal_velocities(
        np.full(1000, ch13.LI_MASS), ch13.T_LI, np.random.default_rng(1),
        remove_drift=False)[:, None, :2]
    TST = ch13.tst_rate(ch13.T_LI) * 1000


    def hopping(friction_per_ps=1.0):
        out = thermostats.run(ch13.surface, np.array([ch13.LI_MASS]), LI0,
                              LIV, ch13.DT_LI, 4000,
                              Langevin(ch13.T_LI, friction_per_ps / 1000),
                              np.random.default_rng(2), every=2,
                              keep=("positions",))
        hops = ch13.count_hops(out["positions"][:, :, 0, :])
        rate = hops.mean() / 20.0
        print(f"friction {friction_per_ps:g}/ps: {rate:.3f} hops per atom "
              f"per ps ({rate / TST:.2f} of the transition-state rate "
              f"{TST:.3f}/ps)")


    widgets.interact(hopping, friction_per_ps=widgets.SelectionSlider(
        options=[0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        value=1.0, **SLOW));
    """),
    md(r"""
    Up to a few per ps the rate stays near 1 per ps,
    about 0.8 of the transition-state rate, the rest being crossings that
    turn back. Beyond, it falls: about 0.35 per ps at 100 per ps and 0.15
    at 300, falling more and more nearly in proportion to $1/\gamma$, as
    friction turns each crossing into a slow struggle over the bridge.
    """),
    ase_check("langevin"),
]

NOSE = [
    md(r"""
    ## 13.5 A friction that learns

    One oscillator ($m = k = k_\mathrm{B}T_0 = 1$, period $2\pi$) under a
    Nosé-Hoover chain of the length you choose, $\tau_\mathrm{T} = 1$: one
    trajectory's phase portrait, and the time average of $x^2$ along each
    of 100 trajectories started at rest from random positions.
    """),
    code(r"""
    def chain(length=1):
        r0 = np.random.default_rng(3).standard_normal((100, 1, 1))
        th = NoseHooverChain(1.0, 1.0, 1, chain=length, **REDUCED)
        out = thermostats.run(oscillators, np.ones(1), r0, np.zeros_like(r0),
                              0.05, 20000, th, every=10,
                              keep=("positions", "velocities"),
                              force_to_accel=1.0)
        x = out["positions"][200:, :, 0, 0]
        p = out["velocities"][200:, :, 0, 0]
        averages = np.mean(x**2, 0)
        fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), dpi=80)
        a.plot(x[:, 0], p[:, 0], ",", alpha=0.6,
               color=viz.THERMOSTAT["Nosé-Hoover"])
        a.set_aspect("equal")
        a.set_xlabel("x")
        a.set_ylabel("p")
        b.hist(averages, bins=25, range=(0.2, 2.2),
               color=viz.THERMOSTAT["Nosé-Hoover"])
        b.axvline(1.0, ls=":", color="0.4")
        b.set_xlabel("time average of $x^2$")
        b.set_ylabel("number of trajectories")
        fig.subplots_adjust(wspace=0.4)
        plt.show()
        print(f"chain of {length}: per-trajectory <x^2> from "
              f"{averages.min():.2f} to {averages.max():.2f}, standard "
              f"deviation {averages.std():.3f}")


    widgets.interact(chain, length=widgets.IntSlider(1, min=1, max=4,
                                                     **SLOW));
    """),
    md(r"""
    With one thermostat the trajectory fills a band,
    not the plane, and each trajectory keeps its own average of $x^2$,
    spread from about 0.8 to 1.6. With a chain of two or more the plane
    fills, and every trajectory's average lies within about 0.15 of 1.
    """),
    code(r"""
    # SymPy: the density exp(-beta (H + Q xi^2/2)) changes along the
    # Nose-Hoover flow at the rate N_f xi rho, which the divergence -N_f xi
    # requires. One atom in one dimension (N_f = 1).
    q, p, xi, m, k, Q, kT = sp.symbols("q p xi m k Q kT", real=True)
    H = p**2 / (2 * m) + k * q**2 / 2
    rates = {q: p / m, p: -k * q - xi * p, xi: (p**2 / m - kT) / Q}
    divergence = sum(sp.diff(rates[s], s) for s in (q, p, xi))
    log_rho = -(H + Q * xi**2 / 2) / kT
    change = sum(sp.diff(log_rho, s) * rates[s] for s in (q, p, xi))
    print("divergence", divergence, "; d ln(rho)/dt", sp.simplify(change))
    assert sp.simplify(change + divergence) == 0
    """),
    ase_check("chain"),
]

CSVR_CELLS = [
    md(r"""
    ## 13.6 Rescaling that samples

    CSVR on the 2000 oscillators of Section 13.1, acting once every
    `stride` steps over their whole length, as TrajCast's acts once per
    learned stride: the kinetic energy against the canonical gamma
    density.
    """),
    code(r"""
    def csvr(tau=2.0, stride=1):
        rng = np.random.default_rng(0)
        r, v = rng.standard_normal((2000, 1, 3)), rng.standard_normal(
            (2000, 1, 3))
        out = thermostats.run(oscillators, np.ones(1), r, v, 0.05, 6000,
                              CSVR(1.0, tau, 3, stride=stride, **REDUCED),
                              np.random.default_rng(1), every=50, keep=(),
                              force_to_accel=1.0)
        k = out["kinetic"][20:].ravel()
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        ax.hist(k, bins=60, range=(0, 8), density=True, alpha=0.4)
        grid = np.linspace(0.01, 8, 300)
        ax.plot(grid, statmech.kinetic_energy_density(grid, 3, 1.0, kb=1.0),
                "k--", lw=1)
        ax.set_xlabel(r"$K / (k_\mathrm{B}T_0)$")
        ax.set_ylabel("probability density")
        plt.show()
        print(f"tau {tau:g}, stride {stride}: c1 = "
              f"{math.exp(-0.05 * stride / tau):.4f}; mean K {k.mean():.3f}, "
              f"variance {k.var():.3f} (canonical 1.5 and 1.5)")


    widgets.interact(
        csvr,
        tau=widgets.FloatLogSlider(2.0, base=10, min=-1, max=1, **SLOW),
        stride=widgets.IntSlider(1, min=1, max=20, **SLOW),
    );
    """),
    md(r"""
    For every coupling time and stride the mean and
    variance of $K$ stay near 1.5 and the histogram follows the gamma
    density: the thermostat update has the required stationary distribution for any
    step. The intervening integration still has its own step-size error.
    Increasing the stride changes how often the velocities are rescaled.
    """),
    code(r"""
    # SymPy: the mean of lambda^2 over the noise is c1 + (1 - c1) K0/K.
    c1, K, K0, Nf = sp.symbols("c1 K K0 N_f", positive=True)
    c2 = (1 - c1) * K0 / (Nf * K)
    mean_lambda2 = c1 + c2 * (1 + (Nf - 1))  # <R1^2> = 1, <S> = N_f - 1
    print(sp.simplify(mean_lambda2 * K - K))
    assert sp.simplify(mean_lambda2 * K - K - (1 - c1) * (K0 - K)) == 0
    """),
    ase_check("csvr"),
]

COMPARE = [
    md(r"""
    ## 13.7 Which thermostat, and when

    The table of the chapter, recomputed from the cached runs
    (`scripts/ch13_thermostats/compare.py`).
    """),
    code(r"""
    _spec = importlib.util.spec_from_file_location(
        "compare_13",
        Path("..") / "scripts" / "ch13_thermostats" / "compare.py")
    _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
    """),
    md(r"""
    Six thermostats side by side: the histogram of the kinetic temperature
    of liquid argon fills as each run goes on, against the canonical
    density (dashed), with the run time and the relative spread so far as
    a fraction of the canonical $\sqrt{2/N_{\mathrm f}}$. Andersen and
    Langevin move the centre of mass, so their temperatures count all 768
    freedoms.
    """),
    code(r"""
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation

    PANELS = (("fixed energy", "nve_s0", N_FREE),
              ("Berendsen, 0.1 ps", "berendsen_100_s0", N_FREE),
              ("Andersen, 1/ps", "andersen_1_s0", N_ALL),
              ("Langevin, 1/ps", "langevin_1_s0", N_ALL),
              ("chain of 3, 0.1 ps", "nhc_100_s0", N_FREE),
              ("CSVR, 0.1 ps", "csvr_100_s0", N_FREE))
    PER_FRAME = 20  # samples every 100 fs: 2 ps a frame
    panel_t = [2 * run(name)["kinetic"][50:] / (n * KB)
               for _, name, n in PANELS]
    bins = np.linspace(110, 160, 41)
    grid = np.linspace(110, 160, 300)

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.2), dpi=72, sharex=True,
                             sharey=True, gridspec_kw=dict(hspace=0.45))
    bars, labels = [], []
    for a, (title, _, n), t in zip(axes.ravel(), PANELS, panel_t):
        k_per_t = 0.5 * n * KB  # K = (N_f k_B / 2) T
        a.plot(grid, statmech.kinetic_energy_density(k_per_t * grid, n,
                                                     135.0) * k_per_t,
               **viz.REFERENCE_STYLE)
        _, _, patches = a.hist(t[:PER_FRAME], bins=bins, density=True,
                               color=viz.ACCENT, alpha=0.7)
        bars.append(patches)
        labels.append(a.text(0.03, 0.9, "", transform=a.transAxes,
                             fontsize=8))
        a.set_title(title, fontsize=9)
        a.set_ylim(0, 0.2)
    for a in axes[1]:
        a.set_xlabel("T / K")
    for a in axes[:, 0]:
        a.set_ylabel("density / K$^{-1}$")


    def spread(t, n):
        return t.std() / t.mean() / math.sqrt(2 / n)


    def draw(frame):
        upto = PER_FRAME * (frame + 1)
        for patches, label, t, (_, _, n) in zip(bars, labels, panel_t,
                                                PANELS):
            heights, _ = np.histogram(t[:upto], bins=bins, density=True)
            for patch, height in zip(patches, heights):
                patch.set_height(height)
            label.set_text(f"{upto * 0.1:.0f} ps, spread "
                           f"{spread(t[:upto], n):.2f}")


    animation = FuncAnimation(fig, draw, frames=len(panel_t[0]) // PER_FRAME,
                              interval=120)
    plt.close(fig)
    for (title, _, n), t in zip(PANELS, panel_t):
        print(f"{title}: spread over the whole run {spread(t, n):.2f} of "
              f"the canonical")
    HTML(animation.to_jshtml())
    """),
    md(r"""
    Over the first few picoseconds every histogram is
    ragged and its spread far from its final value. By the end, Andersen,
    Langevin, the chain and CSVR fill the dashed density, with 0.92 to
    1.04 of the canonical spread in these single runs, while Berendsen
    coupling holds a spike 0.44 as wide and the run at fixed energy
    settles near 0.6, the narrower spread of Chapter 12's fixed-energy
    ensemble.
    """),
]

TRAJECTORY = [
    md(r"""
    ## 13.8 What the trajectory looks like

    The six faults of the chapter, each against a healthy run (dashed).
    """),
    code(r"""
    def _strong():
        a = run("berendsen_100_s0")
        b = run("csvr_100_s0")
        t = a["times"] / 1000
        ta = 2 * a["kinetic"] / (N_FREE * KB)
        tb = 2 * b["kinetic"] / (N_FREE * KB)
        return t, ta, tb, "T / K", (
            f"spread of T: Berendsen {ta[50:].std() / ta[50:].mean():.4f}, "
            f"CSVR {tb[50:].std() / tb[50:].mean():.4f}, canonical "
            f"{CANON:.4f}")


    def _weak():
        means = []
        for s in range(3):
            r = run(f"andersen_0.1_s{s}")
            means.append(2 * r["kinetic"][50:].mean() / (N_ALL * KB))
        r = run("andersen_0.1_s1")
        t = r["times"] / 1000
        temp = 2 * r["kinetic"] / (N_ALL * KB)
        running = np.cumsum(temp) / np.arange(1, len(temp) + 1)
        return t, running, None, "running mean of T / K", (
            "run means " + ", ".join(f"{x:.2f}" for x in means) + " K")


    def _ice():
        out = {}
        for name in ("berendsen", "csvr"):
            r = run(f"ice_{name}")
            p = np.einsum("i,tix->tx", r["masses"], r["velocities"])
            out[name] = 0.5 * units.MV2_TO_EV * (p**2).sum(1) / r[
                "masses"].sum()
            t = r["times"][::5] / 1000
        return t, out["berendsen"], out["csvr"], "K of the centre / eV", (
            f"Berendsen {out['berendsen'][0]:.3f} -> "
            f"{out['berendsen'][-1]:.3f} eV; CSVR {out['csvr'][-1]:.3f} eV")


    def _one_atom():
        bins = np.linspace(0, 1.125 * ch13.BARRIER, 46)
        exact = ch13.exact_energy_density(ch13.T_LI, bins)
        u = run("li_nh")["potential"]
        hist, _ = np.histogram(u.ravel(), bins=bins, density=True)
        centres = 0.5 * (bins[1:] + bins[:-1])
        return centres, hist, exact, "density / eV$^{-1}$", (
            f"Nosé-Hoover misses the exact density by "
            f"{np.abs(hist - exact).max() / exact.max():.3f} of its peak")


    def _wrong_nf():
        out = {}
        for n_free in (3, 2):
            o = thermostats.run(ch13.surface, np.array([ch13.LI_MASS]), LI0,
                                LIV, ch13.DT_LI, 2000,
                                CSVR(ch13.T_LI, 100.0, n_free),
                                np.random.default_rng(8), every=20, keep=())
            out[n_free] = o["kinetic"].mean(1) / KB
            t = o["times"] / 1000
        return t, out[3], out[2], "T over two freedoms / K", (
            f"told 3: {out[3][20:].mean():.0f} K; told 2: "
            f"{out[2][20:].mean():.0f} K")


    def _com():
        pos = run("langevin_0.1_s0")["positions"]
        lags = np.arange(0, 41)
        msd = {}
        for relative in (False, True):
            x = pos - pos.mean(1, keepdims=True) if relative else pos
            msd[relative] = np.array([np.mean(np.sum(
                (x[k:] - x[:len(x) - k]) ** 2, -1)) for k in lags])
        d_cell = ch13.diffusion(pos, 500.0, relative=False) * 1e4
        d_rel = ch13.diffusion(pos, 500.0) * 1e4
        label = "mean squared displacement / Å²"
        return lags * 0.5, msd[False], msd[True], label, (
            f"D in the cell {d_cell:.2f}, from the centre of mass "
            f"{d_rel:.2f} e-4 Å²/fs")


    FAULTS = {"(a) coupling too strong": _strong,
              "(b) coupling too weak": _weak,
              "(c) the flying ice cube": _ice,
              "(d) one atom, one variable": _one_atom,
              "(e) the wrong freedoms in the target": _wrong_nf,
              "(f) the centre counted as diffusion": _com}


    def fault(name="(a) coupling too strong"):
        x, bad, good, label, note = FAULTS[name]()
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        labels = {
            "a": ("Berendsen", "CSVR"),
            "b": ("Andersen", ""),
            "c": ("Berendsen", "CSVR"),
            "d": ("Nosé-Hoover", "canonical"),
            "e": ("target counts 3", "target counts 2"),
            "f": ("in the cell", "from the centre of mass"),
        }[name[1]]
        ax.plot(x, bad, lw=0.7, label=labels[0])
        if good is not None:
            ax.plot(x, good, "--", lw=0.7, color="0.4", label=labels[1])
        if name.startswith("(c)"):
            ax.set_yscale("log")
        ax.set_xlabel("U / eV" if name.startswith("(d)") else
                      "lag / ps" if name.startswith("(f)") else "time / ps")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        plt.show()
        print(note)


    widgets.interact(fault, name=widgets.Dropdown(options=list(FAULTS)));
    """),
    md(r"""
    (a) Berendsen's band of temperature is under
    half as wide as CSVR's. (b) Under weak Andersen coupling the three
    runs average between 130.1 and 133.7 K, short of the 135 K target.
    (c) The kinetic energy of the centre of mass grows steadily under
    Berendsen coupling and wanders without growing under CSVR. (d) A
    single Nosé-Hoover thermostat on one lithium atom misses the exact
    density by about a quarter of its peak. (e) Told three freedoms, the
    thermostat holds the atoms near 1500 K. (f) The displacement measured
    in the cell grows faster than that measured from the centre of mass,
    by about 1.15 in D in this run (1.25 over the book's three runs,
    Table 13.3).
    """),
]

EXERCISES = [
    md(r"""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises' works
    every exercise in full.

    Run the collapsed answer-key cell to prepare the checks. After
    each exercise a collapsed cell holds a worked solution in code. The
    derivations 13.6, 13.7, 13.13, 13.14, 13.15 and 13.16 are checked by
    SymPy or NumPy; 13.3, 13.9, 13.20 and 13.21 ask for reasoning, answered
    in the book.
    """),
    hidden(r"""
    # Reference values, calculated from the exercise data.
    _kt = KB * 135
    _m_ar = 39.948 * units.MV2_TO_EV
    TARGETS = {
        "13.1": 1 - math.exp(-0.002),
        "13.2": math.sqrt(1 + 1 / 500 * (300 / 290 - 1)),
        "13.4": (2.376 / 1.4941) / 2,
        "13.5": math.exp(5e-4 / 0.1 * 1000),
        "13.8": math.sqrt(1 - math.exp(-0.02)) * math.sqrt(_kt / _m_ar),
        "13.10": math.sqrt(KB * 1000 / (2 * math.pi * 6.94
                                       * units.MV2_TO_EV)),
        "13.11": math.exp(-0.3 / (KB * 300)) / 170.3e-15,
        "13.12": 2 * math.pi * 100 / math.sqrt(2),
        "13.17": np.array([500.0, math.exp(-0.01), 70.0]),
        "13.18": _kt / (256 * _m_ar * 1e-4),
        "13.19": 300 * 576 / 381,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 13.1: the probability of at least one collision in a step,
    # nu = 1/ps, dt = 2 fs.
    collision = None

    check(collision, TARGETS["13.1"], rtol=1e-4, name="probability")
    """),
    solution(r"""
    # SOLUTION 13.1. 1 - e^(-nu dt), with nu dt = 0.002; and the limit.
    collision = 1 - math.exp(-0.002)
    for n in (10, 1000, 100000):
        print(n, 1 - (1 - 0.002 / n) ** n)

    check(collision, TARGETS["13.1"], rtol=1e-4, name="probability");
    """),
    code(r"""
    # EXERCISE 13.2: lambda for dt = 1 fs, tau = 0.5 ps, T = 290 K, T0 = 300 K.
    lam = None

    check(lam, TARGETS["13.2"], rtol=1e-9, name="lambda")
    """),
    solution(r"""
    # SOLUTION 13.2. lambda^2 = 1 + (dt/tau)(T0/T - 1).
    lam = math.sqrt(1 + 1 / 500 * (300 / 290 - 1))

    check(lam, TARGETS["13.2"], rtol=1e-9, name="lambda");
    """),
    code(r"""
    # EXERCISE 13.4: the time constant (ps) of a liquid's temperature under
    # Langevin with gamma = 1/ps, C_V/C_K = 2.376/1.4941.
    tau_langevin = None

    check(tau_langevin, TARGETS["13.4"], rtol=1e-3, name="time constant")
    """),
    solution(r"""
    # SOLUTION 13.4. C_V/(2 gamma C_K).
    tau_langevin = (2.376 / 1.4941) / (2 * 1.0)

    check(tau_langevin, TARGETS["13.4"], rtol=1e-3, name="time constant");
    """),
    code(r"""
    # EXERCISE 13.5: the growth factor of K_com in 1 ns, <(dT/T0)^2> = 5e-4,
    # tau = 0.1 ps.
    growth = None

    check(growth, TARGETS["13.5"], rtol=1e-3, name="growth")
    """),
    solution(r"""
    # SOLUTION 13.5. e^(<(dT/T0)^2> t / tau).
    growth = math.exp(5e-4 * 1000 / 0.1)

    check(growth, TARGETS["13.5"], rtol=1e-3, name="growth");
    """),
    code(r"""
    # EXERCISES 13.6 and 13.7, checked by SymPy: the sum of two Gaussians,
    # and the fixed point of the variance (Section 13.4 checks the latter).
    s, x = sp.symbols("s x", real=True)
    a, b = sp.symbols("a b", positive=True)
    gauss = lambda z, var: sp.exp(-z**2 / (2 * var)) / sp.sqrt(  # noqa: E731
        2 * sp.pi * var)
    density = sp.integrate(gauss(x, a) * gauss(s - x, b), (x, -sp.oo, sp.oo))
    print(sp.simplify(density))
    assert sp.simplify(density - gauss(s, a + b)) == 0
    """),
    code(r"""
    # EXERCISE 13.8: the standard deviation (Å/fs) of the random part of one
    # O step: argon, 135 K, gamma = 1/ps, dt = 10 fs.
    noise = None

    check(noise, TARGETS["13.8"], rtol=1e-3, name="noise")
    """),
    solution(r"""
    # SOLUTION 13.8. sqrt(1 - c^2) sqrt(k_B T/m), c = e^(-gamma dt).
    c = math.exp(-0.01)
    noise = math.sqrt(1 - c * c) * math.sqrt(KB * 135
                                             / (39.948 * units.MV2_TO_EV))

    check(noise, TARGETS["13.8"], rtol=1e-3, name="noise");
    """),
    code(r"""
    # EXERCISE 13.10: sqrt(k_B T/(2 pi m)) for lithium at 1000 K, in Å/fs.
    flux = None

    check(flux, TARGETS["13.10"], rtol=1e-3, name="flux")
    """),
    solution(r"""
    # SOLUTION 13.10. And a numerical check of the integral.
    flux = math.sqrt(KB * 1000 / (2 * math.pi * 6.94 * units.MV2_TO_EV))
    beta_m = 6.94 * units.MV2_TO_EV / (KB * 1000)
    v = np.linspace(0, 0.2, 200001)
    print(np.trapezoid(v * np.exp(-beta_m * v * v / 2), v)
          / math.sqrt(2 * math.pi / beta_m), flux)

    check(flux, TARGETS["13.10"], rtol=1e-3, name="flux");
    """),
    code(r"""
    # EXERCISE 13.11: the one-dimensional transition-state rate (hops per
    # second) for a period of 170.3 fs and a 0.3 eV barrier at 300 K.
    rate_1d = None

    check(rate_1d, TARGETS["13.11"], rtol=1e-3, name="rate")
    """),
    solution(r"""
    # SOLUTION 13.11. (omega/2 pi) e^(-U_b/k_BT) = e^(-U_b/k_BT)/period.
    rate_1d = math.exp(-0.3 / (KB * 300)) / 170.3e-15

    check(rate_1d, TARGETS["13.11"], rtol=1e-3, name="rate");
    """),
    code(r"""
    # EXERCISE 13.12: the period (fs) of the friction's swing, tau = 100 fs.
    swing = None

    check(swing, TARGETS["13.12"], rtol=1e-3, name="period")
    """),
    solution(r"""
    # SOLUTION 13.12. angular frequency sqrt(2)/tau.
    swing = 2 * math.pi * 100 / math.sqrt(2)

    check(swing, TARGETS["13.12"], rtol=1e-3, name="period");
    """),
    code(r"""
    # EXERCISES 13.13 and 13.14, checked by SymPy: Hoover's equations from
    # Nosé's Hamiltonian, for one atom in one dimension (N_f = 1), and the
    # divergence -3 xi of one atom in three dimensions.
    q, pt, s_, ps = sp.symbols("q ptilde s p_s", real=True)
    m, Q, kT = sp.symbols("m Q kT", positive=True)
    U = sp.Function("U")
    HN = pt**2 / (2 * m * s_**2) + U(q) + ps**2 / (2 * Q) + kT * sp.log(s_)
    dq, dpt = sp.diff(HN, pt), -sp.diff(HN, q)
    ds, dps = sp.diff(HN, ps), -sp.diff(HN, s_)
    p = pt / s_  # the real momentum; d/dt = s d/dt'
    dp_dt = s_ * (dpt / s_ - pt * ds / s_**2)
    xi = ps / Q
    assert sp.simplify(s_ * dq - p / m) == 0
    assert sp.simplify(dp_dt - (-sp.diff(U(q), q) - xi * p)) == 0
    assert sp.simplify(s_ * dps / Q - (p**2 / m - kT) / Q) == 0
    xs = sp.symbols("x y z px py pz xi")
    F = sp.symbols("Fx Fy Fz")
    rates = [xs[3] / m, xs[4] / m, xs[5] / m,
             F[0] - xs[6] * xs[3], F[1] - xs[6] * xs[4],
             F[2] - xs[6] * xs[5],
             ((xs[3]**2 + xs[4]**2 + xs[5]**2) / m - 3 * kT) / Q]
    divergence = sum(sp.diff(rate, var) for rate, var in zip(rates, xs))
    print("divergence:", divergence)
    assert sp.simplify(divergence + 3 * xs[6]) == 0
    """),
    code(r"""
    # EXERCISE 13.16, checked numerically (13.15 is checked by SymPy in
    # Section 13.6): a sum of n squares as twice a gamma number of shape n/2.
    rng = np.random.default_rng(4)
    n = 7
    squares = (rng.standard_normal((200000, n)) ** 2).sum(1)
    gamma = 2 * rng.standard_gamma(n / 2, 200000)
    print(f"mean {squares.mean():.3f} and {gamma.mean():.3f} (both {n}); "
          f"variance {squares.var():.2f} and {gamma.var():.2f} "
          f"(both {2 * n})")
    assert abs(squares.mean() - gamma.mean()) < 0.05
    """),
    code(r"""
    # EXERCISE 13.17: TrajCast's default tau (fs) at a 5 fs stride, its c1,
    # and tau (fs) for paracetamol at ten strides of 7 fs.
    trajcast = None  # np.array([tau, c1, tau_paracetamol])

    check(trajcast, TARGETS["13.17"], rtol=1e-4, name="TrajCast")
    """),
    solution(r"""
    # SOLUTION 13.17. A hundred strides; e^(-stride/tau); ten strides.
    trajcast = np.array([100 * 5.0, math.exp(-5.0 / 500.0), 10 * 7.0])

    check(trajcast, TARGETS["13.17"], rtol=1e-4, name="TrajCast");
    """),
    code(r"""
    # EXERCISE 13.18: D of the centre of mass (Å²/fs) for 256 argon atoms,
    # 135 K, gamma = 0.1/ps.
    d_com = None

    check(d_com, TARGETS["13.18"], rtol=1e-3, name="D of the centre")
    """),
    solution(r"""
    # SOLUTION 13.18. k_B T/(M gamma), M in amu times MV2_TO_EV.
    d_com = KB * 135 / (256 * 39.948 * units.MV2_TO_EV * 1e-4)

    check(d_com, TARGETS["13.18"], rtol=1e-3, name="D of the centre");
    """),
    code(r"""
    # EXERCISE 13.19: the temperature (K) of 64 rigid water molecules, 192
    # atoms, under a thermostat that counts 3N = 576 freedoms with a target
    # of 300 K.
    water_t = None

    check(water_t, TARGETS["13.19"], rtol=1e-4, name="temperature")
    """),
    solution(r"""
    # SOLUTION 13.19. 300 x 576/381.
    water_t = 300 * 576 / 381

    check(water_t, TARGETS["13.19"], rtol=1e-4, name="temperature");
    """),
]

NOTES = [md(r"""
    ## Working notes

    Choose a thermostat and coupling strength. Record the mean temperature,
    relative kinetic-energy fluctuation and the change in a dynamical
    quantity. State which measurements this choice supports and which
    would need a weaker coupling or a run at fixed energy.
    """)]

CELLS = (SETUP + TESTS + BERENDSEN + ANDERSEN + LANGEVIN + NOSE + CSVR_CELLS
         + COMPARE + TRAJECTORY + EXERCISES + NOTES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "13_thermostats.ipynb"))
