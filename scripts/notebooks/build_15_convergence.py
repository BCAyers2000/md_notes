"""Write notebooks/15_convergence.ipynb, the companion to Chapter 15.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_15_convergence.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/15_convergence.ipynb

The runs are read from data/ch15_convergence/runs/, written by
scripts/ch15_convergence/runs.py. The statistics are checked against
pymbar's timeseries module, where it computes the same quantity, and the
minimiser against ASE's FIRE.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 15: Equilibration, convergence and uncertainty

    Work through the uncertainty estimates on a series whose answer is known,
    then use the same checks on the saved liquid runs. Change one setting at
    a time and keep the diagnostic that explains why the estimate moved.
    The section numbers follow Chapter 15; run the cells in order.
    """),
    code(r"""
    %matplotlib inline
    import contextlib
    import io
    import logging
    import math
    import sys
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats as st

    from mdlab import md, potentials, statmech, units, viz
    from mdlab.analysis import stats
    from mdlab.exercise import check

    sys.path.insert(0, str(Path("..") / "scripts" / "ch15_convergence"))
    import ch15  # the shared set-up of the chapter's scripts

    with (contextlib.redirect_stdout(io.StringIO()),
          contextlib.redirect_stderr(io.StringIO())):
        from pymbar import timeseries  # the independent reference
    logging.getLogger("pymbar").setLevel(logging.ERROR)

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    KB = units.KB
    GPA = units.EV_PER_A3_TO_GPA
    DT = ch15.DT  # 10 fs between samples
    RNG = np.random.default_rng(15)


    def run(name):
        # One cached run of scripts/ch15_convergence/runs.py.
        return np.load(ch15.RUNS / f"{name}.npz")
    """),
]

MEAN = [
    md(r"""
    ## 15.1 The error of an average

    Roll a die $n$ times, average, and repeat many times: the means spread
    by $\sigma_x/\sqrt n$, with $\sigma_x = \sqrt{35/12}$, and gather into a
    Gaussian. The interval $\bar x \pm f\,s/\sqrt n$ should hold 3.5 in 95%
    of the trials, with $f = 1.96$ for many rolls and Student's factor for
    few.
    """),
    code(r"""
    def dice(n=4, student=False):
        rolls = RNG.integers(1, 7, (100_000, n))
        means = rolls.mean(axis=1)
        sigma = math.sqrt(35 / 12)
        factor = st.t.ppf(0.975, n - 1) if student and n > 1 else 1.96
        s = rolls.std(axis=1, ddof=1) / math.sqrt(n) if n > 1 else np.nan
        held = np.mean(np.abs(means - 3.5) <= factor * s)
        grid = np.linspace(1, 6, 300)
        sd = sigma / math.sqrt(n)
        fig, ax = plt.subplots(figsize=(6, 3))
        edges = (np.arange(n, 6 * n + 2) - 0.5) / n
        ax.hist(means, bins=edges, density=True, histtype="step",
                color=viz.ACCENT)
        ax.plot(grid, np.exp(-0.5 * ((grid - 3.5) / sd) ** 2)
                / (sd * math.sqrt(2 * math.pi)), **viz.REFERENCE_STYLE)
        ax.set_xlabel("mean of n rolls")
        ax.set_ylabel("density")
        plt.show()
        print(f"spread of the mean {means.std():.4f}; σ/√n {sd:.4f}; "
              f"±{factor:.3f} s/√n holds 3.5 in {held:.3f} of the trials")


    widgets.interact(dice, n=widgets.IntSlider(4, min=2, max=64, **SLOW),
                     student=widgets.Checkbox(False));
    """),
    md(r"""
    The spread of the mean follows $\sigma_x/\sqrt n$
    for every $n$, and the histogram approaches the dashed Gaussian as $n$
    grows. With 1.96 the interval holds 3.5 in about 61% of trials for two
    rolls, 90% for eight and 94% from about thirty; Student's factor raises
    the fractions for few rolls, though not to 95%, since one roll is far
    from Gaussian.
    """),
]

CORRELATION = [
    md(r"""
    ## 15.2 Samples that remember

    The step $x' = cx + \sqrt{1 - c^2}\,z$, with $z$ a fresh standard Gaussian draw, has $\mathrm{corr}(k) = c^k$ and
    $g = (1 + c)/(1 - c)$. Estimate $g$ from one series and compare the
    naive error bar $s/\sqrt n$ with $\sqrt{g s^2/n}$.
    """),
    code(r"""
    def memory(c=0.95, n=10_000):
        x = ch15.ou(n, c, RNG)
        acf = stats.autocorrelation(x, 300)
        g_est = stats.statistical_inefficiency(x)
        exact = (1 + c) / (1 - c)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.plot(np.arange(400) * DT / 1000, x[:400], color=viz.ACCENT,
                 lw=0.8)
        ax1.set_xlabel("time / ps")
        ax1.set_ylabel("x")
        ax2.plot(np.arange(301) * DT, acf, color=viz.ACCENT)
        ax2.plot(np.arange(301) * DT, c ** np.arange(301),
                 **viz.REFERENCE_STYLE)
        ax2.axhline(0, color="black", lw=0.4)
        ax2.set_xlabel("lag / fs")
        ax2.set_ylabel("corr")
        plt.show()
        mean, error, _ = stats.standard_error(x, g_est)
        print(f"g exact {exact:.1f}, estimated {g_est:.1f}; mean {mean:.3f}"
              f" ± {error:.3f} (naive ± "
              f"{x.std(ddof=1) / math.sqrt(n):.4f})")


    widgets.interact(
        memory,
        c=widgets.FloatSlider(0.95, min=0.0, max=0.99, step=0.01, **SLOW),
        n=widgets.IntSlider(10_000, min=1000, max=50_000, step=1000, **SLOW));
    """),
    md(r"""
    The estimate follows $c^k$ until the noise of the
    tail takes over. When the series is long beside its memory, $g$ comes
    out within about a fifth of the exact value, a little high or low from
    draw to draw; with $c$ near 0.99 and a few thousand samples it can come
    out far too low. The naive error bar is $\sqrt g$ times too short at
    every setting, equal to the right one only at $c = 0$.
    """),
    code(r"""
    # The liquid: g and τ_int of U, K and P under CSVR and at fixed energy.
    for label, name in (("CSVR", "long_csvr"), ("fixed energy", "long_nve")):
        r = run(name)
        for q, a in (("U", r["potential"]), ("K", r["kinetic"]),
                     ("P", r["pressure"] * GPA)):
            g = stats.statistical_inefficiency(a)
            print(f"{label:12s} {q}: g {g:6.1f}, τ_int {g * DT / 2000:.3f} ps")
    """),
    code(r"""
    # Compare with pymbar: the same estimate of g, on the liquid's U.
    u = run("long_csvr")["potential"][:20_000]
    ours = stats.statistical_inefficiency(u)
    theirs = timeseries.statistical_inefficiency(u, fast=False)
    print(f"mdlab {ours:.6f}, pymbar {theirs:.6f}, relative difference "
          f"{abs(ours - theirs) / theirs:.1e}")
    """),
]

BLOCKS = [
    md(r"""
    ## 15.3 Block averages

    Cut the series into blocks of $b$ samples, doubling $b$ each time, and
    plot the error of the mean from the block means. It rises to the
    plateau $\sqrt{g\sigma^2/n}$ (dashed) once the blocks outgrow the memory,
    if the run is long enough to have one.
    """),
    code(r"""
    def blocks(c=0.95, log2_n=15):
        n = 2 ** log2_n
        x = ch15.ou(n, c, RNG)
        result = stats.block_average(x)
        k = np.arange(1, n)
        exact = math.sqrt((1 + 2 * np.sum((1 - k / n) * c**k)) / n)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.errorbar(result["length"], result["error"],
                    yerr=result["error_error"], fmt="o", color=viz.ACCENT)
        ax.axhline(exact, **viz.REFERENCE_STYLE)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("block length / samples")
        ax.set_ylabel("error of the mean")
        plt.show()
        g = (1 + c) / (1 - c)
        print(f"{n / g:.0f} independent samples; blocks of 5g = {5 * g:.0f}"
              f" samples; exact error {exact:.4f}")


    widgets.interact(
        blocks,
        c=widgets.FloatSlider(0.95, min=0.5, max=0.99, step=0.01, **SLOW),
        log2_n=widgets.IntSlider(15, min=8, max=18, **SLOW));
    """),
    md(r"""
    Single samples give an error $\sqrt g$ too small.
    With many independent samples the points level off on the dashed line
    once the blocks are a few times $g$ long; with few (a short series, or
    $c$ near 0.99) they are still rising when the blocks run out, and the
    last points scatter: no plateau, and no trustworthy error bar.
    """),
]

EQUILIBRATION = [
    md(r"""
    ## 15.4 Where a run begins

    Add a decaying offset to the series of Section 15.2 and find the start
    that maximises $(n - t_0)/g(t_0)$.
    """),
    code(r"""
    def transient(offset=3.0, decay_ps=2.0, length_ps=50.0):
        n = int(length_ps * 1000 / DT)
        tau = decay_ps * 1000 / DT
        shift = offset * np.exp(-np.arange(n) / tau)
        x = ch15.ou(n, 0.95, RNG) + shift
        t0, g, neff = stats.detect_equilibration(x, nskip=max(1, n // 300))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.arange(n) * DT / 1000, x, color=viz.ACCENT, lw=0.5)
        ax.plot(np.arange(n) * DT / 1000, shift, **viz.REFERENCE_STYLE)
        ax.axvline(t0 * DT / 1000, **viz.THRESHOLD_STYLE)
        ax.set_xlabel("time / ps")
        ax.set_ylabel("x")
        plt.show()
        m_all, e_all, _ = stats.standard_error(x)
        m_cut, e_cut, _ = stats.standard_error(x[t0:])
        print(f"start {t0 * DT / 1000:.2f} ps ({t0 / tau:.1f} decay times), "
              f"{neff:.0f} independent samples after it")
        print(f"mean from the start {m_all:.3f} ± {e_all:.3f}; after "
              f"{m_cut:.3f} ± {e_cut:.3f}; the true mean is 0")


    widgets.interact(
        transient,
        offset=widgets.FloatSlider(3.0, min=0.0, max=10.0, step=0.5, **SLOW),
        decay_ps=widgets.FloatSlider(2.0, min=0.5, max=10.0, step=0.5,
                                     **SLOW),
        length_ps=widgets.FloatSlider(50.0, min=20.0, max=200.0, step=10.0,
                                      **SLOW));
    """),
    md(r"""
    With an offset that outlasts the series' own
    memory, the start usually falls one to a few decay times in, and the
    mean after it sits closer to zero, with a smaller error bar, than the
    mean from the start; from draw to draw it can fall anywhere from a
    fraction of one decay time to more than twenty, sometimes leaving part
    of the offset in place. With no offset the start is usually near zero,
    though a stretch where $g$ comes out small can pull it several
    picoseconds in. A start in the second half of the run means the run is
    too short to place it.
    """),
    code(r"""
    # Eight runs melting from the lattice: U/N and each run's start.
    fig, ax = plt.subplots(figsize=(6, 3))
    for seed in range(8):
        u = run(f"melt_s{seed}")["potential"] / 256
        start = stats.detect_equilibration(u, nskip=20)[0]
        ax.plot(np.arange(len(u)) * DT / 1000, 1000 * u, lw=0.4,
                color=viz.REFERENCE)
        ax.axvline(start * DT / 1000, color=viz.ACCENT, lw=0.6)
    ax.set_xlim(0, 30)
    ax.set_xlabel("time / ps")
    ax.set_ylabel("U/N / meV")
    plt.show()
    """),
    code(r"""
    # Compare with pymbar: the same start on a melting run (every
    # 20th sample, to keep pymbar's search quick).
    u = run("melt_s1")["potential"][::20]
    ours = stats.detect_equilibration(u)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        theirs = timeseries.detect_equilibration(u, fast=False)[0]
    print(f"start: mdlab {ours}, pymbar {theirs} (samples of 200 fs)")
    """),
]

DRIFT = [
    md(r"""
    ## 15.5 A slow drift

    A tyre read every morning for a month: the least-squares slope and its
    error find a leak hidden under the gauge's scatter. Then the liquid at
    fixed energy at each step.
    """),
    code(r"""
    def tyre(leak=0.01, scatter=0.05, days=30):
        t = np.arange(float(days))
        readings = 2.5 - leak * t + scatter * RNG.standard_normal(days)
        slope, error, ratio = stats.drift_test(t, readings, g=1.0)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t, readings, "o", color=viz.ACCENT)
        ax.plot(t, readings.mean() + slope * (t - t.mean()),
                **viz.REFERENCE_STYLE)
        ax.set_xlabel("day")
        ax.set_ylabel("pressure / bar")
        plt.show()
        print(f"slope {slope:.4f} ± {error:.4f} bar/day, {ratio:+.1f} errors")


    widgets.interact(
        tyre,
        leak=widgets.FloatSlider(0.01, min=0.0, max=0.03, step=0.002,
                                 readout_format=".3f", **SLOW),
        scatter=widgets.FloatSlider(0.05, min=0.01, max=0.2, step=0.01,
                                    **SLOW),
        days=widgets.IntSlider(30, min=5, max=90, **SLOW));
    """),
    md(r"""
    With no leak the slope lies within two errors of
    zero in about 95% of draws once there are a few weeks of readings, and
    in about 85% for five days, whose scatter is itself poorly known. A
    leak of 0.01 bar a day shows within a few days of readings for a small
    scatter, and needs weeks for a large one; the error of the slope falls
    as the inverse of the number of days to the power 3/2.
    """),
    code(r"""
    for dt in (5, 10, 20, 30, 35):
        r = run(f"step_{dt}")
        e = r["potential"] + r["kinetic"]
        every = max(1, len(e) // 20000)
        slope, error, ratio = stats.drift_test(r["times"][::every],
                                               e[::every])
        print(f"δt = {dt:2d} fs: drift {1e6 * slope * 1e6 / 256:+9.2f} ± "
              f"{1e6 * error * 1e6 / 256:7.2f} μeV per atom per ns "
              f"({ratio:+.1f} errors)")
    """),
]

OBSERVABLES = [
    md(r"""
    ## 15.6 How long for each quantity

    The run needed to fix a mean to $\pm\sigma^\ast$ is $T \ge
    2\tau_\mathrm{int}\sigma_A^2/\sigma^{\ast 2}$, with $\sigma_A$ and
    $\tau_\mathrm{int}$ measured from a run.
    """),
    code(r"""
    LONG = {"CSVR": run("long_csvr"), "fixed energy": run("long_nve")}


    def run_length(quantity="P (GPa)", thermostat="CSVR", target=0.001):
        r = LONG[thermostat]
        a = {"U/N (eV)": r["potential"] / 256,
             "T (K)": 2 * r["kinetic"] / (ch15.N_FREE * KB),
             "P (GPa)": r["pressure"] * GPA}[quantity]
        g = stats.statistical_inefficiency(a)
        tau = g * DT / 2
        need = 2 * tau * a.var() / target**2
        print(f"σ {a.std():.4g}, τ_int {tau / 1000:.3f} ps: a run of "
              f"{need / 1e6:.3g} ns for ±{target:g}")


    widgets.interact(
        run_length, quantity=["U/N (eV)", "T (K)", "P (GPa)"],
        thermostat=list(LONG),
        target=widgets.FloatLogSlider(0.001, base=10, min=-6, max=0,
                                      **SLOW));
    """),
    md(r"""
    Halving the target quadruples the run, for every
    quantity. At fixed energy the same target needs a 15th to a 38th of
    the time needed under CSVR, since the thermostat lengthens the memory
    of every quantity tied to the energy.
    """),
]

SIZE = [
    md(r"""
    ## 15.7 Results that depend on the size of the box

    The diffusion coefficient from the runs of 256 to 2048 atoms against
    $1/L$, and the straight line of Yeh and Hummer.
    """),
    code(r"""
    def size_line(include_256=True):
        rows = []
        for n in (256, 500, 864, 1372, 2048):
            if n == 256 and not include_256:
                continue
            ds = []
            for name in (f"size_{n}", f"size_{n}_b"):
                if not (ch15.RUNS / f"{name}.npz").exists():
                    continue
                r = run(name)
                pos = r["positions"].astype(float)
                size = (len(pos) - 1) // 10
                ds += [1e4 * ch15.diffusion(pos[j * size:(j + 1) * size + 1],
                                            1000.0) for j in range(10)]
            rows.append((1 / r["cell"][0, 0], np.mean(ds),
                         np.std(ds, ddof=1) / math.sqrt(len(ds))))
        x, d, e = np.array(rows).T
        w = 1 / e**2
        slope, intercept = np.polyfit(x, d, 1, w=np.sqrt(w))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.errorbar(x, d, yerr=e, fmt="o", color=viz.ACCENT)
        grid = np.linspace(0, x.max(), 20)
        ax.plot(grid, intercept + slope * grid, **viz.REFERENCE_STYLE)
        ax.set_xlabel("1/L / Å⁻¹")
        ax.set_ylabel("D / 1e-4 Å² fs⁻¹")
        plt.show()
        print(f"D at infinite size {intercept:.3f}; slope {slope:.2f}")


    widgets.interact(size_line, include_256=widgets.Checkbox(True));
    """),
    md(r"""
    $D$ rises with the box, and the extrapolated
    value lies above every box simulated; leaving out the smallest box
    moves the extrapolation by about its own error.
    """),
]

ENSEMBLE = [
    md(r"""
    ## 15.8 Does the run sample its ensemble?

    The logarithm of the ratio of the energy densities at 140 and 130 K
    must be a straight line in $E$ with the slope $\beta_{130} -
    \beta_{140}$. Its error comes from the block bootstrap.
    """),
    code(r"""
    REQUIRED = 1 / (KB * 130) - 1 / (KB * 140)


    def ratio_slope(e1, e2):
        lo = max(np.percentile(e1, 0.5), np.percentile(e2, 0.5))
        hi = min(np.percentile(e1, 99.5), np.percentile(e2, 99.5))
        edges = np.linspace(lo, hi, 41)
        n1, _ = np.histogram(e1, edges)
        n2, _ = np.histogram(e2, edges)
        ok = (n1 > 10) & (n2 > 10)
        x = 0.5 * (edges[1:] + edges[:-1])[ok]
        y = np.log(n2[ok] / len(e2)) - np.log(n1[ok] / len(e1))
        return np.polyfit(x, y, 1)[0], x, y


    def ensemble(thermostat="csvr"):
        e = []
        for t in (130, 140):
            r = run(f"ensemble_{thermostat}_{t}")
            e.append((r["potential"] + r["kinetic"])[2000:])
        slope, x, y = ratio_slope(*e)
        boot = stats.block_bootstrap(e, lambda a, b: ratio_slope(a, b)[0],
                                     2000, 40, RNG)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, "o", color=viz.ACCENT)
        ax.plot(x, y.mean() + REQUIRED * (x - x.mean()),
                **viz.REFERENCE_STYLE)
        ax.set_xlabel("E / eV")
        ax.set_ylabel("ln P140/P130")
        plt.show()
        print(f"slope {slope:.2f} ± {boot.std():.2f} per eV; required "
              f"{REQUIRED:.3f}")


    widgets.interact(ensemble, thermostat=["csvr", "berendsen"]);
    """),
    md(r"""
    Under CSVR the points follow the dashed slope
    within the error; under Berendsen coupling they climb several times
    too steeply over a narrow range of energy, although the two runs'
    mean energies are right.
    """),
]

PROTOCOL = [
    md(r"""
    ## 15.9 A protocol and a criterion

    The dashboard and the report of `convergence_report` for any recorded
    series: the cached `mdlab` runs, or a run of ASE's own molecular
    dynamics (`ase_bussi`, 20 ps of ASE's Bussi thermostat with its
    Lennard-Jones calculator). Shorten the run to watch the checks fail.
    """),
    code(r"""
    SERIES = {
        "CSVR, U": ("long_csvr", "potential"),
        "CSVR, P": ("long_csvr", "pressure"),
        "fixed energy, U": ("long_nve", "potential"),
        "protocol at 0.1 GPa, V": ("protocol", "npt_volume"),
        "protocol at 0.1 GPa, U": ("protocol", "npt_potential"),
        "melting run, U": ("melt_s0", "potential"),
        "35 fs step, K + U": ("step_35", None),
        "ASE's Bussi run, U": ("ase_bussi", "potential"),
    }


    def criterion(series="protocol at 0.1 GPa, V", fraction=1.0):
        name, key = SERIES[series]
        r = run(name)
        a = r["potential"] + r["kinetic"] if key is None else r[key]
        if key == "pressure":
            a = a * GPA
        a = a[: max(200, int(len(a) * fraction))]
        step = float(r["dt"]) if name.startswith("step") else DT
        report = stats.convergence_report(a, step)
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        axes = axes.ravel()
        ch15.dashboard(axes, a, step, report)
        unit = "Å³" if key == "npt_volume" else "GPa" if key == "pressure" else "eV"
        quantity = "V" if unit == "Å³" else "P" if unit == "GPa" else "E" if key is None else "U"
        axes[0].set_ylabel(f"{quantity} / {unit}")
        axes[1].set_ylabel(f"running mean / {unit}")
        axes[2].set_ylabel(f"standard error / {unit}")
        fig.tight_layout()
        plt.show()
        print(report)


    widgets.interact(criterion, series=list(SERIES),
                     fraction=widgets.FloatSlider(1.0, min=0.02, max=1.0,
                                                  step=0.02, **SLOW));
    """),
    md(r"""
    The long runs pass all four checks. Cut to a few
    per cent of their length, the runs under a thermostat fail the count
    of independent samples; the run at fixed energy, whose memory is ten
    times shorter, keeps enough independent samples down to a few per cent
    of its length and usually passes, and a check it fails there fails by
    chance, as one in twenty should. The
    melting run of 100 ps and ASE's run of 20 ps hold too few independent
    samples of their energy, and the run at 35 fs fails the drift test.
    """),
    code(r"""
    # Compare with ASE: steepest descent and ASE's FIRE reach the same
    # minimum of a disturbed face-centred cubic lattice of 108 atoms.
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.optimize import FIRE

    a0, side = 5.26, 3
    s = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(side) for j in range(side)
                     for k in range(side)])
    h = side * a0 * np.eye(3)
    r0 = ((grid[:, None] + s[None]).reshape(-1, 3) / side) @ h
    r0 = r0 + 0.2 * np.random.default_rng(10).standard_normal(r0.shape)
    pair = potentials.with_cutoff(
        lambda d: potentials.lennard_jones(d, 0.01034, 3.4), 7.0, "shift")
    ours, energies, largest = md.minimise(md.PairModel(pair, h, 7.0, 0.5),
                                          r0, force_tolerance=1e-4)
    atoms = Atoms(f"Ar{len(r0)}", positions=r0, cell=h, pbc=True)
    atoms.calc = LennardJones(sigma=3.4, epsilon=0.01034, rc=7.0,
                              smooth=False)
    FIRE(atoms, logfile=None).run(fmax=1e-4, steps=5000)
    print(f"energy: mdlab {energies[-1]:.6f} eV, ASE "
          f"{atoms.get_potential_energy():.6f} eV; largest difference in "
          f"position {np.abs(ours - atoms.positions).max():.1e} Å")
    """),
]

TRAJECTORY = [
    md(r"""
    ## 15.10 What the trajectory looks like

    Each fault of Section 15.10 against a healthy run, with the check that
    catches it.
    """),
    code(r"""
    LONG_U = run("long_csvr")["potential"]


    def fault(name="(b) the naive error bar"):
        fig, ax = plt.subplots(figsize=(6, 3))
        if name.startswith("(a)"):
            u = run("melt_s7")["potential"] / 256
            start = stats.detect_equilibration(u, nskip=20)[0]
            t = np.arange(len(u)) * DT / 1000
            ax.plot(t, 1000 * np.cumsum(u) / np.arange(1, len(u) + 1),
                    color=viz.ACCENT)
            kept = u[start:]
            ax.plot(t[start:], 1000 * np.cumsum(kept)
                    / np.arange(1, len(kept) + 1), **viz.REFERENCE_STYLE)
            ax.set_ylim(-50.8, -49.6)
            ax.set_ylabel("running mean of U/N / meV")
            check_value = f"start {start * DT / 1000:.1f} ps"
        elif name.startswith("(b)"):
            pieces = LONG_U[:200_000].reshape(20, -1)
            truth = LONG_U.mean()
            naive = [p.std(ddof=1) / math.sqrt(len(p)) for p in pieces]
            right = [stats.standard_error(p)[1] for p in pieces]
            k = np.arange(20)
            ax.errorbar(k, 1000 * (pieces.mean(1) - truth),
                        yerr=1000 * np.array(right), fmt="none",
                        ecolor=viz.REFERENCE)
            ax.errorbar(k, 1000 * (pieces.mean(1) - truth),
                        yerr=1000 * np.array(naive), fmt="o",
                        color=viz.ACCENT)
            held = np.sum(np.abs(pieces.mean(1) - truth) <= naive)
            check_value = f"{held} of 20 naive bars hold the 2 ns mean"
            ax.set_ylabel("U less 2 ns / meV")
        elif name.startswith("(c)"):
            short = LONG_U[:2000]
            b = stats.block_average(short)
            ax.errorbar(b["length"] * DT / 1000, b["error"],
                        yerr=b["error_error"], fmt="o-", color=viz.ACCENT)
            ax.set_xscale("log")
            ax.set_ylabel("error of U / eV")
            g_short = stats.statistical_inefficiency(short)
            check_value = (f"{len(short) / g_short:.1f} independent "
                           "samples")
        elif name.startswith("(d)"):
            for dt, style in ((10, viz.REFERENCE_STYLE),
                              (35, dict(color=viz.ACCENT))):
                r = run(f"step_{dt}")
                e = (r["potential"] + r["kinetic"]) / 256
                ax.plot(r["times"] / 1000, 1000 * (e - e[0]), **style)
            ax.set_ylabel("(E − E0)/N / meV")
            check_value = "drift test: +9.3 errors at 35 fs"
        elif name.startswith("(e)"):
            for kind, style in (("csvr", viz.REFERENCE_STYLE),
                                ("berendsen", dict(color=viz.ACCENT))):
                e = [(run(f"ensemble_{kind}_{t}")["potential"]
                      + run(f"ensemble_{kind}_{t}")["kinetic"])[2000:]
                     for t in (130, 140)]
                slope, x, y = ratio_slope(*e)
                ax.plot(x - x.mean(), y - y.mean(), "o", ms=3, **style)
            ax.set_ylabel("ln P140/P130")
            check_value = f"required slope {REQUIRED:.2f} per eV"
        else:
            for n, style in ((2048, viz.REFERENCE_STYLE),
                             (256, dict(color=viz.ACCENT))):
                pos = run(f"size_{n}")["positions"].astype(float)
                pos = pos - pos.mean(axis=1, keepdims=True)
                lags = np.arange(1, 31)
                value = [np.mean(np.sum((pos[k:] - pos[:-k]) ** 2, -1))
                         / (6 * k * 1000) for k in lags]
                ax.plot(lags, 1e4 * np.array(value), **style)
            ax.set_ylabel("MSD/6t / 1e-4 Å² fs⁻¹")
            check_value = "D at 20 ps: 3.86 (256 atoms) against 4.18 (2048)"
        axis_labels = {"(b)": "piece number", "(c)": "block length / ps",
                       "(e)": "E − mean(E) / eV"}
        ax.set_xlabel(axis_labels.get(name[:3], "time / ps"))
        plt.show()
        print(check_value)


    FAULTS = ["(a) averaged from the start", "(b) the naive error bar",
              "(c) too short for a plateau", "(d) a drifting energy",
              "(e) the wrong ensemble", "(f) a box too small"]
    widgets.interact(fault, name=FAULTS);
    """),
    md(r"""
    (a) The running mean from the first step creeps
    up for the whole run, the one from the start found settles early. (b)
    The naive bars are too short to see and hold none of the means; the
    grey ones hold 11 of the 20, against the 14 that 68% would give,
    within the spread of two. (c) The block errors are still
    rising when the blocks run out. (d) The energy climbs at 35 fs and is
    flat at 10 fs. (e) Berendsen's points climb far more steeply than
    CSVR's. (f) The small box's curve levels off below the large box's.
    """),
]

EXERCISES = [
    md(r"""
    ## Working notes

    Before extending a run, write down the quantity and precision needed.
    Record the discarded time, effective sample count and block plateau
    together; an error bar without those checks is difficult to interpret.
    When a setting changes the conclusion, keep both plots for comparison.
    """),
    md(r"""
    ## Exercises

    Each exercise cell checks its answer against the key below; the
    collapsed cell after it holds a worked solution.
    """),
    hidden(r"""
    # The answer key. Each target is computed from the exercise's data.
    TARGETS = {
        "15.1": (1.96 * math.sqrt(35 / 12) / 0.01) ** 2,
        "15.2": 2.776 / 1.96 - 1,
        "15.4": 45.0,
        "15.5": np.array([199.0, 995.0, -10 / math.log(0.99), 1e5 / 199]),
        "15.8": np.array([300.0, 2**16 // 300]),
        "15.9": 3 * 2 * (math.exp(-5.34 / 2) - math.exp(-25)) / (50 - 5.34),
        "15.11": 20.8 * 10 ** -1.5,
        "15.13": np.array([2 * 940 * 0.01328**2 / 0.0005**2 / 1e6,
                           2 * 123 * 0.009613**2 / 0.0005**2 / 1e6]),
        "15.14": 410.0 - 150.0,
        "15.15": 1 / (KB * 300) - 1 / (KB * 310),
        "15.17": 2.837297 * 0.011633 * 1.602177e-19
        / (6 * math.pi * 2.18e-4 * 23.26e-10) / 1e-5,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 15.1: the rolls of a die for a 95% interval of ±0.01.
    rolls = None

    check(rolls, TARGETS["15.1"], rtol=1e-4, name="rolls")
    """),
    solution(r"""
    # SOLUTION 15.1. n ≥ (1.96 σ/0.01)² with σ² = 35/12.
    rolls = (1.96 * math.sqrt(35 / 12) / 0.01) ** 2

    check(rolls, TARGETS["15.1"], rtol=1e-4, name="rolls");
    """),
    code(r"""
    # EXERCISE 15.2: how much wider Student's interval is for five samples.
    wider = None

    check(wider, TARGETS["15.2"], rtol=1e-3, name="wider")
    """),
    solution(r"""
    # SOLUTION 15.2.
    wider = 2.776 / 1.96 - 1

    check(wider, TARGETS["15.2"], rtol=1e-3, name="wider");
    """),
    code(r"""
    # EXERCISE 15.4: the steps of c = 0.95, from x0 = 0, that bring the
    # variance within 1% of 1.
    steps = None

    check(steps, TARGETS["15.4"], name="steps")
    """),
    solution(r"""
    # SOLUTION 15.4. 1 − var(x_k) = c^(2k) < 0.01, rounded up.
    steps = float(math.ceil(math.log(100) / (2 * math.log(1 / 0.95))))

    check(steps, TARGETS["15.4"], name="steps");
    """),
    code(r"""
    # EXERCISE 15.5: g, τ_int (fs), τ_c (fs) and the independent samples
    # in 1 ns, for c = 0.99 with samples 10 fs apart.
    answers = None

    check(answers, TARGETS["15.5"], rtol=1e-3, name="g, τ_int, τ_c, n_eff")
    """),
    solution(r"""
    # SOLUTION 15.5. g = (1 + c)/(1 − c), τ_int = g δt/2, τ_c = −δt/ln c,
    # and 1 ns holds 10^5 samples.
    c = 0.99
    g = (1 + c) / (1 - c)
    answers = np.array([g, g * 10 / 2, -10 / math.log(c), 1e5 / g])

    check(answers, TARGETS["15.5"], rtol=1e-3, name="g, τ_int, τ_c, n_eff");
    """),
    code(r"""
    # EXERCISE 15.8: the shortest block (samples) and the number of them,
    # for 2^16 samples with g = 60.
    block_and_count = None

    check(block_and_count, TARGETS["15.8"], name="blocks")
    """),
    solution(r"""
    # SOLUTION 15.8. Blocks of 5g samples.
    block = 5 * 60
    block_and_count = np.array([block, 2**16 // block], dtype=float)

    check(block_and_count, TARGETS["15.8"], name="blocks");
    """),
    code(r"""
    # EXERCISE 15.9: the offset left in the mean after the discard.
    left = None

    check(left, TARGETS["15.9"], rtol=1e-3, name="offset left")
    """),
    solution(r"""
    # SOLUTION 15.9. δA0 τ_r (e^(−t0/τ_r) − e^(−T/τ_r))/(T − t0).
    left = 3 * 2 * (math.exp(-5.34 / 2) - math.exp(-50 / 2)) / (50 - 5.34)

    check(left, TARGETS["15.9"], rtol=1e-3, name="offset left");
    """),
    code(r"""
    # EXERCISE 15.11: the drift limit (μeV per atom per ns) for 10 ns.
    limit = None

    check(limit, TARGETS["15.11"], rtol=1e-3, name="limit")
    """),
    solution(r"""
    # SOLUTION 15.11. The limit falls as T^(−3/2).
    limit = 20.8 * 10 ** -1.5

    check(limit, TARGETS["15.11"], rtol=1e-3, name="limit");
    """),
    code(r"""
    # EXERCISE 15.13: the runs (ns) that fix P to ±0.0005 GPa under CSVR
    # and at fixed energy.
    runs_needed = None

    check(runs_needed, TARGETS["15.13"], rtol=1e-3, name="runs")
    """),
    solution(r"""
    # SOLUTION 15.13. T ≥ 2 τ_int σ²/σ*², with τ_int in fs.
    runs_needed = np.array([2 * 940 * 0.01328**2,
                            2 * 123 * 0.009613**2]) / 0.0005**2 / 1e6

    check(runs_needed, TARGETS["15.13"], rtol=1e-3, name="runs");
    """),
    code(r"""
    # EXERCISE 15.14: how much longer (ps) the run of 150 ps must be.
    longer = None

    check(longer, TARGETS["15.14"], name="longer")
    """),
    solution(r"""
    # SOLUTION 15.14. 100 independent samples need 100 × 2τ_int = 400 ps
    # after the start at 10 ps.
    longer = 100 * 2 * 2.0 + 10 - 150

    check(longer, TARGETS["15.14"], name="longer");
    """),
    code(r"""
    # EXERCISE 15.15: the slope (per eV) for 300 and 310 K.
    slope_300 = None

    check(slope_300, TARGETS["15.15"], rtol=1e-4, name="slope")
    """),
    solution(r"""
    # SOLUTION 15.15. β300 − β310.
    slope_300 = 1 / (KB * 300) - 1 / (KB * 310)

    check(slope_300, TARGETS["15.15"], rtol=1e-4, name="slope");
    """),
    code(r"""
    # EXERCISE 15.17: the Yeh-Hummer correction for 256 atoms, in Å²/fs.
    correction = None

    check(correction, TARGETS["15.17"], rtol=2e-3, name="correction")
    """),
    solution(r"""
    # SOLUTION 15.17. 2.837297 k_BT/(6π η_s L) in m²/s, then 1 Å²/fs is
    # 1e-5 m²/s.
    kt_joule = 0.011633 * 1.602177e-19
    correction = (2.837297 * kt_joule
                  / (6 * math.pi * 2.18e-4 * 23.26e-10)) / 1e-5

    check(correction, TARGETS["15.17"], rtol=2e-3, name="correction");
    """),
]

CELLS = (SETUP + MEAN + CORRELATION + BLOCKS + EQUILIBRATION + DRIFT
         + OBSERVABLES + SIZE + ENSEMBLE + PROTOCOL + TRAJECTORY + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "15_convergence.ipynb"))
