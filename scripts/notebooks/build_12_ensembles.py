"""Write notebooks/12_ensembles.ipynb, the companion to Chapter 12.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_12_ensembles.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/12_ensembles.ipynb

The argon runs are read from data/ch12_ensembles/runs/, written by
scripts/ch12_ensembles/runs.py (about 3 minutes on 8 cores), and the
animation from renders/ch12_ensembles/exchange.py. Sections 12.5 and 12.6
end with checks of mdlab against ASE; the other sections check against
exact results, SciPy's gamma distribution, or SymPy.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 12: Ensembles

    We connect the probability of a state with what a finite trajectory
    actually measures. Work through the probability examples before
    comparing time averages, velocity distributions and temperature
    fluctuations. The section numbers follow Chapter 12.

    Run the cells in order. The liquid-argon trajectories are saved in
    `data/ch12_ensembles/`. Exact distributions, symbolic calculations and
    ASE's thermal velocities provide separate checks on the results.
    """),
    code(r"""
    %matplotlib inline
    import math
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from ase import Atoms
    from ase.constraints import FixBondLengths
    from ase.md.velocitydistribution import thermalize_momenta
    from IPython.display import Image, display
    from scipy import stats

    from mdlab import chain, diagnostics, statmech, units, viz, water
    from mdlab.exercise import check

    viz.use_style()
    rng = np.random.default_rng(12)
    SLOW = dict(continuous_update=False)  # redraw only on release
    DATA = Path("..") / "data" / "ch12_ensembles"
    KB = units.KB


    def run(name):
        # Load a saved trajectory from scripts/ch12_ensembles/runs.py.
        return np.load(DATA / "runs" / f"{name}.npz")


    LIQUID = run("liquid")
    N_FREE = statmech.degrees_of_freedom(256)
    """),
]

PROBABILITY = [
    md(r"""
    ## 12.1 Probability

    Throw two dice, and draw numbers from the normal distribution, as many
    times as you choose; the fractions settle on the probabilities.
    """),
    code(r"""
    TOTALS = np.arange(2, 13)
    EXACT = (6 - np.abs(TOTALS - 7)) / 36


    def throws(n=1000):
        r = np.random.default_rng(n)
        dice = r.integers(1, 7, (n, 2)).sum(1)
        fraction = np.array([(dice == s).mean() for s in TOTALS])
        x = r.standard_normal(n)
        fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), dpi=80)
        a.bar(TOTALS, EXACT, color="0.85")
        a.plot(TOTALS, fraction, "o")
        a.set_xlabel("total of two dice")
        a.set_ylabel("fraction")
        counts, edges = np.histogram(x, bins=30, range=(-4, 4), density=True)
        b.stairs(counts, edges, fill=True, alpha=0.4)
        u = np.linspace(-4, 4, 200)
        b.plot(u, np.exp(-u * u / 2) / math.sqrt(2 * math.pi), "--")
        b.set_xlabel("$x$")
        b.set_ylabel("probability density")
        plt.show()
        print(f"largest gap between fraction and probability: "
              f"{np.abs(fraction - EXACT).max():.4f}; mean of the totals "
              f"{dice.mean():.3f} (7), variance {dice.var():.3f} (5.833)")


    widgets.interact(
        throws,
        n=widgets.SelectionSlider(
            options=[10, 100, 1000, 10_000, 100_000], value=1000, **SLOW),
    );
    """),
    md(r"""
    The fractions scatter about the probabilities,
    and the largest gap shrinks as the throws grow, roughly as $1/\sqrt n$:
    between 6 and 25 times smaller for a hundred times more throws, the
    scatter of a single set of throws blurring the factor of ten. The histogram
    of normal numbers settles on the Gaussian in the same way, and the
    mean and variance of the totals approach 7 and $35/6$.
    """),
    code(r"""
    # SymPy: a die's mean and variance, and the Gaussian's integral and
    # variance, exactly.
    faces = sp.Range(1, 7)
    mean = sp.Rational(sum(faces), 6)
    variance = sp.Rational(sum(f * f for f in faces), 6) - mean**2
    x, s = sp.symbols("x s", positive=True)
    gauss = sp.exp(-x**2 / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
    total = sp.integrate(gauss, (x, -sp.oo, sp.oo))
    var_g = sp.integrate(x**2 * gauss, (x, -sp.oo, sp.oo))
    print(mean, variance, total, var_g)
    assert (mean, variance, total, var_g) == (sp.Rational(7, 2),
                                             sp.Rational(35, 12), 1, s**2)
    print([round(math.erf(k / math.sqrt(2)), 4) for k in (1, 2, 3)])
    """),
]

ERGODIC = [
    md(r"""
    ## 12.2 Many states, one system

    The chain of 32 atoms, started with its energy, 0.0724, in the first
    normal mode, followed for 10 000 units of time with steps of 0.05. The
    coupling $\alpha$ is yours to set; each run takes about a second.
    """),
    code(r"""
    PATTERNS, OMEGA = chain.normal_modes(32)


    def fput(alpha=0.25):
        r = chain.run(4.0 * PATTERNS[:, 0], np.zeros(32), alpha, 0.05,
                      200_000, every=200)
        modes, t = r["modes"], r["times"]
        share = modes / modes.sum(1)[:, None]
        fig, ax = plt.subplots(figsize=(7, 3), dpi=80)
        for k in range(4):
            ax.plot(t, share[:, k], lw=0.8, label=f"mode {k + 1}")
        ax.plot(t, share[:, 4:].sum(1), "k", lw=0.8, label="modes 5-32")
        ax.set_xlabel("time")
        ax.set_ylabel("share of the energy")
        ax.legend(ncol=5, fontsize=8, loc="upper center")
        ax.set_ylim(0, 1.15)
        plt.show()
        print(f"mode 1 falls to {share[:, 0].min():.3f}; modes 5-32 reach "
              f"{share[:, 4:].sum(1).max():.3f} and average "
              f"{share[:, 4:].sum(1).mean():.3f} (equal sharing 0.875); "
              f"energy kept to {np.ptp(r['energy']) / r['energy'][0]:.1e}")


    widgets.interact(
        fput,
        alpha=widgets.FloatSlider(0.25, min=0.0, max=1.0, step=0.05, **SLOW),
    );
    """),
    md(r"""
    With $\alpha = 0$ the first mode keeps all the
    energy for ever. A small coupling passes the energy among the first
    few modes and back, and modes 5 to 32 never hold more than a fifth of
    it while $\alpha \le 0.25$. Larger couplings spread it further: at
    $\alpha = 1$ the high modes hold up to 0.83 of the energy at times,
    but only 0.47 on average over the run, still short of the 0.875 of
    equal sharing.
    """),
    md(r"""
    Liquid argon instead: the running time average of one atom's kinetic
    energy, divided by the average over all atoms and times. Choose the
    atom.
    """),
    code(r"""
    _per_atom = 0.5 * units.MV2_TO_EV * LIQUID["masses"][None, :] * (
        LIQUID["velocities"] ** 2).sum(-1)
    _running = np.cumsum(_per_atom, 0) / np.arange(1, len(_per_atom) + 1)[
        :, None] / _per_atom.mean()


    def atom_average(atom=0):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        ax.plot(LIQUID["frame_times"] / 1000, _running[:, atom], lw=0.8)
        ax.axhline(1.0, ls=":", color="0.4")
        ax.set_xlabel("time / ps")
        ax.set_ylabel("running average / all-atom mean")
        ax.set_ylim(0, 2.5)
        plt.show()
        print(f"atom {atom} after 50 ps: {_running[-1, atom]:.3f}; all "
              f"atoms from {_running[-1].min():.3f} to "
              f"{_running[-1].max():.3f}")


    widgets.interact(atom_average,
                     atom=widgets.IntSlider(0, min=0, max=255, **SLOW));
    """),
    md(r"""
    Every atom's average wanders over the first few
    picoseconds and then settles towards 1, and after 50 ps all 256 lie
    between 0.807 and 1.249: the averages are approaching a common value, as expected if the atoms
    explore the same distribution. Their remaining spread shows that
    50 ps is not yet long enough for close agreement; it does not by
    itself establish ergodicity.
    """),
    code(r"""
    # Checked against the exact result: the squares of the mode frequencies
    # are the eigenvalues of the matrix with 2 on its diagonal and -1 beside
    # it (ASE has no such chain).
    for n in (3, 32):
        matrix = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        _, omega = chain.normal_modes(n)
        assert np.allclose(np.sort(omega**2), np.linalg.eigvalsh(matrix))
    period = 2 * math.pi / omega[-1]
    print("normal_modes agrees with the eigenvalues for N = 3 and 32;",
          f"period of the fastest mode for N = 32: {period:.3f}")
    """),
]

MICRO = [
    md(r"""
    ## 12.3 Counting states at fixed energy

    Two gases of $N$ atoms each share a fixed energy; the probability
    that the first holds the fraction $x$, from the full count
    $x^{3N/2 - 1}(1 - x)^{3N/2 - 1}$, against the Gaussian of standard
    deviation $1/\sqrt{12N}$ from the Taylor series.
    """),
    code(r"""
    def sharing(n=100):
        x = np.linspace(1e-6, 1 - 1e-6, 4001)
        log = (1.5 * n - 1) * (np.log(x) + np.log(1 - x))
        p = np.exp(log - log.max())
        p /= np.trapezoid(p, x)
        spread = math.sqrt(np.trapezoid((x - 0.5) ** 2 * p, x))
        formula = 1 / math.sqrt(12 * n)
        g = np.exp(-((x - 0.5) ** 2) / (2 * formula**2)) / (
            formula * math.sqrt(2 * math.pi))
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        ax.plot(x, p, label="exact")
        ax.plot(x, g, "--", label="Gaussian, $1/\\sqrt{12N}$")
        ax.set_xlabel("share $x$")
        ax.set_ylabel("density")
        ax.legend()
        plt.show()
        print(f"N = {n}: standard deviation of the curve {spread:.4f}, "
              f"formula {formula:.4f}")


    widgets.interact(
        sharing,
        n=widgets.SelectionSlider(options=[1, 3, 10, 30, 100, 1000],
                                  value=100, **SLOW),
    );
    """),
    md(r"""
    The curve narrows as $N$ grows. For one atom in
    each gas its spread is 0.250, against 0.289 from the formula, because
    the formula keeps only the large-$N$ count and the curvature at the
    peak; by $N = 100$ they agree to 0.5% (0.0288 against 0.0289), and the
    Gaussian lies on the curve.
    """),
    code(r"""
    # SymPy: the curvature of ln P at x = 1/2 is -12N.
    x, n = sp.symbols("x N", positive=True)
    log_p = sp.Rational(3, 2) * n * (sp.log(x) + sp.log(1 - x))
    curvature = sp.simplify(sp.diff(log_p, x, 2).subs(x, sp.Rational(1, 2)))
    print(curvature)
    assert sp.simplify(curvature + 12 * n) == 0
    """),
]

CANONICAL = [
    md(r"""
    ## 12.4 A system and its bath

    One oscillator, the system, shares 200 quanta of energy with 99 more,
    the bath, by random moves of one quantum at a time. The histogram of
    the system's quanta builds up against the bath's count of states
    $\Omega_{\mathrm B}(Q - n)$ and the Boltzmann factor.
    """),
    code(r"""
    display(Image(filename=str(DATA / "exchange.gif")))
    """),
    md(r"""
    The bars of the bath change at every frame, and
    the system's histogram settles on the dashed curve: the probability
    of holding $n$ quanta is proportional to the number of ways the bath
    can hold the rest. The dotted Boltzmann factor lies almost on it,
    0.333 against 0.331 for an empty system, because the bath is a hundred
    times larger than the system.
    """),
    md(r"""
    The odds of a barrier: choose its height and the temperature.
    """),
    code(r"""
    def odds(barrier=0.3, temperature=300.0):
        t = np.linspace(50, 1500, 400)
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        ax.semilogy(t, statmech.boltzmann_factor(barrier, t))
        ax.plot(temperature, statmech.boltzmann_factor(barrier, temperature),
                "o")
        ax.set_xlabel("T / K")
        ax.set_ylabel("odds")
        ax.set_ylim(1e-40, 1)
        plt.show()
        here = statmech.boltzmann_factor(barrier, temperature)
        double = statmech.boltzmann_factor(barrier, 2 * temperature)
        print(f"k_B T = {KB * temperature:.5f} eV; odds {here:.3e}; at "
              f"twice the temperature {double:.3e}, {double / here:.3g} "
              f"times larger")


    widgets.interact(
        odds,
        barrier=widgets.FloatSlider(0.3, min=0.05, max=1.0, step=0.05,
                                    **SLOW),
        temperature=widgets.FloatSlider(300, min=50, max=1500, step=10,
                                        **SLOW),
    );
    """),
    md(r"""
    The odds rise steeply with temperature, and the
    more steeply the higher the barrier: doubling the temperature
    multiplies the odds by $\mathrm{e}^{\Delta E/2k_\mathrm{B} T}$, 331 for the
    0.3 eV barrier at 300 K. A 1 eV barrier at room temperature has odds
    near $10^{-17}$, and is essentially never reached in a run of
    nanoseconds.
    """),
    code(r"""
    # SymPy: the spring's partition function gives <E> = k_B T and the
    # variance (k_B T)^2; a two-level system's <E> and C_V. ASE has no
    # partition functions, so the reference is the exact derivative.
    beta, w, eps, kb, temp = sp.symbols("beta omega epsilon k_B T",
                                        positive=True)
    z_spring = 2 * sp.pi / (beta * w)
    mean_e = -sp.diff(sp.log(z_spring), beta)
    var_e = sp.diff(sp.log(z_spring), beta, 2)
    print(sp.simplify(mean_e), sp.simplify(var_e))
    assert sp.simplify(mean_e - 1 / beta) == 0
    assert sp.simplify(var_e - 1 / beta**2) == 0
    z_two = 1 + sp.exp(-eps / (kb * temp))
    e_two = sp.simplify(kb * temp**2 * sp.diff(sp.log(z_two), temp))
    cv_two = sp.diff(e_two, temp)
    assert sp.simplify(e_two - eps / (sp.exp(eps / (kb * temp)) + 1)) == 0
    print("two levels: <E> =", e_two)
    print("C_V/k_B at eps = 0.1 eV, 300 K:",
          float((cv_two / kb).subs({eps: 0.1, kb: KB, temp: 300})))
    """),
]

MAXWELL = [
    md(r"""
    ## 12.5 The distribution of velocities

    Draw velocities at a temperature for atoms of one species with
    `statmech.thermal_velocities`, and compare them with the
    Maxwell-Boltzmann densities.
    """),
    code(r"""
    MASSES = {"H": 1.008, "Li": 6.94, "C": 12.011, "O": 15.999,
              "Ar": 39.948}


    def velocities(species="Ar", temperature=300.0, atoms=1000):
        m = np.full(atoms, MASSES[species])
        v = statmech.thermal_velocities(m, temperature,
                                        np.random.default_rng(atoms),
                                        remove_drift=False)
        sigma = math.sqrt(KB * temperature / (m[0] * units.MV2_TO_EV))
        speeds = np.linalg.norm(v, axis=1)
        fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), dpi=80)
        a.hist(v.ravel(), bins=40, density=True, alpha=0.4)
        u = np.linspace(-5 * sigma, 5 * sigma, 200)
        a.plot(u, np.exp(-u * u / (2 * sigma**2))
               / (sigma * math.sqrt(2 * math.pi)), "--")
        a.set_xlabel("velocity component / Å fs$^{-1}$")
        a.set_ylabel("density / fs Å$^{-1}$")
        b.hist(speeds, bins=40, density=True, alpha=0.4)
        s = np.linspace(0, 5 * sigma, 200)
        b.plot(s, statmech.maxwell_speed_density(s, m[0], temperature), "--")
        b.set_xlabel("speed / Å fs$^{-1}$")
        b.set_ylabel("density / fs Å$^{-1}$")
        fig.subplots_adjust(wspace=0.4)
        plt.show()
        print(f"sigma: drawn {v.std():.5f}, sqrt(k_B T/m) {sigma:.5f} Å/fs; "
              f"mean speed {speeds.mean():.5f}, sqrt(8/pi) sigma "
              f"{math.sqrt(8 / math.pi) * sigma:.5f}; excess kurtosis "
              f"{diagnostics.excess_kurtosis(m, v):+.3f}")


    widgets.interact(
        velocities,
        species=widgets.Dropdown(options=list(MASSES), value="Ar"),
        temperature=widgets.FloatSlider(300, min=10, max=1500, step=10,
                                        **SLOW),
        atoms=widgets.SelectionSlider(options=[100, 1000, 10_000],
                                      value=1000, **SLOW),
    );
    """),
    md(r"""
    At every temperature and mass the histograms
    follow the curves, which widen as $\sqrt{T/m}$: hydrogen's spread is
    $\sqrt{39.948/1.008} = 6.3$ times argon's. The drawn spread is within
    about 10% of $\sqrt{k_\mathrm{B} T/m}$ for 100 atoms and within 1% for 10 000,
    and the excess kurtosis scatters about zero, by less for more atoms.
    """),
    code(r"""
    # SymPy: the speed density peaks at sqrt(2) sigma and has the mean
    # sqrt(8/pi) sigma.
    v, sig = sp.symbols("v sigma", positive=True)
    density = (4 * sp.pi * v**2 * (2 * sp.pi * sig**2) ** sp.Rational(-3, 2)
               * sp.exp(-v**2 / (2 * sig**2)))
    assert sp.integrate(density, (v, 0, sp.oo)) == 1
    peak = sp.solve(sp.diff(density, v), v)
    mean_v = sp.simplify(sp.integrate(v * density, (v, 0, sp.oo)))
    print(peak, mean_v)
    assert peak == [sp.sqrt(2) * sig]
    assert sp.simplify(mean_v - sp.sqrt(8 / sp.pi) * sig) == 0
    """),
    code(r"""
    # Checked against ASE: thermalize_momenta draws the same normal numbers
    # in the same order, so the same seed gives the same velocities, up to
    # the two libraries' values of Boltzmann's constant.
    m = np.random.default_rng(0).uniform(1, 40, 50)
    ours = statmech.thermal_velocities(m, 300.0, np.random.default_rng(7),
                                       remove_drift=False)
    atoms = Atoms("H50", positions=np.zeros((50, 3)), masses=m)
    thermalize_momenta(atoms, 300.0, rng=np.random.default_rng(7))
    theirs = atoms.get_velocities() * math.sqrt(units.FORCE_TO_ACCEL)
    from ase import units as ase_units

    print(f"largest relative difference {np.abs(ours / theirs - 1).max():.1e};"
          f" half the relative difference of k_B, "
          f"{0.5 * (KB / ase_units.kB - 1):.1e}")
    assert np.abs(ours / theirs - 1).max() < 1e-6
    """),
]

EQUIPARTITION = [
    md(r"""
    ## 12.6 Equal shares and the kinetic temperature

    Count the freedoms of a run.
    """),
    code(r"""
    def freedoms(atoms=256, held=0, drift_removed=True, isolated=False,
                 linear=False):
        n_free = statmech.degrees_of_freedom(atoms, held, drift_removed)
        if isolated:
            n_free -= 2 if linear else 3
        if n_free < 1:
            print("more conditions than components of velocity")
            return
        print(f"N_f = {n_free}; ASE's count, 3N less the held distances: "
              f"{3 * atoms - held}; its temperature reads "
              f"{n_free / (3 * atoms - held):.6f} of mdlab's")


    widgets.interact(
        freedoms,
        atoms=widgets.IntSlider(256, min=2, max=1000, **SLOW),
        held=widgets.IntSlider(0, min=0, max=500, **SLOW),
    );
    """),
    md(r"""
    Each independent held distance or condition on the
    total momentum removes one freedom; an isolated molecule whose turning
    is removed loses three more, two if it is linear. ASE's temperature
    differs from `mdlab`'s only by the freedoms `mdlab` removes for the
    drift (and for the turning, when that box is ticked), a ratio that
    approaches 1 for large periodic systems: 0.996 for 256 atoms.
    """),
    md(r"""
    A liquid of two masses, 39.948 and 83.798 amu, started with velocities
    drawn at 130 K with the masses or with one spread for every atom: the
    temperature of each species.
    """),
    code(r"""
    LIGHT, HEAVY = np.arange(0, 256, 2), np.arange(1, 256, 2)


    def species(start="with the masses"):
        r = run("mixture_masses" if start == "with the masses"
                else "mixture_massless")
        t = r["frame_times"] / 1000
        light = statmech.part_temperature(r["masses"], r["velocities"],
                                          atoms=LIGHT)
        heavy = statmech.part_temperature(r["masses"], r["velocities"],
                                          atoms=HEAVY)
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        ax.plot(t, light, lw=0.8, label="light")
        ax.plot(t, heavy, lw=0.8, label="heavy")
        ax.set_xlim(0, 5)
        ax.set_xlabel("time / ps")
        ax.set_ylabel("T of a species / K")
        ax.legend()
        plt.show()
        print(f"at the start: light {light[0]:.0f} K, heavy {heavy[0]:.0f} K;"
              f" last 10 ps: {light[1000:].mean():.0f} and "
              f"{heavy[1000:].mean():.0f} K")


    widgets.interact(species, start=widgets.Dropdown(
        options=["with the masses", "one spread for all"]));
    """),
    md(r"""
    Drawn with the masses, the two species start
    within the scatter of 128 atoms, 128 and 119 K, and stay together.
    Drawn with one spread, the light atoms start at 88 K and the heavy at
    171 K; collisions bring them together within a few hundred
    femtoseconds, and over the last 10 ps both runs have equal species
    temperatures to within 2 K.
    """),
    code(r"""
    # SymPy: equipartition by parts. A square term a x^2 holds k_B T/2; a
    # quartic term a x^4 holds k_B T/4, so only squares share equally.
    x, a, b = sp.symbols("x a beta", positive=True)
    for power, share in ((2, sp.Rational(1, 2)), (4, sp.Rational(1, 4))):
        weight = sp.exp(-b * a * x**power)
        mean = (sp.integrate(a * x**power * weight, (x, -sp.oo, sp.oo))
                / sp.integrate(weight, (x, -sp.oo, sp.oo)))
        print(f"<a x^{power}> =", sp.simplify(mean))
        assert sp.simplify(mean - share / b) == 0
    """),
    code(r"""
    # Checked against ASE: get_temperature shares the kinetic energy among
    # 3N less the freedoms its constraints remove, without the three of
    # the drift. On the liquid, the ratio is 765/768; on rigid water with
    # FixBondLengths, 381/384; both to the 3.4e-7 by which the libraries'
    # values of k_B differ.
    v = LIQUID["velocities"][-1]
    ours = statmech.kinetic_temperature(LIQUID["masses"], v, N_FREE)
    atoms = Atoms("Ar256", positions=np.zeros((256, 3)),
                  masses=LIQUID["masses"])
    atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
    print(f"liquid: mdlab {ours:.4f} K, ASE {atoms.get_temperature():.4f} K,"
          f" ratio {atoms.get_temperature() / ours:.6f} = 765/768 "
          f"{765 / 768:.6f}")
    assert math.isclose(atoms.get_temperature() / ours, 765 / 768,
                        rel_tol=1e-6)
    bonds, _ = water.constraint_bonds(64)
    m = np.tile([15.999, 1.008, 1.008], 64)
    v = statmech.thermal_velocities(m, 300.0, np.random.default_rng(3))
    ours = statmech.kinetic_temperature(
        m, v, statmech.degrees_of_freedom(192, len(bonds)))
    atoms = Atoms("OH2" * 64, positions=np.zeros((192, 3)), masses=m)
    atoms.set_velocities(v / math.sqrt(units.FORCE_TO_ACCEL))
    atoms.set_constraint(FixBondLengths(bonds))
    print(f"rigid water: ratio {atoms.get_temperature() / ours:.6f} = "
          f"381/384 {381 / 384:.6f}")
    assert math.isclose(atoms.get_temperature() / ours, 381 / 384,
                        rel_tol=1e-6)
    """),
]

KINETIC = [
    md(r"""
    ## 12.7 How much the temperature fluctuates

    The kinetic energy of $N$ argon atoms: drawn independently from the
    canonical ensemble (each frame new thermal velocities), and from a run
    at fixed energy, against the gamma density of $N_{\mathrm f} = 3N - 3$
    freedoms with the same mean.
    """),
    code(r"""
    def kinetic(atoms=256):
        r = run(f"size_{atoms}")
        n_free = statmech.degrees_of_freedom(atoms)
        k = r["kinetic"]
        temp = 2 * k.mean() / (n_free * KB)
        draws = np.array([
            statmech.kinetic_energy(
                r["masses"], statmech.thermal_velocities(
                    r["masses"], temp, np.random.default_rng(i)))
            for i in range(2000)])
        grid = np.linspace(0.75, 1.25, 400) * k.mean()
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        ax.hist(draws / k.mean(), bins=40, density=True, alpha=0.4,
                label="canonical draws")
        ax.hist(k / k.mean(), bins=40, density=True, alpha=0.4,
                label="fixed energy")
        ax.plot(grid / k.mean(), k.mean() * statmech.kinetic_energy_density(
            grid, n_free, temp), "k--", lw=1, label="gamma")
        ax.set_xlabel("K / mean")
        ax.set_ylabel("probability density")
        ax.legend()
        plt.show()
        cv = statmech.heat_capacity_nve(k, n_free) / (atoms * KB)
        spread = draws.std() / draws.mean()
        print(f"relative spread: canonical draws {spread:.4f},"
              f" formula {math.sqrt(2 / n_free):.4f}, fixed energy "
              f"{k.std() / k.mean():.4f}; C_V from the spread {cv:.3f} N k_B")


    widgets.interact(kinetic, atoms=widgets.SelectionSlider(
        options=[108, 256, 500, 864], value=256, **SLOW));
    """),
    md(r"""
    The canonical draws follow the gamma curve at
    every size, with a relative spread close to $\sqrt{2/N_{\mathrm f}}$.
    The run at fixed energy is narrower at every size, by a factor near
    0.6, and both narrow as $1/\sqrt N$. The heat capacity from the
    narrowing lies between 2.19 and $2.34\,Nk_\mathrm{B}$ at every size.
    """),
    code(r"""
    # Checked against SciPy: kinetic_energy_density is SciPy's gamma
    # distribution with shape N_f/2 and scale k_B T (ASE has no such
    # density).
    kt = KB * 300.0
    for n_free in (3, 30, 765):
        spread = math.sqrt(2 / n_free)
        k = 0.5 * n_free * kt * np.linspace(max(1 - 4 * spread, 0.01),
                                            1 + 4 * spread, 50)
        ours = statmech.kinetic_energy_density(k, n_free, 300.0)
        theirs = stats.gamma.pdf(k, n_free / 2, scale=kt)
        print(f"N_f = {n_free}: largest relative difference "
              f"{np.abs(ours / theirs - 1).max():.1e}")
        assert np.allclose(ours, theirs, rtol=1e-9)
    """),
    code(r"""
    # SymPy: from Z_K = beta^(-a) Gamma(a), <K> = a/beta and the variance
    # a/beta^2.
    a, beta, kk = sp.symbols("a beta K", positive=True)
    z_k = sp.integrate(kk ** (a - 1) * sp.exp(-beta * kk), (kk, 0, sp.oo),
                       conds="none")
    mean_k = sp.simplify(-sp.diff(sp.log(z_k), beta))
    var_k = sp.simplify(sp.diff(sp.log(z_k), beta, 2))
    print(z_k, mean_k, var_k)
    assert sp.simplify(mean_k - a / beta) == 0
    assert sp.simplify(var_k - a / beta**2) == 0
    """),
]

OTHER = [
    md(r"""
    ## 12.8 Other ensembles

    The Fermi-Dirac occupation of an orbital and its entropy, against
    $\varepsilon_{\mathrm o} - \mu$, at the temperature you choose.
    """),
    code(r"""
    def fermi(temperature=300.0):
        e = np.linspace(-0.3, 0.3, 601)
        f = 1 / (np.exp(e / (KB * temperature)) + 1)
        g = np.clip(f, 1e-300, 1 - 1e-16)
        s = -(g * np.log(g) + (1 - g) * np.log(1 - g))
        fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3), dpi=80)
        a.plot(e, f)
        a.set_xlabel(r"$\varepsilon_\mathrm{o} - \mu$ / eV")
        a.set_ylabel("occupation")
        b.plot(e, s)
        b.axhline(math.log(2), ls=":", color="0.4")
        b.set_xlabel(r"$\varepsilon_\mathrm{o} - \mu$ / eV")
        b.set_ylabel("entropy / $k_B$")
        plt.show()
        print(f"k_B T = {KB * temperature:.4f} eV; occupation 0.1 eV above "
              f"mu {1 / (math.exp(0.1 / (KB * temperature)) + 1):.4f}")


    widgets.interact(fermi, temperature=widgets.FloatSlider(
        300, min=30, max=3000, step=30, **SLOW));
    """),
    md(r"""
    Every curve passes through ½ at
    $\varepsilon_{\mathrm o} = \mu$, and the step from 1 to 0 is a few
    $k_\mathrm{B} T$ wide, with $k_\mathrm{B}T = 0.026$ eV at 300 K. The entropy is negligible outside the
    step and
    is largest, $\ln 2$, at $\varepsilon_{\mathrm o} = \mu$; smearing the occupations
    of a metal adds this entropy for every orbital near the Fermi level.
    """),
    code(r"""
    # SymPy: f(mu + x) = 1 - f(mu - x), the slope -1/(4 k_B T) at mu, and
    # the Legendre transforms of the energy.
    x, kt = sp.symbols("x kT", positive=True)
    f = lambda e: 1 / (sp.exp(e / kt) + 1)  # noqa: E731
    assert sp.simplify(f(x) - (1 - f(-x))) == 0
    assert sp.simplify(sp.diff(f(x), x).subs(x, 0) + 1 / (4 * kt)) == 0
    T, S, P, V, mu, N = sp.symbols("T S P V mu N")
    dT, dS, dP, dV, dmu, dN = sp.symbols("dT dS dP dV dmu dN")
    dE = T * dS - P * dV + mu * dN
    dF = dE - T * dS - S * dT
    dG = dF + P * dV + V * dP
    dOmega = dF - mu * dN - N * dmu
    print("dF =", sp.expand(dF), "; d(F + PV) =", sp.expand(dG),
          "; d(F - mu N) =", sp.expand(dOmega))
    assert sp.expand(dG - (-S * dT + V * dP + mu * dN)) == 0
    assert sp.expand(dOmega - (-S * dT - P * dV - N * dmu)) == 0
    """),
]

TRAJECTORY = [
    md(r"""
    ## 12.9 What the trajectory looks like

    The healthy run: kinetic temperature, the temperatures of the three
    directions, and the excess kurtosis of the velocities.
    """),
    code(r"""
    m, v = LIQUID["masses"], LIQUID["velocities"]
    temp = 2 * LIQUID["kinetic"] / (N_FREE * KB)
    directions = [statmech.part_temperature(m, v, axis=k) for k in range(3)]
    kurt = np.array([diagnostics.excess_kurtosis(m, f) for f in v])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10, 3), dpi=80)
    a.plot(LIQUID["times"] / 1000, temp, lw=0.4)
    a.set_xlabel("time / ps")
    a.set_ylabel("T / K")
    for k, name in enumerate("xyz"):
        running = np.cumsum(directions[k]) / np.arange(1, len(v) + 1)
        b.plot(LIQUID["frame_times"] / 1000, running, label=f"$T_{name}$")
    b.legend()
    b.set_xlabel("time / ps")
    c.plot(LIQUID["frame_times"] / 1000, kurt, lw=0.5)
    c.set_xlabel("time / ps")
    c.set_ylabel("excess kurtosis")
    plt.show()
    print(f"T {temp.mean():.2f} K, spread {temp.std() / temp.mean():.4f} "
          f"(canonical {math.sqrt(2 / N_FREE):.4f}); directions "
          + ", ".join(f"{d.mean():.2f}" for d in directions)
          + f" K; kurtosis {kurt.mean():+.3f} +- {kurt.std():.3f}")
    """),
    md(r"""
    The six faults of the chapter, each through the check that catches
    it, against the healthy run (dashed).
    """),
    code(r"""
    def _water():
        r = np.load(Path("..") / "data" / "ch11_constraints" / "runs"
                    / "rigid_healthy.npz")
        right = 2 * r["kinetic"] / (381 * KB)
        wrong = 2 * r["kinetic"] / (576 * KB)
        t = r["times"] / 1000
        return t, wrong, (t, right), "T / K", (
            f"T over 576: {wrong.mean():.1f} K; over 381: "
            f"{right.mean():.1f} K, the 300 K asked for")


    def _lattice():
        r = run("lattice")
        t = 2 * r["kinetic"] / (N_FREE * KB)
        return r["times"] / 1000, t, None, "T / K", (
            f"start {t[0]:.1f} K, last 10 ps {t[1000:].mean():.1f} K, "
            f"ratio {t[1000:].mean() / t[0]:.3f}")


    def _drift():
        r = run("drift")
        t = 2 * r["kinetic"] / (N_FREE * KB)
        p = np.einsum("i,tix->tx", r["masses"], r["velocities"])
        return r["times"] / 1000, t, (LIQUID["times"] / 1000, temp), "T / K", (
            f"T {t.mean():.1f} K; total momentum "
            f"{np.linalg.norm(p, axis=1).mean():.3f} amu Å/fs")


    def _masses():
        r = run("mixture_massless")
        light = statmech.part_temperature(r["masses"], r["velocities"],
                                          atoms=LIGHT)
        heavy = statmech.part_temperature(r["masses"], r["velocities"],
                                          atoms=HEAVY)
        t = r["frame_times"] / 1000
        return t, light, (t, heavy), "T of a species / K", (
            f"light {light[0]:.0f} K and heavy {heavy[0]:.0f} K at the "
            f"start (shown solid and dashed)")


    def _uniform():
        r = run("uniform")
        k = np.array([diagnostics.excess_kurtosis(r["masses"], f)
                      for f in r["velocities"]])
        healthy = (LIQUID["frame_times"] / 1000, kurt)
        return r["frame_times"] / 1000, k, healthy, "excess kurtosis", (
            f"{k[0]:+.2f} at the start, {k[20]:+.2f} after 200 fs")


    def _sharing():
        r = chain.run(4.0 * PATTERNS[:, 0], np.zeros(32), 0.25, 0.05,
                      400_000, every=500)
        share = r["modes"].mean(0) / (r["energy"][0] / 32)
        modes = np.arange(1, 33)
        label = "mean energy / equal share"
        return modes, share, (modes, np.ones(32)), label, (
            f"modes 1 to 4 hold {share[:4].sum() / share.sum():.3f} of the "
            f"energy")


    FAULTS = {"(a) freedoms miscounted": _water,
              "(b) lattice start": _lattice,
              "(c) drift counted": _drift,
              "(d) masses ignored": _masses,
              "(e) not Gaussian": _uniform,
              "(f) no sharing": _sharing}


    def fault(name="(a) freedoms miscounted"):
        x, bad, good, label, note = FAULTS[name]()
        fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
        labels = {
            "a": ("wrong count", "correct count"),
            "b": ("lattice start", ""),
            "c": ("drift retained", "drift removed"),
            "d": ("light atoms", "heavy atoms"),
            "e": ("uniform start", "thermal start"),
            "f": ("measured", "equal sharing"),
        }[name[1]]
        ax.plot(x, bad, lw=0.6, label=labels[0])
        if good is not None:
            xg, yg = good
            shown = xg <= x.max()
            ax.plot(xg[shown], yg[shown], "--", lw=0.6, color="0.4",
                    label=labels[1])
        if name.startswith("(f)"):
            ax.set_yscale("log")
        ax.set_xlabel("mode number" if name.startswith("(f)") else "time / ps")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        plt.show()
        print(note)


    widgets.interact(fault, name=widgets.Dropdown(options=list(FAULTS)));
    """),
    md(r"""
    (a) The same run reads about 200 K over the
    wrong count and 304 K over the right one. (b) The crystal's
    temperature halves within a fraction of a picosecond. (c) The drift
    adds a steady 12 K and a total momentum of 8.9 amu Å/fs. (d) The light
    atoms start cold and the heavy hot, and they meet within a few hundred
    femtoseconds. (e) The kurtosis starts near $-1.1$ and reaches the
    healthy band within about 200 fs. (f) The chain keeps almost all its
    energy in its first four modes.
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
    derivations 12.7, 12.8, 12.11, 12.13, 12.16 and 12.17 are checked by
    SymPy, 12.19 by numerical integration, and 12.20 by the SymPy cell of
    Section 12.8; 12.4 takes one line.
    """),
    hidden(r"""
    # Reference values, calculated from the exercise data.
    _beta_eps = 0.1 / (KB * 300)
    _peak = 1 / 2.399357
    _sig_li = math.sqrt(KB * 300 / (6.94 * units.MV2_TO_EV))
    _sig_c = math.sqrt(KB * 300 / (12.011 * units.MV2_TO_EV))
    _f = 1 / (math.exp(_beta_eps) + 1)
    _r = 0.025**2 * 1497 / 2
    TARGETS = {
        "12.1": np.array([3.5, 35 / 12, 35 / 6]),
        "12.2": 29167,
        "12.3": 2 * np.sin(np.arange(1, 4) * np.pi / 8),
        "12.5": 0.1 / (2 * 1.5e4 * KB * 300),
        "12.6": np.array([0.1 / (math.exp(_beta_eps) + 1),
                          _beta_eps**2 * math.exp(_beta_eps)
                          / (math.exp(_beta_eps) + 1) ** 2,
                          _peak * 0.1 / KB]),
        "12.9": np.array([0.3 / (KB * math.log(1000)),
                          math.exp(-0.5 / (KB * 300))]),
        "12.10": np.array([math.sqrt(2) * _sig_li,
                           math.sqrt(8 / math.pi) * _sig_li]),
        "12.12": np.array([_sig_li, _sig_c, _sig_li / _sig_c]),
        "12.14": np.array([4, 321, 321 / 324]),
        "12.15": 6668,
        "12.18": 1497 / 2 / (1 - _r) / 500,
        "12.21": -(_f * math.log(_f) + (1 - _f) * math.log(1 - _f)),
        "12.22": math.sqrt(KB * 15 / (39.948 * units.MV2_TO_EV)),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 12.1: the mean and variance of one die, and the variance of
    # the total of two.
    die = None  # np.array([mean, variance, variance_of_two])

    check(die, TARGETS["12.1"], rtol=1e-6, name="die")
    """),
    solution(r"""
    # SOLUTION 12.1. <x> = 21/6, <x^2> = 91/6; variances of independent
    # dice add.
    faces = np.arange(1, 7)
    variance = (faces**2).mean() - faces.mean() ** 2
    die = np.array([faces.mean(), variance, 2 * variance])

    check(die, TARGETS["12.1"], rtol=1e-6, name="die");
    """),
    code(r"""
    # EXERCISE 12.2: how many throws of a die make the standard deviation
    # of their mean no more than 0.01?
    throws_needed = None

    check(throws_needed, TARGETS["12.2"], rtol=0, atol=0.5, name="throws")
    """),
    solution(r"""
    # SOLUTION 12.2. var(mean) = (35/12)/n = 0.01^2. A quick simulation
    # confirms sigma^2/n for n = 100.
    throws_needed = math.ceil((35 / 12) / 0.01**2)
    means = np.random.default_rng(2).integers(1, 7, (20000, 100)).mean(1)
    print(f"simulated variance of a mean of 100: {means.var():.5f}, "
          f"formula {35 / 12 / 100:.5f}")

    check(throws_needed, TARGETS["12.2"], rtol=0, atol=0.5, name="throws");
    """),
    code(r"""
    # EXERCISE 12.3: the three angular frequencies of a chain of three.
    omegas = None  # np.array([w1, w2, w3])

    check(omegas, TARGETS["12.3"], rtol=1e-6, name="frequencies")
    """),
    solution(r"""
    # SOLUTION 12.3. 2 sin(k pi/8), and the eigenvalues of the matrix.
    omegas = 2 * np.sin(np.arange(1, 4) * np.pi / 8)
    matrix = 2 * np.eye(3) - np.eye(3, k=1) - np.eye(3, k=-1)
    print(omegas**2, np.linalg.eigvalsh(matrix))

    check(omegas, TARGETS["12.3"], rtol=1e-6, name="frequencies");
    """),
    md(r"""
    **Exercise 12.4** (entropies add) is one line:
    $k_\mathrm{B}\ln(\Omega_1\Omega_2) = k_\mathrm{B}\ln\Omega_1 + k_\mathrm{B}\ln\Omega_2$.
    """),
    code(r"""
    # EXERCISE 12.5: the ratio E_s/2E of the second-order term to the
    # first for 0.1 eV against a bath of 1e4 ideal-gas atoms at 300 K.
    factor = None

    check(factor, TARGETS["12.5"], rtol=1e-3, name="factor")
    """),
    solution(r"""
    # SOLUTION 12.5. E = 3/2 N k_B T for the bath.
    factor = 0.1 / (2 * 1.5e4 * KB * 300)

    check(factor, TARGETS["12.5"], rtol=1e-3, name="factor");
    """),
    code(r"""
    # EXERCISE 12.6: two levels, eps = 0.1 eV: <E> (eV) and C_V/k_B at
    # 300 K, and the temperature (K) at which C_V is largest.
    two_level = None  # np.array([mean_energy, cv_over_kb, t_peak])

    check(two_level, TARGETS["12.6"], rtol=2e-3, name="two levels")
    """),
    solution(r"""
    # SOLUTION 12.6. <E> = eps/(e^x + 1), C_V/k_B = x^2 e^x/(e^x + 1)^2 with
    # x = eps/k_B T; maximise over x numerically.
    from scipy.optimize import minimize_scalar

    x = 0.1 / (KB * 300)
    best = minimize_scalar(lambda y: -y * y * math.exp(y)
                           / (math.exp(y) + 1) ** 2, bounds=(0.5, 6),
                           method="bounded")
    two_level = np.array([0.1 / (math.exp(x) + 1),
                          x * x * math.exp(x) / (math.exp(x) + 1) ** 2,
                          0.1 / (KB * best.x)])

    check(two_level, TARGETS["12.6"], rtol=2e-3, name="two levels");
    """),
    code(r"""
    # EXERCISE 12.7 and 12.8, checked by SymPy: F = <E> - TS for a system
    # of three levels, and the variance (k_B T)^2 of a vibration's energy.
    b, e1, e2 = sp.symbols("beta e1 e2", positive=True)
    levels = [0, e1, e2]
    z = sum(sp.exp(-b * e) for e in levels)
    probs = [sp.exp(-b * e) / z for e in levels]
    mean_e = sum(e * p for e, p in zip(levels, probs))
    entropy = -sum(p * sp.log(p) for p in probs)  # in units of k_B
    free = -sp.log(z) / b
    assert sp.simplify(free - (mean_e - entropy / b)) == 0
    z_spring = 2 * sp.pi / (b * sp.Symbol("omega", positive=True))
    assert sp.simplify(sp.diff(sp.log(z_spring), b, 2) - 1 / b**2) == 0
    print("12.7 and 12.8: checked")
    """),
    code(r"""
    # EXERCISE 12.9: the temperature (K) at which a 0.3 eV barrier has odds
    # 1e-3, and the odds of a 0.5 eV barrier at 300 K.
    barrier = None  # np.array([temperature, odds])

    check(barrier, TARGETS["12.9"], rtol=1e-3, name="odds")
    """),
    solution(r"""
    # SOLUTION 12.9. 0.3/k_B T = ln 1000.
    barrier = np.array([0.3 / (KB * math.log(1000)),
                        statmech.boltzmann_factor(0.5, 300.0)])

    check(barrier, TARGETS["12.9"], rtol=1e-3, name="odds");
    """),
    code(r"""
    # EXERCISE 12.10: the most probable and the mean speed (Å/fs) of lithium
    # at 300 K.
    li_speeds = None  # np.array([most_probable, mean])

    check(li_speeds, TARGETS["12.10"], rtol=1e-3, name="speeds")
    """),
    solution(r"""
    # SOLUTION 12.10. sqrt(2) sigma and sqrt(8/pi) sigma, sigma^2 = k_B T/m.
    sigma = math.sqrt(KB * 300 / (6.94 * units.MV2_TO_EV))
    li_speeds = np.array([math.sqrt(2) * sigma,
                          math.sqrt(8 / math.pi) * sigma])

    check(li_speeds, TARGETS["12.10"], rtol=1e-3, name="speeds");
    """),
    code(r"""
    # EXERCISE 12.11, checked by SymPy: <x^4> = 3 sigma^4 for the Gaussian,
    # and an even spread has excess kurtosis -6/5.
    x, s, c = sp.symbols("x s c", positive=True)
    gauss = sp.exp(-x**2 / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))
    assert sp.simplify(sp.integrate(x**4 * gauss, (x, -sp.oo, sp.oo))
                       - 3 * s**4) == 0
    m2 = sp.integrate(x**2, (x, -c, c)) / (2 * c)
    m4 = sp.integrate(x**4, (x, -c, c)) / (2 * c)
    assert sp.simplify(m4 / m2**2 - 3) == sp.Rational(-6, 5)
    print("12.11: checked")
    """),
    code(r"""
    # EXERCISE 12.12: sigma_v (Å/fs) of Li and of C at 300 K, and their
    # ratio.
    graphite = None  # np.array([sigma_li, sigma_c, ratio])

    check(graphite, TARGETS["12.12"], rtol=1e-3, name="sigmas")
    """),
    solution(r"""
    # SOLUTION 12.12. sqrt(k_B T/m); equal mean kinetic energies.
    sig = [math.sqrt(KB * 300 / (mass * units.MV2_TO_EV))
           for mass in (6.94, 12.011)]
    graphite = np.array([sig[0], sig[1], sig[0] / sig[1]])

    check(graphite, TARGETS["12.12"], rtol=1e-3, name="sigmas");
    """),
    code(r"""
    # EXERCISE 12.13, checked by SymPy: the Hamiltonian of a turning
    # molecule from its kinetic energy.
    th, ph, thd, phd, i_ = sp.symbols("theta phi thetadot phidot I",
                                      positive=True)
    lagrangian = sp.Rational(1, 2) * i_ * (thd**2 + sp.sin(th)**2 * phd**2)
    p_th, p_ph = sp.diff(lagrangian, thd), sp.diff(lagrangian, phd)
    h = p_th * thd + p_ph * phd - lagrangian
    pt, pp = sp.symbols("p_theta p_phi")
    h_p = sp.simplify(h.subs({thd: pt / i_, phd: pp / (i_ * sp.sin(th)**2)}))
    print(h_p)
    assert sp.simplify(h_p - (pt**2 / (2 * i_)
                              + pp**2 / (2 * i_ * sp.sin(th)**2))) == 0
    """),
    code(r"""
    # EXERCISE 12.14: freedoms of an isolated CO2 with drift and turning
    # removed; of 108 periodic argon atoms, drift removed; and the factor
    # ASE's temperature is of mdlab's for the second.
    counts = None  # np.array([co2, argon, factor])

    check(counts, TARGETS["12.14"], rtol=1e-6, name="freedoms")
    """),
    solution(r"""
    # SOLUTION 12.14. 9 - 3 - 2; 3N - 3; ASE divides by 3N.
    counts = np.array([9 - 3 - 2, statmech.degrees_of_freedom(108),
                       statmech.degrees_of_freedom(108) / 324])

    check(counts, TARGETS["12.14"], rtol=1e-6, name="freedoms");
    """),
    code(r"""
    # EXERCISE 12.15: atoms needed for a 1% canonical spread of the kinetic
    # temperature, drift removed.
    atoms_needed = None

    check(atoms_needed, TARGETS["12.15"], rtol=0, atol=0.5, name="atoms")
    """),
    solution(r"""
    # SOLUTION 12.15. sqrt(2/N_f) = 0.01 gives N_f = 20000 = 3N - 3.
    atoms_needed = math.ceil((2 / 0.01**2 + 3) / 3)

    check(atoms_needed, TARGETS["12.15"], rtol=0, atol=0.5, name="atoms");
    """),
    code(r"""
    # EXERCISES 12.16 and 12.17, checked by SymPy: the peak of the gamma
    # density, and the bracket of the fixed-energy spread for C_U = C_K.
    k, kt, a = sp.symbols("K kT a", positive=True)
    peak = sp.solve(sp.diff((a - 1) * sp.log(k) - k / kt, k), k)
    assert peak == [kt * (a - 1)]
    c_k = sp.Symbol("C_K", positive=True)
    bracket = 1 - c_k / (c_k + c_k)  # 1 - C_K/C_V with C_V = 2 C_K
    assert bracket == sp.Rational(1, 2)
    print("12.16: peak at", peak[0], "; 12.17: variance ratio", bracket)
    """),
    code(r"""
    # EXERCISE 12.18: C_V per atom (in k_B) of 500 atoms whose kinetic
    # energy at fixed energy has a relative spread of 0.025.
    cv_per_atom = None

    check(cv_per_atom, TARGETS["12.18"], rtol=1e-3, name="C_V per atom")
    """),
    solution(r"""
    # SOLUTION 12.18. 2<dK^2>/(N_f (k_B T)^2) = 0.025^2 N_f/2.
    n_free = statmech.degrees_of_freedom(500)
    ratio = 0.025**2 * n_free / 2
    cv_per_atom = n_free / 2 / (1 - ratio) / 500

    check(cv_per_atom, TARGETS["12.18"], rtol=1e-3, name="C_V per atom");
    """),
    code(r"""
    # EXERCISE 12.19, checked numerically: the volume's variance against
    # -k_B T d<V>/dP for a model Z(V) = V^3 e^(-beta c V^2), with k_B T = 1/2.
    from scipy.integrate import quad

    def moments(pressure, beta=2.0, c=0.5):
        weight = lambda v: v**3 * math.exp(-beta * (c * v * v  # noqa: E731
                                                    + pressure * v))
        z = quad(weight, 0, np.inf)[0]
        return [quad(lambda v: v**k * weight(v), 0, np.inf)[0] / z
                for k in (1, 2)]

    mean_v, mean_v2 = moments(0.3)
    h = 1e-4
    slope = (moments(0.3 + h)[0] - moments(0.3 - h)[0]) / (2 * h)
    print(f"variance {mean_v2 - mean_v**2:.6f}, -k_B T d<V>/dP "
          f"{-slope / 2.0:.6f}")
    assert math.isclose(mean_v2 - mean_v**2, -slope / 2.0, rel_tol=1e-6)
    """),
    code(r"""
    # EXERCISE 12.21: the entropy, in units of k_B, of an orbital 0.1 eV
    # above mu at 300 K.
    orbital_entropy = None

    check(orbital_entropy, TARGETS["12.21"], rtol=1e-3, name="entropy")
    """),
    solution(r"""
    # SOLUTION 12.21. -[f ln f + (1 - f) ln(1 - f)].
    f = 1 / (math.exp(0.1 / (KB * 300)) + 1)
    orbital_entropy = -(f * math.log(f) + (1 - f) * math.log(1 - f))

    check(orbital_entropy, TARGETS["12.21"], rtol=1e-3, name="entropy");
    """),
    code(r"""
    # EXERCISE 12.22: the drift velocity (Å/fs) along z of argon that adds
    # 15 K to T_z.
    drift_velocity = None

    check(drift_velocity, TARGETS["12.22"], rtol=1e-3, name="drift")
    """),
    solution(r"""
    # SOLUTION 12.22. T_z gains m v_d^2/k_B. A check on drawn velocities:
    # adding v_d to every z component raises part_temperature along z by
    # 15 K exactly when the drawn velocities have no drift.
    drift_velocity = math.sqrt(KB * 15 / (39.948 * units.MV2_TO_EV))
    m = np.full(500, 39.948)
    v = statmech.thermal_velocities(m, 300.0, np.random.default_rng(5))
    shifted = v + np.array([0.0, 0.0, drift_velocity])
    rise = (statmech.part_temperature(m, shifted, axis=2)
            - statmech.part_temperature(m, v, axis=2))
    print(f"T_z rises by {rise:.3f} K")

    check(drift_velocity, TARGETS["12.22"], rtol=1e-3, name="drift");
    """),
]

NOTES = [md(r"""
    ## Working notes

    Choose a trajectory average and record how it changes when the observed
    interval is shortened. Keep the fluctuation of the observable separate
    from the uncertainty in its mean. Use the chain and liquid examples
    to explain why a longer trajectory need not explore every state.
    """)]

CELLS = (SETUP + PROBABILITY + ERGODIC + MICRO + CANONICAL + MAXWELL
         + EQUIPARTITION + KINETIC + OTHER + TRAJECTORY + EXERCISES + NOTES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "12_ensembles.ipynb"))
