"""Write notebooks/04_oscillations.ipynb, the companion to Chapter 4.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_04_oscillations.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/04_oscillations.ipynb
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md("""
    # Notebook 04: oscillations

    Start with one spring, then change its energy, add drag and a driving
    force, and couple it to another oscillator. The section numbers match
    Chapter 4. Run the cells in order; when changing a parameter, predict
    the effect on the period or amplitude before looking at the new curve.
    The exercises below leave space for your calculation and keep the worked
    solutions folded away until you need them.
    """),
    code("""
    %matplotlib inline
    import math

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation
    from scipy.constants import g as G  # 9.80665 m/s²

    from mdlab import dynamics, energy, oscillators, units, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    """),
]

CIRCLE = [
    md(r"""
    ## 4.1 The shadow of a circle

    The point $P$ goes round a circle of radius $A$ at $\omega$; its
    shadow on the $x$ axis is the block on the spring. The animation runs
    over one period of the block of Section 2.8 started 3 cm out and
    moving outwards at 0.4 m/s ($A$ = 5 cm, $\phi$ = −0.927 rad).
    """),
    code("""
    def animate_reference_circle():
        OMEGA, X0, V0 = 10.0, 0.03, 0.4
        AMP = math.hypot(X0, V0 / OMEGA)
        PHI = math.atan2(-V0 / OMEGA, X0)
        times = np.linspace(0.0, 2 * np.pi / OMEGA, 60)

        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)
        s = np.linspace(0, 2 * np.pi, 200)
        ax.plot(100 * AMP * np.cos(s), 100 * AMP * np.sin(s), color=viz.REFERENCE)
        ax.axhline(0, color="black", lw=0.5)
        (point,) = ax.plot([], [], "o", color=viz.ACCENT, ms=7)
        (shadow,) = ax.plot([], [], "s", color=viz.OCHRE, ms=7)
        (drop,) = ax.plot([], [], ":", color="black")
        ax.set_aspect("equal")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel("$x$ / cm")
        ax.set_ylabel("$y$ / cm")


        def draw(k):
            angle = OMEGA * times[k] + PHI
            px, py = 100 * AMP * np.cos(angle), 100 * AMP * np.sin(angle)
            point.set_data([px], [py])
            shadow.set_data([px], [0.0])
            drop.set_data([px, px], [py, 0.0])
            return point, shadow, drop


        animation = FuncAnimation(fig, draw, frames=len(times), interval=60)
        plt.close(fig)
        return HTML(animation.to_jshtml())


    animate_reference_circle()
    """),
    md("""
    The square (the block) moves fastest when the
    dot crosses the top or bottom of the circle, and stops for an instant
    when the dot crosses the $x$ axis.
    """),
]

SPRING = [
    md(r"""
    ## 4.2 Energy in the oscillator and 4.3 the phase portrait

    The spring explorer: change the stiffness, the mass and the
    amplitude, and move the time cursor. The block, the energy bars and
    the state on its ellipse in the phase plane all move together.
    """),
    code("""
    def spring_explorer(k=50.0, m=0.5, amplitude=4.0, t=0.1):
        omega = math.sqrt(k / m)
        a = amplitude / 100  # m
        x, v = oscillators.harmonic(t, a, 0.0, omega)
        x, v = float(x), float(v)
        kinetic, potential = 0.5 * m * v**2, 0.5 * k * x**2

        fig, (block, bars, phase) = plt.subplots(
            1, 3, figsize=(viz.FULL, 2.3),
            gridspec_kw=dict(width_ratios=(1.6, 0.8, 1.2), wspace=0.5),
        )
        coil = np.linspace(-12, 100 * x - 1.0, 60)
        block.plot(coil, 0.3 * np.sin(np.linspace(0, 16 * np.pi, 60)),
                   color="black", lw=0.8)
        block.add_patch(plt.Rectangle((100 * x - 1.0, -0.8), 2.0, 1.6,
                                      color=viz.ACCENT))
        block.axvline(0, **viz.THRESHOLD_STYLE)
        block.set_xlim(-12, 12)
        block.set_ylim(-1.5, 1.5)
        block.set_yticks([])
        block.set_xlabel("$x$ / cm")
        bars.bar(["$K$", "$U$"], [1000 * kinetic, 1000 * potential],
                 color=[viz.OCHRE, viz.OXBLOOD])
        bars.set_ylim(0, 1000 * 0.5 * k * a**2 * 1.05)
        bars.set_ylabel("mJ")
        s = np.linspace(0, 2 * np.pi, 200)
        phase.plot(100 * a * np.cos(s), a * omega * np.sin(s),
                   color=viz.REFERENCE)
        phase.plot(100 * x, v, "o", color=viz.ACCENT)
        phase.set_xlabel("$x$ / cm")
        phase.set_ylabel("$v$ / m s$^{-1}$")
        plt.show()
        print(f"omega = {omega:.2f} rad/s, period {2 * math.pi / omega:.3f} "
              f"s, E = {1000 * 0.5 * k * a**2:.2f} mJ, K = "
              f"{1000 * kinetic:.2f} mJ, U = {1000 * potential:.2f} mJ")


    widgets.interact(
        spring_explorer,
        k=widgets.FloatSlider(50.0, min=5.0, max=200.0, step=5.0, **SLOW),
        m=widgets.FloatSlider(0.5, min=0.1, max=2.0, step=0.1, **SLOW),
        amplitude=widgets.FloatSlider(4.0, min=1.0, max=10.0, step=0.5),
        t=widgets.FloatSlider(0.1, min=0.0, max=2.0, step=0.01),
    );
    """),
    md("""
    The bars always add to the same total, which
    grows as the square of the amplitude. A stiffer spring or a lighter
    block makes the period shorter; for a given amplitude the energy,
    $\\frac12 kA^2$, depends on the stiffness alone, not on the mass.

    The equal shares of Section 4.2, checked by averaging over a period:
    """),
    code("""
    t = np.linspace(0.0, 2 * np.pi / 10.0, 100001)[:-1]
    x, v = oscillators.harmonic(t, 0.04, 0.0, 10.0)
    print(f"mean K = {np.mean(0.5 * 0.5 * v**2):.6f} J, "
          f"mean U = {np.mean(0.5 * 50 * x**2):.6f} J, E/2 = 0.02 J")
    """),
]

SMALL = [
    md(r"""
    ## 4.4 Small oscillations about any minimum

    SymPy expands the pendulum's well about its bottom; the quadratic
    term is $\frac12(mg/L)s^2$, a spring of stiffness $mg/L$.
    """),
    code("""
    s, m, g, L = sp.symbols("s m g L", positive=True)
    U = m * g * L * (1 - sp.cos(s / L))
    print(sp.series(U, s, 0, 5))
    print("omega_0 =", sp.sqrt(sp.diff(U, s, 2).subs(s, 0) / m))
    print(f"L = 1.2 m: period {2 * math.pi * math.sqrt(1.2 / G):.4f} s")
    """),
    md("""
    The lithium atom in a hollow of the model surface: the curvature
    along the hop, and the period of small rattles.
    """),
    code("""
    ub, a = energy.SURFACE_BARRIER, energy.SURFACE_SPACING
    k_li = 2 * math.pi**2 * ub / a**2  # eV/Å²
    h = 1e-4
    u = lambda x: float(oscillators.cosine_well(x, ub, a)[0])
    numerical = (u(h) - 2 * u(0.0) + u(-h)) / h**2
    omega_li = math.sqrt(k_li * units.FORCE_TO_ACCEL / 6.94)
    print(f"k = {k_li:.4f} eV/Å² (finite differences {numerical:.4f})")
    period_li = 2 * math.pi / omega_li
    print(f"omega = {omega_li:.5f} rad/fs, period {period_li:.1f} fs")
    """),
]

ANHARMONIC = [
    md(r"""
    ## 4.5 Larger swings: anharmonic wells

    Choose a well and an energy, as a fraction of its depth. The plot
    shows the well, the energy and the turning points; the period comes
    from `oscillators.period` and, as a check, from a motion timed by
    `dynamics.solve_newton`.
    """),
    code("""
    WELLS = {
        "cosine": (lambda x: float(oscillators.cosine_well(x, 1.0, 1.0)[0]),
                   0.0, 2 * math.pi**2, (-0.55, 0.55)),
        "Morse": (lambda y: float(oscillators.morse(y, 1.0, 1.0, 0.0)[0]),
                  0.0, 2.0, (-0.8, 4.0)),
        "Lennard-Jones": (
            lambda r: float(oscillators.lennard_jones(r, 1.0, 1.0)[0]) + 1.0,
            2 ** (1 / 6), 72 / 2 ** (1 / 3), (0.95, 2.6)),
    }


    def well_explorer(well="Morse", fraction=0.5):
        u, bottom, curvature, (lo, hi) = WELLS[well]
        t0 = 2 * math.pi / math.sqrt(curvature)  # mass 1
        left = oscillators.turning_point(u, fraction, bottom, -1e-3)
        right = oscillators.turning_point(u, fraction, bottom, 1e-3)
        period = oscillators.period(u, fraction, 1.0, bottom, 1e-3,
                                    force_to_accel=1.0, nodes=400)

        def force(r):
            dx = 1e-6
            return -np.array([(u(float(r[0]) + dx) - u(float(r[0]) - dx))
                              / (2 * dx)])

        t = np.linspace(0.0, 2.2 * period, 6000)
        r, _ = dynamics.solve_newton(force, 1.0, [left], [0.0], t,
                                     force_to_accel=1.0, rtol=1e-9,
                                     atol=1e-11)
        midpoint = 0.5 * (left + right)
        crossings = np.flatnonzero((r[:-1, 0] < midpoint)
                                   & (r[1:, 0] >= midpoint))
        crossing_times = t[crossings] + (midpoint - r[crossings, 0]) * (
            t[crossings + 1] - t[crossings]) / (
            r[crossings + 1, 0] - r[crossings, 0])
        measured_period = crossing_times[1] - crossing_times[0]
        # Keep both turning points visible as the energy approaches escape.
        margin = 0.08 * (right - left)
        xs = np.linspace(min(lo, left - margin), max(hi, right + margin), 600)
        fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.3))
        ax.plot(xs, [u(x) for x in xs], color="black")
        parabola = 0.5 * curvature * (xs - bottom) ** 2
        ax.plot(xs, parabola, **viz.REFERENCE_STYLE)
        ax.axhline(fraction, color=viz.ACCENT)
        ax.plot([left, right], [fraction] * 2, "o", color=viz.ACCENT)
        ax.set_ylim(0, 1.6)
        ax.set_xlabel("position / well length scale")
        ax.set_ylabel("energy / well depth")
        plt.show()
        print(f"T/T0 = {period / t0:.4f}; from successive rightward "
              f"crossings: {measured_period / t0:.4f}. "
              f"Relative mismatch: {abs(measured_period / period - 1):.2e}")


    widgets.interact(
        well_explorer,
        well=widgets.Dropdown(options=list(WELLS), value="Morse"),
        fraction=widgets.FloatSlider(0.5, min=0.02, max=0.98, step=0.02,
                                     **SLOW),
    );
    """),
    md("""
    Near the bottom the well follows its dashed
    parabola and the period is close to the small-swing period. As the
    energy rises, the motion reaches the soft outer side of the Morse and
    Lennard-Jones wells, where it slows down, and the period grows.

    The arcsine integral of Section 4.5 for the harmonic period, done by
    SymPy:
    """),
    code("""
    x, A, w = sp.symbols("x A omega", positive=True)
    print("T =", sp.simplify(2 * sp.integrate(1 / (w * sp.sqrt(A**2 - x**2)),
                                               (x, -A, A))))
    """),
]

DAMPED = [
    md(r"""
    ## 4.6 Damping

    The block with $\omega_0$ = 10 rad/s released 4 cm out. Change the
    drag rate $\gamma$: below 20 per second it swings, at 20 it returns
    without overshooting, above it creeps back.
    """),
    code("""
    def damping(gamma=4.0):
        t = np.linspace(0.0, 2.0, 800)
        x, v = oscillators.damped(t, 0.04, 0.0, 10.0, gamma)
        energy_j = 0.5 * 0.5 * v**2 + 0.5 * 50 * x**2
        fig, (left, right) = plt.subplots(1, 2, figsize=(viz.FULL, 2.2))
        left.plot(t, 100 * x, color=viz.ACCENT)
        if gamma < 20.0:
            omega_d = math.sqrt(100 - gamma**2 / 4)
            envelope = 4 * math.hypot(1.0, gamma / (2 * omega_d))
            left.plot(t, envelope * np.exp(-gamma * t / 2),
                      label="upper envelope", **viz.THRESHOLD_STYLE)
            left.legend(fontsize=8)
        left.axhline(0, color="black", lw=0.4)
        left.set_xlabel("$t$ / s")
        left.set_ylabel("$x$ / cm")
        right.plot(t, 1000 * energy_j, color=viz.ACCENT)
        right.set_xlabel("$t$ / s")
        right.set_ylabel("$E$ / mJ")
        plt.show()
        rate = 100 - gamma**2 / 4
        kind = ("under-damped" if rate > 0 else
                "critical" if rate == 0 else "over-damped")
        print(kind + (f", omega_d = {math.sqrt(rate):.3f} rad/s" if rate > 0
                      else ""))


    widgets.interact(damping, gamma=widgets.FloatSlider(4.0, min=0.0,
                                                        max=60.0, step=1.0));
    """),
    md("""
    SymPy confirms that $x = e^{-\\gamma t/2}y$ removes the drag term:
    """),
    code("""
    t, gamma, w0 = sp.symbols("t gamma omega_0", positive=True)
    y = sp.Function("y")
    x = sp.exp(-gamma * t / 2) * y(t)
    damped_eq = x.diff(t, 2) + gamma * x.diff(t) + w0**2 * x
    print(sp.simplify(damped_eq * sp.exp(gamma * t / 2)))
    """),
]

DRIVEN = [
    md(r"""
    ## 4.7 Driving and resonance

    The block ($\omega_0$ = 10 rad/s, $\gamma$ = 2 per second) starts at
    rest and is pushed with $f$ = 1 m/s² at the angular frequency
    $\omega$. The motion (teal) is the steady response plus a decaying
    damped motion; the dotted curve is the steady response alone.
    """),
    code("""
    def drive(omega=10.0, gamma=2.0):
        t = np.linspace(0.0, 8.0, 3000)
        amp, lag = oscillators.driven_steady_state(omega, 10.0, gamma, 1.0)
        steady = amp * np.cos(omega * t - lag)
        # the damped part cancels the steady state's start: x(0) = v(0) = 0
        x0 = -amp * np.cos(lag)
        v0 = -amp * omega * np.sin(lag)
        transient, _ = oscillators.damped(t, x0, v0, 10.0, gamma)
        fig, ax = plt.subplots(figsize=(viz.FULL, 2.2))
        ax.plot(t, 100 * (steady + transient), color=viz.ACCENT)
        ax.plot(t, 100 * steady, **viz.THRESHOLD_STYLE)
        ax.set_xlabel("$t$ / s")
        ax.set_ylabel("$x$ / cm")
        plt.show()
        print(f"steady amplitude {100 * float(amp):.3f} cm, lag "
              f"{float(lag):.3f} rad")


    widgets.interact(
        drive,
        omega=widgets.FloatSlider(10.0, min=1.0, max=20.0, step=0.25, **SLOW),
        gamma=widgets.FloatSlider(2.0, min=0.5, max=8.0, step=0.5, **SLOW),
    );
    """),
    md("""
    With the default $\\gamma = 2$ per second and $\\omega = 10$ rad/s,
    the swing builds up over a few times $2/\\gamma$ to five times the
    1 cm static stretch. Far from
    resonance it stays small, and the dying motion of the start, added
    to the steady swing, makes its size rise and fall slowly (a beat)
    until it dies away.
    """),
]

BODIES = [
    md(r"""
    ## 4.8 Two bodies and the reduced mass

    The carts' quarter period, and the isotope effect of Exercise 4.15.
    """),
    code("""
    mu = oscillators.reduced_mass(1.0, 3.0)
    print(f"carts: reduced mass {mu} kg, quarter period "
          f"{0.5 * math.pi * math.sqrt(mu / 200):.4f} s")
    for partner, mass in (("H", 1.008), ("D", 2.014)):
        print(f"O-{partner}: reduced mass "
              f"{oscillators.reduced_mass(mass, 15.999):.4f} amu")
    """),
]

MODES = [
    md(r"""
    ## 4.9 Coupled oscillators and normal modes

    NumPy's `eigh` finds the modes of the two carts and of the O-C-O
    chain; each pattern satisfies $\Phi u = \omega^2 M u$.
    """),
    code("""
    w2, modes = oscillators.normal_modes([[2, -1], [-1, 2]], [1, 1],
                                         force_to_accel=1.0)
    print("carts: omega^2 / (k/m) =", w2.round(6))
    print("patterns (columns):")
    print((modes / np.abs(modes).max(axis=0)).round(4))

    mo, mc = 15.999, 12.011
    phi = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], float)
    w2, chain = oscillators.normal_modes(phi, [mo, mc, mo],
                                         force_to_accel=1.0)
    print("O-C-O: omega^2 / k =", w2.round(6))
    print("ratio of the stretches:", math.sqrt(w2[2] / w2[1]))
    print("patterns, scaled so that the first O moves by 1:")
    print((chain / chain[0]).round(4))
    """),
    md("""
    The three modes of the chain, animated side by side (each with its own
    angular frequency; the translation drifts steadily, and the
    animation brings it back at the end of each loop).
    """),
    code("""
    def animate_chain_modes():
        frames = 60
        fig, ax = plt.subplots(figsize=(4.5, 2.4), dpi=120)
        ax.set_xlim(-1.0, 4.0)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2], ["antisymmetric", "symmetric", "translation"])
        ax.set_xticks([])
        rows = []
        for row in range(3):
            (line,) = ax.plot([], [], "-o", color="black", mfc=viz.ACCENT, ms=10)
            rows.append(line)
        base = np.array([0.0, 1.5, 3.0])
        patterns = [chain[:, 2] / chain[0, 2], chain[:, 1] / chain[0, 1],
                    np.ones(3)]
        speeds = [1.0, math.sqrt(w2[1] / w2[2]), 0.0]


        def draw(k):
            phase = 2 * np.pi * k / frames
            for row, (line, pattern, speed) in enumerate(
                    zip(rows, patterns, speeds, strict=True)):
                if speed == 0.0:  # translation: a steady drift, restarted each loop
                    shift = (0.6 * k / frames - 0.3) * pattern
                else:
                    shift = 0.3 * np.cos(speed * phase) * pattern
                line.set_data(base + shift, [row] * 3)
            return rows


        animation = FuncAnimation(fig, draw, frames=frames, interval=80)
        plt.close(fig)
        return HTML(animation.to_jshtml())


    animate_chain_modes()
    """),
]

MOLECULES = [
    md(r"""
    ## 4.10 Molecular vibrations and the time step

    Wavenumbers (Shimanouchi's tables, via the NIST WebBook) turned into
    periods with `units.C_CM_PER_FS`.
    """),
    code("""
    table = {
        "water, antisymmetric O-H stretch": 3756,
        "water, symmetric O-H stretch": 3657,
        "methane, C-H stretch": 3019,
        "methane, C-H stretch in step": 2917,
        "water, bend": 1595,
        "ethane, C-C stretch": 995,
    }
    for name, wavenumber in table.items():
        nu = wavenumber * units.C_CM_PER_FS  # cycles per fs
        print(f"{name:35s} {wavenumber:5d} cm-1  period {1 / nu:6.2f} fs")
    """),
]

EXERCISES = [
    md("""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises' works
    every exercise in full, including 4.1, which asks for an
    explanation.

    Run the collapsed answer-key cell below without opening it. After
    each exercise a collapsed cell holds a worked solution in code. The
    derivations 4.3, 4.6, 4.7, 4.8, 4.10, 4.12, 4.14, 4.16, 4.17 and 4.18 have
    their solutions checked by SymPy.
    """),
    hidden("""
    # The answer key. Each target is computed rather than typed in.
    _w = math.sqrt(80 / 0.2)
    _amp5, _lag5 = oscillators.driven_steady_state(5.0, 10.0, 2.0, 1.0)
    _amp15, _lag15 = oscillators.driven_steady_state(15.0, 10.0, 2.0, 1.0)
    _mu_oh = oscillators.reduced_mass(1.008, 15.999)
    _mu_od = oscillators.reduced_mass(2.014, 15.999)
    _w_oh = 2 * math.pi * 3657 * units.C_CM_PER_FS
    _e4 = 0.5 * 0.5 * 0.4**2 + 0.5 * 50 * 0.03**2

    TARGETS = {
        "4.2 energy": 0.5 * 80 * 0.03**2,
        "4.2 vmax": 0.03 * _w,
        "4.2 v": math.sqrt(2 * (0.5 * 80 * (0.03**2 - 0.015**2)) / 0.2),
        "4.4 energy": _e4,
        "4.4 axes": np.array([math.sqrt(2 * _e4 / 50),
                              math.sqrt(2 * _e4 / 0.5)]),
        "4.5 length": G / math.pi**2,
        "4.6 time": math.pi / 60,
        "4.9 distance": (26 / 7) ** (1 / 6),
        "4.11 period": 2 * math.pi / math.sqrt(100 - 36),
        "4.11 tenth": math.log(10) / 6,
        "4.13 amplitudes": np.array([float(_amp5), float(_amp15)]),
        "4.13 lags": np.array([float(_lag5), float(_lag15)]),
        "4.15 ratio": math.sqrt(_mu_oh / _mu_od),
        "4.19 omega": _w_oh,
        "4.19 k": _mu_oh * _w_oh**2 / units.FORCE_TO_ACCEL,
        "4.20 step": 0.1 / (3756 * units.C_CM_PER_FS),
        "4.20 steps": math.ceil(1000 / (0.1 / (3756 * units.C_CM_PER_FS))),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code("""
    # EXERCISE 4.2: energy (J), largest speed (m/s), speed at 1.5 cm (m/s).
    energy_j = None
    vmax = None
    v = None

    check(energy_j, TARGETS["4.2 energy"], name="energy")
    check(vmax, TARGETS["4.2 vmax"], name="largest speed")
    check(v, TARGETS["4.2 v"], name="speed at 1.5 cm")
    """),
    solution("""
    # SOLUTION 4.2. E = k A²/2; v_max = A ω; K = E − U at x.
    energy_j = 0.5 * 80 * 0.03**2
    vmax = 0.03 * math.sqrt(80 / 0.2)
    v = math.sqrt(2 * (energy_j - 0.5 * 80 * 0.015**2) / 0.2)

    check(energy_j, TARGETS["4.2 energy"], name="energy")
    check(vmax, TARGETS["4.2 vmax"], name="largest speed")
    check(v, TARGETS["4.2 v"], name="speed at 1.5 cm")
    """),
    solution("""
    # SOLUTION 4.3. The average of sin² over a period is 1/2.
    t, w = sp.symbols("t omega", positive=True)
    average = sp.integrate(sp.sin(w * t) ** 2, (t, 0, 2 * sp.pi / w)) / (
        2 * sp.pi / w)
    print("average =", sp.simplify(average))
    assert sp.simplify(average - sp.Rational(1, 2)) == 0
    """),
    code("""
    # EXERCISE 4.4: energy (J), and the semi-axes as [m, m/s].
    energy_j = None
    axes = None

    check(energy_j, TARGETS["4.4 energy"], name="energy")
    check(axes, TARGETS["4.4 axes"], name="semi-axes")
    """),
    solution("""
    # SOLUTION 4.4. E from the state; semi-axes √(2E/k) and √(2E/m).
    energy_j = 0.5 * 0.5 * 0.4**2 + 0.5 * 50 * 0.03**2
    axes = np.array([math.sqrt(2 * energy_j / 50),
                     math.sqrt(2 * energy_j / 0.5)])

    check(energy_j, TARGETS["4.4 energy"], name="energy")
    check(axes, TARGETS["4.4 axes"], name="semi-axes")
    """),
    code("""
    # EXERCISE 4.5: length of a pendulum with a 2 s period (m).
    length = None

    check(length, TARGETS["4.5 length"], name="length")
    """),
    solution("""
    # SOLUTION 4.5. 2π √(L/g) = 2 gives L = g / π².
    length = G * (2 / (2 * math.pi)) ** 2

    check(length, TARGETS["4.5 length"], name="length")
    """),
    code("""
    # EXERCISE 4.6: time from the centre to half the amplitude (s).
    time = None

    check(time, TARGETS["4.6 time"], name="time")
    """),
    solution("""
    # SOLUTION 4.6. (1/ω) arcsin(1/2) = π / (6ω).
    time = math.asin(0.5) / 10.0

    check(time, TARGETS["4.6 time"], name="time")
    """),
    solution("""
    # SOLUTION 4.6, the derivation: the integral of dx/(ω√(A² − x²)) from 0
    # to A/2.
    x, A, w = sp.symbols("x A omega", positive=True)
    t_half = sp.integrate(1 / (w * sp.sqrt(A**2 - x**2)), (x, 0, A / 2))
    print("t =", t_half)
    assert sp.simplify(t_half - sp.pi / (6 * w)) == 0
    """),
    solution("""
    # SOLUTION 4.7. U'' at x = a for the double well.
    x, a, u0 = sp.symbols("x a U_0", positive=True)
    U = u0 * ((x / a) ** 2 - 1) ** 2
    curvature = sp.simplify(sp.diff(U, x, 2).subs(x, a))
    print("U''(a) =", curvature)
    assert sp.simplify(curvature - 8 * u0 / a**2) == 0
    """),
    solution("""
    # SOLUTION 4.8. The curvature across the hop equals that along it.
    y, ub, a = sp.symbols("y U_b a", positive=True)
    g = 4 * sp.pi / (sp.sqrt(3) * a)
    U = ub / 4 * (3 - 2 * sp.cos(g * y / 2) - sp.cos(g * y))
    angles = (-sp.pi / 6, sp.pi / 2, 7 * sp.pi / 6)  # g_1, g_2, g_3
    ripples = sum(sp.cos(g * y * sp.sin(al)) for al in angles)  # g_k·(0, y)
    assert sp.simplify(ub / 4 * (3 - ripples) - U) == 0
    across = sp.simplify(sp.diff(U, y, 2).subs(y, 0))
    print("U'' across =", across)
    assert sp.simplify(across - 2 * sp.pi**2 * ub / a**2) == 0
    """),
    code("""
    # EXERCISE 4.9: distance of the strongest attraction (in units of σ).
    distance = None

    check(distance, TARGETS["4.9 distance"], name="distance / sigma")
    """),
    solution("""
    # SOLUTION 4.9. U''_LJ = 0 where (σ/r)^6 = 42/156.
    distance = (156 / 42) ** (1 / 6)

    check(distance, TARGETS["4.9 distance"], name="distance / sigma")
    """),
    solution("""
    # SOLUTION 4.10. The Morse curvature at r0.
    r, d, a, r0 = sp.symbols("r D a r_0", positive=True)
    U = d * (1 - sp.exp(-a * (r - r0))) ** 2
    curvature = sp.simplify(sp.diff(U, r, 2).subs(r, r0))
    print("U''(r0) =", curvature)
    assert sp.simplify(curvature - 2 * d * a**2) == 0
    """),
    code("""
    # EXERCISE 4.11: period of the swing (s) and time to shrink to a
    # tenth (s).
    period = None
    tenth = None

    check(period, TARGETS["4.11 period"], name="period")
    check(tenth, TARGETS["4.11 tenth"], name="time to a tenth")
    """),
    solution("""
    # SOLUTION 4.11. ω_d = √(ω₀² − γ²/4); the swing shrinks as e^(−γt/2).
    period = 2 * math.pi / math.sqrt(10.0**2 - 12.0**2 / 4)
    tenth = math.log(10) / (12.0 / 2)

    check(period, TARGETS["4.11 period"], name="period")
    check(tenth, TARGETS["4.11 tenth"], name="time to a tenth")
    """),
    solution("""
    # SOLUTION 4.12. (c1 + c2 t) e^(−ω₀ t) solves the critical case.
    t, w0, c1, c2 = sp.symbols("t omega_0 c_1 c_2")
    x = (c1 + c2 * t) * sp.exp(-w0 * t)
    residual = x.diff(t, 2) + 2 * w0 * x.diff(t) + w0**2 * x
    print("residual =", sp.simplify(residual))
    assert sp.simplify(residual) == 0
    """),
    code("""
    # EXERCISE 4.13: amplitudes (m) and lags (rad) at 5 and 15 rad/s,
    # each as an array [at 5, at 15].
    amplitudes = None
    lags = None

    check(amplitudes, TARGETS["4.13 amplitudes"], name="amplitudes")
    check(lags, TARGETS["4.13 lags"], name="lags")
    """),
    solution("""
    # SOLUTION 4.13. Equation (4.9), with the lag in (0, π).
    omegas = np.array([5.0, 15.0])
    amplitudes = 1.0 / np.sqrt((100 - omegas**2) ** 2 + (2 * omegas) ** 2)
    lags = np.arctan2(2 * omegas, 100 - omegas**2)

    check(amplitudes, TARGETS["4.13 amplitudes"], name="amplitudes")
    check(lags, TARGETS["4.13 lags"], name="lags")
    """),
    solution("""
    # SOLUTION 4.14. A combination of two motions is a motion.
    t, gamma, w0, c1, c2 = sp.symbols("t gamma omega_0 c_1 c_2")
    x1, x2 = sp.Function("x1")(t), sp.Function("x2")(t)
    lhs = lambda x: x.diff(t, 2) + gamma * x.diff(t) + w0**2 * x
    difference = sp.expand(lhs(c1 * x1 + c2 * x2)
                           - c1 * lhs(x1) - c2 * lhs(x2))
    print("difference =", difference)
    assert difference == 0
    """),
    code("""
    # EXERCISE 4.15: ratio of the O-D to the O-H angular frequency.
    ratio = None

    check(ratio, TARGETS["4.15 ratio"], name="ratio")
    """),
    solution("""
    # SOLUTION 4.15. ω = √(k/m_r) with the same k.
    ratio = math.sqrt(oscillators.reduced_mass(1.008, 15.999)
                      / oscillators.reduced_mass(2.014, 15.999))

    check(ratio, TARGETS["4.15 ratio"], name="ratio")
    """),
    solution("""
    # SOLUTION 4.16. A different middle spring.
    k, kp, m = sp.symbols("k k' m", positive=True)
    phi = sp.Matrix([[k + kp, -kp], [-kp, k + kp]])
    for u, w2 in ((sp.Matrix([1, 1]), k / m), (sp.Matrix([1, -1]),
                                                (k + 2 * kp) / m)):
        assert sp.simplify(phi * u - m * w2 * u) == sp.zeros(2, 1)
    print("both modes check")
    """),
    solution("""
    # SOLUTION 4.17. The antisymmetric stretch of O-C-O.
    k, mo, mc = sp.symbols("k m_O m_C", positive=True)
    phi = k * sp.Matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    mass = sp.diag(mo, mc, mo)
    u = sp.Matrix([1, -2 * mo / mc, 1])
    w2 = k / mo * (1 + 2 * mo / mc)
    assert sp.simplify(phi * u - w2 * mass * u) == sp.zeros(3, 1)
    print("antisymmetric stretch checks")
    """),
    solution("""
    # SOLUTION 4.18. A free pair: eigenvalues 0 and k/m_r.
    k, m1, m2 = sp.symbols("k m_1 m_2", positive=True)
    weighted = sp.Matrix([[k / m1, -k / sp.sqrt(m1 * m2)],
                          [-k / sp.sqrt(m1 * m2), k / m2]])
    values = list(weighted.eigenvals())
    print("eigenvalues:", [sp.simplify(v) for v in values])
    assert any(sp.simplify(v) == 0 for v in values)
    assert any(sp.simplify(v - k * (m1 + m2) / (m1 * m2)) == 0
               for v in values)
    """),
    code("""
    # EXERCISE 4.19: angular frequency (rad/fs) and stiffness (eV/Å²).
    omega = None
    k = None

    check(omega, TARGETS["4.19 omega"], name="omega")
    check(k, TARGETS["4.19 k"], name="stiffness")
    """),
    solution("""
    # SOLUTION 4.19. ω = 2π c ν̃; k = m_r ω² / FORCE_TO_ACCEL.
    omega = 2 * math.pi * 3657 * units.C_CM_PER_FS
    m_r = oscillators.reduced_mass(1.008, 15.999)  # amu
    k = m_r * omega**2 / units.FORCE_TO_ACCEL

    check(omega, TARGETS["4.19 omega"], name="omega")
    check(k, TARGETS["4.19 k"], name="stiffness")
    """),
    code("""
    # EXERCISE 4.20: the longest step (fs) and the steps per picosecond.
    step = None
    steps = None

    check(step, TARGETS["4.20 step"], name="step")
    check(steps, TARGETS["4.20 steps"], name="steps per ps")
    """),
    solution("""
    # SOLUTION 4.20. A tenth of the shortest period, 1/(c ν̃) at 3756 cm-1.
    step = 0.1 / (3756 * units.C_CM_PER_FS)
    steps = math.ceil(1000 / step)  # rounded up to a whole step


    check(step, TARGETS["4.20 step"], name="step")
    check(steps, TARGETS["4.20 steps"], name="steps per ps")
    """),
]

CELLS = (
    SETUP
    + CIRCLE
    + SPRING
    + SMALL
    + ANHARMONIC
    + DAMPED
    + DRIVEN
    + BODIES
    + MODES
    + MOLECULES
    + EXERCISES
)

if __name__ == "__main__":
    print("wrote", write(CELLS, "04_oscillations.ipynb"))
