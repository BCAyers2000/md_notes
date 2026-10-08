"""Write notebooks/09_integrators.ipynb, the companion to Chapter 9.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_09_integrators.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/09_integrators.ipynb

Section 9.7 reads the runs cached by scripts/ch09_integrators/fig_long.py;
run that script first if data/ch09_integrators/long_runs.npz is missing.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 09: Integrating the equations of motion

    Working through Chapter 9: compare integration errors, reverse a run,
    and find where a step becomes too long. The section numbers match the
    book. Run the cells in order; the later calculations reuse the stepper
    and force functions defined near the start.
    """),
    code(r"""
    %matplotlib inline
    import math
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp

    from mdlab import hamiltonian, integrators, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    RNG = np.random.default_rng(9)  # the fixed seed of every random draw


    def spring(r):
        return -r  # m = k = 1, so omega = 1


    def march(method, dt, n, q0=1.0, v0=0.0, force=spring):
        r, v, f = np.array([[q0]]), np.array([[v0]]), None
        qs, vs = [q0], [v0]
        for _ in range(n):
            if method == "velocity Verlet":
                r, v, f = integrators.velocity_verlet_step(
                    r, v, [1.0], force, dt, f, 1.0)
            elif method == "forward Euler":
                r, v = integrators.euler_step(r, v, [1.0], force, dt, 1.0)
            else:
                r, v = integrators.rk4_step(r, v, [1.0], force, dt, 1.0)
            qs.append(r.item())
            vs.append(v.item())
        return np.array(qs), np.array(vs)
    """),
]

STEPS = [
    md(r"""
    ## 9.1 and 9.2 Steps, errors and the Verlet method

    Compare forward Euler with velocity Verlet for the spring. The fixed
    plotting window keeps the Verlet ellipse visible as Euler spirals
    outwards; the printed energy ratio records how far Euler has grown.
    """),
    code(r"""
    def spiral(omega_dt=0.3, periods=3):
        n = int(periods * 2 * math.pi / omega_dt)
        fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=120)
        s = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(s), np.sin(s), **viz.REFERENCE_STYLE)
        for name, colour in (("forward Euler", viz.METHOD["forward Euler"]),
                             ("velocity Verlet", viz.ACCENT)):
            q, v = march(name, omega_dt, n)
            ax.plot(q, v, ".-", color=colour, ms=3, lw=0.5, label=name)
            print(f"{name}: energy after {n} steps x "
                  f"{(q[-1]**2 + v[-1]**2):.4g}")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xlabel("$q$")
        ax.set_ylabel("$v$")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16),
                  ncols=2, fontsize=8)
        ax.set_title(r"central window: $|q|, |v| \leq 3$", fontsize=8)
        plt.show()


    widgets.interact(
        spiral,
        omega_dt=widgets.FloatSlider(0.3, min=0.02, max=1.0, step=0.02,
                                     **SLOW),
        periods=widgets.IntSlider(3, min=1, max=20, **SLOW),
    );
    """),
    md(r"""
    Euler's points spiral outwards for every step,
    faster for longer ones; Verlet's stay on a closed curve close to the
    true circle, however many periods are followed.
    """),
    code(r"""
    # The order of each method: the error of the pendulum's angle after a
    # time of 4, from q = 1, p = 0.5, against the step.
    def pendulum(r):
        return -np.sin(r)


    q_ref = hamiltonian.flow(lambda q, p: np.sin(q), lambda q, p: p, [1.0],
                             [0.5], [0.0, 4.0])[0][-1, 0]
    steps = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])  # those of Fig. 9.1
    for name in ("forward Euler", "velocity Verlet", "RK4"):
        errors = np.array([march(name, dt, int(round(4 / dt)), 1.0, 0.5,
                                 pendulum)[0][-1] - q_ref for dt in steps])
        slope = np.polyfit(np.log(steps), np.log(np.abs(errors)), 1)[0]
        pairs = np.log2(np.abs(errors[:-1] / errors[1:]))
        print(f"{name}: fitted order {slope:.2f}; between successive steps "
              f"{pairs.round(1)}; signs {np.sign(errors).astype(int)}")
    """),
    code(r"""
    # Checked against ASE: 12 Lennard-Jones atoms in reduced units through
    # mdlab's velocity Verlet and ASE's. With masses in amu, energies in eV
    # and lengths in Å, ASE's own time unit is the reduced one, so the step
    # needs no conversion.
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.md.verlet import VelocityVerlet
    from mdlab import potentials

    grid = np.array([[i, j, k] for i in range(3) for j in range(2)
                     for k in range(2)], float)
    start = 1.12 * grid + 0.03 * np.random.default_rng(9).normal(size=(12, 3))
    v_start = 0.1 * np.random.default_rng(10).normal(size=(12, 3))
    lj = potentials.with_cutoff(potentials.lennard_jones, 6.0, "shift")
    out = integrators.integrate(
        lambda r: potentials.pair_energy_forces(r, lj)[:2], np.ones(12),
        start, v_start, 0.005, 4000, every=250, force_to_accel=1.0)
    atoms = Atoms("H12", positions=start)
    atoms.set_masses(np.ones(12))
    atoms.set_velocities(v_start)
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=6.0, smooth=False)
    dynamics_ase = VelocityVerlet(atoms, timestep=0.005)
    for k in range(1, 17):
        dynamics_ase.run(250)
        gap = np.abs(atoms.get_positions() - out["positions"][k]).max()
        if k in (1, 2, 4, 8, 16):
            print(f"after {250 * k:5d} steps, positions differ by "
                  f"{gap:.1e} sigma")
    """),
    md(r"""
    The two codes agree to the rounding of the
    arithmetic at first, about $10^{-14}\sigma$. The difference then grows
    by roughly a constant factor for every interval of time: the motion of
    many atoms is chaotic, and two paths that differ only in their rounding
    separate in the same way as two that start apart. Codes are compared
    over short runs.
    """),
]

REVERSAL = [
    md(r"""
    ## 9.3 Reversal and volume

    Run forwards, reverse the velocity, run back: how far from the start
    does each method land?
    """),
    code(r"""
    def back_and_forth(n_steps=100, dt=0.1):
        for name in ("forward Euler", "velocity Verlet"):
            q, v = march(name, dt, n_steps, 1.0, 0.5, pendulum)
            qb, vb = march(name, dt, n_steps, q[-1], -v[-1], pendulum)
            print(f"{name}: lands {abs(qb[-1] - 1.0):.2e} from the start, "
                  f"velocity {-vb[-1]:.6f} (started 0.5)")


    widgets.interact(
        back_and_forth,
        n_steps=widgets.IntSlider(100, min=1, max=2000, **SLOW),
        dt=widgets.FloatSlider(0.1, min=0.01, max=0.5, step=0.01, **SLOW),
    );
    """),
    md(r"""
    Velocity Verlet retraces its steps in exact arithmetic. Here the
    remaining difference is round-off, which can grow over longer runs;
    compare the printed error at different settings. Forward
    Euler misses after a single step, by $8\times10^{-3}$ at
    $\delta t = 0.1$, and once the energy it gains carries the pendulum
    over the top it misses by tens or hundreds of radians.
    """),
]

SPLITTING = [
    md(r"""
    ## 9.4 Splitting the Liouville operator

    SymPy with operators that do not commute: the symmetric product of
    exponentials agrees with the exponential of the sum to second order,
    the unsymmetric one only to first.
    """),
    code(r"""
    X, Y = sp.symbols("X Y", commutative=False)
    h = sp.symbols("h")


    def exp_series(x, order=2):
        return sum(x**n / sp.factorial(n) for n in range(order + 1))


    def truncate(expr, order=2):
        expr = sp.expand(expr)
        return sum(expr.coeff(h, n) * h**n for n in range(order + 1))


    exact = truncate(exp_series(h * (X + Y)))
    symmetric = truncate(exp_series(h * X / 2) * exp_series(h * Y)
                         * exp_series(h * X / 2))
    unsymmetric = truncate(exp_series(h * X) * exp_series(h * Y))
    print("symmetric - exact:", sp.simplify(symmetric - exact))
    print("unsymmetric - exact:", sp.simplify(unsymmetric - exact))
    """),
]

SHADOW = [
    md(r"""
    ## 9.5 The energy the method keeps

    The energy $H$ and the shadow energy
    $\tilde H = p^2/2 + (1 - \omega^2\delta t^2/4)q^2/2$ along velocity
    Verlet, with $m = k = \omega = 1$ and a step below the stability limit.
    """),
    code(r"""
    def shadow(omega_dt=0.5):
        q, v = march("velocity Verlet", omega_dt, 300)
        h_true = 0.5 * v**2 + 0.5 * q**2
        h_shadow = 0.5 * v**2 + 0.5 * (1 - omega_dt**2 / 4) * q**2
        fig, ax = plt.subplots(figsize=(4.5, 2.6), dpi=120)
        ax.plot(h_true / h_true[0], color=viz.ACCENT, lw=0.8, label="$H$")
        ax.plot(h_shadow / h_shadow[0], color="black", label=r"$\tilde H$")
        ax.set_xlabel("step")
        ax.set_ylabel("energy / initial value")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=2)
        plt.show()
        print(f"H between {h_true.min() / h_true[0]:.4f} and "
              f"{h_true.max() / h_true[0]:.4f}; predicted lower edge "
              f"{1 - omega_dt**2 / 4:.4f}; shadow constant to "
              f"{np.ptp(h_shadow) / h_shadow[0]:.1e}")


    widgets.interact(
        shadow,
        omega_dt=widgets.FloatSlider(0.5, min=0.05, max=1.95, step=0.05,
                                     **SLOW),
    );
    """),
    md(r"""
    $H$ never rises above its start and dips to at
    most $1 - \omega^2\delta t^2/4$ of it, reached when a step lands near
    $q = 0$; at $\omega\delta t = 1$ the states repeat every six steps and
    miss $q = 0$, so the band is narrower. $\tilde H$ stays constant to
    round-off right up to the limit.
    """),
]

STABILITY = [
    md(r"""
    ## 9.6 Stability and the size of the step

    Push $\omega\,\delta t$ towards 2 and past it.
    """),
    code(r"""
    def stability(omega_dt=1.9):
        q, v = march("velocity Verlet", omega_dt, 200)
        fig, ax = plt.subplots(figsize=(4.5, 2.6), dpi=120)
        ax.semilogy(np.abs(q) + 1e-300, color=viz.ACCENT)
        ax.set_xlabel("step")
        ax.set_ylabel("$|q|$")
        plt.show()
        print(f"largest |q| {np.abs(q).max():.3e}; trace "
              f"{2 - omega_dt**2:.4f}")


    widgets.interact(
        stability,
        omega_dt=widgets.FloatSlider(1.9, min=1.5, max=2.2, step=0.01,
                                     **SLOW),
    );
    """),
    md(r"""
    Up to 1.99 the motion stays at $|q| \le 1$;
    from 2.01 it grows by a fixed factor every step, the trace having
    passed $-2$. Exactly at 2, this special start at rest alternates between
    $q = 1$ and $q = -1$. A nonzero starting velocity would grow without
    bound there, so the endpoint is not stable for general initial states.
    """),
]

LONG = [
    md(r"""
    ## 9.7 Long runs and rough forces

    The cached runs of the 32-atom Lennard-Jones cluster over $200\tau$.
    """),
    code(r"""
    CACHE = Path("..") / "data" / "ch09_integrators" / "long_runs.npz"
    runs = np.load(CACHE)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 2.6), dpi=120,
                                        gridspec_kw={"wspace": 0.4})
    for name, ax, colour, style, label in (
            ("vv", left, viz.ACCENT, "-", "velocity Verlet"),
            ("rk4", left, viz.OCHRE, "--", "RK4"),
            ("switch", right, viz.ACCENT, "-", "switched"),
            ("shift", right, viz.OCHRE, "--", "shifted")):
        e = runs[f"e_{name}"]
        ax.plot(runs[f"t_{name}"], e - e[0], color=colour, ls=style,
                lw=0.8, label=label)
        print(f"{name}: force calls {int(runs[f'calls_{name}'])}, spread "
              f"{np.ptp(e):.2e}, end {e[-1] - e[0]:+.2e}")
    for ax in (left, right):
        ax.set_xlabel(r"time / $\tau$")
        ax.legend(fontsize=8)
        ax.set_ylabel(r"$[E - E(0)] / \varepsilon$")
    plt.show()
    """),
]

EXERCISES = [
    md(r"""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises' works
    every exercise in full.

    Run the collapsed answer-key cell below without opening it. After
    each exercise a collapsed cell holds a worked solution in code. The
    derivations 9.1, 9.3 to 9.14 and 9.19 have their solutions checked by
    SymPy.
    """),
    hidden(r"""
    # The answer key. Each target is computed from the exercise's data.
    _w_oh = 2 * math.pi / 8.88
    _w_li = 2 * math.pi / 170.3
    _t = 2 - 2.05**2
    _x = 0.2
    _x7 = _w_oh * 0.7
    TARGETS = {
        "9.2": np.array([math.log2(16), math.log2(16) + 1]),
        "9.15": abs(_t / 2 - math.sqrt(_t**2 / 4 - 1)),
        "9.16": math.sqrt(0.12) / _w_oh,
        "9.17": np.array([0.2 / _w_li, 2 / _w_li]),
        "9.18": 2 * math.pi / 0.2,
        "9.19": 1 - (1 - _x**6 / 72 + _x**8 / 576) ** 10000,
        "9.20": _w_oh * 5,
        "9.21": 3756 * math.acos(1 - _x7**2 / 2) / _x7,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    solution(r"""
    # SOLUTION 9.1. Euler for free fall: x_n and the error g t_f dt / 2.
    n, dt, g, x0, v0 = sp.symbols("n delta_t g x_0 v_0", positive=True)
    j = sp.symbols("j", integer=True)
    x_n = x0 + dt * sp.summation(v0 - j * g * dt, (j, 0, n - 1))
    euler_x = x0 + v0 * n * dt - g * dt**2 * n * (n - 1) / 2
    assert sp.simplify(x_n - euler_x) == 0
    exact_fall = x0 + v0 * n * dt - g * (n * dt) ** 2 / 2
    assert sp.simplify(x_n - exact_fall - g * (n * dt) * dt / 2) == 0
    print("x_n and the error g t_f dt/2: checked")
    """),
    code(r"""
    # EXERCISE 9.2: the order and the power of dt in the local error when
    # halving dt divides the error by 16.
    order_local = None

    check(order_local, TARGETS["9.2"], name="order, local power")
    """),
    solution(r"""
    # SOLUTION 9.2. 16 = 2**4.
    order_local = np.array([4, 5])

    check(order_local, TARGETS["9.2"], name="order, local power");
    """),
    solution(r"""
    # SOLUTIONS 9.3 to 9.6. Taylor series of the motion: Störmer's error,
    # the central velocity, the local error of velocity Verlet, leapfrog.
    t, d = sp.symbols("t delta")
    x = sp.Function("x")
    ahead = sp.series(x(t + d), d, 0, 6).removeO().doit()
    behind = sp.series(x(t - d), d, 0, 6).removeO().doit()
    total = sp.expand(ahead + behind)
    a4 = sp.diff(x(t), t, 4)
    assert sp.simplify(total.coeff(d, 4) - a4 / 12) == 0
    central = sp.expand((ahead - behind) / (2 * d))
    assert sp.simplify(central.coeff(d, 2) - sp.diff(x(t), t, 3) / 6) == 0
    vv = x(t) + d * sp.diff(x(t), t) + d**2 / 2 * sp.diff(x(t), t, 2)
    assert sp.simplify(sp.expand(ahead - vv).coeff(d, 3)
                       - sp.diff(x(t), t, 3) / 6) == 0
    xs = sp.symbols("x0:3")
    vh, a1 = sp.symbols("v_h a_1")
    x2 = xs[1] + d * (vh + d * a1)  # leapfrog with v_h = (x1 - x0)/d
    stormer = 2 * xs[1] - xs[0] + d**2 * a1
    assert sp.simplify(x2.subs(vh, (xs[1] - xs[0]) / d) - stormer) == 0
    print("Störmer error ä dt⁴/12, central velocity error ȧ dt²/6, "
          "VV local error ȧ dt³/6, leapfrog = Störmer: checked")
    """),
    solution(r"""
    # SOLUTIONS 9.7 and 9.8. Euler does not run back; J^T Ω J = det(J) Ω.
    hh = sp.symbols("h_e", positive=True)
    q1, v1 = 1, -hh  # one Euler step of the spring from (1, 0)
    q2, v2 = q1 + hh * (-v1), -v1 - hh * q1
    assert (sp.simplify(q2 - (1 + hh**2)), sp.simplify(v2)) == (0, 0)
    a_, b_, c_, d_ = sp.symbols("a b c d")
    J = sp.Matrix([[a_, b_], [c_, d_]])
    Om = sp.Matrix([[0, 1], [-1, 0]])
    assert sp.simplify(J.T * Om * J - J.det() * Om) == sp.zeros(2, 2)
    print("Euler lands at (1 + h², 0); J^T Ω J = det(J) Ω: checked")
    """),
    solution(r"""
    # SOLUTIONS 9.9 to 9.11. The unsymmetric product, which acts first,
    # and the drift of q³.
    diff = sp.simplify(unsymmetric - exact)
    assert sp.expand(diff - h**2 * (X * Y - Y * X) / 2) == 0
    q, p, m = sp.symbols("q p m")
    F = sp.Function("F")(q)
    L_r = lambda f: p / m * sp.diff(f, q)  # noqa: E731
    L_p = lambda f: F * sp.diff(f, p)  # noqa: E731


    def exp_op(L, f, order=4):
        total, term = 0, f
        for n_ in range(order + 1):
            total += h**n_ / sp.factorial(n_) * term
            term = L(term)
        return total


    assert sp.simplify(exp_op(L_p, exp_op(L_r, q))
                       - (q + h * (p + h * F) / m)) == 0
    assert sp.expand(exp_op(L_r, q**3) - (q + h * p / m) ** 3) == 0
    print("unsymmetric error h²(XY − YX)/2, kick then drift, drift of q³: "
          "checked")
    """),
    solution(r"""
    # SOLUTIONS 9.12 to 9.14. The shadow energy is kept; symplectic Euler
    # keeps its own, and its band; the step's matrix, det 1, trace 2 − ω²δt².
    k, mm, hs = sp.symbols("k m h_s", positive=True)
    qq, pp = sp.symbols("q p")
    w2 = k / mm
    ph = pp - hs / 2 * k * qq
    qn = qq + hs * ph / mm
    pn = ph - hs / 2 * k * qn
    Ht = lambda a, b: (b**2 / (2 * mm)  # noqa: E731
                       + k * a**2 / 2 * (1 - w2 * hs**2 / 4))
    assert sp.simplify(sp.expand(Ht(qn, pn) - Ht(qq, pp))) == 0
    q1, p1 = sp.symbols("q' p'")
    mirror = ((p1 + hs / 2 * k * q1) ** 2 / (2 * mm) + k * q1**2 / 2
              - hs * k / (2 * mm) * q1 * (p1 + hs / 2 * k * q1))
    assert sp.expand(mirror - Ht(q1, p1)) == 0
    pse = pp - hs * k * qq  # symplectic Euler: kick, then drift
    qse = qq + hs * pse / mm
    Hse = lambda a, b: (b**2 / (2 * mm) + k * a**2 / 2  # noqa: E731
                        - hs * k / (2 * mm) * a * b)
    assert sp.expand(Hse(qse, pse) - Hse(qq, pp)) == 0
    A_, phi, w = sp.symbols("A phi omega", positive=True)
    on_circle = {qq: A_ * sp.cos(phi), pp: mm * w * A_ * sp.sin(phi)}
    H_true = (pp**2 / (2 * mm) + k * qq**2 / 2).subs(on_circle)
    kept = Hse(qq, pp).subs(on_circle)
    band = H_true * (1 - w * hs * sp.sin(2 * phi) / 2)
    assert sp.simplify((kept - band).subs(k, mm * w**2)) == 0
    assert sp.simplify(qn - ((1 - w2 * hs**2 / 2) * qq + hs / mm * pp)) == 0
    assert sp.simplify(pn - (-k * hs * (1 - w2 * hs**2 / 4) * qq
                             + (1 - w2 * hs**2 / 2) * pp)) == 0
    M = sp.Matrix([[sp.diff(qn, qq), sp.diff(qn, pp)],
                   [sp.diff(pn, qq), sp.diff(pn, pp)]])
    assert sp.simplify(M.det()) == 1
    assert sp.simplify(M.trace() - (2 - w2 * hs**2)) == 0
    print("shadow energy kept, and its mirror form; symplectic Euler keeps "
          "H − (hk/2m)qp, band 1/(1 ± ωh/2); step matrix, det 1, trace "
          "2 − ω²h²: checked")
    """),
    code(r"""
    # EXERCISE 9.15: the growth factor per step at omega dt = 2.05.
    growth = None

    check(growth, TARGETS["9.15"], name="growth")
    """),
    solution(r"""
    # SOLUTION 9.15. lambda² − (trace) lambda + 1 = 0, trace 2 − 2.05².
    trace = 2 - 2.05**2
    growth = abs(trace / 2 - math.sqrt(trace**2 / 4 - 1))
    q_, _ = march("velocity Verlet", 2.05, 60)
    print("measured growth over the last steps:",
          round(abs(q_[-1] / q_[-3]) ** 0.5, 4))

    check(growth, TARGETS["9.15"], name="growth");
    """),
    code(r"""
    # EXERCISE 9.16: the longest step (fs) for a 3% band on the O-H stretch.
    dt_water = None

    check(dt_water, TARGETS["9.16"], name="step")
    """),
    solution(r"""
    # SOLUTION 9.16. omega² dt²/4 = 0.03.
    dt_water = math.sqrt(0.12) / (2 * math.pi / 8.88)

    check(dt_water, TARGETS["9.16"], name="step");
    """),
    code(r"""
    # EXERCISE 9.17: the 1% step and the stability limit (fs) for lithium.
    lithium = None

    check(lithium, TARGETS["9.17"], name="steps")
    """),
    solution(r"""
    # SOLUTION 9.17. omega dt = 0.2 and omega dt = 2.
    w = 2 * math.pi / 170.3
    lithium = np.array([0.2 / w, 2 / w])

    check(lithium, TARGETS["9.17"], name="steps");
    """),
    code(r"""
    # EXERCISE 9.18: steps per period for a 1% band.
    per_period = None

    check(per_period, TARGETS["9.18"], name="steps per period")
    """),
    solution(r"""
    # SOLUTION 9.18. omega dt = 0.2, period 2 pi/omega.
    per_period = 2 * math.pi / 0.2

    check(per_period, TARGETS["9.18"], name="steps per period");
    """),
    code(r"""
    # EXERCISE 9.19: the fraction of the energy RK4 loses in 10 000 steps at
    # omega dt = 0.2.
    lost = None

    check(lost, TARGETS["9.19"], name="fraction lost")
    """),
    solution(r"""
    # SOLUTION 9.19. c² + s², checked by SymPy, and a run of RK4.
    xx = sp.symbols("x")
    c = 1 - xx**2 / 2 + xx**4 / 24
    s = xx - xx**3 / 6
    assert sp.expand(c**2 + s**2 - (1 - xx**6 / 72 + xx**8 / 576)) == 0
    qx, ux = sp.symbols("q u")
    after = (c * qx + s * ux) ** 2 + (-s * qx + c * ux) ** 2
    assert sp.expand(after - (c**2 + s**2) * (qx**2 + ux**2)) == 0
    lost = 1 - float((1 - xx**6 / 72 + xx**8 / 576).subs(xx, 0.2)) ** 10000
    q_, v_ = march("RK4", 0.2, 10000)
    print("run of RK4:", round(1 - (q_[-1] ** 2 + v_[-1] ** 2), 5))

    check(lost, TARGETS["9.19"], name="fraction lost");
    """),
    code(r"""
    # EXERCISE 9.20: omega dt for the O-H stretch with a 5 fs step.
    long_step = None

    check(long_step, TARGETS["9.20"], name="omega dt")
    """),
    solution(r"""
    # SOLUTION 9.20. Beyond 2: velocity Verlet would blow up.
    long_step = 2 * math.pi / 8.88 * 5

    check(long_step, TARGETS["9.20"], name="omega dt");
    """),
    code(r"""
    # EXERCISE 9.21: the wavenumber (cm⁻¹) at which the 3756 cm⁻¹ stretch
    # appears with dt = 0.7 fs.
    shifted = None

    check(shifted, TARGETS["9.21"], name="wavenumber")
    """),
    solution(r"""
    # SOLUTION 9.21. cos(omega~ dt) = 1 − omega² dt²/2, measured too.
    x7 = 2 * math.pi / 8.88 * 0.7
    shifted = 3756 * math.acos(1 - x7**2 / 2) / x7
    q_, _ = march("velocity Verlet", x7, 4000)
    crossings = np.nonzero((q_[1:] < 0) & (q_[:-1] >= 0))[0]
    measured = 2 * math.pi * (len(crossings) - 1) / (
        (crossings[-1] - crossings[0]) * x7)
    print("measured frequency ratio:", round(measured, 4))

    check(shifted, TARGETS["9.21"], name="wavenumber");
    """),
]

CELLS = (SETUP + STEPS + REVERSAL + SPLITTING + SHADOW + STABILITY + LONG
         + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "09_integrators.ipynb"))
