"""Write notebooks/06_lagrange.ipynb, the companion to Chapter 6.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_06_lagrange.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/06_lagrange.ipynb
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md("""
    # Notebook 06: Lagrangian mechanics

    Working through Chapter 6: choose coordinates that obey a constraint,
    build the action, then recover the motion and the constraint force.
    Run the cells in order; the symbolic definitions are reused later.
    """),
    code("""
    %matplotlib inline
    import math

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from scipy.constants import g as G  # 9.80665 m/s²
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    from mdlab import dynamics, energy, lagrangian, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    """),
]

COORDINATES = [
    md(r"""
    ## 6.1 Coordinates that fit the problem

    For a bead on $y=h(x)$, one number fixes its position. Differentiating
    both coordinates gives its speed squared and hence its kinetic
    energy. Compare this with a pendulum, where the angle is the free
    coordinate and the rod length stays fixed.
    """),
    code("""
    time = sp.symbols("t", real=True)
    position = sp.Function("x")(time)
    height = sp.Function("h")
    bead_position = sp.Matrix([position, height(position)])
    bead_velocity = bead_position.diff(time)
    print("bead speed squared:", sp.simplify(bead_velocity.dot(bead_velocity)))

    length = sp.symbols("L", positive=True)
    angle = sp.Function("theta")(time)
    bob_position = sp.Matrix([length * sp.sin(angle), -length * sp.cos(angle)])
    bob_velocity = bob_position.diff(time)
    print("pendulum speed squared:", sp.simplify(bob_velocity.dot(bob_velocity)))
    """),
]

ACTION = [
    md(r"""
    ## 6.2 The action and Hamilton's principle

    The ball of 0.145 kg thrown up at 12 m/s, followed for 2 s. Add a bump
    $\zeta\sin(n\pi t/\tau)$ to its true path and watch the action
    $\int(K - U)\,\mathrm dt$: it is least on the true path, $\zeta = 0$.
    """),
    code("""
    MASS, V0, TAU = 0.145, 12.0, 2.0
    t = np.linspace(0.0, TAU, 2001)
    TRUE = V0 * t - 0.5 * G * t**2


    def weight(x):
        return MASS * G * x


    def trial(zeta=1.0, n=1):
        bump = np.sin(n * np.pi * t / TAU)
        zetas = np.linspace(-2.5, 2.5, 51)
        actions = [lagrangian.action(TRUE + z * bump, t, MASS, weight)
                   for z in zetas]
        here = lagrangian.action(TRUE + zeta * bump, t, MASS, weight)
        fig, (left, right) = plt.subplots(1, 2, figsize=(viz.FULL, 2.4),
                                          gridspec_kw={"wspace": 0.4})
        left.plot(t, TRUE, color=viz.ACCENT, lw=1.6, label="true")
        left.plot(t, TRUE + zeta * bump, color=viz.OCHRE, label="trial")
        left.set_xlabel("$t$ / s")
        left.set_ylabel("$x$ / m")
        left.legend(loc="lower center")
        right.plot(zetas, actions, color="black")
        right.plot(zeta, here, "o", color=viz.OCHRE)
        right.set_xlabel(r"$\\zeta$ / m")
        right.set_ylabel(r"action / J s")
        plt.show()
        print(f"action {here:.4f} J s, excess over the true path "
              f"{here - actions[25]:.4f}")


    widgets.interact(
        trial,
        zeta=widgets.FloatSlider(1.0, min=-2.5, max=2.5, step=0.1, **SLOW),
        n=widgets.IntSlider(1, min=1, max=4),
    );
    """),
    md("""
    Every bump raises the action, and the faster
    bumps (larger $n$) raise it more, as $\\zeta^2 m(n\\pi)^2/(4\\tau)$.
    For the spring over half a second (Exercise 6.3) the slowest bump
    lowers it instead: the true path is then a saddle.
    """),
]

EULER = [
    md(r"""
    ## 6.3 The Euler-Lagrange equation

    SymPy's `euler_equations` applies the equation to a Lagrangian. For a
    body on a line in a potential $U(x)$ it gives Newton's second law.
    """),
    code("""
    from sympy.calculus.euler import euler_equations

    t_, m_, g_, k_ = sp.symbols("t m g k", positive=True)
    x = sp.Function("x")(t_)
    U = sp.Function("U")
    print(euler_equations(sp.Rational(1, 2) * m_ * x.diff(t_)**2 - U(x),
                          x, t_)[0])
    print(euler_equations(sp.Rational(1, 2) * m_ * x.diff(t_)**2
                          - sp.Rational(1, 2) * k_ * x**2, x, t_)[0])
    """),
]

CONSTRAINED = [
    md(r"""
    ## 6.4 Constrained motion

    First the bead's equation derived by SymPy from
    $\mathscr L = \frac12 m(1 + h'^2)\dot x^2 - mgh(x)$, compared with the
    book's form.
    """),
    code("""
    h = sp.Function("h")
    lag = (sp.Rational(1, 2) * m_ * (1 + h(x).diff(x)**2) * x.diff(t_)**2
           - m_ * g_ * h(x))
    eq = euler_equations(lag, x, t_)[0]
    acc = sp.solve(eq, x.diff(t_, 2))[0]
    book = -(h(x).diff(x) * (g_ + h(x).diff(x, 2) * x.diff(t_)**2)
             / (1 + h(x).diff(x)**2))
    assert sp.simplify(acc - book) == 0
    print("x'' =", sp.simplify(acc))
    """),
    md(r"""
    The bead on any wire. Choose the wire and the height from which the
    bead is released on its left side, and read off the times.
    """),
    code("""
    WIRES = {
        "hill (Chapter 3)": (
            lambda x: float(energy.hill_track(x)[0]),
            lambda x: float(energy.hill_track(x)[1]),
            lambda x: 1.8 * x**2 - 1.2,
            (-2.6, 2.4),
        ),
        "parabola h = x²/2": (
            lambda x: 0.5 * x**2, lambda x: x, lambda x: 1.0, (-2.0, 2.0),
        ),
        "circular bowl R = 1.2 m": (
            lambda x: 1.2 - math.sqrt(1.44 - x**2),
            lambda x: x / math.sqrt(1.44 - x**2),
            lambda x: 1.44 / (1.44 - x**2) ** 1.5,
            (-1.19, 1.19),
        ),
    }


    def bead(wire="hill (Chapter 3)", height=1.3):
        h_, dh, d2h, (lo, hi) = WIRES[wire]
        bottom = -1.473 if wire.startswith("hill") else 0.0  # valley floor
        if h_(lo) < height:
            print("the wire is not that high on the left")
            return
        start = brentq(lambda x: h_(x) - height, lo, bottom)
        times = np.linspace(0.0, 8.0, 8001)
        xs, vs = lagrangian.solve_wire(lambda x: (h_(x), dh(x), d2h(x)),
                                       start, 0.0, times, g=G)
        back = np.nonzero((vs[1:] >= 0) & (vs[:-1] < 0))[0]
        grid = np.linspace(lo, hi, 400)
        fig, (left, right) = plt.subplots(1, 2, figsize=(viz.FULL, 2.4))
        left.plot(grid, [h_(x) for x in grid], color="black")
        left.axhline(height, **viz.THRESHOLD_STYLE)
        left.plot(start, height, "o", color=viz.ACCENT)
        left.set_xlabel("$x$ / m")
        left.set_ylabel("$h$ / m")
        right.plot(times, xs, color=viz.ACCENT)
        right.set_xlabel("$t$ / s")
        right.set_ylabel("$x$ / m")
        plt.show()
        if len(back):
            print(f"released at x = {start:.3f} m; turns at x = "
                  f"{xs.max():.3f} m; back after {times[back[0] + 1]:.3f} s")


    widgets.interact(
        bead,
        wire=list(WIRES),
        height=widgets.FloatSlider(1.3, min=0.3, max=1.6, step=0.05, **SLOW),
    );
    """),
    md("""
    On the parabola the time to come back hardly
    depends on the height for small swings; on the hill it grows sharply as
    the energy approaches the top of the hill, and jumps when the bead
    passes over it into the second valley.
    """),
]

POLAR = [
    md(r"""
    ## 6.5 Polar coordinates and conserved momenta

    The puck joined to the hole by a spring, $U = \frac12 kr^2$. Change its
    starting speed across the line to the hole and see the effective
    potential, the turning points and the path.
    """),
    code("""
    def orbit(speed=0.5):
        m, k, r0 = 0.2, 2.0, 0.5
        ell = m * r0 * speed
        e = 0.5 * m * speed**2 + 0.5 * k * r0**2
        outer_radius = max(r0, speed / math.sqrt(k / m))
        inner_radius = min(r0, speed / math.sqrt(k / m))
        r = np.linspace(0.2 * inner_radius, 1.25 * outer_radius, 600)
        u_eff = 0.5 * k * r**2 + ell**2 / (2 * m * r**2)
        t = np.linspace(0.0, 2 * math.pi / math.sqrt(k / m), 1001)
        path, _ = dynamics.solve_newton(lambda q: -k * q, m, [[r0, 0.0]],
                                        [[0.0, speed]], t,
                                        force_to_accel=1.0)
        fig, (left, right) = plt.subplots(1, 2, figsize=(viz.FULL, 2.6),
                                          gridspec_kw={"wspace": 0.4})
        left.plot(r, u_eff, color=viz.OXBLOOD, label="effective potential")
        left.axhline(e, color=viz.ACCENT, label="total energy")
        left.plot([inner_radius, outer_radius], [e, e], "o", color="black",
                  ms=3)
        left.set_ylim(0, max(0.6, 1.5 * e))
        left.set_xlabel("$r$ / m")
        left.set_ylabel("energy / J")
        left.legend(fontsize=8)
        right.plot(path[:, 0, 0], path[:, 0, 1], color=viz.ACCENT)
        right.plot(0, 0, "o", color="black", ms=3)
        right.set_aspect("equal")
        right.set_xlim(-1.2 * outer_radius, 1.2 * outer_radius)
        right.set_ylim(-1.2 * outer_radius, 1.2 * outer_radius)
        right.set_xlabel("$x$ / m")
        right.set_ylabel("$y$ / m")
        plt.show()
        radius = np.linalg.norm(path[:, 0], axis=1)
        print(f"L = {ell:.3f} kg m²/s, E = {e:.3f} J; r from "
              f"{radius.min():.4f} to {radius.max():.4f} m")


    widgets.interact(
        orbit, speed=widgets.FloatSlider(0.5, min=0.05, max=2.5, step=0.01,
                                         **SLOW),
    );
    """),
    md("""
    Near $0.5\\sqrt{k/m} = 1.58$ m/s the path is a
    circle of 0.5 m: the energy line touches the bottom of the effective
    potential. Slower or faster, it is an ellipse whose near and far
    points are the turning points of the left panel.
    """),
]

MULTIPLIERS = [
    md(r"""
    ## 6.6 Constraints and Lagrange multipliers

    The pendulum in $x$ and $y$, held at its length by the multiplier of
    Equation (6.8). Release it from any angle and compare the force of the
    rod with $mg(3\cos\theta - 2\cos\theta_0)$.
    """),
    code("""
    def pendulum(release=120.0):
        m, length = 0.5, 1.2
        theta0 = math.radians(release)
        gravity = np.array([0.0, -G])

        def rates(_t, y):
            lam = lagrangian.rod_multiplier(y[:2], y[2:], m, gravity)
            return [*y[2:], *(gravity + lam * y[:2] / m)]

        start = [length * math.sin(theta0), -length * math.cos(theta0), 0, 0]
        times = np.linspace(0.0, 4.0, 4001)
        sol = solve_ivp(rates, (0, 4.0), start, t_eval=times, rtol=1e-10,
                        atol=1e-12)
        r, v = sol.y[:2].T, sol.y[2:].T
        theta = np.arctan2(r[:, 0], -r[:, 1])
        force = [-lagrangian.rod_multiplier(ri, vi, m, gravity) * length
                 for ri, vi in zip(r, v)]
        fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=120)
        ax.plot(np.degrees(theta), np.array(force) / (m * G), ".",
                color=viz.ACCENT, ms=1)
        grid = np.linspace(-theta0, theta0, 200)
        ax.plot(np.degrees(grid), 3 * np.cos(grid) - 2 * math.cos(theta0),
                **viz.REFERENCE_STYLE)
        ax.axhline(0, **viz.THRESHOLD_STYLE)
        ax.set_xlabel("angle / degrees")
        ax.set_ylabel("force of the rod / $mg$")
        plt.show()
        print(f"length kept to within "
              f"{np.ptp(np.linalg.norm(r, axis=1)):.1e} m")


    widgets.interact(
        pendulum, release=widgets.FloatSlider(120.0, min=10.0, max=175.0,
                                              step=5.0, **SLOW),
    );
    """),
    md("""
    Released below 90° the force never goes below
    zero, and a string would do; released higher, it turns negative near
    the ends of the swing, where only a rod can hold the bob.
    """),
]

EXERCISES = [
    md("""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises' works
    every exercise in full.

    Run the collapsed answer-key cell below without opening it. After
    each exercise a collapsed cell holds a worked solution in code. The
    derivations 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.12, 6.15 and
    6.16 have their solutions checked by SymPy.
    """),
    hidden("""
    # The answer key. Each target is computed rather than typed in.
    _m, _k, _tau = 0.5, 50.0, 0.5
    _ell, _e = 0.2 * 0.4 * 0.3, 0.5 * 0.2 * 0.3**2 + 0.5 * 2.0 * 0.4**2
    _c = _ell**2 / (2 * 0.2)
    _roots = sorted(math.sqrt((_e + s * math.sqrt(_e**2 - 4 * _c)) / 2)
                    for s in (-1, 1))


    def _shape(x):
        hh, sl = energy.hill_track(x)
        return float(hh), float(sl), 1.8 * x**2 - 1.2


    _start = brentq(lambda x: float(energy.hill_track(x)[0]) - 0.8, -2.5,
                    -1.5)
    _tt = np.linspace(0.0, 4.0, 40001)
    _xx, _vv = lagrangian.solve_wire(_shape, _start, 0.0, _tt, g=G)
    _back = np.nonzero((_vv[1:] >= 0) & (_vv[:-1] < 0))[0][0] + 1
    TARGETS = {
        "6.1": np.array([1, 3, 6]),
        "6.2": math.e - (math.e - 1),
        "6.3": np.array([(_tau / 4) * (_m * (n * math.pi / _tau)**2 - _k)
                         for n in (1, 2)]),
        "6.11": np.array([_ell, _e, *_roots]),
        "6.12": (0.05**2 / (0.2 * 2.0)) ** 0.25,
        "6.13": 0.5,
        "6.14": np.array([2 * 0.5 * G, 0.5 * 0.5 * G]),
        "6.17": np.array([_xx.max(), _tt[_back]]),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code("""
    # EXERCISE 6.1: coordinates of (a) the bead on a hoop, (b) the rigid
    # pair in a plane, (c) rigid water in space.
    counts = None

    check(counts, TARGETS["6.1"], name="counts")
    """),
    solution("""
    # SOLUTION 6.1. Cartesian coordinates less one per fixed distance.
    counts = np.array([1, 2 * 2 - 1, 3 * 3 - 3])

    check(counts, TARGETS["6.1"], name="counts")
    """),
    code("""
    # EXERCISE 6.2: the integral of x e^x from 0 to 1.
    value = None

    check(value, TARGETS["6.2"], name="integral")
    """),
    solution("""
    # SOLUTION 6.2. By parts, [x e^x] - ∫ e^x; SymPy agrees.
    xs_ = sp.symbols("x")
    value = float(sp.integrate(xs_ * sp.exp(xs_), (xs_, 0, 1)))

    check(value, TARGETS["6.2"], name="integral")
    """),
    code("""
    # EXERCISE 6.3: the change of action per ζ² for n = 1 and 2 (J s/m²).
    coefficients = None

    check(coefficients, TARGETS["6.3"], name="coefficients")
    """),
    solution("""
    # SOLUTION 6.3. ½m∫η'² − ½k∫η², each integral checked by SymPy.
    tt, tau, m, k, n = sp.symbols("t tau m k n", positive=True)
    eta = sp.sin(n * sp.pi * tt / tau)
    kinetic_part = sp.integrate(eta.diff(tt)**2, (tt, 0, tau))
    potential_part = sp.integrate(eta**2, (tt, 0, tau))
    excess = (sp.Rational(1, 2) * m * kinetic_part
              - sp.Rational(1, 2) * k * potential_part)
    formula = tau / 4 * (m * (n * sp.pi / tau)**2 - k)
    for nn in (1, 2):
        assert sp.simplify((excess - formula).subs(n, nn)) == 0
    coefficients = np.array([float(formula.subs({tau: 0.5, m: 0.5, k: 50,
                                                 n: nn})) for nn in (1, 2)])

    check(coefficients, TARGETS["6.3"], name="coefficients")
    """),
    solution("""
    # SOLUTIONS 6.4 and 6.5. Free fall, and two coordinates.
    y = sp.Function("y")(t_)
    fall = euler_equations(sp.Rational(1, 2) * m_ * x.diff(t_)**2
                           - m_ * g_ * x, x, t_)[0]
    assert sp.solve(fall, x.diff(t_, 2)) == [-g_]
    lag_xy = (sp.Rational(1, 2) * m_ * (x.diff(t_)**2 + y.diff(t_)**2)
              - U(x, y))
    for c, eq in zip((x, y), euler_equations(lag_xy, [x, y], t_)):
        assert sp.simplify(sp.solve(eq, c.diff(t_, 2))[0]
                           + U(x, y).diff(c) / m_) == 0
    print("x'' = -g; m x'' = -dU/dx and m y'' = -dU/dy: checked")
    """),
    solution("""
    # SOLUTION 6.6. Near a valley bottom, h' ≈ h''(x0)(x − x0): the
    # bead's equation, linearised, is a spring with ω0² = g h''.
    eps_, x0, c2, c3 = sp.symbols("epsilon x_0 c_2 c_3")
    s_, v_ = sp.symbols("s v")  # displacement x − x0 and velocity
    slope = c2 * s_ + c3 * s_**2  # h'(x0) = 0
    curv = c2 + 2 * c3 * s_
    rhs = -slope * (g_ + curv * v_**2) / (1 + slope**2)
    linear = sp.series(rhs.subs({s_: eps_ * s_, v_: eps_ * v_}), eps_, 0,
                       2).removeO().subs(eps_, 1)
    assert sp.simplify(linear + g_ * c2 * s_) == 0
    print("x'' ≈", linear)
    """),
    solution("""
    # SOLUTION 6.7. The bead in a circular bowl is a pendulum of length R.
    # With x = R sin θ and |θ| < 90°, √(R² − x²) = R cos θ, so h = R(1 − cos θ)
    # and h' = (dh/dθ)/(dx/dθ) = tan θ.
    R, th = sp.symbols("R theta", positive=True)
    thd = sp.symbols("thetadot")
    xb, hb = R * sp.sin(th), R * (1 - sp.cos(th))
    slope_b = sp.diff(hb, th) / sp.diff(xb, th)
    assert sp.simplify(slope_b - sp.tan(th)) == 0
    kin = (sp.Rational(1, 2) * m_ * (1 + slope_b**2)
           * (sp.diff(xb, th) * thd)**2)
    pend = (sp.Rational(1, 2) * m_ * R**2 * thd**2
            - m_ * g_ * R * (1 - sp.cos(th)))
    assert sp.simplify(kin - m_ * g_ * hb - pend) == 0
    print("bowl Lagrangian = pendulum Lagrangian: checked")
    """),
    solution("""
    # SOLUTIONS 6.8 and 6.9. θ'(θ'' + (g/L) sin θ) is the rate of change
    # of the energy per mL², and q' ∂L/∂q' − L does not change.
    L_ = sp.symbols("L", positive=True)
    theta = sp.Function("theta")(t_)
    energy_ = (sp.Rational(1, 2) * m_ * L_**2 * theta.diff(t_)**2
               + m_ * g_ * L_ * (1 - sp.cos(theta)))
    rate = sp.expand(energy_.diff(t_)
                     - m_ * L_**2 * theta.diff(t_)
                     * (theta.diff(t_, 2) + g_ / L_ * sp.sin(theta)))
    assert sp.simplify(rate) == 0
    q = sp.Function("q")(t_)
    F = sp.Function("F")  # any Lagrangian of q and q'
    lag_q = F(q, q.diff(t_))
    energy_q = q.diff(t_) * lag_q.diff(q.diff(t_)) - lag_q
    el = euler_equations(lag_q, q, t_)[0].lhs  # dL/dq - d(dL/dq')/dt
    assert sp.simplify(energy_q.diff(t_) + q.diff(t_) * el) == 0
    h_w = sp.Function("h")
    kin_b = sp.Rational(1, 2) * m_ * (1 + h_w(q).diff(q)**2) * q.diff(t_)**2
    lag_b = kin_b - m_ * g_ * h_w(q)
    energy_b = q.diff(t_) * lag_b.diff(q.diff(t_)) - lag_b
    assert sp.simplify(energy_b - (kin_b + m_ * g_ * h_w(q))) == 0
    print("pendulum energy and the energy function: checked")
    """),
    solution("""
    # SOLUTION 6.10. The carts in X and r: L = ½MX'² + ½ m_r r'² − ½k(r−l)².
    m1, m2, l_ = sp.symbols("m_1 m_2 ell", positive=True)
    X, r = sp.Function("X")(t_), sp.Function("r")(t_)
    M = m1 + m2
    x1, x2 = X - m2 / M * r, X + m1 / M * r
    lag_c = (sp.Rational(1, 2) * m1 * x1.diff(t_)**2
             + sp.Rational(1, 2) * m2 * x2.diff(t_)**2
             - sp.Rational(1, 2) * k_ * (x2 - x1 - l_)**2)
    target = (sp.Rational(1, 2) * M * X.diff(t_)**2
              + sp.Rational(1, 2) * m1 * m2 / M * r.diff(t_)**2
              - sp.Rational(1, 2) * k_ * (r - l_)**2)
    assert sp.simplify(lag_c - target) == 0
    print("L in X and r: checked; X does not appear, so M X' is conserved")
    """),
    code("""
    # EXERCISE 6.11: L (kg m²/s), E (J) and the two turning distances (m),
    # nearer first.
    answers = None

    check(answers, TARGETS["6.11"], name="L, E, turning points")
    """),
    solution("""
    # SOLUTION 6.11. r⁴ − E r² + L²/(2m) = 0 with ½k = 1.
    ell = 0.2 * 0.4 * 0.3
    e = 0.5 * 0.2 * 0.3**2 + 0.5 * 2.0 * 0.4**2
    roots = np.sort(np.sqrt(np.roots([1.0, -e, ell**2 / (2 * 0.2)])))
    answers = np.array([ell, e, *roots])

    check(answers, TARGETS["6.11"], name="L, E, turning points")
    """),
    code("""
    # EXERCISE 6.12: the radius of the circular path for L = 0.05 (m).
    radius = None

    check(radius, TARGETS["6.12"], name="radius")
    """),
    solution("""
    # SOLUTION 6.12. The bottom of U_eff: k r = L²/(m r³).
    rr, kk, mm, LL = sp.symbols("r k m L", positive=True)
    bottom = sp.solve(sp.diff(kk * rr**2 / 2 + LL**2 / (2 * mm * rr**2), rr),
                      rr)[0]
    omega = sp.simplify(LL / (mm * bottom**2))
    assert sp.simplify(omega - sp.sqrt(kk / mm)) == 0
    radius = float(bottom.subs({LL: 0.05, mm: 0.2, kk: 2.0}))

    check(radius, TARGETS["6.12"], name="radius")
    """),
    code("""
    # EXERCISE 6.13: the largest value of xy on the unit circle.
    largest = None

    check(largest, TARGETS["6.13"], name="largest xy")
    """),
    solution("""
    # SOLUTION 6.13. ∇(xy) = λ∇(x² + y²) with x² + y² = 1.
    xa, ya, la = sp.symbols("x y lambda", real=True)
    sols = sp.solve([ya - 2 * la * xa, xa - 2 * la * ya, xa**2 + ya**2 - 1],
                    [xa, ya, la], dict=True)
    largest = max(float(s[xa] * s[ya]) for s in sols)

    check(largest, TARGETS["6.13"], name="largest xy")
    """),
    code("""
    # EXERCISE 6.14: the force of the rod at the bottom and at release (N).
    forces = None

    check(forces, TARGETS["6.14"], atol=1e-9, name="forces")
    """),
    solution("""
    # SOLUTION 6.14. F = mg(3 cos θ − 2 cos θ0) with θ0 = 60°.
    def force(theta, theta0=math.pi / 3, m=0.5):
        return m * G * (3 * math.cos(theta) - 2 * math.cos(theta0))


    forces = np.array([force(0.0), force(math.pi / 3)])

    check(forces, TARGETS["6.14"], atol=1e-9, name="forces")
    """),
    solution("""
    # SOLUTION 6.15. The force is least at the turning points, mg cos θ0.
    th, th0 = sp.symbols("theta theta_0")
    force = 3 * sp.cos(th) - 2 * sp.cos(th0)  # per mg
    assert sp.simplify(force.subs(th, th0) - sp.cos(th0)) == 0
    # the force falls as |θ| grows, so it is least at the turning points
    assert sp.simplify(force.diff(th) + 3 * sp.sin(th)) == 0
    print("least force mg cos θ0: taut throughout only if θ0 ≤ 90°")
    """),
    solution("""
    # SOLUTION 6.16. The gradients of ½(|rj − ri|² − d²).
    ri = sp.Matrix(sp.symbols("x_i y_i z_i"))
    rj = sp.Matrix(sp.symbols("x_j y_j z_j"))
    d = sp.symbols("d")
    gg = ((rj - ri).dot(rj - ri) - d**2) / 2
    assert sp.Matrix([gg.diff(c) for c in rj]) == rj - ri
    assert sp.Matrix([gg.diff(c) for c in ri]) == -(rj - ri)
    print("equal and opposite, along the bond: checked")
    """),
    code("""
    # EXERCISE 6.17: where the bead released at 0.8 m turns (x, m) and
    # when it is back at its start (s).
    answers = None

    check(answers, TARGETS["6.17"], name="turning point, time")
    """),
    solution("""
    # SOLUTION 6.17. solve_wire from the left point at 0.8 m.
    def shape(x):
        hh, sl = energy.hill_track(x)
        return float(hh), float(sl), 1.8 * x**2 - 1.2


    start = brentq(lambda x: float(energy.hill_track(x)[0]) - 0.8, -2.5,
                   -1.5)
    times = np.linspace(0.0, 4.0, 40001)
    xs, vs = lagrangian.solve_wire(shape, start, 0.0, times, g=G)
    back = np.nonzero((vs[1:] >= 0) & (vs[:-1] < 0))[0][0] + 1
    answers = np.array([xs.max(), times[back]])

    check(answers, TARGETS["6.17"], name="turning point, time")
    """),
]

CELLS = (SETUP + COORDINATES + ACTION + EULER + CONSTRAINED + POLAR
         + MULTIPLIERS + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "06_lagrange.ipynb"))
