"""Write notebooks/07_hamilton.ipynb, the companion to Chapter 7.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_07_hamilton.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/07_hamilton.ipynb
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 07: Hamiltonian mechanics and phase space

    Working through Chapter 7 with the pendulum as a running example.
    We follow individual states, then regions of phase space, and compare
    the exact flow with discrete steps. Run the cells in order; the
    Hamiltonian and symbolic tools are reused in the exercises.
    """),
    code(r"""
    %matplotlib inline
    import math

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation
    from scipy.constants import g as G  # 9.80665 m/s²
    from scipy.integrate import solve_ivp

    from mdlab import hamiltonian, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    RNG = np.random.default_rng(7)  # the fixed seed of every random draw

    # the pendulum of the chapter: 0.5 kg on a 1.2 m rod, p = mL²θ̇
    MASS, LENGTH = 0.5, 1.2
    INERTIA, MGL = MASS * LENGTH**2, MASS * G * LENGTH
    OMEGA0 = math.sqrt(G / LENGTH)


    def pend_dh_dq(q, p):
        return MGL * np.sin(q)


    def pend_dh_dp(q, p):
        return p / INERTIA


    def pend_energy(q, p):
        return p**2 / (2 * INERTIA) + MGL * (1 - np.cos(q))
    """),
]

LEGENDRE = [
    md(r"""
    ## 7.1 From velocities to momenta

    The Legendre transform $f^*(p) = pv - f(v)$ at the $v$ where
    $f'(v) = p$ is the largest gap between the line $pv$ and the curve
    $f$. Choose a function and a slope; the cell finds the largest gap by
    brute force on a fine grid and compares it with the formula.
    """),
    code(r"""
    FUNCTIONS = {
        "½ m v², m = 0.5": (lambda v: 0.25 * v**2, lambda p: p**2),
        "e^v": (np.exp, lambda p: p * np.log(p) - p),
        "v⁴/4": (lambda v: v**4 / 4, lambda p: 0.75 * p ** (4 / 3)),
    }


    def legendre(function="½ m v², m = 0.5", p=1.0):
        f, f_star = FUNCTIONS[function]
        v = np.linspace(-3.0, 7.0, 100001)
        gap = p * v - f(v)
        best = np.argmax(gap)
        fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=120)
        ax.plot(v, f(v), color="black", label="$f(v)$")
        ax.plot(v, p * v, **viz.REFERENCE_STYLE, label="$pv$")
        ax.plot(v, p * v - gap[best], color=viz.ACCENT, lw=0.8,
                label="tangent of slope $p$")
        ax.annotate("", xy=(v[best], f(v[best])), xytext=(v[best], p * v[best]),
                    arrowprops=dict(arrowstyle="<->", color=viz.ACCENT))
        ax.set_xlim(-2, max(4, v[best] + 1))
        ax.set_ylim(-3, max(6, f(v[best]) + 2))
        ax.set_xlabel("$v$")
        ax.legend(loc="upper left")
        plt.show()
        print(f"largest gap {gap[best]:.4f} at v = {v[best]:.4f}; "
              f"formula f*(p) = {f_star(p):.4f}")


    widgets.interact(
        legendre, function=list(FUNCTIONS),
        p=widgets.FloatSlider(1.0, min=0.2, max=3.0, step=0.05, **SLOW),
    );
    """),
    md(r"""
    The arrow, the largest gap, always sits where
    the tangent of slope $p$ touches the curve, and its length matches
    the formula. The tangent meets the vertical axis at $-f^*(p)$.
    """),
]

EQUATIONS = [
    md(r"""
    ## 7.2 Hamilton's equations

    The pendulum's phase portrait as contours of
    $H = p^2/(2mL^2) + mgL(1 - \cos\theta)$. Set a starting state with
    the sliders; the cell follows it for 6 s with `hamiltonian.flow`.
    The plot shows two periods of the angle. A rotating pendulum continues
    beyond this window; its angle is not folded back at the edges.
    """),
    code(r"""
    def launch(theta0=1.0, p0=2.0):
        theta = np.linspace(-2 * np.pi, 2 * np.pi, 400)
        p = np.linspace(-7, 7, 300)
        TH, PP = np.meshgrid(theta, p)
        fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=120)
        ax.contour(TH, PP, pend_energy(TH, PP) / MGL,
                   levels=[0.3, 0.9, 1.5, 2.8, 4.2], colors="0.75",
                   linewidths=0.6)
        ax.contour(TH, PP, pend_energy(TH, PP) / MGL, levels=[2.0],
                   colors=viz.ACCENT, linewidths=1.2)
        times = np.linspace(0, 6, 1201)
        q, pp = hamiltonian.flow(pend_dh_dq, pend_dh_dp, [theta0], [p0], times)
        ax.plot(q[:, 0], pp[:, 0], color=viz.OCHRE, lw=1.6)
        ax.plot(theta0, p0, "o", color="black")
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-7, 7)
        ax.set_xlabel(r"$\theta$ / rad")
        ax.set_ylabel(r"$p$ / kg m$^2$ s$^{-1}$")
        plt.show()
        e = pend_energy(theta0, p0)
        drift = np.ptp(pend_energy(q[:, 0], pp[:, 0]))
        print(f"H = {e:.4f} J = {e / MGL:.3f} mgL "
              f"({'swings' if e < 2 * MGL else 'goes over the top'}); "
              f"H changes along the path by {drift:.1e} J")


    widgets.interact(
        launch,
        theta0=widgets.FloatSlider(1.0, min=-3.1, max=3.1, step=0.05, **SLOW),
        p0=widgets.FloatSlider(2.0, min=-6.0, max=6.0, step=0.05, **SLOW),
    );
    """),
    md(r"""
    Every path lies on a contour of $H$, and $H$
    changes along it only by the error of the solver. Start at
    $\theta = 0$ with $p$ near $4.117$, the separatrix: the path creeps
    towards $\pm\pi$ and hesitates there for longer the closer $p$ is.
    """),
    code(r"""
    # The creep along the separatrix: π − θ halves every ln 2/ω0.
    times = np.linspace(0.0, 2.5, 11)
    q, _ = hamiltonian.flow(pend_dh_dq, pend_dh_dp, [0.0],
                            [2 * INERTIA * OMEGA0], times)
    gap = np.pi - q[:, 0]
    for t, g_ in zip(times, gap):
        print(f"t = {t:.2f} s: pi − theta = {g_:.5f} rad")
    print(f"ln 2/omega0 = {math.log(2) / OMEGA0:.4f} s")
    """),
]

BRACKETS = [
    md(r"""
    ## 7.3 Poisson brackets and the Liouville operator

    SymPy computes brackets and the terms $\hat{L}^n A$ of the series
    $A(t) = \sum_n (t^n/n!)\,\hat{L}^nA$.
    """),
    code(r"""
    q, p = sp.symbols("q p", real=True)
    t, m, k, g = sp.symbols("t m k g", positive=True)


    def bracket(a, b, pairs=((q, p),)):
        return sum(sp.diff(a, x) * sp.diff(b, y) - sp.diff(a, y) * sp.diff(b, x)
                   for x, y in pairs)


    def series(a, h, n_terms):
        term, total = a, 0
        for n in range(n_terms):
            total += t**n / sp.factorial(n) * term
            term = sp.expand(bracket(term, h))
        return sp.simplify(total)


    spring = p**2 / (2 * m) + k * q**2 / 2
    ball = p**2 / (2 * m) + m * g * q
    print("{q, p} =", bracket(q, p))
    print("{q, H} =", bracket(q, spring), "  {p, H} =", bracket(p, spring))
    print("ball, q(t):", series(q, ball, 5))
    print("spring, q(t) to t^5:", series(q, spring, 6))
    """),
    md(r"""
    For the pendulum the terms grow more complicated. The cell sums the
    series for $\theta(t)$ to a chosen number of terms and compares it with
    `hamiltonian.flow`.
    """),
    code(r"""
    th = sp.symbols("theta")
    pendulum = p**2 / (2 * INERTIA) + MGL * (1 - sp.cos(th))
    TERMS = [th]
    for _ in range(13):
        TERMS.append(sp.expand(bracket(TERMS[-1], pendulum, ((th, p),))))
    TERM_FNS = [sp.lambdify((th, p), term, "numpy") for term in TERMS]


    def partial_sums(n_terms=4, theta0=1.0, p0=0.0):
        times = np.linspace(0, 1.5, 151)
        exact = hamiltonian.flow(pend_dh_dq, pend_dh_dp, [theta0], [p0],
                                 times)[0][:, 0]
        total = sum(times**n / math.factorial(n) * TERM_FNS[n](theta0, p0)
                    for n in range(n_terms))
        fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=120)
        ax.plot(times, exact, color="black", label="flow")
        ax.plot(times, total, color=viz.ACCENT, ls="--",
                label=f"{n_terms} terms")
        angle_limit = max(2.0, 1.1 * abs(theta0))
        ax.set_ylim(-angle_limit, angle_limit)
        ax.set_xlabel("$t$ / s")
        ax.set_ylabel(r"$\theta$ / rad")
        ax.legend()
        plt.show()
        bad = np.abs(total - exact) >= 1e-3
        t_ok = times[bad.argmax() - 1] if bad.any() else times[-1]
        print(f"within 0.001 rad of the flow up to t = {t_ok:.2f} s")


    widgets.interact(
        partial_sums,
        n_terms=widgets.IntSlider(4, min=1, max=14, **SLOW),
        theta0=widgets.FloatSlider(1.0, min=0.1, max=2.5, step=0.1, **SLOW),
        p0=widgets.fixed(0.0),
    );
    """),
    md(r"""
    Adding non-zero terms improves the approximation near the starting
    time; the printed interval shows where it meets the chosen tolerance.
    The vertical window follows the true swing, so a poor truncated series
    may leave the plot. This does not guarantee accuracy over a whole swing. Chapter 9
    therefore splits $\hat{L}$ into pieces that can each be followed
    exactly.
    """),
]

LIOUVILLE = [
    md(r"""
    ## 7.4 Liouville's theorem

    A blob of 6000 starting states on its edge, carried by the flow. The
    area, computed with `polygon_area`, stays fixed without drag and falls
    as $e^{-\gamma t}$ with it.
    """),
    code(r"""
    s = np.linspace(0, 2 * np.pi, 6000, endpoint=False)
    BLOB_Q, BLOB_P = 1.6 + 0.4 * np.cos(s), 0.8 * np.sin(s)
    FRAMES = np.linspace(0.0, 4.0, 41)


    def blob_animation(gamma):
        qs, ps = hamiltonian.flow(pend_dh_dq, pend_dh_dp, BLOB_Q, BLOB_P,
                                  FRAMES, gamma=gamma)
        area0 = hamiltonian.polygon_area(BLOB_Q, BLOB_P)
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=72)
        th = np.linspace(-np.pi, np.pi, 300)
        sep = 2 * math.sqrt(INERTIA * MGL) * np.cos(th / 2)
        ax.plot(th, sep, color="0.8")
        ax.plot(th, -sep, color="0.8")
        patch = ax.fill(qs[0], ps[0], color=viz.ACCENT, alpha=0.7)[0]
        label = ax.set_title("")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel(r"$\theta$ / rad")
        ax.set_ylabel(r"$p$ / kg m$^2$ s$^{-1}$")

        def draw(i):
            patch.set_xy(np.column_stack([qs[i], ps[i]]))
            ratio = hamiltonian.polygon_area(qs[i], ps[i]) / area0
            label.set_text(f"t = {FRAMES[i]:.1f} s, area ratio {ratio:.4f}")
            return patch, label

        animation = FuncAnimation(fig, draw, frames=len(FRAMES), interval=80)
        plt.close(fig)
        return animation


    HTML(blob_animation(0.0).to_jshtml())
    """),
    code(r"""
    HTML(blob_animation(0.5).to_jshtml())  # with drag, γ = 0.5 per second
    """),
    md(r"""
    Without drag the blob is sheared into a long
    streak, yet the area ratio stays at 1.0000. With drag the ratio after
    $t$ seconds is $e^{-0.5t}$: 0.6065 at 1 s, 0.3679 at 2 s.
    """),
    code(r"""
    # The Jacobian determinant of the flow map, by central differences.
    for gamma in (0.0, 0.5):
        jac = hamiltonian.flow_jacobian(pend_dh_dq, pend_dh_dp, 1.0, 0.5,
                                        2.5, gamma=gamma)
        print(f"gamma = {gamma}: J =\n{jac.round(4)}\n"
              f"det J = {np.linalg.det(jac):.8f}, "
              f"exp(-2.5 gamma) = {math.exp(-2.5 * gamma):.8f}")
    """),
]

NOETHER = [
    md(r"""
    ## 7.5 Symmetries and conserved quantities

    Forces from an energy that depends only on a bond angle, computed by
    central differences, for three atoms placed at random. They are not
    along the lines that join the atoms, yet they add to zero and give no
    torque.
    """),
    code(r"""
    def angle_energy(r, k_theta=2.0, theta0=math.radians(104.5)):
        a, b = r[0] - r[1], r[2] - r[1]
        cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        return 0.5 * k_theta * (math.acos(cos) - theta0) ** 2


    def forces_of(energy_fn, r, h=1e-6):
        out = np.zeros_like(r)
        for i in range(len(r)):
            for a in range(r.shape[1]):
                up, down = r.copy(), r.copy()
                up[i, a] += h
                down[i, a] -= h
                out[i, a] = -(energy_fn(up) - energy_fn(down)) / (2 * h)
        return out


    for trial in range(3):
        r = RNG.normal(size=(3, 3))
        f = forces_of(angle_energy, r)
        print(f"trial {trial}: |sum of forces| = "
              f"{np.linalg.norm(f.sum(axis=0)):.1e}, |torque| = "
              f"{np.linalg.norm(np.cross(r, f).sum(axis=0)):.1e} eV")
    """),
    md(r"""
    Two atoms in a periodic box 10 Å wide, joined by a spring to the
    nearest copy of each other. Move atom 2 with the sliders: the forces
    always add to zero, and the torque about the origin is zero when the
    nearest copy is the atom itself, inside the box, and in general not
    otherwise.
    """),
    code(r"""
    BOX = 10.0


    def box_energy(r, k_bond=1.0, r0=2.0):
        d = r[1] - r[0]
        d -= BOX * np.round(d / BOX)
        return 0.5 * k_bond * (np.linalg.norm(d) - r0) ** 2


    def box_pair(x2=9.0, y2=2.0):
        pair = np.array([[1.0, 1.0], [x2, y2]])
        separation = pair[1] - pair[0]
        if np.allclose(separation, 0):
            print("The atoms coincide, so the spring's force direction is "
                  "undefined. Separate them to compare the forces.")
            return
        if np.any(np.isclose(np.abs(separation), BOX / 2)):
            print("Two periodic copies are equally near on a half-box "
                  "boundary. Move off the boundary to obtain a unique force.")
            return
        f = forces_of(box_energy, pair)
        torque = np.sum(pair[:, 0] * f[:, 1] - pair[:, 1] * f[:, 0])
        print(f"forces {f.round(4).tolist()} eV/Å; sum {f.sum(axis=0).round(9)}; "
              f"torque about the origin {torque:.4f} eV")


    widgets.interact(
        box_pair,
        x2=widgets.FloatSlider(9.0, min=0.5, max=9.5, step=0.5, **SLOW),
        y2=widgets.FloatSlider(2.0, min=0.5, max=9.5, step=0.5, **SLOW),
    );
    """),
    md(r"""
    The forces add to zero for every placement. At
    (9, 2) Å the spring reaches across the boundary and the torque is
    −1.056 eV. Move atom 2 to (3, 2) Å, where its nearest copy is itself,
    and the torque vanishes.
    """),
]

REVERSAL = [
    md(r"""
    ## 7.6 Running the motion backwards

    The puck on springs of 2 N/m (east-west) and 5 N/m (north-south),
    followed forwards, reversed and followed again for the same time.
    """),
    code(r"""
    def puck_run(r0, p0, duration, gamma, m=0.2, kx=2.0, ky=5.0):
        def rates(_t, y):
            return [y[2] / m, y[3] / m, -kx * y[0] - gamma * y[2],
                    -ky * y[1] - gamma * y[3]]

        t = np.linspace(0, duration, 1001)
        return solve_ivp(rates, (0, duration), [*r0, *p0], t_eval=t,
                         method="DOP853", rtol=1e-11, atol=1e-13).y


    def reverse(duration=4.0, gamma=0.0):
        fwd = puck_run((0.3, 0.0), (0.02, 0.1), duration, gamma)
        back = puck_run(fwd[:2, -1], -fwd[2:, -1], duration, gamma)
        fig, ax = plt.subplots(figsize=(4.5, 2.4), dpi=120)
        ax.plot(fwd[0], fwd[1], color=viz.ACCENT, lw=1.6, label="forward")
        ax.plot(back[0], back[1], color=viz.OCHRE, ls="--", label="reversed")
        ax.plot(0.3, 0.0, "o", color="black")
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ / m")
        ax.set_ylabel("$y$ / m")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncols=2)
        plt.show()
        miss = math.hypot(back[0, -1] - 0.3, back[1, -1])
        print(f"returns {miss:.1e} m from the start")


    widgets.interact(
        reverse,
        duration=widgets.FloatSlider(4.0, min=0.5, max=20.0, step=0.5, **SLOW),
        gamma=widgets.FloatSlider(0.0, min=0.0, max=1.0, step=0.05, **SLOW),
    );
    """),
    md(r"""
    Without drag the dashed path lies on the solid
    one and the puck returns to its start, however long the run. Any drag
    at all makes it miss.
    """),
]

SYMPLECTIC = [
    md(r"""
    ## 7.7 Steps that keep area

    The block of 0.5 kg on 50 N/m stepped by the forward Euler and the
    symplectic Euler methods. A small square of starting states is carried
    along too, so the effect of each step map on areas can be seen.
    """),
    code(r"""
    SPRING_M, SPRING_K = 0.5, 50.0


    def spring_dh_dq(q, p):
        return SPRING_K * q


    def spring_dh_dp(q, p):
        return p / SPRING_M


    METHODS = {"forward Euler": hamiltonian.euler_step,
               "symplectic Euler": hamiltonian.symplectic_euler_step}


    def steps(dt=0.02, n_steps=50):
        side = 0.01
        sq_q = 0.04 + side * np.array([0, 1, 1, 0])
        sq_p = side * 10 * np.array([0, 0, 1, 1])
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 2.8), dpi=120,
                                           gridspec_kw={"wspace": 0.4})
        for name, step in METHODS.items():
            q, p = np.array([0.04]), np.array([0.0])
            qq, pp = sq_q.copy(), sq_p.copy()
            path, energies = [(q[0], p[0])], [1.0]
            for _ in range(n_steps):
                q, p = step(spring_dh_dq, spring_dh_dp, q, p, dt)
                qq, pp = step(spring_dh_dq, spring_dh_dp, qq, pp, dt)
                path.append((q[0], p[0]))
                energies.append((p[0]**2 / (2 * SPRING_M)
                                 + 0.5 * SPRING_K * q[0]**2) / 0.04)
            path = np.array(path)
            colour = viz.METHOD[name]
            left.plot(100 * path[:, 0], path[:, 1], ".-", color=colour,
                      ms=2, lw=0.5, label=name)
            left.fill(100 * qq, pp, color=colour, alpha=0.5)
            right.semilogy(dt * np.arange(n_steps + 1), energies, color=colour)
            final_area = hamiltonian.polygon_area(qq, pp)
            area = final_area / hamiltonian.polygon_area(sq_q, sq_p)
            scale = np.max(np.abs(qq)) * np.max(np.abs(pp))
            if final_area <= 100 * np.finfo(float).eps * scale:
                area_text = "unresolved: the square is too thin for round-off"
            else:
                area_text = f"x {area:.6g}"
            print(f"{name}: energy x {energies[-1]:.6g}, square's area "
                  f"{area_text} after {n_steps} steps")
        left.fill(100 * sq_q, sq_p, color="black", alpha=0.3)
        left.set_xlabel("$q$ / cm")
        left.set_ylabel("$p$ / kg m s$^{-1}$")
        left.legend(loc="upper left", fontsize=7)
        right.set_xlabel("$t$ / s")
        right.set_ylabel("energy / starting energy")
        plt.show()
        print(f"1 + (omega dt)^2 to the power {n_steps}: "
              f"{(1 + (10 * dt) ** 2) ** n_steps:.6g}")
        if 10 * dt >= 2:
            print("This step reaches or exceeds the spring's stability "
                  "limit. Symplectic Euler still preserves area in exact "
                  "arithmetic, but its trajectories need not stay bounded.")


    widgets.interact(
        steps,
        dt=widgets.FloatSlider(0.02, min=0.002, max=0.21, step=0.002,
                               readout_format=".3f", **SLOW),
        n_steps=widgets.IntSlider(50, min=1, max=300, **SLOW),
    );
    """),
    md(r"""
    The forward Euler square grows by exactly the
    factor printed below the plots, the same as the energy; the
    symplectic Euler square is sheared but keeps its area, and its energy
    stays in a band. A larger step widens the band but does not make it
    grow, until $\omega\,\delta t$ approaches 2, where the method fails.
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
    derivations 7.1 to 7.6, 7.9 to 7.11, 7.14, 7.16, 7.17, 7.19 and 7.20
    have their solutions checked by SymPy.
    """),
    hidden(r"""
    # The answer key. Each target is computed rather than typed in.
    _p_sep = 2 * INERTIA * OMEGA0
    _pair = np.array([[1.0, 1.0], [9.0, 2.0]])
    _f1 = 0.5 * (math.sqrt(5) - 2) * 2 * np.array([-2.0, 1.0]) / math.sqrt(5)
    TARGETS = {
        "7.7": np.array([_p_sep, _p_sep / (MASS * LENGTH)]),
        "7.8": math.log(10) / OMEGA0,
        "7.12": 0.1 * 0.1 * np.linalg.det(np.array([[1.0, 2.0 / 1.0],
                                                    [0.0, 1.0]])),
        "7.13": math.log(2) / 0.5,
        "7.16": float((_pair[0, 0] - _pair[1, 0]) * _f1[1]
                      - (_pair[0, 1] - _pair[1, 1]) * _f1[0]),
        "7.18": math.log(2) / math.log(1 + (10 * 0.005) ** 2),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    solution(r"""
    # SOLUTION 7.1. The transform of e^v, and its slope.
    v_, pp = sp.symbols("v p", positive=True)
    v_at_p = sp.solve(sp.Eq(sp.exp(v_), pp), v_)[0]  # ln p
    f_star = sp.simplify(pp * v_at_p - sp.exp(v_at_p))
    assert sp.simplify(f_star - (pp * sp.log(pp) - pp)) == 0
    assert sp.simplify(sp.diff(f_star, pp) - v_at_p) == 0
    print("f*(p) =", f_star, "; slope", sp.diff(f_star, pp))
    """),
    solution(r"""
    # SOLUTION 7.2. A product of two rates still gives Σ p q̇ = 2K.
    a, b, c, r1, r2 = sp.symbols("a b c qdot_1 qdot_2")
    K = (a * r1**2 + 2 * b * r1 * r2 + c * r2**2) / 2
    assert sp.expand(sp.diff(K, r1) * r1 + sp.diff(K, r2) * r2 - 2 * K) == 0
    print("p1 q1' + p2 q2' − 2K = 0: checked")
    """),
    solution(r"""
    # SOLUTION 7.3. The hanging spring: H, the equations and the rest state.
    H = p**2 / (2 * m) + k * q**2 / 2 - m * g * q
    rates = [sp.diff(H, p), -sp.diff(H, q)]
    rest = sp.solve(rates, [q, p], dict=True)
    assert rest == [{q: m * g / k, p: 0}]
    print("q' =", rates[0], "; p' =", rates[1], "; rest at", rest[0])
    """),
    solution(r"""
    # SOLUTIONS 7.4 and 7.5. The ball in a plane, and the puck in polar
    # coordinates, from the Lagrangian by the Legendre transform.
    x, y, r, th, px, py, pr, pt = sp.symbols("x y r theta p_x p_y p_r p_theta")
    xd, yd, rd, thd = sp.symbols("xdot ydot rdot thetadot")


    def hamiltonian_of(lag, rates_, momenta):
        sol = sp.solve([sp.Eq(mom, sp.diff(lag, rd_))
                        for mom, rd_ in zip(momenta, rates_)], rates_,
                       dict=True)[0]
        h = sum(mom * rd_ for mom, rd_ in zip(momenta, rates_)) - lag
        return sp.simplify(h.subs(sol))


    H_ball = hamiltonian_of(m * (xd**2 + yd**2) / 2 - m * g * y, [xd, yd],
                            [px, py])
    assert sp.simplify(H_ball - ((px**2 + py**2) / (2 * m) + m * g * y)) == 0
    assert sp.diff(H_ball, x) == 0  # so p_x is conserved
    U = sp.Function("U")
    H_puck = hamiltonian_of(m * (rd**2 + r**2 * thd**2) / 2 - U(r),
                            [rd, thd], [pr, pt])
    assert sp.simplify(H_puck - (pr**2 / (2 * m) + pt**2 / (2 * m * r**2)
                                 + U(r))) == 0
    assert sp.diff(H_puck, th) == 0
    effective = U(r) + pt**2 / (2 * m * r**2)
    assert sp.simplify(-sp.diff(H_puck, r) + sp.diff(effective, r)) == 0
    print("ball:", H_ball, "\npuck:", H_puck)
    """),
    solution(r"""
    # SOLUTION 7.6. The bead's Hamilton's equations give its equation
    # (6.4), here for any wire of degree four, h = c0 + c1 q + ... + c4 q⁴.
    tt = sp.symbols("t")
    cs = sp.symbols("c0:5")
    h_q = sum(ci * q**i for i, ci in enumerate(cs))
    A = 1 + sp.diff(h_q, q) ** 2
    H_bead = p**2 / (2 * m * A) + m * g * h_q
    X = sp.Function("x")(tt)
    on_path = {q: X, p: m * A.subs(q, X) * X.diff(tt)}
    assert sp.cancel(sp.diff(H_bead, p).subs(on_path) - X.diff(tt)) == 0
    pdot_hamilton = (-sp.diff(H_bead, q)).subs(on_path)
    pdot_direct = sp.diff(on_path[p], tt)
    xdd = sp.solve(sp.Eq(pdot_direct, pdot_hamilton), X.diff(tt, 2))[0]
    hp, hpp = sp.diff(h_q, q).subs(q, X), sp.diff(h_q, q, 2).subs(q, X)
    expected = -hp * (g + hpp * X.diff(tt) ** 2) / (1 + hp**2)
    assert sp.cancel(xdd - expected) == 0
    print("x'' = −h'(g + h'' x'²)/(1 + h'²): checked")
    """),
    code(r"""
    # EXERCISE 7.7: momentum (kg m²/s) and speed (m/s) at the bottom of the
    # separatrix.
    answers = None

    check(answers, TARGETS["7.7"], name="momentum, speed");
    """),
    solution(r"""
    # SOLUTION 7.7. p = 2 mL² ω0, and the speed is L θ' = p/(mL).
    p_bottom = 2 * INERTIA * OMEGA0
    answers = np.array([p_bottom, p_bottom / (MASS * LENGTH)])

    check(answers, TARGETS["7.7"], name="momentum, speed");
    """),
    code(r"""
    # EXERCISE 7.8: time (s) from 10° short of the top to 1° short.
    creep = None

    check(creep, TARGETS["7.8"], name="time");
    """),
    solution(r"""
    # SOLUTION 7.8. The gap falls as exp(−ω0 t), so by 10 in ln 10/ω0.
    creep = math.log(10) / OMEGA0

    check(creep, TARGETS["7.8"], name="time");
    """),
    solution(r"""
    # SOLUTIONS 7.9 and 7.10. The product rule for brackets, and L_z.
    A_, B_, C_ = (sp.Function(name)(q, p) for name in "ABC")
    lhs = bracket(A_, B_ * C_)
    assert sp.simplify(lhs - (bracket(A_, B_) * C_ + B_ * bracket(A_, C_))) == 0
    Lz = x * py - y * px
    H_plane = (px**2 + py**2) / (2 * m) + U(sp.sqrt(x**2 + y**2))
    assert sp.simplify(bracket(Lz, H_plane, ((x, px), (y, py)))) == 0
    print("{A, BC} = {A, B} C + B {A, C}; {L_z, H} = 0: checked")
    """),
    solution(r"""
    # SOLUTION 7.11. The series for p(t) of the spring.
    w = sp.symbols("omega", positive=True)
    spring_w = p**2 / (2 * m) + m * w**2 * q**2 / 2
    approx = series(p, spring_w, 10)
    exact = p * sp.cos(w * t) - m * w * q * sp.sin(w * t)
    assert sp.simplify(sp.series(exact, t, 0, 10).removeO() - approx) == 0
    print("p(t) = p0 cos wt − m w q0 sin wt, to t^9")
    """),
    code(r"""
    # EXERCISE 7.12: area (J s) of the image of the square after 2 s.
    area = None

    check(area, TARGETS["7.12"], name="area");
    """),
    solution(r"""
    # SOLUTION 7.12. Free flight of 1 kg for 2 s: q = q0 + 2 p0, p = p0.
    q0s = np.array([0.0, 0.1, 0.1, 0.0])
    p0s = np.array([0.0, 0.0, 0.1, 0.1])
    print("corners go to",
          np.column_stack([q0s + 2 * p0s, p0s]).round(3).tolist())
    area = hamiltonian.polygon_area(q0s + 2 * p0s, p0s)

    check(area, TARGETS["7.12"], name="area");
    """),
    code(r"""
    # EXERCISE 7.13: time (s) for a blob to lose half its area at γ = 0.5/s.
    halving = None

    check(halving, TARGETS["7.13"], name="halving time");
    """),
    solution(r"""
    # SOLUTION 7.13. exp(−γt) = 1/2, and a check with the flow itself.
    halving = math.log(2) / 0.5
    qs, ps = hamiltonian.flow(pend_dh_dq, pend_dh_dp, BLOB_Q, BLOB_P,
                              [0.0, halving], gamma=0.5)
    print("area ratio after that time:",
          round(hamiltonian.polygon_area(qs[1], ps[1])
                / hamiltonian.polygon_area(qs[0], ps[0]), 6))

    check(halving, TARGETS["7.13"], name="halving time");
    """),
    solution(r"""
    # SOLUTION 7.14. Two carts: shifting both leaves L unchanged, and
    # m1 x1' + m2 x2' is constant by the Euler-Lagrange equations.
    m1, m2, ell = sp.symbols("m_1 m_2 ell", positive=True)
    x1, x2 = sp.Function("x_1")(tt), sp.Function("x_2")(tt)
    alpha = sp.symbols("alpha")
    lag = (m1 * x1.diff(tt)**2 / 2 + m2 * x2.diff(tt)**2 / 2
           - k * (x2 - x1 - ell)**2 / 2)
    assert sp.simplify(lag.subs({x1: x1 + alpha, x2: x2 + alpha}).doit()
                       - lag) == 0
    eqs = sp.euler_equations(lag, [x1, x2], tt)
    accel = sp.solve([e.lhs for e in eqs], [x1.diff(tt, 2), x2.diff(tt, 2)])
    total = sp.diff(m1 * x1.diff(tt) + m2 * x2.diff(tt), tt).subs(accel)
    assert sp.simplify(total) == 0
    print("d/dt (m1 x1' + m2 x2') = 0: checked")
    """),
    solution(r"""
    # SOLUTION 7.15. Three atoms joined by springs fall under gravity:
    # measure which components of P and L stay constant.
    masses = np.array([1.0, 2.0, 3.0])
    rest_lengths = {(0, 1): 1.0, (1, 2): 1.2, (0, 2): 1.5}


    def accelerations(r):
        f = np.zeros_like(r)
        for (i, j), d0 in rest_lengths.items():
            d = r[j] - r[i]
            pull = 5.0 * (np.linalg.norm(d) - d0) * d / np.linalg.norm(d)
            f[i] += pull
            f[j] -= pull
        f[:, 2] -= masses * 9.8
        return f / masses[:, None]


    def rhs(_t, y):
        r, v = y[:9].reshape(3, 3), y[9:].reshape(3, 3)
        return np.concatenate([v.ravel(), accelerations(r).ravel()])


    y0 = np.concatenate([RNG.normal(size=9), RNG.normal(size=9)])
    sol = solve_ivp(rhs, (0, 2), y0, t_eval=[0, 2], rtol=1e-10, atol=1e-12)
    for label, idx in (("start", 0), ("end", -1)):
        r = sol.y[:9, idx].reshape(3, 3)
        mom = masses[:, None] * sol.y[9:, idx].reshape(3, 3)
        print(label, "P =", mom.sum(axis=0).round(4),
              "L =", np.cross(r, mom).sum(axis=0).round(4))
    r0, r1 = sol.y[:9, 0].reshape(3, 3), sol.y[:9, -1].reshape(3, 3)
    m0 = masses[:, None] * sol.y[9:, 0].reshape(3, 3)
    m1_ = masses[:, None] * sol.y[9:, -1].reshape(3, 3)
    P0, P1 = m0.sum(axis=0), m1_.sum(axis=0)
    L0, L1 = np.cross(r0, m0).sum(axis=0), np.cross(r1, m1_).sum(axis=0)
    assert np.allclose(P0[:2], P1[:2]) and np.isclose(L0[2], L1[2])
    assert not np.isclose(P0[2], P1[2])
    assert not np.isclose(L0[0], L1[0]) and not np.isclose(L0[1], L1[1])
    print("P_x, P_y and L_z keep their values; P_z, L_x and L_y do not.")
    """),
    code(r"""
    # EXERCISE 7.16: torque (eV) of the box pair about the centre (5, 5) Å.
    torque_centre = None

    check(torque_centre, TARGETS["7.16"], name="torque");
    """),
    solution(r"""
    # SOLUTION 7.16. The forces add to zero, so the torque is the same
    # about every point (SymPy, for three forces in a plane); computed
    # directly about (5, 5) Å.
    ax_, ay_ = sp.symbols("a_x a_y")
    xs_, ys_ = sp.symbols("x1:4"), sp.symbols("y1:4")
    fx_, fy_ = list(sp.symbols("f_x1:3")), list(sp.symbols("f_y1:3"))
    fx_.append(-fx_[0] - fx_[1])  # the third force: the three add to zero
    fy_.append(-fy_[0] - fy_[1])
    about_a = sum((xs_[i] - ax_) * fy_[i] - (ys_[i] - ay_) * fx_[i]
                  for i in range(3))
    about_o = sum(xs_[i] * fy_[i] - ys_[i] * fx_[i] for i in range(3))
    assert sp.expand(about_a - about_o) == 0
    pair = np.array([[1.0, 1.0], [9.0, 2.0]])
    f = forces_of(box_energy, pair)
    rel = pair - [5.0, 5.0]
    torque_centre = float(np.sum(rel[:, 0] * f[:, 1] - rel[:, 1] * f[:, 0]))

    check(torque_centre, TARGETS["7.16"], name="torque");
    """),
    solution(r"""
    # SOLUTION 7.17. Which Hamiltonians are unchanged by p → −p?
    c_, b_ = sp.symbols("c b", nonzero=True)
    candidates = {"H1": p**2 / (2 * m) + U(q), "H2": (p - c_)**2 / (2 * m),
                  "H3": p**2 / (2 * m) + k * q**2 / 2 + b_ * q * p}
    even = [name for name, h in candidates.items()
            if sp.simplify(h.subs(p, -p) - h) == 0]
    assert even == ["H1"]
    print("reversible by the argument of Section 7.6:", even)
    """),
    code(r"""
    # EXERCISE 7.18: number of forward Euler steps for the energy to double
    # with δt = 0.005 s, ω δt = 0.05 (a real number; round up for whole
    # steps).
    n_double = None

    check(n_double, TARGETS["7.18"], name="steps");
    """),
    solution(r"""
    # SOLUTION 7.18. (1 + 0.0025)^n = 2, and a run of the method itself.
    n_double = math.log(2) / math.log(1.0025)
    q1, p1 = np.array([0.04]), np.array([0.0])
    for _ in range(278):
        q1, p1 = hamiltonian.euler_step(spring_dh_dq, spring_dh_dp, q1, p1,
                                        0.005)
    print("energy ratio after 278 steps:",
          round(float((p1**2 / 1.0 + 25.0 * q1**2)[0] / 0.04), 4))

    check(n_double, TARGETS["7.18"], name="steps");
    """),
    solution(r"""
    # SOLUTIONS 7.19 and 7.20. The other order keeps area, and the energy
    # that symplectic Euler keeps for the spring.
    dt_ = sp.symbols("delta_t", positive=True)
    q_new = q + dt_ * p / m
    p_new = p - dt_ * k * q_new
    jac = sp.Matrix([[sp.diff(q_new, q), sp.diff(q_new, p)],
                     [sp.diff(p_new, q), sp.diff(p_new, p)]])
    assert sp.simplify(jac.det()) == 1
    p_kick = p - dt_ * k * q
    q_drift = q + dt_ * p_kick / m
    shadow = p**2 / (2 * m) + k * q**2 / 2 - k * dt_ * q * p / (2 * m)
    after = shadow.subs({q: q_drift, p: p_kick}, simultaneous=True)
    assert sp.simplify(sp.expand(after - shadow)) == 0
    print("det J = 1 for the other order; H~ is unchanged by a step")
    """),
]

CELLS = (SETUP + LEGENDRE + EQUATIONS + BRACKETS + LIOUVILLE + NOETHER
         + REVERSAL + SYMPLECTIC + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "07_hamilton.ipynb"))
