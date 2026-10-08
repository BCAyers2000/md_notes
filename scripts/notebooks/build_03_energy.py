"""Write notebooks/03_energy.ipynb, the companion to Chapter 3.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_03_energy.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/03_energy.ipynb
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md("""
    # Notebook 03: work and energy

    These calculations follow the energy from a force, through the work
    it does, to the motion it permits. The section numbers match Chapter 3.
    Run the cells in order, then use the sliders to test a prediction: where
    should a body turn back, and which changes leave its energy unchanged?
    Keep the prediction and the result beside the plot in your working copy.
    """),
    code("""
    %matplotlib inline
    import math

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from scipy.constants import g as G  # 9.80665 m/s²
    from scipy.integrate import cumulative_trapezoid

    from mdlab import dynamics, energy, units, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    """),
]

CONSTANT = [
    md(r"""
    ## 3.1 Work done by a constant force

    A rope pulls a sledge with 40 N at an angle $\theta$ above the snow,
    over 10 m. Only the component $F\cos\theta$ along the snow does work,
    $W = \mathbf F\cdot\Delta\mathbf r$.
    """),
    code("""
    def sledge(angle=30.0, pull=40.0, distance=10.0):
        theta = np.radians(angle)
        force = pull * np.array([np.cos(theta), np.sin(theta)])
        step = np.array([distance, 0.0])
        work = force @ step  # the scalar product F · Δr

        angles = np.linspace(0.0, 180.0, 181)
        fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.4))
        ax.plot(angles, pull * distance * np.cos(np.radians(angles)),
                color="black")
        ax.plot(angle, work, "o", color=viz.ACCENT)
        ax.axhline(0.0, **viz.THRESHOLD_STYLE)
        ax.set_xlabel(r"angle $\\theta$ / degrees")
        ax.set_ylabel("work $W$ / J")
        plt.show()
        print(f"F = ({force[0]:.2f}, {force[1]:.2f}) N, W = {work:.1f} J")


    widgets.interact(
        sledge,
        angle=widgets.FloatSlider(30.0, min=0.0, max=180.0, step=5.0),
        pull=widgets.FloatSlider(40.0, min=0.0, max=80.0, step=5.0),
        distance=widgets.FloatSlider(10.0, min=0.0, max=20.0, step=1.0),
    );
    """),
    md("""
    The work is largest when the rope is level,
    falls to zero when it pulls straight up ($\\theta$ = 90°), and turns
    negative when the pull has a component against the motion. At 30°
    it is 346.4 J, as in Section 3.1.
    """),
]

KINETIC = [
    md(r"""
    ## 3.2 Kinetic energy

    Under a constant force the change in kinetic energy equals
    $\mathbf F\cdot\Delta\mathbf r$, in any direction. Here a random
    constant force acts on a body of 2 kg for 3 s, starting from a
    random velocity, and the two sides are compared.
    """),
    code("""
    rng = np.random.default_rng(2026)
    mass = 2.0  # kg
    force = rng.normal(size=3)  # N
    v0 = rng.normal(size=3)  # m/s
    t = 3.0  # s
    r, v = dynamics.solve_newton(
        lambda r: force, mass, np.zeros(3), v0, [0.0, t],
        force_to_accel=1.0,
    )
    change = 0.5 * mass * (v[-1] @ v[-1] - v0 @ v0)
    print(f"K(3 s) − K(0) = {change:.10f} J")
    print(f"F · Δr        = {force @ r[-1]:.10f} J")
    """),
    md("""
    The projectile of Section 3.2: launched at 12 m/s and 60°, it is
    moving at 12 cos 60° = 6 m/s at the top, so energy gives the height
    of the top without the time.
    """),
    code("""
    v0, alpha = 12.0, np.radians(60.0)
    top = (v0**2 - (v0 * np.cos(alpha)) ** 2) / (2 * G)
    print(f"top of the path from energy: {top:.4f} m")
    vertical = v0 * np.sin(alpha)  # the vertical part of the launch
    print(f"(v0 sin α)² / 2g:            {vertical**2 / (2 * G):.4f} m")
    """),
]

VARYING = [
    md(r"""
    ## 3.3 Work done by a varying force

    The work of a varying force is the area under $F(x)$. The rectangles
    of Figure 3.2(a) approach the area as they narrow.
    """),
    code("""
    def example_force(x):
        return 3.0 + 1.5 * np.sin(1.4 * x) + 0.4 * x  # N, illustrative


    def rectangles(n=8):
        xa, xb = 0.5, 4.5
        edges = np.linspace(xa, xb, n + 1)
        middles = 0.5 * (edges[1:] + edges[:-1])
        total = np.sum(example_force(middles) * np.diff(edges))
        fine = np.linspace(xa, xb, 20001)
        area = np.trapezoid(example_force(fine), fine)

        fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.4))
        ax.bar(middles, example_force(middles), width=np.diff(edges),
               color=viz.ACCENT, alpha=0.2, edgecolor=viz.ACCENT)
        x = np.linspace(0.0, 5.0, 300)
        ax.plot(x, example_force(x), color="black")
        ax.set_xlabel("$x$ / m")
        ax.set_ylabel("$F$ / N")
        plt.show()
        print(f"{n} rectangles: {total:.5f} J;  area: {area:.5f} J")


    widgets.interact(rectangles, n=widgets.IntSlider(8, min=1, max=64,
                                                     **SLOW));
    """),
    md("""
    One rectangle is a poor guess; eight are close;
    sixty-four agree with the area to about one part in a hundred
    thousand.

    SymPy integrates the hand's force $kx$ for the spring, and checks the
    work-energy theorem for the block of Chapter 2: the kinetic energy at
    the centre equals the work done by the spring.
    """),
    code("""
    k, x, xs = sp.symbols("k x x_s", positive=True)
    print("work to stretch by x_s:", sp.integrate(k * x, (x, 0, xs)))

    k_value, block, stretch = 50.0, 0.5, 0.04
    work = 0.5 * k_value * stretch**2
    print(f"spring's work from 4 cm to 0: {work:.4f} J; "
          f"speed at the centre {math.sqrt(2 * work / block):.4f} m/s")
    """),
    md("""
    The theorem along the actual motion: integrating the power $Fv$ over
    time gives the change in kinetic energy, for the block on the spring
    of Section 2.8.
    """),
    code("""
    t = np.linspace(0.0, 0.3, 3001)
    r, v = dynamics.solve_newton(
        lambda r: -50.0 * r, 0.5, [0.04], [0.0], t, force_to_accel=1.0
    )
    power = -50.0 * r[:, 0] * v[:, 0]  # F v, in W
    supplied = cumulative_trapezoid(power, t, initial=0.0)
    kinetic = 0.5 * 0.5 * v[:, 0] ** 2
    print(f"largest |K(t) − K(0) − ∫F v dt| = "
          f"{np.max(np.abs(kinetic - kinetic[0] - supplied)):.1e} J")
    """),
]

POTENTIAL = [
    md(r"""
    ## 3.4 Potential energy and the conservation of energy

    A ball of 0.5 kg thrown up at 12 m/s. Move the time cursor and watch
    kinetic and potential energy trade, their sum fixed at 36 J.
    """),
    code("""
    MASS_BALL, V_THROW = 0.5, 12.0
    FLIGHT = 2 * V_THROW / G


    def throw(t=0.6):
        times = np.linspace(0.0, FLIGHT, 300)
        height = V_THROW * times - 0.5 * G * times**2
        kinetic = 0.5 * MASS_BALL * (V_THROW - G * times) ** 2
        potential = MASS_BALL * G * height

        k_now = 0.5 * MASS_BALL * (V_THROW - G * t) ** 2
        u_now = MASS_BALL * G * (V_THROW * t - 0.5 * G * t**2)
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(viz.FULL, 2.4),
            gridspec_kw=dict(width_ratios=(3, 1)),
        )
        left.plot(times, kinetic, color=viz.OCHRE, label="$K$")
        left.plot(times, potential, color=viz.OXBLOOD, label="$U$")
        left.plot(times, kinetic + potential, color=viz.ACCENT, label="$E$")
        left.axvline(t, **viz.THRESHOLD_STYLE)
        left.set_xlabel("$t$ / s")
        left.set_ylabel("energy / J")
        left.legend(fontsize=8, ncols=3, loc="lower center",
                    bbox_to_anchor=(0.5, 1.0))
        right.bar(["$K$", "$U$"], [k_now, u_now],
                  color=[viz.OCHRE, viz.OXBLOOD])
        right.set_ylim(0, 40)
        right.set_ylabel("energy / J")
        plt.show()
        print(f"K = {k_now:.2f} J, U = {u_now:.2f} J, "
              f"E = {k_now + u_now:.2f} J")


    widgets.interact(
        throw, t=widgets.FloatSlider(0.6, min=0.0, max=FLIGHT, step=0.01)
    );
    """),
    md("""
    The bars always add up to 36 J. At the top,
    t = 1.224 s, all of it is potential energy, and the ball is
    36 / (0.5 g) = 7.342 m up.

    The proof of Proposition 3.1 in SymPy: for any potential energy $U$
    and any motion obeying the second law, $\\mathrm dE/\\mathrm dt$
    vanishes.
    """),
    code("""
    t, m = sp.symbols("t m", positive=True)
    x = sp.Function("x")(t)
    U = sp.Function("U")
    E = m * x.diff(t) ** 2 / 2 + U(x)
    rate = E.diff(t)
    # the second law: m x'' = F = −dU/dx
    newton = {x.diff(t, 2): -sp.diff(U(x), x) / m}
    print("dE/dt =", sp.simplify(rate.subs(newton)))
    """),
]

TRACK = [
    md(r"""
    ## 3.5 Sliding on a hill-shaped track

    The track $h(x) = 0.15x^4 - 0.6x^2 + 0.15x + 1.0$ is
    `energy.hill_track`. Release the bead from rest at height $h_E$ and
    read its speed, $\sqrt{2g(h_E - h)}$, at the valley bottoms and on the
    hill.
    """),
    code("""
    EQUILIBRIA = np.sort(np.roots([0.6, 0.0, -1.2, 0.15]).real)
    PEAKS, _ = energy.hill_track(EQUILIBRIA)
    NAMES = ("deep valley", "hill", "shallow valley")


    def release(h_E=1.3):
        for name, h0 in zip(NAMES, PEAKS, strict=True):
            if h0 <= h_E:
                print(f"{name:15s} h = {h0:.3f} m:  "
                      f"v = {math.sqrt(2 * G * (h_E - h0)):.3f} m/s")
            else:
                print(f"{name:15s} h = {h0:.3f} m:  out of reach")


    widgets.interact(release, h_E=widgets.FloatSlider(1.3, min=0.0, max=2.0,
                                                    step=0.05));
    """),
]

DIAGRAMS = [
    md(r"""
    ## 3.6 Energy diagrams

    The energy diagram of the track. Move the total energy $E/mg$: the
    solid parts of the line are where the bead may go, the dots its
    turning points, and the lower panel its speed.
    """),
    code("""
    X_TRACK = np.linspace(-2.4, 2.4, 4801)
    H_TRACK, _ = energy.hill_track(X_TRACK)


    def diagram(level=1.3):
        allowed = H_TRACK <= level
        turns = energy.turning_points(X_TRACK, H_TRACK, level)
        speed = np.sqrt(2 * G * np.clip(level - H_TRACK, 0.0, None))

        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(viz.FULL, 3.6), sharex=True,
            gridspec_kw=dict(height_ratios=(1.6, 1.0)),
        )
        top.plot(X_TRACK, H_TRACK, color="black")
        top.plot(X_TRACK, np.where(allowed, level, np.nan),
                 color=viz.ACCENT, lw=2)
        top.plot(turns, np.full_like(turns, level), "o", color=viz.ACCENT)
        top.set_ylim(0, 2.2)
        top.set_ylabel("$U/mg = h$ / m")
        bottom.plot(X_TRACK, np.where(allowed, speed, np.nan),
                    color=viz.ACCENT)
        bottom.set_ylim(0, 6)
        bottom.set_xlabel("$x$ / m")
        bottom.set_ylabel("speed / m s$^{-1}$")
        plt.show()
        print("turning points:", turns.round(3), "m")


    widgets.interact(diagram, level=widgets.FloatSlider(1.3, min=0.0,
                                                        max=2.0, step=0.02,
                                                        **SLOW));
    """),
    md("""
    Below 0.183 m nothing is allowed. Between
    0.607 m and 1.009 m there are two separate allowed regions, one in
    each valley, and the bead keeps to the one it started in. Above
    1.009 m it passes over the hill.

    The equilibria are the roots of the cubic $\\mathrm dh/\\mathrm dx =
    0.6x^3 - 1.2x + 0.15$, and the sign of $h''$ decides their stability.
    """),
    code("""
    for x0, h0 in zip(EQUILIBRIA, PEAKS, strict=True):
        curvature = 1.8 * x0**2 - 1.2  # d²h/dx²
        kind = "stable" if curvature > 0 else "unstable"
        print(f"x = {x0:+.3f} m, h = {h0:.3f} m, h'' = {curvature:+.3f} "
              f"per m: {kind}")
    """),
]

FRICTION = [
    md(r"""
    ## 3.7 Friction and the loss of energy

    A block begins sliding down a straight ramp 3 m long with negligible
    speed. Change the angle and the sliding-friction coefficient and read
    the work over the ramp. This model describes motion once sliding has
    begun; it does not include the static friction that decides whether a
    block initially at rest starts to move.
    """),
    code("""
    def ramp(angle=30.0, mu=0.2, mass=2.0, length=3.0):
        theta = np.radians(angle)
        w_weight = mass * G * length * np.sin(theta)
        w_friction = -mu * mass * G * np.cos(theta) * length
        kinetic = w_weight + w_friction
        print(f"weight's work   {w_weight:7.2f} J")
        print(f"friction's work {w_friction:7.2f} J")
        balance = np.sin(theta) - mu * np.cos(theta)
        if np.isclose(balance, 0.0):
            print("the forces balance: a sliding block keeps its speed")
        elif balance < 0.0:
            print("a sliding block slows down; with negligible initial "
                  "speed it cannot reach the bottom. The work above "
                  "assumes traversal of the whole ramp.")
        else:
            print(f"speed at the bottom {math.sqrt(2 * kinetic / mass):.3f} "
                  f"m/s (smooth ramp: "
                  f"{math.sqrt(2 * G * length * np.sin(theta)):.3f} m/s)")


    widgets.interact(
        ramp,
        angle=widgets.FloatSlider(30.0, min=5.0, max=80.0, step=1.0),
        mu=widgets.FloatSlider(0.2, min=0.0, max=1.0, step=0.01),
        mass=widgets.fixed(2.0), length=widgets.fixed(3.0),
    );
    """),
    md("""
    The bead of Section 2.7 at its terminal velocity: the weight's power
    $mgv_\\infty$ and the drag's $bv_\\infty^2$ are the same.
    """),
    code("""
    bead, b = 0.01, 0.05
    v_inf = bead * G / b
    print(f"m g v∞ = {bead * G * v_inf:.4f} W,  b v∞² = {b * v_inf**2:.4f} W")
    """),
]

BODIES = [
    md(r"""
    ## 3.8 The energy of several bodies

    Two carts pushed apart by a spring of 200 N/m compressed by 0.1 m.
    Energy, $\frac12 m_1v_1^2 + \frac12 m_2v_2^2 = 1$ J, and momentum,
    $m_1v_1 + m_2v_2 = 0$, give the speeds; the computer solution of the
    equations of motion agrees.
    """),
    code("""
    def carts(m1=1.0, m2=3.0, k=200.0, squeeze=0.1):
        stored = 0.5 * k * squeeze**2
        v2 = math.sqrt(2 * stored * m1 / (m2 * (m1 + m2)))
        v1 = -m2 / m1 * v2
        natural = 0.3

        def push(r):
            c = max(natural - (r[1, 0] - r[0, 0]), 0.0)
            return np.array([[-k * c], [k * c]])

        t = np.linspace(0.0, 0.4, 801)
        r, v = dynamics.solve_newton(
            push, [m1, m2], [[0.0], [natural - squeeze]], np.zeros((2, 1)),
            t, force_to_accel=1.0,
        )
        compression = np.maximum(natural - (r[:, 1, 0] - r[:, 0, 0]), 0.0)
        kinetic = 0.5 * np.array([m1, m2]) * v[:, :, 0] ** 2
        spring = 0.5 * k * compression**2

        fig, ax = plt.subplots(figsize=(viz.FULL, 2.2))
        ax.plot(t, spring, color=viz.OXBLOOD, label="spring")
        ax.plot(t, kinetic[:, 0], color=viz.OCHRE, label="cart 1")
        ax.plot(t, kinetic[:, 1], color=viz.OCHRE, ls="--", label="cart 2")
        ax.plot(t, spring + kinetic.sum(axis=1), color=viz.ACCENT,
                label="total")
        ax.set_xlabel("$t$ / s")
        ax.set_ylabel("energy / J")
        ax.legend(fontsize=8, ncols=4, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0))
        plt.show()
        print(f"from energy and momentum: v1 = {v1:+.4f}, v2 = {v2:+.4f} m/s")
        print(f"computer solution:        v1 = {v[-1, 0, 0]:+.4f}, "
              f"v2 = {v[-1, 1, 0]:+.4f} m/s")


    widgets.interact(
        carts,
        m1=widgets.FloatSlider(1.0, min=0.5, max=5.0, step=0.5, **SLOW),
        m2=widgets.FloatSlider(3.0, min=0.5, max=5.0, step=0.5, **SLOW),
        k=widgets.fixed(200.0), squeeze=widgets.fixed(0.1),
    );
    """),
    md("""
    The total stays at 1 J. The lighter cart always
    takes the larger share of the energy, $K = p^2/2m$ with equal and
    opposite momenta; with equal masses they share it equally.
    """),
]

PATHS = [
    md(r"""
    ## 3.9 Work along a path in space

    The line integral of the toolbox, done by SymPy: the swirl
    $c(-y, x)$ once round a circle of radius $R$.
    """),
    code("""
    s, c, R = sp.symbols("s c R", positive=True)
    r = sp.Matrix([R * sp.cos(s), R * sp.sin(s)])
    F = sp.Matrix([-c * r[1], c * r[0]])
    integrand = sp.simplify(F.dot(r.diff(s)))
    print("F · dr/ds =", integrand)
    print("work once round =", sp.integrate(integrand, (s, 0, 2 * sp.pi)))
    """),
    md("""
    Now bend a path from $A = (0, 0)$ to $B = (4, 0)$ m and compare three
    works: the weight of a 1 kg body moving in a vertical plane, sliding
    friction on a floor ($\\mu_\\mathrm{k} = 0.3$, 1 kg), and the swirl
    with $c = 1$ N/m. `energy.work` computes the line integrals.
    """),
    code("""
    A, B = np.array([0.0, 0.0]), np.array([4.0, 0.0])
    s_path = np.linspace(0.0, 1.0, 2001)[:, None]


    def weight(r):
        return np.broadcast_to([0.0, -G], r.shape)  # 1 kg, y upwards


    def swirl_field(r):
        return energy.swirl(r, 1.0)


    def routes(bulge=1.0):
        path = A + s_path * (B - A) + bulge * np.sin(np.pi * s_path) * [0, 1]
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.0))
        ax.plot(*path.T, color=viz.ACCENT)
        ax.plot(*np.array([A, B]).T, "o", color="black")
        ax.set_aspect("equal")
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("$x$ / m")
        ax.set_ylabel("$y$ / m")
        plt.show()
        print(f"weight:   {energy.work(weight, path):+8.3f} J")
        print(f"friction: {-0.3 * G * length:+8.3f} J  (path {length:.3f} m)")
        print(f"swirl:    {energy.work(swirl_field, path):+8.3f} J")


    widgets.interact(routes, bulge=widgets.FloatSlider(1.0, min=-2.0,
                                                       max=2.0, step=0.1));
    """),
    md("""
    The weight does no work whatever the bulge,
    since $A$ and $B$ are at the same height. Friction always does more
    negative work on a longer path. The swirl's work changes sign with
    the bulge: it is $-2c$ times the area between the path and the
    straight line when the path bulges to the left, and $+2c$ times it
    when it bulges to the right.
    """),
]

GRADIENT = [
    md(r"""
    ## 3.10 Conservative forces and the gradient

    The hill $h = 300\,e^{-(x^2 + 2y^2)/2}$ m, with $x$ and $y$ in km.
    SymPy takes the partial derivatives; the widget moves the point $P$
    and shows the gradient there.
    """),
    code("""
    x, y = sp.symbols("x y", real=True)
    h = 300 * sp.exp(-(x**2 + 2 * y**2) / 2)
    dhdx, dhdy = sp.diff(h, x), sp.diff(h, y)
    print("∂h/∂x =", sp.simplify(dhdx))
    print("∂h/∂y =", sp.simplify(dhdy))
    at_P = {x: 0.6, y: 0.5}
    print(f"at P: h = {float(h.subs(at_P)):.1f} m, "
          f"∂h/∂x = {float(dhdx.subs(at_P)):.1f} m/km, "
          f"∂h/∂y = {float(dhdy.subs(at_P)):.1f} m/km")

    hill_height = sp.lambdify((x, y), h)
    slope = sp.lambdify((x, y), [dhdx, dhdy])
    """),
    code("""
    GX, GY = np.meshgrid(np.linspace(-2, 2, 201), np.linspace(-1.5, 1.5, 151))


    def explorer(px=0.6, py=0.5):
        gx, gy = slope(px, py)
        fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.6))
        contours = ax.contour(GX, GY, hill_height(GX, GY),
                              levels=np.arange(50, 300, 50),
                              colors="black", linewidths=0.5)
        ax.clabel(contours, fmt="%d m", fontsize=7)
        ax.quiver(px, py, gx, gy, color=viz.ACCENT, scale=450, **viz.ARROW)
        ax.plot(px, py, "o", color=viz.ACCENT)
        ax.set_aspect("equal")
        ax.set_xlabel("east $x$ / km")
        ax.set_ylabel("north $y$ / km")
        plt.show()
        steep = math.hypot(gx, gy)
        angle = math.degrees(math.atan(steep / 1000))  # m per km: / 1000
        print(f"∇h = ({gx:.1f}, {gy:.1f}) m/km; steepest slope "
              f"{steep:.1f} m/km = {angle:.1f}°")


    widgets.interact(
        explorer,
        px=widgets.FloatSlider(0.6, min=-1.8, max=1.8, step=0.1),
        py=widgets.FloatSlider(0.5, min=-1.3, max=1.3, step=0.1),
    );
    """),
    md("""
    The arrow always crosses the contour at a right
    angle and points uphill, and it is longest where the contours crowd
    together. It points straight at the summit only on the $x$ and $y$
    axes, where the ellipses are symmetric.
    """),
]

CURL = [
    md(r"""
    ## 3.11 The curl test

    A function for the curl in the plane, $\partial F_y/\partial x -
    \partial F_x/\partial y$, applied to the forces of the chapter.
    """),
    code("""
    def curl(Fx, Fy):
        return sp.simplify(sp.diff(Fy, x) - sp.diff(Fx, y))


    a, b, c = sp.symbols("a b c", positive=True)
    print("swirl c(−y, x):        ", curl(-c * y, c * x))
    print("(y, x):                ", curl(y, x))
    print("vortex b(−y, x)/ρ²:    ",
          curl(-b * y / (x**2 + y**2), b * x / (x**2 + y**2)))
    U_hill = -h  # any potential energy; here minus the hill
    print("−∇U for U = −h:        ", curl(-sp.diff(U_hill, x),
                                         -sp.diff(U_hill, y)))
    """),
    md("""
    The work of the swirl round a circle, $2c$ times the area, by
    `energy.work` on a polygon of many sides.
    """),
    code("""
    angle = np.linspace(0.0, 2 * np.pi, 2001)[:, None]
    for radius in (0.5, 1.0, 2.0):
        loop = radius * np.hstack([np.cos(angle), np.sin(angle)])
        work = energy.work(lambda r: energy.swirl(r, 1.0), loop)
        print(f"R = {radius}: work {work:.5f}, 2c πR² = "
              f"{2 * np.pi * radius**2:.5f}")
    """),
]

ATOMS = [
    md(r"""
    ## 3.12 The energy of atoms

    The kinetic-energy factor, derived as in Section 3.12 from the
    values of 1 amu and 1 eV, and the factor `mdlab` computes.
    """),
    code("""
    amu_kg, ev_j = 1.66053906892e-27, 1.602176634e-19
    by_hand = amu_kg * (1e-10 / 1e-15) ** 2 / ev_j
    print(f"1 amu Å²/fs² = {by_hand:.7f} eV;  units.MV2_TO_EV = "
          f"{units.MV2_TO_EV:.7f}")
    print(f"1 / MV2_TO_EV = {1 / units.MV2_TO_EV:.9f};  FORCE_TO_ACCEL = "
          f"{units.FORCE_TO_ACCEL:.9f}")
    lithium = energy.kinetic_energy(6.94, [0.0139])
    print(f"lithium at 0.0139 Å/fs: {lithium:.4f} eV")
    """),
    md(r"""
    Two lithium atoms joined by a spring-like pair energy $\varphi(r) =
    \frac12 k(r - r_0)^2$. The forces from Exercise 3.20 are checked
    against central differences of $U$, and the total energy along a
    motion stays fixed while the total force is zero.
    """),
    code("""
    K_PAIR, R0 = 1.0, 3.0  # eV/Å², Å


    def pair(r):
        bond = r[1] - r[0]
        dist = np.linalg.norm(bond)
        u = 0.5 * K_PAIR * (dist - R0) ** 2
        f2 = -K_PAIR * (dist - R0) * bond / dist  # −φ'(r) (r2 − r1)/r
        return u, np.array([-f2, f2])


    rng = np.random.default_rng(3)
    r = rng.normal(scale=2.0, size=(2, 3))
    _, forces = pair(r)
    h = 1e-6
    numerical = np.zeros_like(r)
    for i in range(2):
        for k in range(3):
            step = np.zeros_like(r)
            step[i, k] = h
            up, down = pair(r + step)[0], pair(r - step)[0]
            numerical[i, k] = -(up - down) / (2 * h)
    print("largest force error:", np.max(np.abs(forces - numerical)))
    print("total force:", forces.sum(axis=0))

    masses = np.array([6.94, 6.94])
    t = np.linspace(0.0, 500.0, 2001)
    rs, vs = dynamics.solve_newton(
        lambda q: pair(q)[1], masses, r, rng.normal(scale=0.01, size=(2, 3)),
        t, force_to_accel=units.FORCE_TO_ACCEL,
    )
    total = [energy.kinetic_energy(masses, v) + pair(q)[0]
             for q, v in zip(rs, vs, strict=True)]
    print(f"E varies by {np.ptp(total):.1e} eV over 500 fs")
    """),
]

SURFACE = [
    md(r"""
    ## 3.13 A lithium atom on a model surface

    `energy.hexagonal_surface` with the barrier 0.3 eV and spacing
    2.46 Å. Launch the atom from the hollow at the origin with a chosen
    kinetic energy and direction; the shading is $U$, and the black lines
    bound the regions it may not enter.
    """),
    code("""
    MASS_LI = 6.94
    SITES = energy.surface_sites()
    print({name: float(energy.hexagonal_surface(site)[0])
           for name, site in SITES.items()})


    def surface_force(r):
        return energy.hexagonal_surface(r)[1]


    def launch(kinetic=0.33, angle=37.5, duration=1500.0):
        speed = math.sqrt(2 * kinetic / (MASS_LI * units.MV2_TO_EV))
        theta = np.radians(angle)
        v0 = speed * np.array([np.cos(theta), np.sin(theta)])
        t = np.linspace(0.0, duration, int(duration) * 2 + 1)
        r, v = dynamics.solve_newton(
            surface_force, MASS_LI, SITES["hollow"], v0, t,
            force_to_accel=units.FORCE_TO_ACCEL,
        )
        k = 0.5 * MASS_LI * np.sum(v**2, axis=-1) * units.MV2_TO_EV
        u = energy.hexagonal_surface(r)[0]

        lo, hi = r.min(axis=0) - 1.5, r.max(axis=0) + 1.5
        X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 300),
                           np.linspace(lo[1], hi[1], 300))
        U = energy.hexagonal_surface(np.stack([X, Y], axis=-1))[0]
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(viz.FULL, 2.8), gridspec_kw=dict(wspace=0.4)
        )
        shading = left.contourf(X, Y, U, levels=np.linspace(0, 0.3375, 13),
                                cmap="cividis_r")
        fig.colorbar(shading, ax=left, orientation="horizontal", pad=0.25,
                     fraction=0.07, ticks=[0, 0.15, 0.3], label="$U$ / eV")
        if U.min() < kinetic < U.max():
            left.contour(X, Y, U, levels=[kinetic], colors="black",
                         linewidths=0.6)
        left.plot(*r.T, color="white", lw=2.6)
        left.plot(*r.T, color=viz.ACCENT, lw=1.1)
        left.set_aspect("equal")
        left.set_xlabel("$x$ / Å")
        left.set_ylabel("$y$ / Å")
        right.plot(t, k, color=viz.OCHRE, lw=0.8, label="$K$")
        right.plot(t, u, color=viz.OXBLOOD, ls="--", lw=0.8, label="$U$")
        right.plot(t, k + u, color=viz.ACCENT, label="$E$")
        right.set_xlabel("$t$ / fs")
        right.set_ylabel("energy / eV")
        right.legend(fontsize=8, ncols=3, loc="lower center",
                     bbox_to_anchor=(0.5, 1.0))
        plt.show()
        print(f"speed {speed:.5f} Å/fs; E varies by {np.ptp(k + u):.1e} eV")


    widgets.interact(
        launch,
        kinetic=widgets.FloatSlider(0.33, min=0.05, max=0.6, step=0.01,
                                    **SLOW),
        angle=widgets.FloatSlider(37.5, min=0.0, max=60.0, step=2.5, **SLOW),
        duration=widgets.FloatSlider(1500.0, min=200.0, max=3000.0,
                                     step=100.0, **SLOW),
    );
    """),
    md("""
    Below 0.3 eV the black boundary closes round the
    starting hollow and the atom rattles inside it. Between 0.3 and
    0.3375 eV it can pass through the bridges but never enters the islands
    round the carbon atoms. Above 0.3375 eV no region is forbidden. In
    every case $E$ stays constant to the numerical accuracy reported below
    the plot. A connected allowed region makes a hop possible; the chosen
    launch direction still determines the trajectory.
    """),
]

DRIFT = [
    md(r"""
    ## 3.14 When the energy changes

    Add the swirl $c\,\hat{\mathbf z}\times\mathbf r$ to the surface force
    and compare the change of $E$ with the work the swirl does,
    $\int_0^t \mathbf F_\mathrm{sw}\cdot\mathbf v\,\mathrm dt'$.
    """),
    code("""
    def drift(c=0.005):
        speed = math.sqrt(2 * 0.33 / (MASS_LI * units.MV2_TO_EV))
        theta = np.radians(37.5)
        v0 = speed * np.array([np.cos(theta), np.sin(theta)])
        t = np.linspace(0.0, 1500.0, 7501)

        def force(r):
            return surface_force(r) + energy.swirl(r, c)

        r, v = dynamics.solve_newton(force, MASS_LI, [0.0, 0.0], v0, t,
                                     force_to_accel=units.FORCE_TO_ACCEL)
        e = (0.5 * MASS_LI * np.sum(v**2, axis=-1) * units.MV2_TO_EV
             + energy.hexagonal_surface(r)[0])
        supplied = cumulative_trapezoid(
            np.sum(energy.swirl(r, c) * v, axis=-1), t, initial=0.0
        )
        fig, ax = plt.subplots(figsize=(viz.FULL, 2.2))
        ax.plot(t, e - e[0], color=viz.OXBLOOD, label="$E - E(0)$")
        ax.plot(t, supplied, **viz.REFERENCE_STYLE, label="work of the swirl")
        ax.set_xlabel("$t$ / fs")
        ax.set_ylabel("eV")
        ax.legend(fontsize=8)
        plt.show()
        print(f"largest mismatch {np.max(np.abs(e - e[0] - supplied)):.1e} eV")


    widgets.interact(
        drift,
        c=widgets.FloatSlider(0.005, min=0.0, max=0.02, step=0.001,
                              readout_format=".3f", **SLOW),
    );
    """),
    md("""
    With $c = 0$ the energy is flat. Any swirl makes
    it wander, and the dashed work of the swirl lies on top of it: the
    energy changes at exactly the rate at which the non-conservative force
    does work, Equation (3.18).
    """),
]

EXERCISES = [
    md("""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises' works
    every exercise in full, including 3.1 and 3.13, which ask for
    explanations to write on paper.

    Run the collapsed answer-key cell below without opening it. After
    each exercise a collapsed cell holds a worked solution in code: try
    the exercise first, then open it to compare. The derivations 3.9,
    3.16, 3.17, 3.19 and 3.20 have their solutions checked by SymPy.
    """),
    hidden("""
    # The answer key. Each target is computed rather than typed in.
    _x = np.linspace(-2.4, 2.4, 48001)
    _h, _ = energy.hill_track(_x)
    _roots = np.sort(np.roots([0.6, 0, -1.2, 0.15]).real)
    _bottom = energy.hill_track(_roots)[0][0]
    _H = _bottom + 3.0**2 / (2 * G)
    _ramp = 2.0 / np.sin(np.radians(25))
    _k_o = 0.5 * 15.999 * 0.01**2 * units.MV2_TO_EV
    _k_li = 0.5 * 6.94 * 0.02**2 * units.MV2_TO_EV
    TARGETS = {
        "3.2 work": 60 * 25 * np.cos(np.radians(50)),
        "3.3 work": 70 * G * 3,
        "3.3 power": 70 * G * 3 / 5,
        "3.4 kinetic": 0.5 * 1200 * (50 / 3.6) ** 2,
        "3.4 distance": 0.5 * 1200 * (50 / 3.6) ** 2 / 6000,
        "3.4 distance 100": 0.5 * 1200 * (100 / 3.6) ** 2 / 6000,
        "3.5 first": 0.5 * 200 * 0.1**2,
        "3.5 second": 0.5 * 200 * (0.2**2 - 0.1**2),
        "3.6 speed": np.sqrt(2 * 0.5 * 500 * 0.08**2 / 0.1),
        "3.6 height": 0.5 * 500 * 0.08**2 / (0.1 * G),
        "3.7 speed": np.sqrt(2 * G * 1.2 * (1 - np.cos(np.radians(40)))),
        "3.8 height": _H,
        "3.8 turning": energy.turning_points(_x, _h, _H)[:2],
        "3.10 distance": 3.0**2 / (2 * 0.05 * G),
        "3.11 speed": np.sqrt(2 * (G * 2.0 - 0.15 * G * np.cos(np.radians(25))
                                   * _ramp)),
        "3.12 speeds": np.array([-2.0, 1.0])
        * np.sqrt(0.5 * 200 * 0.05**2 / 3),
        "3.13 before": 0.5 * 2 * 3.0**2,
        "3.13 after": 0.5 * 3 * 2.0**2,
        "3.14 work": 0.5 * 2 * 1,
        "3.15 force": -2.0 * np.array([1.0, 4.0]),
        "3.16 U": -20.0,
        "3.18 work": 2 * np.pi,
        "3.21 distance": _k_o / 0.5,
        "3.22 kinetic": _k_li * np.exp(-1.0),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code("""
    # EXERCISE 3.2: work done by the pull on the suitcase (J).
    work = None

    check(work, TARGETS["3.2 work"], name="work")
    """),
    solution("""
    # SOLUTION 3.2. Only the component along the floor works: F d cos θ.
    work = 60 * 25 * np.cos(np.radians(50))

    check(work, TARGETS["3.2 work"], name="work")
    """),
    code("""
    # EXERCISE 3.3: work against the weight (J) and average power (W).
    work = None
    power = None

    check(work, TARGETS["3.3 work"], name="work")
    check(power, TARGETS["3.3 power"], name="power")
    """),
    solution("""
    # SOLUTION 3.3. A steady climb: the push equals m g over the rise.
    work = 70 * G * 3
    power = work / 5

    check(work, TARGETS["3.3 work"], name="work")
    check(power, TARGETS["3.3 power"], name="power")
    """),
    code("""
    # EXERCISE 3.4: kinetic energy at 50 km/h (J), stopping distance (m),
    # and stopping distance from 100 km/h (m).
    kinetic = None
    distance = None
    distance_100 = None

    check(kinetic, TARGETS["3.4 kinetic"], name="kinetic energy")
    check(distance, TARGETS["3.4 distance"], name="distance")
    check(distance_100, TARGETS["3.4 distance 100"], name="from 100 km/h")
    """),
    solution("""
    # SOLUTION 3.4. The braking force does work −F d = −K.
    speed = 50 * 1000 / 3600  # m/s
    kinetic = 0.5 * 1200 * speed**2
    distance = kinetic / 6000
    distance_100 = 4 * distance  # twice the speed, four times K

    check(kinetic, TARGETS["3.4 kinetic"], name="kinetic energy")
    check(distance, TARGETS["3.4 distance"], name="distance")
    check(distance_100, TARGETS["3.4 distance 100"], name="from 100 km/h")
    """),
    code("""
    # EXERCISE 3.5: work from 0 to 10 cm, then from 10 to 20 cm (J).
    first = None
    second = None

    check(first, TARGETS["3.5 first"], name="first stretch")
    check(second, TARGETS["3.5 second"], name="second stretch")
    """),
    solution("""
    # SOLUTION 3.5. The work from x_A to x_B is k (x_B² − x_A²)/2.
    k = 200.0
    first = 0.5 * k * (0.1**2 - 0.0**2)
    second = 0.5 * k * (0.2**2 - 0.1**2)

    check(first, TARGETS["3.5 first"], name="first stretch")
    check(second, TARGETS["3.5 second"], name="second stretch")
    """),
    code("""
    # EXERCISE 3.6: the puck's speed (m/s) and the height it reaches (m).
    speed = None
    height = None

    check(speed, TARGETS["3.6 speed"], name="speed")
    check(height, TARGETS["3.6 height"], name="height")
    """),
    solution("""
    # SOLUTION 3.6. Spring energy -> kinetic energy -> potential energy.
    stored = 0.5 * 500 * 0.08**2  # J
    speed = math.sqrt(2 * stored / 0.1)
    height = stored / (0.1 * G)

    check(speed, TARGETS["3.6 speed"], name="speed")
    check(height, TARGETS["3.6 height"], name="height")
    """),
    code("""
    # EXERCISE 3.7: the bob's speed at the bottom of its swing (m/s).
    speed = None

    check(speed, TARGETS["3.7 speed"], name="speed")
    """),
    solution("""
    # SOLUTION 3.7. The string does no work; the bob falls L (1 − cos 40°).
    drop = 1.2 * (1 - math.cos(math.radians(40)))
    speed = math.sqrt(2 * G * drop)

    check(speed, TARGETS["3.7 speed"], name="speed")
    """),
    code("""
    # EXERCISE 3.8: the greatest height h_E (m) and the two turning points
    # in the deep valley (m), as an array [left, right].
    h_E = None
    turning = None

    check(h_E, TARGETS["3.8 height"], name="height")
    check(turning, TARGETS["3.8 turning"], name="turning points")
    """),
    solution("""
    # SOLUTION 3.8. h_E = h + v²/2g at the valley bottom; the bead started in
    # the deep valley, so its turning points are the first two crossings.
    h_E = PEAKS[0] + 3.0**2 / (2 * G)
    turning = energy.turning_points(X_TRACK, H_TRACK, h_E)[:2]
    print(f"h_E = {h_E:.3f} m, below the hill at {PEAKS[1]:.3f} m")

    check(h_E, TARGETS["3.8 height"], name="height")
    check(turning, TARGETS["3.8 turning"], name="turning points")
    """),
    md("""
    **Exercise 3.9** is a derivation. The collapsed cell checks it.
    """),
    solution("""
    # SOLUTION 3.9. Equilibria, their stability and the turning points.
    a0, U0 = sp.symbols("a U_0", positive=True)
    xs = sp.Symbol("x", real=True)
    U = U0 * ((xs / a0) ** 2 - 1) ** 2
    equilibria = sp.solve(sp.diff(U, xs), xs)
    print("equilibria:", equilibria)
    for x0 in equilibria:
        print(f"  U''({x0}) =", sp.simplify(sp.diff(U, xs, 2).subs(xs, x0)))
    turns = sp.solve(sp.Eq(U, U0 / 2), xs)
    print("turning points:", turns)
    found = sorted(float(t.subs(a0, 1)) for t in turns)
    expected = sorted(s1 * math.sqrt(1 + s2 / math.sqrt(2))
                      for s1 in (1, -1) for s2 in (1, -1))
    print("in units of a:", np.round(found, 4))
    assert np.allclose(found, expected)
    """),
    code("""
    # EXERCISE 3.10: how far the puck slides on the ice (m).
    distance = None

    check(distance, TARGETS["3.10 distance"], name="distance")
    """),
    solution("""
    # SOLUTION 3.10. Friction μ m g removes all of m v²/2: d = v²/2μg.
    distance = 3.0**2 / (2 * 0.05 * G)

    check(distance, TARGETS["3.10 distance"], name="distance")
    """),
    code("""
    # EXERCISE 3.11: the block's speed at the bottom of the rough ramp (m/s).
    speed = None

    check(speed, TARGETS["3.11 speed"], name="speed")
    """),
    solution("""
    # SOLUTION 3.11. Per kilogram: g × 2 m from the weight, minus friction.
    theta = math.radians(25)
    length = 2.0 / math.sin(theta)
    per_kg = G * 2.0 - 0.15 * G * math.cos(theta) * length  # v²/2
    speed = math.sqrt(2 * per_kg)

    check(speed, TARGETS["3.11 speed"], name="speed")
    """),
    code("""
    # EXERCISE 3.12: the velocities of the 1 kg cart (on the left) and the
    # 2 kg cart (m/s), x to the right, as an array [v1, v2].
    speeds = None

    check(speeds, TARGETS["3.12 speeds"], name="velocities")
    """),
    solution("""
    # SOLUTION 3.12. Momentum gives v1 = −2 v2; energy gives 3 v2² = 0.25.
    v1, v2 = sp.symbols("v1 v2", real=True)
    found = sp.solve([sp.Eq(v1 + 2 * v2, 0),
                      sp.Eq(v1**2 / 2 + v2**2, sp.Rational(1, 4))], [v1, v2])
    v1_value, v2_value = [s for s in found if s[1] > 0][0]  # cart 2 forwards
    speeds = np.array([float(v1_value), float(v2_value)])

    check(speeds, TARGETS["3.12 speeds"], name="velocities")
    """),
    code("""
    # EXERCISE 3.13: kinetic energy before and after the coupling (J).
    before = None
    after = None

    check(before, TARGETS["3.13 before"], name="before")
    check(after, TARGETS["3.13 after"], name="after")
    """),
    solution("""
    # SOLUTION 3.13. 3 J is lost to deformation and warmth.
    before = 0.5 * 2 * 3.0**2
    after = 0.5 * (2 + 1) * 2.0**2
    print(f"lost: {before - after} J")

    check(before, TARGETS["3.13 before"], name="before")
    check(after, TARGETS["3.13 after"], name="after")
    """),
    code("""
    # EXERCISE 3.14: the work along either route (eV).
    work = None

    check(work, TARGETS["3.14 work"], name="work")
    """),
    solution("""
    # SOLUTION 3.14. Both routes by energy.work, and U = −a x y.
    a_value = 0.5


    def field(r):
        return a_value * np.stack([r[..., 1], r[..., 0]], axis=-1)


    s = np.linspace(0.0, 1.0, 1001)[:, None]
    straight = s * np.array([2.0, 1.0])
    two_legs = np.vstack([s * [2.0, 0.0], [2.0, 0.0] + s * [0.0, 1.0]])
    print(f"straight {energy.work(field, straight):.6f} eV, "
          f"two legs {energy.work(field, two_legs):.6f} eV")
    work = -(-a_value * 2.0 * 1.0)  # U(0) − U(2, 1)

    check(work, TARGETS["3.14 work"], name="work")
    """),
    code("""
    # EXERCISE 3.15: the force at (1, 1) Å (eV/Å), as an array [Fx, Fy].
    force = None

    check(force, TARGETS["3.15 force"], name="force")
    """),
    solution("""
    # SOLUTION 3.15. F = −∇U = −k (x, 4y); the contour tangent is (4, −1).
    k = 2.0
    force = -k * np.array([1.0, 4.0 * 1.0])
    print("F · tangent =", force @ np.array([4.0, -1.0]))

    check(force, TARGETS["3.15 force"], name="force")
    """),
    code("""
    # EXERCISE 3.16: U at (1, 2, 3) Å for a = 1 eV/Å³ (eV).
    U_value = None

    check(U_value, TARGETS["3.16 U"], name="U(1, 2, 3)")
    """),
    solution("""
    # SOLUTION 3.16. Zero curl, and U = −a (x² y + y z²) gives back F.
    a, x, y, z = sp.symbols("a x y z", real=True)  # fresh symbols
    Fv = a * sp.Matrix([2 * x * y, x**2 + z**2, 2 * y * z])
    q = (x, y, z)
    curl3 = [sp.diff(Fv[2], y) - sp.diff(Fv[1], z),
             sp.diff(Fv[0], z) - sp.diff(Fv[2], x),
             sp.diff(Fv[1], x) - sp.diff(Fv[0], y)]
    print("curl:", [sp.simplify(c3) for c3 in curl3])
    U = -a * (x**2 * y + y * z**2)
    assert all(sp.simplify(-sp.diff(U, q[i]) - Fv[i]) == 0 for i in range(3))
    U_value = float(U.subs({a: 1, x: 1, y: 2, z: 3}))

    check(U_value, TARGETS["3.16 U"], name="U(1, 2, 3)")
    """),
    md("""
    **Exercise 3.17** is a calculation by hand; the collapsed cell does it
    side by side in SymPy.
    """),
    solution("""
    # SOLUTION 3.17. Each side as a line integral, anticlockwise.
    L, t, c = sp.symbols("L t c", positive=True)
    sides = [sp.Matrix([t, 0]), sp.Matrix([L, t]),
             sp.Matrix([L - t, L]), sp.Matrix([0, L - t])]
    total = 0
    for side in sides:
        Fs = sp.Matrix([-c * side[1], c * side[0]])
        total += sp.integrate(Fs.dot(side.diff(t)), (t, 0, L))
    print("work round the square:", sp.simplify(total))
    assert sp.simplify(total - 2 * c * L**2) == 0
    """),
    code("""
    # EXERCISE 3.18: the work once round the origin for b = 1 eV (eV).
    work = None

    check(work, TARGETS["3.18 work"], name="work")
    """),
    solution("""
    # SOLUTION 3.18. 2π b on every circle round the origin; 0 on one beside it.
    def vortex(r):
        rho2 = np.sum(r**2, axis=-1, keepdims=True)
        return np.concatenate([-r[..., 1:2], r[..., 0:1]], axis=-1) / rho2


    angle = np.linspace(0.0, 2 * np.pi, 4001)[:, None]
    circle = np.hstack([np.cos(angle), np.sin(angle)])
    for radius in (0.5, 1.0, 2.0):
        print(f"R = {radius}: {energy.work(vortex, radius * circle):.6f} eV")
    beside = energy.work(vortex, circle + [3.0, 0.0])
    print(f"beside the origin: {beside:.1e} eV")
    work = energy.work(vortex, circle)

    check(work, TARGETS["3.18 work"], name="work")
    """),
    md("""
    **Exercises 3.19 and 3.20** are derivations; the collapsed cells check
    them.
    """),
    solution("""
    # SOLUTION 3.19. The three products at the top, and U there.
    a_s, D = sp.symbols("a U_b", positive=True)
    g_len = 4 * sp.pi / (sp.sqrt(3) * a_s)
    top = sp.Matrix([a_s / 2, a_s / (2 * sp.sqrt(3))])
    products = []
    for deg in (-30, 90, 210):
        ang = sp.rad(deg)
        g_k = g_len * sp.Matrix([sp.cos(ang), sp.sin(ang)])
        products.append(sp.simplify(g_k.dot(top)))
    print("g_k · r:", products)
    U_top = sp.simplify(D / 4 * (3 - sum(sp.cos(p) for p in products)))
    print("U at the top:", U_top)
    assert sp.simplify(U_top - 9 * D / 8) == 0
    """),
    solution("""
    # SOLUTION 3.20. Pair forces from U = φ(r12).
    x1, y1, z1, x2, y2, z2 = sp.symbols("x1 y1 z1 x2 y2 z2", real=True)
    phi = sp.Function("phi")
    r12 = sp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    U = phi(r12)
    F1 = sp.Matrix([-sp.diff(U, v) for v in (x1, y1, z1)])
    F2 = sp.Matrix([-sp.diff(U, v) for v in (x2, y2, z2)])
    print("∂r12/∂x2 =", sp.simplify(sp.diff(r12, x2)))
    print("F1 + F2 =", sp.simplify(F1 + F2).T)
    assert sp.simplify(F1 + F2) == sp.zeros(3, 1)
    """),
    code("""
    # EXERCISE 3.21: how far the oxygen atom travels before it stops (Å).
    distance = None

    check(distance, TARGETS["3.21 distance"], name="distance")
    """),
    solution("""
    # SOLUTION 3.21. K in eV from MV2_TO_EV; the force does −F d = −K.
    kinetic = energy.kinetic_energy(15.999, [0.01])
    distance = kinetic / 0.5

    check(distance, TARGETS["3.21 distance"], name="distance")
    """),
    code("""
    # EXERCISE 3.22: the lithium atom's kinetic energy after 100 fs (eV).
    kinetic = None

    check(kinetic, TARGETS["3.22 kinetic"], name="kinetic energy")
    """),
    solution("""
    # SOLUTION 3.22. K(t) = K(0) e^(−2γt), checked against the motion.
    gamma = 0.005  # 1/fs
    k0 = energy.kinetic_energy(6.94, [0.02])
    kinetic = k0 * math.exp(-2 * gamma * 100)
    # the drag −γ m v, in eV/Å: dividing by FORCE_TO_ACCEL turns
    # amu Å/fs² into eV/Å, so that the solver's a = F/m gives −γ v
    _, v = dynamics.solve_newton(
        lambda r, v: -gamma * 6.94 * v / units.FORCE_TO_ACCEL,
        6.94, [0.0], [0.02], [0.0, 100.0],
        force_to_accel=units.FORCE_TO_ACCEL, velocity_dependent=True,
    )
    print(f"from the motion: {energy.kinetic_energy(6.94, v[-1]):.5f} eV")

    check(kinetic, TARGETS["3.22 kinetic"], name="kinetic energy")
    """),
]

CELLS = (
    SETUP
    + CONSTANT
    + KINETIC
    + VARYING
    + POTENTIAL
    + TRACK
    + DIAGRAMS
    + FRICTION
    + BODIES
    + PATHS
    + GRADIENT
    + CURL
    + ATOMS
    + SURFACE
    + DRIFT
    + EXERCISES
)

if __name__ == "__main__":
    print("wrote", write(CELLS, "03_energy.ipynb"))
