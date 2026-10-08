"""Write notebooks/01_motion.ipynb, the companion to Chapter 1.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_01_motion.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/01_motion.ipynb
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md("""
    # Notebook 01: motion

    We start with a thrown ball and use its motion to work out what a
    velocity, an acceleration and a numerical derivative measure. The
    same calculations then describe a vibrating atomic coordinate.

    Run the cells in order, then return to the sliders to change one
    quantity at a time. Section numbers match Chapter 1. Keep a copy of
    the notebook for your calculations and observations.
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
    from scipy.optimize import brentq

    from mdlab import kinematics, viz
    from mdlab.exercise import check

    viz.use_style()
    V0 = 12.0  # launch speed of the ball, m/s
    SLOW = dict(continuous_update=False)  # redraw only on release
    """),
]

PYTHON = [
    md("""
    ## Python for these notebooks

    A notebook is a column of cells. A code cell runs when you press
    Shift-Enter, and what it defines stays defined for every later cell,
    which is why the cells must be run from the top. The first cell
    imports the libraries: NumPy (`np`) for arrays of numbers, Matplotlib
    (`plt`) for plots, SymPy (`sp`) for algebra, and `mdlab`, the code the
    book builds.

    A NumPy array holds many numbers at once, and arithmetic on it acts
    on every element, so one line computes a whole table. Square brackets
    pick elements, counting from 0: `x[0]` is the first, `x[-1]` the last,
    `x[2:]` everything from the third on and `x[:-2]` everything except
    the last two. A function is defined with `def`, takes arguments and
    `return`s a result. The cell below shows each of these once.
    """),
    code("""
    t = np.linspace(0.0, 1.0, 5)  # 5 evenly spaced times from 0 to 1 s
    print("t       =", t)
    print("t**2    =", t**2)  # every element squared
    print("t[0], t[-1] =", t[0], t[-1])
    print("t[2:]   =", t[2:])
    print("t[:-2]  =", t[:-2])


    def height_of(t, v0=12.0):
        # height (m) of a ball thrown up at v0 (m/s), at times t (s)
        return v0 * t - 0.5 * G * t**2


    print(f"height at 0.5 s: {height_of(0.5):.3f} m")  # f-string formatting
    """),
]

UNITS = [
    md("""
    ## 1.1 Quantities and units

    A quantity is a number times a unit. Changing the unit changes the
    number, not the quantity.
    """),
    code("""
    ANGSTROM = 1e-10  # metres
    FEMTOSECOND = 1e-15  # seconds

    print(f"1 Å/fs = {ANGSTROM / FEMTOSECOND:.0e} m/s")
    ball_in_a_per_fs = V0 * FEMTOSECOND / ANGSTROM
    print(f"the ball at {V0} m/s = {ball_in_a_per_fs:.1e} Å/fs")
    """),
]

VELOCITY = [
    md(r"""
    ## 1.2 Velocity

    The height of the ball is $x(t) = v_0 t - \tfrac12 g t^2$. The average
    velocity between $t_1$ and $t_1 + \tau$ is the slope of the secant;
    the velocity at $t_1$ is the limit as $\tau \to 0$, the slope of the
    tangent.
    """),
    code("""
    def ball_height(t):
        x, _ = kinematics.constant_acceleration(t, 0.0, V0, -G)
        return x


    flight_times = np.linspace(0.0, 2 * V0 / G, 300)


    def secant(t1=0.4, tau=1.0):
        slope = (ball_height(t1 + tau) - ball_height(t1)) / tau
        v_t1 = V0 - G * t1

        fig, ax = plt.subplots(figsize=(viz.HALF * 1.4, 2.6))
        ax.plot(flight_times, ball_height(flight_times), color="black", lw=1.0)
        ax.plot(
            [t1, t1 + tau],
            [ball_height(t1), ball_height(t1 + tau)],
            "o-",
            color=viz.OCHRE,
            ms=3,
            label=f"secant, slope {slope:.3f} m/s",
        )
        span = np.array([-0.4, 0.6])
        ax.plot(
            t1 + span,
            ball_height(t1) + v_t1 * span,
            color=viz.ACCENT,
            label=f"tangent, slope {v_t1:.3f} m/s",
        )
        ax.set_xlabel("time $t$ / s")
        ax.set_ylabel("height $x$ / m")
        ax.set_ylim(0, 9)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=8)
        plt.show()

        gap = slope - v_t1
        print(
            f"secant − tangent = {gap:+.4f} m/s;  "
            f"−gτ/2 = {-G * tau / 2:+.4f}"
        )


    widgets.interact(
        secant,
        t1=widgets.FloatSlider(0.4, min=0.0, max=1.4, step=0.05),
        tau=widgets.FloatLogSlider(1.0, base=10, min=-3, max=0, step=0.05),
    );
    """),
    md(r"""
    As $\tau$ shrinks the secant turns onto the
    tangent, and the gap between the two slopes is always exactly
    $-g\tau/2$: halve $\tau$ and the gap halves. Move $t_1$ past the top
    of the flight ($t_1 > 1.22$ s) and the tangent's slope turns negative.

    The same algebra done by SymPy: substitute $t_1 + \tau$, subtract
    $x(t_1)$, divide by $\tau$, then take the limit.
    """),
    code("""
    t1, tau, v0, g = sp.symbols("t_1 tau v_0 g", positive=True)


    def x(t):
        return v0 * t - g * t**2 / 2


    average = sp.expand((x(t1 + tau) - x(t1)) / tau)
    print("average velocity :", average)
    print("limit as τ → 0   :", sp.limit(average, tau, 0))
    """),
]

ACCELERATION = [
    md("""
    ## 1.3 Acceleration

    Move the cursor through the flight and compare the height, velocity
    and acceleration at the same instant. At the top, decide which of
    these quantities vanishes before reading its value.
    """),
    code("""
    def cursor(t_cursor=1.0):
        x, v = kinematics.constant_acceleration(flight_times, 0.0, V0, -G)
        a = np.full_like(flight_times, -G)
        x_now, v_now = kinematics.constant_acceleration(t_cursor, 0.0, V0, -G)

        fig, axes = plt.subplots(3, 1, figsize=(viz.HALF * 1.6, 3.6), sharex=True)
        rows = (
            (x, x_now, "$x$ / m"),
            (v, v_now, "$v$ / m s$^{-1}$"),
            (a, -G, "$a$ / m s$^{-2}$"),
        )
        for ax, (curve, value, label) in zip(axes, rows, strict=True):
            ax.plot(flight_times, curve, color=viz.ACCENT)
            ax.axvline(t_cursor, **viz.THRESHOLD_STYLE)
            ax.plot(t_cursor, value, "o", color="black", ms=4)
            ax.axhline(0.0, color="black", lw=0.4)
            ax.set_ylabel(label)
        axes[-1].set_xlabel("time $t$ / s")
        plt.show()

        print(f"t = {t_cursor:.3f} s:  x = {x_now:.3f} m,  "
              f"v = {v_now:+.3f} m/s,  a = {-G:.3f} m/s²")


    widgets.interact(
        cursor,
        t_cursor=widgets.FloatSlider(
            1.0, min=0.0, max=2 * V0 / G, step=0.01, readout_format=".2f"
        ),
    );
    """),
    md(r"""
    Put the cursor at the top, $t = v_0/g = 1.224$
    s: the height is greatest, the velocity is zero, and the acceleration
    is still $-g$. Before the top $v$ and $a$ have opposite signs and the
    ball slows; after it they share a sign and the ball speeds up.
    """),
]

INTEGRATING = [
    md("""
    ## 1.4 From acceleration back to motion

    The displacement $x(t) - x(0)$ is the signed area under $v(t)$. The
    shaded area is computed numerically and compared with the height.
    """),
    code("""
    def area(t_cursor=1.224):
        _, v = kinematics.constant_acceleration(flight_times, 0.0, V0, -G)
        before = flight_times <= t_cursor

        fig, ax = plt.subplots(figsize=(viz.HALF * 1.4, 2.4))
        ax.plot(flight_times, v, color=viz.ACCENT)
        ax.fill_between(
            flight_times[before], v[before], 0.0, color=viz.ACCENT, alpha=0.2, lw=0
        )
        ax.axhline(0.0, color="black", lw=0.4)
        ax.set_xlabel("time $t$ / s")
        ax.set_ylabel("$v$ / m s$^{-1}$")
        plt.show()

        fine = np.linspace(0.0, t_cursor, 2001)
        _, v_fine = kinematics.constant_acceleration(fine, 0.0, V0, -G)
        print(f"area under v up to {t_cursor:.3f} s: "
              f"{np.trapezoid(v_fine, fine):.4f} m;  "
              f"x(t) = {ball_height(t_cursor):.4f} m")


    widgets.interact(
        area,
        t_cursor=widgets.FloatSlider(
            1.224, min=0.0, max=2 * V0 / G, step=0.01, readout_format=".2f"
        ),
    );
    """),
    md(r"""
    Up to the top the area is the greatest height,
    7.342 m. Past the top, area below the axis counts as negative and
    the total shrinks; at $t = 2v_0/g$ the two areas cancel and the ball
    is back at the hand.

    The elimination of $t$ that gives $v^2 = v_0^2 + 2a(x - x_0)$, checked
    by SymPy: substitute $t = (v - v_0)/a$ into the position and compare.
    """),
    code("""
    v, a, x0 = sp.symbols("v a x_0")
    t_of_v = (v - v0) / a
    position = x0 + v0 * t_of_v + a * t_of_v**2 / 2

    print("x − x0                =", sp.simplify(position - x0))
    print("v² − v0² − 2a(x − x0) =",
          sp.simplify(v**2 - v0**2 - 2 * a * (position - x0)))
    """),
]

TRIG = [
    md("""
    ## 1.5 Angles and the trigonometric functions

    Sine and cosine are the coordinates of a point on the unit circle.
    Turn the angle and watch the point, its two coordinates, and the same
    values on the unrolled graphs.
    """),
    code("""
    angles = np.linspace(0.0, 2 * np.pi, 300)


    def unit_circle(theta=1.1):
        c, s = np.cos(theta), np.sin(theta)
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(viz.FULL, 2.6), gridspec_kw=dict(width_ratios=(1, 1.6))
        )
        # the circle, the arc of length θ, and the two coordinates of P
        left.plot(np.cos(angles), np.sin(angles), color="black", lw=0.8)
        left.axhline(0, color="black", lw=0.4)
        left.axvline(0, color="black", lw=0.4)
        arc = np.linspace(0.0, theta, 80)
        left.plot(np.cos(arc), np.sin(arc), color="black", lw=2.4)
        left.plot([0, c], [0, s], color="black", lw=0.8)
        left.plot([c, c], [0, s], color=viz.ACCENT, lw=1.8)
        left.plot([0, c], [0, 0], color=viz.OCHRE, lw=1.8)
        left.plot(c, s, "o", color="black", ms=5)
        left.set_aspect("equal")
        left.set_xlim(-1.2, 1.2)
        left.set_ylim(-1.2, 1.2)
        left.set_xlabel(r"$x = \\cos\\theta$")
        left.set_ylabel(r"$y = \\sin\\theta$")
        # the same two coordinates as functions of the angle
        right.plot(angles, np.sin(angles), color=viz.ACCENT, label="sin θ")
        right.plot(angles, np.cos(angles), color=viz.OCHRE, ls="--", label="cos θ")
        right.axhline(0, color="black", lw=0.4)
        right.axvline(theta, **viz.THRESHOLD_STYLE)
        right.plot([theta, theta], [s, c], "o", color="black", ms=4)
        viz.pi_ticks(right, 0, 4)
        right.set_xlabel("θ / rad")
        right.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8)
        plt.show()

        print(f"θ = {theta:.3f} rad = {np.degrees(theta):.1f}°:  "
              f"cos θ = {c:+.4f},  sin θ = {s:+.4f},  "
              f"cos² + sin² = {c * c + s * s:.6f}")


    widgets.interact(
        unit_circle,
        theta=widgets.FloatSlider(
            1.1, min=0.0, max=2 * np.pi, step=0.02, readout_format=".2f"
        ),
    );
    """),
    md(r"""
    The teal segment is the height of the point and
    the ochre segment its horizontal position; their lengths are the two
    values marked on the graphs. Past $\pi/2$ the cosine turns negative,
    past $\pi$ the sine does, and $\cos^2\theta + \sin^2\theta$ stays 1
    throughout, because the point never leaves the circle.

    The limit $\sin h / h \to 1$ of Section 1.5, and the derivative it
    gives, checked numerically: the difference quotient of $\sin$
    approaches $\cos$.
    """),
    code("""
    for h in (0.5, 0.1, 0.01, 0.001):
        quotient = (np.sin(1.0 + h) - np.sin(1.0)) / h
        print(f"h = {h:<6}  sin h / h = {np.sin(h) / h:.8f}   "
              f"[sin(1 + h) − sin 1] / h = {quotient:.6f}   "
              f"(cos 1 = {np.cos(1.0):.6f})")
    """),
    md("The addition formulae, checked by SymPy for symbolic angles:"),
    code("""
    p, q = sp.symbols("p q")
    cos_rule = sp.cos(p - q) - (sp.cos(p) * sp.cos(q) + sp.sin(p) * sp.sin(q))
    sin_rule = sp.sin(p + q) - (sp.sin(p) * sp.cos(q) + sp.cos(p) * sp.sin(q))
    print(sp.simplify(cos_rule), sp.simplify(sin_rule))
    """),
    md(r"""
    **A sinusoid** $x = A\sin(\omega t + \phi)$: set its amplitude,
    angular frequency and phase, against the unshifted $A\sin\omega t$
    (dashed).
    """),
    code("""
    def sinusoid(A=1.5, omega=np.pi, phi=np.pi / 3):
        times = np.linspace(0.0, 6.0, 600)
        fig, ax = plt.subplots(figsize=(viz.FULL, 2.3))
        ax.plot(times, A * np.sin(omega * times), lw=1.0, **viz.REFERENCE_STYLE)
        ax.plot(times, A * np.sin(omega * times + phi), color=viz.ACCENT)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("time $t$ / s")
        ax.set_ylabel("$x$ / m")
        plt.show()

        print(f"period 2π/ω = {2 * np.pi / omega:.3f} s, "
              f"frequency = {omega / (2 * np.pi):.3f} Hz, "
              f"crests earlier by φ/ω = {phi / omega:.3f} s")


    widgets.interact(
        sinusoid,
        A=widgets.FloatSlider(1.5, min=0.2, max=3.0, step=0.1),
        omega=widgets.FloatSlider(np.pi, min=0.5, max=10.0, step=0.1),
        phi=widgets.FloatSlider(np.pi / 3, min=-np.pi, max=np.pi, step=0.05),
    );
    """),
    md(r"""
    $A$ stretches the wave vertically without moving
    its crests. Doubling $\omega$ halves the period. $\phi$ slides the
    whole wave to the left by $\phi/\omega$; at $\phi = \pi/2$ the teal
    curve is $A\cos\omega t$.
    """),
]

PLANE = [
    md("""
    ## 1.6 Motion in a plane and in space

    A projectile: each component obeys the constant-acceleration formulas
    on its own. Velocity arrows (teal) follow the path; acceleration
    arrows (red) always point down.
    """),
    code("""
    def projectile(speed=12.0, angle=60.0):
        alpha = np.radians(angle)
        launch = speed * np.array([np.cos(alpha), np.sin(alpha)])
        gravity = np.array([0.0, -G])
        flight = 2 * launch[1] / G

        times = np.linspace(0.0, flight, 200)
        path, _ = kinematics.constant_acceleration(
            times, np.zeros(2), launch, gravity
        )
        marks = np.linspace(0.0, flight, 7)  # odd, so one mark is at the top
        r, v = kinematics.constant_acceleration(
            marks, np.zeros(2), launch, gravity
        )

        fig, ax = plt.subplots(figsize=(viz.FULL, 2.6))
        ax.plot(*path.T, color="black", lw=1.0)
        ax.quiver(*r.T, *v.T, color=viz.ACCENT, scale=1 / 0.12, **viz.ARROW)
        down = np.broadcast_to(gravity, r.shape)
        ax.quiver(*r.T, *down.T, color=viz.OXBLOOD, scale=1 / 0.12, **viz.ARROW)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 26)
        ax.set_ylim(-2, 13)
        ax.set_xlabel("$x$ / m")
        ax.set_ylabel("$y$ / m")
        plt.show()

        exact_range = speed**2 * np.sin(2 * alpha) / G
        print(f"range: computed {path[-1, 0]:.3f} m;  "
              f"v0² sin 2α / g = {exact_range:.3f} m")


    widgets.interact(
        projectile,
        speed=widgets.FloatSlider(12.0, min=4.0, max=15.0, step=0.5),
        angle=widgets.FloatSlider(60.0, min=5.0, max=85.0, step=1.0),
    );
    """),
    md("""
    For a fixed speed the range is largest at 45°,
    and angles that add up to 90° (30° and 60°, say) land at the same
    place. At the top of every path the velocity is horizontal and the
    acceleration is still $g$ downwards.

    The same flight as an animation.
    """),
    code("""
    def animate_projectile():
        alpha = np.radians(60.0)
        launch = V0 * np.array([np.cos(alpha), np.sin(alpha)])
        frames = np.linspace(0.0, 2 * launch[1] / G, 51)
        path, velocities = kinematics.constant_acceleration(
            frames, np.zeros(2), launch, [0.0, -G]
        )

        fig, ax = plt.subplots(figsize=(viz.FULL * 0.8, 2.4), dpi=120)
        ax.plot(*path.T, color="black", lw=0.6, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 15)
        ax.set_ylim(-2, 7.5)
        ax.set_xlabel("$x$ / m")
        ax.set_ylabel("$y$ / m")
        (ball,) = ax.plot([], [], "o", color="black", ms=5)
        arrow = dict(scale=1 / 0.12, **viz.ARROW)  # drawn length 0.12 per unit
        v_arrow = ax.quiver([0], [0], [0], [0], color=viz.ACCENT, **arrow)
        a_arrow = ax.quiver([0], [0], [0], [0], color=viz.OXBLOOD, **arrow)


        def draw_frame(k):
            ball.set_data([path[k, 0]], [path[k, 1]])
            v_arrow.set_offsets(path[k])
            v_arrow.set_UVC(*velocities[k])
            a_arrow.set_offsets(path[k])
            a_arrow.set_UVC(0.0, -G)
            return ball, v_arrow, a_arrow


        animation = FuncAnimation(fig, draw_frame, frames=len(frames), interval=60)
        plt.close(fig)
        return HTML(animation.to_jshtml())


    animate_projectile()
    """),
]

CIRCLE = [
    md(r"""
    ## 1.7 Circular motion

    A point going round a circle at constant speed, with its velocity
    (teal) and acceleration (red), and its projection on the $x$ axis
    (grey), which moves as $x = R\cos\omega t$.
    """),
    code("""
    def animate_circle():
        R, OMEGA = 1.0, 1.0
        times = np.linspace(0.0, 2 * np.pi, 60, endpoint=False)
        r, v, a = kinematics.circular_motion(times, R, OMEGA)
        FLOOR = -1.4  # height of the line the projection moves along

        fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=120)
        ax.plot(R * np.cos(angles), R * np.sin(angles), color="black", lw=0.8)
        ax.axhline(FLOOR, color=viz.REFERENCE, lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xlabel("$x / R$")
        ax.set_ylabel("$y / R$")
        (point,) = ax.plot([], [], "o", color="black", ms=5)
        (shadow,) = ax.plot([], [], "o", color=viz.REFERENCE, ms=6)
        (drop,) = ax.plot([], [], ":", color=viz.REFERENCE, lw=0.8)
        arrow = dict(scale=1 / 0.5, **viz.ARROW)  # drawn length 0.5 per unit
        v_arrow = ax.quiver([0], [0], [0], [0], color=viz.ACCENT, **arrow)
        a_arrow = ax.quiver([0], [0], [0], [0], color=viz.OXBLOOD, **arrow)


        def draw_frame(k):
            x, y = r[k]
            point.set_data([x], [y])
            shadow.set_data([x], [FLOOR])
            drop.set_data([x, x], [y, FLOOR])
            v_arrow.set_offsets(r[k])
            v_arrow.set_UVC(*v[k])
            a_arrow.set_offsets(r[k])
            a_arrow.set_UVC(*a[k])
            return point, shadow, drop, v_arrow, a_arrow


        animation = FuncAnimation(fig, draw_frame, frames=len(times), interval=60)
        plt.close(fig)
        return HTML(animation.to_jshtml())


    animate_circle()
    """),
    md(r"""
    The red arrow always points at the centre and
    the teal arrow along the circle: the speed never changes, but the
    velocity's direction does, and that change is the acceleration. The
    grey projection slows at the ends and is fastest in the middle, the
    motion Chapter 4 finds for a block on a spring.

    A numerical check of $\mathbf a = -\omega^2 \mathbf r$: differentiate
    the positions twice by central differences.
    """),
    code("""
    R, OMEGA = 1.0, 1.0
    fine = np.linspace(0.0, 2 * np.pi, 2001)
    tau = fine[1] - fine[0]
    r_fine, _, _ = kinematics.circular_motion(fine, R, OMEGA)
    v_estimate = kinematics.central_difference(r_fine, tau)
    a_estimate = kinematics.central_difference(v_estimate, tau)
    mismatch = a_estimate + OMEGA**2 * r_fine[2:-2]
    print(f"largest |a_estimate + ω² r| = {np.max(np.abs(mismatch)):.1e}")
    """),
]

TAYLOR = [
    md(r"""
    ## 1.8 Short intervals and the Taylor series

    Polynomials of increasing degree, built from the derivatives of
    $\sin t$ at $t_1$, close in on the function near $t_1$. Choose the
    degree and the expansion point.
    """),
    code("""
    def taylor_sin(degree=2, t1=0.785):
        # the derivatives of sin repeat: sin, cos, −sin, −cos, ...
        derivatives = [np.sin(t1), np.cos(t1), -np.sin(t1), -np.cos(t1)]

        def polynomial(step):
            return sum(
                derivatives[n % 4] * step**n / math.factorial(n)
                for n in range(degree + 1)
            )

        tt = np.linspace(-2.0, 4.0, 400)
        fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.6))
        ax.plot(tt, np.sin(tt), color="black", lw=1.4, label=r"$\\sin t$")
        ax.plot(
            tt, polynomial(tt - t1), color=viz.ACCENT, label=f"degree {degree}"
        )
        ax.plot(t1, np.sin(t1), "o", color="black", ms=4)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x(t)$")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8)
        plt.show()

        for step in (0.1, 0.05):
            error = abs(np.sin(t1 + step) - polynomial(step))
            print(f"error at t1 + {step}: {error:.3e}")


    widgets.interact(
        taylor_sin,
        degree=widgets.IntSlider(2, min=0, max=20),
        t1=widgets.FloatSlider(0.785, min=-1.5, max=3.0, step=0.05),
    );
    """),
    md(r"""
    Each extra degree matches one more derivative at the expansion point.
    Compare the two printed errors before increasing the degree: halving
    the interval usually divides the leading error by $2^{n+1}$. At an
    expansion point where the next derivative vanishes, the first missing
    term is of higher order and the error falls faster.
    """),
]

ATOMS = [
    md(r"""
    ## 1.9 From balls to atoms

    An atomic coordinate oscillating with amplitude 0.1 Å and period
    20 fs is saved every $\tau$. The central difference of the saved
    frames (points) is compared with the true velocity (line). A phase of
    0.7 rad keeps the frames off the zeros of $x$.
    """),
    code("""
    AMPLITUDE = 0.1  # Å
    PERIOD = 20.0  # fs
    PHASE = 0.7  # rad
    W = 2 * np.pi / PERIOD
    t_true = np.linspace(0.0, 2 * PERIOD, 800)


    def sampling(tau=1.0):
        saved = np.arange(0.0, 2 * PERIOD + tau, tau)
        x_saved = AMPLITUDE * np.sin(W * saved + PHASE)
        v_estimate = kinematics.central_difference(x_saved, tau)

        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(viz.FULL * 0.8, 3.2), sharex=True
        )
        exact = AMPLITUDE * np.sin(W * t_true + PHASE)
        top.plot(t_true, exact, color="black", lw=0.8)
        top.plot(saved, x_saved, "o", color=viz.OCHRE, ms=3)
        top.set_ylabel("$x$ / Å")
        bottom.plot(
            t_true,
            AMPLITUDE * W * np.cos(W * t_true + PHASE),
            color="black",
            lw=0.8,
            label="true",
        )
        bottom.plot(
            saved[1:-1], v_estimate, "o", color=viz.ACCENT, ms=3,
            label="central difference",
        )
        bottom.set_ylabel("$v$ / Å fs$^{-1}$")
        bottom.set_xlabel("time $t$ / fs")
        bottom.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
        plt.show()

        wt = W * tau
        print(
            f"τ = {tau:.2f} fs: relative error 1 − sin(ωτ)/(ωτ) = "
            f"{1 - np.sin(wt) / wt:.4%}  "
            f"(leading term (ωτ)²/6 = {wt**2 / 6:.4%})"
        )


    widgets.interact(
        sampling,
        tau=widgets.FloatSlider(1.0, min=0.1, max=10.0, step=0.1, **SLOW),
    );
    """),
    md(r"""
    At $\tau = 1$ fs the points sit on the line
    (1.6% low). As $\tau$ grows every velocity shrinks by the same factor.
    At $\tau = 10$ fs, half a period, successive frames still alternate up
    and down, but the two frames each central difference uses are a whole
    period apart and equal, so the recovered velocity is zero everywhere.
    """),
]

EXERCISES = [
    md("""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises', at the
    end of the book, works every exercise in full.

    Run the collapsed answer-key cell below without opening it. After
    each exercise a collapsed cell holds a worked solution in code: try
    the exercise first, then open it to compare.
    """),
    hidden("""
    # The answer key. Each target is computed rather than typed in.
    _times = np.linspace(0.0, 3.0, 300001)
    _velocity = 3 * _times**2 - 12
    _low = np.degrees(0.5 * np.arcsin(10.0 * G / 12.0**2))
    _w_tau = brentq(lambda y: 1 - np.sin(y) / y - 0.01, 1e-3, 1.0)
    TARGETS = {
        "1.1": [False, True, True, False],
        "1.2 time": 15.0 / G,
        "1.2 height": 15.0**2 / (2 * G),
        "1.2 speed": 15.0,
        "1.3 rest": 2.0,
        "1.3 acceleration": 12.0,
        "1.3 displacement": np.trapezoid(_velocity, _times),
        "1.3 distance": np.trapezoid(np.abs(_velocity), _times),
        "1.4 degrees": np.degrees(1.1),
        "1.4 radians": 3 * np.pi / 4,
        "1.4 arc": 0.35 * 3 * np.pi / 4,
        "1.5": np.sqrt((1 - np.cos(np.pi / 4)) / 2),
        "1.6 amplitude": 0.2,
        "1.6 period": 2 * np.pi / 5,
        "1.6 frequency": 5 / (2 * np.pi),
        "1.6 speed": 0.2 * 5,
        "1.6 acceleration": 0.2 * 25,
        "1.6 crest": (np.pi / 2 - np.pi / 6) / 5,
        "1.7 low": _low,
        "1.7 high": 90.0 - _low,
        "1.8 acceleration": 0.02**2 / 1.0,
        "1.8 period": 2 * np.pi * 1.0 / 0.02,
        "1.9 estimate": 1 - 0.1**2 / 2,
        "1.9 error": 0.1**4 / 24,
        "1.10": _w_tau / (2 * np.pi / 20.0),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code("""
    # EXERCISE 1.1: which equations balance in dimension? True or False for
    # each of (i) x = v0 t², (ii) v² = v0² − 2 g x, (iii) t = sqrt(2x/g),
    # (iv) x = v0 exp(−g t).
    balances = None  # for example [True, False, True, True]

    check(balances, TARGETS["1.1"], name="dimensions")
    """),
    solution("""
    # SOLUTION 1.1. Write each dimension as powers of L and T, (L, T).
    length, time = np.array([1, 0]), np.array([0, 1])
    velocity = length - time  # L T⁻¹
    acceleration = length - 2 * time  # L T⁻²

    i = np.array_equal(length, velocity + 2 * time)  # v0 t² is L T
    ii = np.array_equal(2 * velocity, acceleration + length)  # g x
    iii = np.array_equal(time, (length - acceleration) / 2)  # sqrt(x/g)
    # (iv): the exponent g t must be a pure number, but it is L T⁻¹
    iv = np.array_equal(acceleration + time, np.zeros(2))
    check([i, ii, iii, iv], TARGETS["1.1"], name="dimensions")
    """),
    code("""
    # EXERCISE 1.2: thrown up at 15 m/s. Time to the top (s), greatest
    # height (m), speed on return to the hand (m/s).
    time_to_top = None
    height = None
    speed = None

    check(time_to_top, TARGETS["1.2 time"], name="time to the top")
    check(height, TARGETS["1.2 height"], name="greatest height")
    check(speed, TARGETS["1.2 speed"], name="speed on return")
    """),
    solution("""
    # SOLUTION 1.2
    v0 = 15.0
    time_to_top = v0 / G  # v = v0 − g t = 0 at the top
    height = v0**2 / (2 * G)  # v² = v0² − 2 g x with v = 0
    speed = np.sqrt(v0**2 - 2 * G * 0.0)  # back at x = 0, so |v| = v0

    check(time_to_top, TARGETS["1.2 time"], name="time to the top")
    check(height, TARGETS["1.2 height"], name="greatest height")
    check(speed, TARGETS["1.2 speed"], name="speed on return")
    """),
    code("""
    # EXERCISE 1.3: v(t) = 3t² − 12 m/s. Time at rest (s), acceleration
    # then (m/s²), displacement over 0–3 s (m), distance travelled (m).
    rest = None
    acceleration = None
    displacement = None
    distance = None

    check(rest, TARGETS["1.3 rest"], name="time at rest")
    check(acceleration, TARGETS["1.3 acceleration"], name="acceleration")
    check(displacement, TARGETS["1.3 displacement"], name="displacement")
    check(distance, TARGETS["1.3 distance"], name="distance")
    """),
    solution("""
    # SOLUTION 1.3. The function X(t) = t³ − 12t has derivative v(t).
    def X(t):
        return t**3 - 12 * t

    rest = np.sqrt(12 / 3)  # 3t² = 12, taking t ≥ 0
    acceleration = 6 * rest  # a = dv/dt = 6t
    displacement = X(3) - X(0)
    # v < 0 before t = 2 and v > 0 after: count both parts as positive
    distance = -(X(2) - X(0)) + (X(3) - X(2))

    check(rest, TARGETS["1.3 rest"], name="time at rest")
    check(acceleration, TARGETS["1.3 acceleration"], name="acceleration")
    check(displacement, TARGETS["1.3 displacement"], name="displacement")
    check(distance, TARGETS["1.3 distance"], name="distance")
    """),
    code("""
    # EXERCISE 1.4: 1.1 rad in degrees, 135° in radians, and the arc (m)
    # cut off by 135° on a wheel of radius 0.35 m.
    degrees = None
    radians = None
    arc = None

    check(degrees, TARGETS["1.4 degrees"], name="1.1 rad in degrees")
    check(radians, TARGETS["1.4 radians"], name="135° in radians")
    check(arc, TARGETS["1.4 arc"], name="arc length")
    """),
    solution("""
    # SOLUTION 1.4. A whole turn is 2π rad or 360°.
    degrees = 1.1 * 180 / np.pi
    radians = 135 * np.pi / 180
    arc = 0.35 * radians  # arc = R θ, with θ in radians

    check(degrees, TARGETS["1.4 degrees"], name="1.1 rad in degrees")
    check(radians, TARGETS["1.4 radians"], name="135° in radians")
    check(arc, TARGETS["1.4 arc"], name="arc length")
    """),
    code("""
    # EXERCISE 1.5: sin(π/8) to four decimal places, from
    # sin²θ = (1 − cos 2θ)/2.
    sin_pi_8 = None

    check(sin_pi_8, TARGETS["1.5"], atol=5e-5, name="sin(π/8)")
    """),
    solution("""
    # SOLUTION 1.5. SymPy confirms the identity; then θ = π/8.
    theta = sp.symbols("theta")
    identity = sp.cos(2 * theta) - (1 - 2 * sp.sin(theta) ** 2)
    print("cos 2θ − (1 − 2 sin²θ) =", sp.simplify(identity))

    exact = sp.sqrt((1 - sp.cos(sp.pi / 4)) / 2)  # positive root
    print("sin(π/8) =", sp.radsimp(exact))
    sin_pi_8 = round(float(exact), 4)
    check(sin_pi_8, TARGETS["1.5"], atol=5e-5, name="sin(π/8)")
    """),
    code("""
    # EXERCISE 1.6: x(t) = 0.2 sin(5t + π/6) m. Amplitude (m), period
    # (s), frequency (Hz), largest speed (m/s), largest acceleration
    # (m/s²), and the first time after t = 0 at which x is greatest (s).
    amplitude = None
    period = None
    frequency = None
    speed = None
    acceleration = None
    crest = None

    check(amplitude, TARGETS["1.6 amplitude"], name="amplitude")
    check(period, TARGETS["1.6 period"], name="period")
    check(frequency, TARGETS["1.6 frequency"], name="frequency")
    check(speed, TARGETS["1.6 speed"], name="largest speed")
    check(acceleration, TARGETS["1.6 acceleration"],
          name="largest acceleration")
    check(crest, TARGETS["1.6 crest"], name="first crest")
    """),
    solution("""
    # SOLUTION 1.6. Compare with A sin(ω t + φ).
    amplitude, omega, phase = 0.2, 5.0, np.pi / 6
    period = 2 * np.pi / omega
    frequency = 1 / period
    speed = amplitude * omega  # largest size of A ω cos(ω t + φ)
    acceleration = amplitude * omega**2  # largest size of −A ω² sin(...)
    crest = (np.pi / 2 - phase) / omega  # ω t + φ = π/2

    check(amplitude, TARGETS["1.6 amplitude"], name="amplitude")
    check(period, TARGETS["1.6 period"], name="period")
    check(frequency, TARGETS["1.6 frequency"], name="frequency")
    check(speed, TARGETS["1.6 speed"], name="largest speed")
    check(acceleration, TARGETS["1.6 acceleration"],
          name="largest acceleration")
    check(crest, TARGETS["1.6 crest"], name="first crest")
    """),
    code("""
    # EXERCISE 1.7: launch angles in degrees, smaller first, for a 10 m
    # range at 12 m/s.
    low = None
    high = None

    check(low, TARGETS["1.7 low"], name="lower angle")
    check(high, TARGETS["1.7 high"], name="higher angle")
    """),
    solution("""
    # SOLUTION 1.7. d = v0² sin 2α / g, so sin 2α = g d / v0².
    sine = G * 10.0 / 12.0**2
    two_alpha = np.arcsin(sine)  # one angle with this sine ...
    other = np.pi - two_alpha  # ... and the other, since sin(π − θ) = sin θ
    low, high = np.degrees(two_alpha / 2), np.degrees(other / 2)
    print(f"sin 2α = {sine:.4f}")

    check(low, TARGETS["1.7 low"], name="lower angle")
    check(high, TARGETS["1.7 high"], name="higher angle")
    """),
    code("""
    # EXERCISE 1.8: circle of radius 1.0 Å at 0.02 Å/fs. Acceleration
    # (Å/fs²) and period (fs).
    acceleration = None
    period = None

    check(acceleration, TARGETS["1.8 acceleration"], name="acceleration")
    check(period, TARGETS["1.8 period"], name="period")
    """),
    solution("""
    # SOLUTION 1.8
    radius, speed = 1.0, 0.02
    omega = speed / radius  # v = R ω
    acceleration = radius * omega**2  # = v² / R
    period = 2 * np.pi * radius / speed  # one turn at speed v

    # the same from mdlab's circular motion
    _, _, a = kinematics.circular_motion([0.0], radius, omega)
    print(f"mdlab: |a| = {np.linalg.norm(a):.1e} Å/fs²")
    check(acceleration, TARGETS["1.8 acceleration"], name="acceleration")
    check(period, TARGETS["1.8 period"], name="period")
    """),
    code("""
    # EXERCISE 1.9: cos 0.1 from 1 − y²/2, and the error estimate from the
    # first term left out.
    estimate = None
    error_estimate = None

    check(estimate, TARGETS["1.9 estimate"], rtol=1e-9, name="estimate")
    check(error_estimate, TARGETS["1.9 error"], name="error estimate")
    """),
    solution("""
    # SOLUTION 1.9. cos y = 1 − y²/2! + y⁴/4! − ...
    y = 0.1
    estimate = 1 - y**2 / math.factorial(2)
    error_estimate = y**4 / math.factorial(4)
    print(f"true error: {np.cos(y) - estimate:.4e}")

    check(estimate, TARGETS["1.9 estimate"], rtol=1e-9, name="estimate")
    check(error_estimate, TARGETS["1.9 error"], name="error estimate")
    """),
    code("""
    # EXERCISE 1.10: period 20 fs. The largest frame interval (fs) for
    # central-difference velocities within 1%.
    interval = None

    check(interval, TARGETS["1.10"], rtol=5e-3, name="interval")
    """),
    solution("""
    # SOLUTION 1.10. The first part: SymPy expands the forward difference
    # of A sin ωt; its leading error is (ωτ/2) times A ω at most.
    A, w, t, tau = sp.symbols("A omega t tau", positive=True)
    x = A * sp.sin(w * t)
    forward = (x.subs(t, t + tau) - x) / tau
    error = sp.series(forward - x.diff(t), tau, 0, 2).removeO()
    print("forward difference − v ≈", sp.simplify(error))

    # The second part: the relative error (ωτ)²/6 must not exceed 0.01.
    omega = 2 * np.pi / 20.0  # rad/fs
    interval = np.sqrt(6 * 0.01) / omega
    exact = brentq(lambda y: 1 - np.sin(y) / y - 0.01, 1e-3, 1.0) / omega
    print(f"leading term: {interval:.3f} fs; exact: {exact:.3f} fs")
    check(interval, TARGETS["1.10"], rtol=5e-3, name="interval")
    """),
]

CELLS = (
    SETUP
    + PYTHON
    + UNITS
    + VELOCITY
    + ACCELERATION
    + INTEGRATING
    + TRIG
    + PLANE
    + CIRCLE
    + TAYLOR
    + ATOMS
    + EXERCISES
)

if __name__ == "__main__":
    print("wrote", write(CELLS, "01_motion.ipynb"))
