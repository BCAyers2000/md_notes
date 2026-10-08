"""Write notebooks/02_newton.ipynb, the companion to Chapter 2.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_02_newton.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/02_newton.ipynb
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md("""
    # Notebook 02: forces and Newton's laws

    We now supply the forces that determine the motions of Chapter 1.
    The examples compare observers, masses and starting conditions so
    that each change has a physical interpretation.

    Run the cells in order before returning to a slider. Section numbers
    match Chapter 2; the worked solutions follow the exercises in
    collapsed cells, ready to compare with your own calculation.
    """),
    code("""
    %matplotlib inline
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from ase.data import atomic_masses, atomic_numbers
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation
    from scipy.constants import g as G  # 9.80665 m/s²

    from mdlab import dynamics, kinematics, units, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    """),
]

FORCE = [
    md("""
    ## 2.1 Force and the first law

    Forces add as vectors. Two forces act on a body, and the teal arrow
    is their sum, the net force. Change their sizes and directions.
    """),
    code("""
    def as_vector(size, angle_in_degrees):
        angle = np.radians(angle_in_degrees)
        return size * np.array([np.cos(angle), np.sin(angle)])


    def net_force(F1=4.0, angle1=0.0, F2=3.0, angle2=90.0):
        f1, f2 = as_vector(F1, angle1), as_vector(F2, angle2)
        total = f1 + f2

        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        arrows = ((f1, viz.OCHRE), (f2, viz.OXBLOOD), (total, viz.ACCENT))
        for vector, colour in arrows:
            ax.quiver(0, 0, *vector, color=colour, scale=1, **viz.ARROW)
        # the second force drawn again from the tip of the first: tip to tail
        ax.plot([f1[0], total[0]], [f1[1], total[1]], ":", color="black")
        ax.set_aspect("equal")
        limit = max(8.0, 1.15 * np.max(np.abs([f1, f2, total])))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("$F_x$ / units")
        ax.set_ylabel("$F_y$ / units")
        plt.show()

        size = np.linalg.norm(total)
        if size < 1e-12:
            print("net force is zero; it has no direction")
        else:
            direction = np.degrees(np.arctan2(total[1], total[0]))
            print(f"net force {size:.3f} units at {direction:.1f}°")


    widgets.interact(
        net_force,
        F1=widgets.FloatSlider(4.0, min=0.0, max=6.0, step=0.5),
        angle1=widgets.FloatSlider(0.0, min=-180, max=180, step=5),
        F2=widgets.FloatSlider(3.0, min=0.0, max=6.0, step=0.5),
        angle2=widgets.FloatSlider(90.0, min=-180, max=180, step=5),
    );
    """),
    md("""
    Pulls of 4 and 3 units at right angles add to
    5 units, not 7. Turn one against the other and the sum shrinks to 1;
    equal and opposite pulls leave no net force at all.
    """),
]

FRAMES = [
    md(r"""
    ## 2.2 Frames of reference

    A passenger in a train moving at $u$ throws a ball straight up at
    $w$. The same motion is drawn in the frame of reference of the train
    and from the platform; `dynamics.change_frame` converts between them
    with $\mathbf r' = \mathbf r - \mathbf u t$.
    """),
    code("""
    def two_views(u=5.0, w=6.0):
        flight = 2 * w / G
        t = np.linspace(0.0, flight, 200)
        # the throw as seen from the platform ...
        r, v = kinematics.constant_acceleration(
            t, np.zeros(2), [u, w], [0.0, -G]
        )
        # ... and the same throw seen in the train
        r_train, _ = dynamics.change_frame(t, r, v, frame_velocity=[u, 0.0])

        fig, (in_train, on_platform) = plt.subplots(
            1, 2, figsize=(viz.FULL, 2.4),
            gridspec_kw=dict(width_ratios=(1, 3)),
        )
        in_train.plot(*r_train.T, color="black")
        in_train.set_xlim(-1, 1)
        in_train.set_title("in the train", loc="left")
        on_platform.plot(*r.T, color="black")
        hand_times = t[::40]
        on_platform.plot(
            u * hand_times, 0 * hand_times, "s", color=viz.REFERENCE, ms=4
        )
        on_platform.set_title("from the platform", loc="left")
        for ax in (in_train, on_platform):
            ax.set_ylim(-0.3, 3.5)
            ax.set_xlabel("$x$ / m")
        in_train.set_ylabel("$y$ / m")
        plt.show()

        print(
            f"launch speed: in the train {w:.2f} m/s, "
            f"from the platform {np.hypot(u, w):.3f} m/s;  "
            f"lands {u * flight:.3f} m along the track after {flight:.3f} s"
        )


    widgets.interact(
        two_views,
        u=widgets.FloatSlider(5.0, min=0.0, max=15.0, step=0.5),
        w=widgets.FloatSlider(6.0, min=1.0, max=8.0, step=0.5),
    );
    """),
    md("""
    In the train the path never changes shape: it
    is always a vertical line. From the platform it is a parabola that
    stretches as $u$ grows, yet the ball always lands where the
    passenger's hand (grey squares) has got to, because the two frames of
    reference share the same vertical motion and the same acceleration.

    The same throw as an animation, both views side by side.
    """),
    code("""
    def animate_frames():
        u, w = 5.0, 6.0
        times = np.linspace(0.0, 2 * w / G, 50)
        r, v = kinematics.constant_acceleration(
            times, np.zeros(2), [u, w], [0.0, -G]
        )
        r_train, _ = dynamics.change_frame(times, r, v, frame_velocity=[u, 0.0])

        fig, (in_train, on_platform) = plt.subplots(
            1, 2, figsize=(viz.FULL * 0.9, 2.2), dpi=120,
            gridspec_kw=dict(width_ratios=(1, 3)),
        )
        in_train.plot(*r_train.T, color="black", lw=0.5, alpha=0.4)
        on_platform.plot(*r.T, color="black", lw=0.5, alpha=0.4)
        in_train.set_xlim(-1, 1)
        on_platform.set_xlim(-0.5, 6.8)
        for ax in (in_train, on_platform):
            ax.set_ylim(-0.3, 2.2)
        (ball_in_train,) = in_train.plot([], [], "o", color=viz.ACCENT, ms=6)
        (ball_seen,) = on_platform.plot([], [], "o", color=viz.ACCENT, ms=6)
        (hand,) = on_platform.plot([], [], "s", color=viz.REFERENCE, ms=6)


        def draw(k):
            ball_in_train.set_data([r_train[k, 0]], [r_train[k, 1]])
            ball_seen.set_data([r[k, 0]], [r[k, 1]])
            hand.set_data([u * times[k]], [0.0])
            return ball_in_train, ball_seen, hand


        animation = FuncAnimation(fig, draw, frames=len(times), interval=60)
        plt.close(fig)
        return HTML(animation.to_jshtml())


    animate_frames()
    """),
]

SECOND = [
    md(r"""
    ## 2.3 Mass and the second law

    One push of 10 N acts on trolleys of different mass, starting from
    rest. By the second law each accelerates at $a = F/m$, so its
    velocity grows as $v = (F/m)\,t$: the heavier the trolley, the more
    slowly it gathers speed.
    """),
    code("""
    push = 10.0  # N
    t = np.linspace(0.0, 4.0, 100)
    fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.4))
    for mass, colour in zip((5.0, 10.0, 20.0), viz.CYCLE, strict=False):
        ax.plot(t, push / mass * t, color=colour, label=f"{mass:g} kg")
    ax.set_xlabel("time $t$ / s")
    ax.set_ylabel("velocity $v$ / m s$^{-1}$")
    ax.legend(fontsize=8)
    plt.show()
    print(f"20 kg pushed by {push:g} N: a = {push / 20:.2f} m/s²")
    """),
]

THIRD = [
    md(r"""
    ## 2.4 The third law

    The Earth and a falling ball pull on each other with forces of the
    same size, the ball's weight. Dividing that one force by each mass
    gives two very different accelerations.
    """),
    code("""
    ball, earth = 0.5, 5.972e24  # kg
    weight = ball * G  # N, the size of both forces of the pair
    print(f"force on each: {weight:.2f} N")
    print(f"ball accelerates at  {weight / ball:.3f} m/s²")
    print(f"Earth accelerates at {weight / earth:.1e} m/s²")
    """),
]

GRAVITY = [
    md(r"""
    ## 2.5 Weight, free fall and free-body diagrams

    Newton's second law with the weight as the only force, solved by
    SymPy. Two integrations, with the launch velocity as the starting
    value, recover Chapter 1's $x = v_0 t - \tfrac12 g t^2$, and the
    mass cancels.
    """),
    code("""
    t, g, v0, m = sp.symbols("t g v_0 m", positive=True)
    x = sp.Function("x")
    newton = sp.Eq(m * x(t).diff(t, 2), -m * g)
    start = {x(0): 0, x(t).diff(t).subs(t, 0): v0}
    print(sp.dsolve(newton, x(t), ics=start))
    """),
]

STATE = [
    md(r"""
    ## 2.6 Equations of motion and the state

    The step-by-step picture of Section 2.6: from the state $(x, v)$,
    advance by $x \to x + v\tau$ and $v \to v + (F/m)\tau$, then repeat.
    For the thrown ball the steps (points) are compared with the exact
    motion (line). Chapter 9 turns this into a proper method; here it
    shows that the state is enough to say what happens next.
    """),
    code("""
    def stepping(tau=0.15):
        speed = 12.0
        flight = 2 * speed / G
        x, v = 0.0, speed
        times, heights = [0.0], [0.0]
        while times[-1] < flight:
            x, v = x + v * tau, v - G * tau  # one step from the state
            times.append(times[-1] + tau)
            heights.append(x)

        exact_t = np.linspace(0.0, flight, 200)
        exact_x = speed * exact_t - 0.5 * G * exact_t**2
        fig, ax = plt.subplots(figsize=(viz.HALF * 1.6, 2.4))
        ax.plot(exact_t, exact_x, color="black", lw=1.0, label="exact")
        ax.plot(times, heights, "o-", color=viz.ACCENT, ms=3, lw=0.8,
                label=f"steps of {tau} s")
        ax.set_xlabel("time $t$ / s")
        ax.set_ylabel("height $x$ / m")
        ax.set_ylim(min(-1, min(heights) - 0.5), max(9, max(heights) + 0.5))
        ax.legend(fontsize=8)
        plt.show()
        print(f"overshoot after one step: g τ²/2 = {0.5 * G * tau**2:.4f} m")


    widgets.interact(
        stepping,
        tau=widgets.FloatSlider(0.15, min=0.01, max=0.4, step=0.01, **SLOW),
    );
    """),
    md(r"""
    With long steps the points overshoot the true
    height, because each step uses the velocity at its start, before
    gravity has slowed the ball; as $\tau$ shrinks they close in on the
    exact curve. Chapter 9 explains why, and how to do much better.
    """),
]

DRAG = [
    md(r"""
    ## 2.7 Drag and terminal velocity

    A bead released from rest in a thick liquid obeys
    $\mathrm{d}v/\mathrm{d}t = g - \gamma v$ with $\gamma = b/m$. Change
    the mass and the drag coefficient.
    """),
    code("""
    def drag(mass=0.01, b=0.05):
        gamma = b / mass
        v_inf = G / gamma  # the terminal velocity
        t = np.linspace(0.0, 5.0 / gamma, 400)  # show five relaxation times
        depth, speed = dynamics.drag_motion(t, x0=0.0, v0=0.0, g=G,
                                            gamma=gamma)

        fig, (left, right) = plt.subplots(
            1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.45)
        )
        left.plot(t, speed, color=viz.ACCENT, label="with drag")
        left.axhline(v_inf, **viz.THRESHOLD_STYLE)
        left.plot(1 / gamma, (1 - np.exp(-1)) * v_inf, "o",
                  color=viz.ACCENT)
        left.set_ylim(0, 1.15 * v_inf)
        left.set_xlabel("$t$ / s")
        left.set_ylabel("$v$ / m s$^{-1}$")
        right.plot(t, depth, color=viz.ACCENT, label="with drag")
        right.plot(t, 0.5 * G * t**2, label="free fall", **viz.REFERENCE_STYLE)
        right.set_ylim(0, 1.05 * 0.5 * G * t[-1]**2)
        right.set_xlabel("$t$ / s")
        right.set_ylabel("$x$ / m")
        right.legend(loc="upper left", fontsize=8)
        plt.show()

        print(f"γ = {gamma:.2f} /s,  1/γ = {1 / gamma:.3f} s,  "
              f"v_∞ = {v_inf:.3f} m/s,  "
              f"90% reached at {np.log(10) / gamma:.3f} s")


    widgets.interact(
        drag,
        mass=widgets.FloatLogSlider(0.01, base=10, min=-3, max=-1,
                                    step=0.05),
        b=widgets.FloatLogSlider(0.05, base=10, min=-3, max=-0.5,
                                 step=0.05),
    );
    """),
    md(r"""
    The dot at $t = 1/\gamma$ always sits at 63% of
    the terminal velocity, whatever the mass and the drag coefficient. A
    heavier bead, or a thinner liquid, has a smaller $\gamma$: it takes
    longer to settle and settles at a higher speed. On the right, the
    dashed free-fall curve and the true depth agree at first and then
    part. Each plot spans five relaxation times, so read the time axis
    as well as the shape when changing the mass or drag coefficient.

    SymPy solves the drag equation directly; the result is the book's
    $v(t) = v_\infty(1 - e^{-\gamma t})$ with $v_\infty = g/\gamma$.
    """),
    code("""
    t, g, gamma = sp.symbols("t g gamma", positive=True)
    v = sp.Function("v")
    drag_law = sp.Eq(v(t).diff(t), g - gamma * v(t))
    print(sp.dsolve(drag_law, v(t), ics={v(0): 0}))
    """),
    md("""
    The exponential series of the toolbox, summed term by term, gives e.
    Each term is the one before divided by the next whole number.
    """),
    code("""
    term, total = 1.0, 1.0  # the first term of 1 + 1 + 1/2! + 1/3! + ...
    for n in range(1, 15):
        term /= n  # 1/n! from 1/(n − 1)!
        total += term
    print(f"1 + 1 + 1/2! + ... = {total:.12f};  e = {np.e:.12f}")
    """),
]

SPRING = [
    md(r"""
    ## 2.8 The spring

    A block on a spring, started from any state, moves as
    $x(t) = x_0\cos\omega t + (v_0/\omega)\sin\omega t$. The amplitude
    and phase describe the same motion,
    $A = \sqrt{x_0^2 + (v_0/\omega)^2}$ with $\cos\phi = x_0/A$ and
    $\sin\phi = -v_0/(\omega A)$. Change the mass, the stiffness and the
    starting state, including starts on the wall side ($x_0 < 0$).
    """),
    code("""
    def spring(mass=0.5, stiffness=50.0, x0_cm=3.0, v0=0.4):
        omega = np.sqrt(stiffness / mass)
        x0 = x0_cm / 100  # m
        amplitude = np.hypot(x0, v0 / omega)
        # the angle whose cosine and sine have the signs of x0 and −v0/ω,
        # in (−π, π]; arctan2 needs no division, so A = 0 is safe
        phase = np.arctan2(-v0 / omega, x0)

        t = np.linspace(0.0, 2.0, 600)
        x = x0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(viz.FULL, 3.0), sharex=True
        )
        top.plot(t, 100 * x, color=viz.ACCENT)
        top.axhline(0.0, color="black", lw=0.4)
        top.set_ylabel("$x$ / cm")
        bottom.plot(t, v, color=viz.ACCENT)
        bottom.axhline(0.0, color="black", lw=0.4)
        bottom.set_ylabel("$v$ / m s$^{-1}$")
        bottom.set_xlabel("time $t$ / s")
        plt.show()

        print(f"ω = {omega:.2f} rad/s, period {2 * np.pi / omega:.4f} s;  "
              f"A = {100 * amplitude:.2f} cm, φ = {phase:+.3f} rad;  "
              f"largest speed {amplitude * omega:.3f} m/s")


    widgets.interact(
        spring,
        mass=widgets.FloatSlider(0.5, min=0.1, max=2.0, step=0.1),
        stiffness=widgets.FloatSlider(50.0, min=10.0, max=200.0, step=10.0),
        x0_cm=widgets.FloatSlider(3.0, min=-5.0, max=5.0, step=0.5),
        v0=widgets.FloatSlider(0.4, min=-1.0, max=1.0, step=0.1),
    );
    """),
    md("""
    The period changes with the mass and the
    stiffness but never with the starting state. The defaults are the
    book's worked example: 5 cm amplitude and φ = −0.927 rad. Move $x_0$
    to −3 cm: the cosine of φ turns negative and the printed φ, −2.214
    rad, is the book's 4.069 rad less one full turn. Set both $x_0$ and
    $v_0$ to zero: the block stays at rest.

    SymPy confirms that the motion satisfies $m\\ddot x = -kx$, starts
    from $(x_0, v_0)$, and equals the single sinusoid.
    """),
    code("""
    t, m, k, A, phi = sp.symbols("t m k A phi", positive=True)
    x0, v0 = sp.symbols("x_0 v_0", real=True)
    omega = sp.sqrt(k / m)
    x = x0 * sp.cos(omega * t) + v0 / omega * sp.sin(omega * t)
    print("m x'' + k x =", sp.simplify(m * x.diff(t, 2) + k * x))
    print("x(0) =", x.subs(t, 0), ";  x'(0) =", x.diff(t).subs(t, 0))

    single = A * sp.cos(omega * t + phi)
    matched = x.subs({x0: A * sp.cos(phi), v0: -A * omega * sp.sin(phi)})
    difference = sp.simplify(sp.expand_trig(single) - matched)
    print("A cos(ωt + φ) − x(t) =", difference)
    """),
]

BODIES = [
    md("""
    ## 2.9 Several bodies: momentum and the centre of mass

    Two carts pushed apart by a compressed spring. Change their masses:
    the momenta stay equal and opposite, and the centre of mass (dashed)
    never moves.
    """),
    code("""
    def carts(m1=1.0, m2=3.0, stiffness=200.0):
        masses = np.array([m1, m2])
        natural_length, starting_gap = 0.30, 0.20  # m

        def spring_push(r):
            # the spring pushes only while compressed, then falls away
            gap = r[1, 0] - r[0, 0]
            push = stiffness * max(natural_length - gap, 0.0)
            return np.array([[-push], [+push]])

        t = np.linspace(0.0, 0.6, 601)
        r0 = np.array([[0.0], [starting_gap]])
        r, v = dynamics.solve_newton(
            spring_push, masses, r0, np.zeros_like(r0), t,
            force_to_accel=1.0,
        )
        centre = dynamics.centre_of_mass(masses, r)[:, 0]
        p = dynamics.momentum(masses, v)[:, :, 0]  # (time, cart)
        total = dynamics.total_momentum(masses, v)[:, 0]

        fig, (left, right) = plt.subplots(
            1, 2, figsize=(viz.FULL, 2.4), gridspec_kw=dict(wspace=0.45)
        )
        left.plot(t, r[:, 0, 0], color=viz.OCHRE, label="cart 1")
        left.plot(t, r[:, 1, 0], color=viz.ACCENT, label="cart 2")
        left.plot(t, centre, label="centre of mass", **viz.REFERENCE_STYLE)
        left.legend(fontsize=8)
        left.set_xlabel("$t$ / s")
        left.set_ylabel("$x$ / m")
        right.plot(t, p[:, 0], color=viz.OCHRE)
        right.plot(t, p[:, 1], color=viz.ACCENT)
        right.plot(t, total, label="total", **viz.REFERENCE_STYLE)
        right.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62))
        right.set_xlabel("$t$ / s")
        right.set_ylabel("momentum / kg m s$^{-1}$")
        plt.show()

        v1, v2 = v[-1, 0, 0], v[-1, 1, 0]
        print(f"final velocities {v1:+.3f} and {v2:+.3f} m/s, "
              f"ratio {v1 / v2:.3f} = −m2/m1 = {-m2 / m1:.3f};  "
              f"largest |P| = {np.abs(total).max():.1e}")


    widgets.interact(
        carts,
        m1=widgets.FloatSlider(1.0, min=0.5, max=5.0, step=0.5, **SLOW),
        m2=widgets.FloatSlider(3.0, min=0.5, max=5.0, step=0.5, **SLOW),
        stiffness=widgets.FloatSlider(200.0, min=50.0, max=500.0,
                                      step=50.0, **SLOW),
    );
    """),
    md("""
    Make the carts equal and they move apart at
    equal speeds; make one much heavier and it barely moves while the
    light one shoots off. In every case the dashed total momentum stays at
    zero and the centre of mass stays where it began.
    """),
]

ATOMS = [
    md(r"""
    ## 2.10 From balls to atoms

    Newton's law in eV, Å, fs and amu needs one factor, computed from the
    CODATA constants.
    """),
    code("""
    print(f"1 eV/(Å amu) = {units.FORCE_TO_ACCEL:.6e} Å/fs²")

    lithium = atomic_masses[atomic_numbers["Li"]]  # amu
    a = dynamics.acceleration([[1.0, 0.0, 0.0]], [lithium])[0, 0]
    print(f"Li under 1 eV/Å: a = {a:.4e} Å/fs²;  "
          f"after 10 fs, v = {a * 10:.4f} Å/fs")

    weight = lithium * units.AMU * G  # N
    newtons_per_ev_per_angstrom = units.ELEMENTARY_CHARGE / units.ANGSTROM
    print(f"its weight: {weight / newtons_per_ev_per_angstrom:.1e} eV/Å")
    """),
    md(r"""
    The shape trap of Section 2.10. With three atoms the $3\times3$ forces
    divided by the three masses run without complaint, but divide columns;
    the $3\times1$ column of masses divides rows, as Newton's law needs.
    """),
    code("""
    forces = np.ones((3, 3))  # eV/Å, the same force on every atom
    masses = np.array([1.0, 2.0, 4.0])  # amu
    print("forces / masses (wrong, by column):")
    print(forces / masses)
    print("forces / masses[:, np.newaxis] (right, by row):")
    print(forces / masses[:, np.newaxis])
    a = dynamics.acceleration(forces, masses) / units.FORCE_TO_ACCEL
    assert np.allclose(a, forces / masses[:, np.newaxis])
    """),
    md(r"""
    Removing the drift of the centre of mass. Ten atoms with random
    velocities carry a total momentum $\mathbf P$; subtracting
    $\mathbf P/M$ from every velocity, a change of frame of reference,
    removes it.
    """),
    code("""
    rng = np.random.default_rng(2026)
    masses = rng.choice([1.008, 6.94, 12.011, 15.999], size=10)  # amu
    velocities = rng.normal(scale=0.01, size=(10, 3))  # Å/fs

    P = dynamics.total_momentum(masses, velocities)
    print("P before:", P.round(5), "amu Å/fs")

    velocities = velocities - P / masses.sum()
    print("P after: ", dynamics.total_momentum(masses, velocities).round(12))
    """),
]

EXERCISES = [
    md("""
    ## Exercises

    Each exercise cell sets its answers to `None`. Replace `None` with
    your result, in the units stated, and run the cell; the check says
    whether it is right. The book's 'Solutions to the exercises', at the
    end of the book, works every exercise in full, including 2.1 and 2.5,
    which are explanations to write on paper.

    Run the collapsed answer-key cell below without opening it. After
    each exercise a collapsed cell holds a worked solution in code: try
    the exercise first, then open it to compare. The derivations 2.9,
    2.12, 2.15 and 2.19 have their solutions checked by SymPy.
    """),
    hidden("""
    # The answer key. Each target is computed rather than typed in.
    _oxygen = dynamics.acceleration([[0.5, 0.0, 0.0]], [15.999])[0, 0]
    _, _bead = dynamics.drag_motion(0.2, x0=0.0, v0=4.0, g=G, gamma=5.0)
    _stiff = np.sqrt(80 / 0.2)
    TARGETS = {
        "2.2 forwards": 10.0,
        "2.2 backwards": 2.0,
        "2.3 force": 5.0,
        "2.3 angle": np.degrees(np.arctan2(3, 4)),
        "2.3 acceleration": 5.0 / 3.0,
        "2.4 mass": 2.0 * 0.6 / 1.5,
        "2.4 force": 2.0 * 0.6,
        "2.6 tension": 2.0 * (G + 1.5),
        "2.7 mass": 70.0,
        "2.7 weight": 70.0 * 1.62,
        "2.7 earth weight": 70.0 * G,
        "2.8 time": (-2.0 + np.sqrt(2.0**2 + 2 * G * 3.0)) / G,
        "2.10 velocity": _bead,
        "2.11 time": np.log(100) / 5.0,
        "2.13 omega": _stiff,
        "2.13 period": 2 * np.pi / _stiff,
        "2.13 speed": 0.03 * _stiff,
        "2.13 acceleration": 0.03 * _stiff**2,
        "2.14 amplitude": 0.5 / 10.0,
        "2.14 phase": -np.pi / 2,
        "2.16 velocity": 2.0 * 3.0 / (2.0 + 1.0),
        "2.17 boat": -70.0 * 3.0 / (70.0 + 140.0),
        "2.18 acceleration": _oxygen,
        "2.18 velocity": 5 * _oxygen,
        "2.18 velocity m/s": 5 * _oxygen * 1e5,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code("""
    # EXERCISE 2.2: speeds of the ball relative to the ground (m/s),
    # thrown forwards and then backwards.
    forwards = None
    backwards = None

    check(forwards, TARGETS["2.2 forwards"], name="forwards")
    check(backwards, TARGETS["2.2 backwards"], name="backwards")
    """),
    solution("""
    # SOLUTION 2.2. v = v' + u, forwards positive.
    u = 6.0  # the bicycle, m/s
    forwards = +4.0 + u
    backwards = -4.0 + u  # still forwards, more slowly than the bicycle

    check(forwards, TARGETS["2.2 forwards"], name="forwards")
    check(backwards, TARGETS["2.2 backwards"], name="backwards")
    """),
    code("""
    # EXERCISE 2.3: net force (N), its angle north of east (degrees) and
    # the acceleration (m/s²).
    force = None
    angle = None
    acceleration = None

    check(force, TARGETS["2.3 force"], name="net force")
    check(angle, TARGETS["2.3 angle"], name="angle")
    check(acceleration, TARGETS["2.3 acceleration"], name="acceleration")
    """),
    solution("""
    # SOLUTION 2.3. East is x, north is y.
    net = np.array([4.0, 0.0]) + np.array([0.0, 3.0])  # N
    force = np.linalg.norm(net)  # (4² + 3²)^(1/2)
    angle = np.degrees(np.arctan(net[1] / net[0]))  # tan θ = 3/4
    acceleration = force / 3.0  # a = F/m, along the force

    check(force, TARGETS["2.3 force"], name="net force")
    check(angle, TARGETS["2.3 angle"], name="angle")
    check(acceleration, TARGETS["2.3 acceleration"], name="acceleration")
    """),
    code("""
    # EXERCISE 2.4: mass of the second trolley (kg) and force of the
    # spring (N).
    mass = None
    force = None

    check(mass, TARGETS["2.4 mass"], name="mass")
    check(force, TARGETS["2.4 force"], name="force")
    """),
    solution("""
    # SOLUTION 2.4. One force: accelerations in inverse ratio of masses.
    m1, a1, a2 = 2.0, 0.6, 1.5
    mass = m1 * a1 / a2
    force = m1 * a1  # F = m a, the same from either trolley
    print(f"check: m2 a2 = {mass * a2:.2f} N")

    check(mass, TARGETS["2.4 mass"], name="mass")
    check(force, TARGETS["2.4 force"], name="force")
    """),
    code("""
    # EXERCISE 2.6: tension in the rope (N).
    tension = None

    check(tension, TARGETS["2.6 tension"], name="tension")
    """),
    solution("""
    # SOLUTION 2.6. Upwards positive: F_rope − m g = m a.
    m, a = 2.0, 1.5
    tension = m * (G + a)
    print(f"holding the weight: {m * G:.2f} N; accelerating: {m * a:.2f} N")

    check(tension, TARGETS["2.6 tension"], name="tension")
    """),
    code("""
    # EXERCISE 2.7: the astronaut's mass on the Moon (kg), weight on the
    # Moon (N) and weight on the Earth (N).
    mass = None
    weight_moon = None
    weight_earth = None

    check(mass, TARGETS["2.7 mass"], name="mass")
    check(weight_moon, TARGETS["2.7 weight"], name="weight on the Moon")
    check(weight_earth, TARGETS["2.7 earth weight"],
          name="weight on the Earth")
    """),
    solution("""
    # SOLUTION 2.7. Mass is the same everywhere; weight is m g.
    mass = 70.0
    weight_moon = mass * 1.62
    weight_earth = mass * G
    print(f"ratio: {weight_moon / weight_earth:.3f}, about a sixth")

    check(mass, TARGETS["2.7 mass"], name="mass")
    check(weight_moon, TARGETS["2.7 weight"], name="weight on the Moon")
    check(weight_earth, TARGETS["2.7 earth weight"],
          name="weight on the Earth")
    """),
    code("""
    # EXERCISE 2.8: time (s) to reach the ground from 3 m, moving down at
    # 2 m/s.
    time = None

    check(time, TARGETS["2.8 time"], name="time to the ground")
    """),
    solution("""
    # SOLUTION 2.8. x(t) = 3 − 2t − g t²/2 = 0, or (g/2) t² + 2t − 3 = 0.
    A, B, C = G / 2, 2.0, -3.0
    roots = np.roots([A, B, C])
    print("both roots:", roots.round(4))
    time = roots[roots > 0][0]  # the root after the start

    check(time, TARGETS["2.8 time"], name="time to the ground")
    """),
    md("""
    **Exercise 2.9** is a derivation. The collapsed cell checks the book's
    solution with SymPy.
    """),
    solution("""
    # SOLUTION 2.9. One step of τ, then two steps of τ/2, from x = 0, v = v0.
    v0, g, tau = sp.symbols("v_0 g tau", positive=True)
    exact = v0 * tau - g * tau**2 / 2

    x_one = 0 + v0 * tau
    x1, v1 = 0 + v0 * tau / 2, v0 - g * tau / 2
    x_two = x1 + v1 * tau / 2
    print("one step overshoots by  ", sp.simplify(x_one - exact))
    print("two half steps overshoot", sp.simplify(x_two - exact))
    assert sp.simplify(x_one - exact - g * tau**2 / 2) == 0
    assert sp.simplify(x_two - exact - g * tau**2 / 4) == 0
    """),
    code("""
    # EXERCISE 2.10: velocity (m/s) 0.2 s after being fired down at 4 m/s.
    velocity = None

    check(velocity, TARGETS["2.10 velocity"], name="velocity")
    """),
    solution("""
    # SOLUTION 2.10. The gap v − v∞ shrinks by e^(−γt).
    gamma, v0, t = 5.0, 4.0, 0.2
    v_inf = G / gamma
    velocity = v_inf + (v0 - v_inf) * np.exp(-gamma * t)
    print(f"v∞ = {v_inf:.3f} m/s; starting gap {v0 - v_inf:.3f} m/s")

    check(velocity, TARGETS["2.10 velocity"], name="velocity")
    """),
    code("""
    # EXERCISE 2.11: time (s) to reach 99% of the terminal velocity.
    time = None

    check(time, TARGETS["2.11 time"], name="time to 99%")
    """),
    solution("""
    # SOLUTION 2.11. 1 − e^(−γt) = 0.99, so e^(−γt) = 0.01.
    gamma = 5.0
    time = -np.log(0.01) / gamma

    check(time, TARGETS["2.11 time"], name="time to 99%")
    """),
    md("""
    **Exercise 2.12** is a derivation. The collapsed cell checks it.
    """),
    solution("""
    # SOLUTION 2.12. Differentiate x(t) and substitute into the equation.
    t, g, gamma = sp.symbols("t g gamma", positive=True)
    v_inf = g / gamma
    x = v_inf * t - v_inf / gamma * (1 - sp.exp(-gamma * t))
    velocity = x.diff(t)
    print("dx/dt =", sp.simplify(velocity))
    residual = sp.simplify(x.diff(t, 2) - (g - gamma * velocity))
    print("x'' − (g − γ x') =", residual)
    assert residual == 0
    assert sp.simplify(velocity - v_inf * (1 - sp.exp(-gamma * t))) == 0
    """),
    code("""
    # EXERCISE 2.13: angular frequency (rad/s), period (s), largest speed
    # (m/s) and largest acceleration (m/s²).
    omega = None
    period = None
    speed = None
    acceleration = None

    check(omega, TARGETS["2.13 omega"], name="omega")
    check(period, TARGETS["2.13 period"], name="period")
    check(speed, TARGETS["2.13 speed"], name="largest speed")
    check(acceleration, TARGETS["2.13 acceleration"],
          name="largest acceleration")
    """),
    solution("""
    # SOLUTION 2.13. Released from rest, so the amplitude is the pull.
    m, k, amplitude = 0.2, 80.0, 0.03
    omega = np.sqrt(k / m)
    period = 2 * np.pi / omega
    speed = amplitude * omega  # at the centre
    acceleration = amplitude * omega**2  # at the turning points

    check(omega, TARGETS["2.13 omega"], name="omega")
    check(period, TARGETS["2.13 period"], name="period")
    check(speed, TARGETS["2.13 speed"], name="largest speed")
    check(acceleration, TARGETS["2.13 acceleration"],
          name="largest acceleration")
    """),
    code("""
    # EXERCISE 2.14: amplitude (m) and phase (rad, between −π and π).
    amplitude = None
    phase = None

    check(amplitude, TARGETS["2.14 amplitude"], name="amplitude")
    check(phase, TARGETS["2.14 phase"], name="phase")
    """),
    solution("""
    # SOLUTION 2.14. A from x0 and v0; φ from its cosine and sine.
    omega, x0, v0 = 10.0, 0.0, 0.5
    amplitude = np.hypot(x0, v0 / omega)
    cos_phi, sin_phi = x0 / amplitude, -v0 / (omega * amplitude)
    phase = np.arctan2(sin_phi, cos_phi)  # the angle with both
    print(f"cos φ = {cos_phi:.1f}, sin φ = {sin_phi:.1f}")
    t = np.linspace(0.0, 1.0, 11)
    same = np.allclose(amplitude * np.cos(omega * t + phase),
                       amplitude * np.sin(omega * t))
    print("A cos(ωt + φ) equals A sin ωt:", same)

    check(amplitude, TARGETS["2.14 amplitude"], name="amplitude")
    check(phase, TARGETS["2.14 phase"], name="phase")
    """),
    md("""
    **Exercise 2.15** is a derivation. The collapsed cell checks it.
    """),
    solution("""
    # SOLUTION 2.15. x is measured downwards from the hanging rest.
    m, g, k, x = sp.symbols("m g k x", positive=True)
    stretch = sp.solve(sp.Eq(k * sp.Symbol("s"), m * g), sp.Symbol("s"))[0]
    print("resting stretch s =", stretch)
    net_force = m * g - k * (stretch + x)  # downwards positive
    print("net force at x:", sp.simplify(net_force))
    assert sp.simplify(net_force + k * x) == 0  # Hooke's law about the rest
    """),
    code("""
    # EXERCISE 2.16: common velocity (m/s) after the carts couple.
    velocity = None

    check(velocity, TARGETS["2.16 velocity"], name="common velocity")
    """),
    solution("""
    # SOLUTION 2.16. Momentum before = momentum after.
    masses = np.array([2.0, 1.0])
    before = np.array([[3.0], [0.0]])  # velocities, m/s
    P = dynamics.total_momentum(masses, before)[0]
    velocity = P / masses.sum()

    check(velocity, TARGETS["2.16 velocity"], name="common velocity")
    """),
    code("""
    # EXERCISE 2.17: how far the boat moves (m), positive in the direction
    # the person walks.
    boat = None

    check(boat, TARGETS["2.17 boat"], name="boat")
    """),
    solution("""
    # SOLUTION 2.17. The centre of mass stays put:
    # 70 Δx_p + 140 Δx_b = 0 with Δx_p = Δx_b + 3.
    db = sp.symbols("Delta_x_b")
    boat = float(sp.solve(sp.Eq(70 * (db + 3) + 140 * db, 0), db)[0])
    print(f"boat {boat:+.1f} m, person {boat + 3:+.1f} m")

    check(boat, TARGETS["2.17 boat"], name="boat")
    """),
    code("""
    # EXERCISE 2.18: acceleration (Å/fs²), then the velocity after 5 fs in
    # Å/fs and in m/s.
    acceleration = None
    velocity = None
    velocity_si = None

    check(acceleration, TARGETS["2.18 acceleration"], name="acceleration")
    check(velocity, TARGETS["2.18 velocity"], name="velocity in Å/fs")
    check(velocity_si, TARGETS["2.18 velocity m/s"], name="velocity in m/s")
    """),
    solution("""
    # SOLUTION 2.18. a = F/m times the unit factor; v = a t from rest.
    acceleration = 0.5 / 15.999 * units.FORCE_TO_ACCEL
    velocity = acceleration * 5.0
    velocity_si = velocity * 1e5  # 1 Å/fs = 1e-10 m / 1e-15 s

    check(acceleration, TARGETS["2.18 acceleration"], name="acceleration")
    check(velocity, TARGETS["2.18 velocity"], name="velocity in Å/fs")
    check(velocity_si, TARGETS["2.18 velocity m/s"], name="velocity in m/s")
    """),
    md("""
    **Exercise 2.19** is a derivation. The collapsed cell checks it for
    three atoms with symbolic masses and velocities.
    """),
    solution("""
    # SOLUTION 2.19. Subtract P/M from every velocity.
    m = sp.symbols("m1:4", positive=True)
    v = sp.symbols("v1:4")  # one component; the others work the same way
    M = sum(m)
    P = sum(mi * vi for mi, vi in zip(m, v, strict=True))
    new_P = sum(mi * (vi - P / M) for mi, vi in zip(m, v, strict=True))
    print("new total momentum:", sp.simplify(new_P))
    assert sp.simplify(new_P) == 0
    """),
]

CELLS = (
    SETUP
    + FORCE
    + FRAMES
    + SECOND
    + THIRD
    + GRAVITY
    + STATE
    + DRAG
    + SPRING
    + BODIES
    + ATOMS
    + EXERCISES
)

if __name__ == "__main__":
    print("wrote", write(CELLS, "02_newton.ipynb"))
