"""Write notebooks/05_rotation.ipynb, the companion to Chapter 5.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_05_rotation.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/05_rotation.ipynb
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md("""
    # Notebook 05: rotation and angular momentum

    Working through Chapter 5: how angular momentum is conserved, how
    mass distribution affects rotation, and how to separate a molecule's
    rigid motion from its vibration. Run the cells in order; later
    sections reuse the geometry and calculations developed above them.
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
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.constants import g as G  # 9.80665 m/s²

    from mdlab import dynamics, energy, oscillators, rotation, units, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    rng = np.random.default_rng(5)
    """),
]

PLANE = [
    md(r"""
    ## 5.1 Turning in a plane

    First the algebra: with $x = r\cos\theta$ and $y = r\sin\theta$, both
    changing with time, $x\dot y - y\dot x = r^2\dot\theta$.
    """),
    code("""
    t = sp.symbols("t")
    r, th = sp.Function("r")(t), sp.Function("theta")(t)
    x, y = r * sp.cos(th), r * sp.sin(th)
    combination = sp.simplify(x * y.diff(t) - y * x.diff(t))
    print("x y' - y x' =", combination)
    assert sp.simplify(combination - r**2 * th.diff(t)) == 0
    """),
    md(r"""
    The puck on its string: a puck of 0.2 kg circling at 0.5 m and
    1.2 m/s, so $L$ = 0.12 kg m²/s. Move the radius slider to pull the
    string in or let it out; $L$ stays fixed, and the speed and the
    kinetic energy follow.
    """),
    code("""
    MASS, R1, V1 = 0.2, 0.5, 1.2
    L_PUCK = MASS * R1 * V1


    def puck(radius=0.5):
        speed = L_PUCK / (MASS * radius)
        kinetic = 0.5 * MASS * speed**2
        fig, (top, bars) = plt.subplots(
            1, 2, figsize=(viz.FULL, 2.5),
            gridspec_kw=dict(width_ratios=(1.0, 1.3), wspace=0.4),
        )
        s = np.linspace(0, 2 * np.pi, 200)
        top.plot(radius * np.cos(s), radius * np.sin(s), color=viz.REFERENCE)
        top.plot(0, 0, "o", color="black", ms=3)
        top.plot(radius, 0, "o", color=viz.ACCENT, ms=8)
        top.quiver(radius, 0, 0, speed, color=viz.ACCENT, scale=6,
                   scale_units="xy", angles="xy")
        top.set_aspect("equal")
        top.set_xlim(-1.1, 1.1)
        top.set_ylim(-1.1, 1.1)
        top.set_xlabel("$x$ / m")
        top.set_ylabel("$y$ / m")
        ratios = [1.0, speed / V1, kinetic / (0.5 * MASS * V1**2)]
        bars.bar(["$L$", "speed", "$K$"], ratios,
                 color=[viz.ACCENT, "black", viz.OCHRE])
        bars.axhline(1, **viz.THRESHOLD_STYLE)
        bars.set_ylabel("ratio to the starting value")
        bars.set_ylim(0, max(4.5, 1.15 * max(ratios)))
        plt.show()
        print(f"r = {radius:.2f} m: speed {speed:.2f} m/s, K = "
              f"{kinetic:.3f} J; the hand has done {kinetic - 0.144:+.3f} J")


    widgets.interact(
        puck, radius=widgets.FloatSlider(0.5, min=0.1, max=1.0, step=0.05),
    );
    """),
    md("""
    Halving the radius doubles the speed and
    quadruples the kinetic energy, while $L$ never moves. Letting the
    string out makes the work of the hand negative.
    """),
]

CROSS = [
    md(r"""
    ## 5.2 Angular momentum in space

    The cross product in NumPy, the example of the toolbox, and the
    properties checked for symbols.
    """),
    code("""
    a, b = np.array([1.0, 2.0, 0.0]), np.array([3.0, -1.0, 2.0])
    c = np.cross(a, b)
    print("a x b =", c, "  a.(a x b) =", a @ c, "  b.(a x b) =", b @ c)
    lagrange_value = math.sqrt((a @ a) * (b @ b) - (a @ b) ** 2)
    print("|a x b| =", np.linalg.norm(c), " sqrt(|a|²|b|² - (a.b)²) =",
          lagrange_value)

    A = sp.Matrix(sp.symbols("a_x a_y a_z"))
    B = sp.Matrix(sp.symbols("b_x b_y b_z"))
    AxB = A.cross(B)
    assert sp.simplify(A.dot(AxB)) == 0 and sp.simplify(B.dot(AxB)) == 0
    assert sp.simplify(B.cross(A) + AxB) == sp.zeros(3, 1)
    lagrange = AxB.dot(AxB) - (A.dot(A) * B.dot(B) - A.dot(B) ** 2)
    assert sp.expand(lagrange) == 0
    print("perpendicular, antisymmetric, Lagrange's identity: checked")
    """),
    md(r"""
    Turn $\mathbf b$ in the $xy$ plane and watch $\mathbf a\times\mathbf b$,
    with $\mathbf a$ fixed along $x$: its length is the shaded area, and it
    flips from $+z$ to $-z$ as $\mathbf b$ passes $\mathbf a$.
    """),
    code("""
    def cross_explorer(angle=60.0, length=1.0):
        a = np.array([1.0, 0.0, 0.0])
        theta = math.radians(angle)
        b = length * np.array([math.cos(theta), math.sin(theta), 0.0])
        c = np.cross(a, b)
        fig = plt.figure(figsize=(3.6, 3.2), dpi=120)
        ax = fig.add_subplot(projection="3d")
        for vec, colour, name in ((a, "black", "a"), (b, "black", "b"),
                                  (c, viz.ACCENT, "a x b")):
            ax.quiver(0, 0, 0, *vec, color=colour)
            ax.text(*(1.1 * vec), name)
        corners = np.array([[0, 0, 0], a, a + b, b])
        # A polygon also handles the zero-area cases at 0° and 180°.
        ax.add_collection3d(Poly3DCollection([corners], facecolor="0.85",
                                            alpha=0.6))
        ax.set_xlim(-1.7, 2.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_zlim(-1.7, 1.7)
        ax.set_box_aspect((4.4, 3.4, 3.4))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        plt.show()
        print(f"a x b = {np.round(c, 3)}, length {np.linalg.norm(c):.3f} = "
              f"|a||b| sin(angle) = {length * abs(math.sin(theta)):.3f}")


    widgets.interact(
        cross_explorer,
        angle=widgets.FloatSlider(60.0, min=0.0, max=360.0, step=5.0, **SLOW),
        length=widgets.FloatSlider(1.0, min=0.2, max=1.5, step=0.1, **SLOW),
    );
    """),
    md("""
    The arrow $\\mathbf a\\times\\mathbf b$ is
    longest when $\\mathbf b$ is at right angles to $\\mathbf a$, vanishes
    at 0° and 180°, and points down for angles past 180°.
    """),
]

SYSTEM = [
    md(r"""
    ## 5.3 The angular momentum of many bodies

    Five atoms joined by a pair potential, moved for a while with
    `solve_newton`: the internal forces are along the lines joining the
    atoms, so the total angular momentum about any fixed point stays put.
    The split $\mathbf L = \mathbf R\times\mathbf P + \mathbf L'$ is checked
    at the end.
    """),
    code("""
    masses = np.array([6.94, 12.011, 15.999, 1.008, 18.998])  # amu
    r0 = rng.normal(scale=1.5, size=(5, 3))
    v0 = rng.normal(scale=0.01, size=(5, 3))


    def pair_forces(r):
        forces = np.zeros_like(r)
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                dist = np.linalg.norm(d)
                # a soft spring towards 2 Å: phi'(r) = 0.5 (r - 2)
                f = 0.5 * (dist - 2.0) * d / dist
                forces[i] += f
                forces[j] -= f
        return forces


    times = np.linspace(0.0, 500.0, 6)  # fs
    r, v = dynamics.solve_newton(pair_forces, masses, r0, v0, times,
                                 force_to_accel=units.FORCE_TO_ACCEL)
    origin = np.array([3.0, -1.0, 2.0])
    for k, time in enumerate(times):
        ang = rotation.angular_momentum(masses, r[k], v[k], origin)
        print(f"t = {time:5.0f} fs: L = {np.round(ang, 8)} amu Å²/fs")
    centre = dynamics.centre_of_mass(masses, r[-1])
    total_p = dynamics.total_momentum(masses, v[-1])
    split = np.cross(centre, total_p) + rotation.angular_momentum(
        masses, r[-1], v[-1] - total_p / masses.sum(), centre)
    assert np.allclose(split, rotation.angular_momentum(masses, r[-1], v[-1]))
    print("L = R x P + L' checked")
    """),
]

RIGID = [
    md(r"""
    ## 5.4 Rigid rotation

    The disc's moment of inertia by integration over rings, and the
    race down a slope of 30°: a block sliding without friction, a disc
    and a hoop rolling.
    """),
    code("""
    rho, R, M = sp.symbols("rho R M", positive=True)
    disc = sp.integrate(rho**2 * 2 * M * rho / R**2, (rho, 0, R))
    print("disc: I =", disc)
    assert sp.simplify(disc - M * R**2 / 2) == 0

    SLOPE, LENGTH = math.radians(30.0), 2.0
    BODIES = (("block", 0.0, viz.REFERENCE), ("disc", 0.5, viz.ACCENT),
              ("hoop", 1.0, viz.OCHRE))
    times = np.linspace(0.0, 1.35, 60)
    fig, ax = plt.subplots(figsize=(4.5, 2.6), dpi=120)
    ax.plot([0, LENGTH * math.cos(SLOPE)], [LENGTH * math.sin(SLOPE), 0],
            color="black", lw=1)
    markers = [ax.plot([], [], "o", color=c, ms=9, label=n)[0]
               for n, _, c in BODIES]
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.9)
    ax.set_ylim(-0.1, 1.2)
    ax.legend(loc="upper right")


    def draw(k):
        for marker, (_, kappa, _) in zip(markers, BODIES, strict=True):
            accel = G * math.sin(SLOPE) / (1 + kappa)
            s = min(0.5 * accel * times[k] ** 2, LENGTH)
            marker.set_data([s * math.cos(SLOPE)],
                            [(LENGTH - s) * math.sin(SLOPE) + 0.05])
        return markers


    animation = FuncAnimation(fig, draw, frames=len(times), interval=60)
    plt.close(fig)
    HTML(animation.to_jshtml())
    """),
    md("""
    The block arrives first, followed by the disc and then the hoop,
    whatever their masses and sizes: the more of the mass lies far from
    the axis, the more of the energy goes into turning.
    """),
]

TENSOR = [
    md(r"""
    ## 5.5 The inertia tensor

    The two triple-product rules checked for symbols, then the tensor of
    water at its measured geometry (O-H 0.958 Å, 104.4776°), one bond along
    $x$, and its principal axes from `eigh`.
    """),
    code("""
    C = sp.Matrix(sp.symbols("c_x c_y c_z"))
    assert sp.simplify(A.cross(B.cross(C)) - (B * A.dot(C) - C * A.dot(B))) \\
        == sp.zeros(3, 1)
    assert sp.expand(A.dot(B.cross(C)) - B.dot(C.cross(A))) == 0
    print("a x (b x c) = b(a.c) - c(a.b) and the cyclic rule: checked")

    R_OH, ANGLE = 0.958, math.radians(104.4776)
    water = np.array([[0.0, 0.0, 0.0], [R_OH, 0.0, 0.0],
                      [R_OH * math.cos(ANGLE), R_OH * math.sin(ANGLE), 0.0]])
    water_masses = np.array([15.999, 1.008, 1.008])
    centre = dynamics.centre_of_mass(water_masses, water)
    tensor = rotation.inertia_tensor(water_masses, water, centre)
    moments, axes = np.linalg.eigh(tensor)
    print("tensor about the centre of mass (amu Å²):")
    print(tensor.round(4))
    print("principal moments:", moments.round(4))
    print("axes (columns):")
    print(axes.round(4))
    axis_angle = math.degrees(math.atan2(axes[1, 1], axes[0, 1]))
    print(f"axis of I_2 at {axis_angle:.2f}° to x; half the bond angle is "
          f"{math.degrees(ANGLE / 2):.2f}°")
    """),
    md(r"""
    $\mathbf L = \mathsf I\boldsymbol\omega$ is parallel to
    $\boldsymbol\omega$ only along a principal axis. Turn
    $\boldsymbol\omega$ in the plane of the molecule and compare their
    directions. The two quantities have different units, so the arrow
    lengths should not be compared.
    """),
    code("""
    def tilt(angle=20.0):
        omega = np.array([math.cos(math.radians(angle)),
                          math.sin(math.radians(angle)), 0.0])
        ang = tensor @ omega
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)
        ax.quiver(0, 0, *omega[:2], color="black", scale=1, angles="xy",
                  scale_units="xy", label="omega")
        ax.quiver(0, 0, *ang[:2], color=viz.ACCENT, scale=1, angles="xy",
                  scale_units="xy", label="L")
        for k in range(2):
            ax.plot(*np.outer([-1.3, 1.3], axes[:2, k]).T, ":", color="0.6")
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.legend(loc="lower right")
        plt.show()
        between = math.degrees(math.atan2(
            omega[0] * ang[1] - omega[1] * ang[0], omega @ ang))
        print(f"L is {between:+.1f}° from omega")


    widgets.interact(
        tilt, angle=widgets.FloatSlider(20.0, min=0.0, max=180.0, step=2.0),
    );
    """),
    md("""
    At 52° and at 142°, along the dotted principal
    axes, $\\mathbf L$ lies along $\\boldsymbol\\omega$; in between it
    leans towards the axis of the larger moment.
    """),
]

REMOVE = [
    md(r"""
    ## 5.6 Removing translation and rotation

    The ring of six carbon atoms of Figure 5.6. Switch the removal of the
    drift and of the turning on and off, and watch the arrows and the
    three parts of the kinetic energy.
    """),
    code("""
    N_RING = 6
    angles = 2 * np.pi * np.arange(N_RING) / N_RING
    ring = 1.4 * np.column_stack([np.cos(angles), np.sin(angles),
                                  np.zeros(N_RING)])
    ring_masses = np.full(N_RING, 12.011)
    ring_v = np.zeros((N_RING, 3))
    ring_v[:, :2] = 0.003 * np.random.default_rng(5).normal(size=(N_RING, 2))
    ring_v += np.array([0.006, 0.003, 0.0]) + np.cross([0, 0, 0.006], ring)


    def remover(drift=False, turning=False):
        v = ring_v.copy()
        if drift:
            v = rotation.remove_rigid_motion(ring_masses, ring, v,
                                             rotation=False)
        if turning:
            centre = dynamics.centre_of_mass(ring_masses, ring)
            omega = rotation.angular_velocity(ring_masses, ring, v)
            v -= np.cross(omega, ring - centre)
        fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=120)
        ax.scatter(ring[:, 0], ring[:, 1], s=60, color="0.2", zorder=3)
        ax.quiver(ring[:, 0], ring[:, 1], v[:, 0], v[:, 1], color=viz.ACCENT,
                  scale=0.01, scale_units="xy", angles="xy")
        ax.plot(0, 0, "+", color="black")
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("$x$ / Å")
        ax.set_ylabel("$y$ / Å")
        plt.show()
        centre = dynamics.centre_of_mass(ring_masses, ring)
        p = dynamics.total_momentum(ring_masses, v)
        ang = rotation.angular_momentum(ring_masses, ring, v, centre)
        print(f"P = {np.round(p, 4)} amu Å/fs, L'_z = {ang[2]:+.4f} amu Å²/fs,"
              f" K = {energy.kinetic_energy(ring_masses, v):.4f} eV")


    widgets.interact(remover, drift=False, turning=False);
    """),
    md("""
    Removing the drift zeroes $\\mathbf P$ but
    leaves $L'_z$ = 0.7616; removing the turning as well zeroes both. The
    kinetic energy falls from 0.4317 to 0.2628 and then to 0.0500 eV,
    the energy of the motion within the ring. The switches act
    independently: removing only the turning leaves the original drift.
    """),
    code("""
    # Checked against ASE: Stationary removes the drift and ZeroRotation
    # the turning; get_moments_of_inertia gives the principal moments.
    from ase import Atoms
    from ase.md.velocitydistribution import Stationary, ZeroRotation

    atoms = Atoms("C6", positions=ring)
    atoms.set_masses(ring_masses)
    atoms.set_velocities(ring_v)
    Stationary(atoms, preserve_temperature=False)
    ZeroRotation(atoms, preserve_temperature=False)
    ours = rotation.remove_rigid_motion(ring_masses, ring, ring_v)
    print("velocities differ from ASE's by",
          np.abs(ours - atoms.get_velocities()).max(), "Å/fs")
    centre = dynamics.centre_of_mass(ring_masses, ring)
    moments = np.linalg.eigvalsh(rotation.inertia_tensor(ring_masses, ring,
                                                         centre))
    print("principal moments: mdlab", moments.round(6), "ASE",
          np.sort(atoms.get_moments_of_inertia()).round(6), "amu Å²")
    """),
]

FREEDOM = [
    md(r"""
    ## 5.7 Degrees of freedom

    The three-spring model of water: its $9\times9$ Hessian has six zero
    eigenvalues, the three drifts and three turnings, and three
    vibrations. The stiffnesses are chosen to match the measured bend
    and antisymmetric stretch.
    """),
    code("""
    pairs = [[0, 1], [0, 2], [1, 2]]
    masses9 = np.repeat(water_masses, 3)
    hessian = rotation.spring_hessian(water, pairs, [48.48, 48.48, 17.66])
    w2, modes = oscillators.normal_modes(hessian, masses9,
                                         force_to_accel=units.FORCE_TO_ACCEL)
    print("omega² (fs⁻²):", np.array2string(w2, precision=3))
    wavenumbers = np.sqrt(np.clip(w2[6:], 0, None)) / (
        2 * math.pi * units.C_CM_PER_FS)
    print("vibrations (cm⁻¹):", wavenumbers.round(0),
          " measured: bend 1595, antisymmetric 3756, symmetric 3657")
    for axis in np.eye(3):  # every rigid pattern is turned into zero
        assert np.allclose(hessian @ np.tile(axis, 3), 0, atol=1e-12)
        assert np.allclose(hessian @ np.cross(axis, water).ravel(), 0,
                           atol=1e-12)
    print("drifts and turnings are zero modes: checked")
    """),
    md(r"""
    The three vibrations of the model, drawn as arrows on the atoms (each
    pattern scaled to be visible).
    """),
    code("""
    fig, axes_ = plt.subplots(1, 3, figsize=(viz.FULL, 2.0), dpi=120)
    for ax, k, name in zip(axes_, (6, 7, 8),
                           ("bend", "antisymmetric", "symmetric stretch"),
                           strict=True):
        pattern = modes[:, k].reshape(3, 3)
        pattern = 0.5 * pattern / np.abs(pattern).max()
        ax.scatter(water[:, 0], water[:, 1], s=[120, 50, 50],
                   color=[viz.ELEMENT["O"]["colour"], "0.8", "0.8"],
                   edgecolor="black", zorder=3)
        ax.quiver(water[:, 0], water[:, 1], pattern[:, 0], pattern[:, 1],
                  color=viz.ACCENT, scale=1, scale_units="xy", angles="xy")
        ax.set_title(f"{name}\\n{wavenumbers[k - 6]:.0f} cm⁻¹", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.9, 1.6)
        ax.set_ylim(-0.6, 1.5)
        ax.axis("off")
    plt.show()
    """),
]

MOLECULES = [
    md(r"""
    ## 5.8 Spinning molecules

    LiF ($d$ = 1.5639 Å) turning with $K = k_\mathrm{B}T$ at 300 K, its
    angular momentum drawn along the axis. The animation shows one turn,
    about 992 fs; in that time the bond vibrates 27 times.
    """),
    code("""
    D_LIF = 1.5639
    m_r = oscillators.reduced_mass(6.94, 18.998)
    inertia = m_r * D_LIF**2
    w_rot = math.sqrt(2 * units.KB * 300 * units.FORCE_TO_ACCEL / inertia)
    d_li = 18.998 / (6.94 + 18.998) * D_LIF
    print(f"I = {inertia:.3f} amu Å², omega = {w_rot:.6f} rad/fs, period "
          f"{2 * math.pi / w_rot:.0f} fs, L = {inertia * w_rot:.4f} amu Å²/fs")

    frames = 48
    fig = plt.figure(figsize=(3.6, 3.2), dpi=120)
    ax = fig.add_subplot(projection="3d")
    ax.quiver(0, 0, 0, 0, 0, 1.2, color=viz.ACCENT)
    ax.text(0, 0, 1.3, "L")
    (bond,) = ax.plot([], [], [], color="0.4", lw=2)
    (li,) = ax.plot([], [], [], "o", color=viz.ELEMENT["Li"]["colour"], ms=10)
    (f_atom,) = ax.plot([], [], [], "o", color=viz.ELEMENT["F"]["colour"],
                        ms=7)
    for setter in (ax.set_xlim, ax.set_ylim):
        setter(-1.3, 1.3)
    ax.set_zlim(-0.6, 1.4)


    def draw(k):
        phi = 2 * np.pi * k / frames
        u = np.array([math.cos(phi), math.sin(phi), 0.0])
        p_li, p_f = -d_li * u, (D_LIF - d_li) * u
        bond.set_data_3d(*np.column_stack([p_li, p_f]))
        li.set_data_3d([p_li[0]], [p_li[1]], [0.0])
        f_atom.set_data_3d([p_f[0]], [p_f[1]], [0.0])
        return bond, li, f_atom


    animation = FuncAnimation(fig, draw, frames=frames, interval=70)
    plt.close(fig)
    HTML(animation.to_jshtml())
    """),
    md("""
    The lighter lithium atom sweeps the large
    circle; $\\mathbf L$ never moves, since nothing outside the molecule
    exerts a torque on it.
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
    derivations 5.3, 5.5, 5.6, 5.7, 5.8, 5.11, 5.13, 5.14, 5.18 and 5.19
    have their solutions checked by SymPy.
    """),
    hidden("""
    # The answer key. Each target is computed rather than typed in.
    _ell = 0.2 * 0.5 * 1.2
    _v02 = _ell / (0.2 * 0.1)
    _i1, _i2 = 2.0 + 2 * 3.0 * 0.7**2, 2.0 + 2 * 3.0 * 0.3**2
    _w9 = 1.5 * _i1 / _i2
    _a10 = G * math.sin(math.radians(30)) / 1.4
    _square = np.array([[1.0, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
    _r16 = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    _v16 = np.array([[0.01, 0.01, 0], [-0.01, 0.03, 0]])
    _mr = oscillators.reduced_mass(6.94, 18.998)
    _inertia = _mr * 1.5639**2
    _wrot = math.sqrt(2 * units.KB * 300 * units.FORCE_TO_ACCEL / _inertia)
    _wvib = 2 * math.pi * 910.34 * units.C_CM_PER_FS
    _what = math.sqrt(2 * units.KB * 1000 * units.FORCE_TO_ACCEL / _inertia)
    TARGETS = {
        "5.1 torque": 0.8 * 10 * math.sin(math.radians(30)),
        "5.2": np.array([_v02, 0.5 * 0.2 * _v02**2,
                         0.5 * 0.2 * _v02**2 - 0.5 * 0.2 * 1.2**2]),
        "5.4 product": np.cross([2.0, 0, 1], [1.0, 3, -1]),
        "5.4 area": float(np.linalg.norm(np.cross([2.0, 0, 1], [1.0, 3, -1]))),
        "5.9": np.array([_w9, 0.5 * _i2 * _w9**2 - 0.5 * _i1 * 1.5**2]),
        "5.10": np.array([math.sqrt(2 * G / 1.4), math.sqrt(4 / _a10)]),
        "5.11 friction": G * math.sin(math.radians(30)) * 0.5 / 1.5,
        "5.12 moments": np.linalg.eigvalsh(
            rotation.inertia_tensor([1.0] * 4, _square)),
        "5.15": np.linalg.solve([[4.0, 1, 0], [1, 4, 1], [0, 1, 4]],
                                [6.0, 12, 14]),
        "5.16 rest": rotation.remove_rigid_motion([1.0, 1.0], _r16, _v16),
        "5.16 energy": energy.kinetic_energy([1.0, 1.0], _v16),
        "5.17": np.array([12 - 6, 12 - 6]),
        "5.19 stretch": 1.5639 * (_wrot / _wvib) ** 2,
        "5.20 period": 2 * math.pi / _what,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code("""
    # EXERCISE 5.1: torque of 10 N at 30° to the door, 0.8 m out (N m).
    torque = None

    check(torque, TARGETS["5.1 torque"], name="torque")
    """),
    solution("""
    # SOLUTION 5.1. Only the part across the door, F sin 30°, turns it.
    torque = 0.8 * 10 * math.sin(math.radians(30))

    check(torque, TARGETS["5.1 torque"], name="torque")
    """),
    code("""
    # EXERCISE 5.2: speed (m/s), kinetic energy (J), work of the hand (J).
    answers = None

    check(answers, TARGETS["5.2"], name="speed, K, work")
    """),
    solution("""
    # SOLUTION 5.2. L = m r v is conserved.
    speed = 0.12 / (0.2 * 0.1)
    kinetic = 0.5 * 0.2 * speed**2
    answers = np.array([speed, kinetic, kinetic - 0.144])

    check(answers, TARGETS["5.2"], name="speed, K, work")
    """),
    solution("""
    # SOLUTION 5.3. With no force, (r0 + v t - a) x v does not depend on t.
    t, m = sp.symbols("t m")
    r0, v, a = (sp.Matrix(sp.symbols(f"{n}_x {n}_y {n}_z")) for n in "rva")
    ang = m * (r0 + v * t - a).cross(v)
    assert sp.simplify(ang.diff(t)) == sp.zeros(3, 1)
    print("dL/dt = 0 for free motion: checked")
    """),
    code("""
    # EXERCISE 5.4: the cross product (a vector) and the area.
    product = None
    area = None

    check(product, TARGETS["5.4 product"], name="product")
    check(area, TARGETS["5.4 area"], name="area")
    """),
    solution("""
    # SOLUTION 5.4.
    product = np.cross([2.0, 0, 1], [1.0, 3, -1])
    area = math.sqrt(5 * 11 - 1**2)

    check(product, TARGETS["5.4 product"], name="product")
    check(area, TARGETS["5.4 area"], name="area")
    """),
    solution("""
    # SOLUTION 5.5. r . (r x v) = 0, so r stays at right angles to L.
    r_, v_ = (sp.Matrix(sp.symbols(f"{n}_x {n}_y {n}_z")) for n in "rv")
    assert sp.expand(r_.dot(r_.cross(v_))) == 0
    print("r . L = 0 at every instant: checked")
    """),
    solution("""
    # SOLUTION 5.6. About a: L - a x P, checked for three bodies.
    rs = [sp.Matrix(sp.symbols(f"x{i} y{i} z{i}")) for i in range(3)]
    ps = [sp.Matrix(sp.symbols(f"p{i} q{i} s{i}")) for i in range(3)]
    a = sp.Matrix(sp.symbols("a_x a_y a_z"))
    about_a = sum(((r - a).cross(p) for r, p in zip(rs, ps)), sp.zeros(3, 1))
    about_0 = sum((r.cross(p) for r, p in zip(rs, ps)), sp.zeros(3, 1))
    total_p = sum(ps, sp.zeros(3, 1))
    assert sp.expand(about_a - (about_0 - a.cross(total_p))) == sp.zeros(3, 1)
    print("L about a = L - a x P: checked")
    """),
    solution("""
    # SOLUTION 5.7. The rod about its middle and about one end.
    x, M, ell = sp.symbols("x M ell", positive=True)
    middle = sp.integrate(x**2 * M / ell, (x, -ell / 2, ell / 2))
    end = sp.integrate(x**2 * M / ell, (x, 0, ell))
    assert sp.simplify(middle - M * ell**2 / 12) == 0
    assert sp.simplify(end - M * ell**2 / 3) == 0
    print("rod:", middle, "and", end)
    """),
    solution("""
    # SOLUTION 5.8. The parallel axis, for three bodies with symbols.
    ms_ = sp.symbols("m1:4", positive=True)
    xs, ys = sp.symbols("x1:4"), sp.symbols("y1:4")
    s_x, s_y = sp.symbols("s_x s_y")
    total = sum(ms_)
    com_x = sum(m * x for m, x in zip(ms_, xs)) / total  # centre of mass
    com_y = sum(m * y for m, y in zip(ms_, ys)) / total


    def about(cx, cy):  # moment about the axis along z through (cx, cy)
        return sum(m * ((x - cx) ** 2 + (y - cy) ** 2)
                   for m, x, y in zip(ms_, xs, ys))


    shifted = about(com_x + s_x, com_y + s_y)
    assert sp.simplify(shifted - about(com_x, com_y)
                       - total * (s_x**2 + s_y**2)) == 0
    print("I = I_cm + M s²: checked")
    """),
    code("""
    # EXERCISE 5.9: new angular speed (rad/s) and work of the arms (J).
    answers = None

    check(answers, TARGETS["5.9"], name="omega, work")
    """),
    solution("""
    # SOLUTION 5.9. I omega is conserved.
    i1, i2 = 2.0 + 2 * 3.0 * 0.7**2, 2.0 + 2 * 3.0 * 0.3**2
    omega = 1.5 * i1 / i2
    answers = np.array([omega, 0.5 * i2 * omega**2 - 0.5 * i1 * 1.5**2])

    check(answers, TARGETS["5.9"], name="omega, work")
    """),
    code("""
    # EXERCISE 5.10: speed after 1 m (m/s) and time for 2 m at 30° (s).
    answers = None

    check(answers, TARGETS["5.10"], name="speed, time")
    """),
    solution("""
    # SOLUTION 5.10. kappa = 2/5.
    accel = G * math.sin(math.radians(30)) / 1.4
    answers = np.array([math.sqrt(2 * G * 1.0 / 1.4),
                        math.sqrt(2 * 2 / accel)])

    check(answers, TARGETS["5.10"], name="speed, time")
    """),
    code("""
    # EXERCISE 5.11: the friction force on the 1 kg disc (N).
    friction = None

    check(friction, TARGETS["5.11 friction"], name="friction")
    """),
    solution("""
    # SOLUTION 5.11. M a = M g sin(alpha) - f with a = g sin(alpha)/(1+kappa).
    kappa_, g_, alpha, M_ = sp.symbols("kappa g alpha M", positive=True)
    f = M_ * g_ * sp.sin(alpha) - M_ * g_ * sp.sin(alpha) / (1 + kappa_)
    assert sp.simplify(f - kappa_ * M_ * g_ * sp.sin(alpha) / (1 + kappa_)) == 0
    friction = float(f.subs({kappa_: 0.5, g_: G, alpha: sp.pi / 6, M_: 1}))

    check(friction, TARGETS["5.11 friction"], name="friction")
    """),
    code("""
    # EXERCISE 5.12: the principal moments of the square (amu Å²),
    # smallest first.
    moments = None

    check(moments, TARGETS["5.12 moments"], name="moments")
    """),
    solution("""
    # SOLUTION 5.12. The sums of the book, every mass 1.
    square = np.array([[1.0, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
    x_, y_, z_ = square.T
    i_xy = -np.sum(x_ * y_)
    moments = np.array([np.sum(y_**2 + z_**2), np.sum(x_**2 + z_**2),
                        np.sum(x_**2 + y_**2)])
    print("I_xx, I_yy, I_zz =", moments, " I_xy =", i_xy)

    check(moments, TARGETS["5.12 moments"], name="moments")
    """),
    solution("""
    # SOLUTION 5.13. A flat body: I_zz = I_xx + I_yy, I_xz = I_yz = 0.
    ms_ = sp.symbols("m1:4", positive=True)
    arms = [sp.Matrix([*sp.symbols(f"x{i} y{i}"), 0]) for i in range(1, 4)]
    t_ = sum((m * (r_.dot(r_) * sp.eye(3) - r_ * r_.T)
              for m, r_ in zip(ms_, arms)), sp.zeros(3, 3))
    assert sp.expand(t_[2, 2] - t_[0, 0] - t_[1, 1]) == 0
    assert t_[0, 2] == 0 and t_[1, 2] == 0
    assert t_ * sp.Matrix([0, 0, 1]) == t_[2, 2] * sp.Matrix([0, 0, 1])
    print("flat body: checked")
    """),
    solution("""
    # SOLUTION 5.14. Two atoms on z, about their centre of mass.
    m1, m2, d = sp.symbols("m_1 m_2 d", positive=True)
    d1, d2 = m2 * d / (m1 + m2), m1 * d / (m1 + m2)
    atoms = [(m1, sp.Matrix([0, 0, -d1])), (m2, sp.Matrix([0, 0, d2]))]
    assert sp.simplify(m1 * atoms[0][1] + m2 * atoms[1][1]) == sp.zeros(3, 1)
    t_ = sum((m * (r_.dot(r_) * sp.eye(3) - r_ * r_.T) for m, r_ in atoms),
             sp.zeros(3, 3))
    reduced = m1 * m2 / (m1 + m2)
    assert sp.simplify(t_ - sp.diag(reduced * d**2, reduced * d**2, 0)) \
        == sp.zeros(3, 3)
    print("I = diag(m_r d², m_r d², 0): checked")
    """),
    code("""
    # EXERCISE 5.15: (omega_x, omega_y, omega_z).
    omega = None

    check(omega, TARGETS["5.15"], name="solution")
    """),
    solution("""
    # SOLUTION 5.15. By elimination, as in the book: (1, 2, 3).
    omega_x = (15 * 6 - 34) / (15 * 4 - 4)  # 4 wx + 15 (6 - 4 wx) = 34
    omega_y = 6 - 4 * omega_x
    omega_z = (14 - omega_y) / 4
    omega = np.array([omega_x, omega_y, omega_z])

    check(omega, TARGETS["5.15"], name="solution")
    """),
    code("""
    # EXERCISE 5.16: the velocities that remain (two rows of three, Å/fs)
    # and the kinetic energy (eV).
    rest = None
    kinetic = None

    check(rest, TARGETS["5.16 rest"], atol=1e-12, name="what remains")
    check(kinetic, TARGETS["5.16 energy"], name="kinetic energy")
    """),
    solution("""
    # SOLUTION 5.16. Drift (0, 0.02, 0), turning 0.01 rad/fs about z; a
    # stretch along x remains.
    r16 = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    v16 = np.array([[0.01, 0.01, 0], [-0.01, 0.03, 0]])
    rest = rotation.remove_rigid_motion([1.0, 1.0], r16, v16)
    kinetic = energy.kinetic_energy([1.0, 1.0], v16)
    print("remains:", rest.round(6).tolist())

    check(rest, TARGETS["5.16 rest"], atol=1e-12, name="what remains")
    check(kinetic, TARGETS["5.16 energy"], name="kinetic energy")
    """),
    code("""
    # EXERCISE 5.17: internal degrees of freedom of NH3 and H2O2.
    counts = None

    check(counts, TARGETS["5.17"], name="counts")
    """),
    solution("""
    # SOLUTION 5.17. Four atoms, not linear: 3 x 4 - 6 each.
    counts = np.array([3 * 4 - 6, 3 * 4 - 6])

    check(counts, TARGETS["5.17"], name="counts")
    """),
    solution("""
    # SOLUTION 5.18. One spring along z: the turning about x is a zero mode.
    k_, d_, eps = sp.symbols("k d epsilon", positive=True)
    q = sp.symbols("x1 y1 z1 x2 y2 z2")
    along = sp.Matrix([0, 0, 1])  # the unit vector e along the spring
    energy_ = k_ / 2 * along.dot(sp.Matrix(q[3:]) - sp.Matrix(q[:3])) ** 2
    phi = sp.hessian(energy_, q)
    turn = sp.Matrix([0, 0, 0, *(eps * sp.Matrix([1, 0, 0]).cross(
        sp.Matrix([0, 0, d_])))])
    assert phi * turn == sp.zeros(6, 1)
    print("Hessian:", phi.tolist())
    print("Phi times the turning about x is zero: checked")
    """),
    code("""
    # EXERCISE 5.19: the stretch of the spinning LiF bond (Å).
    stretch = None

    check(stretch, TARGETS["5.19 stretch"], rtol=1e-2, name="stretch")
    """),
    solution("""
    # SOLUTION 5.19. m_r omega² r = k (r - d) gives r - d ≈ d (omega/omega_v)².
    w_, wv, dd, rr = sp.symbols("omega omega_v d r", positive=True)
    exact = sp.solve(sp.Eq(w_**2 * rr, wv**2 * (rr - dd)), rr)[0] - dd
    series = sp.series(exact, w_, 0, 3).removeO()
    assert sp.simplify(series - dd * w_**2 / wv**2) == 0
    w_vib = 2 * math.pi * 910.34 * units.C_CM_PER_FS
    stretch = 1.5639 * (w_rot / w_vib) ** 2

    check(stretch, TARGETS["5.19 stretch"], rtol=1e-2, name="stretch")
    """),
    code("""
    # EXERCISE 5.20: the period of turning of LiF at 1000 K (fs).
    period = None

    check(period, TARGETS["5.20 period"], name="period")
    """),
    solution("""
    # SOLUTION 5.20. omega grows as the square root of the energy.
    period = 2 * math.pi / w_rot * math.sqrt(300 / 1000)

    check(period, TARGETS["5.20 period"], name="period")
    """),
]

CELLS = (
    SETUP
    + PLANE
    + CROSS
    + SYSTEM
    + RIGID
    + TENSOR
    + REMOVE
    + FREEDOM
    + MOLECULES
    + EXERCISES
)

if __name__ == "__main__":
    print("wrote", write(CELLS, "05_rotation.ipynb"))
