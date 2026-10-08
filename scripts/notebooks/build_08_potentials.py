"""Write notebooks/08_potentials.ipynb, the companion to Chapter 8.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_08_potentials.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/08_potentials.ipynb

Section 8.5 reads the energy records cached by
scripts/ch08_potentials/fig_cutoff.py; run that script first if
data/ch08_potentials/cutoff_runs.npz is missing.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 08: Potential energy surfaces and interatomic models

    Working through Chapter 8: inspect an energy surface, compare pair
    potentials, and check that the forces agree with the energy. Run the
    cells in order. The cutoff comparison uses the saved cluster
    trajectories, so we can inspect their behaviour without repeating
    the simulation.
    """),
    code(r"""
    %matplotlib inline
    import math
    import time
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from scipy import constants as c

    from mdlab import potentials, units, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    RNG = np.random.default_rng(8)  # the fixed seed of every random draw
    KT300 = units.KB * 300.0  # eV
    HC = c.h * c.c / c.e * 100.0  # eV cm
    """),
]

ORIGIN = [
    md(r"""
    ## 8.1 Where the potential energy comes from

    A potential energy surface in two dimensions: the Müller-Brown
    surface (Müller and Brown, 1979), a standard test landscape with three
    minima and two saddle points, drawn with its forces $-\nabla U$ as
    arrows. The coordinates and energy are in arbitrary units. Move the
    probe with the sliders; arrows show direction with a fixed display
    length, and the printed components give the force magnitude.
    """),
    code(r"""
    MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
    MB_a = np.array([-1.0, -1.0, -6.5, 0.7])
    MB_b = np.array([0.0, 0.0, 11.0, 0.6])
    MB_c = np.array([-10.0, -10.0, -6.5, 0.7])
    MB_x0 = np.array([1.0, 0.0, -0.5, -1.0])
    MB_y0 = np.array([0.0, 0.5, 1.5, 1.0])


    def muller_brown(x, y):
        x, y = np.asarray(x)[..., None], np.asarray(y)[..., None]
        dx, dy = x - MB_x0, y - MB_y0
        terms = MB_A * np.exp(MB_a * dx**2 + MB_b * dx * dy + MB_c * dy**2)
        u = terms.sum(axis=-1)
        fx = -(terms * (2 * MB_a * dx + MB_b * dy)).sum(axis=-1)
        fy = -(terms * (MB_b * dx + 2 * MB_c * dy)).sum(axis=-1)
        return u, fx, fy


    def surface(x=0.0, y=0.6):
        gx, gy = np.meshgrid(np.linspace(-1.6, 1.2, 300),
                             np.linspace(-0.4, 2.1, 300))
        u = muller_brown(gx, gy)[0]
        fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=120)
        levels = np.linspace(u.min(), 100, 31)
        contours = ax.contourf(gx, gy, u, levels=levels, cmap="cividis",
                              extend="max")
        fig.colorbar(contours, ax=ax, label="$U$ / arbitrary units")
        ax.contour(gx, gy, u, levels=levels, colors="white",
                   linewidths=0.3)
        ax_, ay_ = np.meshgrid(np.linspace(-1.5, 1.1, 14),
                               np.linspace(-0.3, 2.0, 12))
        _, fx, fy = muller_brown(ax_, ay_)
        size = np.hypot(fx, fy) + 1e-9
        ax.quiver(ax_, ay_, fx / size, fy / size, color="white", alpha=0.7,
                  scale=30)
        u0, fx0, fy0 = muller_brown(x, y)
        ax.plot(x, y, "o", color=viz.OCHRE)
        norm = max(float(np.hypot(fx0, fy0)), 1e-12)
        ax.annotate("", xy=(x + 0.3 * fx0 / norm, y + 0.3 * fy0 / norm),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=viz.OCHRE))
        ax.set_xlim(-1.9, 1.5)
        ax.set_ylim(-0.7, 2.4)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ / arbitrary units")
        ax.set_ylabel("$y$ / arbitrary units")
        plt.show()
        print(f"U = {float(u0):.1f}, force = ({float(fx0):.1f}, "
              f"{float(fy0):.1f})")


    widgets.interact(
        surface,
        x=widgets.FloatSlider(0.0, min=-1.5, max=1.1, step=0.01, **SLOW),
        y=widgets.FloatSlider(0.6, min=-0.3, max=2.0, step=0.01, **SLOW),
    );
    """),
    md(r"""
    The arrows point downhill, at right angles to
    the contours. At the three minima and the two saddle points between
    them the force vanishes; move the probe to (−0.56, 1.44), (0.62, 0.03)
    or (−0.05, 0.47) to find the minima.
    """),
    code(r"""
    # The cost of a dense eigenproblem grows as n³ (one thread is not
    # enforced here, so the slope may come out lower than 3).
    for n in (100, 200, 400, 800):
        a = RNG.normal(size=(n, n))
        a = a + a.T
        start = time.perf_counter()
        np.linalg.eigh(a)
        print(f"n = {n}: {time.perf_counter() - start:.4f} s")
    """),
]

CLASSICAL = [
    md(r"""
    ## 8.2 When the nuclei may be treated as classical

    Compare the quantum step $\hbar\omega = 2\pi\hbar c\tilde\nu$ with
    $k_BT$. A ratio much smaller than 1 supports the classical limit for
    this vibration; a ratio near 1 needs more care. We retain the ratio
    itself rather than assigning a universal threshold.
    """),
    code(r"""
    def quantum(wavenumber=1000.0, temperature=300.0):
        step = HC * wavenumber
        ratio = step / (units.KB * temperature)
        print(f"hbar omega = {step:.4f} eV, k_B T = "
              f"{units.KB * temperature:.4f} eV, ratio {ratio:.2f}")


    widgets.interact(
        quantum,
        wavenumber=widgets.FloatLogSlider(1000.0, min=1, max=4, step=0.05,
                                          **SLOW),
        temperature=widgets.FloatSlider(300.0, min=10.0, max=1500.0,
                                        step=10.0, **SLOW),
    );
    """),
    md(r"""
    At 300 K the two energies are equal, a ratio of 1, at
    about 208 cm⁻¹: slow rattles of heavy atoms fall below it, stretches
    of bonds to hydrogen far above.
    """),
]

PAIRS = [
    md(r"""
    ## 8.3 Pair potentials

    Compare the three potential shapes and their forces. Here lengths
    are in Å and energies in eV; the sliders set $\varepsilon$ in eV,
    $\sigma$ in Å, and the dimensionless shape parameters.
    """),
    code(r"""
    def explorer(epsilon=1.0, sigma=1.0, morse_width=6.0, alpha=14.0):
        r = np.linspace(0.6, 3.0, 600)
        rm = 2 ** (1 / 6) * sigma
        shapes = {
            "Lennard-Jones": potentials.lennard_jones(r, epsilon, sigma),
            "Morse": potentials.morse(r, epsilon, morse_width / rm, rm),
            "exp-6": potentials.buckingham(
                r, 6 * epsilon * math.exp(alpha) / (alpha - 6), rm / alpha,
                alpha * epsilon * rm**6 / (alpha - 6)),
        }
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 2.8), dpi=120,
                                           gridspec_kw={"wspace": 0.4})
        for (name, (phi, dphi)), colour in zip(
                shapes.items(), (viz.ACCENT, viz.OCHRE, viz.OXBLOOD)):
            left.plot(r, phi, color=colour, label=name)
            right.plot(r, -dphi, color=colour)
        left.set_ylim(-1.3 * epsilon, 2 * epsilon)
        right.set_ylim(-3 * epsilon / sigma, 20 * epsilon / sigma)
        for ax in (left, right):
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_xlabel("$r$ / Å")
        left.set_ylabel(r"$\varphi$ / eV")
        right.set_ylabel(r"$-d\varphi/dr$ / (eV/Å)")
        left.legend()
        plt.show()


    widgets.interact(
        explorer,
        epsilon=widgets.FloatSlider(1.0, min=0.2, max=2.0, step=0.1, **SLOW),
        sigma=widgets.FloatSlider(1.0, min=0.8, max=1.5, step=0.05, **SLOW),
        morse_width=widgets.FloatSlider(6.0, min=3.0, max=10.0, step=0.5,
                                        **SLOW),
        alpha=widgets.FloatSlider(14.0, min=8.0, max=20.0, step=0.5, **SLOW),
    );
    """),
    md(r"""
    All three wells keep their minimum at
    $2^{1/6}\sigma$ with depth $\varepsilon$. A smaller Morse width makes a
    wider, softer well and a longer tail; a smaller exp-6 steepness pulls
    its turnover outwards, towards distances a hot simulation can reach.
    """),
]

FORCES = [
    md(r"""
    ## 8.4 Forces, the virial, and checking forces

    A force routine with a deliberate mistake, a factor of 12 written as 11
    in the slope of the Lennard-Jones repulsion, and the check that catches
    it.
    """),
    code(r"""
    def wrong_lj(r):
        r = np.asarray(r, dtype=float)
        s6 = r**-6
        return 4 * (s6 * s6 - s6), 4 * (-11 * s6 * s6 + 6 * s6) / r


    grid = np.array([[i, j, k] for i in range(3) for j in range(3)
                     for k in range(3)], dtype=float)[:12]
    cluster = 1.15 * grid + 0.08 * RNG.normal(size=grid.shape)
    for name, pair in (("correct", potentials.lennard_jones),
                       ("wrong", wrong_lj)):
        f = potentials.pair_energy_forces(cluster, pair)[1]
        numeric = potentials.finite_difference_forces(
            lambda r: potentials.pair_energy_forces(r, pair)[0], cluster, 1e-5)
        print(f"{name}: largest |analytic − numerical| = "
              f"{np.abs(f - numeric).max():.2e}")
    """),
    md(r"""
    The correct routine agrees with the differences
    to about $10^{-7}$; the wrong one disagrees by whole units of
    $\varepsilon/\sigma$, though both still give forces that add to zero,
    so the third law alone would not catch it.
    """),
    code(r"""
    # Checked against ASE: its Lennard-Jones and Morse calculators on the
    # same cluster. ASE's Morse takes rho0 = a r0 and a smooth cutoff, here
    # moved far beyond the cluster.
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.calculators.morse import MorsePotential

    atoms = Atoms("Ar12", positions=cluster)
    shifted = potentials.with_cutoff(potentials.lennard_jones, 2.5, "shift")
    u, f, _ = potentials.pair_energy_forces(cluster, shifted)
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5, smooth=False)
    print(f"Lennard-Jones: energy differs from ASE's by "
          f"{abs(u - atoms.get_potential_energy()):.1e}, forces by "
          f"{np.abs(f - atoms.get_forces()).max():.1e}")
    u, f, _ = potentials.pair_energy_forces(
        cluster, lambda r: potentials.morse(r, 1.0, 6.0, 1.1))
    atoms.calc = MorsePotential(epsilon=1.0, r0=1.1, rho0=6.0 * 1.1,
                                rcut1=100.0, rcut2=101.0)
    print(f"Morse: energy differs from ASE's by "
          f"{abs(u - atoms.get_potential_energy()):.1e}, forces by "
          f"{np.abs(f - atoms.get_forces()).max():.1e}")
    """),
]

CUTOFFS = [
    md(r"""
    ## 8.5 Cutoffs

    The energy records of the 32-atom cluster, read from the cache written
    by `scripts/ch08_potentials/fig_cutoff.py`.
    """),
    code(r"""
    CACHE = Path("..") / "data" / "ch08_potentials" / "cutoff_runs.npz"
    runs = np.load(CACHE)


    def records(scheme="truncate"):
        e = runs[f"energy_{scheme}"]
        fig, ax = plt.subplots(figsize=(4.5, 2.4), dpi=120)
        ax.plot(runs["times"], e - e[0], color=viz.ACCENT)
        ax.set_xlabel(r"time / $\tau$")
        ax.set_ylabel(r"$E - E(0)$ / $\varepsilon$")
        plt.show()
        print(f"largest change {np.abs(e - e[0]).max():.2e}, force calls "
              f"{int(runs[f'calls_{scheme}'])}")


    widgets.interact(records,
                     scheme=["none", "truncate", "shift", "switch"]);
    """),
    md(r"""
    Only truncation changes the energy visibly;
    look at the vertical scale of the other three. The force calls show
    the hidden cost of shifting: the accurate solver needed 46 times as
    many as without a cutoff.
    """),
]

MANYBODY = [
    md(r"""
    ## 8.6 Many-body potentials for metals and carbon

    The second-moment model: the bond energy of a central atom with $z$
    neighbours on a circle of radius $r_0$, and its forces checked.
    """),
    code(r"""
    PARAMS = dict(repulsion=0.0, hopping=1.0, p=10.0, q=2.0, r0=1.0)
    for z in (1, 2, 3, 4, 6, 8, 12):
        angles = 2 * np.pi * np.arange(z) / z
        star = np.vstack([[0.0, 0.0, 0.0],
                          np.column_stack([np.cos(angles), np.sin(angles),
                                           np.zeros(z)])])
        d = np.linalg.norm(star[1:], axis=1)
        rho = np.sum(np.exp(-2 * PARAMS["q"] * (d - 1)))
        print(f"z = {z:2d}: bond energy of the centre {-math.sqrt(rho):.3f}, "
              f"per bond {-math.sqrt(rho) / z:.3f}")
    r = 1.0 * grid[:10] + 0.08 * RNG.normal(size=(10, 3))
    params = dict(PARAMS, repulsion=0.1, hopping=1.2)
    u, f = potentials.second_moment(r, **params)
    numeric = potentials.finite_difference_forces(
        lambda x: potentials.second_moment(x, **params)[0], r, 1e-5)
    print(f"forces against differences: {np.abs(f - numeric).max():.1e}; "
          f"total force {np.abs(f.sum(axis=0)).max():.1e}")
    """),
]

BONDED = [
    md(r"""
    ## 8.7 Bonded force fields

    Twist the last atom of a chain of four about the middle bond and
    watch the dihedral angle and the OPLS torsion energy.
    """),
    code(r"""
    def twist(angle=180.0, v1=0.1, v2=0.02, v3=0.05):
        t = math.radians(angle)
        a, b, cc = [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]
        d = [1.5, math.cos(t), math.sin(t)]
        phi = potentials.dihedral_angle(a, b, cc, d)
        grid_ = np.linspace(-np.pi, np.pi, 361)
        energy = potentials.opls_torsion(grid_, [v1, v2, v3])
        fig, ax = plt.subplots(figsize=(4.5, 2.6), dpi=120)
        ax.plot(np.degrees(grid_), energy, color=viz.ACCENT)
        ax.plot(math.degrees(phi), potentials.opls_torsion(phi, [v1, v2, v3]),
                "o", color=viz.OCHRE)
        ax.set_xlabel(r"dihedral angle $\psi$ / degrees")
        ax.set_ylabel("torsion energy / eV")
        plt.show()
        print(f"psi = {math.degrees(phi):.1f} degrees")


    widgets.interact(
        twist,
        angle=widgets.FloatSlider(180.0, min=-180.0, max=180.0, step=5.0,
                                  **SLOW),
        v1=widgets.FloatSlider(0.1, min=-0.2, max=0.2, step=0.01, **SLOW),
        v2=widgets.FloatSlider(0.02, min=-0.2, max=0.2, step=0.01, **SLOW),
        v3=widgets.FloatSlider(0.05, min=-0.2, max=0.2, step=0.01, **SLOW),
    );
    """),
    md(r"""
    With $\kappa_2$ at its default and positive
    $\kappa_1$ and $\kappa_3$, the trans arrangement, 180°, is lowest and
    the cis, 0°, highest; $\kappa_3$ alone makes three equal minima, at
    ±60° and 180°. (The sliders `v1`, `v2`, `v3` set $\kappa_1$,
    $\kappa_2$, $\kappa_3$ in eV.)
    """),
]

COULOMB = [
    md(r"""
    ## 8.8 Charges and the slow decay of $1/r$

    The Madelung sum of rock salt over a sphere of any radius, against
    the neutral cube of the same half-side.
    """),
    code(r"""
    def madelung(radius=6.0):
        sphere = potentials.madelung_partial_sum(radius, "sphere")
        cube = potentials.madelung_partial_sum(max(1, int(radius)), "cube")
        print(f"sphere of radius {radius:.2f} d: {sphere:8.4f}; cube of "
              f"half-side {max(1, int(radius))} d: {cube:.6f}; "
              f"limit 1.747565")


    widgets.interact(
        madelung,
        radius=widgets.FloatSlider(6.0, min=1.0, max=14.0, step=0.05,
                                   **SLOW),
    );
    """),
    md(r"""
    Small changes of the radius swing the sphere's
    sum by several units, as each new shell brings its own net charge;
    from a half-side of $2d$ on, the cube's sum hardly moves from 1.7476.
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
    derivations 8.3, 8.4, 8.5, 8.7, 8.10, 8.11, 8.12, 8.13, 8.15 and 8.16
    have their solutions checked by SymPy.
    """),
    hidden(r"""
    # The answer key. Each target is computed rather than typed in.
    _li = 6.94 / 5.486e-4
    _hollow = units.HBAR * 2 * math.pi / 170.3
    _tau = 3.4 * math.sqrt(39.948 * units.MV2_TO_EV / (120 * units.KB))
    _w = 12 / 2 ** (1 / 6)
    _phi, _dphi = potentials.lennard_jones(1.5)
    _psi = potentials.dihedral_angle([1, 0, 0], [0, 0, 0], [0, 0, 1],
                                     [0, 1, 1])
    TARGETS = {
        "8.1": math.sqrt(_li),
        "8.2": np.array([_hollow, HC * 3657.0]) / units.KB,
        "8.6": np.array([2 * math.pi / _w, 2 * math.pi / _w * _tau,
                         2 * math.pi / _w / 0.005]),
        "8.8": float(-1.5 * _dphi),
        "8.9": (3 * np.finfo(float).eps * 10.75 / 1e4) ** (1 / 3),
        "8.13": 8 / 3 * math.pi * 0.8 * ((1 / 3) * 0.4**9 - 0.4**3),
        "8.14": 0.49 / abs(float(potentials.lennard_jones(2.5)[0])),
        "8.16": 12 * (1 - math.sqrt(1 - 1 / 12)),
        "8.17": np.array([math.degrees(_psi),
                          float(potentials.opls_torsion(_psi,
                                                        [0.1, 0.02, 0.05]))]),
        "8.18": units.COULOMB / KT300,
        "8.19": np.array([potentials.madelung_partial_sum(r + 1e-9, "sphere")
                          for r in (1, 2**0.5, 3**0.5, 2)]),
        "8.20": potentials.madelung_partial_sum(1, "cube"),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 8.1: how many times faster an electron vibrates than a
    # lithium nucleus, with the same stiffness.
    ratio = None

    check(ratio, TARGETS["8.1"], name="ratio")
    """),
    solution(r"""
    # SOLUTION 8.1. omega = sqrt(k/m), so the ratio is sqrt of the masses.
    ratio = math.sqrt(6.94 / 5.486e-4)

    check(ratio, TARGETS["8.1"], name="ratio");
    """),
    code(r"""
    # EXERCISE 8.2: temperatures (K) at which k_B T equals hbar omega for
    # the lithium hollow (period 170.3 fs) and the O-H stretch (3657 cm⁻¹).
    temperatures = None

    check(temperatures, TARGETS["8.2"], name="temperatures")
    """),
    solution(r"""
    # SOLUTION 8.2. T = hbar omega / k_B.
    temperatures = np.array([units.HBAR * 2 * math.pi / 170.3,
                             HC * 3657.0]) / units.KB

    check(temperatures, TARGETS["8.2"], name="temperatures");
    """),
    solution(r"""
    # SOLUTIONS 8.3, 8.4 and 8.5. The strongest Lennard-Jones pull, the
    # lowered Morse well, and the exp-6 minimum.
    r, eps, sig, D, a, r0, al, rm = sp.symbols(
        "r epsilon sigma D a r_0 alpha r_m", positive=True)
    lj = 4 * eps * ((sig / r)**12 - (sig / r)**6)
    r_min = 2 ** sp.Rational(1, 6) * sig
    assert sp.simplify(sp.diff(lj, r).subs(r, r_min)) == 0
    r_pull = (sp.Rational(26, 7)) ** sp.Rational(1, 6) * sig
    assert sp.simplify(sp.diff(lj, r, 2).subs(r, r_pull)) == 0
    pull = sp.diff(lj, r).subs(r, r_pull)
    assert abs(float(pull.subs({eps: 1, sig: 1})) - 2.396) < 1e-3
    morse = D * (sp.exp(-2 * a * (r - r0)) - 2 * sp.exp(-a * (r - r0)))
    assert sp.simplify(morse.subs(r, r0) + D) == 0
    assert sp.simplify(sp.diff(morse, r).subs(r, r0)) == 0
    assert sp.simplify(sp.diff(morse, r, 2).subs(r, r0) - 2 * D * a**2) == 0
    A = 6 * eps * sp.exp(al) / (al - 6)
    exp6 = A * sp.exp(-r * al / rm) - al * eps * rm**6 / (al - 6) / r**6
    assert sp.simplify(sp.diff(exp6, r).subs(r, rm)) == 0
    assert sp.simplify(exp6.subs(r, rm) + eps) == 0
    assert sp.simplify(sp.diff(exp6, r, 2).subs(r, rm)
                       - 6 * al * (al - 7) * eps / ((al - 6) * rm**2)) == 0
    print("LJ strongest pull 2.396 eps/sigma at 1.2445 sigma, Morse minimum "
          "-D with 2Da², exp-6 minimum -eps at r_m for alpha > 7: checked")
    """),
    code(r"""
    # EXERCISE 8.6: the period of an argon pair in units of tau, in fs,
    # and in steps of 0.005 tau.
    period = None

    check(period, TARGETS["8.6"], rtol=5e-3, name="period")
    """),
    solution(r"""
    # SOLUTION 8.6. omega = sqrt(2k/m) with k = 72 eps/r_m², = 12/(2^(1/6) tau).
    omega_tau = 12 / 2 ** (1 / 6)
    tau_fs = 3.4 * math.sqrt(39.948 * units.MV2_TO_EV / (120 * units.KB))
    period = np.array([2 * math.pi / omega_tau,
                       2 * math.pi / omega_tau * tau_fs,
                       2 * math.pi / omega_tau / 0.005])

    check(period, TARGETS["8.6"], rtol=5e-3, name="period");
    """),
    solution(r"""
    # SOLUTION 8.7. The forces of a pair potential add to zero, checked for
    # an arbitrary φ on four atoms: each pair appears twice with opposite
    # signs.
    phi_f = sp.Function("phi")  # a function of the squared distance
    pos = [sp.Matrix(sp.symbols(f"x{a} y{a} z{a}")) for a in range(4)]
    U4 = sum(phi_f(((pos[j] - pos[i]).T * (pos[j] - pos[i]))[0])
             for i in range(4) for j in range(i + 1, 4))
    for comp in range(3):
        assert sp.simplify(sum(sp.diff(U4, p_[comp]) for p_ in pos)) == 0
    print("four atoms in 3D: each component of the total force is zero")
    """),
    code(r"""
    # EXERCISE 8.8: the virial (in eps) of two Lennard-Jones atoms 1.5 sigma
    # apart.
    virial = None

    check(virial, TARGETS["8.8"], name="virial")
    """),
    solution(r"""
    # SOLUTION 8.8. W = -r phi'(r), positive inside the minimum only.
    virial = float(potentials.pair_energy_forces(
        [[0, 0, 0], [1.5, 0, 0]], potentials.lennard_jones)[2])
    rs = np.linspace(0.9, 3.0, 200)
    ws = -rs * potentials.lennard_jones(rs)[1]
    assert np.all((ws > 0) == (rs < 2 ** (1 / 6)))

    check(virial, TARGETS["8.8"], name="virial");
    """),
    code(r"""
    # EXERCISE 8.9: the best step (in sigma) for |U| = 10.75 and
    # |U'''| = 1e4.
    best = None

    check(best, TARGETS["8.9"], rtol=1e-2, name="step")
    """),
    solution(r"""
    # SOLUTION 8.9. h³ = 3 eps_m |U| / |U'''|.
    best = (3 * np.finfo(float).eps * 10.75 / 1e4) ** (1 / 3)

    check(best, TARGETS["8.9"], rtol=1e-2, name="step");
    """),
    solution(r"""
    # SOLUTIONS 8.10, 8.11 and 8.12. The forward difference, the switching
    # function, and TrajCast's envelope.
    x, U1, U3_ = sp.symbols("x U1 U3")
    h, U0, U2, em = sp.symbols("h U0 U2 epsilon_m", positive=True)
    taylor = U0 + U1 * h + U2 * h**2 / 2 + U3_ * h**3 / 6
    forward = sp.expand((taylor - U0) / h)
    assert sp.expand(forward - U1 - U2 * h / 2 - U3_ * h**2 / 6) == 0
    best_h = sp.solve(sp.diff(U2 * h / 2 + em * U0 / h, h), h)
    assert len(best_h) == 1
    assert sp.simplify(best_h[0] ** 2 - 2 * em * U0 / U2) == 0
    least = (U2 * h / 2 + em * U0 / h).subs(h, best_h[0])
    assert sp.simplify(least**2 - 2 * em * U0 * U2) == 0  # ∝ sqrt(eps_m)
    print("forward difference: best h = sqrt(2 eps_m U0/U2), least error "
          "sqrt(2 eps_m U0 U2): checked")
    S = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
    for point, value in ((0, 1), (1, 0)):
        assert S.subs(x, point) == value
        assert sp.diff(S, x).subs(x, point) == 0
        assert sp.diff(S, x, 2).subs(x, point) == 0
    u = 1 - 28 * x**6 + 48 * x**7 - 21 * x**8
    assert u.subs(x, 1) == 0
    assert sp.diff(u, x).subs(x, 1) == 0 and sp.diff(u, x, 2).subs(x, 1) == 0
    print("S and the envelope: values and two derivatives checked")
    """),
    code(r"""
    # EXERCISE 8.13: the Lennard-Jones energy per atom (eps) beyond
    # r_c = 2.5 sigma at n = 0.8 sigma⁻³.
    tail = None

    check(tail, TARGETS["8.13"], name="tail")
    """),
    solution(r"""
    # SOLUTION 8.13. One half of the integral of 4 pi r² n phi(r) beyond
    # r_c, done by SymPy.
    rr, n_, rc = sp.symbols("r n r_c", positive=True)
    integrand = 4 * sp.pi * rr**2 * n_ * 4 * (rr**-12 - rr**-6) / 2
    formula = sp.integrate(integrand, (rr, rc, sp.oo))
    closed = sp.Rational(8, 3) * sp.pi * n_ * (rc**-9 / 3 - rc**-3)
    assert sp.simplify(formula - closed) == 0
    tail = float(closed.subs({n_: 0.8, rc: 2.5}))

    check(tail, TARGETS["8.13"], name="tail");
    """),
    code(r"""
    # EXERCISE 8.14: how many fewer pairs a rise of 0.49 eps corresponds to.
    pairs = None

    check(pairs, TARGETS["8.14"], atol=0.5, name="pairs")
    """),
    solution(r"""
    # SOLUTION 8.14. Each crossing changes the truncated energy by
    # |phi(2.5 sigma)|.
    pairs = 0.49 / abs(float(potentials.lennard_jones(2.5)[0]))

    check(pairs, TARGETS["8.14"], atol=0.5, name="pairs");
    """),
    code(r"""
    # EXERCISE 8.16: the vacancy energy over the cohesive energy for z = 12
    # in the second-moment model.
    vacancy = None

    check(vacancy, TARGETS["8.16"], name="ratio")
    """),
    solution(r"""
    # SOLUTIONS 8.15 and 8.16. The force of the square root (SymPy) and the
    # vacancy ratio.
    rho, drho, z = sp.symbols("rho drho z", positive=True)
    assert sp.simplify(-sp.diff(-sp.sqrt(rho), rho) * drho
                       - drho / (2 * sp.sqrt(rho))) == 0
    xi, q_, r0_ = sp.symbols("xi q r_0", positive=True)
    ri = sp.Matrix(sp.symbols("x_i y_i z_i", real=True))
    rk = sp.Matrix(sp.symbols("x_k y_k z_k", real=True))
    dist = sp.sqrt(((rk - ri).T * (rk - ri))[0])
    term = xi**2 * sp.exp(-2 * q_ * (dist / r0_ - 1))
    grad = sp.Matrix([sp.diff(term, v) for v in rk])
    stated = -(2 * q_ / r0_) * term * (rk - ri) / dist
    assert sp.simplify(grad - stated) == sp.zeros(3, 1)
    ratio_z = z * (sp.sqrt(z) - sp.sqrt(z - 1)) / sp.sqrt(z)
    assert sp.simplify(ratio_z - z * (1 - sp.sqrt(1 - 1 / z))) == 0
    vacancy = float(ratio_z.subs(z, 12))

    check(vacancy, TARGETS["8.16"], name="ratio");
    """),
    code(r"""
    # EXERCISE 8.17: the dihedral angle (degrees) and torsion energy (eV).
    twist_answer = None

    check(twist_answer, TARGETS["8.17"], name="angle, energy")
    """),
    solution(r"""
    # SOLUTION 8.17. Two cross products and the cosine series.
    phi = potentials.dihedral_angle([1, 0, 0], [0, 0, 0], [0, 0, 1],
                                    [0, 1, 1])
    twist_answer = np.array([math.degrees(phi),
                             float(potentials.opls_torsion(phi,
                                                           [0.1, 0.02, 0.05]))])

    check(twist_answer, TARGETS["8.17"], name="angle, energy");
    """),
    code(r"""
    # EXERCISE 8.18: the distance (Å) at which k/r equals k_B T at 300 K.
    distance = None

    check(distance, TARGETS["8.18"], name="distance")
    """),
    solution(r"""
    # SOLUTION 8.18. r = k / (k_B T).
    distance = units.COULOMB / KT300

    check(distance, TARGETS["8.18"], name="distance");
    """),
    code(r"""
    # EXERCISE 8.19: the sphere sums up to d, sqrt2 d, sqrt3 d and 2d.
    shells = None

    check(shells, TARGETS["8.19"], atol=1e-3, name="sums")
    """),
    solution(r"""
    # SOLUTION 8.19. Shells of 6 (opposite), 12 (like), 8 (opposite), 6
    # (like) ions.
    counts = [(6, -1, 1.0), (12, 1, 2**0.5), (8, -1, 3**0.5), (6, 1, 2.0)]
    shells = np.cumsum([-sign * n / dist for n, sign, dist in counts])
    pts = np.array([(a_, b_, c_) for a_ in range(-2, 3) for b_ in range(-2, 3)
                    for c_ in range(-2, 3)])
    for n_ions, sign, dist in counts:
        at = pts[np.isclose(np.linalg.norm(pts, axis=1), dist)]
        assert len(at) == n_ions
        assert np.all((-1.0) ** at.sum(axis=1) == sign)

    check(shells, TARGETS["8.19"], atol=1e-3, name="sums");
    """),
    code(r"""
    # EXERCISE 8.20: the cube sum with half-side d.
    cube1 = None

    check(cube1, TARGETS["8.20"], name="cube")
    """),
    solution(r"""
    # SOLUTION 8.20. Faces, edges and corners with weights 1/2, 1/4, 1/8.
    cube1 = 6 * 0.5 - 12 * 0.25 / math.sqrt(2) + 8 * 0.125 / math.sqrt(3)

    check(cube1, TARGETS["8.20"], name="cube");
    """),
]

CELLS = (SETUP + ORIGIN + CLASSICAL + PAIRS + FORCES + CUTOFFS + MANYBODY
         + BONDED + COULOMB + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "08_potentials.ipynb"))
