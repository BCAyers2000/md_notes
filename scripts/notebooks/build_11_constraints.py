"""Write notebooks/11_constraints.ipynb, the companion to Chapter 11.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_11_constraints.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/11_constraints.ipynb

The water runs are read from data/ch11_constraints/, written by
scripts/ch11_constraints/prepare_water.py and runs.py (about 10 minutes
together), and the animation from renders/ch11_constraints/
water_frames.py --movie. Section 11.2 ends with a check of mdlab's RATTLE
against ASE; the other sections check against exact results, against the
plain method they extend, or against a healthy run.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 11: Constraints and multiple time steps

    The question here is how much longer a step we can take without losing
    control of the trajectory. Start with one constrained bond, then
    compare flexible and rigid water, multiple time steps and heavier
    hydrogens. The section numbers follow Chapter 11.

    Run the cells in order. Water trajectories are saved in
    `data/ch11_constraints/`; the short bond examples run here. RATTLE is
    compared with ASE, and RESPA is checked against velocity Verlet with
    one inner step and against the two-spring stability calculation.
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
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixBondLengths
    from ase.md.verlet import VelocityVerlet
    from IPython.display import Image, display

    from mdlab import constraints, diagnostics, integrators, oscillators
    from mdlab import respa, rotation, units, viz, water
    from mdlab.exercise import check

    viz.use_style()
    rng = np.random.default_rng(11)
    SLOW = dict(continuous_update=False)  # redraw only on release
    DATA = Path("..") / "data" / "ch11_constraints"
    START = np.load(DATA / "water_start.npz")
    H0, M0 = START["cell"], START["masses"]
    N_MOL = len(M0) // 3
    BONDS, LENGTHS = water.constraint_bonds(N_MOL)


    def run(name):
        # Load a saved trajectory from scripts/ch11_constraints/runs.py.
        return np.load(DATA / "runs" / f"{name}.npz")


    def total(r):
        e = r["potential"] + r["kinetic"]
        return e - e[0]
    """),
]

SHAKE = [
    md(r"""
    ## 11.1 Holding a bond fixed

    One bond of length 1, drifted to a stretch you choose and turned away
    from the old bond by an angle; SHAKE corrects it sweep by sweep. The
    masses only decide how the correction is shared.
    """),
    code(r"""
    def shake_sweeps(stretch=0.1, angle=18.0, mass_ratio=3.0):
        old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        a = math.radians(angle)
        r = np.array([[0.0, 0.0, 0.0],
                      [(1 + stretch) * math.cos(a),
                       (1 + stretch) * math.sin(a), 0.0]])
        start = r.copy()
        errors = [abs(np.linalg.norm(r[1] - r[0]) - 1)]
        for _ in range(8):
            r, _ = constraints.shake(r, old, [1.0, mass_ratio], [[0, 1]],
                                     [1.0], tolerance=0.0, max_sweeps=1,
                                     strict=False)
            errors.append(abs(np.linalg.norm(r[1] - r[0]) - 1))
        errors = np.array(errors)
        fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
        shown = errors > 1e-16
        ax.semilogy(np.arange(9)[shown], errors[shown], "o-")
        ax.set_xlabel("sweeps")
        ax.set_ylabel("relative error of the length")
        plt.show()
        moves = np.linalg.norm(r - start, axis=1)
        print("errors:", ", ".join(f"{e:.1e}" for e in errors[:5]))
        print(f"atom i moved {moves[0]:.4f}, atom j {moves[1]:.4f}: ratio "
              f"{moves[0] / moves[1]:.2f} (the mass ratio)")


    widgets.interact(
        shake_sweeps,
        stretch=widgets.FloatSlider(0.1, min=0.01, max=0.5, step=0.01,
                                    **SLOW),
        angle=widgets.FloatSlider(18, min=0, max=60, step=1, **SLOW),
        mass_ratio=widgets.FloatSlider(3, min=1, max=16, step=0.5, **SLOW),
    );
    """),
    md(r"""
    Once the error is small, each sweep roughly
    squares it, so the number of correct digits about doubles, and the
    bond reaches the rounding of the arithmetic in three to eight sweeps,
    more only within a fraction of a degree of the limit below. When the
    stretched bond is turned so far that its part at right angles to the
    old bond exceeds the length, for an angle above
    $\arcsin(1/(1 + \text{stretch}))$, about 42° at a stretch of 0.5, no
    move along the old bond can restore the length: the errors fall for a
    few sweeps, then stall and jump about, and in a simulation SHAKE would
    fail. The mass ratio leaves the
    errors unchanged and sets only how far each atom moves: the lighter
    one moves further, in the ratio of the masses.
    """),
    code(r"""
    # Newton's method for y^2 = 2 from y = 1 (the toolbox).
    y = 1.0
    for n in range(5):
        print(f"y_{n} = {y:.13f}, error {abs(y - math.sqrt(2)):.2e}")
        y -= (y * y - 2) / (2 * y)
    """),
    code(r"""
    # SymPy: the constraint at the end of the step is the quadratic of
    # Equation 11.1, and its first-order solution is a Newton step from 0.
    eta, d = sp.symbols("eta d", positive=True)
    rt = sp.Matrix(sp.symbols("tx ty tz"))  # the provisional bond
    ro = sp.Matrix(sp.symbols("ox oy oz"))  # the old bond, |ro|^2 = d^2
    f = (rt + eta * ro).dot(rt + eta * ro) - d**2
    quadratic = d**2 * eta**2 + 2 * rt.dot(ro) * eta + rt.dot(rt) - d**2
    on_bond = {ro[0]: sp.sqrt(d**2 - ro[1]**2 - ro[2]**2)}
    assert sp.simplify((f - quadratic).subs(on_bond)) == 0
    newton = -f.subs(eta, 0) / sp.diff(f, eta).subs(eta, 0)
    linear = (d**2 - rt.dot(rt)) / (2 * rt.dot(ro))
    assert sp.simplify(newton - linear) == 0
    print("quadratic and its Newton step: checked")
    """),
    code(r"""
    # The box of 64 rigid molecules after one unconstrained step of 2 fs:
    # sweeps against the tolerance (Figure 11.2b).
    r0, v0 = START["positions"], START["velocities"]
    f0 = water.WaterModel(H0, N_MOL)(r0)[1]
    drifted = r0 + 2.0 * (v0 + units.FORCE_TO_ACCEL * f0 / M0[:, None])
    before = constraints.bond_errors(drifted, BONDS, LENGTHS, H0)
    print(f"largest bond error before SHAKE: {np.abs(before).max():.3f} Å")
    for tol in (1e-2, 1e-6, 1e-10, 1e-14):
        r, n = constraints.shake(drifted, r0, M0, BONDS, LENGTHS, H0, tol)
        left = np.abs(constraints.bond_errors(r, BONDS, LENGTHS, H0)).max()
        print(f"tolerance {tol:.0e}: {n} sweeps, largest error {left:.1e} Å")
    """),
]

RATTLE = [
    md(r"""
    ## 11.2 Velocities too: RATTLE

    The velocity correction of one pair, as in the book: masses 1 and 3,
    joined along $x$.
    """),
    code(r"""
    v_before = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]])
    m_pair = np.array([1.0, 3.0])
    v_after, _ = constraints.rattle_velocities(
        [[0.0, 0, 0], [1.0, 0, 0]], v_before, m_pair, [[0, 1]], [1.0])
    print("corrected velocities:", v_after.round(4).tolist())
    print("momentum before", m_pair @ v_before, "after", m_pair @ v_after)
    """),
    code(r"""
    # Out and back: 20 steps of 2 fs from the prepared box, then the same
    # from the reversed velocities, with the bonds held to 1e-14.
    w = water.WaterModel(H0, N_MOL)
    rs, _ = constraints.shake(START["positions"], START["positions"], M0,
                              BONDS, LENGTHS, H0, tolerance=1e-14)
    vs, _ = constraints.rattle_velocities(rs, START["velocities"], M0,
                                          BONDS, LENGTHS, H0,
                                          tolerance=1e-14)
    go = constraints.run(w, M0, rs, vs, H0, 2.0, 20, BONDS, LENGTHS,
                         every=20, tolerance=1e-14)
    back = constraints.run(w, M0, go["positions"][-1],
                           -go["velocities"][-1], H0, 2.0, 20, BONDS,
                           LENGTHS, every=20, tolerance=1e-14)
    print(f"start recovered to {np.abs(back['positions'][-1] - rs).max():.1e}"
          f" Å")
    print(f"freedoms: flexible {9 * N_MOL - 3}, O-H held "
          f"{9 * N_MOL - 3 - 2 * N_MOL}, rigid {9 * N_MOL - 3 - 3 * N_MOL}")
    """),
    code(r"""
    # Checked against ASE: its VelocityVerlet with FixBondLengths, given
    # the same forces, start and femtosecond, for 20 steps of 1 fs.
    class Wrapped(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, model):
            super().__init__()
            self.model = model

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            u, f = self.model(self.atoms.positions)
            self.results = {"energy": u, "forces": f}


    ours = constraints.run(w, M0, rs, vs, H0, 1.0, 20, BONDS, LENGTHS,
                           every=20, tolerance=1e-13)
    atoms = Atoms(["O", "H", "H"] * N_MOL, positions=rs, cell=H0.T,
                  pbc=True, masses=M0)
    fs = math.sqrt(units.FORCE_TO_ACCEL)  # mdlab's fs in ASE's time unit
    atoms.set_velocities(vs / fs)
    atoms.constraints = FixBondLengths(BONDS[:, ::-1], tolerance=1e-13,
                                       bondlengths=LENGTHS)
    atoms.calc = Wrapped(w)
    VelocityVerlet(atoms, timestep=1.0 * fs).run(20)
    dr = np.abs(atoms.positions - ours["positions"][-1]).max()
    dv = np.abs(atoms.get_velocities() * fs - ours["velocities"][-1]).max()
    print(f"positions differ by {dr:.1e} Å, velocities by {dv:.1e} Å/fs")
    """),
    md(r"""
    Both codes hold the bonds by the same two
    corrections, so with the same forces and the same femtosecond they
    agree to about $10^{-12}$ Å, the rounding of the arithmetic.
    """),
]

WATER = [
    md(r"""
    ## 11.3 Rigid water

    Choose a model and a step from the saved runs of Figure 11.3. Compare
    the change in total energy with the fluctuation ratio of Section 9.6;
    a small drift and a narrow band are separate checks.
    """),
    code(r"""
    SCANS = {
        "flexible": ("flexible", (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)),
        "O-H held": ("bonds", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)),
        "O-H held, heavy H": ("bonds_hmr", (1.0, 2.0, 3.0, 4.0, 5.0)),
        "rigid": ("rigid", (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
        "rigid, heavy H": ("rigid_hmr", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                         7.0, 8.0)),
    }
    CHOICES = [f"{label}, {dt:g} fs" for label, (_, steps) in SCANS.items()
               for dt in steps]


    def step_run(choice="rigid, 2 fs"):
        label, dt = choice.rsplit(", ", 1)
        prefix = SCANS[label][0]
        r = run(f"{prefix}_{float(dt[:-3]):g}")
        fig, ax = plt.subplots(figsize=(6, 2.6), dpi=80)
        ax.plot(r["times"] / 1000, total(r), lw=0.7)
        ax.set_xlabel("time / ps")
        ax.set_ylabel("E - E(0) / eV")
        plt.show()
        ratio = diagnostics.energy_fluctuation(r["potential"], r["kinetic"])
        drift = diagnostics.energy_drift(r["times"], r["potential"],
                                         r["kinetic"])
        print(f"std(E)/std(K) = {ratio:.4f}; drift {1000 * drift:+.4f} "
              f"eV/ps")


    widgets.interact(step_run, choice=widgets.Dropdown(options=CHOICES,
                                                       value="rigid, 2 fs"));
    """),
    md(r"""
    For every model, a doubled step widens the band
    of the total energy about fourfold while the step is short; the band
    stays centred, its mean moving far less than its width, until the
    longest steps. At equal steps the flexible model's band is the widest
    and rigid water's the narrowest, heavier hydrogens narrowing it
    further.
    """),
    code(r"""
    # The step at which each model reaches a fluctuation of 1%.
    def at_level(steps, values, level=0.01):
        for k in range(len(steps) - 1):
            if values[k] <= level < values[k + 1]:
                w_ = math.log(level / values[k]) / math.log(
                    values[k + 1] / values[k])
                return steps[k] * (steps[k + 1] / steps[k]) ** w_
        return float("nan")


    LEVEL_STEP = {}
    for label, (prefix, steps) in SCANS.items():
        values = [diagnostics.energy_fluctuation(
            run(f"{prefix}_{dt:g}")["potential"],
            run(f"{prefix}_{dt:g}")["kinetic"]) for dt in steps]
        LEVEL_STEP[label] = at_level(steps, values)
        print(f"{label:18s} 1% at {LEVEL_STEP[label]:.2f} fs, gain "
              f"{LEVEL_STEP[label] / LEVEL_STEP.get('flexible', 1):.2f}")
    """),
]

RESPA = [
    md(r"""
    ## 11.4 Multiple time steps

    First the exact check: with one inner step, RESPA is velocity Verlet
    with the two forces added.
    """),
    code(r"""
    def chain_springs(x):
        dd = x[1:] - x[:-1]
        dist = np.linalg.norm(dd, axis=1)
        pull = (50.0 * (dist - 1.2) / dist)[:, None] * dd
        f = np.zeros_like(x)
        f[:-1] += pull
        f[1:] -= pull
        return 25.0 * float(np.sum((dist - 1.2) ** 2)), f


    def pair_lj(x):
        from mdlab import potentials
        u, f, _ = potentials.pair_energy_forces(x, potentials.lennard_jones)
        return u, f


    xc = 1.2 * np.arange(12)[:, None] * np.array([1.0, 0, 0])
    xc = xc + rng.normal(scale=0.02, size=xc.shape)
    vc = rng.normal(scale=0.05, size=xc.shape)
    one = respa.run(chain_springs, pair_lj, np.ones(12), xc, vc, 0.002, 1,
                    50, every=50, force_to_accel=1)
    r_, v_, f_ = xc.copy(), vc.copy(), None
    for _ in range(50):
        r_, v_, f_ = integrators.velocity_verlet_step(
            r_, v_, np.ones(12),
            lambda x: chain_springs(x)[1] + pair_lj(x)[1], 0.002, f_, 1)
    print(f"RESPA with one inner step against velocity Verlet: "
          f"{np.abs(one['positions'][-1] - r_).max():.1e}")
    """),
    code(r"""
    OUTER = [round(0.25 * k, 2) for k in range(1, 19)]
    VV_STEPS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


    def outer_step(dt=1.0):
        r = run(f"respa_{dt:g}")
        fig, ax = plt.subplots(figsize=(6, 2.6), dpi=80)
        ax.plot(r["times"] / 1000, total(r), lw=0.7, label="RESPA")
        if dt in VV_STEPS:
            v = run(f"flexible_{dt:g}")
            ax.plot(v["times"] / 1000, total(v), lw=0.7, ls="--",
                    label="velocity Verlet")
        ax.set_xlabel("time / ps")
        ax.set_ylabel("E - E(0) / eV")
        ax.legend(fontsize=7)
        plt.show()
        ratio = diagnostics.energy_fluctuation(r["potential"], r["kinetic"])
        print(f"RESPA: std(E)/std(K) {ratio:.4f}, slow calls per ps "
              f"{1000 / dt:.0f}")


    widgets.interact(outer_step, dt=widgets.SelectionSlider(
        options=OUTER, value=1.0, description="outer / fs", **SLOW));
    """),
    md(r"""
    At 0.25 fs the two methods are the same, one
    inner step per outer step. Wherever velocity Verlet is shown at a
    longer step, RESPA's band is narrower, since it follows the springs
    with steps of 0.25 fs. Up to an outer step of about 1.75 fs the band widens
    smoothly; beyond it the band widens much faster, and from 3.5 fs the
    total energy climbs, until it blows up at 4.5 fs.
    """),
]

RESONANCE = [
    md(r"""
    ## 11.5 Resonance

    SymPy multiplies the outer step $\mathbf{K}\mathbf{R}\mathbf{K}$ and
    finds its trace and determinant.
    """),
    code(r"""
    th, wf, ws, dts = sp.symbols("theta omega_f omega_s delta_t",
                                 positive=True)
    R = sp.Matrix([[sp.cos(th), sp.sin(th) / wf],
                   [-wf * sp.sin(th), sp.cos(th)]])
    K = sp.Matrix([[1, 0], [-dts * ws**2 / 2, 1]])
    M = K * R * K
    assert sp.simplify(M.trace() - (2 * sp.cos(th)
                                    - ws**2 * dts / wf * sp.sin(th))) == 0
    assert sp.simplify(M.det() - 1) == 0
    a_, b_, c_, d_ = sp.symbols("a b c d")
    G = sp.Matrix([[a_, b_], [c_, d_]])
    assert sp.simplify(G**2 - G.trace() * G + G.det() * sp.eye(2)) == \
        sp.zeros(2, 2)
    print("trace, determinant and M^2 = (tr M) M - (det M) I: checked")
    """),
    code(r"""
    def resonance(ratio=0.3, step=0.48):
        period = 2 * math.pi
        grid = np.linspace(0.02, 1.15, 600)
        half = [np.trace(respa.outer_step_matrix(x * period, 1.0, ratio,
                                                 50)) / 2 for x in grid]

        def fast(q):
            return 0.5 * float(np.sum(q * q)), -q

        def slow(q):
            return 0.5 * ratio**2 * float(np.sum(q * q)), -ratio**2 * q

        out = respa.run(fast, slow, [1.0], [[1.0]], [[0.0]], step * period,
                        50, 300, force_to_accel=1)
        fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), dpi=80)
        a.plot(grid, half)
        a.axhline(1, ls=":", c="grey")
        a.axhline(-1, ls=":", c="grey")
        a.axvline(step, c="k", lw=0.8)
        a.set_xlabel("outer step / fast period")
        a.set_ylabel("half the trace")
        b.semilogy(np.abs(out["positions"][:, 0, 0]) + 1e-300)
        b.set_xlabel("outer steps")
        b.set_ylabel("|q|")
        plt.show()
        mid = np.trace(respa.outer_step_matrix(step * period, 1.0, ratio,
                                               50))
        print(f"half the trace at this step {mid / 2:+.4f}; largest |q| "
              f"{np.abs(out['positions']).max():.2e}; predicted band below "
              f"1/2 from {(1 - ratio**2) / 2:.3f} to 0.5")


    widgets.interact(
        resonance,
        ratio=widgets.FloatSlider(0.3, min=0.05, max=0.5, step=0.01, **SLOW),
        step=widgets.FloatSlider(0.48, min=0.05, max=1.15, step=0.005,
                                 description="dt / T_f", **SLOW),
    );
    """),
    md(r"""
    The curve is the trace of the outer step with the
    same 50 inner steps as the run. $|q|$ grows only when the marked step
    lies where half the trace is beyond $\pm1$, in the bands just below
    half the fast period and below the whole period; elsewhere it stays at
    1 or below.
    A larger ratio widens the bands, roughly as its square, and moves
    their lower edges down; the band below the period is about twice as
    wide as the one below half of it.
    """),
]

MASS = [
    md(r"""
    ## 11.6 Heavier hydrogens

    The spring model of water with the hydrogen mass of your choice, taken
    from the oxygen: the periods of its three vibrations, the stability
    limit of velocity Verlet, and the principal moments of inertia.
    """),
    code(r"""
    SHAPE = water.molecule()
    HESSIAN = rotation.spring_hessian(SHAPE, [(0, 1), (0, 2), (1, 2)],
                                      [water.K_OH, water.K_OH, water.K_HH])


    def heavier(hydrogen=3.024):
        masses = constraints.repartition_masses(
            [water.MASS_O, water.MASS_H, water.MASS_H], [[0, 1], [0, 2]],
            hydrogen)
        w2, _ = oscillators.normal_modes(HESSIAN, np.repeat(masses, 3),
                                         force_to_accel=units.FORCE_TO_ACCEL)
        omega = np.sqrt(w2[-3:])
        centre = masses @ SHAPE / masses.sum()
        moments = np.linalg.eigvalsh(rotation.inertia_tensor(masses,
                                                             SHAPE - centre))
        print(f"masses O {masses[0]:.3f}, H {masses[1]:.3f} amu "
              f"(total {masses.sum():.3f})")
        print("periods / fs:", (2 * math.pi / omega).round(2))
        print(f"stability limit 2/omega of the fastest: "
              f"{2 / omega.max():.2f} fs")
        print("principal moments / amu Å^2:", moments.round(3))


    widgets.interact(heavier, hydrogen=widgets.FloatSlider(
        3.024, min=1.008, max=4.0, step=0.004, **SLOW));
    """),
    md(r"""
    The total mass stays 18.015 amu at every
    setting. As the hydrogen grows heavier every period lengthens, the
    stretches' in the largest proportion, and the stability limit rises
    with them; the moments
    of inertia rise too, so a rigid molecule turns more slowly.
    """),
]

LIMITS = [
    md(r"""
    ## 11.7 How far these methods go

    Table 11.1 from the cache: the step at 1% for each model, its gain,
    and the calls of the forces between molecules per picosecond.
    """),
    code(r"""
    respa_values = [diagnostics.energy_fluctuation(
        run(f"respa_{dt:g}")["potential"], run(f"respa_{dt:g}")["kinetic"])
        for dt in OUTER]
    rows = [("flexible", LEVEL_STEP["flexible"]),
            ("flexible, RESPA", at_level(OUTER, respa_values))]
    rows += [(label, LEVEL_STEP[label]) for label in list(SCANS)[1:]]
    for label, step in rows:
        print(f"{label:18s} {step:5.2f} fs  gain "
              f"{step / LEVEL_STEP['flexible']:4.2f}  calls per ps "
              f"{1000 / step:5.0f}")
    """),
]

TRAJECTORY = [
    md(r"""
    ## 11.8 What the trajectory looks like

    The healthy run of rigid water, 10 ps at 2 fs, rendered every 200 fs
    (three molecules' oxygens in teal), then its record.
    """),
    code(r"""
    display(Image(filename=str(DATA / "frames.gif")))
    """),
    code(r"""
    healthy = run("rigid_healthy")
    t = healthy["times"] / 1000
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=80)
    early = t <= 2
    per = 1000 / N_MOL
    for key, label in (("kinetic", "K"), ("potential", "U")):
        axes[0].plot(t[early], per * (healthy[key][early]
                                      - healthy[key][0]), lw=0.5,
                     label=label)
    axes[0].plot(t[early], per * total(healthy)[early], label="E")
    axes[0].set_ylabel("change per molecule / meV")
    axes[0].legend(fontsize=7)
    axes[1].semilogy(t, healthy["bond_error"], lw=0.5, label="bond error")
    com = diagnostics.centre_of_mass_path(M0, healthy["frames"])
    axes[1].semilogy(healthy["frame_times"][1:] / 1000, com[1:],
                     label="centre of mass moved")
    axes[1].legend(fontsize=7)
    rms = diagnostics.rms_displacement(healthy["frames"][:, ::3])
    axes[2].plot(healthy["frame_times"] / 1000, rms)
    axes[2].set_ylabel("RMS displacement of O / Å")
    for a in axes:
        a.set_xlabel("time / ps")
    plt.show()
    u_, k_ = healthy["potential"], healthy["kinetic"]
    drift = diagnostics.energy_drift(healthy["times"], u_, k_)
    print(f"std(E)/std(K) {diagnostics.energy_fluctuation(u_, k_):.4f}, "
          f"drift {1000 * drift:+.1e} eV/ps, largest bond error "
          f"{healthy['bond_error'].max():.1e} Å")
    """),
    md(r"""
    Now break it. Each fault of Table 11.2, with the healthy run beside it
    (dashed) through the quantity that catches the fault.
    """),
    code(r"""
    FAULTS = {
        "(a) too long a step": ("fault_step", "flexible_0.5", "energy"),
        "(b) a cutoff that jumps": ("fault_cutoff", "rigid_2", "energy"),
        "(c) wrong units": ("fault_units", "rigid_2", "kinetic"),
        "(d) wrapped where it must not be": ("fault_wrapped", "rigid_2",
                                             "bonds"),
        "(e) overlapping atoms": ("fault_overlap", "rigid_2", "forces"),
        "(f) constraints drifting": ("fault_tolerance", "rigid_2",
                                     "energy"),
        "(g) drift of the centre of mass": ("fault_drift", "rigid_2",
                                            "centre"),
        "(h) resonance": ("respa_4", "respa_1", "energy"),
    }
    FREE = 6 * N_MOL - 3
    FIRST_FORCES = run("fault_tolerance")["first_forces"]


    def broken(fault="(a) too long a step"):
        bad_name, good_name, what = FAULTS[fault]
        bad, good = run(bad_name), run(good_name)
        fig, ax = plt.subplots(figsize=(6, 2.8), dpi=80)
        if what == "energy":
            ax.plot(bad["times"] / 1000, total(bad), lw=0.7)
            ax.plot(good["times"] / 1000, total(good), "--", lw=0.7)
            ax.set_ylabel("E - E(0) / eV")
        elif what == "kinetic":
            for r, style in ((bad, "-"), (good, "--")):
                ax.plot(r["times"] / 1000, 1000 * r["kinetic"] / FREE,
                        style, lw=0.7)
            ax.axhline(12.93, ls=":", c="grey")
            ax.set_ylabel("K per freedom / meV")
        elif what == "bonds":
            ax.semilogy(bad["times"], bad["bond_error"], "o-")
            ax.semilogy(good["times"][:6], good["bond_error"][:6], "o--")
            ax.set_ylabel("largest bond error / Å")
        elif what == "forces":
            for f, style in ((bad["first_forces"], "o"),
                             (FIRST_FORCES, "s")):
                ax.semilogy(np.sort(np.linalg.norm(f, axis=1))[::-1][:12],
                            style)
            ax.set_ylabel("force / eV/Å, largest first")
        else:
            for r, style in ((bad, "-"), (good, "--")):
                ax.plot(r["frame_times"] / 1000,
                        diagnostics.centre_of_mass_path(M0, r["frames"]),
                        style)
            ax.set_ylabel("centre of mass moved / Å")
        ax.set_xlabel("time / ps" if what not in ("bonds", "forces")
                      else ("time / fs" if what == "bonds" else "atom"))
        plt.show()
        for name, r in (("broken", bad), ("healthy", good)):
            line = f"{name}: {len(r['times']) - 1} steps"
            if len(r["times"]) > 3:
                u_, k_ = r["potential"], r["kinetic"]
                drift = diagnostics.energy_drift(r["times"], u_, k_)
                line += (f", std(E)/std(K) "
                         f"{diagnostics.energy_fluctuation(u_, k_):.4f}"
                         f", drift {1000 * drift:+.2e} eV/ps")
            if "bond_error" in r.files:
                line += f", bond error {r['bond_error'].max():.1e} Å"
            print(line)


    widgets.interact(broken, fault=widgets.Dropdown(options=list(FAULTS)));
    """),
    md(r"""
    Each fault leaves its own mark against the
    dashed healthy run: a wide band (a, h), a total energy that wanders or
    drifts (b, f), a kinetic energy that starts a hundred times below the
    dotted value asked for and stays below it while the energy is
    conserved well (c), a bond error that bursts at once (d), two forces
    some $10^5$ times the rest, beside the healthy squares (e), or a centre
    of mass that moves in a straight line (g). ASE has no checks of this kind
    to compare with; the reference is the healthy run, and the tests of
    `mdlab.diagnostics` run each check on a run with a known fault.
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
    derivations 11.4 to 11.6, 11.9, 11.11 and 11.12 are checked by SymPy
    or NumPy.
    """),
    hidden(r"""
    # Reference values, calculated from the exercise data.
    _old, _new = np.array([1.5, 0, 0]), np.array([1.6, 0.3, 0])
    _eta = max(np.roots([_old @ _old, 2 * _new @ _old, _new @ _new - 2.25]))
    _mr = 10 / 7
    _y = [2.0]
    for _ in range(3):
        _y.append(_y[-1] - (_y[-1] ** 2 - 3) / (2 * _y[-1]))
    _vo = np.array([0.001, 0, 0])
    _vh = np.array([0.02, 0.01, 0])
    _zeta = -(_vh - _vo)[0] * 0.96 / 0.96**2
    _mr_oh = 16 / 17
    _ox = 15.999 - 2 * 1.008
    _ratio = (oscillators.reduced_mass(_ox, 2.016)
              / oscillators.reduced_mass(15.999, 1.008))
    _cosine = math.cos(math.radians(104.52 / 2))
    _p = np.array([0.2817, 0.2936, 0.4033])
    TARGETS = {
        "11.1": np.array([14, 1397]),
        "11.2": np.array([_eta, _eta * _mr / 5 * 1.5, -_eta * _mr / 2 * 1.5]),
        "11.3": np.array(_y[1:]),
        "11.7": np.array([_zeta, _vo[0] - _mr_oh * _zeta * 0.96 / 16,
                          _vh[0] + _mr_oh * _zeta * 0.96,
                          0.5 * _mr_oh * 0.019**2]),
        "11.8": np.array([6 * 216 - 3, (6 * 216 - 3) * 0.01293]),
        "11.10": np.array([1e6 / 0.41 * (24e-3 + 0.04e-3),
                           1e6 / 0.72 * (24e-3 + 3 * 0.04e-3)]),
        "11.13": np.array([0.455, 0.5, 0.91, 1.0]),
        "11.15": np.array([_ox, oscillators.reduced_mass(_ox, 2.016),
                           math.sqrt(_ratio)]),
        "11.16": np.array([2 * m * 0.9572 * _cosine / 18.015
                           for m in (1.008, 3.024)]),
        "11.18": np.array([1e6 * np.linalg.norm(_p) / (64 * 18.015),
                           0.5 * units.MV2_TO_EV * (_p @ _p)
                           / (64 * 18.015)]),
        "11.19": np.array([2 / math.sqrt(units.FORCE_TO_ACCEL),
                           2 * math.pi / 8.88 * 2
                           / math.sqrt(units.FORCE_TO_ACCEL)]),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 11.1: freedoms of one methanol molecule with its four bonds
    # to hydrogen held, and of 100 molecules with the drift removed.
    freedoms = None  # np.array([one, hundred])

    check(freedoms, TARGETS["11.1"], rtol=0, atol=0.5, name="freedoms")
    """),
    solution(r"""
    # SOLUTION 11.1. 3 x 6 - 4 per molecule; 100 of them less 3 for the drift.
    freedoms = np.array([3 * 6 - 4, 100 * (3 * 6 - 4) - 3])

    check(freedoms, TARGETS["11.1"], rtol=0, atol=0.5, name="freedoms");
    """),
    code(r"""
    # EXERCISE 11.2: eta, and the moves (Å along x) of atoms j and i, for
    # masses 2 (i) and 5 (j), r_ij = (1.5, 0, 0), provisional (1.6, 0.3, 0).
    one_bond = None  # np.array([eta, move_j, move_i])

    check(one_bond, TARGETS["11.2"], rtol=1e-3, name="eta and moves")
    """),
    solution(r"""
    # SOLUTION 11.2. The quadratic d^2 eta^2 + 2 (rt.r) eta + |rt|^2 - d^2,
    # its root nearer zero, and the moves eta m_r/m_j r and -eta m_r/m_i r.
    roots = np.roots([2.25, 2 * 2.4, 2.65 - 2.25])
    eta = roots[np.argmin(np.abs(roots))]
    m_r = 2 * 5 / (2 + 5)
    one_bond = np.array([eta, eta * m_r / 5 * 1.5, -eta * m_r / 2 * 1.5])

    check(one_bond, TARGETS["11.2"], rtol=1e-3, name="eta and moves");
    """),
    code(r"""
    # EXERCISE 11.3: Newton's method for y^2 = 3 from y = 2: y1, y2, y3.
    guesses = None  # np.array([y1, y2, y3])

    check(guesses, TARGETS["11.3"], rtol=1e-9, name="guesses")
    """),
    solution(r"""
    # SOLUTION 11.3. y -> y - (y^2 - 3)/(2y).
    y, guesses = 2.0, []
    for _ in range(3):
        y = y - (y * y - 3) / (2 * y)
        guesses.append(y)
    guesses = np.array(guesses)

    check(guesses, TARGETS["11.3"], rtol=1e-9, name="guesses");
    """),
    solution(r"""
    # EXERCISE 11.4 (show that): Newton's rule for y^2 = a and its error.
    # SymPy checks the two identities.
    yn, a = sp.symbols("y_n a", positive=True)
    nxt = yn - (yn**2 - a) / (2 * yn)
    assert sp.simplify(nxt - (yn + a / yn) / 2) == 0
    assert sp.simplify(nxt - sp.sqrt(a) - (yn - sp.sqrt(a))**2 / (2 * yn)) == 0
    print("11.4: checked")
    """),
    solution(r"""
    # EXERCISE 11.5 (show that): a SHAKE correction keeps the pair's centre
    # of mass, for any eta. SymPy checks it.
    mi, mj, et = sp.symbols("m_i m_j eta", positive=True)
    rij = sp.Matrix(sp.symbols("x y z"))
    mr = mi * mj / (mi + mj)
    shift = mi * (-et * mr / mi * rij) + mj * (et * mr / mj * rij)
    assert sp.simplify(shift) == sp.zeros(3, 1)
    print("11.5: checked")
    """),
    solution(r"""
    # EXERCISE 11.6 (show that): the kinetic energy of a pair splits into
    # centre-of-mass and relative parts. SymPy checks it in one dimension
    # per component.
    V, u = sp.symbols("V u")
    Mt = mi + mj
    vi, vj = V - mj / Mt * u, V + mi / Mt * u
    split = mi * vi**2 / 2 + mj * vj**2 / 2 - (Mt * V**2 / 2 + mr * u**2 / 2)
    assert sp.simplify(split) == 0
    m2 = np.array([16.0, 1.0])
    v2 = rng.normal(size=(2, 3))
    after, _ = constraints.rattle_velocities([[0, 0, 0], [0.96, 0, 0]], v2,
                                             m2, [[0, 1]], [0.96],
                                             tolerance=1e-15)
    ke = lambda v: 0.5 * np.sum(m2 * np.sum(v * v, 1))
    assert np.isclose(ke(v2) - ke(after),
                      0.5 * 16 / 17 * (v2[1, 0] - v2[0, 0]) ** 2)
    assert np.allclose(m2 @ v2, m2 @ after)
    print("11.6: checked")
    """),
    code(r"""
    # EXERCISE 11.7: r_H - r_O = (0.96, 0, 0) Å. zeta (1/fs), the corrected
    # x-velocities of O and H (Å/fs) and the kinetic energy removed
    # (amu Å^2/fs^2).
    oh = None  # np.array([zeta, vx_O, vx_H, removed])

    check(oh, TARGETS["11.7"], rtol=1e-4, name="O-H correction")
    """),
    solution(r"""
    # SOLUTION 11.7. mdlab's correction, and the energy before and after.
    vo, vh = np.array([0.001, 0, 0]), np.array([0.02, 0.01, 0])
    v, _ = constraints.rattle_velocities([[0, 0, 0], [0.96, 0, 0]],
                                         [vo, vh], [16.0, 1.0], [[0, 1]],
                                         [0.96], tolerance=1e-15)
    k = lambda vel: 0.5 * (16 * vel[0] @ vel[0] + 1 * vel[1] @ vel[1])
    zeta = -(vh - vo) @ np.array([0.96, 0, 0]) / 0.96**2
    oh = np.array([zeta, v[0, 0], v[1, 0], k([vo, vh]) - k(v)])

    check(oh, TARGETS["11.7"], rtol=1e-4, name="O-H correction");
    """),
    code(r"""
    # EXERCISE 11.8: freedoms of 216 rigid molecules with the drift removed,
    # and the kinetic energy (eV) at 12.93 meV per freedom.
    larger = None  # np.array([freedoms, energy])

    check(larger, TARGETS["11.8"], rtol=1e-3, name="larger box")
    """),
    solution(r"""
    # SOLUTION 11.8. Six per rigid molecule, less three.
    larger = np.array([6 * 216 - 3, (6 * 216 - 3) * 0.01293])

    check(larger, TARGETS["11.8"], rtol=1e-3, name="larger box");
    """),
    solution(r"""
    # EXERCISE 11.9 (show that): one inner step is velocity Verlet with the
    # summed force. The Section 11.4 runs of both must agree to rounding.
    assert np.abs(one["positions"][-1] - r_).max() < 1e-12
    assert np.abs(one["velocities"][-1] - v_).max() < 1e-12
    print("11.9: checked")
    """),
    code(r"""
    # EXERCISE 11.10: seconds per nanosecond for velocity Verlet at 0.41 fs
    # and for RESPA at 0.72 fs with three inner steps (24 ms, 0.04 ms).
    seconds = None  # np.array([verlet, respa])

    check(seconds, TARGETS["11.10"], rtol=1e-3, name="seconds")
    """),
    solution(r"""
    # SOLUTION 11.10. Calls per nanosecond times the time per call.
    seconds = np.array([1e6 / 0.41 * (0.024 + 0.04e-3),
                        1e6 / 0.72 * (0.024 + 3 * 0.04e-3)])

    check(seconds, TARGETS["11.10"], rtol=1e-3, name="seconds");
    """),
    solution(r"""
    # EXERCISE 11.11 (show that): M^2 = (tr M) M - (det M) I for rows (1, 2)
    # and (3, 4). NumPy checks it.
    Mn = np.array([[1, 2], [3, 4]])
    assert np.array_equal(Mn @ Mn, np.trace(Mn) * Mn
                          - round(np.linalg.det(Mn)) * np.eye(2))
    print("11.11: checked")
    """),
    solution(r"""
    # EXERCISE 11.12 (show that): just below theta = 2 pi the trace exceeds 2
    # when epsilon < 2 pi b. SymPy expands the trace to second order.
    eps_, b = sp.symbols("epsilon b", positive=True)
    trace = 2 * sp.cos(2 * sp.pi - eps_) - b * (2 * sp.pi - eps_) * sp.sin(
        2 * sp.pi - eps_)
    series = sp.series(trace, eps_, 0, 3).removeO()
    # the full trace, less the approximation, is the term theta ~ 2 pi drops
    assert sp.expand(series - (2 - eps_**2 + 2 * sp.pi * b * eps_)) == \
        -b * eps_**2
    print("11.12: checked")
    """),
    code(r"""
    # EXERCISE 11.13: the predicted band edges for omega_s = 0.3 omega_f, in
    # units of T_f: below half the period, then below the period.
    edges = None  # np.array([lo1, hi1, lo2, hi2])

    check(edges, TARGETS["11.13"], rtol=1e-3, name="edges")
    """),
    solution(r"""
    # SOLUTION 11.13. b = 0.09; (1 - b)/2 to 1/2, and 1 - b to 1.
    b_ = 0.3**2
    edges = np.array([(1 - b_) / 2, 0.5, 1 - b_, 1.0])

    check(edges, TARGETS["11.13"], rtol=1e-3, name="edges");
    """),
    md(r"""
    **Exercise 11.14** asks for an explanation: why halving the inner step
    at an outer step of 2 fs does not help. The cell below compares the two
    cached runs; the book's solution gives the reason.
    """),
    code(r"""
    for name in ("respa_2", "respa_2_inner0.125"):
        r = run(name)
        ratio = diagnostics.energy_fluctuation(r["potential"], r["kinetic"])
        print(f"{name}: std(E)/std(K) {ratio:.4f}")
    """),
    code(r"""
    # EXERCISE 11.15: hydrogen doubled at the oxygen's expense: the oxygen's
    # mass (amu), the O-H reduced mass (amu), the factor the stretch slows by.
    doubled = None  # np.array([m_O, m_r, factor])

    check(doubled, TARGETS["11.15"], rtol=1e-4, name="doubled")
    """),
    solution(r"""
    # SOLUTION 11.15.
    m_o = 15.999 - 2 * 1.008
    m_r_new = m_o * 2.016 / (m_o + 2.016)
    doubled = np.array([m_o, m_r_new,
                        math.sqrt(m_r_new / (15.999 * 1.008 / 17.007))])

    check(doubled, TARGETS["11.15"], rtol=1e-4, name="doubled");
    """),
    code(r"""
    # EXERCISE 11.16: the distance (Å) of the centre of mass from the oxygen,
    # ordinary hydrogens, then three times heavier.
    centre = None  # np.array([ordinary, heavy])

    check(centre, TARGETS["11.16"], rtol=1e-3, name="centre")
    """),
    solution(r"""
    # SOLUTION 11.16. Two hydrogens at 0.9572 cos(52.26 deg) along the
    # bisector, weighted by their mass over the molecule's.
    along = 0.9572 * math.cos(math.radians(52.26))
    centre = np.array([2 * m * along / 18.015 for m in (1.008, 3.024)])

    check(centre, TARGETS["11.16"], rtol=1e-3, name="centre");
    """),
    md(r"""
    **Exercise 11.17** asks which properties repartitioning changes; the
    widget of Section 11.6 shows the vibrations slowing, and the book's
    solution gives the reason the averages over arrangements stay the
    same.
    """),
    code(r"""
    # EXERCISE 11.18: how far the centre of mass moves in a nanosecond (Å),
    # and the kinetic energy of the drift (eV).
    drift = None  # np.array([distance, energy])

    check(drift, TARGETS["11.18"], rtol=1e-3, name="drift")
    """),
    solution(r"""
    # SOLUTION 11.18. |P|/M times 1e6 fs; |P|^2/(2M) in eV.
    p = np.array([0.2817, 0.2936, 0.4033])
    mass = 64 * 18.015
    drift = np.array([1e6 * np.linalg.norm(p) / mass,
                      0.5 * units.MV2_TO_EV * (p @ p) / mass])

    check(drift, TARGETS["11.18"], rtol=1e-3, name="drift");
    """),
    code(r"""
    # EXERCISE 11.19: a step written as 2 in ASE's units: its length (fs)
    # and omega dt for the O-H stretch of period 8.88 fs.
    ase_step = None  # np.array([step, omega_dt])

    check(ase_step, TARGETS["11.19"], rtol=1e-3, name="ASE step")
    """),
    solution(r"""
    # SOLUTION 11.19. ASE's time unit is 1/sqrt(FORCE_TO_ACCEL) fs.
    unit = 1 / math.sqrt(units.FORCE_TO_ACCEL)
    ase_step = np.array([2 * unit, 2 * math.pi / 8.88 * 2 * unit])

    check(ase_step, TARGETS["11.19"], rtol=1e-3, name="ASE step");
    """),
    md(r"""
    **Exercise 11.20** asks for a diagnosis. Compare with fault (b) in the
    widget of Section 11.8: a drift that the step does not change points
    to the forces.
    """),
]

NOTES = [md(r"""
    ## Working notes

    For one water model, record the step, constraint tolerance and energy
    fluctuation ratio. Change only the step and compare again. Use the
    result to justify which motion limits the step, then check whether
    changing the tolerance leads to the same diagnosis.
    """)]

CELLS = (SETUP + SHAKE + RATTLE + WATER + RESPA + RESONANCE + MASS + LIMITS
         + TRAJECTORY + EXERCISES + NOTES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "11_constraints.ipynb"))
