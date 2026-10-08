"""Write notebooks/10_code.ipynb, the companion to Chapter 10.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_10_code.py
    jupyter nbconvert --execute --to notebook --inplace notebooks/10_code.ipynb

Section 10.6 reads the runs cached by scripts/ch10_code/fig_run.py; run
that script first if data/ch10_code/runs.npz is missing. Sections 10.1,
10.2 and 10.4 to 10.6 end with a check of mdlab against ASE, an
independent library; 10.3 checks against known values and 10.7 against
the plain loop.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 10: Building a molecular dynamics code

    We build up a periodic simulation and check each piece before joining
    it to the velocity Verlet loop. The section numbers follow Chapter 10.
    Start with wrapping and nearest images, then compare the Ewald sum,
    neighbour lists and force kernels with their reference calculations.

    Run the cells in order from this notebook's directory. Later cells
    reuse the objects prepared above them. The longer trajectories come
    from `data/ch10_code/runs.npz`; the ASE comparisons use the same
    positions, forces and units so that a discrepancy can be traced.
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
    from ase import Atoms
    from ase.build import bulk
    from ase.calculators.lj import LennardJones
    from ase.cell import Cell
    from ase.geometry import find_mic, wrap_positions
    from ase.md.velocitydistribution import Stationary
    from ase.md.verlet import VelocityVerlet
    from ase.neighborlist import neighbor_list
    from ase.spacegroup import crystal
    from ase.units import fs as ASE_FS

    from mdlab import cell, ewald, io, kernels, md, neighbours, potentials
    from mdlab import units, viz
    from mdlab.exercise import check

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    A_G, C_G = 2.464, 6.711  # graphite's cell, Trucano and Chen (1975)
    GRAPHITE = np.array([[A_G, 0, 0], [-A_G / 2, A_G * math.sqrt(3) / 2, 0],
                         [0, 0, C_G]]).T  # lattice vectors as columns
    FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


    def supercell(fractional, h, n):
        # The atoms of a cell copied into an n[0] x n[1] x n[2] supercell.
        grid = np.array([[i, j, k] for i in range(n[0]) for j in range(n[1])
                         for k in range(n[2])], float)
        s = (np.asarray(fractional)[None, :, :] + grid[:, None, :]) / n
        big = h * np.asarray(n, float)  # column k times n[k]
        return cell.to_cartesian(s.reshape(-1, 3), big), big


    def outline(ax, h, **style):
        # The parallelogram of the cell's first two lattice vectors.
        corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, 0]])
        xy = cell.to_cartesian(corners, h)
        ax.plot(xy[:, 0], xy[:, 1], **style)
    """),
]

PERIODIC = [
    md(r"""
    ## 10.1 A box repeated in every direction

    Five atoms take random steps. Their unwrapped paths leave the cell;
    wrapped, they stay inside it.
    """),
    code(r"""
    def wrapping(shape="hexagonal", n_steps=200):
        h = 4 * GRAPHITE if shape == "hexagonal" else 8 * np.eye(3)
        rng = np.random.default_rng(1)
        start = cell.to_cartesian(rng.uniform(0, 1, (5, 3)), h)
        paths = start + np.cumsum(rng.normal(scale=0.4,
                                             size=(n_steps, 5, 3)), axis=0)
        wrapped = np.array([cell.wrap(p, h) for p in paths])
        fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
        outline(ax, h, color="black", lw=1)
        for k, colour in enumerate(viz.CYCLE[:5]):
            ax.plot(paths[:, k, 0], paths[:, k, 1], lw=0.8, alpha=0.5,
                    color=colour)
            ax.plot(wrapped[:, k, 0], wrapped[:, k, 1], ".", ms=2,
                    color=colour)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ / Å")
        ax.set_ylabel("$y$ / Å")
        plt.show()
        s = cell.to_fractional(wrapped.reshape(-1, 3), h)
        print(f"wrapped fractional coordinates between {s.min():.3f} and "
              f"{s.max():.3f}")


    widgets.interact(
        wrapping,
        shape=widgets.Dropdown(options=["hexagonal", "cubic"]),
        n_steps=widgets.IntSlider(200, min=10, max=2000, step=10, **SLOW),
    );
    """),
    md(r"""
    The faint unwrapped paths wander out of the
    parallelogram; the dots, wrapped, stay inside it, an atom leaving
    through one face reappearing at the opposite one. Every fractional
    coordinate stays between 0 and 1.
    """),
    code(r"""
    # Checked against ASE: volume, reciprocal vectors and wrapping.
    rng = np.random.default_rng(2)
    ase_cell = Cell(GRAPHITE.T)  # ASE holds the lattice vectors as rows
    r = rng.normal(scale=15, size=(500, 3))
    print("volume:", cell.cell_volume(GRAPHITE), "ASE", ase_cell.volume)
    print("reciprocal vectors differ by",
          np.abs(cell.reciprocal_vectors(GRAPHITE)
                 - 2 * np.pi * ase_cell.reciprocal()).max())
    print("wrapped positions differ by",
          np.abs(cell.wrap(r, GRAPHITE)
                 - wrap_positions(r, GRAPHITE.T, eps=0)).max(), "Å")
    """),
]

IMAGE = [
    md(r"""
    ## 10.2 The nearest image

    Choose the length and angle of a separation in graphite's layer plane.
    Compare the image returned by rounding with the nearest image found
    by searching neighbouring cells.
    """),
    code(r"""
    def image(angle=30, length=1.3):
        theta = math.radians(angle)
        d = np.array([[length * math.cos(theta), length * math.sin(theta),
                       0.0]])
        rounded = cell.minimum_image(d, GRAPHITE)[0]
        nearest = cell.nearest_image(d, GRAPHITE)[0]
        half = 0.5 * cell.perpendicular_widths(GRAPHITE).min()
        fig, ax = plt.subplots(figsize=(4.5, 4), dpi=80)
        corners = cell.to_cartesian(np.array([[-.5, -.5, 0], [.5, -.5, 0],
                                              [.5, .5, 0], [-.5, .5, 0],
                                              [-.5, -.5, 0]]), GRAPHITE)
        ax.plot(corners[:, 0], corners[:, 1], color="black", lw=1)
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(half * np.cos(t), half * np.sin(t), **viz.REFERENCE_STYLE)
        for vec, colour, label in ((rounded, viz.OXBLOOD, "rounded"),
                                   (nearest, viz.ACCENT, "nearest")):
            ax.annotate("", xy=vec[:2], xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=colour))
            ax.plot([], [], color=colour, label=label)
        ax.set_aspect("equal")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("$x$ / Å")
        ax.set_ylabel("$y$ / Å")
        ax.legend(loc="upper right", fontsize=8)
        plt.show()
        print(f"rounded {np.linalg.norm(rounded):.3f} Å, nearest "
              f"{np.linalg.norm(nearest):.3f} Å, half the width {half:.3f} Å")


    widgets.interact(
        image,
        angle=widgets.FloatSlider(30, min=-180, max=180, step=1, **SLOW),
        length=widgets.FloatSlider(1.3, min=0.1, max=2.5, step=0.01,
                                   **SLOW),
    );
    """),
    md(r"""
    Whenever the nearest copy is shorter than the
    dashed circle of radius $w_{\min}/2$ = 1.067 Å, the two arrows coincide.
    Beyond it they can part, as at the starting 30° and 1.3 Å, and at
    $-30^\circ$ with a length of 2: rounding then returns a longer copy,
    whose arrow ends in one of the parallelogram's two acute corners,
    towards $-30^\circ$ or $150^\circ$.
    """),
    code(r"""
    # Checked against ASE: find_mic searches for the nearest copy.
    rng = np.random.default_rng(3)
    d = cell.to_cartesian(rng.uniform(-3, 3, size=(5000, 3)), GRAPHITE)
    ase_d, ase_len = find_mic(d, GRAPHITE.T, pbc=True)
    ours = np.linalg.norm(cell.nearest_image(d, GRAPHITE), axis=1)
    print("nearest-copy lengths differ from ASE's by",
          np.abs(ours - ase_len).max(), "Å")
    short = ase_len < 0.5 * cell.perpendicular_widths(GRAPHITE).min()
    print("rounding equals ASE's copy for all", short.sum(), "short ones:",
          np.allclose(cell.minimum_image(d, GRAPHITE)[short], ase_d[short]))
    """),
]

EWALD = [
    md(r"""
    ## 10.3 Charges in a periodic box: the Ewald sum

    Rock salt in its cubic cell of side 2 ($d = 1$, $k_e = 1$). The slider
    moves the work between the real-space and reciprocal-space sums.
    """),
    code(r"""
    SALT = np.vstack([2 * FCC, 2 * FCC + [1.0, 0.0, 0.0]])
    SALT_Q = np.array([1.0] * 4 + [-1.0] * 4)


    def split(alpha=2.0):
        e, f, parts = ewald.ewald_energy_forces(SALT_Q, SALT, 2 * np.eye(3),
                                                alpha=alpha, coulomb=1.0)
        names = ["real", "reciprocal", "self"]
        fig, ax = plt.subplots(figsize=(4.5, 2.6), dpi=80)
        ax.bar(names + ["sum"], [parts[k] / 4 for k in names] + [e / 4],
               color=[viz.ACCENT, viz.OCHRE, viz.OXBLOOD, "black"])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("energy per ion pair")
        plt.show()
        print(f"Madelung constant {-e / 4:.10f}; largest force "
              f"{np.abs(f).max():.1e}")


    widgets.interact(
        split, alpha=widgets.FloatSlider(2.0, min=0.5, max=6.0, step=0.1,
                                         **SLOW)
    );
    """),
    md(r"""
    As $\alpha$ rises the real-space part shrinks to
    nothing, the reciprocal-space part grows and the self part grows more
    negative, and their sum stays at $-1.7475646$ per ion pair. Every
    force is zero: each ion sits at a centre of symmetry.
    """),
    md(r"""
    ASE has no Ewald sum for point charges, so the checks here are the
    Madelung constant of Chapter 8, the same energy in other cells, and the
    neutralising background of a charged cell.
    """),
    code(r"""
    big, _ = supercell(SALT / 2, 2 * np.eye(3), (2, 2, 2))
    e8 = ewald.ewald_energy_forces(np.tile(SALT_Q, 8), big, 4 * np.eye(3),
                                   coulomb=1.0)[0]
    prim = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float).T
    ep = ewald.ewald_energy_forces([1.0, -1.0], [[0, 0, 0], [1, 0, 0]],
                                   prim, coulomb=1.0)[0]
    print(f"per ion pair: supercell {-e8 / 32:.10f}, primitive cell "
          f"{-ep:.10f}, Evjen 1.747565")
    rng = np.random.default_rng(4)
    r5, q5 = rng.uniform(0, 6, (5, 3)), np.array([1, 1, -1, .5, .2])
    for a in (0.5, 0.8, 1.2):
        e, _, p = ewald.ewald_energy_forces(q5, r5, 6 * np.eye(3), alpha=a)
        print(f"alpha {a}: charged cell {e:.10f} eV, without the "
              f"background {e - p['background']:.6f} eV")
    """),
]

NEIGHBOURS = [
    md(r"""
    ## 10.4 Neighbour lists

    Place atoms randomly at $\rho = 0.8\sigma^{-3}$ with $r_c = 2.5\sigma$.
    Time the search for interacting pairs, first by testing all pairs and
    then with a cell list. The density and cutoff stay fixed as $N$ grows.
    """),
    code(r"""
    def race(n=2000):
        side = (n / 0.8) ** (1 / 3)
        r = np.random.default_rng(5).uniform(0, side, (n, 3))
        h = side * np.eye(3)
        for name in ("all_pairs", "cell_list_pairs"):
            start = time.perf_counter()
            i, j, _, _ = getattr(neighbours, name)(r, h, 2.5)
            print(f"{name}: {time.perf_counter() - start:.3f} s, "
                  f"{len(i) / n:.1f} pairs per atom")


    widgets.interact(
        race, n=widgets.IntSlider(2000, min=500, max=8000, step=500, **SLOW)
    );
    """),
    md(r"""
    About 26 pairs per atom at every $N$. Doubling
    $N$ roughly quadruples the time of all pairs and doubles that of the
    cell list; they cost about the same near $N = 1000$.
    """),
    code(r"""
    # Checked against ASE: neighbor_list lists every pair both ways.
    rng = np.random.default_rng(6)
    h = 4.5 * GRAPHITE
    r = cell.to_cartesian(rng.uniform(0, 1, (400, 3)), h)
    r_cut = 0.3 * cell.perpendicular_widths(h).min()
    i, j, _, dist = neighbours.cell_list_pairs(r, h, r_cut)
    atoms = Atoms("C" * len(r), positions=r, cell=h.T, pbc=True)
    ai, aj, ad = neighbor_list("ijd", atoms, r_cut)
    keep = ai < aj
    print(len(i), "pairs from mdlab,", keep.sum(), "from ASE; same pairs:",
          set(zip(i.tolist(), j.tolist())) == set(zip(ai[keep].tolist(),
                                                      aj[keep].tolist())))
    """),
]

START = [
    md(r"""
    ## 10.5 Starting a simulation

    Build supercells of argon and graphite, then give the atoms random
    velocities with the centre-of-mass drift removed.
    """),
    code(r"""
    a0 = 1.54916 * 3.4  # the model's own spacing, Å (Section 10.5)
    r_ar, h_ar = supercell(FCC, a0 * np.eye(3), (5, 5, 5))
    print(f"{len(r_ar)} argon atoms in a cube of side {h_ar[0, 0]:.2f} Å")
    graphite_s = np.array([[0, 0, .25], [0, 0, .75], [1 / 3, 2 / 3, .25],
                           [2 / 3, 1 / 3, .75]])
    r_c, h_c = supercell(graphite_s, GRAPHITE, (10, 10, 3))
    print(len(r_c), "carbon atoms; widths",
          cell.perpendicular_widths(h_c).round(2), "Å")
    m = np.full(len(r_ar), 39.948)
    v = md.starting_velocities(m, 0.02 * len(r_ar),
                               np.random.default_rng(7))
    print("total momentum", np.abs(m @ v).max(), "; kinetic energy per atom",
          0.5 * units.MV2_TO_EV * float(m @ (v * v).sum(1)) / len(m), "eV")
    """),
    code(r"""
    # Checked against ASE: its builders and its drift removal.
    ar = bulk("Ar", "fcc", a=a0, cubic=True).repeat((5, 5, 5))
    ours = np.round(cell.to_fractional(r_ar, h_ar) % 1, 8)
    theirs = np.round(ar.get_scaled_positions() % 1, 8)
    print("argon: same positions:",
          set(map(tuple, ours)) == set(map(tuple, theirs)))
    gr = crystal(["C", "C"], [(0, 0, .25), (1 / 3, 2 / 3, .25)],
                 spacegroup=194, cellpar=[A_G, A_G, C_G, 90, 90, 120])
    print("graphite from its space group:\n",
          np.round(gr.get_scaled_positions(), 4))
    atoms = Atoms("Ar" * len(m), positions=r_ar)
    atoms.set_masses(m)
    atoms.set_velocities(v + 0.01)  # add a drift, then remove it
    Stationary(atoms, preserve_temperature=False)
    print("drift removal differs from ASE's by",
          np.abs(atoms.get_velocities() - v).max())
    """),
]

LOOP = [
    md(r"""
    ## 10.6 The loop

    The two runs of Figure 10.4, from the cache written by
    `scripts/ch10_code/fig_run.py`.
    """),
    code(r"""
    runs = np.load(Path("..") / "data" / "ch10_code" / "runs.npz")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 2.8), dpi=80)
    t = runs["argon_times"] / 1000
    left.plot(t, runs["argon_kinetic"] / 500, label="kinetic")
    left.plot(t, (runs["argon_potential"] - runs["argon_potential"][0])
              / 500, label="potential - U(0)")
    left.set_xlabel("time / ps")
    left.set_ylabel("eV per atom")
    left.legend(fontsize=7)
    for dt in (0.002, 0.004):
        e = runs[f"salt_{dt}_total"]
        right.plot(runs[f"salt_{dt}_times"], e - e[0], label=f"dt = {dt}")
    right.set_xlabel(r"time / $\tau$")
    right.set_ylabel(r"$E - E(0)$ / $\varepsilon$")
    fig.subplots_adjust(wspace=0.4)
    right.legend(fontsize=7)
    plt.show()
    spreads = [np.ptp(runs[f"salt_{dt}_total"]) for dt in (0.002, 0.004)]
    print(f"salt: spread ratio for a doubled step "
          f"{spreads[1] / spreads[0]:.2f}")
    """),
    code(r"""
    # Checked against ASE: the same 256 atoms through ASE's VelocityVerlet,
    # with ASE's own femtosecond and with mdlab's.
    eps, sig, rc = 0.01034, 3.4, 8.5
    r0, h0 = supercell(FCC, 5.26 * np.eye(3), (4, 4, 4))
    m0 = np.full(len(r0), 39.948)
    v0 = md.starting_velocities(m0, 0.02 * len(r0), np.random.default_rng(8))
    pair = potentials.with_cutoff(
        lambda x: potentials.lennard_jones(x, eps, sig), rc, "shift")
    out = md.run(md.PairModel(pair, h0, rc, 1.0), m0, r0, v0, h0, 5.0, 200,
                 every=200, path="argon.extxyz", symbols=["Ar"] * len(r0))
    for name, fs in (("ASE's fs", ASE_FS),
                     ("mdlab's fs", math.sqrt(units.FORCE_TO_ACCEL))):
        atoms = Atoms("Ar" * len(r0), positions=r0, cell=h0.T, pbc=True)
        atoms.set_masses(m0)
        atoms.set_velocities(v0 / fs)
        atoms.calc = LennardJones(epsilon=eps, sigma=sig, rc=rc,
                                  smooth=False)
        VelocityVerlet(atoms, timestep=5.0 * fs).run(200)
        gap = np.abs(atoms.get_positions() - out["positions"][-1]).max()
        print(f"{name}: positions after 200 steps differ by {gap:.1e} Å")
    """),
    md(r"""
    With ASE's femtosecond the positions differ by
    about $10^{-8}$ Å, the effect of two editions of the physical
    constants; with the same femtosecond they agree to about $10^{-14}$ Å,
    the rounding of the arithmetic.
    """),
    code(r"""
    # Checked against ASE: it reads the file mdlab wrote.
    from ase.io import read

    frames = read("argon.extxyz", index=":")
    print(len(frames), "frames;", len(frames[-1]), "atoms; cell matches:",
          np.allclose(frames[-1].cell.array, h0.T))
    print("potential energy read by ASE", frames[-1].get_potential_energy(),
          "eV; written by mdlab", out["potential"][-1], "eV")
    """),
]

SPEED = [
    md(r"""
    ## 10.7 Making it fast

    One call of each version of the force kernel, on the same atoms. The
    plain loop is skipped above 2000 atoms. Python, NumPy and Numba run
    by default. To include Fortran, install a Fortran compiler, Meson and
    Ninja, then set `RUN_FORTRAN = True` in the next cell and rerun it.
    """),
    code(r"""
    RUN_FORTRAN = False
    if RUN_FORTRAN:
        kernels.build_fortran()


    def kernel_race(n=2000):
        k = int(np.ceil(n ** (1 / 3)))
        side = (n / 0.8) ** (1 / 3)
        grid = np.array([[a, b, c] for a in range(k) for b in range(k)
                         for c in range(k)], float)[:n]
        r = grid / k * side + np.random.default_rng(9).normal(0, 0.05,
                                                              (n, 3))
        i, j, _, _ = neighbours.cell_list_pairs(r, side * np.eye(3), 2.8)
        args = (r, np.full(3, side), i, j, 1.0, 1.0, 2.5)
        reference = kernels.lj_numpy(*args)
        versions = ["lj_loop", "lj_numpy", "lj_numba"]
        if RUN_FORTRAN:
            versions.append("lj_fortran")
        for name in versions:
            if name == "lj_loop" and n > 2000:
                continue
            getattr(kernels, name)(*args)  # compile on first use
            start = time.perf_counter()
            e, f = getattr(kernels, name)(*args)
            took = time.perf_counter() - start
            print(f"{name:11s} {1e3 * took:8.3f} ms; energy differs by "
                  f"{abs(e - reference[0]) / abs(reference[0]):.0e}")


    widgets.interact(
        kernel_race,
        n=widgets.IntSlider(2000, min=500, max=16000, step=500, **SLOW),
    );
    """),
    md(r"""
    The book's timings put NumPy about thirty times ahead of the plain
    loop and Numba and Fortran another ten to twenty times ahead. The
    ratios on this machine may differ. Check that the energies still
    agree to about a part in $10^{12}$ before comparing the timings.
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
    derivations 10.6 to 10.8, 10.10 to 10.12 and 10.17 and 10.18 are
    checked by SymPy or NumPy.
    """),
    hidden(r"""
    # Reference values, calculated from the exercise data.
    _n = 1
    while 1 - (1 - 2 / _n) ** 3 >= 0.01:
        _n += 1
    _prim = 5.267 * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]).T
    _reach = math.sqrt(-math.log(1e-6))
    _alpha = _reach / 15
    _v = math.sqrt(2 * 0.0104 / (39.948 * units.MV2_TO_EV))
    TARGETS = {
        "10.1": _n**3,
        "10.2": np.array([1.0, 1.0, 0.25]),
        "10.3": np.array([cell.cell_volume(GRAPHITE),
                          *cell.perpendicular_widths(GRAPHITE)[[0, 2]]]),
        "10.4": np.array([cell.cell_volume(_prim),
                          cell.perpendicular_widths(_prim)[0],
                          0.5 * cell.perpendicular_widths(_prim)[0]]),
        "10.5": cell.minimum_image([[1.8, -2.9, 0.1]], np.diag([2, 3, 4]))[0],
        "10.9": np.linalg.norm(cell.reciprocal_vectors(GRAPHITE),
                               axis=1)[[0, 2]],
        "10.13": np.array([_alpha, 2 * _alpha * _reach]),
        "10.14": 4 / 3 * math.pi * 10**3 * 0.0334,
        "10.15": np.array([math.floor(s / 9.5) ** 3 for s in (40, 30)]),
        "10.15 all pairs": [s for s in (40, 30, 25)
                            if math.floor(s / 9.5) < 3][0],
        "10.16": 0.5 / (10 * _v),
        "10.19": 0.0031 * 1000 * 4.6e-9,
        "10.2 wrapped": np.array([0.0, 0.0, C_G / 4]),
        "10.13 waves": 2896,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 10.1: the number of atoms N = n^3 at which fewer than 1% are
    # on the surface, from the exact fraction 1 - (1 - 2/n)^3, not the
    # estimate 6/n.
    n_atoms = None

    check(n_atoms, TARGETS["10.1"], name="N")
    """),
    solution(r"""
    # SOLUTION 10.1. (1 - 2/n)^3 > 0.99 first at n = 598.
    n = 1
    while 1 - (1 - 2 / n) ** 3 >= 0.01:
        n += 1
    n_atoms = n**3

    check(n_atoms, TARGETS["10.1"], name="N");
    """),
    code(r"""
    # EXERCISE 10.2: the fractional coordinates of (a/2, sqrt(3)a/2, c/4),
    # and the position (Å) to which wrapping moves it.
    s_graphite = None
    wrapped = None

    check(s_graphite, TARGETS["10.2"], name="s")
    check(wrapped, TARGETS["10.2 wrapped"], atol=1e-3, name="wrapped")
    """),
    solution(r"""
    # SOLUTION 10.2. Solve h s = r.
    point = [[A_G / 2, A_G * math.sqrt(3) / 2, C_G / 4]]
    s_graphite = cell.to_fractional(point, GRAPHITE)[0]
    wrapped = cell.wrap(point, GRAPHITE)[0]

    check(s_graphite, TARGETS["10.2"], name="s")
    check(wrapped, TARGETS["10.2 wrapped"], atol=1e-3, name="wrapped");
    """),
    code(r"""
    # EXERCISE 10.3: graphite's volume (Å³), in-plane width and width
    # across the layers (Å).
    graphite_cell = None

    check(graphite_cell, TARGETS["10.3"], name="V, widths")
    """),
    solution(r"""
    # SOLUTION 10.3. V = (sqrt3/2) a^2 c; widths (sqrt3/2) a and c.
    graphite_cell = np.array([math.sqrt(3) / 2 * A_G**2 * C_G,
                              math.sqrt(3) / 2 * A_G, C_G])

    check(graphite_cell, TARGETS["10.3"], name="V, widths");
    """),
    code(r"""
    # EXERCISE 10.4: the primitive fcc cell for a0 = 5.267 Å: volume (Å³),
    # width (Å) and largest cutoff (Å).
    primitive = None

    check(primitive, TARGETS["10.4"], name="V, width, cutoff")
    """),
    solution(r"""
    # SOLUTION 10.4. V = a0^3/4, width a0/sqrt3, cutoff half of it.
    a0 = 5.267
    primitive = np.array([a0**3 / 4, a0 / math.sqrt(3),
                          a0 / (2 * math.sqrt(3))])

    check(primitive, TARGETS["10.4"], name="V, width, cutoff");
    """),
    code(r"""
    # EXERCISE 10.5: the nearest copy (Å) of (1.8, -2.9, 0.1) in the
    # 2 x 3 x 4 Å cell.
    copy = None

    check(copy, TARGETS["10.5"], name="copy")
    """),
    solution(r"""
    # SOLUTION 10.5. Round s = (0.9, -0.967, 0.025) to (1, -1, 0).
    copy = np.array([1.8, -2.9, 0.1]) - np.array([2, -3, 0])

    check(copy, TARGETS["10.5"], name="copy");
    """),
    solution(r"""
    # SOLUTIONS 10.6 to 10.8. A normalised Gaussian, erf near zero, and
    # waves that do not mix.
    x, x0 = sp.symbols("x x_0", real=True)
    s_ = sp.symbols("s", positive=True)
    assert sp.simplify(sp.integrate(sp.exp(-(x - x0)**2 / (2 * s_**2)),
                                    (x, -sp.oo, sp.oo))
                       - s_ * sp.sqrt(2 * sp.pi)) == 0
    u, r_, al = sp.symbols("u r alpha", positive=True)
    series = sp.series(sp.erf(x), x, 0, 6).removeO()
    assert sp.expand(series - 2 / sp.sqrt(sp.pi) * (x - x**3 / 3
                                                   + x**5 / 10)) == 0
    assert sp.limit(sp.erf(al * r_) / r_, r_, 0) == 2 * al / sp.sqrt(sp.pi)
    L = sp.symbols("L", positive=True)
    for m1 in range(4):
        for m2 in range(4):
            if m1 == m2 == 0:
                continue
            avg = sp.integrate(sp.cos(2 * sp.pi * m1 * x / L)
                               * sp.cos(2 * sp.pi * m2 * x / L),
                               (x, 0, L)) / L
            want = sp.Rational(1, 2) if m1 == m2 else 0
            assert sp.simplify(avg - want) == 0
    print("normalised Gaussian, erf series and its limit, orthogonality: "
          "checked")
    """),
    code(r"""
    # EXERCISE 10.9: the lengths (1/Å) of graphite's a* and c*.
    lengths = None

    check(lengths, TARGETS["10.9"], name="lengths")
    """),
    solution(r"""
    # SOLUTION 10.9. |a*| = 4 pi/(sqrt3 a), |c*| = 2 pi/c.
    lengths = np.array([4 * math.pi / (math.sqrt(3) * A_G), 2 * math.pi / C_G])
    print(np.round(cell.reciprocal_vectors(GRAPHITE) * A_G / (2 * math.pi), 4))

    check(lengths, TARGETS["10.9"], name="lengths");
    """),
    solution(r"""
    # SOLUTIONS 10.10 and 10.11. The real-space slope by SymPy, the
    # reciprocal force from its formula, and the waves of rock salt.
    r_, al = sp.symbols("r alpha", positive=True)
    slope = sp.diff(sp.erfc(al * r_) / r_, r_)
    claim = -(sp.erfc(al * r_) / r_
              + 2 * al / sp.sqrt(sp.pi) * sp.exp(-al**2 * r_**2)) / r_
    assert sp.simplify(slope - claim) == 0
    rng = np.random.default_rng(10)
    pos, q = rng.uniform(0, 5, (4, 3)), np.array([1.0, -1.0, 0.5, -0.5])
    alpha, side = 1.0, 5.0
    g_c = 2 * alpha * math.sqrt(-math.log(1e-10))  # mdlab's default
    m = np.arange(-10, 11)
    g = 2 * np.pi / side * np.stack(np.meshgrid(m, m, m), -1).reshape(-1, 3)
    g2 = np.sum(g * g, axis=1)
    keep = (g2 > 0) & (g2 <= g_c**2)
    g, g2 = g[keep], g2[keep]
    weight = np.exp(-g2 / (4 * alpha**2)) / g2
    phase = pos @ g.T
    c_sum, s_sum = q @ np.cos(phase), q @ np.sin(phase)
    bracket = c_sum * np.sin(phase) - s_sum * np.cos(phase)
    formula = (4 * np.pi * units.COULOMB / side**3 * q[:, None]
               * ((weight * bracket) @ g))
    numeric = potentials.finite_difference_forces(
        lambda x: ewald.ewald_energy_forces(
            q, x, side * np.eye(3), alpha=alpha)[2]["reciprocal"], pos, 1e-5)
    assert np.allclose(formula, numeric, atol=1e-7)
    for m1 in range(-3, 4):
        for m2 in range(-3, 4):
            for m3 in range(-3, 4):
                g = np.pi * np.array([m1, m2, m3], float)
                c_sum = SALT_Q @ np.cos(SALT @ g)
                s_sum = SALT_Q @ np.sin(SALT @ g)
                odd = m1 % 2 and m2 % 2 and m3 % 2
                assert abs(s_sum) < 1e-12
                assert abs(c_sum - (8 if odd else 0)) < 1e-12
    print("real-space slope, reciprocal force, rock-salt waves: checked")
    """),
    solution(r"""
    # SOLUTION 10.12. The cheapest alpha and the cost N^(3/2).
    A, B, N, V, al = sp.symbols("A B N V alpha", positive=True)
    rho = N / V
    cost = A * N * rho / al**3 + B * N * V * al**3
    best = sp.solve(sp.diff(cost, al), al)[0]
    assert sp.simplify(best**6 - A * N / (B * V**2)) == 0
    assert sp.simplify(cost.subs(al, best)
                       - 2 * sp.sqrt(A * B) * N**sp.Rational(3, 2)) == 0
    print("cost at the best alpha:", sp.simplify(cost.subs(al, best)))
    """),
    code(r"""
    # EXERCISE 10.13: alpha (1/Å) and G_c (1/Å) for a 30 Å cube,
    # eps_E = 1e-6, real-space cutoff 15 Å, and roughly how many waves.
    ewald_parameters = None
    waves = None

    check(ewald_parameters, TARGETS["10.13"], name="alpha, G_c")
    check(waves, TARGETS["10.13 waves"], rtol=0.05, name="waves")
    """),
    solution(r"""
    # SOLUTION 10.13. alpha = sqrt(-ln eps)/15, G_c = 2 alpha sqrt(-ln eps).
    reach = math.sqrt(-math.log(1e-6))
    ewald_parameters = np.array([reach / 15, 2 * reach**2 / 15])
    m = np.arange(-9, 10)
    grid = np.stack(np.meshgrid(m, m, m), -1).reshape(-1, 3)
    g = 2 * math.pi / 30 * np.linalg.norm(grid, axis=1)
    waves = int(np.sum((g > 0) & (g <= ewald_parameters[1])))
    print("waves within G_c:", waves, "; sphere estimate",
          round(4 / 3 * math.pi * ewald_parameters[1]**3
                / (2 * math.pi / 30)**3))

    check(ewald_parameters, TARGETS["10.13"], name="alpha, G_c")
    check(waves, TARGETS["10.13 waves"], rtol=0.05, name="waves");
    """),
    code(r"""
    # EXERCISE 10.14: molecules within 10 Å in water at 0.0334 per Å³.
    in_water = None

    check(in_water, TARGETS["10.14"], rtol=1e-2, name="molecules")
    """),
    solution(r"""
    # SOLUTION 10.14. (4/3) pi r^3 rho.
    in_water = 4 / 3 * math.pi * 10**3 * 0.0334

    check(in_water, TARGETS["10.14"], rtol=1e-2, name="molecules");
    """),
    code(r"""
    # EXERCISE 10.15: the numbers of small cells for cubes of side 40 and
    # 30 Å with r_c + skin = 9.5 Å, and the side (Å) of the cube, of 40, 30
    # and 25 Å, in which mdlab tests all pairs instead.
    small_cells = None
    all_pairs_side = None

    check(small_cells, TARGETS["10.15"], name="small cells")
    check(all_pairs_side, TARGETS["10.15 all pairs"], name="side")
    """),
    solution(r"""
    # SOLUTION 10.15. floor(side/9.5) cubed; 25 Å gives 2 a side.
    small_cells = np.array([4**3, 3**3])
    all_pairs_side = 25

    check(small_cells, TARGETS["10.15"], name="small cells")
    check(all_pairs_side, TARGETS["10.15 all pairs"], name="side");
    """),
    code(r"""
    # EXERCISE 10.16: steps of 10 fs for an argon atom at the rms speed to
    # move half a 1 Å skin.
    rebuild_steps = None

    check(rebuild_steps, TARGETS["10.16"], rtol=0.05, name="steps")
    """),
    solution(r"""
    # SOLUTION 10.16. v = sqrt(2K/m) with K in eV, m in amu.
    v_rms = math.sqrt(2 * 0.0104 / (39.948 * units.MV2_TO_EV))
    rebuild_steps = 0.5 / (10 * v_rms)

    check(rebuild_steps, TARGETS["10.16"], rtol=0.05, name="steps");
    """),
    solution(r"""
    # SOLUTIONS 10.17 and 10.18. No drift after subtracting P/M; half the
    # energy on average.
    ms = sp.symbols("m0:4", positive=True)
    vs = sp.symbols("v0:4")
    P, M = sum(a * b for a, b in zip(ms, vs)), sum(ms)
    assert sp.simplify(sum(a * (b - P / M) for a, b in zip(ms, vs))) == 0
    t, w, A_, mass = sp.symbols("t omega A m", positive=True)
    x_t = A_ * sp.sin(w * t)  # started at the minimum
    K = mass * sp.diff(x_t, t)**2 / 2
    average = sp.integrate(K, (t, 0, 2 * sp.pi / w)) / (2 * sp.pi / w)
    assert sp.simplify(average - K.subs(t, 0) / 2) == 0
    print("no drift; average kinetic energy K0/2: checked")
    """),
    code(r"""
    # EXERCISE 10.19: the difference in position (Å) after 1000 fs at
    # 0.0031 Å/fs from two femtoseconds 4.6e-9 apart.
    shift = None

    check(shift, TARGETS["10.19"], rtol=0.05, name="shift")
    """),
    solution(r"""
    # SOLUTION 10.19. speed times the difference in time.
    shift = 0.0031 * 1000 * 4.6e-9

    check(shift, TARGETS["10.19"], rtol=0.05, name="shift");
    """),
]

NOTES = [md(r"""
    ## Working notes

    Choose one boundary-crossing example and one force-kernel comparison.
    Record the cell, cutoff and atom count with the result. Explain why
    wrapping changes the stored coordinates without changing the physical
    separation, and separate agreement of the forces from their timing.
    """)]

CELLS = (SETUP + PERIODIC + IMAGE + EWALD + NEIGHBOURS + START + LOOP + SPEED
         + EXERCISES + NOTES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "10_code.ipynb"))
