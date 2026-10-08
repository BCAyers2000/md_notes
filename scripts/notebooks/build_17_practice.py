"""Write notebooks/17_practice.ipynb, the companion to Chapter 17.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_17_practice.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/17_practice.ipynb

The runs are read from data/ch17_practice/runs/ (written by the scripts
of scripts/ch17_practice/, the long ones on another machine). The short ASE check runs here; the LAMMPS force check and MACE scan
are optional.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 17: Molecular dynamics in practice

    Compare the same argon model across codes before interpreting the
    saved lithium-in-graphite runs. The unit conversions and atom ordering
    matter as much as the integrator settings. The section numbers follow
    Chapter 17; run the cells in order.

    The stored runs supply the longer calculations. Set
    `RUN_EXTERNAL_CALCULATORS = True` only when deliberately repeating the
    optional LAMMPS force check and MACE graphite scan in a prepared
    environment; the default uses the saved comparison.
    """),
    code(r"""
    %matplotlib inline
    import math
    import os
    import sys
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from ase import Atoms
    from ase.io import read, write as ase_write

    from mdlab import diagnostics, io, md, statmech, units, viz
    from mdlab.cell import wrap
    from mdlab.analysis import hops, stats, transport
    from mdlab.exercise import check

    os.environ.setdefault("FI_PROVIDER", "tcp")  # MPICH under LAMMPS
    sys.path.insert(0, str(Path("..") / "scripts" / "ch17_practice"))
    import ch17  # the shared set-up of the chapter's scripts

    viz.use_style()
    RNG = np.random.default_rng(17)
    SLOW = dict(continuous_update=False)  # redraw only on release
    KB = units.KB
    RUN_EXTERNAL_CALCULATORS = False
    RUNS = ch17.RUNS
    R0, V0, H0, M0 = ch17.liquid_start(0)
    V0 = V0 - np.average(V0, axis=0, weights=M0)
    """),
]

ASE = [
    md(r"""
    ## 17.1 Molecular dynamics with ASE

    The same liquid argon in `mdlab` and in ASE, the second through
    `io.mdlab_calculator`. ASE's time unit is 10.18 fs, so the step and the
    velocities are converted on the way in.
    """),
    code(r"""
    from ase.md.verlet import VelocityVerlet

    atoms = Atoms(["Ar"] * len(M0), positions=R0, cell=H0.T, pbc=True,
                  masses=M0)
    atoms.set_velocities(V0 * np.sqrt(1 / units.FORCE_TO_ACCEL))
    atoms.calc = io.mdlab_calculator(ch17.argon_model(H0))
    VelocityVerlet(atoms, timestep=10.0 / ch17.ASE_FS).run(50)
    ours = md.run(ch17.argon_model(H0), M0, R0, V0, H0, 10.0, 50)
    gap = np.abs(atoms.get_positions() - ours["positions"][-1]).max()
    print(f"after 50 steps of 10 fs the two codes differ by {gap:.1e} Å")
    """),
    code(r"""
    NVE = {c: np.load(RUNS / f"argon_{c}_nve.npz")
           for c in ("mdlab", "ase", "lammps")}


    def divergence(upto_ps=5.0):
        fig, ax = plt.subplots(figsize=(6, 3))
        ref = NVE["mdlab"]["positions"]
        for c, colour in (("ase", viz.OCHRE), ("lammps", viz.OXBLOOD)):
            run = NVE[c]
            gap = np.abs(run["positions"] - ref).max(axis=(1, 2))
            keep = (run["times"] > 0) & (run["times"] <= upto_ps * 1000)
            ax.semilogy(run["times"][keep] / 1000, gap[keep], color=colour,
                        label=c)
        ax.set_xlabel("t / ps")
        ax.set_ylabel("largest difference / Å")
        ax.legend()
        plt.show()


    widgets.interact(divergence,
                     upto_ps=widgets.FloatSlider(5.0, min=0.2, max=5.0,
                                                 step=0.2, **SLOW));
    """),
    md(r"""
    Both differences grow by about ten times a
    picosecond, the chaos of the liquid amplifying whatever separates the
    two paths: rounding for ASE, which starts near 10⁻¹⁴ Å, and the
    interpolation of LAMMPS's table, which starts near 10⁻⁶ Å. Neither is
    an error of either code; the energies and the averages agree.
    """),
    md(r"""
    ### Compare with ASE: diffusion under three thermostats

    $D$ from the MSD over 2 to 20 ps of 200 ps runs in each code, after
    the first 10 ps, with the error of five blocks.
    """),
    code(r"""
    def diffusion(run, blocks=5):
        late = run["times"] > 10000
        lag = run["times"][1] - run["times"][0]
        n = int(round(20000 / lag))
        ds = [transport.diffusion_coefficient(
                  lag * np.arange(n + 1),
                  transport.msd(b, n, remove_drift=True).sum(1), 2000, 20000)
              for b in np.array_split(run["positions"][late], blocks)]
        return np.mean(ds), np.std(ds, ddof=1) / np.sqrt(blocks)


    D_TABLE = {}
    for kind in ("langevin", "csvr", "nhc"):
        row = []
        for c in ("mdlab", "ase", "lammps"):
            d, e = diffusion(np.load(RUNS / f"argon_{c}_{kind}.npz"))
            D_TABLE[c, kind] = (d, e)
            row.append(f"{c} {d * 1e4:.2f} ± {e * 1e4:.2f}")
        print(f"{kind:9s} " + ";  ".join(row) + "  (1e-4 Å²/fs)")
    """),
]

LAMMPS = [
    md(r"""
    ## 17.2 LAMMPS

    The switched Lennard-Jones model written as a LAMMPS table with a
    chosen number of points, and LAMMPS's forces on one frame against
    `mdlab`'s.
    """),
    code(r"""
    if RUN_EXTERNAL_CALCULATORS:
        from lammps import lammps

        FOLDER = RUNS / "lammps"
        FOLDER.mkdir(parents=True, exist_ok=True)
        MODEL = ch17.argon_model(H0)
        U0, F0 = MODEL(R0)
        frame = Atoms(["Ar"] * len(M0), positions=R0, cell=H0.T, pbc=True,
                      masses=M0)
        ase_write(FOLDER / "notebook.data", frame, format="lammps-data",
                  masses=True, units="metal")


        def lammps_forces(points):
            io.lammps_table(MODEL.pair, 0.5, ch17.ch12.R_CUT, points,
                            str(FOLDER / "notebook.table"), "ARGON")
            lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
            lmp.commands_string(f'''
        units metal
        atom_style atomic
        atom_modify map array
        read_data {FOLDER / "notebook.data"}
        pair_style table linear {points}
        pair_coeff 1 1 {FOLDER / "notebook.table"} ARGON {ch17.ch12.R_CUT}
        run 0
        ''')
            order = np.argsort(lmp.numpy.extract_atom("id")[:len(M0)])
            f = lmp.numpy.extract_atom("f")[:len(M0)][order].copy()
            e = lmp.get_thermo("pe")
            lmp.close()
            return e, f


        def table_view(log_points=4.3):
            points = int(round(10**log_points))
            e, f = lammps_forces(points)
            print(f"{points} points: energy off by {abs(e - U0):.2e} eV, "
                  f"largest force off by {np.abs(f - F0).max():.2e} eV/Å")


        widgets.interact(table_view,
                         log_points=widgets.FloatSlider(4.3, min=2.0, max=4.3,
                                                        step=0.1, **SLOW));
    else:
        print("Optional LAMMPS table check: enable RUN_EXTERNAL_CALCULATORS to repeat it.")
    """),
    md(r"""
    Ten times more points makes the miss of the
    forces about a hundred times smaller, as the error $\delta x^2y''/8$
    of a straight line between points $\delta x$ apart says; LAMMPS
    rebuilds the table with the same number of points equally spaced in
    $r^2$ and interpolates there. With 20 001 points the
    miss on this frame is about 2 × 10⁻⁷ eV/Å, small but not zero, which is why LAMMPS's
    path parts from `mdlab`'s above.
    """),
]

LEARNED = [
    md(r"""
    ## 17.3 A learned potential in both

    MACE-MP-0 on graphite, with and without Grimme's D3 dispersion, and
    LiC₆'s energy at fixed energy against the time step (cached runs).
    """),
    code(r"""
    if RUN_EXTERNAL_CALCULATORS:
        CS = np.linspace(5.9, 9.0, 24)
        CURVES = {}
        for label, d3 in (("MACE-MP-0", False), ("with D3", True)):
            calc = ch17.mace(dtype="float64", dispersion=d3)
            energies = []
            for c in CS:
                g = ch17.graphite(c=c)
                g.calc = calc
                energies.append(g.get_potential_energy() / len(g))
            CURVES[label] = np.array(energies)


        def graphite_view(d3=True):
            fig, ax = plt.subplots(figsize=(6, 3))
            for label in ("MACE-MP-0", "with D3"):
                if label == "with D3" and not d3:
                    continue
                e = CURVES[label]
                ax.plot(CS, (e - e.min()) * 1000, label=label)
                print(f"{label}: least energy near c = {CS[np.argmin(e)]:.2f} Å")
            ax.axvline(ch17.C_GRAPHITE, **viz.THRESHOLD_STYLE)
            ax.set_xlabel("c / Å")
            ax.set_ylabel("energy per atom / meV")
            ax.set_ylim(0, 60)
            ax.legend()
            plt.show()


        widgets.interact(graphite_view, d3=True);
    else:
        from IPython.display import Image, display

        display(Image(filename=str(Path(viz.THEORY) / "book" / "figures"
                                   / "ch17_practice" / "learned.png")))
        print("Saved comparison from Section 17.3; graphite is the right-hand panel.")
    """),
    md(r"""
    Without D3 the least energy lies near 7.8 Å,
    far beyond the measured 6.711 Å (dotted), and the curve is nearly flat:
    PBE, on which the model was trained, binds graphite's sheets hardly at
    all. With D3 the minimum moves to about 6.6 Å and the binding is eight
    times stronger.
    """),
    code(r"""
    def step_view(dt=1.0, dtype="float64"):
        name = RUNS / f"nve_{dt:g}_{dtype}.npz"
        if not name.exists():
            print("no run with that step in that precision")
            return
        run = np.load(name)
        e = run["kinetic"] + run["potential"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(run["times"] / 1000, (e - e[0]) * 1000, lw=0.6)
        ax.set_xlabel("t / ps")
        ax.set_ylabel("K + U, change / meV")
        plt.show()
        print(f"spread of K + U over that of K: "
              f"{e.std() / run['kinetic'].std():.4f}")


    widgets.interact(step_view, dt=[0.5, 1.0, 2.0, 3.0],
                     dtype=["float64", "float32"]);
    """),
    md(r"""
    Each doubling of the step makes the band of the
    total energy about four times wider, up to 2 fs; at 3 fs it widens
    more than $\delta t^2$ says and the energy climbs. Single precision,
    run on the GPU, gives the same band at 1 and 2 fs.
    """),
]

AIMD = [
    md(r"""
    ## 17.4 First-principles molecular dynamics and its cost

    The FHI-aims single points of a 28-atom LiC₆ frame, read from their
    cached results.
    """),
    code(r"""
    finest = np.load(RUNS / "aims_kgrid_4.npz")
    for n_ in (2, 3, 4):
        r = np.load(RUNS / f"aims_kgrid_{n_}.npz")
        carbon = np.array(read(RUNS / "aims_frames.extxyz", 0)
                          .get_chemical_symbols()) == "C"
        gap = np.linalg.norm(r["forces"] - finest["forces"], axis=1)[carbon]
        print(f"k {tuple(int(k) for k in r['kgrid'])}: {float(r['wall']):.0f}"
              f" s, {int(r['cycles'])} cycles; carbon's forces from the "
              f"finest grid's, up to {gap.max():.3f} eV/Å")
    step = float(finest["wall"]) / 2
    mace_step = float(np.load(RUNS / "aims_frames.npz")["wall"])
    print(f"a first-principles step at half the time: {step:.0f} s; "
          f"MACE-MP-0: {mace_step * 1000:.0f} ms, "
          f"{step / mace_step:.0f} times less")
    """),
]

LIC6 = [
    md(r"""
    ## 17.5 LiC₆ with a foundation model

    The spacing of LiC₆'s sheets at 300 K under the two couplings, and the
    convergence report on it.
    """),
    code(r"""
    def lengths(run):
        norms = np.linalg.norm(run["cell"], axis=2)
        return norms[:, 0] / 3 / np.sqrt(3), norms[:, 2]


    NPT = {name: np.load(RUNS / f"{name}.npz")
           for name in ("npt_aniso", "npt_iso", "npt_long")
           if (RUNS / f"{name}.npz").exists()}


    def npt_view(run="npt_aniso"):
        r = NPT[run]
        a, c = lengths(r)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(r["times"] / 1000, c, lw=0.5)
        ax.axhline(ch17.C_LIC6, **viz.THRESHOLD_STYLE)
        ax.set_xlabel("t / ps")
        ax.set_ylabel("c / Å")
        plt.show()
        print(stats.convergence_report(c, r["times"][1] - r["times"][0]))


    widgets.interact(npt_view, run=list(NPT));
    """),
    md(r"""
    Under isotropic coupling the spacing stays at its
    value at 0 K, 3.616 Å, and the report passes on a quantity that never
    moved. With the lengths coupled apart, the first 20 ps rise towards
    3.78 Å and the report refuses them. These saved 20 ps do not yet
    establish the equilibrium spacing; the next step is a longer run.
    """),
]

HOPS = [
    md(r"""
    ## 17.6 Lithium on the move

    One lithium atom among 64 carbon atoms at 300, 450 and 600 K, 20 ps
    each, counted from the start that Chapter 15's test finds. Its sites
    are the hollows of the sheet below (0 to 15) and of the sheet above
    (16 to 31), each read in the frame of its own sheet. Choose
    the temperature, both families of sites or the sheet below's alone,
    and the least stay a visit needs to count: the left panel shows the
    site against time with the counted hops, the right the count against
    the least stay.
    """),
    code(r"""
    DILUTE = {t: np.load(RUNS / f"dilute_{t}.npz") for t in (300, 450, 600)}
    # count from the start that Chapter 15's test finds in the temperature
    START = {t: stats.detect_equilibration(
        2 * run["kinetic"] / ((3 * 65 - 3) * KB))[0]
        for t, run in DILUTE.items()}
    SITES = {(t, f): ch17.li_sites(run, families=f)["sites"][START[t]:]
             for t, run in DILUTE.items() for f in (1, 2)}
    STAYS = np.arange(1, 21)  # frames of 10 fs


    def hop_view(temperature=300, families=2, least_stay=50):
        run, seq = DILUTE[temperature], SITES[temperature, families]
        times = run["times"][START[temperature]:]
        found = hops.hops(seq, max(1, least_stay // 10))
        span = (times[-1] - times[0]) / 1000  # ps
        r = hops.rate(len(found), span)
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3))
        ax.plot(times / 1000, seq, lw=0.4, color="0.6")
        if len(found):
            ax.plot(times[found[:, 0]] / 1000, found[:, 2], "o", ms=2)
        ax.set_xlabel("t / ps")
        ax.set_ylabel("site")
        bx.plot(STAYS * 10, [len(hops.hops(seq, s)) for s in STAYS], "o-",
                ms=3)
        bx.axhline(25, **viz.THRESHOLD_STYLE)
        bx.axvline(least_stay, color="0.7", lw=0.8)
        bx.set_xlabel("least stay / fs")
        bx.set_ylabel(f"hops in {span:.0f} ps")
        plt.show()
        if r["resolved"]:
            print(f"{len(found)} hops: {r['rate']:.2f} ± {r['error']:.2f} "
                  f"per ps")
        else:
            print(f"{len(found)} hops: unresolved; {r['needed']:.0f} ps "
                  f"would resolve it")


    widgets.interact(hop_view, temperature=[300, 450, 600],
                     families=widgets.Dropdown(
                         options=[("both sheets", 2), ("sheet below", 1)]),
                     least_stay=widgets.IntSlider(
                         value=50, min=10, max=200, step=10, **SLOW));
    """),
    md(r"""
    With both families the count at 300 K moves
    only from 14 to 11 across the slider, and the hops alternate between
    the sites below 16 and those above: neighbours on the honeycomb belong
    to opposite families. With the sheet below alone it falls from 28 to
    8, and the left panel shows the reason: while lithium sits over a
    site of the missing family, the plot flickers among three sites of the
    other. At 50 fs the rate is resolved at 450 and 600 K and not at 300 K,
    where the print gives the run that would resolve it.
    """),
]

PROTOCOL = [
    md(r"""
    ## 17.7 A protocol for a production run, and 17.8 a health report

    `diagnostics.health_report` on records of this chapter, healthy and
    broken. Choose one and read which checks it fails.
    """),
    code(r"""
    def records():
        nve = np.load(RUNS / "argon_mdlab_nve.npz")
        csvr = np.load(RUNS / "argon_mdlab_csvr.npz")
        late = csvr["times"] > 10000
        langevin = np.load(RUNS / "argon_mdlab_langevin.npz")
        late_l = langevin["times"] > 10000
        kt_m = KB * 135.0 / (39.948 * units.MV2_TO_EV)
        drift = math.sqrt(3 * kt_m / len(M0))
        out = {
            "argon at fixed energy": dict(
                positions=nve["positions"], cell=H0, masses=M0,
                times=nve["times"], potential=nve["potential"],
                kinetic=nve["kinetic"], symbols=["Ar"] * len(M0)),
            "the same, wrapped": dict(
                positions=np.array([wrap(x, H0)
                                    for x in nve["positions"][::10]]),
                cell=H0, masses=M0),
            "the same, its centre carried along": dict(
                positions=nve["positions"] + drift * nve["times"][:, None,
                                                                  None]
                * np.array([1.0, 0, 0]), cell=H0, masses=M0),
            "argon under CSVR": dict(
                positions=csvr["positions"][late], cell=H0, masses=M0,
                kinetic=csvr["kinetic"][late], temperature=135.0,
                ensemble="nvt"),
            "argon under Langevin, as if the momentum were kept": dict(
                positions=langevin["positions"][late_l], cell=H0, masses=M0),
            "argon under Langevin, the momentum not kept": dict(
                positions=langevin["positions"][late_l], cell=H0, masses=M0,
                momentum_kept=False),
        }
        for dt in (1.0, 3.0):
            run = np.load(RUNS / f"nve_{dt:g}_float64.npz")
            out[f"LiC6 at fixed energy, {dt:g} fs"] = dict(  # energies only
                positions=None, cell=None, times=run["times"],
                potential=run["potential"], kinetic=run["kinetic"])
        return out


    RECORDS = records()


    def report_view(record="argon at fixed energy"):
        print(diagnostics.health_report(**RECORDS[record]))


    widgets.interact(report_view, record=list(RECORDS));
    """),
    md(r"""
    The healthy argon passes every check its record
    allows. Wrapping fails the continuity check with moves of nearly a
    whole cell; carrying the centre of mass fails the second check; CSVR
    passes its temperature and its canonical spread; the healthy Langevin
    run fails the centre-of-mass check, since its random forces move the
    centre as one free particle, and passes once the check is left out for
    a run that does not keep its momentum; LiC₆ passes at 1 fs
    and fails both energy checks at 3 fs.
    """),
]

EXERCISES = [
    md(r"""
    ## Working notes

    Keep the code version, unit conversion and thermostat options beside
    each comparison. When trajectories separate, compare energies and
    uncertainty in the observables before changing the integrator. For
    lithium in graphite, record which cell lengths are free and how many
    persistent hops support the reported rate.
    """),
    md(r"""
    ## Exercises

    Each exercise cell checks its answer against the key below; the
    collapsed cell after it holds a worked solution.
    """),
    hidden(r"""
    # The answer key. Each target is computed from the exercise's data.
    AMU, EV = 1.66053906892e-27, 1.602176634e-19
    T_ASE = 1e-10 * math.sqrt(AMU / EV) / 1e-15  # fs
    T_AKMA = 1e-10 * math.sqrt(AMU / (4184 / 6.02214076e23)) / 1e-15
    TARGETS = {
        "17.1": np.array([T_AKMA, 2 / T_AKMA]),
        "17.2": np.array([0.01 * T_ASE, 10.0]),
        "17.3": np.array([2.0, 0.5e-3 * T_ASE]),
        "17.4": np.array([768 / 765, math.sqrt(768 / 765)]),
        "17.6": 20000 * math.sqrt(7.6e-7 / 1e-9) + 1,
        "17.7": np.array([1, -2, 3]),
        "17.8": 0.25 * 1000 / 126 * 1e6 / 86400,
        "17.9": KB * 300 * 1000,
        "17.10": math.sqrt(0.05 / 0.0067),
        "17.11": np.array([1e4 * 192 * 4 / 3600, 1e4 * 192 * 4 / 3600 / 192]),
        "17.12": np.array([120.0, 3.0]),
        "17.13": np.array([(-1.00 - 1.11 + 1.68) / 3,
                           math.sqrt(0.45**2 + 0.36**2 + 0.10**2) / 3]),
        "17.14": np.array([12.5, -math.log(0.05) / 20,
                           25 / (-math.log(0.05) / 20)]),
        "17.15": 2.4648 / math.sqrt(3),
        "17.19": 3 * math.sqrt(24 / 768),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 17.1: the unit of time in fs of a code in kcal/mol, Å and
    # amu (1 kcal = 4184 J, 6.02214076e23 per mole), and the number such a
    # code needs for a step of 2 fs.
    unit = None

    check(unit, TARGETS["17.1"], rtol=1e-4, name="unit and step")
    """),
    solution(r"""
    # SOLUTION 17.1. (kcal/mol)/Å = amu Å/t², so t = Å √(amu/(kcal/mol)).
    kcal_mol = 4184 / 6.02214076e23  # J for one molecule
    t = 1e-10 * math.sqrt(1.66053906892e-27 / kcal_mol) / 1e-15
    unit = np.array([t, 2 / t])

    check(unit, TARGETS["17.1"], rtol=1e-4, name="unit and step");
    """),
    code(r"""
    # EXERCISE 17.2: 0.01 Å/fs in ASE's units and in LAMMPS's metal units.
    velocity = None

    check(velocity, TARGETS["17.2"], rtol=1e-4, name="velocities")
    """),
    solution(r"""
    # SOLUTION 17.2. ASE's unit is Å per 10.1805 fs; metal's is Å/ps.
    velocity = np.array([0.01 * ch17.ASE_FS, 0.01 * 1000])

    check(velocity, TARGETS["17.2"], rtol=1e-4, name="velocities");
    """),
    code(r"""
    # EXERCISE 17.3: for γ = 0.5 /ps, LAMMPS's damping parameter in ps and
    # ASE's friction per ASE unit of time.
    friction = None

    check(friction, TARGETS["17.3"], rtol=1e-4, name="damp and friction")
    """),
    solution(r"""
    # SOLUTION 17.3. LAMMPS takes 1/γ; ASE takes γ in 1/(ASE time).
    gamma = 0.5e-3  # per fs
    friction = np.array([1 / gamma / 1000, gamma * ch17.ASE_FS])

    check(friction, TARGETS["17.3"], rtol=1e-4, name="damp and friction");
    """),
    code(r"""
    # EXERCISE 17.4: the ratio of ASE's chain mass 3N kT τ² to mdlab's
    # N_f kT τ² for 256 atoms, and the ratio of the periods.
    ratios = None

    check(ratios, TARGETS["17.4"], rtol=1e-6, name="mass and period")
    """),
    solution(r"""
    # SOLUTION 17.4. N_f = 3N − 3; the period goes as √mass.
    n = 256
    ratios = np.array([3 * n / (3 * n - 3), math.sqrt(3 * n / (3 * n - 3))])

    check(ratios, TARGETS["17.4"], rtol=1e-6, name="mass and period");
    """),
    code(r"""
    # EXERCISE 17.6: entries of LAMMPS's table that bring its miss beyond
    # 3 Å from 7.6e-7 to 1e-9 eV/Å over the same range, from 20001 now.
    points = None

    check(points, TARGETS["17.6"], rtol=1e-3, name="points")
    """),
    solution(r"""
    # SOLUTION 17.6. The miss goes as δx², so the intervals grow by √ratio.
    points = 20000 * math.sqrt(7.6e-7 / 1e-9) + 1

    check(points, TARGETS["17.6"], rtol=1e-3, name="points");
    """),
    code(r"""
    # EXERCISE 17.7: the image counts (ix, iy, iz) packed in 540539393.
    image = None

    check(image, TARGETS["17.7"], rtol=0, atol=0, name="image counts")
    """),
    solution(r"""
    # SOLUTION 17.7. Remainders and whole quotients on division by 1024.
    flag = 540539393
    image = np.array([flag % 1024 - 512, flag // 1024 % 1024 - 512,
                      flag // 1024**2 - 512])

    check(image, TARGETS["17.7"], rtol=0, atol=0, name="image counts");
    """),
    code(r"""
    # EXERCISE 17.8: days for 1 ns of 1000 atoms with steps of 1 fs, at
    # 0.25 s a call for 126 atoms, in proportion to the atoms.
    days = None

    check(days, TARGETS["17.8"], rtol=1e-3, name="days")
    """),
    solution(r"""
    # SOLUTION 17.8. 1.98 s a step for 1000 atoms; 10⁶ steps.
    days = 0.25 * 1000 / 126 * 1e6 / 86400

    check(days, TARGETS["17.8"], rtol=1e-3, name="days");
    """),
    code(r"""
    # EXERCISE 17.9: k_BT at 300 K in meV, against 5.2 and 41.2 meV.
    kt = None

    check(kt, TARGETS["17.9"], rtol=1e-3, name="k_BT")
    """),
    solution(r"""
    # SOLUTION 17.9. KB is in eV/K.
    kt = KB * 300 * 1000
    print(f"{kt:.1f} meV: 5.2 is a fifth of it; 41.2 is larger")

    check(kt, TARGETS["17.9"], rtol=1e-3, name="k_BT");
    """),
    code(r"""
    # EXERCISE 17.10: the step in fs at which 0.0067 (δt / 1 fs)² reaches
    # 0.05.
    longest = None

    check(longest, TARGETS["17.10"], rtol=1e-3, name="longest step")
    """),
    solution(r"""
    # SOLUTION 17.10. Solve 0.0067 δt² = 0.05.
    longest = math.sqrt(0.05 / 0.0067)

    check(longest, TARGETS["17.10"], rtol=1e-3, name="longest step");
    """),
    code(r"""
    # EXERCISE 17.11: core-hours for 10 ps of steps of 1 fs at 192 s a step
    # on four cores, and hours on 192 cores.
    cost = None

    check(cost, TARGETS["17.11"], rtol=1e-3, name="core-hours and hours")
    """),
    solution(r"""
    # SOLUTION 17.11. 10⁴ steps of 192 s on 4 cores.
    core_hours = 1e4 * 192 * 4 / 3600
    cost = np.array([core_hours, core_hours / 192])

    check(cost, TARGETS["17.11"], rtol=1e-3, name="core-hours and hours");
    """),
    code(r"""
    # EXERCISE 17.12: the angle in degrees between 2a₁ + a₂ and −a₁ + a₂,
    # and the area of their cell over the sheet's.
    cell_check = None

    check(cell_check, TARGETS["17.12"], rtol=1e-9, name="angle and area")
    """),
    solution(r"""
    # SOLUTION 17.12. Dot product over lengths; areas by determinants.
    a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, math.sqrt(3) / 2])
    b1, b2 = 2 * a1 + a2, -a1 + a2
    angle = math.degrees(math.acos(b1 @ b2 / (np.linalg.norm(b1)
                                              * np.linalg.norm(b2))))
    cell_check = np.array([angle, abs(np.linalg.det([b1, b2])
                                      / np.linalg.det([a1, a2]))])

    check(cell_check, TARGETS["17.12"], rtol=1e-9, name="angle and area");
    """),
    code(r"""
    # EXERCISE 17.13: the mean of −1.00 ± 0.45, −1.11 ± 0.36 and
    # 1.68 ± 0.10 GPa, and its error if the three are independent.
    mean_p = None

    check(mean_p, TARGETS["17.13"], rtol=1e-3, name="mean and error")
    """),
    solution(r"""
    # SOLUTION 17.13. Errors of independent terms add in quadrature.
    p = np.array([-1.00, -1.11, 1.68])
    e = np.array([0.45, 0.36, 0.10])
    mean_p = np.array([p.mean(), np.sqrt(np.sum(e**2)) / 3])

    check(mean_p, TARGETS["17.13"], rtol=1e-3, name="mean and error");
    """),
    code(r"""
    # EXERCISE 17.14: the ps to see 25 hops at 2 per ps; the 95% bound on
    # the rate, per ps, after none in 20 ps; and the ps that 25 hops at
    # that bound would need.
    rate_answers = None

    check(rate_answers, TARGETS["17.14"], rtol=1e-3, name="rate answers")
    """),
    solution(r"""
    # SOLUTION 17.14. hops.rate holds the same rules.
    none = hops.rate(0, 20.0)
    rate_answers = np.array([hops.rate(4, 2.0)["needed"], none["upper"],
                             none["needed"]])

    check(rate_answers, TARGETS["17.14"], rtol=1e-3, name="rate answers");
    """),
    code(r"""
    # EXERCISE 17.15: the distance ℓ in Å between neighbouring sites of
    # lithium in dilute graphite, a = 2.4648 Å.
    ell = None

    check(ell, TARGETS["17.15"], rtol=1e-4, name="ℓ")
    """),
    solution(r"""
    # SOLUTION 17.15. A hollow lies a/√3 from each corner of its hexagon.
    ell = 2.4648 / math.sqrt(3)

    check(ell, TARGETS["17.15"], rtol=1e-4, name="ℓ");
    """),
    code(r"""
    # EXERCISE 17.19: health_report's limit on the excess kurtosis of the
    # 768 velocity components of 256 atoms.
    limit = None

    check(limit, TARGETS["17.19"], rtol=1e-6, name="kurtosis limit")
    """),
    solution(r"""
    # SOLUTION 17.19. Three times √(24/n); −1.2 lies far outside it.
    limit = 3 * math.sqrt(24 / 768)

    check(limit, TARGETS["17.19"], rtol=1e-6, name="kurtosis limit");
    """),
]

CELLS = (SETUP + ASE + LAMMPS + LEARNED + AIMD + LIC6 + HOPS + PROTOCOL
         + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "17_practice.ipynb"))
