"""Write notebooks/16_observables.ipynb, the companion to Chapter 16.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_16_observables.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/16_observables.ipynb

The runs are read from data/ch16_observables/runs/ (written by
scripts/ch16_observables/runs.py) and data/ch15_convergence/runs/. The
radial distribution function, the diffusion coefficient and the bond
criterion are checked against ASE.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 16: Observables

    Use the saved trajectories to work out what each observable measures.
    Start with structure, then follow motion and transport, and finish with
    spectra and free energy. Keep the units beside each estimate and check
    how the result changes with the analysis window. The section numbers
    follow Chapter 16; run the cells in order.
    """),
    code(r"""
    %matplotlib inline
    import math
    import sys
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import HTML
    from matplotlib import animation

    from mdlab import bonds, cell, diagnostics, units, viz
    from mdlab.analysis import landscape, spectra, structure, transport
    from mdlab.exercise import check

    sys.path.insert(0, str(Path("..") / "scripts" / "ch16_observables"))
    import ch16  # the shared set-up of the chapter's scripts

    viz.use_style()
    RNG = np.random.default_rng(16)
    SLOW = dict(continuous_update=False)  # redraw only on release
    KB = units.KB
    THZ = 1000.0  # 1/fs to THz
    PA_S = units.ELEMENTARY_CHARGE / units.ANGSTROM**3 * units.FEMTOSECOND


    def run(name, chapter=16):
        # One cached run of the chapter's runs.py (or Chapter 15's).
        folder = ch16.RUNS if chapter == 16 else ch16.ch15.RUNS
        return np.load(folder / f"{name}.npz")


    LIQUID = run("long_csvr", 15)
    H = LIQUID["cell"]
    FRAMES = LIQUID["positions"][1::4].astype(float)  # every 2 ps
    RHO = 256 / np.linalg.det(H)
    """),
]

RDF = [
    md(r"""
    ## 16.1 How atoms arrange themselves

    The animation grows shells of equal width around one atom of the liquid
    and counts the partners in each, divided by what an even scatter would
    put there. One frame gives a ragged histogram; the average over frames
    and centres is $g(r)$.
    """),
    code(r"""
    def shells_animation(frames=30):
        r0 = FRAMES[0]
        centre = r0[0]
        d = np.linalg.norm(cell.minimum_image(r0 - centre, H), axis=1)[1:]
        x, g_all = structure.rdf(FRAMES, H, 11.6, 116)
        edges = np.linspace(0, 11.6, 117)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
        sep = cell.minimum_image(r0 - centre, H)
        slab = np.abs(sep[:, 2]) < 1.7  # a slice through the centre
        ax1.scatter(sep[slab, 0], sep[slab, 1], s=12, color=viz.REFERENCE)
        ax1.plot(0, 0, "o", color=viz.OXBLOOD)
        ring = plt.Circle((0, 0), 0.1, fill=False, color=viz.ACCENT, lw=2)
        ax1.add_patch(ring)
        ax1.set_xlim(-11.6, 11.6)
        ax1.set_ylim(-11.6, 11.6)
        ax1.set_aspect("equal")
        ax1.set_title("one atom and a slice of its neighbours", fontsize=9)
        ax1.set_xlabel("x / Å")
        ax1.set_ylabel("y / Å")
        ax2.plot(x, g_all, **viz.REFERENCE_STYLE)
        bars = ax2.bar(x, np.zeros_like(x), width=0.1, color=viz.ACCENT)
        ax2.set_xlim(0, 11.6)
        ax2.set_ylim(0, 6)
        ax2.set_xlabel("r / Å")
        ax2.set_ylabel("g(r)")
        one = np.histogram(d, edges)[0] / (RHO
                                           * structure.shell_volumes(edges))

        def draw(k):
            r = 11.6 * (k + 1) / frames
            ring.set_radius(r)
            for b, value, mid in zip(bars, one, x):
                b.set_height(value if mid <= r else 0)
            return [ring, *bars]

        anim = animation.FuncAnimation(fig, draw, frames=frames, blit=True)
        plt.close(fig)
        return HTML(anim.to_jshtml())


    shells_animation()
    """),
    md(r"""
    One atom's histogram (bars) is ragged: a shell
    holds a few partners or none. It already shows the gap inside
    3 Å and the crowd near 3.7 Å that the average over every atom and
    1000 frames (dashed) makes smooth.
    """),
    code(r"""
    def rdf_view(n_frames=10, centre_fraction=1.0):
        n_centres = max(1, int(256 * centre_fraction))
        centres = np.arange(n_centres)
        x, g = structure.rdf(FRAMES[:n_frames], H, 11.6, 232,
                             centres=centres)
        x0, g0 = structure.rdf(FRAMES, H, 11.6, 232)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x0, g0, **viz.REFERENCE_STYLE)
        ax.plot(x, g, color=viz.ACCENT, lw=0.8)
        ax.set_xlabel("r / Å")
        ax.set_ylabel("g(r)")
        plt.show()
        n_c = structure.running_coordination(x0, g0, RHO)
        low = np.argmin(np.where((x0 > 4.5) & (x0 < 6.5), g0, 9))
        print(f"{n_frames} frames, {n_centres} centres; largest difference "
              f"from all 1000 frames {np.abs(g - g0).max():.3f}; "
              f"coordination to the first minimum {n_c[low]:.2f}")


    widgets.interact(rdf_view,
                     n_frames=widgets.IntSlider(10, min=1, max=1000, **SLOW),
                     centre_fraction=widgets.FloatSlider(1.0, min=0.004,
                                                         max=1.0, step=0.004,
                                                         **SLOW));
    """),
    md(r"""
    With every atom as a centre, even one frame
    gives the shape of $g(r)$, within 0.75 of the whole run everywhere;
    ten frames come within 0.17 and a hundred within 0.05, the difference
    shrinking roughly as one over the square root of the frames. With one
    centre and one frame the curve is all noise. The first shell holds
    about 12 neighbours.
    """),
    md(r"""
    ### Compare with ASE

    ASE's `get_rdf` uses the same normalisation, $N\rho$ times the shell
    volume, so on the same frames the two agree to rounding.
    """),
    code(r"""
    from ase import Atoms
    from ase.geometry.rdf import get_rdf

    images = [Atoms("Ar256", positions=f, cell=H.T, pbc=True)
              for f in FRAMES[:50]]
    ours = structure.rdf(FRAMES[:50], H, 11.6, 232)[1]
    theirs = get_rdf(images, 11.6, 232, no_dists=True)
    print(f"largest difference from ASE: {np.abs(ours - theirs).max():.1e}")
    assert np.allclose(ours, theirs, atol=1e-10)
    """),
]

SQ = [
    md(r"""
    ## 16.2 What diffraction sees

    $S(G)$ from the positions, on the wave vectors of the cell, against the
    transform of $g(r)$ cut at a distance you choose. The cut spoils the
    transform at small $G$.
    """),
    code(r"""
    Q, S_DIRECT = structure.structure_factor(FRAMES[::4], H, 10)
    X_FINE, G_FINE = structure.rdf(FRAMES, H, 11.6, 232)


    def sq_view(r_cut=11.6):
        keep = X_FINE <= r_cut
        grid = np.linspace(0.2, Q.max(), 300)
        s = structure.structure_factor_from_rdf(X_FINE[keep], G_FINE[keep],
                                                RHO, grid)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(Q, S_DIRECT, "o", ms=2, color=viz.ACCENT)
        ax.plot(grid, s, **viz.REFERENCE_STYLE)
        ax.set_xlabel("G / Å⁻¹")
        ax.set_ylabel("S(G)")
        plt.show()
        at = structure.structure_factor_from_rdf(X_FINE[keep], G_FINE[keep],
                                                 RHO, Q[:1])[0]
        print(f"smallest G {Q[0]:.3f} Å⁻¹: from positions {S_DIRECT[0]:.3f}, "
              f"from g(r) cut at {r_cut:.1f} Å {at:.3f}")


    widgets.interact(sq_view, r_cut=widgets.FloatSlider(11.6, min=5.0,
                                                        max=11.6, step=0.2,
                                                        **SLOW));
    """),
    md(r"""
    Above about 2.5 Å⁻¹ the transform follows the
    points whatever the cut. Near the first peak it falls short, by 0.15
    even at the longest cut, and below 1 Å⁻¹ it ripples, its miss at the
    smallest $G$ changing with the cut without settling. The points, which
    need no cut, sit near 0.06 there, close to $\rhok_{\mathrm B} T\kappa_T =
    0.055$.
    """),
]

BONDS = [
    md(r"""
    ## 16.3 Bonds and molecules

    The first 5 ps of the A-B gas at 2000 K, every 10 fs. Choose $r_{\rm
    on}$ and $r_{\rm off}$ and count the bonds formed and broken; with
    $r_{\rm off} = r_{\rm on}$ there is no band.
    """),
    code(r"""
    GAS = run("gas_2000")
    GAS_FRAMES = GAS["positions"].astype(float)
    GAS_H = GAS["cell"]
    print(f"r_on = {ch16.MORSE_ON:.3f} Å, r_off = {ch16.MORSE_OFF:.3f} Å; "
          f"{len(GAS_FRAMES)} frames")


    from mdlab.neighbours import cell_list_pairs

    # The pairs within 3 Å of each frame, found once for every slider move.
    GAS_PAIRS = [cell_list_pairs(x, GAS_H, 3.0) for x in GAS_FRAMES]


    def count_events(r_on, r_off, pairs=GAS_PAIRS):
        tracker = ch16.gas_tracker(r_on, max(r_on, r_off))
        events = 0
        for k, (i, j, _, d) in enumerate(pairs):
            formed, broken = tracker.update(i, j, d)
            if k:
                events += len(formed) + len(broken)
        return events, len(tracker.keys)


    def flicker(r_on=1.81, r_off=2.57):
        events, held = count_events(r_on, r_off)
        span = (len(GAS_FRAMES) - 1) * 10 / 1000
        print(f"r_on {r_on:.2f}, r_off {max(r_on, r_off):.2f} Å: "
              f"{events / span:.0f} events per ps; {held} bonds at the end")


    widgets.interact(flicker,
                     r_on=widgets.FloatSlider(1.81, min=1.4, max=2.4,
                                              step=0.01, **SLOW),
                     r_off=widgets.FloatSlider(2.57, min=1.4, max=2.9,
                                               step=0.01, **SLOW));
    """),
    md(r"""
    With $r_{\rm off} = r_{\rm on}$ the counts run
    into hundreds per picosecond, most of them vibrations crossing the
    cutoff; the chapter's band cuts them about sixfold over these first
    5 ps. Moving $r_{\rm on}$ out counts loosely held pairs as bonds, and
    a single cutoff moved in to 1.4 Å, inside most vibrations, flickers
    thousands of times per picosecond.
    """),
    md(r"""
    ### Compare with ASE

    With $r_{\rm on}$ the sum of the covalent radii and no history, the
    bonds of the first frame are the pairs closer than that sum, which ASE's
    `neighbor_list` finds with `natural_cutoffs`.
    """),
    code(r"""
    from ase.neighborlist import natural_cutoffs, neighbor_list
    from mdlab.neighbours import cell_list_pairs

    rng = np.random.default_rng(16)
    symbols = ["C"] * 40 + ["H"] * 40
    r = rng.uniform(0, 8, (80, 3))
    h = 8.0 * np.eye(3)
    tracker = bonds.BondTracker(symbols, on=1.0, off=1.3)
    i, j, _, d = cell_list_pairs(r, h, tracker.reach)
    tracker.update(i, j, d)
    atoms = Atoms(symbols, positions=r, cell=h, pbc=True)
    a, b = neighbor_list("ij", atoms, natural_cutoffs(atoms))
    theirs = np.unique(np.minimum(a, b) * 80 + np.maximum(a, b))
    print(f"{len(tracker.keys)} bonds; the same as ASE's: "
          f"{np.array_equal(tracker.keys, theirs)}")
    assert np.array_equal(tracker.keys, theirs)
    """),
]

REACTIONS = [
    md(r"""
    ## 16.4 Reactions and their rates

    Species counts and the reaction network of the gas at each
    temperature, and the rate of breaking with the error $\sqrt n$ of its
    count.
    """),
    code(r"""
    def network(temperature=2000, least_life=0.5):
        r = run(f"gas_{temperature}")
        names = list(r["band_species"])
        counts = r["band_counts"]
        t = r["times"] / 1000
        fig, ax = plt.subplots(figsize=(6, 3))
        for name, colour in (("AB", viz.ACCENT), ("A", viz.OCHRE)):
            ax.plot(t, counts[:, names.index(name)], color=colour, lw=0.7,
                    label=name)
        ax.set_xlabel("t / ps")
        ax.set_ylabel("number")
        ax.legend()
        plt.show()
        events = [(float(a), int(i), int(j), int(s))
                  for a, i, j, s in r["band_events"]]
        kept = [e for e in bonds.persistent(events, least_life * 1000)
                if e[0] > 10_000]
        broken = sum(1 for e in kept if e[3] == -1)
        late = r["times"] > 10_000
        exposure = counts[late][:, names.index("AB")].sum() * 0.01  # bond-ps
        print(f"{broken} bonds broken after 10 ps (least lifetime "
              f"{least_life} ps): k_b ≈ {broken / exposure:.4f} ± "
              f"{math.sqrt(broken) / exposure:.4f} per ps")
        kinds, n = np.unique(r["band_reactions"], return_counts=True)
        for k in np.argsort(-n)[:6]:
            print(f"   {kinds[k]}: {n[k]}")


    widgets.interact(network,
                     temperature=[1500, 1750, 2000, 2250, 2500],
                     least_life=widgets.FloatSlider(0.5, min=0.0, max=2.0,
                                                    step=0.1, **SLOW));
    """),
    md(r"""
    The molecules fall from 250 in the first few
    picoseconds and then fluctuate about a level that drops as the
    temperature rises. Every reaction and its reverse occur about equally
    often. The rate of breaking grows with the temperature at every least
    lifetime, and its value falls as the least lifetime grows, quickly at
    first, as passing encounters drop out. (The exposure here counts only
    AB molecules, so the rates sit a little above those of the chapter,
    which counts every bond.)
    """),
]

MSD = [
    md(r"""
    ## 16.5 How far atoms wander

    The MSD of liquid argon at fixed energy, with a window for the fit of
    $D$ that you choose.
    """),
    code(r"""
    ARGON = run("argon_nve")
    V_ARGON = ARGON["velocities"].astype(float)
    M_ARGON = ARGON["masses"]
    START = ARGON["positions"][0].astype(float)
    PATHS = START + np.concatenate(
        [np.zeros((1, 256, 3)),
         np.cumsum(5.0 * (V_ARGON[1:] + V_ARGON[:-1]), axis=0)])
    T_FULL = 10.0 * np.arange(2001)
    MSD_FULL = transport.msd(PATHS, 2000, masses=M_ARGON,
                             origin_step=10).sum(1)


    def fit_window(start_ps=2.0, stop_ps=20.0):
        stop_ps = max(stop_ps, start_ps + 0.05)
        d = transport.diffusion_coefficient(T_FULL, MSD_FULL,
                                            start_ps * 1000, stop_ps * 1000)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.loglog(T_FULL[1:] / 1000, MSD_FULL[1:], color=viz.ACCENT)
        ax.axvspan(start_ps, stop_ps, color=viz.tint(viz.OCHRE, 0.3))
        ax.set_xlabel("t / ps")
        ax.set_ylabel("MSD / Å²")
        plt.show()
        print(f"D = {d * 1e4:.3f}e-4 Å²/fs from {start_ps:.2f} to "
              f"{stop_ps:.2f} ps")


    widgets.interact(fit_window,
                     start_ps=widgets.FloatLogSlider(2.0, min=-2, max=1,
                                                     **SLOW),
                     stop_ps=widgets.FloatLogSlider(20.0, min=-1, max=1.31,
                                                    **SLOW));
    """),
    md(r"""
    Windows that start beyond about 0.05 ps and end
    ten times later or more give $D$ within about 6 per cent of $3.76\times
    10^{-4}$ Å²/fs. A window inside the first 0.2 ps, where the MSD grows
    as $t^2$, gives a value too low, by a third for 0.02 to 0.2 ps.
    """),
    md(r"""
    ### Compare with ASE

    ASE's `DiffusionCoefficient` fits each axis's MSD from the first frame
    of each segment, with no centre removed; `msd` with one start per
    segment does the same.
    """),
    code(r"""
    from ase.md.analysis import DiffusionCoefficient

    long = run("long_nve", 15)
    pos = long["positions"][1:401].astype(float)
    ase_dt = 500.0 * np.sqrt(units.FORCE_TO_ACCEL)  # 500 fs in ASE time
    images = [Atoms("Ar256", positions=x, cell=long["cell"].T) for x in pos]
    theirs = DiffusionCoefficient(images, ase_dt)
    theirs.calculate(number_of_segments=4)
    theirs_d = np.mean(theirs.slopes[0], axis=1) * np.sqrt(
        units.FORCE_TO_ACCEL)
    ours_d = []
    for seg in np.array_split(pos, 4):
        o = transport.msd(seg, remove_drift=False, origin_step=len(seg))
        ours_d.append(transport.diffusion_coefficient(
            500 * np.arange(len(seg)), o.sum(1), 0.0, 500 * (len(seg) - 1)))
    gap = np.max(np.abs(np.array(ours_d) / theirs_d - 1))
    print(f"largest relative difference from ASE: {gap:.1e}")
    assert gap < 1e-10
    """),
]

VACF = [
    md(r"""
    ## 16.6 Velocities that remember

    The running Green-Kubo integrals of $D$, from one run, and of the
    shear viscosity, from one of Chapter 15's runs or all twelve.
    """),
    code(r"""
    C_VV = transport.velocity_autocorrelation(V_ARGON, 2000)
    GK_D = transport.running_integral(C_VV, 10.0) / 3
    NAMES = ["long_nve", "long_csvr"] + [f"size_{n}{t}" for n in
                                         (256, 500, 864, 1372, 2048)
                                         for t in ("", "_b")]
    ETA = []
    for name in NAMES:
        r = run(name, 15)
        n = r["positions"].shape[1]
        temp = 2 * r["kinetic"].mean() / ((3 * n - 3) * KB)
        corr = transport.correlation(r["shear"], max_lag=2000) / 3
        ETA.append(transport.running_integral(corr, 10.0)
                   * r["volume"].mean() / (KB * temp) * PA_S)
    ETA = np.array(ETA)


    def green_kubo(upper_ps=3.0, which="all twelve"):
        k = round(upper_ps * 100)
        eta = ETA.mean(0) if which == "all twelve" else ETA[NAMES.index(which)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        lag = 10.0 * np.arange(len(GK_D)) / 1000
        ax1.plot(lag, GK_D * 1e4, color=viz.ACCENT)
        ax1.axvline(upper_ps, **viz.THRESHOLD_STYLE)
        ax1.set_xlabel("upper limit / ps")
        ax1.set_ylabel("D / 1e-4 Å² fs⁻¹")
        ax2.plot(10.0 * np.arange(len(eta)) / 1000, eta * 1e4,
                 color=viz.ACCENT)
        ax2.axvline(upper_ps, **viz.THRESHOLD_STYLE)
        ax2.axhline(2.18, **viz.REFERENCE_STYLE)
        ax2.set_xlabel("upper limit / ps")
        ax2.set_ylabel("η / 1e-4 Pa s")
        plt.show()
        print(f"read at {upper_ps:.1f} ps: D = {GK_D[k] * 1e4:.3f}e-4 Å²/fs,"
              f" η = {eta[k] * 1e4:.3f}e-4 Pa s ({which})")


    widgets.interact(green_kubo,
                     upper_ps=widgets.FloatSlider(3.0, min=0.1, max=20.0,
                                                  step=0.1, **SLOW),
                     which=["all twelve"] + NAMES);
    """),
    md(r"""
    The integral for $D$ overshoots near 0.32 ps,
    where $C_{vv}$ crosses zero, and settles near $3.78\times10^{-4}$ from
    about a picosecond. The viscosity of all twelve runs lies near
    $1.85\times10^{-4}$ Pa s from 1 to 3 ps, below Chapter 15's 2.18
    (dashed), and then drifts as noise accumulates, to about 1.7 by 20 ps;
    a single run's integral wanders by a tenth or more beyond a few
    picoseconds.
    """),
]

CHARGE = [
    md(r"""
    ## 16.7 Charge transport

    For each run of the molten salt: the MSD of the charge displacement
    $\sum_iq_i\mathbf{r}_i$, against the sum of the ions' own MSDs, which is
    what Nernst-Einstein assumes.
    """),
    code(r"""
    def salt_view(seed=0):
        r = run(f"salt_s{seed}")
        q, m = r["charges"], r["masses"]
        lags = np.arange(0, 2501, 25)
        t = 2.0 * lags
        total = transport.msd(r["charge_displacement"][:, None, :],
                              remove_drift=False, origin_step=5,
                              lags=lags).sum(1)
        pos = r["positions"].astype(float)
        frame = float(r["frame_dt"])
        own = sum(32 * transport.msd(pos, 50, masses=m,
                                     select=np.flatnonzero(q == s)).sum(1)
                  for s in (1.0, -1.0))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t / 1000, total, color=viz.ACCENT, label="charge displacement")
        ax.plot(frame * np.arange(51) / 1000, own, **viz.REFERENCE_STYLE,
                label="sum of the ions' own")
        ax.set_xlabel("t / ps")
        ax.set_ylabel("charge MSD / e² Å²")
        ax.legend()
        plt.show()
        fit = (t >= 1000) & (t <= 5000)
        slope = np.polyfit(t[fit], total[fit], 1)[0]
        own_slope = np.polyfit(frame * np.arange(51)[10:],
                               own[10:], 1)[0]
        temp = 2 * r["kinetic"].mean() / (189 * KB)
        print(f"T = {temp:.0f} K; slopes {slope:.4f} (charge) and "
              f"{own_slope:.4f} (own) e² Å²/fs: H_R ≈ {own_slope / slope:.2f}")


    widgets.interact(salt_view, seed=widgets.IntSlider(0, min=0, max=7));
    """),
    md(r"""
    In most runs the dashed sum rises a little
    faster than the charge displacement, and the ratio of the slopes, the
    Haven ratio, lies between about 0.97 and 1.14; the mean over the eight
    runs is $1.06 \pm 0.02$. One run on its own cannot show a difference of
    6%.
    """),
]

LAYERS = [
    md(r"""
    ## 16.8 Diffusion in layers

    The guests' MSD along each axis.
    """),
    code(r"""
    LAYERED = run("layered_150")
    GUESTS = LAYERED["guests"].astype(float) - LAYERED["centre"][:, None, :]
    PER_AXIS = transport.msd(GUESTS, 500, remove_drift=False)


    def layers_view(axis="x"):
        k = "xyz".index(axis)
        t = 100.0 * np.arange(501)
        d = transport.diffusion_coefficient(t, PER_AXIS[:, k], 5000.0,
                                            50000.0, dimensions=1)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t / 1000, PER_AXIS[:, k], color=viz.ACCENT)
        ax.set_xlabel("t / ps")
        ax.set_ylabel(rf"$\langle\Delta {axis}^2\rangle$ / Å²")
        plt.show()
        if axis == "z":
            print(f"⟨Δz²⟩ levels off at {PER_AXIS[-1, 2]:.3f} Å² by 50 ps: no "
                  f"guest leaves its gap, so the slope is no D_zz")
        else:
            print(f"D_{axis}{axis} = {d:.2e} Å²/fs")


    widgets.interact(layers_view, axis=["x", "y", "z"]);
    """),
    md(r"""
    Along $x$ and $y$ the MSD grows steadily, to
    about 11 Å² in 50 ps, and the two coefficients are near
    $1\times10^{-4}$ Å²/fs, agreeing within the errors of the chapter.
    Along $z$ it levels off near 0.13 Å² within a few picoseconds: the
    guests rattle between their sheets.
    """),
]

FOURIER = [
    md(r"""
    ## 16.9 Frequencies in a signal

    A note sampled at a rate you choose. Below twice its frequency the
    samples show a lower note, its alias.
    """),
    code(r"""
    def sample_note(rate=8000.0, note=659.25):
        t = np.arange(0, 0.05, 1 / rate)
        x = np.sin(2 * np.pi * note * t)
        n = 2 ** int(np.ceil(np.log2(len(x))) + 1)
        spectrum = np.abs(spectra.fft(np.concatenate(
            [x, np.zeros(n - len(x))])))
        freq = np.fft.fftfreq(n, 1 / rate)
        half = freq >= 0
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(freq[half], spectrum[half], color=viz.ACCENT)
        ax.axvline(note, **viz.REFERENCE_STYLE)
        ax.set_xlim(0, max(rate / 2, note * 1.2))
        ax.set_xlabel("frequency / Hz")
        ax.set_ylabel("transform magnitude")
        plt.show()
        seen = freq[half][np.argmax(spectrum[half])]
        print(f"Nyquist {rate / 2:.0f} Hz; the note appears at {seen:.0f} Hz")


    widgets.interact(sample_note,
                     rate=widgets.FloatSlider(8000, min=500, max=8000,
                                              step=50, **SLOW),
                     note=widgets.FloatSlider(659.25, min=100, max=2000,
                                              step=1, **SLOW));
    """),
    md(r"""
    While the rate is above twice the note, the peak
    sits on the dashed line. With the note between half the rate and the
    rate, the peak appears at the rate less the note, folded below the
    Nyquist frequency, and moves the wrong way as the note rises; a note
    above the rate folds again, to its distance from the nearest whole
    multiple of the rate. A note at a whole multiple of half the rate is
    sampled only at its zeros, and the spectrum shows nothing.
    """),
]

VDOS = [
    md(r"""
    ## 16.10 The vibrational density of states

    The chain's spectrum from frames a stride $\Delta t$ apart, with the
    correlation function taken to a lag you choose.
    """),
    code(r"""
    CHAIN = run("chain_dense")
    COPIES = CHAIN["velocities"].astype(float)
    COPIES = COPIES - COPIES.mean(axis=2, keepdims=True)
    V_CHAIN = COPIES.reshape(len(COPIES), -1, 3)
    NU1, DOS1 = spectra.vdos(V_CHAIN, 1.0, 2000)


    def vdos_view(stride=1, lag_fs=2000):
        nu, dos = spectra.vdos(V_CHAIN[::stride], float(stride),
                               max(4, lag_fs // stride))
        nyquist = THZ / (2 * stride)
        # every frame, with the same lag, so that only the stride differs
        nu_ref, dos_ref = spectra.vdos(V_CHAIN, 1.0, lag_fs)
        ref = np.interp(nu * THZ, nu_ref * THZ, dos_ref / THZ)
        keep = nu * THZ <= nyquist
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(nu_ref * THZ, dos_ref / THZ, **viz.REFERENCE_STYLE)
        ax.plot(nu * THZ, dos / THZ, color=viz.ACCENT)
        ax.axvline(nyquist, **viz.THRESHOLD_STYLE)
        ax.set_xlim(0, 30)
        ax.set_xlabel("ν / THz")
        ax.set_ylabel(r"$\mathcal{D}(\nu)$ / THz$^{-1}$")
        plt.show()
        score = spectra.overlap_score(dos[keep] / THZ, ref[keep])
        print(f"stride {stride} fs: Nyquist {nyquist:.1f} THz; overlap with "
              f"every frame {score:.3f}")


    widgets.interact(vdos_view,
                     stride=widgets.IntSlider(1, min=1, max=40, **SLOW),
                     lag_fs=widgets.IntSlider(2000, min=200, max=10000,
                                              step=100, **SLOW));
    """),
    md(r"""
    Up to a stride of about 20 fs the spectrum below
    its Nyquist frequency matches the dashed one and the overlap is near 1.
    Past 23.8 fs the bond stretch at 21.00 THz folds back below the limit,
    and past 27.4 fs the one at 18.25 THz too; peaks appear where nothing
    vibrates, and the overlap falls to about 0.8. A short lag smooths every
    peak into a broad hump; a long one sharpens them and makes them
    noisier. The overlap compares spectra taken to the same lag, so it
    measures the stride alone.
    """),
]

FES = [
    md(r"""
    ## 16.11 Free energy from a histogram

    The chain's dihedral free energy against the torsion energy, with error
    bars from blocks in time or from blocks of copies, after leaving out
    the start.
    """),
    code(r"""
    def fes_view(temperature=200, blocks="copies", skip_ps=200):
        psi = run(f"chain_{temperature}")["psi"].astype(float)
        kept = psi[int(skip_ps * 10):]
        order = kept.T.ravel() if blocks == "copies" else kept.ravel()
        (mid,), f, df = landscape.free_energy(
            order, 72, temperature, value_range=[(-np.pi, np.pi)])
        kt = KB * temperature
        fine = np.linspace(-np.pi, np.pi, 72 * 200 + 1)
        from mdlab import potentials
        w = np.exp(-potentials.opls_torsion(fine, ch16.CHAIN_TORSION) / kt)
        exact = -kt * np.log(np.array([w[k * 200:(k + 1) * 200 + 1].mean()
                                       for k in range(72)]))
        shift = np.average(f - exact, weights=1 / df**2)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.degrees(mid), exact * 1000, color="black", lw=0.6)
        ax.errorbar(np.degrees(mid), (f - shift) * 1000, 2 * df * 1000,
                    fmt="o", ms=2, color=viz.ACCENT)
        ax.set_xlabel("ψ / degrees")
        ax.set_ylabel("F / meV")
        plt.show()
        ratio = np.mean(((f - exact - shift) / df) ** 2)
        print(f"{temperature} K, blocks of {blocks}: squared misses over "
              f"squared errors {ratio:.2f} per bin")


    widgets.interact(fes_view, temperature=[300, 200],
                     blocks=["copies", "time"],
                     skip_ps=widgets.IntSlider(200, min=0, max=1000,
                                               step=100, **SLOW));
    """),
    md(r"""
    At 300 K the points fall on the line within their
    errors with either kind of block and any start, the squared misses
    over squared errors between about 0.5 and 1 per bin. At 200 K, with
    blocks in time, the ratio lies between about 2 and 3.4: the gauche
    wells, which each copy visits only about once in 250 ps, sit apart
    from the line by more than their bars. Blocks of copies give larger
    bars; leaving out more of the start brings the ratio towards 1, while
    keeping the start raises it, since every copy shares the trans well it
    began in.
    """),
]

TRAJECTORY = [
    md(r"""
    ## 16.12 What the trajectory looks like

    Each fault of the chapter's table, against the correct curve.
    """),
    code(r"""
    def fault(name="(a) wrapped positions"):
        fig, ax = plt.subplots(figsize=(6, 3))
        sparse = PATHS[::10]
        t = 100.0 * np.arange(1001)
        good = transport.msd(sparse, 1000, masses=M_ARGON,
                             origin_step=5).sum(1)
        if name.startswith("(a)"):
            wrapped = np.array([cell.wrap(x, ARGON["cell"]) for x in sparse])
            bad = transport.msd(wrapped, 1000, masses=M_ARGON,
                                origin_step=5).sum(1)
            ax.plot(t / 1000, good, color=viz.ACCENT)
            ax.plot(t / 1000, bad, color=viz.OXBLOOD)
            ax.set_ylabel("MSD / Å²")
            jump = diagnostics.largest_jump(wrapped, ARGON["cell"])
            true = diagnostics.largest_jump(sparse, ARGON["cell"])
            note = (f"at 20 ps: {bad[200]:.0f} against {good[200]:.0f} Å²; "
                    f"the check, the largest move between frames, is "
                    f"{jump:.2f} cell widths against {true:.3f}")
        elif name.startswith("(b)"):
            kt_m = KB * 132.1 / (39.948 * units.MV2_TO_EV)
            drift = math.sqrt(3 * kt_m / 256)
            moved = sparse + drift * t[:, None, None] * np.array([1, 0, 0])
            bad = transport.msd(moved, 1000, remove_drift=False,
                                origin_step=5).sum(1)
            ax.plot(t / 1000, good, color=viz.ACCENT)
            ax.plot(t / 1000, bad, color=viz.OXBLOOD)
            ax.set_xlim(0, 30)
            ax.set_ylim(0, 150)
            ax.set_ylabel("MSD / Å²")
            path = diagnostics.centre_of_mass_path(M_ARGON, moved)[-1]
            note = (f"the drift adds a t² term; the check, the path of the "
                    f"centre of mass, has carried it {path:.0f} Å by 100 ps")
        else:
            lang = run("argon_langevin")
            c_l = transport.velocity_autocorrelation(
                lang["velocities"].astype(float), 200)
            ax.plot(10 * np.arange(201), C_VV[:201] / C_VV[0],
                    color=viz.ACCENT)
            ax.plot(10 * np.arange(201), c_l / c_l[0], color=viz.OXBLOOD)
            ax.set_ylabel("C_vv(t)/C_vv(0)")
            # D from each running integral averaged over 2 to 10 ps
            d_l = transport.running_integral(
                transport.velocity_autocorrelation(
                    lang["velocities"].astype(float), 1000), 10.0)[200:] / 3
            cut = 1 - d_l.mean() / GK_D[200:1001].mean()
            note = f"friction of 10 ps⁻¹ cuts D by {100 * cut:.0f}%"
        ax.set_xlabel("t / ps" if not name.startswith("(c)") else "t / fs")
        plt.show()
        print(note)


    widgets.interact(fault, name=["(a) wrapped positions",
                                  "(b) a drifting centre of mass",
                                  "(c) Langevin friction"]);
    """),
    md(r"""
    (a) The wrapped MSD shoots up and then levels off
    towards 270 Å², $L^2/2$, while the true one keeps rising steadily; its
    check, the largest move between frames, is more than a whole cell width
    against a few hundredths. (b) The drifting run curves upwards away from
    the straight line, and the centre of mass, which should stay put,
    wanders by nearly 20 Å. (c) Under friction the velocity correlation
    falls to half in 60 fs instead of 160 and crosses zero at 170 fs
    instead of 320. The other faults of the chapter's table have their
    widgets in 16.1, 16.3, 16.5, 16.6, 16.10 and 16.11.
    """),
]

EXERCISES = [
    md(r"""
    ## Working notes

    For each observable, record the trajectory, discarded interval and
    analysis window. Keep a short note of the feature used to choose a
    cutoff, fit range or integration limit. If two sensible choices disagree,
    show that disagreement before reporting a single value.
    """),
    md(r"""
    ## Exercises

    Each exercise cell checks its answer against the key below; the
    collapsed cell after it holds a worked solution.
    """),
    hidden(r"""
    # The answer key. Each target is computed from the exercise's data.
    MV2 = units.MV2_TO_EV
    TARGETS = {
        "16.1": np.array([1 - 4 * math.pi * 0.05 / (4 / 3 * math.pi
                                                    * (1.05**3 - 1)),
                          1 - 4 * math.pi * 25 * 0.05
                          / (4 / 3 * math.pi * (5.05**3 - 125))]),
        "16.2": np.array([1 - 1 / 256, 1 - 1 / 2048]),
        "16.4": -16 / 9 * math.pi * 0.8 * 0.01034,
        "16.6": 0.03 * KB * 300 * 0.5 * units.EV_PER_A3_TO_GPA,
        "16.7": 2 * 1000 / 60,
        "16.8": (np.array([15, 29]), np.array([8]),
                 np.array([[0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 1, 1]])),
        "16.10": np.array([2500.0, 4.0e-4, math.sqrt(40) / 1e5]),
        "16.12": (3.46 - 1) / (3.46 + 1),
        "16.13": 2 * 5.24e-4 * 18.015 * MV2 / (KB * 304),
        "16.15": (256 / 0.02035) / (KB * 135) * (0.0121
                 / units.EV_PER_A3_TO_GPA) ** 2 * 180 * PA_S,
        "16.16": units.ELEMENTARY_CHARGE * 32 * (5.19e-4 + 5.03e-4) * 1e-5
        / (13.4**3 * 1e-30 * KB * 1487),
        "16.14": 1 / 3,
        "16.19": np.where(np.isin(np.arange(16), [3, 13]), 8.0, 0.0),
        "16.20": 2**20 / 20,
        "16.21": np.array([2 * 1000 / 70 - 18.25, 1000 / (2 * 18.25)]),
        "16.23": 12 * 15.999 * MV2 * 5.24e-4 / (KB * 304) / 1000,
        "16.24": math.sqrt(KB * 300 / 0.9785),
        "16.25": 400.0,
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 16.1: the fraction by which 4πr²δr falls short of the exact
    # shell volume for δr = 0.05 Å, at r = 1 and 5 Å.
    short = None

    check(short, TARGETS["16.1"], rtol=1e-3, name="shortfall")
    """),
    solution(r"""
    # SOLUTION 16.1. Exact (4/3)π[(r + δr)³ − r³] against 4πr²δr.
    short = np.array([1 - 4 * math.pi * r * r * 0.05
                      / (4 / 3 * math.pi * ((r + 0.05) ** 3 - r**3))
                      for r in (1.0, 5.0)])

    check(short, TARGETS["16.1"], rtol=1e-3, name="shortfall");
    """),
    code(r"""
    # EXERCISE 16.2: the far value of g(r) for 256 and 2048 atoms placed
    # independently at random.
    far = None

    check(far, TARGETS["16.2"], rtol=1e-9, name="far value")
    """),
    solution(r"""
    # SOLUTION 16.2. 1 − 1/N.
    far = np.array([1 - 1 / 256, 1 - 1 / 2048])

    check(far, TARGETS["16.2"], rtol=1e-9, name="far value");
    """),
    code(r"""
    # EXERCISE 16.4: U/N (eV) with g a step at σ, for ρσ³ = 0.8 and
    # ε = 0.01034 eV.
    u_step = None

    check(u_step, TARGETS["16.4"], rtol=1e-4, name="U/N")
    """),
    solution(r"""
    # SOLUTION 16.4. −(16/9)πρσ³ε.
    u_step = -16 / 9 * math.pi * 0.8 * 0.01034

    check(u_step, TARGETS["16.4"], rtol=1e-4, name="U/N");
    """),
    code(r"""
    # EXERCISE 16.6: S(0) for κ_T = 0.5 GPa⁻¹, ρ = 0.03 Å⁻³, 300 K.
    s0 = None

    check(s0, TARGETS["16.6"], rtol=1e-3, name="S(0)")
    """),
    solution(r"""
    # SOLUTION 16.6. ρ k_BT κ_T, with κ_T in Å³/eV.
    s0 = 0.03 * KB * 300 * 0.5 * units.EV_PER_A3_TO_GPA

    check(s0, TARGETS["16.6"], rtol=1e-3, name="S(0)");
    """),
    code(r"""
    # EXERCISE 16.7: events per ps from one pair vibrating with a period of
    # 60 fs across a single cutoff.
    per_ps = None

    check(per_ps, TARGETS["16.7"], rtol=1e-6, name="events per ps")
    """),
    solution(r"""
    # SOLUTION 16.7. Two crossings a period.
    per_ps = 2 * 1000 / 60

    check(per_ps, TARGETS["16.7"], rtol=1e-6, name="events per ps");
    """),
    code(r"""
    # EXERCISE 16.8: the keys formed and broken between the two frames,
    # and each frame's molecule labels from bonds.molecules.
    formed, broken, labels = None, None, None

    for got, want, label in zip((formed, broken, labels), TARGETS["16.8"],
                                ("formed", "broken", "molecules")):
        check(got, want, rtol=0, name=label)
    """),
    solution(r"""
    # SOLUTION 16.8. Keys iN + j with N = 6; set differences of sorted keys.
    old = np.array([0 * 6 + 1, 1 * 6 + 2, 3 * 6 + 4])
    new = np.array([0 * 6 + 1, 2 * 6 + 3, 3 * 6 + 4, 4 * 6 + 5])
    formed, broken = np.setdiff1d(new, old), np.setdiff1d(old, new)
    labels = np.array([bonds.molecules(6, old), bonds.molecules(6, new)])

    for got, want, label in zip((formed, broken, labels), TARGETS["16.8"],
                                ("formed", "broken", "molecules")):
        check(got, want, rtol=0, name=label);
    """),
    code(r"""
    # EXERCISE 16.10: the events for 2%, and the rate (per ps) with its
    # error for 40 dissociations of 100 molecules in 1 ns.
    poisson = None

    check(poisson, TARGETS["16.10"], rtol=1e-6, name="events, rate, error")
    """),
    solution(r"""
    # SOLUTION 16.10. n = 1/0.02²; k = n/X and √n/X with X = 1e5 molecule-ps.
    poisson = np.array([1 / 0.02**2, 40 / 1e5, math.sqrt(40) / 1e5])

    check(poisson, TARGETS["16.10"], rtol=1e-6, name="events, rate, error");
    """),
    code(r"""
    # EXERCISE 16.12: the step correlation c that gives a memory of
    # direction (1 + c)/(1 − c) = 3.46.
    c = None

    check(c, TARGETS["16.12"], rtol=1e-6, name="c")
    """),
    solution(r"""
    # SOLUTION 16.12. (1 + c)/(1 − c) = 3.46, so c = (3.46 − 1)/(3.46 + 1).
    c = (3.46 - 1) / (3.46 + 1)

    check(c, TARGETS["16.12"], rtol=1e-6, name="c");
    """),
    code(r"""
    # EXERCISE 16.13: 2Dm/k_BT (fs) for water, D = 5.24e-4 Å²/fs,
    # m = 18.015 amu, 304 K.
    crossover = None

    check(crossover, TARGETS["16.13"], rtol=1e-3, name="crossover")
    """),
    solution(r"""
    # SOLUTION 16.13. m in eV fs²/Å² is m × MV2_TO_EV.
    crossover = 2 * 5.24e-4 * 18.015 * units.MV2_TO_EV / (KB * 304)

    check(crossover, TARGETS["16.13"], rtol=1e-3, name="crossover");
    """),
    code(r"""
    # EXERCISE 16.14: with SymPy, the MSD of an exponential memory
    # C_0 e^{−s/τ_c} from 2∫(t − s)C ds, and D/(C_0 τ_c) from its slope.
    d_over = None

    check(d_over, TARGETS["16.14"], rtol=1e-12, name="D/(C_0 τ_c)")
    """),
    solution(r"""
    # SOLUTION 16.14. The integral, its two limits, and the slope over 6.
    import sympy as sp

    t, s, tau, c0 = sp.symbols("t s tau_c C_0", positive=True)
    msd_exp = 2 * sp.integrate((t - s) * c0 * sp.exp(-s / tau), (s, 0, t))
    assert sp.simplify(msd_exp - 2 * c0 * tau
                       * (t - tau * (1 - sp.exp(-t / tau)))) == 0
    print("small t:", sp.series(msd_exp, t, 0, 3).removeO())
    slope = sp.limit(sp.diff(msd_exp, t), t, sp.oo)
    d_over = float(slope / 6 / (c0 * tau))

    check(d_over, TARGETS["16.14"], rtol=1e-12, name="D/(C_0 τ_c)");
    """),
    code(r"""
    # EXERCISE 16.15: η ≈ (V/k_BT) σ_xy² τ_int (Pa s) for 256 atoms at
    # 0.02035 Å⁻³ and 135 K, σ_xy = 0.0121 GPa, τ_int = 180 fs.
    eta = None

    check(eta, TARGETS["16.15"], rtol=1e-3, name="η")
    """),
    solution(r"""
    # SOLUTION 16.15. Everything in eV, Å and fs, then to Pa s.
    volume = 256 / 0.02035
    sigma = 0.0121 / units.EV_PER_A3_TO_GPA
    eta = volume / (KB * 135) * sigma**2 * 180 * PA_S

    check(eta, TARGETS["16.15"], rtol=1e-3, name="η");
    """),
    code(r"""
    # EXERCISE 16.16: σ_NE (S/m) for the salt.
    sigma_ne = None

    check(sigma_ne, TARGETS["16.16"], rtol=1e-3, name="σ_NE")
    """),
    solution(r"""
    # SOLUTION 16.16. e² Σ D_i/(V k_BT) in SI; q_i² = e² for every ion.
    sum_d = 32 * (5.19e-4 + 5.03e-4) * 1e-5  # m²/s
    sigma_ne = (units.ELEMENTARY_CHARGE * sum_d
                / (13.4**3 * 1e-30 * KB * 1487))

    check(sigma_ne, TARGETS["16.16"], rtol=1e-3, name="σ_NE");
    """),
    code(r"""
    # EXERCISE 16.19: spectra.dft of cos(2π·3n/16), n = 0, ..., 15.
    x_hat = None

    check(x_hat, TARGETS["16.19"], atol=1e-9, name="transform")
    """),
    solution(r"""
    # SOLUTION 16.19. N_s/2 = 8 at k = 3 and k = 16 − 3, zero elsewhere.
    x_hat = spectra.dft(np.cos(2 * np.pi * 3 * np.arange(16) / 16))

    check(x_hat, TARGETS["16.19"], atol=1e-9, name="transform");
    """),
    code(r"""
    # EXERCISE 16.20: N_s² / (N_s log₂N_s) for N_s = 2^20.
    ratio = None

    check(ratio, TARGETS["16.20"], rtol=1e-9, name="ratio")
    """),
    solution(r"""
    # SOLUTION 16.20. N_s / log₂N_s.
    ratio = 2**20 / 20

    check(ratio, TARGETS["16.20"], rtol=1e-9, name="ratio");
    """),
    code(r"""
    # EXERCISE 16.21: where 18.25 THz appears with frames 35 fs apart, and
    # the longest stride (fs) that shows it where it is.
    fold = None

    check(fold, TARGETS["16.21"], rtol=1e-6, name="alias, stride")
    """),
    solution(r"""
    # SOLUTION 16.21. Folded to 1/Δt − ν, with 1/Δt = 1000/35 THz.
    fold = np.array([2 * 1000 / 70 - 18.25, 1000 / (2 * 18.25)])

    check(fold, TARGETS["16.21"], rtol=1e-6, name="alias, stride");
    """),
    code(r"""
    # EXERCISE 16.23: 𝒟(0) per THz for water's oxygen atoms.
    zero = None

    check(zero, TARGETS["16.23"], rtol=1e-3, name="D(0)")
    """),
    solution(r"""
    # SOLUTION 16.23. 12 m D/k_BT in fs, then per THz is ÷ 1000.
    zero = 12 * 15.999 * units.MV2_TO_EV * 5.24e-4 / (KB * 304) / 1000

    check(zero, TARGETS["16.23"], rtol=1e-3, name="D(0)");
    """),
    code(r"""
    # EXERCISE 16.24: where F(r_h) is least (Å) at 300 K for k = 0.9785.
    r_least = None

    check(r_least, TARGETS["16.24"], rtol=1e-4, name="r_h")
    """),
    solution(r"""
    # SOLUTION 16.24. √(k_BT/k).
    r_least = math.sqrt(KB * 300 / 0.9785)

    check(r_least, TARGETS["16.24"], rtol=1e-4, name="r_h");
    """),
    code(r"""
    # EXERCISE 16.25: the crossings that give F to 0.1 k_BT.
    crossings = None

    check(crossings, TARGETS["16.25"], rtol=1e-9, name="crossings")
    """),
    solution(r"""
    # SOLUTION 16.25. 2/√n = 0.1.
    crossings = (2 / 0.1) ** 2

    check(crossings, TARGETS["16.25"], rtol=1e-9, name="crossings");
    """),
]

CELLS = (SETUP + RDF + SQ + BONDS + REACTIONS + MSD + VACF + CHARGE + LAYERS
         + FOURIER + VDOS + FES + TRAJECTORY + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "16_observables.ipynb"))
