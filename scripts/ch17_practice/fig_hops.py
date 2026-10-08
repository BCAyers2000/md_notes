"""Figure fig:ip-hops: lithium on the move in dilute graphite.

scan_li.py's two paths of lithium between sites, scan_map.py's map with
the carbons held, and runs_mace.py's dilute_300, dilute_450 and
dilute_600 (one lithium among 64 carbon atoms, CSVR, 20 ps, lithium's
position every 10 fs, every atom's every 100 fs). Lithium's sites are
the hollows of the two sheets bounding its gallery, two families each
moving with its own sheet: the sheets slide, and each sheet's slide is
taken from the frames and interpolated. (a) The energy along the
straight path to the nearest site of the other family (a/√3) and along
the path to the next site of the same family across a C-C bond (a),
every atom but lithium and one carbon of each sheet relaxed, those held
in the plane, with D3. Hops are counted from the start that Chapter 15's
test finds in each run's kinetic temperature, since the runs start from
the relaxed cell and take a few picoseconds to warm. (b)
Hops against the least stay a visit needs to count, at each temperature,
with sites over the hollows of both sheets (solid) and of the sheet
below alone (dashed), and the 25 hops that resolve a rate to a fifth
(dotted). (c) The rate of hops from a least stay of 50 fs, with its
Poisson error (open: fewer than 25 hops), against 1/k_BT, and the line
exp(−E/k_BT) through the middle point for E the highest point of the
shorter path of (a).

Prints the highest points of both paths, the energy along the shorter
path on the map with the carbons held, the slides of the sheets, the
start of each run, the counts in ten windows against those of
independent hops, the counts with each definition of a site and with
both families fixed in the cell where the sheets started, the rates
with their errors or the exposure still needed, the fraction of hops
that return to the site just left, the slope of ln k against 1/k_BT
with its error, D from lithium's MSD in the frame of the gallery's mean
over 0.5 to 2 ps in four blocks against k_h ℓ²/4, and lithium's carbon
neighbours by the bond tracker of Section 16.3 with its default
thresholds, 2.45 and 2.86 Å for Li-C.
"""

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ch17 import DATA, RUNS, hollows, li_sites, sheets
from matplotlib.ticker import NullFormatter

from mdlab import bonds, units, viz
from mdlab.analysis import hops, stats, transport
from mdlab.cell import minimum_image
from mdlab.diagnostics import unwrap
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, THRESHOLD_STYLE

TEMPERATURES = (300, 450, 600)
COLOURS = {300: ACCENT, 450: OCHRE, 600: OXBLOOD}
STAYS = np.array([1, 2, 3, 5, 7, 10, 15, 20])  # frames of 10 fs
STAY = 5  # the least stay used for rates, 50 fs
WINDOWS = 10  # for the test of independent hops
FREEDOMS = 3 * 65 - 3  # the drift was removed at the start

scan = np.load(RUNS / "li_scan.npz")
for path in ("hollow", "bridge"):
    print(f"{path} path, {float(scan[path + '_length']):.4f} Å: highest "
          f"{1000 * scan[path + '_energy'].max():.1f} meV with D3, "
          f"{1000 * scan[path + '_learned'].max():.1f} without")
barrier = scan["hollow_energy"].max()


def held_path():
    """The shorter path on scan_map.py's map, with the carbons held.

    The map's grid, 12 points to a cell, covers one cell of the sheet with
    lithium's first site at (4, 8). The host was relaxed around that site,
    so the map is not periodic: of the three nearest sites of the other
    family, the hollows of the sheet above, the path goes to the one whose
    straight line stays on the grid.
    """
    run = np.load(RUNS / "dilute_300.npz")
    cell = run["cell"][0]
    above = sheets(run)[1]
    li = run["frames"][0, 64] * [1, 1, 0]
    sites = hollows(run["frames"][0, :64][above, :2], cell)
    gap = minimum_image(sites - li, cell.T)[:, :2]
    nearest = np.argsort(np.linalg.norm(gap, axis=1))[:3]
    steps = gap[nearest] @ np.linalg.inv(cell[:2, :2] / 4)
    ends = np.round([4, 8] + 12 * steps).astype(int)
    toward = steps[np.all((ends >= 0) & (ends < 12), axis=1)][0]
    held = np.load(RUNS / "li_map.npz")["energy"]
    return [held[tuple(np.round([4, 8] + 12 * f * toward).astype(int))]
            for f in np.linspace(0, 1, 5)]


relaxed = read(DATA / "structures" / "dilute.extxyz")
gap = minimum_image(relaxed.positions[:-1] - relaxed.positions[-1],
                    np.array(relaxed.get_cell()).T)
near = np.argsort(np.linalg.norm(gap, axis=1))[:10]
print("relaxed at 0 K, lithium's ten nearest carbon atoms: " + ", ".join(
    f"{np.linalg.norm(gap[k]):.3f} Å {'above' if gap[k, 2] > 0 else 'below'}"
    for k in near))
along = held_path()
print(f"with the carbons held: the other family's site "
      f"{1000 * along[-1]:.0f} meV, the way to it "
      + ", ".join(f"{1000 * e:.0f}" for e in along) + " meV")


counts, single, rates = {}, {}, {}
for t in TEMPERATURES:
    run = np.load(RUNS / f"dilute_{t}.npz")
    cell = run["cell"][0]
    both, one = li_sites(run), li_sites(run, families=1)
    moves = both["moves"]
    print(f"{t} K: the sheet below slides up to "
          f"{np.linalg.norm(moves[0], axis=1).max():.2f} Å, the sheet above "
          f"moves up to "
          f"{np.linalg.norm(moves[1] - moves[0], axis=1).max():.2f} Å "
          "against it")
    # the runs start from the relaxed cell: count from the start that
    # Chapter 15's test finds in the kinetic temperature, kept every 10 fs
    kinetic_t = 2 * run["kinetic"] / (FREEDOMS * units.KB)
    t0 = stats.detect_equilibration(kinetic_t)[0]
    print(f"   the first ps at {kinetic_t[:100].mean():.0f} K; start found "
          f"at {run['times'][t0] / 1000:.2f} ps, "
          f"{kinetic_t[t0:].mean():.0f} K after it")
    seq = both["sites"][t0:]
    counts[t] = np.array([len(hops.hops(seq, s)) for s in STAYS])
    single[t] = np.array([len(hops.hops(one["sites"][t0:], s))
                          for s in STAYS])
    fixed = li_sites(run, follow=False)["sites"][t0:]
    print("   with both families fixed in the cell where the sheets "
          "started: " + ", ".join(str(len(hops.hops(fixed, s)))
                                  for s in STAYS))
    span = (run["times"][-1] - run["times"][t0]) / 1000  # ps
    seen = hops.hops(seq, STAY)
    rates[t] = r = hops.rate(len(seen), span) | {"n": len(seen)}
    window = np.histogram(seen[:, 0], np.linspace(0, len(seq),
                                                  WINDOWS + 1))[0]
    whole = np.histogram(hops.hops(both["sites"], STAY)[:, 0],
                         np.linspace(0, len(both["sites"]), WINDOWS + 1))[0]
    print(f"   hops in {WINDOWS} windows: "
          + " ".join(str(c) for c in window)
          + f"; variance over mean {window.var(ddof=1) / window.mean():.2f},"
          f" 1 ± {np.sqrt(2 / (WINDOWS - 1)):.2f} for independent hops; "
          f"over the whole run, warming included, "
          f"{whole.var(ddof=1) / whole.mean():.2f}")
    verdict = (f"{r['rate']:.2f} ± {r['error']:.2f} per ps" if r["resolved"]
               else f"{r['rate']:.2f} ± {r['error']:.2f} per ps, unresolved, "
               f"{r['needed']:.0f} ps needed")
    print(f"   over the hollows of the sheet below "
          f"{np.mean(seq < 16):.2f} of the time; hops for least "
          "stays of 10 to 200 fs: " + ", ".join(str(c) for c in counts[t])
          + "; with the sheet below alone: "
          + ", ".join(str(c) for c in single[t]))
    back = np.mean(seen[1:, 2] == seen[:-1, 1])
    print(f"   at 50 fs {len(seen)} in {span:.1f} ps, {verdict}; "
          f"{np.mean((seen[:, 1] < 16) != (seen[:, 2] < 16)):.2f} between "
          f"families; back to the site just left {back:.2f} ± "
          f"{np.sqrt(back * (1 - back) / (len(seen) - 1)):.2f} (1/3 at "
          "random)")
    gallery = run["li"].copy()
    gallery[:, :2] -= (moves[0] + moves[1]) / 2
    gallery[:, 2] = 0.0
    path = unwrap(gallery[t0:, None, :], cell.T)
    ds = []
    for block in np.array_split(path, 4):
        msd = transport.msd(block, 200, remove_drift=False).sum(1)
        ds.append(transport.diffusion_coefficient(
            10.0 * np.arange(201), msd, 500, 2000, dimensions=2))
    ds = np.array(ds)
    spread = ds.std(ddof=1) / 2 / ds.mean()
    step = np.linalg.norm(cell[0]) / 4 / np.sqrt(3)
    print(f"   D from the MSD ({ds.mean() * 1e4:.1f} ± "
          f"{ds.std(ddof=1) / 2 * 1e4:.1f})e-4 Å²/fs, a fractional error of "
          f"{spread:.2f}, {(spread / 0.05)**2:.0f} times the run for 5%; "
          f"k_h ℓ²/4 = "
          f"{r['rate'] / 1000 * step**2 / 4 * 1e4:.1f}e-4 with ℓ = "
          f"{step:.4f} Å")
    tracker = bonds.BondTracker(["C"] * 64 + ["Li"])
    bonded = []
    for frame in run["frames"][t0 // 10:]:
        gap = np.linalg.norm(minimum_image(frame[:64] - frame[64], cell.T),
                             axis=1)
        close = np.flatnonzero(gap < tracker.reach)
        tracker.update(close, np.full(len(close), 64), gap[close])
        bonded.append(len(tracker.keys))
    print(f"   lithium bonded to 10 carbon atoms in "
          f"{np.mean(np.array(bonded) == 10):.2f} of the frames, 9 in "
          f"{np.mean(np.array(bonded) == 9):.2f}, fewer in "
          f"{np.mean(np.array(bonded) < 9):.2f}; a visit lasts "
          f"{span / len(seen):.2f} ps on average")

beta = np.array([1 / (units.KB * t) for t in TEMPERATURES])
k = np.array([rates[t]["rate"] for t in TEMPERATURES])
e = np.array([rates[t]["error"] for t in TEMPERATURES])
n = np.array([rates[t]["n"] for t in TEMPERATURES])
fit, cov = np.polyfit(beta, np.log(k), 1, w=np.sqrt(n), cov="unscaled")
print(f"ln k against 1/k_BT, weighted by the counts: slope "
      f"{fit[0]:.3f} ± {np.sqrt(cov[0, 0]):.3f} eV, an apparent activation "
      f"energy of ({-fit[0] * 1000:.0f} ± {np.sqrt(cov[0, 0]) * 1000:.0f})"
      f" meV against the shorter path's {1000 * barrier:.0f}; the rate rises "
      f"{k[-1] / k[0]:.1f} times from 300 to 600 K, "
      f"{np.exp(barrier * (beta[0] - beta[-1])):.1f} times for that barrier")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.1),
                         gridspec_kw=dict(wspace=0.5))
ax = axes[0]
for path, colour, mark, label in (
        ("hollow", ACCENT, "o-", "to the nearest site"),
        ("bridge", OCHRE, "s--", "across a bond")):
    ax.plot(scan["fraction"] * float(scan[path + "_length"]),
            scan[path + "_energy"] * 1000, mark, ms=3, lw=0.9, color=colour,
            label=label)
ax.set_ylim(-5, 150)
ax.set_xlabel(r"distance along the path / \AA")
ax.set_ylabel("energy / meV")
ax.legend(fontsize=7, loc="upper center")
viz.panel_tag(ax, "a")

ax = axes[1]
for t in TEMPERATURES:
    ax.plot(STAYS * 10, counts[t], "o-", ms=3, lw=0.8, color=COLOURS[t],
            label=f"{t} K")
    ax.plot(STAYS * 10, single[t], "--", lw=0.7, color=COLOURS[t], alpha=0.6)
ax.axhline(25, **THRESHOLD_STYLE)
ax.set_xlabel("least stay / fs")
ax.set_ylabel("hops counted")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "b")

ax = axes[2]
for b, rate, err, count in zip(beta, k, e, n, strict=True):
    ax.errorbar(b, rate, err, fmt="o", ms=3, color=ACCENT, capsize=0,
                elinewidth=0.8, mfc=ACCENT if count >= 25 else "white")
line = np.linspace(beta.min() * 0.95, beta.max() * 1.05, 50)
ax.plot(line, k[1] * np.exp(-barrier * (line - beta[1])), **THRESHOLD_STYLE)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"$1/k_{\mathrm{B}}T$ / eV$^{-1}$")
ax.set_ylabel("hops per ps")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch17_practice", "hops.pdf")))
