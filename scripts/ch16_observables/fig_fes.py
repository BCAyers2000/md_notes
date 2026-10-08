"""Figure fig:ob-fes: free energy from where a run spends its time.

(a) Lithium on the model surface at 600 K (runs.py, surface_600: 1000
atoms, 200 ps, every 50 fs), positions folded into one cell of the
hollows: F(x, y) = −k_BT ln P (shading) and contours of U every 0.1 eV
(lines). (b) The same atoms' free energy against d, the distance to the
nearest hollow (points, with twice their error from 10 blocks), against
the exact −k_BT ln[d ∮e^{−βU}dθ] (line) and U along the line from a
hollow to a bridge (dashed). (c) The dihedral angle of the four-bead
chain (chain_300 and chain_200: 200 copies, 2 ns): F(ψ) at 300 K and
200 K, with twice their error, against the torsion energy (line).

Prints the agreement of F with U in two dimensions, the minimum of F(r_h)
and where U along the path reaches the same value, the chain's
crossings per copy and the agreement of F(ψ) with the torsion energy.
"""

import matplotlib.pyplot as plt
import numpy as np
from ch16 import CHAIN_TORSION, RUNS, ch13, ch15

from mdlab import energy, potentials, units, viz
from mdlab.analysis import landscape, spectra
from mdlab.viz import ACCENT, OCHRE, REFERENCE

T_SURF = 600.0
#: The curvature of U at a hollow, 3U_b|g|²/8 (Section 16.10), eV/Å².
K_HOLLOW = 3 * 0.3 * (4 * np.pi / (np.sqrt(3) * ch13.SPACING)) ** 2 / 8
kt = units.KB * T_SURF
surface = np.load(RUNS / "surface_600.npz")
paths = surface["positions"].astype(float)
points = paths.reshape(-1, 2)
fractional = points @ np.linalg.inv(ch13.LATTICE)
folded = (fractional - np.floor(fractional)) @ ch13.LATTICE
(xs, ys), f2, df2 = landscape.free_energy(
    folded, (60, 52), T_SURF,
    value_range=[(0, 1.5 * ch13.SPACING), (0, ch13.SPACING * np.sqrt(3) / 2)])
grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)
u2, _ = energy.hexagonal_surface(grid)
inside = (grid @ np.linalg.inv(ch13.LATTICE))
inside = np.all((inside > 0.02) & (inside < 0.98), axis=-1) & np.isfinite(f2)
inside &= u2 < 0.3
surface_shift = np.average((f2 - u2)[inside], weights=1 / df2[inside] ** 2)
dev = (f2 - u2 - surface_shift)[inside] / df2[inside]
print(f"F(x, y) − U, below 0.3 eV: χ² per bin {np.mean(dev**2):.2f} over "
      f"{inside.sum()} bins; largest |F − U − c| "
      f"{np.abs(f2 - u2 - surface_shift)[inside].max() * 1000:.1f} meV")

site, dist = ch13.nearest_hollow(paths)
# Out to a/2, where a circle about a hollow still lies wholly in the
# region nearer that hollow than any other.
(d_mid,), f_d, df_d = landscape.free_energy(dist.ravel(), 40, T_SURF,
                                            value_range=[(0, 1.2)])
theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
exact = []
for d in d_mid:
    ring = d * np.column_stack([np.cos(theta), np.sin(theta)])
    u_ring, _ = energy.hexagonal_surface(ring)
    exact.append(-kt * np.log(d * np.mean(np.exp(-u_ring / kt))))
exact = np.array(exact)
ok = np.isfinite(f_d)
offset = np.average((f_d - exact)[ok], weights=1 / df_d[ok] ** 2)
exact = exact + offset
dev_d = (f_d - exact)[ok] / df_d[ok]
line = np.column_stack([d_mid, np.zeros_like(d_mid)])
u_line, _ = energy.hexagonal_surface(line)
low = np.argmin(f_d)
print(f"F(r_h): least at r_h = {d_mid[low]:.3f} Å; χ² per bin against the "
      f"exact curve {np.mean(dev_d**2):.2f} over {ok.sum()} bins; F at the "
      f"smallest r_h is {(f_d[0] - f_d[low]) * 1000:.0f} meV above its least")
print(f"harmonic estimate of the least: r_h = √(k_BT/k) with k = "
      f"3U_b|g|²/8: {np.sqrt(kt / K_HOLLOW):.3f} Å")

SKIP = 2000  # frames, 200 ps: every copy starts in the trans well
chains = {}
for temperature in (300, 200):
    run = np.load(RUNS / f"chain_{temperature}.npz")
    psi = run["psi"].astype(float)
    kt_c = units.KB * temperature
    basin = np.where(np.abs(psi) > np.radians(120), 0, np.sign(psi))
    crossings = np.sum(basin[1:] != basin[:-1])
    span = len(psi) * float(run["frame_dt"]) / 1e6  # ns
    kept = psi[SKIP:]
    skip_ps = SKIP * float(run["frame_dt"]) / 1000
    # the exact_psi answer averaged over each bin, as a histogram's bin is
    fine = np.linspace(-np.pi, np.pi, 72 * 200 + 1)
    weight = np.exp(-potentials.opls_torsion(fine, CHAIN_TORSION) / kt_c)
    exact_psi = -kt_c * np.log(np.array(
        [weight[k * 200:(k + 1) * 200 + 1].mean() for k in range(72)]))
    for label, order in (("blocks in time", kept.ravel()),
                         ("blocks of copies", kept.T.ravel())):
        (mid,), f, df = landscape.free_energy(order, 72, temperature,
                                              value_range=[(-np.pi, np.pi)])
        shift = np.average(f - exact_psi, weights=1 / df**2)
        dev = (f - exact_psi - shift) / df
        print(f"chain at {temperature} K, {label} (10 blocks, after "
              f"{skip_ps:.0f} ps): F − exact, "
              f"χ² per bin {np.mean(dev**2):.2f} over 72 bins; median error "
              f"{np.median(df) * 1000:.2f} meV")
        if label == "blocks in time":
            regions = (("−120° to 0", (mid > -2 * np.pi / 3) & (mid < 0)),
                       ("0 to 120°", (mid > 0) & (mid < 2 * np.pi / 3)),
                       ("trans", np.abs(mid) > 2 * np.pi / 3))
            print("    mean squared miss over squared error by region: "
                  + ", ".join(f"{n} {np.mean(dev[m] ** 2):.2f}"
                              for n, m in regions))
    folded_exact = exact_psi[36:]  # ψ ≥ 0, bins symmetric about 0
    for label, order in (("time", np.abs(kept).ravel()),
                         ("copies", np.abs(kept).T.ravel())):
        (_,), ff, dff = landscape.free_energy(order, 36, temperature,
                                              value_range=[(0, np.pi)])
        sh = np.average(ff - folded_exact, weights=1 / dff**2)
        print(f"    folded to |ψ|, blocks of {label}: "
              f"{np.mean(((ff - folded_exact - sh) / dff) ** 2):.2f}")
    chains[temperature] = (mid, f - shift, df)
    gauche = [((kept > 0) & (kept < np.radians(120))).mean(0),
              ((kept < 0) & (kept > -np.radians(120))).mean(0)]
    diff = gauche[0] - gauche[1]
    trans = (np.abs(kept) > np.radians(120)).mean()
    trans_exact = weight[np.abs(fine) > np.radians(120)].sum() / np.sum(
        weight)
    changes = crossings / psi.shape[1] / span
    print(f"    {changes:.1f} changes of well per copy "
          f"per ns; trans fraction {trans:.3f} against {trans_exact:.3f} "
          f"exact; gauche+ − gauche− = {diff.mean():.4f} ± "
          f"{diff.std(ddof=1) / np.sqrt(len(diff)):.4f} over the copies")
    top = np.argmin(np.abs(mid))
    print(f"    at ψ ≈ 0: F = {(f[top] - shift) * 1000:.0f} ± "
          f"{df[top] * 1000:.0f} meV against {exact_psi[top] * 1000:.0f} meV")
    first = psi[:SKIP]
    print(f"    trans fraction over the first {skip_ps:.0f} ps: "
          f"{(np.abs(first) > np.radians(120)).mean():.3f}")

# Potential-energy distributions of Chapter 15's argon, per atom.
nve = ch15.np.load(ch15.RUNS / "long_nve.npz")["potential"] / 256
csvr = ch15.np.load(ch15.RUNS / "long_csvr.npz")["potential"] / 256
edges = np.linspace(min(nve.min(), csvr.min()), max(nve.max(), csvr.max()),
                    80)
h_nve = np.histogram(nve, edges, density=True)[0]
h_csvr = np.histogram(csvr, edges, density=True)[0]
half = len(csvr) // 2
h_a = np.histogram(csvr[:half], edges, density=True)[0]
h_b = np.histogram(csvr[half:], edges, density=True)[0]
print(f"P(U): spreads {csvr.std() * 1000:.3f} (CSVR) and "
      f"{nve.std() * 1000:.3f} meV (fixed energy) per atom; overlap "
      f"{spectra.overlap_score(h_csvr, h_nve):.3f}; the two halves of the "
      f"CSVR run {spectra.overlap_score(h_a, h_b):.3f}")

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.55),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
shown = np.where(np.isfinite(f2), f2 - surface_shift, np.nan)
mesh = ax.pcolormesh(xs, ys, shown.T, cmap="cividis", vmin=0, vmax=0.35,
              shading="auto", rasterized=True)
ax.contour(xs, ys, u2.T, levels=np.arange(0.05, 0.35, 0.1), colors="white",
           linewidths=0.5)
colourbar = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                        fraction=0.08, pad=0.30, ticks=[0, 0.1, 0.2, 0.3])
colourbar.set_label(r"$F(x,y)$ / eV")
ax.set_xlabel(r"$x$ / \AA")
ax.set_ylabel(r"$y$ / \AA")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.plot(d_mid, u_line * 1000, color=REFERENCE, ls="--", lw=0.8)
ax.plot(d_mid, (exact - exact.min()) * 1000, color="black", lw=0.6)
ax.errorbar(d_mid, (f_d - exact.min()) * 1000, 2 * df_d * 1000, fmt="o",
            color=ACCENT, ms=2, lw=0.6)
ax.set_xlim(0, 1.2)
ax.set_ylim(-20, 320)
ax.set_xlabel(r"$r_{\mathrm h}$ / \AA")
ax.set_ylabel("energy / meV")
viz.panel_tag(ax, "b")

ax = axes[2]
mid = chains[300][0]
ax.plot(np.degrees(mid), potentials.opls_torsion(mid, CHAIN_TORSION) * 1000,
        color="black", lw=0.6)
for temperature, colour in ((300, ACCENT), (200, OCHRE)):
    m, f, df = chains[temperature]
    ok = np.isfinite(f)
    ax.errorbar(np.degrees(m[ok]), f[ok] * 1000, 2 * df[ok] * 1000, fmt="o",
                color=colour, ms=1.6, lw=0.5, label=f"{temperature} K")
ax.set_xlim(-180, 180)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_xlabel(r"$\psi$ / degrees")
ax.set_ylabel(r"$F(\psi)$ / meV")
ax.legend(fontsize=7, loc="upper right")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables", "fes.pdf")))
