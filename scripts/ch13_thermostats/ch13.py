"""Shared set-up of the Chapter 13 scripts.

Two test systems. The liquid argon of Chapter 12 (256 atoms, 0.8σ⁻³),
started from the last frame of its cached run (scripts/ch12_ensembles/
runs.py), thermostatted at 135 K. And lithium on the model hexagonal
surface of Chapter 3 (barrier 0.3 eV, hollows 2.46 Å apart), many
independent atoms near 1000 K, each with its own thermostat, for which
the canonical density of positions, e^{−βU}, and the rate of hops over
the bridges are computed exactly or measured.
"""

import sys
from pathlib import Path

import numpy as np

from mdlab import energy, statmech, units, viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch12_ensembles"))
import ch12  # noqa: E402

DATA = Path(viz.THEORY, "data", "ch13_thermostats")
RUNS = DATA / "runs"
T_LIQUID = 135.0
N_ATOMS = 256
N_FREE = statmech.degrees_of_freedom(N_ATOMS)  # 765
DT_LIQUID = 10.0

LI_MASS = 6.94
T_LI = 1000.0
DT_LI = 5.0
SPACING = energy.SURFACE_SPACING
BARRIER = energy.SURFACE_BARRIER
#: Lattice vectors of the hollows, as rows.
LATTICE = SPACING * np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])


#: Frames of Chapter 12's liquid run (every 50 fs) used as independent
#: starts: at 50, 30 and 10 ps.
STARTS = (1000, 600, 200)


def liquid_start(seed=0):
    """Positions, velocities, cell and masses of the equilibrated liquid."""
    run = np.load(ch12.RUNS / "liquid.npz")
    frame = STARTS[seed]
    return (run["positions"][frame], run["velocities"][frame], run["cell"],
            run["masses"])


def argon_model(cell):
    """The switched Lennard-Jones model of Chapter 12."""
    return ch12.model(cell)


def surface(r):
    """Energy and forces of lithium atoms (R, 1, 2) on the model surface."""
    u, f = energy.hexagonal_surface(r[..., 0, :])
    return u, f[..., None, :]


def nearest_hollow(r):
    """Integer lattice coordinates of the hollow nearest each point (…, 2)."""
    base = np.floor(r @ np.linalg.inv(LATTICE))
    best, best_d = None, None
    for di in (0, 1):
        for dj in (0, 1):
            cand = base + np.array([di, dj])
            d = np.linalg.norm(r - cand @ LATTICE, axis=-1)
            if best is None:
                best, best_d = cand, d
            else:
                closer = d < best_d
                best = np.where(closer[..., None], cand, best)
                best_d = np.where(closer, d, best_d)
    return best.astype(int), best_d


def count_hops(path, core=0.7):
    """Hops of each replica along ``path`` (frames, R, 2).

    An atom is assigned to a hollow once it comes within ``core`` Å of it,
    and keeps that assignment until it reaches another: a hop is a change
    of assignment, so crossings of a bridge that turn back do not count.
    """
    site, dist = nearest_hollow(path)
    state = site[0].copy()
    hops = np.zeros(path.shape[1], dtype=int)
    for frame in range(1, len(path)):
        inside = dist[frame] < core
        moved = inside & np.any(site[frame] != state, axis=-1)
        hops += moved
        state = np.where(inside[:, None], site[frame], state)
    return hops


def cell_grid(n=400):
    """Points filling one cell of the surface evenly, and the area of each.

    The cell is the rhombus of the two lattice vectors, centred on a
    hollow; any cell gives the same integral of a periodic function.
    """
    s = (np.arange(n) + 0.5) / n
    s1, s2 = np.meshgrid(s - 0.5, s - 0.5, indexing="ij")
    pts = np.stack([s1, s2], -1) @ LATTICE  # a rhombus of one cell's area
    area = abs(np.linalg.det(LATTICE)) / n**2
    return pts.reshape(-1, 2), area


def canonical_positions(replicas, temperature, rng):
    """Positions (R, 1, 2) drawn from the density e^{−βU} over one cell."""
    kt = units.KB * temperature
    n = 600
    pts, _ = cell_grid(n)
    u, _ = energy.hexagonal_surface(pts)
    w = np.exp(-u / kt)
    pick = rng.choice(len(pts), size=replicas, p=w / w.sum())
    jitter = (rng.random((replicas, 2)) - 0.5) / n @ LATTICE
    return (pts[pick] + jitter)[:, None, :]


def tst_rate(temperature, mass=LI_MASS, n=4000):
    """Transition-state rate of leaving a hollow, per fs.

    The flux of atoms through the six edges of the hollow's cell, each
    a/√3 long and through a bridge, divided by the number in the cell:
    k = √(k_BT/2πm) ∮ e^{−βU} dl / ∫ e^{−βU} dA.
    """
    kt = units.KB * temperature
    pts, area = cell_grid(600)
    u, _ = energy.hexagonal_surface(pts)
    inside = np.sum(np.exp(-u / kt)) * area
    t = (np.arange(n) + 0.5) / n - 0.5
    edge_length = SPACING / np.sqrt(3)
    bridge = np.array([SPACING / 2, 0.0])
    along = np.array([0.0, 1.0])
    line = bridge + np.outer(t * edge_length, along)
    u_line, _ = energy.hexagonal_surface(line)
    edge = np.sum(np.exp(-u_line / kt)) * edge_length / n
    speed = np.sqrt(kt / (2 * np.pi * mass * units.MV2_TO_EV))
    return speed * 6 * edge / inside


def exact_energy_density(temperature, bins):
    """The canonical density of U for one atom on the surface."""
    kt = units.KB * temperature
    pts, _ = cell_grid(600)
    u, _ = energy.hexagonal_surface(pts)
    hist, _ = np.histogram(u, bins=bins, weights=np.exp(-u / kt),
                           density=True)
    return hist


def diffusion(positions, frame_dt, lags=(2000.0, 20000.0), relative=True):
    """The diffusion coefficient from the mean squared displacement, Å²/fs.

    ``positions`` (frames, N, 3) unwrapped, of atoms of equal mass,
    ``frame_dt`` in fs. With ``relative``, positions are measured from the
    centre of mass, which a thermostat that does not keep the momentum
    lets wander. The squared displacement over each lag is averaged over
    every starting frame and every atom, and D is a sixth of the slope of
    a straight line through the lags between ``lags`` (Chapter 16 treats
    this properly).
    """
    if relative:
        positions = positions - positions.mean(axis=1, keepdims=True)
    n = len(positions)
    steps = np.arange(int(lags[0] / frame_dt), int(lags[1] / frame_dt) + 1)
    msd = np.array([np.mean(np.sum((positions[k:] - positions[:n - k]) ** 2,
                                   -1)) for k in steps])
    return np.polyfit(steps * frame_dt, msd, 1)[0] / 6
