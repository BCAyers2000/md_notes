"""Shared set-up of the Chapter 14 scripts.

The switched Lennard-Jones argon of Chapters 10 to 13 (ε = 0.01034 eV,
σ = 3.4 Å, switched off between 2σ and 2.5σ): its liquid, 256 atoms at
0.8σ⁻³ near 135 K, started from the equilibrated frames Chapter 13 used;
its face-centred cubic crystal; the same atoms between soft walls; an
ideal gas of argon atoms. And a layered model solid: triangular sheets of
atoms bound strongly within a sheet and weakly between sheets, each of
three sheets in turn its own pair type, with guest atoms of lithium's
mass between the sheets (Section 14.8).
"""

import math
import sys
from pathlib import Path

import numpy as np

from mdlab import cell, md, potentials, units, viz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch12_ensembles"))
sys.path.insert(0, str(HERE.parent / "ch13_thermostats"))
import ch12  # noqa: E402
import ch13  # noqa: E402

DATA = Path(viz.THEORY, "data", "ch14_pressure")
RUNS = DATA / "runs"
EPS, SIG, MASS = ch12.EPS, ch12.SIG, ch12.MASS
R_CUT, R_SWITCH = ch12.R_CUT, ch12.R_SWITCH
T_LIQUID = ch13.T_LIQUID
N_ATOMS = 256
DT = 10.0
GPA = units.EV_PER_A3_TO_GPA  # 1 eV/Å³ in GPa
BAR = 1e4  # bar per GPa
#: Target pressure of the liquid runs, 0.1 GPa, in eV/Å³.
P_LIQUID = 0.1 / GPA
#: A rough guess at the liquid's compressibility, 2.5 GPa⁻¹, in Å³/eV.
KAPPA_LIQUID = 2.5 * GPA
fcc = ch12.fcc
liquid_start = ch13.liquid_start


def argon_pair():
    """φ and dφ/dr of the switched Lennard-Jones model."""
    return potentials.with_cutoff(
        lambda r: potentials.lennard_jones(r, EPS, SIG), R_CUT, "switch",
        R_SWITCH)


def argon_model(h, skin=1.0):
    """The model in the cell h, with its virial tensor."""
    return md.PairModel(argon_pair(), h, R_CUT, skin)


def walled_model(lengths, stiffness, interacting=True):
    """Atoms between soft walls, without periodic copies.

    Returns a function of the positions giving U and the forces, as
    ``thermostats.run`` expects; the pairs, if any, are all counted.
    """
    from mdlab.virial import harmonic_walls

    pair = argon_pair()

    def model(r):
        u, f, _ = harmonic_walls(r, lengths, stiffness)
        if interacting:
            up, fp, _ = potentials.pair_energy_forces(r, pair)
            u, f = u + up, f + fp
        return u, f

    return model


def volume_mean_and_spread(volumes):
    """Mean and standard deviation of a series of volumes."""
    v = np.asarray(volumes)
    return float(v.mean()), float(v.std())


def isothermal_compressibility(volumes, temperature):
    """κ_T from the volume's fluctuations, var V/(k_BT⟨V⟩), in Å³/eV."""
    v = np.asarray(volumes)
    return float(v.var() / (units.KB * temperature * v.mean()))


def ideal_side(n, temperature, pressure):
    """The side of the cube holding the mean NPT volume of an ideal gas."""
    return ((n + 1) * units.KB * temperature / pressure) ** (1 / 3)


def fixed_volume_pressures():
    """Mean pressure at the three fixed volumes of runs.py, three starts.

    Returns the volumes, the mean pressures and their standard errors
    over the starts, in Å³ and eV/Å³.
    """
    volumes, means, errors = [], [], []
    for scale in ("0.963", "0.978", "0.992"):
        values = []
        for seed in range(3):
            r = np.load(RUNS / f"nvt_v{scale}_s{seed}.npz")
            keep = r["times"] >= 10000
            vol = float(r["volume"])
            values.append((2 * r["kinetic"][keep].mean()
                           + r["virial"][keep].mean()) / (3 * vol))
        volumes.append(vol)
        means.append(np.mean(values))
        errors.append(np.std(values, ddof=1) / math.sqrt(len(values)))
    return np.array(volumes), np.array(means), np.array(errors)


def compressibility_at(volume):
    """κ_T = −1/(V dP/dV) at ``volume``, in Å³/eV.

    The slope is that of a parabola through the fixed-volume pressures.
    """
    volumes, means, _ = fixed_volume_pressures()
    coeffs = np.polyfit(volumes, means, 2)
    slope = np.polyval(np.polyder(coeffs), volume)
    return float(-1 / (volume * slope))


def crossings_period(times, series):
    """Mean period from the crossings of a series through its mean."""
    x = np.asarray(series) - np.mean(series)
    cross = np.nonzero(np.signbit(x[1:]) != np.signbit(x[:-1]))[0]
    if len(cross) < 3:
        return math.nan
    return float(2 * (times[cross[-1]] - times[cross[0]])
                 / (len(cross) - 1))


# ------------------------------------------------------------ layered ---
#: Within a sheet: deep and short; between sheets: argon's own.
EPS_IN, SIG_IN = 8 * EPS, 0.95 * SIG
EPS_OUT, SIG_OUT = EPS, SIG
D_IN = 2 ** (1 / 6) * SIG_IN  # nearest neighbours within a sheet
#: Guests: lithium's mass; their pull on the sheets, ε, and the reach σ
#: they grow to; and between guests.
GUEST_MASS = 6.94
EPS_GUEST, SIG_GUEST = 2 * EPS, 2.5
SIG_GROW = 1.6  # the reach the guests start from
EPS_GG, SIG_GG = 0.2 * EPS, 1.2 * SIG
NX, NY, SHEETS = 6, 4, 9
T_LAYERED = 40.0
DT_LAYERED = 5.0
#: In-plane offsets of the three stacking positions A, B and C.
STACKING = np.array([[0.0, 0.0], [0.5 * D_IN, D_IN / (2 * math.sqrt(3))],
                     [D_IN, D_IN / math.sqrt(3)]])


def _lj(eps, sig):
    return potentials.with_cutoff(
        lambda r: potentials.lennard_jones(r, eps, sig), R_CUT, "switch",
        R_SWITCH)


def layered_table(guest_sigma=None):
    """Pair functions for sheet types 0, 1, 2 and, if given, guests (3)."""
    table = {}
    for a in range(3):
        for b in range(a, 3):
            table[(a, b)] = (_lj(EPS_IN, SIG_IN) if a == b
                             else _lj(EPS_OUT, SIG_OUT))
    if guest_sigma is not None:
        for a in range(3):
            table[(a, 3)] = _lj(EPS_GUEST, guest_sigma)
        table[(3, 3)] = _lj(EPS_GG, SIG_GG)
    return table


def layered_sheets(spacing=0.95 * SIG):
    """Stacked triangular sheets: positions, types and cell.

    NX × NY rectangles of two atoms per sheet, SHEETS sheets stacked A, B,
    C, A, ...; each atom's type is its sheet's place in that cycle.
    """
    rect = np.array([[0.0, 0.0], [0.5, 0.5]])
    grid = np.array([[i + x, j + y] for i in range(NX) for j in range(NY)
                     for x, y in rect]) * np.array([D_IN, D_IN * math.sqrt(3)])
    r, types = [], []
    for k in range(SHEETS):
        xy = grid + STACKING[k % 3]
        r.append(np.column_stack([xy, np.full(len(xy), k * spacing)]))
        types += [k % 3] * len(xy)
    h = np.diag([NX * D_IN, NY * D_IN * math.sqrt(3), SHEETS * spacing])
    return cell.wrap(np.vstack(r), h), np.array(types), h


def guest_sites(h):
    """Guest positions midway between sheets, one for four sheet atoms.

    Each sits over the stacking position that neither neighbouring sheet
    uses, a hollow of both.
    """
    scale = np.diag(h)[:2] / np.array([NX * D_IN, NY * D_IN * math.sqrt(3)])
    spacing = h[2, 2] / SHEETS
    sites = []
    for k in range(SHEETS):
        third = STACKING[(k + 2) % 3]
        for i in range(0, NX, 2):
            for j in range(NY):
                xy = (np.array([i * D_IN, j * D_IN * math.sqrt(3)]) + third)
                xy = xy * scale
                sites.append([xy[0], xy[1], (k + 0.5) * spacing])
    return np.array(sites)
