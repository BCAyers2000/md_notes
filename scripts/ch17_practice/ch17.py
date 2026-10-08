"""Shared set-up of the Chapter 17 scripts.

The liquid argon of Chapters 12 to 16 (256 atoms, 0.8σ⁻³, switched
Lennard-Jones), run by ``mdlab``, by ASE and by LAMMPS with the same
model; and lithium in graphite with the foundation model MACE-MP-0
(medium, the 2023-12-03 weights) plus Grimme's D3 dispersion with
Becke-Johnson damping, which PBE, the functional of the model's training
data, lacks:

- graphite, stacked A, B, with Trucano and Chen's lattice constants as
  the start, a = 2.464 Å and c = 6.711 Å;
- LiC₆, stacked A, A, lithium over every third hexagon (√3×√3 R30°),
  the sheets 3.70 Å apart at the start (Hazrati et al.'s experimental
  spacing);
- dilute lithium, one atom in a 4×4×1 cell of graphite (64 carbon atoms).
"""

import os
import sys
import warnings
from functools import cache
from pathlib import Path

import numpy as np

from mdlab import units, viz

# MPICH, under LAMMPS, fails at exit on this Mac with its default network
# provider; TCP works.
os.environ.setdefault("FI_PROVIDER", "tcp")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch16_observables"))
import ch16  # noqa: E402

ch12, ch13, ch15 = ch16.ch12, ch16.ch13, ch16.ch15
DATA = Path(viz.THEORY, "data", "ch17_practice")
RUNS = DATA / "runs"
GPA = units.EV_PER_A3_TO_GPA
#: ASE's unit of time in fs: ASE's time unit is Å√(amu/eV).
ASE_FS = 1 / np.sqrt(units.FORCE_TO_ACCEL)

# -------------------------------------------------------------- argon ---
T_LIQUID = 135.0
argon_model = ch16.argon_model
liquid_start = ch16.liquid_start

# ----------------------------------------------------------- graphite ---
A_GRAPHITE, C_GRAPHITE = 2.464, 6.711  # Å, Trucano and Chen (1975)
C_LIC6 = 3.70  # Å, the spacing of LiC₆'s sheets (Hazrati et al. 2014)
MODEL = "2023-12-03-mace-128-L1_epoch-199.model"  # MACE-MP-0 medium
SMALL = "2023-12-10-mace-128-L0_energy_epoch-249.model"  # MACE-MP-0 small


def hexagonal(a: float, c: float) -> np.ndarray:
    """Lattice vectors of a hexagonal cell as rows (ASE's convention)."""
    return np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])


def graphite(a: float = A_GRAPHITE, c: float = C_GRAPHITE):
    """AB graphite, four atoms: sheet A at z = c/4, sheet B at 3c/4."""
    from ase import Atoms
    frac = [[0, 0, 0.25], [1 / 3, 2 / 3, 0.25],
            [0, 0, 0.75], [2 / 3, 1 / 3, 0.75]]
    return Atoms("C4", scaled_positions=frac, cell=hexagonal(a, c),
                 pbc=True)


def lic6(a: float = A_GRAPHITE, c: float = C_LIC6):
    """LiC₆ in the √3×√3 R30° cell: one sheet of six carbons.

    Lithium sits over the hexagon at the origin, half a cell above the
    sheet.
    """
    from ase import Atoms
    from ase.build import make_supercell
    sheet = Atoms("C2", scaled_positions=[[1 / 3, 2 / 3, 0],
                                          [2 / 3, 1 / 3, 0]],
                  cell=hexagonal(a, c), pbc=True)
    cell = make_supercell(sheet, [[2, 1, 0], [-1, 1, 0], [0, 0, 1]])
    cell.append("Li")
    cell.positions[-1] = [0, 0, c / 2]
    return cell


def dilute(a: float = A_GRAPHITE, c: float = C_GRAPHITE, n: int = 4):
    """One lithium atom in an n×n×1 cell of AB graphite.

    It sits in the gallery between sheet B and the periodic copy of sheet A
    above it, over a hexagon of sheet B.
    """
    atoms = graphite(a, c).repeat((n, n, 1))
    atoms.append("Li")
    # (1/3, 2/3) of the small cell is an atom of A but the centre of a
    # hexagon of B, whose atoms sit at (0, 0) and (2/3, 1/3)
    atoms.positions[-1] = (np.array([1 / 3, 2 / 3, 0]) @ hexagonal(a, c)
                           + [0, 0, c])
    atoms.wrap()
    return atoms


# ----------------------------------------------------- lithium's sites ---
def sheets(run) -> tuple[np.ndarray, np.ndarray]:
    """The carbon atoms of the sheets below and above lithium's gallery.

    ``run`` a dilute run of runs_mace.py: 64 carbon atoms, then lithium,
    in the gallery at z = c, between sheet B at 3c/4 and the copy of A.
    """
    cell = run["cell"][0]
    c = np.linalg.norm(cell[2])
    z0 = run["frames"][0, :64, 2] % c
    return np.abs(z0 - 0.75 * c) < 0.5, np.abs(z0 - 0.25 * c) < 0.5


def slide(run, sheet: np.ndarray) -> np.ndarray:
    """How far a sheet has slid in the plane, at the times of lithium.

    The mean displacement of its atoms in the frames kept every 100 fs,
    interpolated to the times at which lithium was kept, every 10 fs.
    """
    from mdlab.cell import minimum_image
    cell = run["cell"][0]
    frames = run["frames"]
    d = frames[:, :64] - frames[0, :64]
    d = minimum_image(d.reshape(-1, 3), cell.T).reshape(d.shape)
    mean = d[:, sheet, :2].mean(axis=1)
    frame_times = run["times"][::10][: len(frames)]
    return np.stack([np.interp(run["times"], frame_times, mean[:, k])
                     for k in range(2)], axis=1)


def hollows(xy: np.ndarray, cell: np.ndarray, n: int = 4) -> np.ndarray:
    """The hollows of one sheet of an n×n cell, from its atoms' positions.

    ``xy`` (atoms, 2) in Å; ``cell`` with the lattice vectors as rows. An
    atom less its bond to its nearest neighbour lies at the centre of a
    hexagon; these points, reduced to the sheet's own lattice, are
    averaged, and the sheet's n² hollows follow by its lattice vectors.
    Returns (n², 3), z = 0.
    """
    from mdlab.cell import minimum_image
    p = np.c_[xy, np.zeros(len(xy))]
    d = (p[None] - p[:, None]).reshape(-1, 3)
    d = minimum_image(d, cell.T).reshape(len(p), len(p), 3)
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    centres = p - d[np.arange(len(p)), r.argmin(axis=1)]
    sheet = np.vstack([cell[:2] / n, cell[2]])
    first = centres[0] + minimum_image(centres - centres[0],
                                       sheet.T).mean(axis=0)
    grid = np.array([[i, j, 0] for i in range(n) for j in range(n)])
    return first * [1, 1, 0] + grid @ sheet


def li_sites(run, families: int = 2, follow: bool = True) -> dict:
    """Lithium's site at each of its times, in the frame of each sheet.

    The sites are the hollows of the sheet below (0 to 15) and, with
    ``families`` 2, of the sheet above (16 to 31); lithium is assigned to
    the nearest, each family read in the frame of its own sheet, or, with
    ``follow`` False, fixed in the cell where the sheets started. Returns
    "sites" (times,), "gap" the distance to that site in Å, and "moves"
    the slides of the two sheets (2, times, 2) in Å.
    """
    from mdlab.analysis import hops
    from mdlab.cell import minimum_image
    cell = run["cell"][0]
    moves = [slide(run, s) for s in sheets(run)]
    if not follow:
        moves = [np.zeros_like(m) for m in moves]
    found = []
    for sheet, move in zip(sheets(run), moves, strict=True):
        li = run["li"].copy()
        li[:, :2] -= move  # lithium in the frame of this sheet
        li[:, 2] = 0.0
        sites = hollows(run["frames"][0, :64][sheet, :2], cell)
        index = hops.nearest_site(li, sites, cell.T)
        gap = np.linalg.norm(minimum_image(li - sites[index], cell.T),
                             axis=1)
        found.append((index, gap))
    (low, low_gap), (high, high_gap) = found
    if families == 1:
        return {"sites": low, "gap": low_gap, "moves": np.array(moves)}
    below = low_gap <= high_gap
    return {"sites": np.where(below, low, 16 + high),
            "gap": np.where(below, low_gap, high_gap),
            "moves": np.array(moves)}


def threads(n: int) -> None:
    """Limit PyTorch to n threads, for runs side by side."""
    import torch
    torch.set_num_threads(n)


def _weights(model: str) -> Path:
    """The cached file of a MACE-MP-0 model, downloaded if need be."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    # mace_mp caches each download under the name with '-' and '.' dropped
    path = Path.home() / ".cache" / "mace" / model.replace("-", "").replace(
        ".", "")
    if not path.exists():
        from mace.calculators import mace_mp
        mace_mp(model="small" if model == SMALL else "medium")
    return path


def _float32(model: str) -> Path:
    """A float32 copy of the weights, made on the CPU.

    Apple's GPU (MPS) holds no float64, the weights' own type.
    """
    import torch
    path = _weights(model)
    out = path.with_name(path.name + "_float32")
    if not out.exists():
        torch.save(torch.load(path, map_location="cpu").float(), out)
    return out


def _gpu_without_float64() -> None:
    """Keep the per-atom energies in float32 on the GPU.

    mace-torch 0.3.16 casts the energy of each atom to float64 (models.py,
    ScaleShiftMACE.forward), which MPS cannot hold; on MPS that one output
    is kept in float32. Forces do not pass through it.
    """
    import torch
    if getattr(torch.Tensor.double, "_mps_float32", False):
        return
    double = torch.Tensor.double

    def double_or_float(self, *args, **kwargs):
        if self.device.type == "mps":
            return self.float()
        return double(self, *args, **kwargs)

    double_or_float._mps_float32 = True
    torch.Tensor.double = double_or_float


@cache
def mace(model: str = MODEL, dtype: str = "float32", dispersion: bool = True,
         device: str = "cpu"):
    """MACE-MP-0, with D3(BJ) for PBE added when asked.

    ``device`` 'cpu' (float32 or float64) or 'mps', Apple's GPU (float32
    only).
    """
    warnings.filterwarnings("ignore")
    from ase.calculators.mixing import SumCalculator
    from mace.calculators import MACECalculator
    if device == "mps":
        if dtype != "float32":
            raise ValueError("the MPS device holds no float64")
        _gpu_without_float64()
        path = _float32(model)
    else:
        path = _weights(model)
    learned = MACECalculator(model_paths=str(path), device=device,
                             default_dtype=dtype)
    if not dispersion:
        return learned
    from dftd3.ase import DFTD3
    return SumCalculator([learned, DFTD3(method="PBE", damping="d3bj")])
