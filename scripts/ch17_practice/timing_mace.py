"""What a force call of MACE-MP-0 costs on this machine.

LiC₆ cells of 28 to 1008 atoms; MACE-MP-0 small and medium on the CPU
(five threads) in float32 and float64 and on Apple's GPU (MPS) in
float32, each with D3 added; and D3 alone. For each, the least of three
tries of the mean time of five calls after two to warm up, each call
after a small move of every atom so that nothing is reused.

Writes data/ch17_practice/runs/mace_timing.npz: "atoms", and for each
setting "<model>_<device>_<dtype>" the seconds per call; "d3" likewise.
Prints the times, and the speed of the GPU against the CPU.
"""

import time

import numpy as np
from ch17 import MODEL, RUNS, SMALL, lic6, mace, threads

SIZES = [(2, 2, 1), (3, 3, 1), (3, 3, 2), (6, 6, 1), (6, 6, 2), (6, 6, 4)]
SETTINGS = [(model, device, dtype) for model in ("small", "medium")
            for device, dtype in (("cpu", "float64"), ("cpu", "float32"),
                                  ("mps", "float32"))]


def seconds_per_call(atoms, calc, calls=5, tries=3):
    atoms = atoms.copy()
    atoms.calc = calc
    rng = np.random.default_rng(0)
    for _ in range(2):
        atoms.positions += rng.normal(0, 1e-3, atoms.positions.shape)
        atoms.get_forces()
    best = np.inf
    for _ in range(tries):
        start = time.perf_counter()
        for _ in range(calls):
            atoms.positions += rng.normal(0, 1e-3, atoms.positions.shape)
            atoms.get_forces()
        best = min(best, (time.perf_counter() - start) / calls)
    return best


if __name__ == "__main__":
    threads(5)
    cells = [lic6().repeat(n) for n in SIZES]
    out = {"atoms": np.array([len(c) for c in cells])}
    from dftd3.ase import DFTD3
    out["d3"] = np.array([seconds_per_call(c, DFTD3(method="PBE",
                                                     damping="d3bj"))
                          for c in cells])
    for name, device, dtype in SETTINGS:
        calc = mace(SMALL if name == "small" else MODEL, dtype, True, device)
        key = f"{name}_{device}_{dtype}"
        out[key] = np.array([seconds_per_call(c, calc) for c in cells])
        print(key, " ".join(f"{len(c)}: {t * 1000:.0f} ms"
                            for c, t in zip(cells, out[key], strict=True)),
              flush=True)
    print("D3 alone", " ".join(f"{t * 1000:.1f} ms" for t in out["d3"]))
    for name in ("small", "medium"):
        ratio = out[f"{name}_cpu_float32"] / out[f"{name}_mps_float32"]
        print(f"{name}: CPU float32 time / MPS time = "
              + ", ".join(f"{r:.2f}" for r in ratio))
    np.savez(RUNS / "mace_timing.npz", **out)
