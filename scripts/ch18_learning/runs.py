"""The dense runs of Section 18.8: liquid argon saved at every step.

Chapter 12's liquid of 256 atoms at 135 K (ch16.liquid_start), two runs
under CSVR (τ = 1 ps) from the same start with different seeds, 30 ps in
steps of 10 fs, every step's positions and potential energy kept, as a
first-principles run saved at every step would be. Writes
data/ch18_learning/argon_dense_<a, b>.npz with "times" (fs), "positions"
(frames, 256, 3) in Å, "potential" in eV and "cell" (lattice vectors as
columns). A few seconds each.
"""

import sys

import numpy as np
from ch18 import DATA, HERE

from mdlab import statmech, thermostats

sys.path.insert(0, str(HERE.parent / "ch16_observables"))
import ch16  # noqa: E402

if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    r, v, h, m = ch16.liquid_start(0)
    model = ch16.argon_model(h)
    nf = statmech.degrees_of_freedom(len(m))
    for name, seed in (("a", 1), ("b", 2)):
        out = thermostats.run(model, m, r, v, 10.0, 3000,
                              thermostats.CSVR(135.0, 1000.0, nf),
                              rng=np.random.default_rng(seed), every=1,
                              keep=("positions",))
        np.savez_compressed(DATA / f"argon_dense_{name}.npz",
                            times=out["times"], positions=out["positions"],
                            potential=out["potential"], cell=h)
        print(f"run {name}: {len(out['times'])} frames")
