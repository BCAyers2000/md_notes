"""The chain of Fermi, Pasta, Ulam and Tsingou over a long run.

The run of fig_ergodic.py, 32 atoms with α = 0.25 and the energy 0.0724 in
the first mode, continued a hundred times longer, to t = 2 × 10⁶ (4 × 10⁷
steps of 0.05), recording every 2000 steps. Prints the average and the
largest share of the energy in modes 5 to 32. About 3 minutes.

Run from theory/: python scripts/ch12_ensembles/fput_long.py
"""

import numpy as np

from mdlab import chain

N, ALPHA, DT, AMPLITUDE = 32, 0.25, 0.05, 4.0

if __name__ == "__main__":
    patterns, _ = chain.normal_modes(N)
    r = chain.run(AMPLITUDE * patterns[:, 0], np.zeros(N), ALPHA, DT,
                  40_000_000, every=2000)
    share = r["modes"][:, 4:].sum(1) / r["modes"].sum(1)
    print(f"to t = {r['times'][-1]:.0f}: modes 5 to 32 hold {share.mean():.3f}"
          f" of the energy on average and at most {share.max():.3f}; equal "
          f"sharing 0.875")
