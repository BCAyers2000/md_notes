"""Cost of each thermostat on the liquid, as a ratio to no thermostat.

Run from scripts/ch13_thermostats/: python timing.py

1000 steps of the 256 atoms of liquid argon, the best of three timings
for each method, divided by the best for velocity Verlet alone.
"""

import time

import numpy as np
from ch13 import DT_LIQUID, N_FREE, T_LIQUID, argon_model, liquid_start

from mdlab import thermostats
from mdlab.thermostats import (
    CSVR,
    Andersen,
    Berendsen,
    Langevin,
    NoseHooverChain,
    Thermostat,
)

r, v, h, m = liquid_start()
model = argon_model(h)
t, nf = T_LIQUID, N_FREE
methods = {
    "none": Thermostat,
    "Berendsen": lambda: Berendsen(t, 100.0, nf),
    "Andersen": lambda: Andersen(t, 0.001),
    "Langevin": lambda: Langevin(t, 0.001),
    "Nosé-Hoover chain": lambda: NoseHooverChain(t, 100.0, nf),
    "CSVR": lambda: CSVR(t, 100.0, nf),
}
best = {}
for name, make in methods.items():
    times = []
    for _ in range(3):
        start = time.perf_counter()
        thermostats.run(model, m, r, v, DT_LIQUID, 1000, make(),
                        np.random.default_rng(0), every=100, keep=())
        times.append(time.perf_counter() - start)
    best[name] = min(times)
for name, value in best.items():
    print(f"{name}: {value / best['none']:.2f} times the cost of no "
          f"thermostat")
