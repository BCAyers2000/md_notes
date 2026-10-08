"""Analysis of trajectories, from Chapter 15 on.

- ``stats``: correlation, error bars, equilibration and drift of a
  recorded series, and the convergence criterion (Chapter 15);
- ``structure``: the radial distribution function, coordination numbers
  and the structure factor (Chapter 16);
- ``transport``: mean squared displacements, correlation functions and
  their integrals, for diffusion, conduction and viscosity (Chapter 16);
- ``spectra``: the discrete and fast Fourier transforms, the vibrational
  density of states and the overlap of two spectra (Chapter 16);
- ``landscape``: free energy from a histogram (Chapter 16);
- ``hops``: hops between sites and rates that may be unresolved
  (Chapter 17).
"""

from . import hops, landscape, spectra, stats, structure, transport

__all__ = ["hops", "landscape", "spectra", "stats", "structure", "transport"]
