"""Figure fig:ob-fourier: frequencies in a signal.

(a) A chord, the notes A (440 Hz) and E (659.25 Hz) of equal strength,
sampled 8000 times a second for 50 ms, and the size of its discrete
Fourier transform (inset scale, Hz). (b) The E alone sampled 1000 times
a second: the samples (points) lie equally on the wave and on one of
1000 − 659.25 = 340.75 Hz (dashed), its alias. (c) The time taken by
the direct sum (N² operations), by the radix-2 algorithm of mdlab and by
NumPy's FFT, against N, relative to each at N = 64; each the least of
five tries, from the last of three passes.

Prints the transform's peaks, the alias, the agreement of mdlab's two
transforms with NumPy's, and the range over three passes of the ratios of
time between N = 64 and 4096.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from mdlab import viz
from mdlab.analysis import spectra
from mdlab.viz import ACCENT, OCHRE, OXBLOOD, REFERENCE_STYLE

RATE, LENGTH = 8000.0, 0.05  # samples per second; seconds
t = np.arange(int(RATE * LENGTH)) / RATE
chord = np.sin(2 * np.pi * 440.0 * t) + np.sin(2 * np.pi * 659.25 * t)
x = spectra.fft(np.concatenate([chord, np.zeros(512 - len(chord))]))
freq = np.fft.fftfreq(512, 1 / RATE)
half = freq >= 0
size = np.abs(x[half]) / len(chord) * 2
peaks = freq[half][[k for k in range(1, half.sum() - 1)
                    if size[k] > size[k - 1] and size[k] > size[k + 1]
                    and size[k] > 0.3]]
print(f"{len(chord)} samples padded to 512; peaks at "
      + ", ".join(f"{p:.2f}" for p in peaks) + f" Hz; spacing "
      f"{RATE / 512:.2f} Hz; resolution 1/T = {1 / LENGTH:.0f} Hz")
sample = chord[:256]
reference = np.fft.fft(sample)
print("direct sum and radix-2 against NumPy: "
      f"{np.abs(spectra.dft(sample) - reference).max():.1e}"
      f", {np.abs(spectra.fft(sample) - reference).max():.1e}")

slow = 1000.0
alias = slow - 659.25
ts = np.arange(0, 0.01, 1 / slow)
fine = np.linspace(0, 0.01, 2000)
note = np.sin(2 * np.pi * 659.25 * ts)
print(f"E sampled at {slow:.0f} Hz: Nyquist {slow / 2:.0f} Hz; alias "
      f"{alias:.2f} Hz; largest difference at the samples "
      f"{np.abs(note + np.sin(2 * np.pi * alias * ts)).max():.1e}")

sizes = 2 ** np.arange(6, 13)
rng = np.random.default_rng(0)
METHODS = (("direct", spectra.dft), ("radix 2", spectra.fft),
           ("NumPy", np.fft.fft))


def least_time(function, signal):
    """The least time of one call, over five tries of 20 ms each."""
    best = np.inf
    for _ in range(5):
        calls, start = 0, time.perf_counter()
        while time.perf_counter() - start < 0.02:
            function(signal)
            calls += 1
        best = min(best, (time.perf_counter() - start) / calls)
    return best


passes = []  # three separate passes, for the spread between them
for _ in range(3):
    timings = {name: [] for name, _ in METHODS}
    for n in sizes:
        signal = rng.normal(size=n)
        for name, function in METHODS:
            timings[name].append(least_time(function, signal))
    passes.append(timings)
for name, _ in METHODS:
    growth = [t[name][-1] / t[name][0] for t in passes]
    print(f"{name}: time at 4096 / time at 64 = {min(growth):.0f} to "
          f"{max(growth):.0f} over three passes")
print(f"N² grows by {(4096 / 64) ** 2:.0f}, N log₂N by "
      f"{4096 * 12 / (64 * 6):.0f}")
timings = passes[-1]

viz.use_style(notebook=False)
fig, axes = plt.subplots(1, 3, figsize=(viz.FULL, 2.0),
                         gridspec_kw=dict(wspace=0.45))
ax = axes[0]
ax.plot(freq[half], size, color=ACCENT, lw=0.9)
ax.set_xlim(0, 1200)
ax.set_xlabel("frequency / Hz")
ax.set_ylabel(r"$2|X_k|/400$")
viz.panel_tag(ax, "a")

ax = axes[1]
ax.plot(fine * 1000, np.sin(2 * np.pi * 659.25 * fine), color=ACCENT, lw=0.8)
ax.plot(fine * 1000, -np.sin(2 * np.pi * alias * fine), **REFERENCE_STYLE,
        lw=0.8)
ax.plot(ts * 1000, np.sin(2 * np.pi * 659.25 * ts), "o", color=OXBLOOD,
        ms=3)
ax.set_xlabel("time / ms")
ax.set_ylabel("signal")
viz.panel_tag(ax, "b")

ax = axes[2]
for (name, values), colour in zip(timings.items(), (OXBLOOD, ACCENT, OCHRE),
                                  strict=True):
    ax.loglog(sizes, np.array(values) / values[0], "o-", color=colour,
              ms=2.5, lw=0.8, label=name)
ax.set_xlabel(r"$N$")
ax.set_ylabel("time / time at 64")
ax.legend(fontsize=6, loc="upper left")
viz.panel_tag(ax, "c")

print("wrote", viz.save(fig, viz.figure_path("ch16_observables",
                                             "fourier.pdf")))
