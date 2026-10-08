"""A system and its bath sharing energy: the animation of Notebook 12.

    python renders/ch12_ensembles/exchange.py

One oscillator, the system, and 99 more, the bath, share 200 quanta of
energy. At each move a quantum passes from one oscillator, chosen at
random, to another, chosen at random; a move from an empty oscillator is
skipped. The moves are symmetric, so every division of the quanta is
equally likely in the long run: the 100 oscillators together are
microcanonical (Section 12.3). The left panel shows the quanta of every
oscillator, the system's in the accent colour; the right panel the
histogram of the system's quanta, collected after every move, against
the bath's count of states Ω_B(Q − n), normalised (dashed), and the
Boltzmann factor e^{−n ε/k_BT} with e^{−ε/k_BT} = q/(1 + q), q the mean
quanta per oscillator (dotted). Written as an animated GIF to
data/ch12_ensembles/exchange.gif.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.special import gammaln

from mdlab import viz
from mdlab.viz import ACCENT, REFERENCE, REFERENCE_STYLE, THRESHOLD_STYLE

N_OSC = 100  # the system is oscillator 0
QUANTA = 200
MOVES_PER_FRAME = 10000
FRAMES = 60
N_MAX = 15  # largest number of quanta shown for the system


def bath_counts(n):
    """Ω_B(Q − n), the ways 99 oscillators hold Q − n quanta, normalised."""
    bath = N_OSC - 1
    log = gammaln(QUANTA - n + bath) - gammaln(QUANTA - n + 1) - gammaln(bath)
    p = np.exp(log - log.max())
    return p / p.sum()


def main():
    """Run the moves, draw a frame every MOVES_PER_FRAME, write the GIF."""
    rng = np.random.default_rng(12)
    quanta = np.full(N_OSC, QUANTA // N_OSC)
    counts = np.zeros(QUANTA + 1)
    givers = rng.integers(0, N_OSC, FRAMES * MOVES_PER_FRAME)
    takers = rng.integers(0, N_OSC, FRAMES * MOVES_PER_FRAME)

    viz.use_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=80,
                               gridspec_kw=dict(wspace=0.35,
                                                width_ratios=(1.4, 1)))
    colours = [ACCENT] + [REFERENCE] * (N_OSC - 1)
    bars = a.bar(np.arange(N_OSC), quanta, color=colours, width=0.8)
    a.set_ylim(0, 14)
    a.set_xlim(-1, N_OSC)
    a.set_xlabel("oscillator")
    a.annotate("system", xy=(0, 0), xytext=(6, 12.5), color=ACCENT,
               arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    a.set_ylabel("quanta")
    n = np.arange(N_MAX + 1)
    exact = bath_counts(np.arange(QUANTA + 1))[: N_MAX + 1]
    q = QUANTA / N_OSC
    boltzmann = (1 / (1 + q)) * (q / (1 + q)) ** n
    hist = b.bar(n, np.zeros(N_MAX + 1), color=ACCENT, alpha=0.5, width=0.8)
    b.plot(n, exact, **REFERENCE_STYLE, lw=1.0, label=r"$\Omega_B(Q - n)$")
    b.plot(n, boltzmann, **THRESHOLD_STYLE, label="Boltzmann")
    b.set_xlabel("quanta $n$ of the system")
    b.set_ylabel("fraction of moves")
    b.set_ylim(0, 0.4)
    b.legend(loc="upper right")
    title = fig.suptitle("")
    fig.subplots_adjust(bottom=0.2, top=0.86, left=0.08, right=0.97)

    def frame(k):
        start = k * MOVES_PER_FRAME
        for g, t in zip(givers[start:start + MOVES_PER_FRAME],
                        takers[start:start + MOVES_PER_FRAME], strict=True):
            if quanta[g] > 0:
                quanta[g] -= 1
                quanta[t] += 1
            counts[quanta[0]] += 1
        for bar, h in zip(bars, quanta, strict=True):
            bar.set_height(h)
        share = counts[: N_MAX + 1] / counts.sum()
        for bar, h in zip(hist, share, strict=True):
            bar.set_height(h)
        title.set_text(f"{(k + 1) * MOVES_PER_FRAME:,} moves")
        return [*bars, *hist, title]

    anim = FuncAnimation(fig, frame, frames=FRAMES, blit=False)
    target = viz.THEORY / "data" / "ch12_ensembles" / "exchange.gif"
    anim.save(target, writer=PillowWriter(fps=8))
    share = counts[: N_MAX + 1] / counts.sum()
    print(f"after {FRAMES * MOVES_PER_FRAME} moves: system empty "
          f"{share[0]:.3f} of the time; bath count {exact[0]:.3f}, "
          f"Boltzmann {boltzmann[0]:.3f}; largest gap to the bath count "
          f"{np.abs(share - exact).max():.3f}")
    print("wrote", target)


if __name__ == "__main__":
    main()
