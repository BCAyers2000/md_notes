"""Write notebooks/18_learning.ipynb, the companion to Chapter 18.

Run from theory/, then execute the notebook:

    python scripts/notebooks/build_18_learning.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/18_learning.ipynb

The dense argon runs of Section 18.8 are read from data/ch18_learning/
(written by scripts/ch18_learning/runs.py); everything else is computed
in the notebook, in a few seconds.
"""

from nbtools import code, hidden, md, solution, write

SETUP = [
    md(r"""
    # Notebook 18: Learning as optimisation

    Fit the bond forces first, then use the noisy pair energies to separate
    model choice from optimisation. The comparisons with scikit-learn and
    PyTorch check the calculations along the way. Keep the validation data
    separate when choosing settings, and note which change improves the
    held-out prediction. The section numbers follow Chapter 18; run the
    cells in order.
    """),
    code(r"""
    %matplotlib inline
    import math
    import sys
    from pathlib import Path

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation

    from mdlab import learn, units, viz
    from mdlab.analysis import stats
    from mdlab.cell import minimum_image
    from mdlab.exercise import check

    sys.path.insert(0, str(Path("..") / "scripts" / "ch18_learning"))
    import ch18  # the shared set-up of the chapter's scripts

    viz.use_style()
    SLOW = dict(continuous_update=False)  # redraw only on release
    RNG = np.random.default_rng(18)
    KB = units.KB
    """),
]

LOSS = [
    md(r"""
    ## 18.1 Data, model and loss

    The A–B molecule of Chapter 16 under Langevin dynamics at a chosen
    temperature, 20 ps, a reading of stretch and force every 10 fs. A
    polynomial of the chosen degree in the stretch is fitted by least
    squares; its slope at the bottom is the stiffness it reports, against
    2Da² = 8 eV/Å².
    """),
    code(r"""
    BONDS = {t: ch18.bond_run(t, 20.0, 0) for t in (100, 300, 1000)}


    def bond_view(temperature=300, degree=1):
        x, f = BONDS[temperature]
        X = learn.polynomial_basis(x, degree)
        w = learn.least_squares(X, f)
        grid = np.linspace(x.min(), x.max(), 200)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(x[::3], f[::3], ".", ms=2, alpha=0.5, label="bond samples")
        ax.plot(grid, learn.polynomial_basis(grid, degree) @ w, lw=1.5,
                label=f"degree {degree} fit")
        ax.legend()
        ax.set_xlabel("stretch x / Å")
        ax.set_ylabel("force along the bond / eV Å⁻¹")
        plt.show()
        miss = np.sqrt(np.mean((X @ w - f) ** 2))
        print(f"slope at the bottom {-w[1]:.3f} eV/Å² against "
              f"{ch18.STIFFNESS:.0f}; rms miss {miss:.5f} eV/Å")


    widgets.interact(bond_view, temperature=[100, 300, 1000],
                     degree=widgets.IntSlider(1, min=1, max=4, **SLOW));
    """),
    md(r"""
    The straight line's stiffness falls as the run
    gets hotter, 7.66, 7.07 and 5.26 eV/Å² at 100, 300 and 1000 K: the
    line averages the well's stiffness over the stretches the run
    explores. Raising the degree lets the curve bend with the force, and
    by degree 4 the slope at the bottom is 8.00 at 100 and 300 K and 8.08
    at 1000 K.
    """),
    code(r"""
    # Compare with scikit-learn: the same straight line at 300 K.
    from sklearn.linear_model import LinearRegression

    x, f = BONDS[300]
    ours = learn.least_squares(learn.polynomial_basis(x, 1), f)
    theirs = LinearRegression().fit(x[:, None], f)
    print(f"mdlab: intercept {ours[0]:.6f}, slope {ours[1]:.6f}")
    print(f"scikit-learn: intercept {theirs.intercept_:.6f}, slope "
          f"{theirs.coef_[0]:.6f}")
    """),
]

BASIS = [
    md(r"""
    ## 18.2 Basis functions and overfitting

    Argon's pair energy read at 15 distances with noise of 0.2 meV, fitted
    by a polynomial in the scaled distance of the chosen degree, and
    judged on 200 noise-free test distances.
    """),
    code(r"""
    RNG = np.random.default_rng(182)
    R_TRAIN, E_TRAIN = ch18.readings(15, RNG)
    R_VALID, E_VALID = ch18.readings(15, RNG)
    R_TEST = np.linspace(ch18.R_LOW, ch18.R_HIGH, 200)
    E_TEST = ch18.argon_pair(R_TEST)


    def rms(a, b):
        return 1000 * np.sqrt(np.mean((a - b) ** 2))  # meV


    def basis_view(degree=8):
        w = learn.least_squares(ch18.basis(R_TRAIN, degree), E_TRAIN)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(R_TEST, E_TEST * 1000, "--", color="0.5", label="pair energy")
        ax.plot(R_TRAIN, E_TRAIN * 1000, "ko", ms=4, label="training readings")
        ax.plot(R_TEST, ch18.basis(R_TEST, degree) @ w * 1000, label="fit")
        ax.legend()
        ax.set_ylim(-14, 8)
        ax.set_xlabel("r / Å")
        ax.set_ylabel("φ(r) / meV")
        plt.show()
        print(f"rms miss: training "
              f"{rms(ch18.basis(R_TRAIN, degree) @ w, E_TRAIN):.4f} meV, "
              f"test {rms(ch18.basis(R_TEST, degree) @ w, E_TEST):.4f} meV")


    widgets.interact(basis_view,
                     degree=widgets.IntSlider(8, min=0, max=14, **SLOW));
    """),
    md(r"""
    The training miss falls with every degree, to
    zero at 14, where 15 weights pass the curve through 15 readings. The
    test miss is least at degree 8, 0.28 meV, and then grows: 3.9 meV at
    9 and 41 eV at 14, as the curve swings between the readings.
    """),
]

RIDGE = [
    md(r"""
    ## 18.3 Ridge regularisation

    The polynomial of degree 14 with the penalty λ|w|². The right panel
    shows each component of the weights along an eigenvector of XᵀX,
    without the penalty (grey) and with it (colour), against the
    eigenvalue d_k: the penalty shrinks the components whose d_k is small
    against λ.
    """),
    code(r"""
    X14 = ch18.basis(R_TRAIN, 14)
    D, V = np.linalg.eigh(X14.T @ X14)


    def ridge_view(log_lambda=-3.0):
        lam = 10.0**log_lambda
        w = learn.least_squares(X14, E_TRAIN, ridge=lam)
        w0 = learn.least_squares(X14, E_TRAIN)
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.2))
        ax.plot(R_TEST, E_TEST * 1000, "--", color="0.5", label="pair energy")
        ax.plot(R_TRAIN, E_TRAIN * 1000, "ko", ms=4, label="training readings")
        ax.plot(R_TEST, ch18.basis(R_TEST, 14) @ w * 1000, label="ridge fit")
        ax.legend()
        ax.set_ylim(-14, 8)
        ax.set_xlabel("r / Å")
        ax.set_ylabel("φ(r) / meV")
        resolved = D > 0
        bx.loglog(D[resolved], np.abs(V.T @ w0)[resolved], "o", color="0.6", ms=4, label="no penalty")
        bx.loglog(D[resolved], np.abs(V.T @ w)[resolved], "o", ms=4, label="ridge")
        bx.legend()
        bx.axvline(lam, color="0.7", lw=0.8)
        bx.set_xlabel(r"eigenvalue $d_k$ of $X^{\mathsf{T}}X$")
        bx.set_ylabel("|component of w| / eV")
        plt.show()
        print(f"λ = {lam:.1e}: misses {rms(X14 @ w, E_TRAIN):.3f} (training),"
              f" {rms(ch18.basis(R_VALID, 14) @ w, E_VALID):.3f} (validation),"
              f" {rms(ch18.basis(R_TEST, 14) @ w, E_TEST):.3f} meV (test)")


    widgets.interact(ridge_view, log_lambda=widgets.FloatSlider(
        -3.0, min=-12.0, max=1.0, step=0.5, **SLOW));
    """),
    md(r"""
    At λ = 10⁻³, the value of least validation miss,
    the test miss is 0.15 meV, half the best without the penalty. The
    right panel shows why: components along eigenvectors with d_k well
    below λ, which the readings barely constrain and the noise inflates to
    thousands of eV, are pulled down by d_k/(d_k + λ), while those with
    d_k well above λ hardly move.
    """),
    code(r"""
    # Compare with scikit-learn's Ridge, without an intercept.
    from sklearn.linear_model import Ridge

    w = learn.least_squares(X14, E_TRAIN, ridge=1e-3)
    theirs = Ridge(alpha=1e-3, fit_intercept=False).fit(X14, E_TRAIN).coef_
    print(f"largest difference of the weights: {np.abs(w - theirs).max():.2e}"
          f" eV, against weights of size {np.abs(w).max():.2e}")
    """),
]

GP = [
    md(r"""
    ## 18.4 Gaussian processes

    A Gaussian process with height 10 meV and the known noise, conditioned
    on the first n readings of eight; the band is twice the posterior
    standard deviation.
    """),
    code(r"""
    R8, E8 = ch18.readings(8, np.random.default_rng(184))
    GRID = np.linspace(3.0, 7.1, 300)


    def gp_view(length=0.7, n=8):
        gp = learn.GaussianProcess(length, 0.010, ch18.NOISE).fit(R8[:n],
                                                                  E8[:n])
        mean, var = gp.predict(GRID)
        sd = np.sqrt(var)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(GRID, ch18.argon_pair(GRID) * 1000, "--", color="0.5")
        ax.fill_between(GRID, (mean - 2 * sd) * 1000, (mean + 2 * sd) * 1000,
                        alpha=0.25, lw=0)
        ax.plot(GRID, mean * 1000)
        ax.plot(R8[:n], E8[:n] * 1000, "ko", ms=4)
        ax.set_ylim(-16, 10)
        ax.set_xlabel("r / Å")
        ax.set_ylabel("φ(r) / meV")
        plt.show()
        m, v = gp.predict(R_TEST)
        inside = np.mean(np.abs(m - E_TEST) <= 2 * np.sqrt(v))
        print(f"log marginal likelihood {gp.log_marginal_likelihood():.2f}; "
              f"test miss {rms(m, E_TEST):.3f} meV, {inside:.2f} of the test "
              f"inside the band")


    widgets.interact(gp_view,
                     length=widgets.FloatLogSlider(0.7, base=10, min=-1.3,
                                                   max=0.7, step=0.05, **SLOW),
                     n=widgets.IntSlider(8, min=1, max=8, **SLOW));
    """),
    md(r"""
    Near a reading the band narrows to the noise;
    between readings it widens, most across the gap from 3.36 to 4.13 Å.
    The log marginal likelihood is greatest near ℓ = 0.70 Å. Even there
    only 0.75 of the test lies inside the band, against the 0.95 a correct
    band holds: one length scale cannot fit both the steep wall and the
    slow tail. Short lengths sag to zero between readings with a wide
    band; long ones are stiff with a narrow band that misses.
    """),
    code(r"""
    # Compare with scikit-learn's GaussianProcessRegressor, kernel fixed.
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    gp = learn.GaussianProcess(0.7, 0.010, ch18.NOISE).fit(R8, E8)
    kernel = (ConstantKernel(0.010**2, constant_value_bounds="fixed")
              * RBF(0.7, length_scale_bounds="fixed"))
    sk = GaussianProcessRegressor(kernel, alpha=ch18.NOISE**2,
                                  optimizer=None).fit(R8[:, None], E8)
    m, v = gp.predict(GRID)
    m2, s2 = sk.predict(GRID[:, None], return_std=True)
    print(f"largest difference: mean {np.abs(m - m2).max():.1e} eV, "
          f"standard deviation {np.abs(np.sqrt(v) - s2).max():.1e} eV; "
          f"log marginal likelihood {gp.log_marginal_likelihood():.4f} "
          f"against {sk.log_marginal_likelihood_value_:.4f}")
    """),
]

DESCENT = [
    md(r"""
    ## 18.5 Gradient descent

    A quadratic loss of two weights whose Hessian has the eigenvalues 1
    and κ, along axes turned by 30°, walked by gradient descent at a rate
    given as a fraction of the stability limit 2/λ_max.
    """),
    code(r"""
    def valley(kappa, turn=np.radians(30)):
        c, s = np.cos(turn), np.sin(turn)
        R = np.array([[c, -s], [s, c]])
        A = R @ np.diag([1.0, kappa]) @ R.T
        return A, lambda w: (0.5 * w @ A @ w, A @ w)


    def descent_view(kappa=10.0, fraction=0.9):
        A, fn = valley(kappa)
        rate = fraction * 2 / kappa
        path, loss = learn.minimise(fn, [-2.0, 1.5],
                                    learn.GradientDescent(rate), 60)
        g = np.linspace(-2.6, 2.6, 150)
        G1, G2 = np.meshgrid(g, g)
        W = np.stack([G1, G2], axis=-1)
        Z = 0.5 * np.einsum("...i,ij,...j->...", W, A, W)
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6))
        ax.contour(G1, G2, Z, levels=np.geomspace(0.02, 50, 10),
                   colors="0.75", linewidths=0.5)
        ax.plot(path[:30, 0], path[:30, 1], "o-", ms=2)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xlabel("weight w₁")
        ax.set_ylabel("weight w₂")
        bx.semilogy(np.maximum(loss, 1e-16))
        bx.set_xlabel("step")
        bx.set_ylabel("loss")
        plt.show()
        f = np.abs(1 - rate * np.array([1.0, kappa])).max()
        print(f"rate {rate:.4f}; slowest factor {f:.3f} a step; "
              f"loss after 60 steps {loss[-1]:.2e}")


    widgets.interact(descent_view,
                     kappa=widgets.FloatLogSlider(10, base=10, min=0, max=3,
                                                  step=0.25, **SLOW),
                     fraction=widgets.FloatSlider(0.9, min=0.05, max=1.1,
                                                  step=0.05, **SLOW));
    """),
    md(r"""
    Above a fraction of 1 the path climbs out of the
    valley in a growing zig-zag, the stability limit of an integrator
    again. Below it, the shallow direction sets the pace, and the larger
    κ the slower it is: at κ = 10 and the best rate the loss falls a
    millionfold in 34 steps, at κ = 1000 it would take about 3500.
    """),
]

SGD = [
    md(r"""
    ## 18.6 Stochastic gradients

    The A–B bond's force at 300 K fitted by a straight line in the scaled
    stretch, trained by mini-batches of the chosen size, at a fixed rate
    or one that falls as 1/(1 + epoch/5).
    """),
    code(r"""
    XS, FS = BONDS[300]
    XS = learn.polynomial_basis(XS / XS.std(), 1)
    W_BEST = learn.least_squares(XS, FS)
    FLOOR = np.mean((XS @ W_BEST - FS) ** 2)
    RATE = 1.0 / np.linalg.eigvalsh(2 * XS.T @ XS / len(FS)).max()


    def sgd_view(batch=10, falling=False, epochs=40):
        rng = np.random.default_rng(0)
        w = np.zeros(2)
        history = [np.mean((XS @ w - FS) ** 2)]
        for epoch in range(epochs):
            eta = RATE / (1 + epoch / 5) if falling else RATE
            for rows in learn.batches(len(FS), batch, rng):
                miss = XS[rows] @ w - FS[rows]
                w = w - eta * 2 * XS[rows].T @ miss / len(rows)
            history.append(np.mean((XS @ w - FS) ** 2))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.semilogy(history)
        ax.axhline(FLOOR, color="0.6", ls="--")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss / (eV Å⁻¹)²")
        plt.show()
        print(f"loss after {epochs} epochs {history[-1] - FLOOR:.2e} above "
              f"the least; {int(np.ceil(len(FS) / batch))} steps an epoch")


    widgets.interact(sgd_view, batch=[1, 10, 100, 2001], falling=False,
                     epochs=widgets.IntSlider(40, min=5, max=80, **SLOW));
    """),
    md(r"""
    With all 2001 readings a batch, the loss reaches
    its least in a few epochs. Small batches take many more steps an
    epoch and, at a fixed rate, keep hopping about the least; with the
    falling rate the hops die away.
    """),
]

MOMENTUM = [
    md(r"""
    ## 18.7 Momentum and Adam

    Gradient descent at its best fixed rate, momentum with Polyak's rate
    and μ, and Adam at the rate 0.05, in a valley whose Hessian has the
    eigenvalues 1 and 100, turned by the chosen angle. The animation draws
    their paths step by step.
    """),
    code(r"""
    def race(turn_degrees=0.0, steps=120):
        A, fn = valley(100.0, np.radians(turn_degrees))
        sk = math.sqrt(100.0)
        optimisers = {
            "gradient descent": learn.GradientDescent(2 / 101),
            "momentum": learn.Momentum(4 / (sk + 1) ** 2,
                                       ((sk - 1) / (sk + 1)) ** 2),
            "Adam": learn.Adam(0.05)}
        return {name: learn.minimise(fn, [-2.5, 0.5], opt, steps)
                for name, opt in optimisers.items()}


    def race_view(turn_degrees=0.0):
        runs = race(turn_degrees, 300)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for name, (_, loss) in runs.items():
            ax.semilogy(np.maximum(loss, 1e-30), label=name)
        ax.set_ylim(1e-12, 1e3)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        plt.show()
        for name, (_, loss) in runs.items():
            print(f"{name}: loss {loss[-1]:.2e} after 300 steps")


    widgets.interact(race_view, turn_degrees=widgets.FloatSlider(
        0.0, min=0.0, max=45.0, step=5.0, **SLOW));
    """),
    md(r"""
    With the valley along the weights, Adam evens
    out the two scales and reaches 10⁻⁸ in about 190 steps, momentum in
    77, and gradient descent at its best rate is still far off. Turned by
    45°, Adam has nothing to even out and stalls near 0.5, worse than plain
    descent, while momentum is unchanged.
    """),
    code(r"""
    def race_animation(turn_degrees=0.0, frames=60):
        runs = race(turn_degrees, frames)
        A, _ = valley(100.0, np.radians(turn_degrees))
        g1, g2 = np.meshgrid(np.linspace(-3, 3, 150),
                             np.linspace(-1.8, 1.8, 150))
        W = np.stack([g1, g2], axis=-1)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.contour(g1, g2, 0.5 * np.einsum("...i,ij,...j->...", W, A, W),
                   levels=np.geomspace(0.05, 150, 10), colors="0.75",
                   linewidths=0.5)
        lines = {name: ax.plot([], [], "o-", ms=2, lw=0.8, label=name)[0]
                 for name in runs}
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel("weight w₁")
        ax.set_ylabel("weight w₂")
        ax.legend(loc="lower right")

        def draw(k):
            for name, (path, _) in runs.items():
                lines[name].set_data(path[:k + 1, 0], path[:k + 1, 1])
            return list(lines.values())

        animation = FuncAnimation(fig, draw, frames=frames, interval=80)
        plt.close(fig)
        return HTML(animation.to_jshtml())


    race_animation()
    """),
    code(r"""
    # Compare with PyTorch: Adam's and momentum's steps on one loss.
    import torch

    A_T = np.array([[3.0, 1.0], [1.0, 0.5]])


    def ours(optimiser, steps=20):
        def fn(w):
            return 0.5 * w @ A_T @ w + np.sin(w).sum(), A_T @ w + np.cos(w)
        return learn.minimise(fn, [1.0, -2.0], optimiser, steps)[0]


    def theirs(make, steps=20):
        A = torch.tensor(A_T)
        p = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
        opt = make([p])
        path = [p.detach().clone().numpy()]
        for _ in range(steps):
            opt.zero_grad()
            (0.5 * p @ A @ p + torch.sin(p).sum()).backward()
            opt.step()
            path.append(p.detach().clone().numpy())
        return np.array(path)


    a = np.abs(ours(learn.Adam(0.1)) - theirs(
        lambda p: torch.optim.Adam(p, lr=0.1))).max()
    m = np.abs(ours(learn.Momentum(0.05, 0.8)) - theirs(
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.8))).max()
    print(f"largest difference over 20 steps: Adam {a:.1e}, momentum {m:.1e}")
    """),
]

SPLITS = [
    md(r"""
    ## 18.8 Training, validation and test

    Liquid argon, a frame every 10 fs; ridge regression learns each
    frame's potential energy from the histogram of its pair distances.
    Choose the length of the training run and the split; the misses are
    averaged over ten splits, and the test is another run.
    """),
    code(r"""
    EDGES = np.linspace(3.0, 8.5, 111)
    DENSE = {name: np.load(ch18.DATA / f"argon_dense_{name}.npz")
             for name in ("a", "b")}


    def histograms(run, frames):
        h = run["cell"]
        i, j = np.triu_indices(run["positions"].shape[1], 1)
        out = []
        for f in run["positions"][frames]:
            d = np.linalg.norm(minimum_image(f[i] - f[j], h), axis=1)
            out.append(np.histogram(d, EDGES)[0])
        return np.array(out, dtype=float)


    HA = histograms(DENSE["a"], np.arange(1000, 3001))
    UA = DENSE["a"]["potential"][1000:3001]
    HB = histograms(DENSE["b"], np.arange(1000, 3001, 5))
    UB = DENSE["b"]["potential"][1000:3001:5]
    print(f"statistical inefficiency of U at 10 fs: "
          f"{stats.statistical_inefficiency(UA):.0f} frames")


    def fit_miss(n, train, valid):
        mu, sd = HA[train].mean(0), HA[train].std(0) + 1e-12
        off = UA[train].mean()

        def design(H):
            return np.c_[np.ones(len(H)), (H - mu) / sd]

        w = learn.least_squares(design(HA[train]), UA[train] - off,
                                ridge=1e-2)
        return (rms(design(HA[valid]) @ w + off, UA[valid]),
                rms(design(HB) @ w + off, UB))


    def split_view(picoseconds=2, split="random"):
        n = int(picoseconds * 100)
        found = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            train, valid = (learn.random_split(n, 0.2, rng)
                            if split == "random"
                            else learn.blocked_split(n, 0.2, 50, rng))
            found.append(fit_miss(n, train, valid))
        v, t = np.mean(found, axis=0)
        print(f"{n} frames: validation miss {v:.1f} meV, an independent "
              f"run's {t:.1f} meV")


    widgets.interact(split_view, picoseconds=[2, 4, 10, 20],
                     split=["random", "blocked"]);
    """),
    md(r"""
    With 2 ps of training, a random split reports a
    validation miss of about 30 meV while the independent run gives over
    100: each validation frame has training frames 10 fs either side of
    it. The blocked split reports about 92 meV, much closer to the independent run. By 10 ps,
    about nine independent frames, all agree near the binning's own 22
    meV.
    """),
]

TRAINING = [
    md(r"""
    ## 18.9 What the training looks like

    The fifteen readings of argon's pair energy trained from zero weights;
    choose the run and read its loss curves.
    """),
    code(r"""
    def train_curves(X, Xv, optimiser, steps):
        def fn(w):
            miss = X @ w - E_TRAIN
            return np.mean(miss**2), 2 * X.T @ miss / len(E_TRAIN)
        path, _ = learn.minimise(fn, np.zeros(X.shape[1]), optimiser, steps)
        return (1000 * np.sqrt(np.mean((path @ X.T - E_TRAIN) ** 2, axis=1)),
                1000 * np.sqrt(np.mean((path @ Xv.T - E_VALID) ** 2, axis=1)))


    X8, V8 = ch18.basis(R_TRAIN, 8), ch18.basis(R_VALID, 8)
    LIMIT = 2 / np.linalg.eigvalsh(2 * X8.T @ X8 / 15).max()
    RUNS = {
        "healthy (degree 8, Adam 1e-4)":
            lambda: train_curves(X8, V8, learn.Adam(1e-4), 40000),
        "rate past the limit (1.1 × 2/λ_max)":
            lambda: train_curves(X8, V8, learn.GradientDescent(1.1 * LIMIT),
                                 300),
        "rate far too small (0.001 × 2/λ_max)":
            lambda: train_curves(X8, V8,
                                 learn.GradientDescent(1e-3 * LIMIT), 300),
        "Adam's rate too large (1e-3)":
            lambda: train_curves(X8, V8, learn.Adam(1e-3), 20000),
        "overfitting (degree 14, Adam 1e-4)":
            lambda: train_curves(ch18.basis(R_TRAIN, 14),
                                 ch18.basis(R_VALID, 14),
                                 learn.Adam(1e-4), 100000),
    }


    def training_view(run=list(RUNS)[0]):
        train, valid = RUNS[run]()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.semilogy(np.maximum(train, 1e-3), label="training")
        ax.semilogy(np.maximum(valid, 1e-3), label="validation")
        ax.axhline(1000 * ch18.NOISE, color="0.6", ls=":")
        finite = np.concatenate([train[np.isfinite(train)], valid[np.isfinite(valid)]])
        top = min(1e3, max(1.0, 1.2 * finite.max()))
        ax.set_ylim(0.1, top)
        if np.any(~np.isfinite(train)) or np.any(~np.isfinite(valid)) or finite.max() > top:
            ax.text(0.02, 0.94, "divergence continues above the plotted range",
                    transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("rms miss / meV")
        ax.legend()
        plt.show()
        print(f"at the end: training {train[-1]:.3g}, validation "
              f"{valid[-1]:.3g} meV; least validation {valid.min():.3g} "
              f"meV at step {int(np.argmin(valid))}")


    widgets.interact(training_view, run=list(RUNS));
    """),
    md(r"""
    Healthy: both curves fall to the noise, 0.2 meV,
    together. Past the limit the training miss climbs off the plot; far
    below it, it barely moves. Adam at 10⁻³ reaches the noise but jumps
    in single steps later on. Overfitting: the validation miss is least
    after about 160 steps and climbs to near 3 meV by step 10⁵ while the
    training miss sinks below the noise.
    """),
]

EXERCISES = [
    md(r"""
    ## Working notes

    For each fit, record the basis, regularisation, learning rate and split.
    Keep training and validation errors together, including settings that
    make one improve while the other worsens. Before using a trajectory as
    training data, decide what counts as a separate run or time block.
    """),
    md(r"""
    ## Exercises

    Each exercise cell checks its answer against the key below; the
    collapsed cell after it holds a worked solution.
    """),
    hidden(r"""
    # The answer key. Each target is computed from the exercise's data.
    x_m = np.arange(1, 11) / 100
    kappa_p = 100.0
    TARGETS = {
        "18.2": 0.05 / math.sqrt(np.sum(x_m**2)),
        "18.4": 3 * 2.0 * 0.0345,
        "18.7": np.array([0.2 / math.sqrt(1e-3), (0.2 / 1.0) ** 2]),
        "18.8": np.array([1.5 * 2 / 1.0, 4 - 1.5**2 / 1.0]),
        "18.10": np.array([2 / 200, 2 / 202, 99 / 101,
                           math.log(1e6) / (-2 * math.log(99 / 101))]),
        "18.12": (0.585 / 0.05) ** 2,
        "18.13": np.array([-1 / math.log(0.9), 0.1]),
        "18.14": np.array([9 / 11, (9 / 11) ** 2, 4 / 121,
                           math.log(1e-4) / math.log(9 / 11),
                           math.log(1e-4) / math.log(99 / 101)]),
        "18.15": np.array([1 - 0.9**10, 1 - 0.999**10,
                           (1 - 0.9**10) / math.sqrt(1 - 0.999**10)]),
        "18.17": np.array([200 / 109, 2000 / 109]),
    }
    print(f"{len(TARGETS)} targets loaded")
    """),
    code(r"""
    # EXERCISE 18.2: the error of the stiffness, in N/m, for stretches of
    # 1 to 10 cm and readings that err by 0.05 N.
    error_k = None

    check(error_k, TARGETS["18.2"], rtol=1e-3, name="error of k")
    """),
    solution(r"""
    # SOLUTION 18.2. The error is σ/√Σx².
    error_k = 0.05 / math.sqrt(np.sum((np.arange(1, 11) / 100) ** 2))

    check(error_k, TARGETS["18.2"], rtol=1e-3, name="error of k");
    """),
    code(r"""
    # EXERCISE 18.4: 3ax, the fraction by which the Morse stiffness changes
    # at x = 0.0345 Å for a = 2 Å⁻¹.
    fraction = None

    check(fraction, TARGETS["18.4"], rtol=1e-6, name="fraction")
    """),
    solution(r"""
    # SOLUTION 18.4. The stiffness is 2Da²(1 − 3ax) to first order.
    fraction = 3 * 2.0 * 0.0345

    check(fraction, TARGETS["18.4"], rtol=1e-6, name="fraction");
    """),
    code(r"""
    # EXERCISE 18.7: σ_w in meV for σ = 0.2 meV and λ = 1e-3, and the λ
    # for σ_w = 1 meV.
    prior = None

    check(prior, TARGETS["18.7"], rtol=1e-6, name="σ_w and λ")
    """),
    solution(r"""
    # SOLUTION 18.7. λ = σ²/σ_w².
    prior = np.array([0.2 / math.sqrt(1e-3), (0.2 / 1.0) ** 2])

    check(prior, TARGETS["18.7"], rtol=1e-6, name="σ_w and λ");
    """),
    code(r"""
    # EXERCISE 18.8: the mean and variance of a given b = 2, for variances
    # 4 and 1 and covariance 1.5.
    conditioned = None

    check(conditioned, TARGETS["18.8"], rtol=1e-9, name="mean and variance")
    """),
    solution(r"""
    # SOLUTION 18.8. Mean (c/σ_b²)b, variance σ_a² − c²/σ_b².
    conditioned = np.array([1.5 / 1.0 * 2, 4 - 1.5**2 / 1.0])

    check(conditioned, TARGETS["18.8"], rtol=1e-9, name="mean and variance");
    """),
    code(r"""
    # EXERCISE 18.10: for eigenvalues 2 and 200, the upper stability boundary,
    # the best fixed rate, its slow factor and the steps for a fall of 10⁶.
    rates = None

    check(rates, TARGETS["18.10"], rtol=1e-3, name="rates and steps")
    """),
    solution(r"""
    # SOLUTION 18.10. 2/λ_max; 2/(λ_min + λ_max); (κ − 1)/(κ + 1); the loss
    # falls as the factor squared.
    factor = 99 / 101
    rates = np.array([2 / 200, 2 / 202, factor,
                      math.log(1e6) / (-2 * math.log(factor))])

    check(rates, TARGETS["18.10"], rtol=1e-3, name="rates and steps");
    """),
    code(r"""
    # EXERCISE 18.12: the batch that gives a scatter of 0.05 for σ_g = 0.585.
    batch = None

    check(batch, TARGETS["18.12"], rtol=1e-3, name="batch")
    """),
    solution(r"""
    # SOLUTION 18.12. σ_g/√b = 0.05.
    batch = (0.585 / 0.05) ** 2

    check(batch, TARGETS["18.12"], rtol=1e-3, name="batch");
    """),
    code(r"""
    # EXERCISE 18.13: the steps for the velocity to fall to 1/e with
    # μ = 0.9, and the friction 1 − μ.
    memory = None

    check(memory, TARGETS["18.13"], rtol=1e-6, name="steps and friction")
    """),
    solution(r"""
    # SOLUTION 18.13. 0.9^t = 1/e.
    memory = np.array([-1 / math.log(0.9), 1 - 0.9])

    check(memory, TARGETS["18.13"], rtol=1e-6, name="steps and friction");
    """),
    code(r"""
    # EXERCISE 18.14: Polyak's √μ, μ and η for eigenvalues 1 and 100, and
    # the steps to shrink by 1e-4 with momentum and without.
    polyak = None

    check(polyak, TARGETS["18.14"], rtol=1e-3, name="Polyak")
    """),
    solution(r"""
    # SOLUTION 18.14. √μ = (√κ − 1)/(√κ + 1), η = 4/(√λ_max + √λ_min)².
    s = (10 - 1) / (10 + 1)
    polyak = np.array([s, s * s, 4 / (10 + 1) ** 2,
                       math.log(1e-4) / math.log(s),
                       math.log(1e-4) / math.log(99 / 101)])

    check(polyak, TARGETS["18.14"], rtol=1e-3, name="Polyak");
    """),
    code(r"""
    # EXERCISE 18.15: m/g and s/g² after 10 steps of a steady gradient, and
    # the factor by which the uncorrected step is too large.
    adam_start = None

    check(adam_start, TARGETS["18.15"], rtol=1e-4, name="Adam's start")
    """),
    solution(r"""
    # SOLUTION 18.15. m = (1 − β₁ᵗ)g, s = (1 − β₂ᵗ)g².
    m, s = 1 - 0.9**10, 1 - 0.999**10
    adam_start = np.array([m, s, m / math.sqrt(s)])

    check(adam_start, TARGETS["18.15"], rtol=1e-4, name="Adam's start");
    """),
    code(r"""
    # EXERCISE 18.17: independent frames in 2 and 20 ps at 10 fs, g = 109.
    independent = None

    check(independent, TARGETS["18.17"], rtol=1e-6, name="independent frames")
    """),
    solution(r"""
    # SOLUTION 18.17. The number of frames over g.
    independent = np.array([200 / 109, 2000 / 109])

    check(independent, TARGETS["18.17"], rtol=1e-6, name="independent frames");
    """),
]

CELLS = (SETUP + LOSS + BASIS + RIDGE + GP + DESCENT + SGD + MOMENTUM
         + SPLITS + TRAINING + EXERCISES)

if __name__ == "__main__":
    print("wrote", write(CELLS, "18_learning.ipynb"))
