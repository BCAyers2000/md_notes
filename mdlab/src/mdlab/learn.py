"""A model learned from data by lowering a loss: Chapter 18.

- ``polynomial_basis``: the matrix whose columns are the powers of a
  variable, from 0 to a degree (Section 18.2);
- ``least_squares``: the weights that make the sum of squared misses
  least, with an optional ridge penalty λ|w|² (Sections 18.1 and 18.3);
- ``GaussianProcess``: regression with a squared-exponential kernel, its
  posterior mean and variance at new points and the log marginal
  likelihood of the data (Section 18.4);
- ``GradientDescent``, ``Momentum`` and ``Adam``: optimisers, each with a
  ``step(w, g)`` that returns the next weights (Sections 18.5 and 18.7);
- ``minimise``: an optimiser run on a loss for a number of steps;
- ``batches``: the mini-batches of one pass over the data (Section 18.6);
- ``random_split`` and ``blocked_split``: training and validation
  indices, the second keeping runs of consecutive items together
  (Section 18.8).

Units
-----
Whatever the data carry: the weights take the units that make the model
match the targets.
"""

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray


def polynomial_basis(x: ArrayLike, degree: int) -> NDArray:
    """The design matrix [1, x, x², …, x^degree], one row per point.

    >>> polynomial_basis([2.0, 3.0], 2)
    array([[1., 2., 4.],
           [1., 3., 9.]])
    """
    x = np.asarray(x, dtype=float)
    return x[:, None] ** np.arange(degree + 1)


def least_squares(X: ArrayLike, y: ArrayLike, ridge: float = 0.0) -> NDArray:
    """The weights w that make |Xw − y|² + ridge·|w|² least.

    They solve (XᵀX + ridge·I)w = Xᵀy. The equivalent system of the stacked
    matrix [X; √ridge·I] against [y; 0] is solved by NumPy's least
    squares, which does not form XᵀX and so keeps the precision that its
    square would lose.

    >>> least_squares([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], [1.0, 3.0, 5.0])
    array([1., 2.])
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if ridge:
        p = X.shape[1]
        X = np.vstack([X, np.sqrt(ridge) * np.eye(p)])
        y = np.concatenate([y, np.zeros((p, *y.shape[1:]))])
    return np.linalg.lstsq(X, y, rcond=None)[0]


class GaussianProcess:
    """Regression by a Gaussian process with a squared-exponential kernel.

    The prior makes f(x) Gaussian with mean 0 and the covariance
    k(x, x′) = height²·exp(−|x − x′|²/2·length²); each target is f plus
    Gaussian noise of standard deviation ``noise``. ``fit`` conditions on
    the data, ``predict`` gives the posterior mean and variance of f.

    >>> gp = GaussianProcess(length=1.0).fit([0.0, 1.0], [0.0, 1.0])
    >>> np.round(gp.predict([1.0])[0], 6)
    array([1.])
    """

    def __init__(self, length: float, height: float = 1.0,
                 noise: float = 1e-6):
        self.length, self.height, self.noise = length, height, noise

    def kernel(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """k(a_i, b_j) for points a (n, d) or (n,) and b (m, d) or (m,)."""
        a = np.asarray(a, dtype=float).reshape(len(a), -1)
        b = np.asarray(b, dtype=float).reshape(len(b), -1)
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return self.height**2 * np.exp(-0.5 * d2 / self.length**2)

    def fit(self, x: ArrayLike, y: ArrayLike) -> "GaussianProcess":
        """Condition on the points x and their targets y."""
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        K = self.kernel(self.x, self.x) + self.noise**2 * np.eye(len(self.y))
        self.chol = np.linalg.cholesky(K)  # K = L Lᵀ
        z = np.linalg.solve(self.chol, self.y)
        self.alpha = np.linalg.solve(self.chol.T, z)  # K⁻¹ y
        return self

    def predict(self, x: ArrayLike) -> tuple[NDArray, NDArray]:
        """The posterior mean and variance of f at the points x."""
        ks = self.kernel(x, self.x)
        mean = ks @ self.alpha
        v = np.linalg.solve(self.chol, ks.T)
        variance = self.height**2 - np.sum(v * v, axis=0)
        return mean, np.maximum(variance, 0.0)

    def log_marginal_likelihood(self) -> float:
        """The log of p(y | x): −½yᵀK⁻¹y − ½ln det K − (n/2)ln 2π."""
        n = len(self.y)
        return float(-0.5 * self.y @ self.alpha
                     - np.sum(np.log(np.diag(self.chol)))
                     - 0.5 * n * np.log(2 * np.pi))


class GradientDescent:
    """w ← w − rate·g."""

    def __init__(self, rate: float):
        self.rate = rate

    def step(self, w: NDArray, g: NDArray) -> NDArray:
        """The next weights from w and the gradient g."""
        return w - self.rate * g


class Momentum:
    """Gradient descent with momentum: v ← μv + g, w ← w − rate·v.

    The velocity v remembers past gradients, so steps that agree add up
    and steps that alternate cancel; PyTorch's ``SGD`` with ``momentum``
    takes the same steps.
    """

    def __init__(self, rate: float, momentum: float = 0.9):
        self.rate, self.momentum = rate, momentum
        self.v = None

    def step(self, w: NDArray, g: NDArray) -> NDArray:
        """The next weights from w and the gradient g."""
        self.v = g if self.v is None else self.momentum * self.v + g
        return w - self.rate * self.v


class Adam:
    """Adam: steps scaled by running means of the gradient and its square.

    m ← β₁m + (1 − β₁)g and s ← β₂s + (1 − β₂)g², each divided by
    1 − βᵗ to undo its start at zero, and w ← w − rate·m̂/(√ŝ + ε): every
    weight moves about ``rate`` a step once its gradient is steady,
    whatever the gradient's size.
    """

    def __init__(self, rate: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.rate, self.beta1, self.beta2, self.eps = rate, beta1, beta2, eps
        self.m = self.s = None
        self.t = 0

    def step(self, w: NDArray, g: NDArray) -> NDArray:
        """The next weights from w and the gradient g."""
        if self.m is None:
            self.m, self.s = np.zeros_like(g), np.zeros_like(g)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.s = self.beta2 * self.s + (1 - self.beta2) * g * g
        m_hat = self.m / (1 - self.beta1**self.t)
        s_hat = self.s / (1 - self.beta2**self.t)
        return w - self.rate * m_hat / (np.sqrt(s_hat) + self.eps)


def minimise(
    loss_and_gradient: Callable[[NDArray], tuple[float, NDArray]],
    w0: ArrayLike,
    optimiser,
    steps: int,
) -> tuple[NDArray, NDArray]:
    """Take ``steps`` steps of ``optimiser`` from w0.

    Returns the weights before each step and after the last (steps + 1,
    …) and the loss at each of them.
    """
    w = np.asarray(w0, dtype=float)
    path, losses = [w], []
    for _ in range(steps):
        loss, g = loss_and_gradient(w)
        losses.append(loss)
        w = optimiser.step(w, g)
        path.append(w)
    losses.append(loss_and_gradient(w)[0])
    return np.array(path), np.array(losses)


def batches(n: int, size: int, rng: np.random.Generator) -> Iterator[NDArray]:
    """The mini-batches of one pass over n items, in a random order."""
    order = rng.permutation(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def random_split(n: int, fraction: float,
                 rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """Training and validation indices, the validation ones drawn at random.

    >>> train, valid = random_split(10, 0.2, np.random.default_rng(0))
    >>> len(train), len(valid)
    (8, 2)
    """
    order = rng.permutation(n)
    k = int(round(fraction * n))
    return np.sort(order[k:]), np.sort(order[:k])


def blocked_split(n: int, fraction: float, block: int,
                  rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """Training and validation indices, whole blocks of consecutive items.

    The n items are cut into blocks of ``block`` consecutive items, and
    whole blocks, chosen at random, go to validation until it holds about
    ``fraction`` of the items, so that no validation item has a neighbour
    in time among the training items except at the ends of its block.

    >>> blocked_split(12, 0.25, 3, np.random.default_rng(0))[1]
    array([6, 7, 8])
    """
    starts = np.arange(0, n, block)
    chosen = rng.permutation(len(starts))[:max(1, int(round(
        fraction * len(starts))))]
    valid = np.concatenate([np.arange(s, min(s + block, n))
                            for s in np.sort(starts[chosen])])
    train = np.setdiff1d(np.arange(n), valid)
    return train, valid
