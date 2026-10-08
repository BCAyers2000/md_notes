"""Tests of mdlab.learn against NumPy, scikit-learn and PyTorch."""

import numpy as np
import pytest

from mdlab import learn


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n))
    y = np.sin(3 * x) + 0.1 * rng.normal(size=n)
    return x, y


def test_least_squares_matches_numpy():
    x, y = _data()
    X = learn.polynomial_basis(x, 5)
    assert np.allclose(learn.least_squares(X, y),
                       np.linalg.lstsq(X, y, rcond=None)[0])


def test_ridge_matches_its_normal_equations_and_scikit_learn():
    from sklearn.linear_model import Ridge

    x, y = _data()
    X = learn.polynomial_basis(x, 9)
    lam = 0.3
    w = learn.least_squares(X, y, ridge=lam)
    normal = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    assert np.allclose(w, normal)
    ridge = Ridge(alpha=lam, fit_intercept=False).fit(X, y)
    assert np.allclose(w, ridge.coef_)


def test_gaussian_process_matches_scikit_learn():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    x, y = _data(15)
    gp = learn.GaussianProcess(length=0.4, height=1.3, noise=0.1).fit(x, y)
    grid = np.linspace(-1.2, 1.2, 50)
    mean, var = gp.predict(grid)
    kernel = (ConstantKernel(1.3**2, constant_value_bounds="fixed")
              * RBF(0.4, length_scale_bounds="fixed"))
    ref = GaussianProcessRegressor(kernel, alpha=0.1**2, optimizer=None,
                                   normalize_y=False).fit(x[:, None], y)
    ref_mean, ref_std = ref.predict(grid[:, None], return_std=True)
    assert np.allclose(mean, ref_mean)
    assert np.allclose(np.sqrt(var), ref_std, atol=1e-8)
    assert gp.log_marginal_likelihood() == pytest.approx(
        ref.log_marginal_likelihood_value_)


def test_gaussian_process_passes_through_noise_free_points():
    x = np.array([0.0, 0.5, 1.3])
    gp = learn.GaussianProcess(length=0.7, noise=1e-8).fit(x, np.cos(x))
    mean, var = gp.predict(x)
    assert np.allclose(mean, np.cos(x), atol=1e-6)
    assert np.all(var < 1e-10)


def test_gradient_descent_contracts_each_direction_by_1_less_rate_lambda():
    A = np.diag([1.0, 10.0])
    w = learn.minimise(lambda w: (0.5 * w @ A @ w, A @ w), [1.0, 1.0],
                       learn.GradientDescent(0.15), 5)[0]
    assert np.allclose(w[-1], [(1 - 0.15) ** 5, (1 - 1.5) ** 5])


def _torch_path(optimiser, steps=20):
    import torch

    A = torch.tensor([[3.0, 1.0], [1.0, 0.5]], dtype=torch.float64)
    p = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    opt = optimiser([p])
    path = [p.detach().clone().numpy()]
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * p @ A @ p + torch.sin(p).sum()
        loss.backward()
        opt.step()
        path.append(p.detach().clone().numpy())
    return np.array(path)


def _ours(optimiser, steps=20):
    A = np.array([[3.0, 1.0], [1.0, 0.5]])

    def loss_and_gradient(w):
        return 0.5 * w @ A @ w + np.sin(w).sum(), A @ w + np.cos(w)

    return learn.minimise(loss_and_gradient, [1.0, -2.0], optimiser,
                          steps)[0]


def test_momentum_matches_torch_sgd():
    import torch

    ours = _ours(learn.Momentum(0.05, momentum=0.8))
    theirs = _torch_path(lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.8))
    assert np.allclose(ours, theirs)


def test_adam_matches_torch_adam():
    import torch

    ours = _ours(learn.Adam(0.1, beta1=0.85, beta2=0.99, eps=1e-8))
    theirs = _torch_path(lambda p: torch.optim.Adam(
        p, lr=0.1, betas=(0.85, 0.99), eps=1e-8))
    assert np.allclose(ours, theirs)


def test_batches_cover_every_item_once():
    seen = np.concatenate(list(learn.batches(23, 5,
                                             np.random.default_rng(1))))
    assert sorted(seen) == list(range(23))


def test_splits_partition_the_items():
    rng = np.random.default_rng(2)
    for train, valid in (learn.random_split(100, 0.2, rng),
                         learn.blocked_split(100, 0.2, 10, rng)):
        assert len(np.intersect1d(train, valid)) == 0
        assert len(train) + len(valid) == 100
    train, valid = learn.blocked_split(100, 0.2, 10, rng)
    assert len(valid) == 20 and np.all(valid.reshape(2, 10) % 10
                                       == np.arange(10))
