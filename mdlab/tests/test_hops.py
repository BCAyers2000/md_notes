"""Hops between sites and rates that may be unresolved (Ch. 17)."""

import numpy as np
import pytest

from mdlab.analysis import hops


def test_nearest_site_through_the_boundary():
    sites = np.array([[0.0, 0, 0], [5.0, 0, 0]])
    got = hops.nearest_site([[9.7, 0, 0], [3.0, 0, 0], [7.6, 0, 0]], sites,
                            10 * np.eye(3))
    assert list(got) == [0, 1, 0]


def test_brief_visits_are_not_hops():
    seq = [0, 0, 0, 1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0]
    rows = hops.hops(seq, least_stay=2)
    assert rows.tolist() == [[6, 0, 1], [12, 1, 0]]
    assert len(hops.hops(seq, least_stay=1)) == 6


def test_rate_against_a_poisson_process():
    """Counts of a Poisson process give the rate within its error, and
    'resolved' only from 25 hops."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.5 * 400, 2000)
    rates = np.array([hops.rate(c, 400)["rate"] for c in counts])
    errors = np.array([hops.rate(c, 400)["error"] for c in counts])
    assert rates.mean() == pytest.approx(0.5, rel=0.01)
    assert np.std(rates) == pytest.approx(np.mean(errors), rel=0.05)
    assert not hops.rate(24, 10.0)["resolved"]
    assert hops.rate(25, 10.0)["resolved"]
    none = hops.rate(0, 10.0)
    assert none["upper"] == pytest.approx(-np.log(0.05) / 10)
    assert none["needed"] == pytest.approx(25 / none["upper"])
