"""Bonds, molecules and reactions (Chapter 16)."""

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, neighbor_list

from mdlab import bonds
from mdlab.neighbours import all_pairs, cell_list_pairs

BOX = 30.0 * np.eye(3)


def _step(log, time, r):
    i, j, _, dist = all_pairs(r, BOX, log.tracker.reach)
    log.add(time, i, j, dist)


def test_first_frame_against_brute_force_and_ase():
    rng = np.random.default_rng(0)
    symbols = ["C"] * 40 + ["H"] * 40
    r = rng.uniform(0, 8, (80, 3))
    h = 8.0 * np.eye(3)
    tracker = bonds.BondTracker(symbols, on=1.0, off=1.3)
    i, j, _, dist = cell_list_pairs(r, h, tracker.reach)
    tracker.update(i, j, dist)
    atoms = Atoms(symbols, positions=r, cell=h, pbc=True)
    a, b = neighbor_list("ij", atoms, natural_cutoffs(atoms))
    theirs = np.unique(np.minimum(a, b) * 80 + np.maximum(a, b))
    assert np.array_equal(tracker.keys, theirs)
    radius = bonds.covalent_radii(symbols)
    i, j, _, dist = all_pairs(r, h, tracker.reach)
    brute = np.sort(i[dist < radius[i] + radius[j]] * 80
                    + j[dist < radius[i] + radius[j]])
    assert np.array_equal(tracker.keys, brute)


def test_hysteresis_band():
    tracker = bonds.BondTracker(["X", "X"], pairs={("X", "X"): (1.0, 1.5)})
    seen = []
    for d in (1.2, 0.9, 1.2, 1.4, 1.6, 1.2):
        formed, broken = tracker.update([0], [1], [d])
        seen.append((len(formed), len(broken), len(tracker.keys)))
    assert seen == [(0, 0, 0), (1, 0, 1), (0, 0, 1), (0, 0, 1), (0, 1, 0),
                    (0, 0, 0)]


def test_dissociation_and_recombination():
    log = bonds.ReactionLog(bonds.BondTracker(["N"] * 4))
    r = np.array([[0, 0, 0], [1.1, 0, 0], [10, 0, 0], [11.1, 0, 0]], float)
    _step(log, 0.0, r)
    far = r.copy()
    far[1, 0] = 5.0
    _step(log, 1.0, far)
    near = far.copy()
    near[1, 0] = 8.9
    _step(log, 2.0, near)
    assert log.counts[0] == {"N2": 2}
    assert log.counts[1] == {"N2": 1, "N": 2}
    assert log.counts[2] == {"N3": 1, "N": 1}
    network = log.network()
    assert network[(("N2",), ("N", "N"))] == 1
    assert network[(("N", "N2"), ("N3",))] == 1


def test_a_ring_that_opens_is_an_isomer():
    angle = 2 * np.pi * np.arange(6) / 6
    ring = 1.4 * np.column_stack([np.cos(angle), np.sin(angle),
                                  np.zeros(6)]) + 15
    log = bonds.ReactionLog(bonds.BondTracker(["C"] * 6))
    _step(log, 0.0, ring)
    opened = ring.copy()
    away = np.array([1.0, -0.3, 0.0]) / np.hypot(1.0, 0.3)
    opened[0] = ring[5] + 1.4 * away  # still bonded to atom 5 only
    _step(log, 1.0, opened)
    assert len(log.tracker.keys) == 5
    assert log.reactions == [(1.0, ("C6",), ("C6'",))]


def test_hash_ignores_numbering():
    rng = np.random.default_rng(4)
    keys = np.array([0 * 7 + 1, 1 * 7 + 2, 2 * 7 + 3, 3 * 7 + 4, 4 * 7 + 5,
                     5 * 7 + 6, 1 * 7 + 5])
    symbols = ["C", "C", "O", "C", "N", "C", "H"]
    labels = bonds.molecules(7, keys)
    code = bonds.graph_hash(symbols, keys, labels)
    order = rng.permutation(7)
    new = np.empty(7, dtype=int)
    new[order] = np.arange(7)
    i, j = np.divmod(keys, 7)
    a, b = new[i], new[j]
    moved = np.minimum(a, b) * 7 + np.maximum(a, b)
    moved_symbols = [symbols[k] for k in order]
    code2 = bonds.graph_hash(moved_symbols, moved,
                             bonds.molecules(7, moved))
    assert code2[0] == code[0]
    assert bonds.formulas(symbols, labels) == ["C4HNO"]


def test_brief_contacts_are_dropped():
    events = [(0.0, 0, 1, 1), (10.0, 0, 1, -1), (12.0, 0, 1, 1),
              (100.0, 0, 1, -1), (5.0, 2, 3, 1)]
    kept = bonds.persistent(sorted(events), 5.0)
    assert kept == [(0.0, 0, 1, 1), (5.0, 2, 3, 1), (100.0, 0, 1, -1)]


def test_update_against_the_rule_over_many_frames():
    rng = np.random.default_rng(7)
    symbols = ["C"] * 30 + ["O"] * 30
    h = 9.0 * np.eye(3)
    tracker = bonds.BondTracker(symbols, on=1.1, off=1.4)
    radius = bonds.covalent_radii(symbols)
    r = rng.uniform(0, 9, (60, 3))
    reference: set = set()
    for _ in range(40):
        r = r + rng.normal(0, 0.15, r.shape)
        i, j, _, dist = cell_list_pairs(r, h, tracker.reach)
        tracker.update(i, j, dist)
        total = radius[i] + radius[j]
        close = {(a, b) for a, b, d, s in zip(i, j, dist, total, strict=True)
                 if d < 1.1 * s}
        held = {(a, b) for a, b, d, s in zip(i, j, dist, total, strict=True)
                if d < 1.4 * s}
        reference = close | (reference & held)
        assert sorted(a * 60 + b for a, b in reference) == list(tracker.keys)


def test_hill_order_with_and_without_carbon():
    symbols = ["O", "H", "H", "C", "H", "H", "H", "H", "Li", "F"]
    keys = np.array([0 * 10 + 1, 0 * 10 + 2, 3 * 10 + 4, 3 * 10 + 5,
                     3 * 10 + 6, 3 * 10 + 7, 8 * 10 + 9])
    labels = bonds.molecules(10, keys)
    assert sorted(bonds.formulas(symbols, labels)) == ["CH4", "FLi", "H2O"]


def test_pairs_of_absent_elements_are_ignored():
    """Thresholds for an element the system lacks change no other pair."""
    plain = bonds.BondTracker(["C", "H", "O"])
    extra = bonds.BondTracker(["C", "H", "O"], pairs={("C", "N"): (9.0, 9.5)})
    assert np.array_equal(plain.r_on, extra.r_on)
    assert np.array_equal(plain.r_off, extra.r_off)
