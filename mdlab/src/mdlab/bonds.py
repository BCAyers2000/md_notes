"""Bonds, molecules and reactions in a trajectory: Chapter 16.

- ``BondTracker``: the bonds of each frame from the pairs a neighbour list
  already holds, with a band of hysteresis: a bond forms when two atoms
  come closer than r_on and breaks only when they part beyond r_off,
  r_off > r_on, so that a pair near one threshold does not flicker. By
  default r_on and r_off are the sum of the two atoms' covalent radii
  (ASE's table) times the factors ``on`` and ``off`` (Section 16.3). Each
  bond is the whole number i·N + j, i < j, and the bonds formed or broken
  between frames are the differences of two sorted sets of them;
- ``molecules``: the connected components of the bond graph;
- ``formulas`` and ``graph_hash``: a name for each molecule, its formula
  in the Hill order and a number that is the same for molecules bonded
  the same way, whatever the numbering of their atoms;
- ``ReactionLog``: species counts, bond events and reactions, frame by
  frame; a reaction is a change of the molecules that some atoms belong
  to (Section 16.4);
- ``persistent``: the events left when a bond that forms and breaks again
  within a minimum lifetime is not counted.

Units
-----
Lengths in Å, times in fs.
"""

from collections import Counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_compiled: dict = {}


def _jit(function):
    """``function`` compiled by Numba on its first use."""
    if function.__name__ not in _compiled:
        import numba

        _compiled[function.__name__] = numba.njit(cache=False)(function)
    return _compiled[function.__name__]


def _held_keys(i, j, dist, kind, r_on, r_off, reach, n):
    """2(i·N + j) + [d < r_on] for each pair closer than its r_off.

    One pass over the candidates, compiled: most of them lie beyond the
    largest r_off and are passed over at once.
    """
    count = 0
    for p in range(len(dist)):
        if dist[p] < reach and dist[p] < r_off[kind[i[p]], kind[j[p]]]:
            count += 1
    out = np.empty(count, np.int64)
    k = 0
    for p in range(len(dist)):
        if dist[p] < reach:
            a, b = kind[i[p]], kind[j[p]]
            if dist[p] < r_off[a, b]:
                out[k] = 2 * (i[p] * n + j[p]) + (dist[p] < r_on[a, b])
                k += 1
    return out


def _rounds(lab, i, j, labels, size, rounds):
    """The rounds of ``graph_hash`` and the sum over each molecule.

    ``_mix`` is written out for one number at a time, for Numba.
    """
    golden = np.uint64(0x9E3779B97F4A7C15)
    m1, m2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
    s30, s27, s31 = np.uint64(30), np.uint64(27), np.uint64(31)
    n = len(lab)
    mixed = np.empty(n, np.uint64)
    for r in range(rounds + 1):
        for a in range(n):
            z = lab[a] + golden
            z = (z ^ (z >> s30)) * m1
            z = (z ^ (z >> s27)) * m2
            mixed[a] = z ^ (z >> s31)
        if r == rounds:
            break
        around = np.zeros(n, np.uint64)
        for p in range(len(i)):
            around[i[p]] += mixed[j[p]]
            around[j[p]] += mixed[i[p]]
        for a in range(n):
            z = lab[a] * np.uint64(0x100000001B3) + around[a] + golden
            z = (z ^ (z >> s30)) * m1
            z = (z ^ (z >> s27)) * m2
            lab[a] = z ^ (z >> s31)
    total = np.zeros(size, np.uint64)
    for a in range(n):
        total[labels[a]] += mixed[a]
    return total


def _merge(keys, close, old):
    """The new bonds, those formed and those broken, in one pass.

    ``keys`` (sorted) are the pairs within r_off, ``close`` marks those
    within r_on, ``old`` (sorted) the bonds of the last frame. A pair is
    bonded if it is close, or within r_off and bonded before.
    """
    new = np.empty(len(keys), np.int64)
    formed = np.empty(len(keys), np.int64)
    broken = np.empty(len(old), np.int64)
    n_new = n_formed = n_broken = 0
    p = q = 0
    while p < len(keys) or q < len(old):
        if q == len(old) or (p < len(keys) and keys[p] < old[q]):
            if close[p]:  # a pair that was not bonded comes within r_on
                new[n_new] = keys[p]
                formed[n_formed] = keys[p]
                n_new += 1
                n_formed += 1
            p += 1
        elif p == len(keys) or old[q] < keys[p]:  # an old bond beyond r_off
            broken[n_broken] = old[q]
            n_broken += 1
            q += 1
        else:  # bonded before and still within r_off
            new[n_new] = keys[p]
            n_new += 1
            p += 1
            q += 1
    return new[:n_new], formed[:n_formed], broken[:n_broken]


def _mix(x: NDArray) -> NDArray:
    """A fixed scrambling of 64-bit whole numbers (splitmix64's finaliser)."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def covalent_radii(symbols) -> NDArray:
    """ASE's covalent radius of each atom in Å; NaN for a symbol it lacks."""
    from ase.data import atomic_numbers
    from ase.data import covalent_radii as table

    return np.array([table[atomic_numbers[s]] if s in atomic_numbers
                     else np.nan for s in symbols])


def _symbol_codes(symbols) -> NDArray:
    """A whole number for each symbol, the same in every run."""
    return np.array([int.from_bytes(s.encode()[:8].ljust(8, b"\0"), "little")
                     for s in symbols], dtype=np.uint64)


class BondTracker:
    """The bonds of a trajectory, frame after frame, with hysteresis.

    ``on`` and ``off`` multiply the sum of the covalent radii; ``pairs``
    maps a pair of symbols, e.g. ("C", "H"), to its own (r_on, r_off) in
    Å; it must cover every pair of symbols that ASE's table lacks, such
    as the 'A' and 'B' of a model. Candidate pairs given to ``update``
    must include every pair closer than the largest r_off, as a neighbour
    list's pairs do.
    """

    def __init__(self, symbols, on: float = 1.2, off: float = 1.4,
                 pairs: dict | None = None):
        self.symbols = list(symbols)
        self.n = len(self.symbols)
        kinds, self.kind = np.unique(self.symbols, return_inverse=True)
        radius = covalent_radii(kinds)
        total = radius[:, None] + radius[None, :]
        self.r_on, self.r_off = on * total, off * total
        for (a, b), (r_on, r_off) in (pairs or {}).items():
            if a not in kinds or b not in kinds:
                continue  # a pair of elements this system does not hold
            ka, kb = np.searchsorted(kinds, a), np.searchsorted(kinds, b)
            self.r_on[ka, kb] = self.r_on[kb, ka] = r_on
            self.r_off[ka, kb] = self.r_off[kb, ka] = r_off
        if np.isnan(self.r_on).any():
            raise ValueError("give (r_on, r_off) for every pair of symbols "
                             "without a covalent radius")
        self._reach = self.r_off.max()
        self.keys = np.zeros(0, dtype=np.int64)

    @property
    def reach(self) -> float:
        """The largest r_off: candidate pairs must reach this far."""
        return float(self._reach)

    def update(
        self, i: ArrayLike, j: ArrayLike, dist: ArrayLike
    ) -> tuple[NDArray, NDArray]:
        """Move to the next frame; return the keys formed and broken.

        ``i`` < ``j`` and ``dist`` are the candidate pairs and their
        distances in the new frame, each pair once.
        """
        # the keys within r_off, each carrying its "closer than r_on" flag
        # in an extra lowest bit, so that one sort orders both
        packed = np.sort(_jit(_held_keys)(
            np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64),
            np.asarray(dist, dtype=float), self.kind, self.r_on, self.r_off,
            self._reach, self.n))
        self.keys, formed, broken = _jit(_merge)(
            packed >> 1, (packed & 1).astype(np.bool_), self.keys)
        return formed, broken


def molecules(n_atoms: int, keys: ArrayLike) -> NDArray:
    """The molecule of each atom: the connected components of the bonds."""
    i, j = np.divmod(np.asarray(keys, dtype=np.int64), n_atoms)
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n_atoms, n_atoms))
    return connected_components(graph, directed=False)[1]


def graph_hash(
    symbols, keys: ArrayLike, labels: ArrayLike, rounds: int = 4
) -> NDArray:
    """A 64-bit number per molecule that does not depend on the numbering.

    Each atom starts with a number for its element; in each round it takes
    a scrambled mix of its own number and the sum of its neighbours'
    scrambled numbers, so that after k rounds its number encodes the
    bonded neighbourhood k bonds deep (the Weisfeiler-Lehman scheme). The
    molecule's number is the sum of its atoms' scrambled numbers. Sums do
    not depend on order, so renumbering the atoms changes nothing; two
    molecules bonded differently almost always differ, though a few pairs
    of distinct graphs share every such number.
    """
    n = len(symbols)
    i, j = np.divmod(np.asarray(keys, dtype=np.int64), n)
    labels = np.asarray(labels, dtype=np.int64)
    return _jit(_rounds)(_mix(_symbol_codes(symbols)), i, j, labels,
                         labels.max() + 1, rounds)


def formulas(symbols, labels: ArrayLike) -> list[str]:
    """The formula of each molecule in the Hill order.

    In a molecule that holds carbon, C comes first and H second, then the
    other elements alphabetically; in one that does not, all of them come
    alphabetically.
    """
    kinds, kind = np.unique(list(symbols), return_inverse=True)
    labels = np.asarray(labels)
    table = np.zeros((labels.max() + 1, len(kinds)), dtype=int)
    np.add.at(table, (labels, kind), 1)
    carbon = {"C": 0, "H": 1}
    hill = sorted(range(len(kinds)),
                  key=lambda k: (carbon.get(kinds[k], 2), kinds[k]))
    alphabetical = sorted(range(len(kinds)), key=lambda k: kinds[k])
    has_carbon = "C" in kinds
    cache: dict[tuple, str] = {}
    names = []
    for row in table:
        key = tuple(row)
        if key not in cache:
            with_carbon = has_carbon and row[list(kinds).index("C")] > 0
            order = hill if with_carbon else alphabetical
            cache[key] = "".join(
                kinds[k] + (str(row[k]) if row[k] > 1 else "")
                for k in order if row[k])
        names.append(cache[key])
    return names


class ReactionLog:
    """Species, bond events and reactions, one frame at a time.

    Each species is a graph hash; ``names`` gives it its formula, primed
    for each further isomer of the same formula, in order of appearance.
    """

    def __init__(self, tracker: BondTracker):
        self.tracker = tracker
        self.times: list[float] = []
        self.counts: list[Counter] = []
        self._events: list[NDArray] = []
        self.reactions: list[tuple[float, tuple, tuple]] = []
        self.names: dict[int, str] = {}
        self._labels: NDArray | None = None
        self._species: NDArray | None = None

    def _name(self, code: int, formula: str) -> str:
        if code not in self.names:
            taken = sum(1 for v in self.names.values()
                        if v.rstrip("'") == formula)
            self.names[code] = formula + "'" * taken
        return self.names[code]

    def add(self, time: float, i: ArrayLike, j: ArrayLike,
            dist: ArrayLike) -> None:
        """Take the next frame's candidate pairs and record what changed.

        The bonds found in the first frame start the record; only those
        formed or broken after it are events.
        """
        t = self.tracker
        formed, broken = t.update(i, j, dist)
        first = self._labels is None  # the first frame's bonds are no events
        for keys, sign in ((formed, 1), (broken, -1)):
            if len(keys) and not first:
                a, b = np.divmod(keys, t.n)
                self._events.append(np.column_stack(
                    [np.full(len(a), time), a, b, np.full(len(a), sign)]))
        labels = molecules(t.n, t.keys)
        codes = graph_hash(t.symbols, t.keys, labels)
        changed = self._labels is None or len(formed) or len(broken)
        if changed:
            forms = formulas(t.symbols, labels)
            names = [self._name(int(c), f)
                     for c, f in zip(codes, forms, strict=True)]
            self._names_now = names
        names = self._names_now
        self.times.append(time)
        self.counts.append(Counter(names))
        if self._labels is not None and changed:
            self._react(time, labels, codes)
        self._labels, self._species = labels, codes

    def _react(self, time, labels, codes):
        old, old_codes = self._labels, self._species
        n_old = old.max() + 1
        size = n_old + labels.max() + 1
        edges = np.unique(old.astype(np.int64) * size + n_old + labels)
        graph = coo_matrix((np.ones(len(edges)), np.divmod(edges, size)),
                           shape=(size, size))
        group = connected_components(graph, directed=False)[1]
        nodes_old, nodes_new = group[:n_old], group[n_old:]
        for g in np.unique(group):
            before = np.flatnonzero(nodes_old == g)
            after = np.flatnonzero(nodes_new == g)
            if (len(before) == 1 and len(after) == 1
                    and old_codes[before[0]] == codes[after[0]]):
                continue
            reactants = tuple(sorted(self.names[int(old_codes[k])]
                                     for k in before))
            products = tuple(sorted(self.names[int(codes[k])]
                                    for k in after))
            self.reactions.append((time, reactants, products))

    @property
    def events(self) -> list[tuple[float, int, int, int]]:
        """Every bond formed (+1) or broken (−1): (time, i, j, ±1)."""
        if not self._events:
            return []
        rows = np.concatenate(self._events)
        return [(float(t), int(i), int(j), int(s)) for t, i, j, s in rows]

    def network(self) -> Counter:
        """How often each reaction (reactants, products) happened."""
        return Counter((r, p) for _, r, p in self.reactions)


def persistent(events: list, min_lifetime: float) -> list:
    """The events left after removing bonds that lived too briefly.

    A bond formed and broken again (or broken and formed again) within
    ``min_lifetime`` is taken as a passing contact, and both of its events
    are dropped. ``events`` are (time, i, j, ±1) in order of time.
    """
    by_pair: dict[tuple, list] = {}
    for event in events:
        by_pair.setdefault(event[1:3], []).append(event)
    kept = []
    for history in by_pair.values():
        stack: list = []
        for event in history:
            if (stack and stack[-1][3] == -event[3]
                    and event[0] - stack[-1][0] < min_lifetime):
                stack.pop()
            else:
                stack.append(event)
        kept += stack
    return sorted(kept)
