"""3-way all-in outcome tensor for the 169 hand classes (built by csrc/equity3.c).

T[x, y, z] = (number of card-disjoint combo triples of classes x, y, z) * P(x ends best, y second, z worst).
Summing T over the six permutations of its axes gives the combo triple count W3. T splits ties evenly
between the orders they fit; the ties themselves are in a second file, as corrections (tie_tensors).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from mbr.ranges import N_CLASSES

EQUITY3_FILE = Path(__file__).resolve().parent.parent / "build" / "equity3.f32"
TIES3_FILE = EQUITY3_FILE.with_name("equity3_ties.f32")
BUILD = ("gcc -O3 -march=native -fopenmp -o build/equity3 csrc/equity3.c && "
         "build/equity3 1000 build/equity3.f32 build/equity3_ties.f32")

# strength orders of the three roles (0 = shover, 1 = caller, 2 = overcaller), best first
ORDERS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
# tied outcomes: groups of roles from the best hand to the worst
TIES = [((0, 1), (2,)), ((0, 2), (1,)), ((1, 2), (0,)), ((0,), (1, 2)), ((1,), (0, 2)), ((2,), (0, 1)), ((0, 1, 2),)]


def available():
    return EQUITY3_FILE.exists() and TIES3_FILE.exists()


def _missing():
    return FileNotFoundError(f"{EQUITY3_FILE} or {TIES3_FILE.name} is missing; build them with: {BUILD}")


def _layout(t, roles):
    """t[x, y, z] with role roles[p] at axis p, as a (169, 169 * 169) matrix over (overcaller; shover, caller)."""
    n = N_CLASSES
    return np.ascontiguousarray(t.transpose(roles.index(2), roles.index(0), roles.index(1))).reshape(n, n * n)


@lru_cache(maxsize=1)
def order_tensors():
    """For each strength order o in ORDERS, the matrix V_o of shape (169, 169 * 169) with
    V_o[c, a * 169 + b] = U_o[a, b, c] = combo weight * P(order o) for shover hand a, caller b, overcaller c."""
    if not available():
        raise _missing()
    n = N_CLASSES
    t = np.fromfile(EQUITY3_FILE, dtype=np.float32).reshape(n, n, n)
    # position p of T holds the hand of role o[p]; order the axes as overcaller, shover, caller
    return tuple(_layout(t, o) for o in ORDERS)


@lru_cache(maxsize=1)
def tie_tensors():
    """For each tied outcome in TIES, the matrix like order_tensors() of combo weight * P(outcome).

    The outcome's probability is also in order_tensors(), split evenly between the orders it fits, so
    these come in as corrections: weight * (value of the tie - mean value of those orders)."""
    if not available():
        raise _missing()
    n = N_CLASSES
    best, last, every = np.fromfile(TIES3_FILE, dtype=np.float32).reshape(3, n, n, n)
    out = []
    for groups in TIES:
        roles = tuple(r for g in groups for r in g)
        out.append(_layout(every if len(groups) == 1 else best if len(groups[0]) == 2 else last, roles))
    return tuple(out)


def fitting_orders(groups):
    """The strict orders (indices into ORDERS) that a tied outcome fits."""
    rank = {r: k for k, g in enumerate(groups) for r in g}
    return [q for q, o in enumerate(ORDERS) if all(rank[o[p]] <= rank[o[p + 1]] for p in range(2))]


@lru_cache(maxsize=1)
def order_mass():
    """D[a * 169 + b] = sum over c and the strength orders of U_o[a, b, c] (the combo triple count), at least 1e-9."""
    return np.maximum(sum(t.sum(axis=0) for t in order_tensors()), 1e-9).astype(np.float32)
