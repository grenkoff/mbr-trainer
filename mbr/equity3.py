"""3-way all-in outcome tensor for the 169 hand classes (built by csrc/equity3.c).

T[x, y, z] = (number of card-disjoint combo triples of classes x, y, z) * P(x ends best, y second, z worst).
Summing T over the six permutations of its axes gives the combo triple count W3.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from mbr.ranges import N_CLASSES

EQUITY3_FILE = Path(__file__).resolve().parent.parent / "build" / "equity3.f32"

# strength orders of the three roles (0 = shover, 1 = caller, 2 = overcaller), best first
ORDERS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def available():
    return EQUITY3_FILE.exists()


@lru_cache(maxsize=1)
def order_tensors():
    """For each strength order o in ORDERS, the matrix V_o of shape (169, 169 * 169) with
    V_o[c, a * 169 + b] = U_o[a, b, c] = combo weight * P(order o) for shover hand a, caller b, overcaller c."""
    if not available():
        raise FileNotFoundError(
            f"{EQUITY3_FILE} is missing; build it with: "
            "gcc -O3 -march=native -fopenmp -o build/equity3 csrc/equity3.c && build/equity3 1000 build/equity3.f32"
        )
    n = N_CLASSES
    t = np.fromfile(EQUITY3_FILE, dtype=np.float32).reshape(n, n, n)
    # position p of T holds the hand of role o[p]; order the axes as overcaller, shover, caller
    return tuple(np.ascontiguousarray(t.transpose(o.index(2), o.index(0), o.index(1))).reshape(n, n * n)
                 for o in ORDERS)


@lru_cache(maxsize=1)
def order_mass():
    """D[a * 169 + b] = sum over c and the strength orders of U_o[a, b, c] (the combo triple count), at least 1e-9."""
    return np.maximum(sum(t.sum(axis=0) for t in order_tensors()), 1e-9).astype(np.float32)
