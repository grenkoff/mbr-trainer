"""Showdowns between known hands: how the pots go on every run-out of the board (csrc/showdown.c).

Used for the all-in adjusted results of the hand histories: a hand where the betting ended before the river
with every remaining player's cards shown is counted at its expectation over the boards still to come.
"""

import ctypes
import subprocess
from collections import Counter
from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = ROOT / "build" / "libshowdown.so"
BUILD = ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(LIB_PATH), str(ROOT / "csrc" / "showdown.c")]
RANKS, SUITS = "23456789TJQKA", "cdhs"
MAX_BOARDS = 100_000  # more run-outs than this (preflop: 1.7M) are sampled: ±0.15% equity


@lru_cache(maxsize=1)
def _lib():
    if not LIB_PATH.exists():
        LIB_PATH.parent.mkdir(exist_ok=True)
        subprocess.run(BUILD, check=True)
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.showdown_sigs.restype = ctypes.c_long
    lib.showdown_sigs.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int), ctypes.c_long, ctypes.c_uint64,
                                  np.ctypeslib.ndpointer(np.uint64)]
    return lib


def card(text):
    """'Ah' -> 48 + 2: rank * 4 + suit, as csrc/eval7.h."""
    return RANKS.index(text[0]) * 4 + SUITS.index(text[1])


def rankings(holes, board):
    """[(better, probability)] over the run-outs of the board (sampled if there are more than MAX_BOARDS):
    better[i] is how many players have a strictly better hand than player i. holes: two cards per player
    ('Ah', 'Kd'); board: the cards dealt so far."""
    n, nb = len(holes), len(board)
    hole = (ctypes.c_int * (2 * n))(*[card(c) for h in holes for c in h])
    brd = (ctypes.c_int * max(nb, 1))(*[card(c) for c in board])
    boards = comb(52 - 2 * n - nb, 5 - nb)
    samples = MAX_BOARDS if boards > MAX_BOARDS else 0
    out = np.empty(samples or boards, dtype=np.uint64)
    count = _lib().showdown_sigs(n, hole, nb, brd, samples, 0x9E3779B97F4A7C15, out)
    sigs, counts = np.unique(out[:count], return_counts=True)
    return [(tuple(int(s) >> (4 * i) & 15 for i in range(n)), c / count) for s, c in zip(sigs, counts)]


def split_pots(put_in, live, better):
    """Chips each player wins: put_in holds every player's chips in the pot (folded ones too), live the
    players in the showdown, better their rankings (see rankings()). Returns (won, layer winners): every pot
    layer goes to its best eligible hands, split between ties; a layer is named by its top level."""
    won = Counter()
    layers = {}
    prev = 0
    for level in sorted({put_in[x] for x in live}):
        chips = sum(min(p, level) - min(p, prev) for p in put_in.values())
        eligible = [x for x in live if put_in[x] >= level]
        best = min(better[x] for x in eligible)
        winners = layers[level] = [x for x in eligible if better[x] == best]
        for x in winners:
            won[x] += chips / len(winners)
        prev = level
    return won, layers
