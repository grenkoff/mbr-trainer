"""169 starting-hand classes on the standard 13x13 grid and combo bookkeeping."""

from functools import lru_cache
from pathlib import Path

import numpy as np

RANKS = "AKQJT98765432"  # grid order: index 0 = Ace
SUITS = "cdhs"
N_CLASSES = 169
EQUITY_FILE = Path(__file__).resolve().parent.parent / "data" / "equity_169.txt"
TIES_FILE = Path(__file__).resolve().parent.parent / "data" / "ties_169.txt"


def class_label(cls):
    row, col = divmod(cls, 13)
    if row == col:
        return RANKS[row] * 2
    if row < col:
        return RANKS[row] + RANKS[col] + "s"
    return RANKS[col] + RANKS[row] + "o"


def class_of_label(label):
    a, b = RANKS.index(label[0]), RANKS.index(label[1])
    if a == b:
        return a * 13 + a
    hi, lo = min(a, b), max(a, b)
    return hi * 13 + lo if label.endswith("s") else lo * 13 + hi


def class_of_cards(cards):
    """'Ah Jd' -> class index."""
    c1, c2 = cards.split()
    r1, r2 = RANKS.index(c1[0]), RANKS.index(c2[0])
    if r1 == r2:
        return r1 * 13 + r1
    hi, lo = min(r1, r2), max(r1, r2)
    return hi * 13 + lo if c1[1] == c2[1] else lo * 13 + hi


def class_combos(cls):
    row, col = divmod(cls, 13)
    if row == col:
        r = RANKS[row]
        return [(r + s1, r + s2) for i, s1 in enumerate(SUITS) for s2 in SUITS[i + 1:]]
    hi, lo = RANKS[min(row, col)], RANKS[max(row, col)]
    if row < col:
        return [(hi + s, lo + s) for s in SUITS]
    return [(hi + s1, lo + s2) for s1 in SUITS for s2 in SUITS if s1 != s2]


@lru_cache(maxsize=1)
def combo_counts():
    return np.array([len(class_combos(c)) for c in range(N_CLASSES)], dtype=float)


@lru_cache(maxsize=1)
def pair_weights():
    """W[a, b] = number of card-disjoint combo pairs between classes a and b."""
    combos = [class_combos(c) for c in range(N_CLASSES)]
    w = np.zeros((N_CLASSES, N_CLASSES))
    for a in range(N_CLASSES):
        for b in range(a, N_CLASSES):
            n = sum(1 for x in combos[a] for y in combos[b] if not set(x) & set(y))
            w[a, b] = w[b, a] = n
    return w


@lru_cache(maxsize=1)
def equity_matrix():
    """E[a, b] = all-in equity of class a vs class b."""
    return np.loadtxt(EQUITY_FILE)


@lru_cache(maxsize=1)
def tie_matrix():
    """T[a, b] = probability that class a and class b tie all-in (counted as half a win in E)."""
    return np.loadtxt(TIES_FILE)


def range_grid(freqs, fmt="{:>4}"):
    """Render a 169-vector of frequencies (0..1) as a 13x13 text grid."""
    lines = ["     " + "".join(fmt.format(r) for r in RANKS)]
    for row in range(13):
        cells = []
        for col in range(13):
            f = freqs[row * 13 + col]
            cells.append(fmt.format("##" if f >= 0.95 else ("--" if f <= 0.05 else f"{int(f * 100)}")))
        lines.append(f"  {RANKS[row]}  " + "".join(cells))
    return "\n".join(lines)


def range_percent(freqs):
    cc = combo_counts()
    return float(np.dot(freqs, cc) / cc.sum() * 100)
