"""169 starting-hand classes on the standard 13x13 grid and combo bookkeeping."""

from functools import lru_cache
from pathlib import Path

import numpy as np

RANKS = "AKQJT98765432"  # grid order: index 0 = Ace
SUITS = "cdhs"
N_CLASSES = 169
EQUITY_FILE = Path(__file__).resolve().parent.parent / "data" / "equity_169.txt"
TIES_FILE = Path(__file__).resolve().parent.parent / "data" / "ties_169.txt"
DEAD_FILE = Path(__file__).resolve().parent.parent / "data" / "dead_169.f32"


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


@lru_cache(maxsize=1)
def dead_card_effects():
    """(DE, DT), each [a, b, r]: how much one dead card of rank r (grid order, 0 = Ace) changes the
    equity and the tie probability of class a vs class b, built by csrc/dead.c. With dead cards from
    folded hands, E[a, b] + DE[a, b] @ nu is about the equity, nu[r] = expected dead cards of rank r."""
    d = np.fromfile(DEAD_FILE, dtype=np.float32).reshape(2, N_CLASSES, N_CLASSES, 13).astype(float)
    return d[0], d[1]


@lru_cache(maxsize=1)
def dead_card_strength():
    """H[c, r]: a dead card of rank r moves the equity of class a vs class b by about H[a, r] - H[b, r]
    (least squares over DE weighted by the combo pairs, about 89% of it), for 3-way all-ins."""
    de = dead_card_effects()[0]
    w = pair_weights().copy()
    np.fill_diagonal(w, 0)
    laplacian = np.diag(w.sum(axis=1)) - w
    h = np.linalg.lstsq(laplacian, np.einsum("ab,abr->ar", w, de), rcond=None)[0]
    return h - h.mean(axis=0)


@lru_cache(maxsize=1)
def rank_counts():
    """K[c, r] = number of cards of rank r (grid order) in a hand of class c."""
    k = np.zeros((N_CLASSES, 13))
    for c in range(N_CLASSES):
        for card in class_combos(c)[0]:
            k[c, RANKS.index(card[0])] += 1
    return k


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
