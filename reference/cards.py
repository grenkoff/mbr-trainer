"""Cards for the reference solver: a vectorized 7-card evaluator and random deals.

Written independently of csrc/ and mbr/: nothing here is shared with the main solver.
A card is rank * 4 + suit with rank 0 = deuce ... 12 = ace.
"""

import multiprocessing
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "build" / "reference"

# category of a 7-card hand, from the worst to the best
CATEGORIES = ["high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads",
              "straight flush"]


def _bit_tables():
    masks = np.arange(1 << 13)
    high = np.full(1 << 13, -1)
    for r in range(13):
        high[(masks >> r) & 1 == 1] = r
    straight = np.full(1 << 13, -1)
    for top in range(3, 13):  # top card of the straight: 3 = five (the wheel) ... 12 = ace
        need = sum(1 << r for r in range(top - 4, top + 1)) if top >= 4 else 0b1111 | (1 << 12)
        straight[(masks & need) == need] = top
    popcount = np.array([bin(x).count("1") for x in range(1 << 13)])
    tops = {}
    for k in (1, 2, 3, 5):  # the mask cut down to its k highest ranks
        t = masks.copy()
        for _ in range(13):
            t = np.where(popcount[t] > k, t & (t - 1), t)  # drop the lowest rank while there are too many
        tops[k] = t
    return high, straight, tops


HIGH, STRAIGHT, TOP = _bit_tables()


def _bit(r):
    return np.where(r >= 0, np.left_shift(1, np.maximum(r, 0)), 0)


def evaluate7(cards):
    """Strength of each 7-card hand (cards: int array (k, 7)); a larger number is a better hand.

    The number is category << 21 | (first rank + 1) << 17 | (second rank + 1) << 13 | kicker ranks mask.
    """
    cards = np.asarray(cards)
    k = len(cards)
    ranks, suits = cards // 4, cards % 4
    count = (ranks[:, :, None] == np.arange(13)).sum(axis=1)
    suit_count = (suits[:, :, None] == np.arange(4)).sum(axis=1)
    weights = 1 << np.arange(13)
    present = (count > 0) @ weights
    pairs = (count == 2) @ weights
    trips = (count == 3) @ weights
    quads = (count == 4) @ weights
    flush_suit = suit_count.argmax(axis=1)
    has_flush = suit_count.max(axis=1) >= 5
    in_flush = suits == flush_suit[:, None]
    flush = np.bitwise_or.reduce(np.where(in_flush, 1 << ranks, 0), axis=1)

    cat = np.zeros(k, dtype=np.int64)
    p1 = np.full(k, -1)
    p2 = np.full(k, -1)
    kick = TOP[5][present]

    def put(mask, c, a=None, b=None, kk=None):
        cat[mask] = c
        p1[mask] = -1 if a is None else a[mask]
        p2[mask] = -1 if b is None else b[mask]
        kick[mask] = 0 if kk is None else kk[mask]

    hp = HIGH[pairs]
    lp = HIGH[pairs & ~_bit(hp)]
    put(pairs > 0, 1, hp, None, TOP[3][present & ~_bit(hp)])
    put(lp >= 0, 2, hp, lp, TOP[1][present & ~_bit(hp) & ~_bit(lp)])
    t = HIGH[trips]
    put(t >= 0, 3, t, None, TOP[2][present & ~_bit(t)])
    st = STRAIGHT[present]
    put(st >= 0, 4, st)
    put(has_flush, 5, None, None, TOP[5][flush])
    full_pair = HIGH[(trips & ~_bit(t)) | pairs]
    put((t >= 0) & (full_pair >= 0), 6, t, full_pair)
    q = HIGH[quads]
    put(q >= 0, 7, q, None, TOP[1][present & ~_bit(q)])
    sf = np.where(has_flush, STRAIGHT[flush], -1)
    put(sf >= 0, 8, sf)
    return cat << 21 | (p1 + 1) << 17 | (p2 + 1) << 13 | kick


def hand_class(c1, c2):
    """Index of the 13x13 grid (row and column 0 = ace; suited above the diagonal, offsuit below)."""
    g1, g2 = 12 - c1 // 4, 12 - c2 // 4
    hi, lo = np.minimum(g1, g2), np.maximum(g1, g2)
    suited = c1 % 4 == c2 % 4
    return np.where(hi == lo, hi * 14, np.where(suited, hi * 13 + lo, lo * 13 + hi)).astype(np.uint8)


def _sample_chunk(args):
    players, k, seed = args
    rng = np.random.default_rng(seed)
    need = 2 * players + 5
    cards = rng.random((k, 52)).argpartition(need, axis=1)[:, :need]
    board = cards[:, 2 * players:]
    classes = np.empty((k, players), dtype=np.uint8)
    strength = np.empty((k, players), dtype=np.int64)
    for x in range(players):
        hole = cards[:, 2 * x:2 * x + 2]
        classes[:, x] = hand_class(hole[:, 0], hole[:, 1])
        strength[:, x] = evaluate7(np.concatenate([hole, board], axis=1))
    rank = (strength[:, None, :] > strength[:, :, None]).sum(axis=2)  # rank[d, x] = players beating x
    return classes, (rank @ (players ** np.arange(players))).astype(np.uint16)


def sample_deals(players, n, seed, workers=1, chunk=250_000):
    """n random deals: each player's hand class (n, players) and the showdown order code (n,).

    The code is sum over players x of rank_x * players**x, where rank_x is the number of players whose
    hand beats x's on the five board cards (ties share a rank).
    """
    jobs = [(players, min(chunk, n - start), (seed, start)) for start in range(0, n, chunk)]
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_sample_chunk, jobs)
    else:
        parts = [_sample_chunk(job) for job in jobs]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def deals(players, n, seed, workers=1):
    """Directory with a cached sample_deals (classes.npy, codes.npy) in build/reference/."""
    path = CACHE_DIR / f"deals_{players}p_{n}_{seed}"
    if not (path / "codes.npy").exists():
        classes, codes = sample_deals(players, n, seed, workers)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "classes.npy", classes)
        np.save(path / "codes.npy", codes)
    return path


def load_deals(path):
    """(classes, codes) of a deals directory, memory-mapped."""
    return np.load(Path(path) / "classes.npy", mmap_mode="r"), np.load(Path(path) / "codes.npy", mmap_mode="r")
