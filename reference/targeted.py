"""Targeted Monte Carlo: exact EVs of one decision of a push/fold strategy profile.

The reference solver samples deals uniformly, so a decision reached in a small share of hands gets few
of them. Here every deal is drawn given that the decision is reached: the deciding player's hand is
fixed to one class at a time, the players who acted before get hands from their ranges (the shove or
call range of the ones all-in, the rest of the hands for the ones who folded), the players behind and
the board come from the rest of the deck. All cards come from one deal, so card removal is exact,
folded players included; pots and bounties are split between tied hands (reference.solver.outcome).

It evaluates given strategies, it does not solve: for every hand class of the deciding player it gives
the EV of both actions when everybody plays the profile, to check the EVs a solver reports for it.

The game tree is the main solver's: players shove or fold until somebody shoves, the players behind
call or fold, and at most three players get all-in (whoever acts after the second caller folds), or
two without overcalls.
"""

import multiprocessing
from dataclasses import dataclass
from itertools import product
from math import comb

import numpy as np

from reference.cards import evaluate7, hand_class
from reference.solver import N_CLASSES, outcome

COMBOS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)])
COMBO_CLASS = hand_class(COMBOS[:, 0], COMBOS[:, 1]).astype(np.intp)
CHUNK = 5_000  # deals drawn at once (small enough for the CPU cache)
MAX_TRIES = 1000  # redraws of a hand that collides with the cards already dealt


@dataclass
class Profile:
    """Play probability per hand class: push[i] when folded to i, call[i, j] for j facing i's shove,
    over[i, j, k] for k facing i's shove and j's call (empty without overcalls)."""
    push: dict
    call: dict
    over: dict


def node_ranges(profile, player, shover, callers):
    """[(x, weights per class)] for the players who act before `player` at the node."""
    out = []
    for x in range(player):
        if shover < 0 or x < shover:
            w = 1 - profile.push[x]
        elif x == shover:
            w = profile.push[x]
        else:
            before = [c for c in callers if c < x]
            p = profile.call[shover, x] if not before else profile.over[shover, before[0], x]
            w = p if x in callers else 1 - p
        out.append((x, np.asarray(w, dtype=float)))
    return out


def _value_table(spot, allin):
    """values[code, player] for the players in `allin` all-in, code = sum of rank * len(allin)**index."""
    k, m = len(allin), spot.players
    out = np.zeros((k ** k, m))
    for r in product(range(k), repeat=k):
        rank = [0] * m
        for x, rx in zip(allin, r):
            rank[x] = rx
        out[sum(rx * k ** t for t, rx in enumerate(r))] = outcome(spot, list(allin), rank)
    return out


def _deal(rng, m, player, c, ranges, n):
    """n deals given the node: (hole cards (n, m, 2), board (n, 5), weights (n,)).

    The players before `player` are dealt one by one, each from his range among the hands that do not
    collide with the cards already dealt. That is sequential importance sampling: the weight
    prod over them of Z / C(cards left, 2), Z = the range's weight left after the cards dealt, makes
    the mean weight the probability of reaching the node with this hand and the weighted deals exact.
    """
    rows = np.arange(n)
    holes = np.zeros((n, m, 2), dtype=np.int64)
    holes[:, player] = COMBOS[rng.choice(np.flatnonzero(COMBO_CLASS == c), n)]
    used = np.zeros((n, 52), dtype=bool)
    used[rows, holes[:, player, 0]] = used[rows, holes[:, player, 1]] = True
    dealt = [holes[:, player, 0], holes[:, player, 1]]
    weight = np.ones(n)
    for x, w in ranges:
        wc = w[COMBO_CLASS]
        total = wc.sum()
        if total <= 0:
            return holes, np.zeros((n, 5), dtype=np.int64), np.zeros(n)
        pair = np.zeros((52, 52))
        pair[COMBOS[:, 0], COMBOS[:, 1]] = wc
        pair += pair.T
        u = np.stack(dealt, axis=1)
        left = total - pair.sum(axis=1)[u].sum(axis=1)
        for a in range(u.shape[1]):  # combos made of two dealt cards were subtracted twice
            for b in range(a + 1, u.shape[1]):
                left += pair[u[:, a], u[:, b]]
        left = np.maximum(left, 0)
        weight *= left / comb(52 - len(dealt), 2)
        prob = wc / total
        pick = rng.choice(len(COMBOS), n, p=prob)
        bad = used[rows, COMBOS[pick, 0]] | used[rows, COMBOS[pick, 1]]
        for _ in range(MAX_TRIES):
            idx = np.flatnonzero(bad & (left > 0))
            if not len(idx):
                break
            pick[idx] = rng.choice(len(COMBOS), len(idx), p=prob)
            bad[idx] = used[idx, COMBOS[pick[idx], 0]] | used[idx, COMBOS[pick[idx], 1]]
        weight[bad] = 0
        holes[:, x] = COMBOS[pick]
        used[rows, holes[:, x, 0]] = used[rows, holes[:, x, 1]] = True
        dealt += [holes[:, x, 0], holes[:, x, 1]]
    behind = list(range(player + 1, m))
    need = 2 * len(behind) + 5
    keys = np.random.default_rng(rng.integers(1 << 63)).random((n, 52))
    keys[used] = 2
    rest = keys.argpartition(need, axis=1)[:, :need]
    for t, x in enumerate(behind):
        holes[:, x] = rest[:, 2 * t:2 * t + 2]
    return holes, rest[:, need - 5:], weight


_worker = {}


def _init_worker(spot, profile):
    _worker.update(spot=spot, profile=profile, tables={})


def _chunk_values(node, c, n, rng):
    """Weighted sums over n deals for the deciding hand class c."""
    spot, profile, tables = _worker["spot"], _worker["profile"], _worker["tables"]
    m = spot.players
    player, shover, callers = node
    max_callers = 2 if profile.over else 1
    holes, board, weight = _deal(rng, m, player, c, node_ranges(profile, player, shover, callers), n)
    cls = hand_class(holes[:, :, 0], holes[:, :, 1]).astype(np.intp)
    # only the shover, the callers, the player and the players behind can get all-in
    can_allin = ({shover} if shover >= 0 else set()) | set(callers) | set(range(player, m))
    strength = np.full((n, m), -1, dtype=np.int64)
    for x in can_allin:
        strength[:, x] = evaluate7(np.concatenate([holes[:, x], board], axis=1))
    codes = {}

    def value(allin):
        if allin not in tables:
            tables[allin] = _value_table(spot, allin)
        if allin not in codes:
            s = strength[:, list(allin)]
            rank = (s[:, None, :] > s[:, :, None]).sum(axis=2)
            codes[allin] = rank @ (len(allin) ** np.arange(len(allin)))
        return tables[allin][codes[allin], player]

    def first_in(x):
        if x == m - 1:
            return value((x,))
        s = profile.push[x][cls[:, x]]
        return s * after_shove(x, (), x + 1) + (1 - s) * first_in(x + 1)

    def after_shove(sh, called, x):
        if x == m or len(called) == max_callers:
            return value((sh, *called))
        q = (profile.call[sh, x] if not called else profile.over[sh, called[0], x])[cls[:, x]]
        return q * after_shove(sh, called + (x,), x + 1) + (1 - q) * after_shove(sh, called, x + 1)

    if shover < 0:
        v_play, v_fold = after_shove(player, (), player + 1), first_in(player + 1)
    else:
        v_play, v_fold = after_shove(shover, callers + (player,), player + 1), after_shove(shover, callers, player + 1)
    g = v_play - v_fold
    w2 = weight ** 2
    return np.array([n, weight.sum(), w2.sum(), weight @ v_play, weight @ v_fold, w2 @ g, w2 @ g ** 2])


def _job(args):
    node, c, n, seed = args
    rng = np.random.default_rng(seed)
    return sum(_chunk_values(node, c, min(CHUNK, n - start), rng) for start in range(0, n, CHUNK))


class TargetedCheck:
    """Exact EVs of the decisions of `profile` (a Profile) in `spot` (a reference.solver.Spot)."""

    def __init__(self, spot, profile, workers=1):
        self.pool = multiprocessing.get_context("fork").Pool(
            workers, initializer=_init_worker, initargs=(spot, profile)) if workers > 1 else None
        if self.pool is None:
            _init_worker(spot, profile)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, node, samples, seed=0):
        """node = (player, shover or -1, callers); samples = deals per hand class.

        Returns per hand class: play and fold EV, the standard error of their difference, and reach,
        the probability that the node is reached when the player holds a hand of that class.
        """
        player, shover, callers = node
        tag = [seed, player, shover + 1, *[c + 1 for c in callers]]
        jobs = [((player, shover, tuple(callers)), c, samples, [*tag, c]) for c in range(N_CLASSES)]
        parts = self.pool.map(_job, jobs, chunksize=1) if self.pool else [_job(job) for job in jobs]
        n, sw, sw2, svp, svf, sw2g, sw2g2 = np.array(parts).T
        with np.errstate(invalid="ignore", divide="ignore"):
            play, fold = svp / sw, svf / sw
            g = play - fold
            se = np.sqrt(np.maximum(sw2g2 - 2 * g * sw2g + g ** 2 * sw2, 0)) / sw
        return {"play": play, "fold": fold, "se": se, "reach": sw / n}
