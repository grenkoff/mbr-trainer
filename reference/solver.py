"""Reference push/fold solver: slow, exact in the model, independent of mbr.pushfold.

The game is the push/fold final-table hand of the main solver, without its shortcuts:
- every player behind the shover may call, so any number of players can be all-in;
- every pot layer is split between tied hands, the bounty of a busted player too;
- the cards of all players, folded ones included, come from one deal, so card removal is exact;
- deals are a large random sample of complete deals (reference.cards): the only approximation, and
  its size shows by solving on one sample and scoring on another.

Values are in buy-ins: Malmuth-Harville ICM on the place prizes, plus the bounties won in the hand,
plus (optionally) the bounties still to be won, split in proportion to chips.

The equilibrium is found with CFR+ (regret matching with floored regrets and weighted averaging) on
the sampled game, vectorized over the deals.
"""

import multiprocessing
from dataclasses import dataclass, field
from itertools import permutations, product

import numpy as np

from reference.cards import load_deals

N_CLASSES = 169
CHUNK = 250_000  # deals walked at once, by one worker


@dataclass
class Spot:
    stacks: list  # chips before posting, in preflop order (first to act ... big blind)
    posted: list  # antes + blinds of each player
    payouts: list = field(default_factory=lambda: [4.0, 3.0, 2.0])
    bounty: float = 0.945
    future_bounty: bool = True
    chip_value: float = 0.0  # if set, a cash game: every chip is worth this much (no ICM, no bounties)

    @property
    def players(self):
        return len(self.stacks)


# --- values -------------------------------------------------------------------
def icm_prizes(chips, payouts):
    """Malmuth-Harville: every finishing order of the top places, by explicit enumeration."""
    alive = [x for x in range(len(chips)) if chips[x] > 0]
    places = min(len(payouts), len(alive))
    total = sum(chips[x] for x in alive)
    out = [0.0] * len(chips)
    for order in permutations(alive, places):
        p, left = 1.0, total
        for x in order:
            p *= chips[x] / left
            left -= chips[x]
        for place, x in enumerate(order):
            out[x] += p * payouts[place]
    return out


def outcome(spot, allin, rank):
    """Value of every player when the players in `allin` are all-in and the rest folded.

    allin: player indices (one player means everybody else folded to him); rank[x] = number of hands
    that beat x's at showdown. Returns a list of values in buy-ins.
    """
    m = spot.players
    stacks, posted = spot.stacks, spot.posted
    after = [stacks[x] - posted[x] for x in range(m)]
    won_bounty = [0.0] * m
    if len(allin) == 1:
        (w,) = allin
        after[w] = stacks[w] + sum(posted[x] for x in range(m) if x != w)
        busted = []
    else:
        put = {x: min(stacks[x], max(stacks[y] for y in allin if y != x)) for x in allin}
        for x in allin:
            after[x] = stacks[x] - put[x]
        dead = sum(posted[x] for x in range(m) if x not in allin)
        layer_winners = {}
        prev = 0
        for level in sorted(set(put.values())):
            eligible = [x for x in allin if put[x] >= level]
            best = min(rank[x] for x in eligible)
            winners = [x for x in eligible if rank[x] == best]
            pot = (level - prev) * len(eligible) + (dead if prev == 0 else 0)
            for x in winners:
                after[x] += pot / len(winners)
            layer_winners[level] = winners
            prev = level
        busted = [x for x in allin if after[x] <= 1e-9]
        for x in busted:  # the bounty goes to whoever takes the busted player's last chips
            winners = layer_winners[put[x]]
            for w in winners:
                won_bounty[w] += spot.bounty / len(winners)
    if spot.chip_value:
        return [a * spot.chip_value for a in after]
    value = icm_prizes([max(a, 0.0) for a in after], spot.payouts)
    if busted:
        # players busted in this hand take the places below the survivors, the bigger starting stack
        # higher; equal stacks share their places
        place = m - len(busted)  # 0-based index of the best place left
        prize = lambda k: spot.payouts[k] if k < len(spot.payouts) else 0.0
        for s in sorted({stacks[x] for x in busted}, reverse=True):
            same = [x for x in busted if stacks[x] == s]
            share = sum(prize(place + k) for k in range(len(same))) / len(same)
            for x in same:
                value[x] = share
            place += len(same)
    alive = [x for x in range(m) if after[x] > 1e-9]
    total = sum(after[x] for x in alive)
    for x in range(m):
        value[x] += won_bounty[x]
        if spot.future_bounty and x in alive:
            value[x] += (len(alive) - 1) * spot.bounty * after[x] / total
    return value


# --- game tree ----------------------------------------------------------------
@dataclass
class Decision:
    player: int
    shover: int  # -1 when nobody has shoved yet (the player opens or folds)
    callers: tuple  # players who called the shove so far
    play: object = None  # child after shove / call
    fold: object = None
    index: int = -1


@dataclass
class Terminal:
    allin: tuple  # players who put their chips in (one = everybody folded to him)
    index: int = -1


def build_tree(m):
    decisions, terminals = [], []

    def add(node):
        target = decisions if isinstance(node, Decision) else terminals
        node.index = len(target)
        target.append(node)
        return node

    def first_in(p):
        if p == m - 1:
            return add(Terminal((p,)))
        node = add(Decision(p, -1, ()))
        node.play = after_shove(p, (), p + 1)
        node.fold = first_in(p + 1)
        return node

    def after_shove(shover, callers, p):
        if p == m:
            return add(Terminal((shover, *callers)))
        node = add(Decision(p, shover, callers))
        node.play = after_shove(shover, callers + (p,), p + 1)
        node.fold = after_shove(shover, callers, p + 1)
        return node

    return first_in(0), decisions, terminals


def payoff_tables(spot, terminals):
    """values[t, code, x] for every terminal t and every showdown order code (see reference.cards)."""
    m = spot.players
    codes = list(product(range(m), repeat=m))  # rank tuples; codes that never occur are harmless
    out = np.zeros((len(terminals), m ** m, m), dtype=np.float32)
    for t in terminals:
        cache = {}
        for rank in codes:
            code = sum(r * m ** x for x, r in enumerate(rank))
            key = tuple(rank[x] for x in t.allin) if len(t.allin) > 1 else ()
            if key not in cache:
                cache[key] = outcome(spot, list(t.allin), rank)
            out[t.index, code] = cache[key]
    return out


# --- solving ------------------------------------------------------------------
def _walk(root, values, play, classes, codes, nd):
    """Counterfactual value sums over a block of deals: (cfv[decision, play/fold, class], reach[decision, class])."""
    cfv = np.zeros((nd, 2, N_CLASSES))
    reach_sum = np.zeros((nd, N_CLASSES))
    play = play.astype(np.float32)
    cls = np.asarray(classes, dtype=np.intp)
    code = np.asarray(codes, dtype=np.intp)

    def walk(node, reach):
        # every action of every decision is walked for every deal: CFR needs the value of the
        # actions the current strategy does not take too
        if isinstance(node, Terminal):
            return values[node.index][code]
        p, d = node.player, node.index
        c = cls[:, p]
        s = play[d][c]
        v_play = walk(node.play, reach * s)
        v_fold = walk(node.fold, reach * (1 - s))
        # the player acts once per hand, so the reach of the others is the whole reach
        cfv[d, 0] += np.bincount(c, weights=reach * v_play[:, p], minlength=N_CLASSES)
        cfv[d, 1] += np.bincount(c, weights=reach * v_fold[:, p], minlength=N_CLASSES)
        reach_sum[d] += np.bincount(c, weights=reach, minlength=N_CLASSES)
        return s[:, None] * v_play + (1 - s)[:, None] * v_fold

    walk(root, np.ones(len(code), dtype=np.float32))
    return cfv, reach_sum


_worker = {}


def _init_worker(players, values):
    root, decisions, _ = build_tree(players)
    _worker.update(root=root, nd=len(decisions), values=values, deals={})


def _walk_block(args):
    path, start, end, play = args
    if path not in _worker["deals"]:
        _worker["deals"][path] = load_deals(path)
    classes, codes = _worker["deals"][path]
    return _walk(_worker["root"], _worker["values"], play, classes[start:end], codes[start:end], _worker["nd"])


class ReferenceSolver:
    """CFR+ on the deals in `deals` (a reference.cards.deals directory), in `workers` processes."""

    def __init__(self, spot, deals, workers=1):
        self.spot = spot
        self.m = spot.players
        self.root, self.decisions, self.terminals = build_tree(self.m)
        self.values = payoff_tables(spot, self.terminals)
        self.deals = deals
        nd = len(self.decisions)
        self.regret = np.zeros((nd, 2, N_CLASSES))  # [decision, play/fold, hand class]
        self.average = np.zeros((nd, N_CLASSES))
        self.average_weight = 0.0
        self.iterations = 0
        self.pool = multiprocessing.get_context("fork").Pool(
            workers, initializer=_init_worker, initargs=(self.m, self.values)) if workers > 1 else None
        if self.pool is None:
            _init_worker(self.m, self.values)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def current(self):
        """Play probabilities from the regrets (regret matching; 1/2 when there is no regret yet)."""
        pos = np.maximum(self.regret, 0)
        tot = pos.sum(axis=1)
        return np.divide(pos[:, 0], tot, out=np.full_like(tot, 0.5), where=tot > 0)

    def strategy(self):
        """The averaged strategies: play probability per decision and hand class."""
        if self.average_weight == 0:
            return self.current()
        return self.average / self.average_weight

    def action_values(self, play, deals=None, blocks=False):
        """Counterfactual values of both actions at every decision, per hand class, per deal.

        Returns (cfv[decision, play/fold, class], reach[decision, class]): cfv / reach is the EV of the
        action for a hand of that class when the decision is reached. With blocks, a list of them, one per
        block of CHUNK deals, to estimate the sampling error.
        """
        path = str(self.deals if deals is None else deals)
        n = len(load_deals(path)[1])
        jobs = [(path, start, min(start + CHUNK, n), play) for start in range(0, n, CHUNK)]
        parts = self.pool.map(_walk_block, jobs) if self.pool else [_walk_block(job) for job in jobs]
        if blocks:
            return [(cfv / (job[2] - job[1]), reach / (job[2] - job[1])) for job, (cfv, reach) in zip(jobs, parts)]
        return sum(p[0] for p in parts) / n, sum(p[1] for p in parts) / n

    def iterate(self, iterations, averaging=2):
        """CFR+ iterations; iteration t enters the average with weight t ** averaging."""
        for _ in range(iterations):
            self.iterations += 1
            play = self.current()
            cfv, _ = self.action_values(play)
            node = play * cfv[:, 0] + (1 - play) * cfv[:, 1]
            self.regret = np.maximum(self.regret + cfv - node[:, None], 0)
            w = self.iterations ** averaging
            self.average += w * play
            self.average_weight += w
        return self

    def exploitability(self, play=None, deals=None):
        """Per player: how much a best response to `play` (default: the averaged strategies) gains per hand.

        On deals other than the solving ones it is biased upwards by the sampling noise."""
        play = self.strategy() if play is None else play
        cfv, _ = self.action_values(play, deals)
        node = play * cfv[:, 0] + (1 - play) * cfv[:, 1]
        gain = np.zeros(self.m)
        for d in self.decisions:
            gain[d.player] += (cfv[d.index].max(axis=0) - node[d.index]).sum()
        return gain

    def find(self, player, shover, callers):
        return next(d for d in self.decisions
                    if d.player == player and d.shover == shover and d.callers == tuple(callers))
