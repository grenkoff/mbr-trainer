"""Push/fold equilibrium for Mystery Battle Royale final tables (ICM + mystery bounties).

Game model:
- players act in preflop order; the first player to act may shove or fold (no limps, no non-all-in raises);
- after a shove the remaining players, in order, may call or fold; the first caller goes to showdown with
  the shover and everybody else folds (no overcalls);
- if everybody folds, the big blind wins the blinds and antes.

Values are in buy-ins:
- ICM (Malmuth-Harville) on the place prizes;
- plus the mystery bounty for busting a player;
- plus, optionally, the future bounties still to be won at the table, split in proportion to chips.

Strategies are solved with fictitious play: every iteration each player best-responds to the averaged
strategies of the others, and the averages move towards the best responses.
"""

from dataclasses import dataclass, field

import numpy as np

from mbr.icm import icm
from mbr.ranges import N_CLASSES, combo_counts, equity_matrix, pair_weights

PAYOUTS_BI = [4.0, 3.0, 2.0]
MEAN_BOUNTY_BI = 0.945


@dataclass
class Table:
    names: list  # players in preflop action order (first to act ... big blind)
    stacks: list  # chips before posting antes and blinds
    posted: list  # antes + blinds posted by each player
    bb: int
    payouts: list = field(default_factory=lambda: list(PAYOUTS_BI))
    bounty: float = MEAN_BOUNTY_BI
    future_bounty: bool = True

    @property
    def m(self):
        return len(self.names)

    def values(self, stacks_after, ko_winner=None):
        v = np.array(icm(stacks_after, self.payouts))
        if ko_winner is not None:
            v[ko_winner] += self.bounty
        if self.future_bounty:
            alive = sum(1 for s in stacks_after if s > 0)
            total = sum(stacks_after)
            v += (alive - 1) * self.bounty * np.array(stacks_after, dtype=float) / total
        return v

    def steal_values(self, i):
        s = [st - p for st, p in zip(self.stacks, self.posted)]
        s[i] += sum(self.posted)
        return self.values(s)

    def showdown_values(self, winner, loser):
        c = min(self.stacks[winner], self.stacks[loser])
        dead = sum(p for k, p in enumerate(self.posted) if k not in (winner, loser))
        s = [st - p for st, p in zip(self.stacks, self.posted)]
        s[winner] = self.stacks[winner] + c + dead
        s[loser] = self.stacks[loser] - c
        return self.values(s, ko_winner=winner if s[loser] == 0 else None)


class Solver:
    def __init__(self, table):
        self.t = table
        m = table.m
        self.W = pair_weights()
        self.WE = self.W * equity_matrix()
        self.Wsum = self.W.sum(axis=1)
        self.cc = combo_counts()
        # terminal values (vectors over players)
        self.v_steal = [table.steal_values(i) for i in range(m)]
        self.v_iwin = {}
        self.v_jwin = {}
        for i in range(m - 1):
            for j in range(i + 1, m):
                self.v_iwin[i, j] = table.showdown_values(i, j)
                self.v_jwin[i, j] = table.showdown_values(j, i)
        # strategies: push[i] for opener i (the big blind never opens), call[i, j] for j facing i's shove
        self.push = {i: np.full(N_CLASSES, 0.5) for i in range(m - 1)}
        self.call = {(i, j): np.full(N_CLASSES, 0.5) for i in range(m - 1) for j in range(i + 1, m)}

    # --- range statistics -------------------------------------------------
    def _caller_stats(self, i, j):
        """Call probability and shover win probability for the j-call node, conditioned on i's range."""
        p, c = self.push[i] + 1e-12, self.call[i, j]
        wc = self.W @ c
        cp = float(p @ wc / (p @ self.Wsum))
        denom = float(p @ wc)
        pwin = float(p @ (self.WE @ c) / denom) if denom > 1e-12 else 0.5
        return cp, pwin

    def _after_shove_value(self, i, from_j, stats):
        """Value vector after i shoves and players before from_j have folded."""
        acc = np.zeros(self.t.m)
        q = 1.0
        for k in range(from_j, self.t.m):
            cp, pwin = stats[i, k]
            acc += q * cp * (pwin * self.v_iwin[i, k] + (1 - pwin) * self.v_jwin[i, k])
            q *= 1 - cp
        return acc + q * self.v_steal[i]

    def evaluate(self):
        """Per-hand EVs of every decision against the current average strategies."""
        m = self.t.m
        stats = {(i, j): self._caller_stats(i, j) for (i, j) in self.call}
        node = [None] * m
        node[m - 1] = self.v_steal[m - 1]  # folded to the big blind: walk
        for i in range(m - 2, -1, -1):
            ppush = float(self.push[i] @ self.cc / self.cc.sum())
            node[i] = ppush * self._after_shove_value(i, i + 1, stats) + (1 - ppush) * node[i + 1]

        ev = {}
        for i in range(m - 1):
            # opener i with hand a
            ev_push = np.zeros(N_CLASSES)
            q = np.ones(N_CLASSES)
            for j in range(i + 1, m):
                c = self.call[i, j]
                wc = self.W @ c
                callp = wc / self.Wsum
                eqc = np.divide(self.WE @ c, wc, out=np.full(N_CLASSES, 0.5), where=wc > 1e-12)
                ev_push += q * callp * (eqc * self.v_iwin[i, j][i] + (1 - eqc) * self.v_jwin[i, j][i])
                q *= 1 - callp
            ev_push += q * self.v_steal[i][i]
            ev["push", i] = (ev_push, float(node[i + 1][i]))
            # callers j facing i's shove with hand b
            p = self.push[i] + 1e-12
            wp = self.W @ p
            eqp = (self.WE @ p) / wp  # equity of caller's hand b vs shover range
            for j in range(i + 1, m):
                ev_call = eqp * self.v_jwin[i, j][j] + (1 - eqp) * self.v_iwin[i, j][j]
                ev_fold = float(self._after_shove_value(i, j + 1, stats)[j])
                ev["call", i, j] = (ev_call, ev_fold)
        return ev

    def solve(self, iterations=300, tol=1e-4):
        for it in range(iterations):
            ev = self.evaluate()
            step = 1.0 / (it + 2)
            delta = 0.0
            for i in self.push:
                br = (ev["push", i][0] > ev["push", i][1]).astype(float)
                d = step * (br - self.push[i])
                self.push[i] += d
                delta = max(delta, np.abs(d).max())
            for key in self.call:
                br = (ev["call", *key][0] > ev["call", *key][1]).astype(float)
                d = step * (br - self.call[key])
                self.call[key] += d
                delta = max(delta, np.abs(d).max())
            if it > 50 and delta < tol:
                break
        self.ev = self.evaluate()
        return self


def table_from_hand(hand, bounty=MEAN_BOUNTY_BI, future_bounty=True):
    """Build a Table from a parsed final-table hand, or None if it does not fit the model."""
    stacks = hand.stacks
    sb_name = next((a.name for a in hand.actions if a.verb == "sb"), None)
    bb_name = next((a.name for a in hand.actions if a.verb == "bb"), None)
    if bb_name is None:
        return None
    posted = {n: 0 for n in stacks}
    for a in hand.actions:
        if a.verb in ("ante", "sb", "bb"):
            posted[a.name] += a.amount
    if any(posted[n] >= stacks[n] for n in stacks):
        return None  # someone is all-in from posting
    seats = sorted(hand.seats)
    start = next((k for k, s in enumerate(seats) if s > hand.button), 0)
    names = [hand.seats[s][0] for s in seats[start:] + seats[:start]]  # left of the button first
    b = names.index(bb_name)
    order = names[b + 1:] + names[:b + 1]  # preflop order ends with the big blind
    if sb_name and order[-2] != sb_name:
        return None
    return Table(
        names=order, stacks=[stacks[n] for n in order], posted=[posted[n] for n in order],
        bb=hand.bb, bounty=bounty, future_bounty=future_bounty,
    )
