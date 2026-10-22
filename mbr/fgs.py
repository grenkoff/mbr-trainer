"""FGS (future game simulation) of depth 1, the way HoldemResources Calculator (HRC) values outcomes.

ICM values the stacks right after the hand. FGS plays one more hand first: the button moves on, the
blinds and antes of the same level are posted, everybody plays the push/fold equilibrium of that hand
with ICM (and the bounties) after it, and each player's value at the start of that hand is what the
outcome of the current hand is worth. As in HRC:
- outcomes where tied hands split the pot are valued with ICM;
- the hand ends the tournament (one player left): ICM too;
- a player left with no more than his antes and blinds is all-in from posting in the next hand: he
  always plays, the others call him or fold (two such players: ICM);
- bounties won in the current hand are added; those won later are in the next hand's values.

`table_for(...)` builds the table of a model, `MODELS` lists them, "hrc" being HRC's push/fold with FGS:
it also leaves the folded players' cards out, as HRC's (non Monte Carlo) push/fold does.
"""

from dataclasses import dataclass, field

import numpy as np

from mbr.icm import icm
from mbr.pushfold import MEAN_BOUNTY_BI, Solver, Table

POSITIONS = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]

MODELS = {
    "hrc": dict(bounty=MEAN_BOUNTY_BI, future_bounty=True, fgs=True),
    "bounty": dict(bounty=MEAN_BOUNTY_BI, future_bounty=True),
    "bounty-now": dict(bounty=MEAN_BOUNTY_BI, future_bounty=False),
    "icm": dict(bounty=0.0, future_bounty=False),
}


def positions(n):
    return POSITIONS[-n:]


@dataclass
class FgsTable(Table):
    sb: int = 0
    ante: int = 0
    overcalls: bool = True  # the next hand is solved with overcalls (at most three players all-in)
    bunching: bool = False
    _next: dict = field(default_factory=dict, repr=False)  # stacks after the hand -> next hand's values

    def values(self, stacks_after, ko_winners=(), dead_order=None, split=False):
        alive = [x for x, s in enumerate(stacks_after) if s > 0]
        nxt = None if split or len(alive) < 2 else self.next_hand_values(tuple(stacks_after))
        if nxt is None:
            return super().values(stacks_after, ko_winners, dead_order)
        v = np.array(icm(stacks_after, self.payouts, dead_order))  # the busted players' places
        v[alive] = nxt[alive]
        for w, share in ko_winners:
            v[w] += share * self.bounty
        return v

    def next_hand(self, stacks):
        """(Table, seats in its action order, whether the first one is all-in from posting) of the next
        hand with these stacks, or None if two players would be all-in from posting."""
        m = self.m
        # this table's seats in preflop order: a player all-in from posting may have been put first
        by_seat = [self.names.index(p) for p in positions(m)]
        button = m - 3 if m >= 3 else 0
        # the players left, clockwise from the seat after the button: the next button, SB, BB, ...
        seats = [x for x in (by_seat[(button + k) % m] for k in range(1, m + 1)) if stacks[x] > 0]
        n = len(seats)
        preflop = seats[3:] + seats[:3] if n >= 3 else seats
        arranged = action_order([stacks[x] for x in preflop], self.sb, self.bb, self.ante)
        if arranged is None:
            return None
        order, posted, forced = arranged
        table = Table(names=[positions(n)[k] for k in order], stacks=[stacks[preflop[k]] for k in order],
                      posted=posted, bb=self.bb, payouts=self.payouts[:n], bounty=self.bounty,
                      future_bounty=self.future_bounty, bunching=False)
        return table, [preflop[k] for k in order], forced

    def next_hand_values(self, stacks):
        """Each player's value at the start of the next hand (0 for the busted ones), or None."""
        if stacks not in self._next:
            hand = self.next_hand(stacks)
            out = None
            if hand is not None:
                table, order, forced = hand
                solver = Solver(table, overcalls=self.overcalls and table.m >= 3, forced_open=forced).solve()
                out = np.zeros(self.m)
                out[order] = solver.state_values()
            self._next[stacks] = out
        return self._next[stacks]


def action_order(stacks, sb, bb, ante):
    """(players in the solver's order, their posted chips in that order, whether the first one is all-in
    from posting) for stacks before posting in preflop order, or None if two players are all-in from
    posting. Such a player posts everything and is the "shover" every other player can call, in their
    order (mbr.pushfold.Solver with forced_open)."""
    n = len(stacks)
    posted = [ante] * n
    posted[-2] += sb
    posted[-1] += bb
    forced = [k for k in range(n) if stacks[k] <= posted[k]]
    if len(forced) > 1:
        return None
    order = list(range(n))
    if forced:
        (k,) = forced
        posted[k] = stacks[k]
        order = [k] + order[:k] + order[k + 1:]
    return order, [posted[k] for k in order], bool(forced)


def table_for(stacks, sb, bb, ante, model, overcalls=True):
    """The table of a model: stacks before posting in preflop order (first to act ... big blind). A player
    all-in from posting is put first (see action_order); the table's names are the positions. overcalls:
    whether FGS solves the next hand with them."""
    spec = dict(MODELS[model])
    arranged = action_order(stacks, sb, bb, ante)
    if arranged is None:
        raise ValueError("two players are all-in from posting")
    order, posted, _ = arranged
    names = [positions(len(stacks))[k] for k in order]
    stacks = [stacks[k] for k in order]
    if spec.pop("fgs", False):
        return FgsTable(names=names, stacks=stacks, posted=posted, bb=bb, sb=sb, ante=ante, overcalls=overcalls,
                        **spec)
    return Table(names=names, stacks=stacks, posted=posted, bb=bb, **spec)


def forced_open(table):
    """Whether the table's first player is all-in from posting (table_for put him first)."""
    return table.posted[0] >= table.stacks[0]


def preflop_seats(table):
    """For each player of a table from table_for, his seat in preflop order."""
    return [positions(table.m).index(p) for p in table.names]
