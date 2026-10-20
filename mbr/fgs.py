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
        button = m - 3 if m >= 3 else 0
        # the players left, clockwise from the seat after the button: the next button, SB, BB, ...
        seats = [x for x in ((button + k) % m for k in range(1, m + 1)) if stacks[x] > 0]
        n = len(seats)
        order = seats[3:] + seats[:3] if n >= 3 else seats
        posted = [self.ante] * n
        posted[-2] += self.sb
        posted[-1] += self.bb
        forced = [k for k, x in enumerate(order) if stacks[x] <= posted[k]]
        if len(forced) > 1:
            return None
        if forced:
            # he posts everything and is the "shover" every other player can call, in their order
            (k,) = forced
            posted[k] = stacks[order[k]]
            order = [order[k]] + order[:k] + order[k + 1:]
            posted = [posted[k]] + posted[:k] + posted[k + 1:]
        table = Table(names=positions(n), stacks=[stacks[x] for x in order], posted=posted, bb=self.bb,
                      payouts=self.payouts[:n], bounty=self.bounty, future_bounty=self.future_bounty,
                      bunching=False)
        return table, order, bool(forced)

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


def table_for(stacks, sb, bb, ante, model):
    """The table of a model: stacks before posting in preflop order (first to act ... big blind)."""
    spec = dict(MODELS[model])
    n = len(stacks)
    posted = [ante] * n
    posted[-1] += bb
    posted[-2] += sb
    if spec.pop("fgs", False):
        return FgsTable(names=positions(n), stacks=list(stacks), posted=posted, bb=bb, sb=sb, ante=ante, **spec)
    return Table(names=positions(n), stacks=list(stacks), posted=posted, bb=bb, **spec)
