"""Push/fold equilibrium for Mystery Battle Royale final tables (ICM + mystery bounties).

Game model:
- players act in preflop order; the first player to act may shove or fold (no limps, no non-all-in raises);
- after a shove the remaining players, in order, may call or fold;
- without overcalls the first caller goes to showdown with the shover and everybody else folds;
- with overcalls the players behind the first caller may overcall, up to three players all-in in total
  (everybody after the overcaller folds); side pots are paid by the 3-way outcome tensor (mbr.equity3);
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

from mbr import equity3
from mbr.icm import icm
from mbr.ranges import N_CLASSES, combo_counts, equity_matrix, pair_weights

PAYOUTS_BI = [4.0, 3.0, 2.0]
MEAN_BOUNTY_BI = 0.945
# fictitious play with overcalls starts from the solution without them
WARM_START_WEIGHT = 1
WARM_START_MIN_FREQ = 0.02
MULTIWAY_MIN_FREQ = 0.03
MULTIWAY_ITERATIONS = 40


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

    def values(self, stacks_after, ko_winners=(), dead_order=None):
        v = np.array(icm(stacks_after, self.payouts, dead_order))
        for w in ko_winners:
            v[w] += self.bounty
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
        return self.allin_values([winner, loser])

    def allin_values(self, ranking):
        """Values after an all-in between the players in ranking, ordered from the best hand to the worst.

        Each player risks up to the second largest stack among them; every pot layer goes to the best
        eligible hand, the dead money to the best hand overall. A busted player's bounty goes to the winner
        of the top layer they paid into; players busted together finish in order of their stacks.
        """
        st = self.stacks
        cap = sorted(st[x] for x in ranking)[-2]
        risk = {x: min(st[x], cap) for x in ranking}
        s = [x - p for x, p in zip(st, self.posted)]
        for x in ranking:
            s[x] = st[x] - risk[x]
        s[ranking[0]] += sum(p for k, p in enumerate(self.posted) if k not in ranking)
        prev = 0
        for level in sorted(set(risk.values())):
            eligible = [x for x in ranking if risk[x] >= level]
            s[eligible[0]] += (level - prev) * len(eligible)
            prev = level
        busted = [x for x in ranking if s[x] == 0]
        ko_winners = [next(y for y in ranking if risk[y] >= risk[x] and y != x) for x in busted]
        dead_order = sorted(busted, key=lambda x: -st[x]) if len(busted) > 1 else None
        return self.values(s, ko_winners, dead_order)


class Solver:
    def __init__(self, table, overcalls=False):
        self.t = table
        m = table.m
        self.overcalls = overcalls and m >= 3
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
        # strategies: push[i] for opener i (the big blind never opens), call[i, j] for j facing i's shove,
        # over[i, j, k] for k facing i's shove and j's call
        self.push = {i: np.full(N_CLASSES, 0.5) for i in range(m - 1)}
        self.call = {(i, j): np.full(N_CLASSES, 0.5) for i in range(m - 1) for j in range(i + 1, m)}
        self.over = {}
        if self.overcalls:
            self._init_multiway()

    def _init_multiway(self):
        m, n = self.t.m, N_CLASSES
        self.T3 = equity3.order_tensors()
        self._slices = {}
        self.D_all = np.maximum(self._order_marginals(np.ones((1, n)))[0].sum(axis=0), 1e-9)  # sum_c W3[a, b, c]
        self.W32 = self.W.astype(np.float32)
        E = equity_matrix()
        # heads-up showdown values of every player over the (shover hand, caller hand) grid: shape (m, n, n)
        self.hu = {(i, j): (np.multiply.outer(self.v_iwin[i, j], E)
                            + np.multiply.outer(self.v_jwin[i, j], 1 - E)).astype(np.float32)
                   for (i, j) in self.call}
        self.v3 = {}
        for i in range(m - 2):
            for j in range(i + 1, m - 1):
                for k in range(j + 1, m):
                    roles = (i, j, k)
                    self.v3[i, j, k] = np.array([self.t.allin_values([roles[r] for r in o]) for o in equity3.ORDERS],
                                                dtype=np.float32)
                    self.over[i, j, k] = np.full(n, 0.5)

    # --- range statistics -------------------------------------------------
    def _caller_stats(self, i, j):
        """Call probability and shover win probability for the j-call node, conditioned on i's range."""
        p, c = self.push[i] + 1e-12, self.call[i, j]
        wc = self.W @ c
        cp = float(p @ wc / (p @ self.Wsum))
        denom = float(p @ wc)
        pwin = float(p @ (self.WE @ c) / denom) if denom > 1e-12 else 0.5
        return cp, pwin

    def _after_shove_value(self, i, from_j, cps, called):
        """Value vector after i shoves and players before from_j have folded.

        cps[i, k] is the probability that k calls first, called[i, k] the value vector once k has called.
        """
        acc = np.zeros(self.t.m)
        q = 1.0
        for k in range(from_j, self.t.m):
            acc += q * cps[i, k] * called[i, k]
            q *= 1 - cps[i, k]
        return acc + q * self.v_steal[i]

    def _support_slices(self, name, key, take):
        """The order tensors cut down to the hands in play; supports rarely change between iterations."""
        cache = self._slices.setdefault(name, {})
        if key not in cache:
            cache.clear()
            cache[key] = [take(t) for t in self.T3]
        return cache[key]

    def _order_marginals(self, vectors):
        """S[t, o, a, b] = sum_c U_o[a, b, c] * v_t[c] for range vectors v_t over the overcaller's hand c,
        where U_o[a, b, c] is the combo weight times the probability of strength order o."""
        n = N_CLASSES
        v = np.asarray(vectors, dtype=np.float32)
        rows = np.flatnonzero(v.any(axis=0))  # only the hands in some overcalling range
        if len(rows) > n // 2:
            rows, tensors = np.arange(n), self.T3
        else:
            tensors = self._support_slices("rows", rows.tobytes(), lambda t: t[rows])
        out = np.empty((len(v), len(self.T3), n, n), dtype=np.float32)
        for o, t in enumerate(tensors):
            out[:, o] = (v[:, rows] @ t).reshape(len(v), n, n)
        return out

    def _overcaller_sums(self, mats):
        """X[t, o, c] = sum_{a, b} mats[t, a, b] * U_o[a, b, c] for the overcaller's hand c."""
        n = N_CLASSES
        mats = mats.astype(np.float32)
        a_idx = np.flatnonzero(mats.any(axis=(0, 2)))  # shover hands in some shoving range
        b_idx = np.flatnonzero(mats.any(axis=(0, 1)))  # caller hands in some calling range
        if len(a_idx) * len(b_idx) > n * n // 3:
            a_idx = b_idx = np.arange(n)
            tensors = self.T3
        else:
            cols = (a_idx[:, None] * n + b_idx[None, :]).ravel()
            tensors = self._support_slices("cols", cols.tobytes(), lambda t: t[:, cols])
        sub = mats[:, a_idx][:, :, b_idx].reshape(len(mats), -1)
        out = np.empty((len(mats), len(self.T3), n))
        for o, t in enumerate(tensors):
            out[:, o] = (t @ sub.T).T
        return out

    def _node_values(self, cps, called):
        """node[i] = value vector when it is folded to player i."""
        m = self.t.m
        node = [None] * m
        node[m - 1] = self.v_steal[m - 1]  # folded to the big blind: walk
        for i in range(m - 2, -1, -1):
            ppush = float(self.push[i] @ self.cc / self.cc.sum())
            node[i] = ppush * self._after_shove_value(i, i + 1, cps, called) + (1 - ppush) * node[i + 1]
        return node

    def evaluate(self):
        """Per-hand EVs of every decision against the current average strategies."""
        if self.overcalls:
            return self._evaluate_multiway()
        m = self.t.m
        stats = {(i, j): self._caller_stats(i, j) for (i, j) in self.call}
        cps = {key: cp for key, (cp, _) in stats.items()}
        called = {key: pwin * self.v_iwin[key] + (1 - pwin) * self.v_jwin[key] for key, (_, pwin) in stats.items()}
        node = self._node_values(cps, called)

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
                ev_fold = float(self._after_shove_value(i, j + 1, cps, called)[j])
                ev["call", i, j] = (ev_call, ev_fold)
        return ev

    def _evaluate_multiway(self):
        m, n = self.t.m, N_CLASSES
        W, Wsum = self.W32, self.Wsum
        f32 = lambda x: x.astype(np.float32)
        push = {i: f32(x) for i, x in self.push.items()}
        call = {key: f32(x) for key, x in self.call.items()}
        # the 3-way contractions skip hands left in a range only by early best responses
        prune = lambda x: np.where(x < MULTIWAY_MIN_FREQ, 0, x).astype(np.float32)
        keys = list(self.over)
        marg = dict(zip(keys, self._order_marginals([prune(self.over[key]) for key in keys])))
        # R[i, j, k][x, a, b] = value of player x when i shoved with a, j called with b, j+1 .. k-1 folded
        # and k acts; R[i, j, m] is the heads-up showdown
        R, over_p = {}, {}
        for (i, j) in self.call:
            cur = R[i, j, m] = self.hu[i, j]
            for k in range(m - 1, j, -1):
                s = marg[i, j, k]
                over_p[i, j, k] = s.sum(axis=0) / self.D_all
                cur = ((self.v3[i, j, k].T @ s.reshape(len(s), -1)).reshape(m, n, n) / self.D_all
                       + (1 - over_p[i, j, k]) * cur)
                R[i, j, k] = cur

        cps = {key: self._caller_stats(*key)[0] for key in self.call}
        called = {}
        for (i, j) in self.call:
            w = push[i][:, None] * W * call[i, j][None, :]
            called[i, j] = R[i, j, j + 1].reshape(m, -1) @ w.ravel() / max(w.sum(), 1e-12)
        node = self._node_values(cps, called)

        ev = {}
        for i in range(m - 1):
            ev_push = np.zeros(n)
            q = np.ones(n)
            for j in range(i + 1, m):
                wc = W * call[i, j][None, :]
                ev_push += q * (wc * R[i, j, j + 1][i]).sum(axis=1) / Wsum
                q *= 1 - wc.sum(axis=1) / Wsum
            ev_push += q * self.v_steal[i][i]
            ev["push", i] = (ev_push, float(node[i + 1][i]))
            wp = W * (push[i] + 1e-12)[:, None]
            for j in range(i + 1, m):
                ev_call = (wp * R[i, j, j + 1][j]).sum(axis=0) / wp.sum(axis=0)
                ev_fold = float(self._after_shove_value(i, j + 1, cps, called)[j])
                ev["call", i, j] = (ev_call, ev_fold)

        mats, folds = [], []
        for (i, j, k) in keys:
            reach = np.ones((n, n), dtype=np.float32)  # everybody between the caller and k folded
            for kk in range(j + 1, k):
                reach = reach * (1 - over_p[i, j, kk])
            mat = np.outer(prune(push[i]), prune(call[i, j])) * reach
            mats.append(mat)
            w = W * mat
            folds.append(float((w * R[i, j, k + 1][k]).sum() / max(w.sum(), 1e-12)))
        sums = self._overcaller_sums(np.array(mats))
        for t, (i, j, k) in enumerate(keys):
            num, den = self.v3[i, j, k][:, k] @ sums[t], sums[t].sum(axis=0)
            ev_over = np.divide(num, den, out=np.full(n, folds[t] - 1.0), where=den > 1e-9)
            ev["over", i, j, k] = (ev_over, folds[t])
        return ev

    def solve(self, iterations=None, tol=1e-4, warm_start=True, warm_weight=None):
        """Fictitious play. With overcalls the strategies start from the solution without overcalls,
        which counts as warm_weight iterations of the average, and the overcalls from a best response."""
        offset = 0
        if self.overcalls and warm_start:
            base = Solver(self.t).solve()
            # drop the hands that were (almost) never a best response so the ranges stay sparse
            clean = lambda x: np.where(x < WARM_START_MIN_FREQ, 0.0, x)
            self.push = {i: clean(x) for i, x in base.push.items()}
            self.call = {key: clean(x) for key, x in base.call.items()}
            self.over = {key: np.zeros(N_CLASSES) for key in self.over}
            offset = WARM_START_WEIGHT if warm_weight is None else warm_weight
            iterations = iterations or MULTIWAY_ITERATIONS
        iterations = iterations or 300
        strategies = [(self.push, "push"), (self.call, "call"), (self.over, "over")]
        for it in range(iterations):
            ev = self.evaluate()
            step = 1.0 / (it + 2 + offset)
            delta = 0.0
            for strat, name in strategies:
                rate = 1.0 if offset and it == 0 and name == "over" else step
                for key in strat:
                    e = ev[(name, *key) if isinstance(key, tuple) else (name, key)]
                    d = rate * ((e[0] > e[1]).astype(float) - strat[key])
                    strat[key] += d
                    if rate == step:
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
