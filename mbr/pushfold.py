"""Push/fold equilibrium for Mystery Battle Royale final tables (ICM + mystery bounties).

Game model:
- players act in preflop order; the first player to act may shove or fold (no limps, no non-all-in raises);
- after a shove the remaining players, in order, may call or fold;
- without overcalls the first caller goes to showdown with the shover and everybody else folds;
- with overcalls the players behind the first caller may overcall, up to three players all-in in total
  (everybody after the overcaller folds); side pots are paid by the 3-way outcome tensor (mbr.equity3);
- if everybody folds, the big blind wins the blinds and antes;
- hands that tie split the pot and the bounty (mbr.ranges.tie_matrix, mbr.equity3.tie_tensors);
- the cards of the players who folded (bunching), to first order: they make the ranges of the players
  still in more likely to hold the hands that leave the folds likely, and they are dead for the board,
  which moves the heads-up equities and ties (mbr.ranges.dead_card_effects); for 3-way all-ins each
  hand's strength moves by mbr.ranges.dead_card_strength and the order probabilities by the smallest
  change that moves the three pairs that much (mbr.equity3.pair_shift_orders).

Values are in buy-ins:
- ICM (Malmuth-Harville) on the place prizes;
- plus the mystery bounty for busting a player;
- plus, optionally, the future bounties still to be won at the table, split in proportion to chips.

Strategies are solved with fictitious play: every iteration each player best-responds to the averaged
strategies of the others, and the averages move towards the best responses. Later iterations weigh
more in the average (AVERAGING), which converges far faster than the plain mean.
"""

from dataclasses import dataclass, field

import numpy as np

import os

from mbr import equity3, native
from mbr.icm import icm
from mbr.ranges import (N_CLASSES, combo_counts, dead_card_effects, dead_card_strength, equity_matrix, pair_weights,
                        rank_counts, tie_matrix)

PAYOUTS_BI = [4.0, 3.0, 2.0]
MEAN_BOUNTY_BI = 0.945
# fictitious play with overcalls starts from the solution without them
WARM_START_WEIGHT = 1
WARM_START_MIN_FREQ = 0.02
MULTIWAY_MIN_FREQ = 0.03
MULTIWAY_ITERATIONS = 40
COMBOS = 1326
# iteration t enters the average with weight ~ t^(AVERAGING - 1); 1 is plain fictitious play
AVERAGING = 4


@dataclass
class Table:
    names: list  # players in preflop action order (first to act ... big blind)
    stacks: list  # chips before posting antes and blinds
    posted: list  # antes + blinds posted by each player
    bb: int
    payouts: list = field(default_factory=lambda: list(PAYOUTS_BI))
    bounty: float = MEAN_BOUNTY_BI
    future_bounty: bool = True
    bunching: bool = True  # the solver takes the folded players' cards into account

    @property
    def m(self):
        return len(self.names)

    def values(self, stacks_after, ko_winners=(), dead_order=None, split=False):
        """ko_winners: (player, share of a bounty) for every bounty won in the hand; split: the pot was
        split between tied hands (mbr.fgs values those outcomes differently)."""
        v = np.array(icm(stacks_after, self.payouts, dead_order))
        for w, share in ko_winners:
            v[w] += share * self.bounty
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
        """Values after an all-in between the players in ranking, ordered from the best hand to the worst;
        an entry may be a tuple of players whose hands tie.

        Each player risks up to the second largest stack among them; every pot layer goes to the best
        eligible hands, split between ties, the dead money with the first layer. A busted player's bounty
        goes to the winners of the top layer they paid into; players busted together finish in order of
        their stacks.
        """
        groups = [tuple(g) if isinstance(g, (tuple, list)) else (g,) for g in ranking]
        rank = {x: k for k, g in enumerate(groups) for x in g}
        st = self.stacks
        cap = sorted(st[x] for x in rank)[-2]
        risk = {x: min(st[x], cap) for x in rank}
        s = [x - p for x, p in zip(st, self.posted)]
        for x in rank:
            s[x] = st[x] - risk[x]
        prev = 0
        pot = sum(p for k, p in enumerate(self.posted) if k not in rank)
        layer_winners = {}
        for level in sorted(set(risk.values())):
            eligible = [x for x in rank if risk[x] >= level]
            best = min(rank[x] for x in eligible)
            winners = layer_winners[level] = [x for x in eligible if rank[x] == best]
            pot += (level - prev) * len(eligible)
            for x in winners:
                s[x] += pot / len(winners)
            pot, prev = 0, level
        busted = [x for x in rank if s[x] == 0]
        ko_winners = [(w, 1 / len(layer_winners[risk[x]])) for x in busted for w in layer_winners[risk[x]]]
        dead_order = sorted(busted, key=lambda x: -st[x]) if len(busted) > 1 else None
        return self.values(s, ko_winners, dead_order, split=any(len(g) > 1 for g in groups))


class Solver:
    def __init__(self, table, overcalls=False, bunching=None, forced_open=False):
        """forced_open: the first player is all-in from posting; he always "shoves" and the others can
        only call him (mbr.fgs)."""
        self.t = table
        m = table.m
        self.overcalls = overcalls and m >= 3
        self.bunching = table.bunching if bunching is None else bunching
        self.forced_open = forced_open
        self.W = pair_weights()
        self.E, self.T = equity_matrix(), tie_matrix()
        self.Wsum = self.W.sum(axis=1)
        self.cc = combo_counts()
        # terminal values (vectors over players)
        self.v_steal = [table.steal_values(i) for i in range(m)]
        self.v_iwin = {}
        self.v_jwin = {}
        self.v_tie = {}  # a tie minus the mean of a win and a loss: E counts a tie as half a win
        for i in range(m - 1):
            for j in range(i + 1, m):
                self.v_iwin[i, j] = table.showdown_values(i, j)
                self.v_jwin[i, j] = table.showdown_values(j, i)
                self.v_tie[i, j] = table.allin_values([(i, j)]) - (self.v_iwin[i, j] + self.v_jwin[i, j]) / 2
        # strategies: push[i] for opener i (the big blind never opens), call[i, j] for j facing i's shove,
        # over[i, j, k] for k facing i's shove and j's call
        self.push = {i: np.full(N_CLASSES, 1.0 if forced_open and i == 0 else 0.5) for i in range(m - 1)}
        self.call = {(i, j): np.full(N_CLASSES, 0.5) for i in range(m - 1) for j in range(i + 1, m)}
        self.over = {}
        # 3-way all-in values: for each strength order (equity3.ORDERS), then for each tie (equity3.TIES)
        # the value of the tie minus the mean value of the orders it fits
        self.v3 = {}
        if self.overcalls:
            for i in range(m - 2):
                for j in range(i + 1, m - 1):
                    for k in range(j + 1, m):
                        roles = (i, j, k)
                        orders = [table.allin_values([roles[r] for r in o]) for o in equity3.ORDERS]
                        ties = [table.allin_values([tuple(roles[r] for r in g) for g in groups])
                                - np.mean([orders[q] for q in equity3.fitting_orders(groups)], axis=0)
                                for groups in equity3.TIES]
                        self.v3[i, j, k] = np.array(orders + ties, dtype=np.float32)
                        self.over[i, j, k] = np.full(N_CLASSES, 0.5)

    def _init_multiway(self):
        """Arrays for the numpy version of the multiway evaluation."""
        if hasattr(self, "T3"):
            return
        self.T3 = equity3.order_tensors() + equity3.tie_tensors()  # the orders first, as in self.v3
        self._slices = {}
        self.D_all = equity3.order_mass().reshape(N_CLASSES, N_CLASSES)  # sum_c W3[a, b, c]
        self.W32 = self.W.astype(np.float32)

    # --- folded players' cards ----------------------------------------------
    def _fold_stats(self, fold):
        """For a player folding the hands in `fold`: the expected number of cards of each rank he holds
        when he folds, and phi[x] = P(he folds | another player holds x) / P(he folds)."""
        mass = float(fold @ self.cc)
        if not self.bunching or mass <= 1e-12:
            return np.zeros(13), np.ones(N_CLASSES)
        return (fold * self.cc) @ rank_counts() / mass, (self.W @ fold) / self.Wsum / (mass / COMBOS)

    def _bunching(self):
        """Who folded before each decision, as the weights and equities it is played with.

        pair[i, j] = (W, E, T) when i shoved and j calls, given the players before i and between i and j
        folded: W[a, b] times the folds' phi of a and b (phi[i, j]), the equities and ties moved by their
        dead cards;
        showdown[i, j] = (E, T) once everybody behind j folded too (with overcalls);
        over[i, j, k] = phi of k's hands at the overcall node and strength[i, j, k] how much the dead cards
        there move each hand (dead_card_strength); opener[i] = phi of i's hands when folded to."""
        m = self.t.m
        DE, DT = dead_card_effects()
        stats = {}
        for i in range(m - 1):
            stats["push", i] = self._fold_stats(1 - self.push[i])
        for key, c in self.call.items():
            stats["call", *key] = self._fold_stats(1 - c)
        for key, o in self.over.items():
            stats["over", *key] = self._fold_stats(1 - o)

        def combine(keys):
            nu, phi = np.zeros(13), np.ones(N_CLASSES)
            for key in keys:
                nu, phi = nu + stats[key][0], phi * stats[key][1]
            return nu, phi

        out = {"pair": {}, "phi": {}, "showdown": {}, "over": {}, "strength": {},
               "opener": [combine([("push", x) for x in range(i)])[1] for i in range(m)]}
        for (i, j) in self.call:
            before = [("push", x) for x in range(i)] + [("call", i, x) for x in range(i + 1, j)]
            nu, phi = combine(before)
            out["pair"][i, j] = (self.W * np.outer(phi, phi), self.E + DE @ nu, self.T + DT @ nu)
            out["phi"][i, j] = phi
            if self.over:
                behind = [("over", i, j, x) for x in range(j + 1, m)]
                nu_all = nu + combine(behind)[0]
                out["showdown"][i, j] = (self.E + DE @ nu_all, self.T + DT @ nu_all)
                for k in range(j + 1, m):
                    nu_k, phi_k = combine([("over", i, j, x) for x in range(j + 1, k)])
                    out["over"][i, j, k] = phi * phi_k
                    out["strength"][i, j, k] = dead_card_strength() @ (nu + nu_k)
        return out

    # --- range statistics -------------------------------------------------
    def _caller_stats(self, i, j, bunch=None):
        """Call probability, shover equity and tie probability for the j-call node, over i's range."""
        W, E, T = (bunch or self._bunching())["pair"][i, j]
        p, c = self.push[i] + 1e-12, self.call[i, j]
        wc = W @ c
        cp = float(p @ wc / (p @ W.sum(axis=1)))
        denom = float(p @ wc)
        pwin = float(p @ ((W * E) @ c) / denom) if denom > 1e-12 else 0.5
        ptie = float(p @ ((W * T) @ c) / denom) if denom > 1e-12 else 0.0
        return cp, pwin, ptie

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

    def _node_values(self, cps, called, bunch):
        """node[i] = value vector when it is folded to player i."""
        m = self.t.m
        node = [None] * m
        node[m - 1] = self.v_steal[m - 1]  # folded to the big blind: walk
        for i in range(m - 2, -1, -1):
            w = self.cc * bunch["opener"][i]
            ppush = float(self.push[i] @ w / w.sum())
            node[i] = ppush * self._after_shove_value(i, i + 1, cps, called) + (1 - ppush) * node[i + 1]
        return node

    def evaluate(self):
        """Per-hand EVs of every decision against the current average strategies."""
        if self.overcalls:
            return self._evaluate_multiway()
        m = self.t.m
        bunch = self._bunching()
        stats = {(i, j): self._caller_stats(i, j, bunch) for (i, j) in self.call}
        cps = {key: cp for key, (cp, _, _) in stats.items()}
        called = {key: pwin * self.v_iwin[key] + (1 - pwin) * self.v_jwin[key] + ptie * self.v_tie[key]
                  for key, (_, pwin, ptie) in stats.items()}
        node = self._node = self._node_values(cps, called, bunch)

        ev = {}
        for i in range(m - 1):
            # opener i with hand a
            ev_push = np.zeros(N_CLASSES)
            q = np.ones(N_CLASSES)
            p = self.push[i] + 1e-12
            for j in range(i + 1, m):
                W, E, T = bunch["pair"][i, j]
                c = self.call[i, j]
                wc = W @ c
                callp = wc / W.sum(axis=1)
                eqc = np.divide((W * E) @ c, wc, out=np.full(N_CLASSES, 0.5), where=wc > 1e-12)
                tiec = np.divide((W * T) @ c, wc, out=np.zeros(N_CLASSES), where=wc > 1e-12)
                ev_push += q * callp * (eqc * self.v_iwin[i, j][i] + (1 - eqc) * self.v_jwin[i, j][i]
                                        + tiec * self.v_tie[i, j][i])
                q *= 1 - callp
                # caller j facing i's shove with hand b
                wp = W @ p
                eqp = ((W * E) @ p) / wp  # equity of caller's hand b vs shover range
                tiep = ((W * T) @ p) / wp
                ev_call = eqp * self.v_jwin[i, j][j] + (1 - eqp) * self.v_iwin[i, j][j] + tiep * self.v_tie[i, j][j]
                ev_fold = float(self._after_shove_value(i, j + 1, cps, called)[j])
                ev["call", i, j] = (ev_call, ev_fold)
            ev_push += q * self.v_steal[i][i]
            ev["push", i] = (ev_push, float(node[i + 1][i]))
        return ev

    def _evaluate_multiway(self):
        self._init_multiway()
        m, n = self.t.m, N_CLASSES
        bunch = self._bunching()
        f32 = lambda x: x.astype(np.float32)
        push = {i: f32(x) for i, x in self.push.items()}
        call = {key: f32(x) for key, x in self.call.items()}
        pair_w = {key: f32(w) for key, (w, _, _) in bunch["pair"].items()}
        # the 3-way contractions skip hands left in a range only by early best responses
        prune = lambda x: np.where(x < MULTIWAY_MIN_FREQ, 0, x).astype(np.float32)
        keys = list(self.over)
        marg = dict(zip(keys, self._order_marginals([prune(self.over[key]) * f32(bunch["over"][key])
                                                     for key in keys])))
        # dead cards in 3-way all-ins: the pair (u, v) of roles ends in the order u above v more often by
        # strength[u's hand] - strength[v's hand], which moves the value of player x by
        # sum over roles of alpha[role, x] * strength[the role's hand] (pair_shift_orders)
        n_ord = len(equity3.ORDERS)
        strength = {key: f32(bunch["strength"][key]) for key in keys}
        alpha = {}
        for key in keys:
            g = equity3.pair_shift_orders().T @ self.v3[key][:n_ord]  # [pair (0,1), (0,2), (1,2), x]
            alpha[key] = np.array([g[0] + g[1], g[2] - g[0], -g[1] - g[2]], dtype=np.float32)
        mass = equity3.triple_counts()
        over_strength = {key: (prune(self.over[key]) * f32(bunch["over"][key]) * strength[key]) @ mass
                         for key in keys}  # [a * n + b] sum over c of W3 * over * phi * strength
        # R[i, j, k][x, a, b] = value of player x when i shoved with a, j called with b, j+1 .. k-1 folded
        # and k acts; R[i, j, m] is the heads-up showdown
        R, over_p = {}, {}
        for (i, j) in self.call:
            E, T = bunch["showdown"][i, j]
            cur = R[i, j, m] = (np.multiply.outer(self.v_iwin[i, j], E) + np.multiply.outer(self.v_jwin[i, j], 1 - E)
                                + np.multiply.outer(self.v_tie[i, j], T)).astype(np.float32)
            for k in range(m - 1, j, -1):
                s = marg[i, j, k]
                # at most 1: rounding could push a full overcalling range over and make weights negative
                over_mass = s[:n_ord].sum(axis=0)
                over_p[i, j, k] = np.minimum(over_mass / self.D_all, 1.0)
                al, st = alpha[i, j, k], strength[i, j, k]
                shift = ((al[0][:, None, None] * st[None, :, None] + al[1][:, None, None] * st[None, None, :]) * over_mass
                         + al[2][:, None, None] * over_strength[i, j, k].reshape(n, n))
                cur = ((self.v3[i, j, k].T @ s.reshape(len(s), -1)).reshape(m, n, n) + shift) / self.D_all \
                    + (1 - over_p[i, j, k]) * cur
                R[i, j, k] = cur

        cps = {key: self._caller_stats(*key, bunch)[0] for key in self.call}
        called = {}
        for (i, j) in self.call:
            w = push[i][:, None] * pair_w[i, j] * call[i, j][None, :]
            called[i, j] = R[i, j, j + 1].reshape(m, -1) @ w.ravel() / max(w.sum(), 1e-12)
        node = self._node = self._node_values(cps, called, bunch)

        ev = {}
        for i in range(m - 1):
            ev_push = np.zeros(n)
            q = np.ones(n)
            for j in range(i + 1, m):
                W = pair_w[i, j]
                wc = W * call[i, j][None, :]
                ev_push += q * (wc * R[i, j, j + 1][i]).sum(axis=1) / W.sum(axis=1)
                q *= 1 - wc.sum(axis=1) / W.sum(axis=1)
            ev_push += q * self.v_steal[i][i]
            ev["push", i] = (ev_push, float(node[i + 1][i]))
            for j in range(i + 1, m):
                wp = pair_w[i, j] * (push[i] + 1e-12)[:, None]
                ev_call = (wp * R[i, j, j + 1][j]).sum(axis=0) / wp.sum(axis=0)
                ev_fold = float(self._after_shove_value(i, j + 1, cps, called)[j])
                ev["call", i, j] = (ev_call, ev_fold)

        mats, folds = [], []
        for (i, j, k) in keys:
            reach = np.ones((n, n), dtype=np.float32)  # everybody between the caller and k folded
            for kk in range(j + 1, k):
                reach = reach * (1 - over_p[i, j, kk])
            phi = f32(bunch["phi"][i, j])
            mat = np.outer(prune(push[i]) * phi, prune(call[i, j]) * phi) * reach
            mats.append(mat)
            w = self.W32 * mat
            folds.append(float((w * R[i, j, k + 1][k]).sum() / max(w.sum(), 1e-12)))
        sums = self._overcaller_sums(np.array(mats))
        for t, (i, j, k) in enumerate(keys):
            al, st = alpha[i, j, k][:, k], strength[i, j, k]
            shifted = mats[t] * (al[0] * st[:, None] + al[1] * st[None, :])
            den = sums[t][:n_ord].sum(axis=0)
            num = self.v3[i, j, k][:, k] @ sums[t] + mass @ shifted.ravel() + al[2] * st * den
            ev_over = np.divide(num, den, out=np.full(n, folds[t] - 1.0), where=den > 1e-9)
            ev["over", i, j, k] = (ev_over, folds[t])
        return ev

    def state_values(self):
        """Each player's value at the start of the hand, before the cards are dealt, when everybody plays
        the current strategies (numpy)."""
        self.evaluate()
        return self._node[0]

    def solve(self, iterations=None, tol=1e-4, warm_start=True, warm_weight=None):
        """Fictitious play. With overcalls the strategies start from the solution without overcalls,
        which counts as warm_weight iterations of the average, and the overcalls from a best response.

        The default schedule runs in the Rust solver when it is built (see mbr.native), otherwise in numpy.
        """
        if native.enabled() and iterations is None and warm_start and warm_weight is None:
            return self._solve_native(tol)
        return self.solve_python(iterations, tol, warm_start, warm_weight)

    def _solve_native(self, tol):
        m = self.t.m
        pairs, triples = list(self.call), list(self.over)
        params = [300, tol, MULTIWAY_ITERATIONS, WARM_START_WEIGHT, WARM_START_MIN_FREQ, MULTIWAY_MIN_FREQ,
                  float(os.environ.get("MBR_THREADS", 0)), AVERAGING, float(self.forced_open)]
        strategy, ev_play, ev_fold = native.solve(
            m, self.W, equity_matrix(), tie_matrix(), self.cc, np.array(self.v_steal),
            [self.v_iwin[p] for p in pairs], [self.v_jwin[p] for p in pairs], [self.v_tie[p] for p in pairs],
            [self.v3[t] for t in triples],
            equity3.order_tensors() + equity3.tie_tensors() + (equity3.triple_counts(),) if self.overcalls else None,
            equity3.order_mass() if self.overcalls else None,
            (*dead_card_effects(), rank_counts(), dead_card_strength()) if self.bunching else None, params)
        self.ev = {}
        nodes = [("push", self.push, i) for i in self.push] + [("call", self.call, p) for p in pairs] \
            + [("over", self.over, t) for t in triples]
        for node, (name, strat, key) in enumerate(nodes):
            strat[key] = strategy[node]
            self.ev[(name, *key) if isinstance(key, tuple) else (name, key)] = (ev_play[node], float(ev_fold[node]))
        return self

    def solve_python(self, iterations=None, tol=1e-4, warm_start=True, warm_weight=None):
        """The numpy version of solve()."""
        offset = 0
        if self.overcalls and warm_start:
            base = Solver(self.t, bunching=self.bunching, forced_open=self.forced_open).solve_python()
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
            step = AVERAGING / (it + AVERAGING + 1 + offset)
            delta = 0.0
            for strat, name in strategies:
                rate = 1.0 if offset and it == 0 and name == "over" else step
                for key in strat:
                    if self.forced_open and name == "push" and key == 0:
                        continue
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
