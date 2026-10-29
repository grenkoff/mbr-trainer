"""Per-tournament results of Hero's tournaments, kept in the tournament_results table of mbr.db.

All-in adjusted results: a hand where the betting ended before the river with every remaining player's cards shown
counts at its expectation over the boards still to come, whoever played it: at the final table the others' all-ins
move Hero's money too. Money is valued as the "bounty" model does (ICM of 4/3/2 BI plus the bounties still to come
split by chips), with every knockout at the mean bounty of 0.945 BI instead of the envelope drawn.

Bounties are paid from 9th place on, so a knockout in the last Rush hands pays one too when the final table starts
with fewer than 9 players. Which of Hero's knockouts there were paid is not in the hand history: it is read from the
bounties of the tournament summary, as the number of envelopes that most likely adds up to them.

Besides: Hero's preflop stats by stage, position and stack, and his final-table push/fold decisions scored against
the HRC library (mbr.library).
"""

from collections import Counter

from psycopg.types.json import Jsonb

from mbr import db, library
from mbr.hh import parse_hand
from mbr.pushfold import MEAN_BOUNTY_BI, Table, table_from_hand
from mbr.ranges import class_label, class_of_cards
from mbr.showdown import rankings, split_pots
from mbr.spots import PLACE_PRIZE_BI, hero_preflop_spot, stack_bucket
from mbr.temperature import cool_down

VERSION = 1  # bump when what is computed changes: older results are then computed again
STREET_BOARD = {"PRE": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}
BETTING = ("fold", "check", "call", "bet", "raise")
VALUES = Table(names=[], stacks=[], posted=[], bb=1)  # only its values(): ICM + bounties
# mystery bounty envelopes of the $1 tournaments in BI: frequency per 100M (mean 0.945)
ENVELOPES = {10000: 60, 1000: 400, 100: 4000, 10: 3_500_000, 2: 3_600_000, 1.5: 3_800_000, 1: 4_000_000,
             0.75: 23_000_000, 0.5: 33_704_460, 0.25: 28_391_080}


def outcomes(hand):
    """[(probability, stacks after, {busted: [(winner, share of his bounty)]})] and whether the hand was
    adjusted (run out over the remaining boards) or is taken as it happened."""
    start, put = hand.stacks, hand.chips_put_in()
    folded = {a.name for a in hand.actions if a.verb == "fold"}
    live = [n for n in hand.players if n not in folded]
    shown = hand.shown_cards()
    betting = [a for a in hand.actions if a.verb in BETTING]
    nb = STREET_BOARD[betting[-1].street] if betting else 0
    if len(live) >= 2 and all(n in shown for n in live) and len(hand.board) >= nb:
        out = []
        for better, p in rankings([shown[n] for n in live], hand.board[:nb]):
            won, layers = split_pots(put, live, dict(zip(live, better)))
            stacks = {n: start[n] - put[n] + won[n] for n in start}
            out.append((p, stacks, knockouts(start, put, stacks, layers)))
        return out, nb < 5
    stacks = hand.final_stacks()
    # nobody busts without a showdown; a bust whose cards were not all shown goes to the players who won chips
    winners = [n for n in hand.collected if n not in folded]
    kos = {x: [(w, 1 / len(winners)) for w in winners] for x in start if stacks[x] <= 0 and start[x] > 0}
    return [(1.0, stacks, kos)], False


def knockouts(start, put, stacks, layers):
    """{busted: [(winner, share of his bounty)]}: a busted player's bounty goes to the winners of the lowest
    pot layer covering his chips. He may have "folded" a hand he was all-in in with his ante."""
    kos = {}
    for x in start:
        if start[x] > 0 and stacks[x] <= 0:
            winners = layers[min(level for level in layers if level >= put[x])]
            kos[x] = [(w, 1 / len(winners)) for w in winners]
    return kos


def hero_value(hand, stacks, kos=None):
    """Hero's money in BI after the hand: ICM, the bounties still to come and those won in this hand."""
    names = hand.players
    kos = kos or {}
    busted = sorted(kos, key=lambda x: -hand.stacks[x])
    ko_winners = [(names.index(w), share) for x in busted for w, share in kos[x]]
    dead_order = [names.index(x) for x in busted] if len(busted) > 1 else None
    v = VALUES.values([stacks[n] for n in names], ko_winners, dead_order)
    return v[names.index("Hero")]


def tournament(hands, summary=None):
    """Per tournament: Rush chips (actual, all-in adjusted), final-table value at the start and its change
    (actual, adjusted; the adjusted one also split into Hero's hands and the others'), knockouts with a bounty
    (at the final table, at the end of the Rush stage: actual and expected) and the bounties received."""
    r = Counter()
    ft_started = False
    ft_shares = []  # Hero's share of every bounty won at the final table
    last_rush = rush_end = None
    for hand in hands:
        outs, adjusted = outcomes(hand)
        actual = None if adjusted else outs[0]
        hero_in = not any(a.verb == "fold" and a.name == "Hero" for a in hand.actions)
        final = hand.final_stacks()
        if actual is None:  # the adjusted hand as it really ran out: the full board
            real = outcomes_on_board(hand)
            actual = real if real else (1.0, final, {})
        if not hand.is_final_stage:
            r["rush_actual"] += final["Hero"] - hand.stacks["Hero"]
            r["rush_ev"] += sum(p * s["Hero"] for p, s, _ in outs) - hand.stacks["Hero"]
            r["rush_adjusted"] += adjusted and hero_in
            last_rush = hand, outs, actual
            continue
        if not ft_started:
            ft_started = True
            r["ft"] = 1
            r["v0"] = hero_value(hand, hand.stacks)
            if len(hand.players) < 9 and last_rush:  # the Rush stage ended below 9 players
                rush_end = last_rush, 9 - len(hand.players)
        before = hero_value(hand, hand.stacks)
        ev = sum(p * hero_value(hand, s, k) for p, s, k in outs)
        busted = {x for x in hand.players if final[x] <= 0 and hand.stacks[x] > 0}
        r["ft_actual"] += hero_value(hand, actual[1], actual[2]) - before
        r["ft_ev"] += ev - before
        r["ft_ev_hero" if hero_in else "ft_ev_others"] += ev - before
        r["ft_actual_hero" if hero_in else "ft_actual_others"] += hero_value(hand, actual[1], actual[2]) - before
        ft_shares += [share for x in busted for w, share in actual[2].get(x, []) if w == "Hero"]
        # the run-out on the real board must give the chips of the hand history (an odd chip aside)
        r["check"] = max(r["check"], max(abs(actual[1][n] - final[n]) for n in final))
    r["kos"] = sum(ft_shares)
    paid = []
    if summary:
        r["bounties"] = (summary.received - summary.buyin * PLACE_PRIZE_BI.get(summary.place, 0)) / summary.buyin
    if rush_end:
        (hand, outs, actual), slots = rush_end
        # Hero's knockouts in the last Rush hand, best place first: players busted together finish in order of
        # their stacks, and the first `slots` of them (from 9th place on) were paid
        mine = sorted(((x, share) for x, ws in actual[2].items() for w, share in ws if w == "Hero"),
                      key=lambda xs: -hand.stacks[xs[0]])
        r["rush_candidates"] = len(mine)
        paid = paid_knockouts(r["bounties"], ft_shares, [share for _, share in mine][:slots]) if summary else []
        r["rush_kos"] = sum(paid)
        names = [x for x, _ in mine[:len(paid)]]
        r["rush_kos_ev"] = sum(p * share for p, _, kos in outs for x in names for w, share in kos.get(x, [])
                               if w == "Hero")
    r["unexplained"] = summary is not None and not bounty_likelihood(r["bounties"], ft_shares + paid)
    return r


def paid_knockouts(bounties, ft_shares, candidates):
    """The shares of the first of candidates (Hero's knockouts at the end of the Rush stage, best place first)
    whose bounties were paid: as many as most likely add up, with the final table ones, to the bounties received
    (in BI)."""
    best = max(range(len(candidates) + 1), key=lambda m: bounty_likelihood(bounties, ft_shares + candidates[:m]))
    return candidates[:best]


def bounty_likelihood(bounties, shares):
    """The probability that envelopes won with these shares add up to the bounties received (in BI)."""
    dist = {0.0: 1.0}
    for share in shares:
        nxt = Counter()
        for total, p in dist.items():
            for value, freq in ENVELOPES.items():
                nxt[round(total + share * value, 3)] += p * freq / 1e8
        dist = nxt
    # GG rounds a split bounty to the cent
    return sum(p for total, p in dist.items() if abs(total - bounties) < 0.011)


def outcomes_on_board(hand):
    """The outcome of an adjusted hand on the board that really came."""
    put = hand.chips_put_in()
    folded = {a.name for a in hand.actions if a.verb == "fold"}
    live = [n for n in hand.players if n not in folded]
    shown = hand.shown_cards()
    if len(hand.board) < 5:
        return None
    ((better, _),) = rankings([shown[n] for n in live], hand.board)
    won, layers = split_pots(put, live, dict(zip(live, better)))
    stacks = {n: hand.stacks[n] - put[n] + won[n] for n in hand.stacks}
    return 1.0, stacks, knockouts(hand.stacks, put, stacks, layers)


PREFLOP_KEYS = ("hands", "vpip", "pfr", "rfi_opp", "rfi", "open_shove", "limp", "threebet_opp", "threebet",
                "vs_shove_opp", "vs_shove_call")


def preflop(hands):
    """Hero's preflop counts by (stage, position, stack bucket of his stack in bb):
    hands dealt; VPIP, PFR; folded to him (rfi_opp): raised first in (rfi), of which all-in (open_shove), or
    limped; facing one raise (threebet_opp): re-raised; facing one shove (vs_shove_opp): called or re-shoved."""
    out = {}
    for hand in hands:
        if "Hero" not in hand.stacks:
            continue
        stage = "ft" if hand.is_final_stage else "rush"
        spot = hero_preflop_spot(hand)
        position = hand.positions().get("Hero", "?")
        c = out.setdefault((stage, position, stack_bucket(hand.stacks["Hero"] / hand.bb)), Counter())
        c["hands"] += 1
        mine = [a for a in hand.actions if a.street == "PRE" and a.name == "Hero" and a.verb in BETTING]
        c["vpip"] += any(a.verb in ("call", "raise", "bet") for a in mine)
        c["pfr"] += any(a.verb in ("raise", "bet") for a in mine)
        if spot is None:  # all-in from posting
            continue
        if spot.facing == "unopened" and position != "BB":
            c["rfi_opp"] += 1
            c["rfi"] += spot.hero_action in ("raise", "allin")
            c["open_shove"] += spot.hero_action == "allin"
            c["limp"] += spot.hero_action == "limp"
        elif spot.facing == "raise":
            c["threebet_opp"] += 1
            c["threebet"] += spot.hero_action in ("raise", "allin")
        elif spot.facing == "allin":
            c["vs_shove_opp"] += 1
            c["vs_shove_call"] += spot.hero_action in ("call", "allin")
    return [[*key, {k: c[k] for k in PREFLOP_KEYS}] for key, c in out.items()]


MAX_EFF_BB = 20  # deeper decisions are not push/fold


def decision(hand, setups, with_range=False):
    """Hero's final-table push/fold decision scored against the HRC library, or None: open shove or fold, call
    or fold facing a lone shove, overcall or fold facing a shove and a call. With overcalls up to
    OVERCALLS_MAX_PLAYERS, without them beyond. Limps and smaller raises are not scored.
    with_range adds HRC's frequency of playing and EV of playing less folding for each of the 169 hands."""
    setup = setups.get(hand.hand_id)
    spot = hero_preflop_spot(hand)
    if setup is None or setup["forced"] or spot is None or spot.eff_bb > MAX_EFF_BB or not hand.hero_cards:
        return None
    pre = [a for a in hand.actions if a.street == "PRE" and a.verb not in ("ante", "sb", "bb", "show")]
    before = [a for a in pre[:next(i for i, a in enumerate(pre) if a.name == "Hero")] if a.verb != "fold"]
    names = table_from_hand(hand).names  # preflop order, as the set-up's stacks and the solver's players
    h = names.index("Hero")
    if spot.facing == "unopened":
        kind, key = "open", ("push", h)
    elif spot.facing == "allin" and len(before) == 1 and before[0].verb in ("raise", "bet"):
        kind, key = "call", ("call", names.index(before[0].name), h)
    elif (spot.facing == "allin+call" and len(before) == 2 and before[0].verb in ("raise", "bet")
          and before[1].verb == "call"):
        kind, key = "overcall", ("over", names.index(before[0].name), names.index(before[1].name), h)
    else:
        return None
    overcalls = len(names) <= library.OVERCALLS_MAX_PLAYERS
    if kind == "overcall" and not overcalls or library.variant("hrc", overcalls) not in setup["solved"]:
        return None
    solver = library.load(setup, "hrc", overcalls)
    if key not in solver.ev:
        return None
    cls = class_of_cards(hand.hero_cards)
    ev_play, ev_fold = float(solver.ev[key][0][cls]), float(solver.ev[key][1])
    target = {"push": solver.push, "call": solver.call, "over": solver.over}[key[0]]
    freq = float(target[key[1] if key[0] == "push" else key[1:]][cls])
    action = {"allin": "play", "call": "play", "fold": "fold"}.get(spot.hero_action, "other")
    if action == "other":
        return None
    best = "play" if ev_play > ev_fold else "fold"
    out = {"hand_id": hand.hand_id, "kind": kind, "players": len(names), "position": spot.position,
           "eff_bb": round(spot.eff_bb, 1), "bucket": stack_bucket(spot.eff_bb), "cards": hand.hero_cards,
           "class": class_label(cls), "action": action, "best": best, "freq": round(freq, 3),
           "ev_play": round(ev_play, 4), "ev_fold": round(ev_fold, 4),
           "loss": round(abs(ev_play - ev_fold), 4) if action != best else 0.0, "overcalls": overcalls}
    if with_range:
        play = target[key[1] if key[0] == "push" else key[1:]]
        out["range"] = {"hand_class": cls, "freq": [round(float(f), 3) for f in play],
                        "ev_diff": [round(float(e) - ev_fold, 4) for e in solver.ev[key][0]]}
    return out


def compute(hands, summary, setups):
    """The results of one tournament, as kept in the database."""
    r = tournament(hands, summary)
    data = {k: float(v) if isinstance(v, float) else v for k, v in r.items()}
    data["date"] = hands[0].ts.isoformat() if hands else None
    if summary:
        data.update(buyin=summary.buyin, place=summary.place, received=summary.received,
                    roi=summary.received / summary.buyin - 1, place_prize=PLACE_PRIZE_BI.get(summary.place, 0))
    data["preflop"] = preflop(hands)
    data["decisions"] = [d for h in hands if h.is_final_stage and (d := decision(h, setups))]
    return data


def setups_for(conn, tournament_ids):
    """{hand id: set-up, with the variants solved} of these tournaments' final-table hands."""
    rows = conn.execute("SELECT u.*, array_remove(array_agg(s.variant), NULL) AS solved FROM setups u"
                        " JOIN hands h ON h.id = u.hand_id LEFT JOIN solutions s ON s.hand_id = u.hand_id"
                        " WHERE h.tournament_id = ANY(%s) GROUP BY u.hand_id", (list(tournament_ids),))
    return {r["hand_id"]: r for r in rows}


def pending():
    """Tournament ids without current results, oldest first."""
    with db.connect() as conn:
        rows = conn.execute("SELECT t.id FROM tournaments t LEFT JOIN tournament_results r ON r.tournament_id = t.id"
                            " WHERE (r.version IS NULL OR r.version <> %s) AND EXISTS"
                            " (SELECT 1 FROM hands h WHERE h.tournament_id = t.id) ORDER BY t.id", (VERSION,))
        return [r["id"] for r in rows]


def total():
    """How many tournaments have hands, so results to compute."""
    with db.connect() as conn:
        return conn.execute("SELECT count(DISTINCT tournament_id) AS n FROM hands").fetchone()["n"]


def update(tournament_ids, log=None, stop=None):
    """Compute and keep the results of these tournaments, cooling down the CPU between them."""
    summaries = db.load_summaries()
    for k, tid in enumerate(tournament_ids, 1):
        if stop is not None and stop.is_set():
            return
        cool_down(log=log)
        hands = db.load_hands([tid]).get(tid, [])
        with db.connect() as conn:
            data = compute(hands, summaries.get(tid), setups_for(conn, [tid]))
            conn.execute("INSERT INTO tournament_results (tournament_id, version, data) VALUES (%s, %s, %s)"
                         " ON CONFLICT (tournament_id) DO UPDATE SET version = EXCLUDED.version, data = EXCLUDED.data",
                         (tid, VERSION, Jsonb(data)))
        if log:
            log(f"{k}/{len(tournament_ids)}")


def hand_detail(hand_id):
    """A hand's history text and Hero's scored decision in it with HRC's range, or None if it is not uploaded."""
    with db.connect() as conn:
        row = conn.execute("SELECT raw, tournament_id FROM hands WHERE id = %s", (hand_id,)).fetchone()
        if row is None:
            return None
        hand = parse_hand(row["raw"])
        return {"hand_id": hand_id, "tid": row["tournament_id"], "text": row["raw"],
                "decision": decision(hand, setups_for(conn, [row["tournament_id"]]), with_range=True)}


def load():
    """{tournament id: results} of every tournament with current results."""
    with db.connect() as conn:
        rows = conn.execute("SELECT tournament_id, data FROM tournament_results WHERE version = %s", (VERSION,))
        return {r["tournament_id"]: r["data"] for r in rows}


class Worker:
    """Keeps the results current in the background: after uploads and newly solved set-ups (which drop the results
    of their tournaments), one tournament at a time, cooling down the CPU between them."""

    def __init__(self):
        import threading

        self.stop = threading.Event()
        self.wake = threading.Event()
        self.cooling = False
        self.left, self.total = len(pending()), total()
        threading.Thread(target=self._loop, daemon=True).start()

    def notify(self):
        self.wake.set()

    def _log(self, message):
        self.cooling = message.startswith("CPU")

    def _loop(self):
        import traceback

        while not self.stop.is_set():
            try:
                ids = pending()
                self.left, self.total = len(ids), total()
                if not ids:
                    self.wake.wait(10)
                    self.wake.clear()
                    continue
                update(ids[:1], log=self._log, stop=self.stop)
                self.cooling = False
            except Exception:
                traceback.print_exc()
                self.stop.wait(30)

    def status(self):
        return {"left": self.left, "total": self.total, "cooling": self.cooling}

    def shutdown(self):
        self.stop.set()
        self.wake.set()
