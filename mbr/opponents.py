"""The average opponent: how everybody but Hero plays preflop in the uploaded hands, kept per tournament in the
opponent_results table of mbr.db and brought up to date in the background with Hero's results (mbr.results).

Per tournament: the preflop counts of all opponents together (as Hero's in mbr.results.preflop), their final-table
push/fold decisions against the HRC library (how often they play against how much of the hands HRC plays there,
and, when their hand was shown, whether HRC plays it), and the hands shown when they called a shove or open-shoved.
A call of a shove is always shown down, so those hands are the whole calling range; an open shove only when called.
Their limps and raises short of all-in where push/fold has a decision are counted apart, unscored.
"""

from collections import Counter

from psycopg.types.json import Jsonb

from mbr import db, results
from mbr.ranges import class_of_cards
from mbr.spots import preflop_spot, stack_bucket
from mbr.temperature import cool_down

VERSION = 1
DECISION_KEYS = ("kind", "players", "position", "bucket", "action", "range_pct", "class", "best", "freq", "loss")


def shown(hands):
    """{"stage kind bucket": {hand class: times}} of the hands opponents showed after calling a shove ("call")
    or open-shoving ("shove"), by the effective stack."""
    out = {}
    for hand in hands:
        stage = "ft" if hand.is_final_stage else "rush"
        for name in hand.players:
            if name == "Hero" or (spot := preflop_spot(hand, name)) is None or not spot.cards:
                continue
            if spot.facing == "allin" and spot.hero_action in ("call", "allin"):
                kind = "call"
            elif spot.facing == "unopened" and spot.hero_action == "allin":
                kind = "shove"
            else:
                continue
            counts = out.setdefault(f"{stage} {kind} {stack_bucket(spot.eff_bb)}", Counter())
            counts[class_of_cards(spot.cards)] += 1
    return {k: dict(c) for k, c in out.items()}


def compute(hands, summary, setups):
    """The opponents' data of one tournament, as kept in the database."""
    solvers = {}  # each hand's solution loaded once for all its players
    decisions = [{k: d[k] for k in DECISION_KEYS if k in d}
                 for hand in hands if hand.is_final_stage for name in hand.players if name != "Hero"
                 if (d := results.decision(hand, setups, name=name, solvers=solvers, others=True))]
    return {"date": hands[0].ts.isoformat() if hands else None, "buyin": summary.buyin if summary else None,
            "preflop": results.preflop(hands, opponents=True), "decisions": decisions, "shown": shown(hands)}


def pending():
    """Tournament ids without current opponents' data, oldest first."""
    with db.connect() as conn:
        rows = conn.execute("SELECT t.id FROM tournaments t LEFT JOIN opponent_results r ON r.tournament_id = t.id"
                            " WHERE (r.version IS NULL OR r.version <> %s) AND EXISTS"
                            " (SELECT 1 FROM hands h WHERE h.tournament_id = t.id) ORDER BY t.id", (VERSION,))
        return [r["id"] for r in rows]


def update(tournament_ids, log=None, stop=None):
    """Compute and keep the opponents' data of these tournaments, cooling down the CPU between them."""
    summaries = db.load_summaries()
    for tid in tournament_ids:
        if stop is not None and stop.is_set():
            return
        cool_down(log=log)
        hands = db.load_hands([tid]).get(tid, [])
        with db.connect() as conn:
            data = compute(hands, summaries.get(tid), results.setups_for(conn, [tid]))
            conn.execute("INSERT INTO opponent_results (tournament_id, version, data) VALUES (%s, %s, %s)"
                         " ON CONFLICT (tournament_id) DO UPDATE SET version = EXCLUDED.version, data = EXCLUDED.data",
                         (tid, VERSION, Jsonb(data)))


def load():
    """{tournament id: opponents' data} of every tournament with current data."""
    with db.connect() as conn:
        rows = conn.execute("SELECT tournament_id, data FROM opponent_results WHERE version = %s", (VERSION,))
        return {r["tournament_id"]: r["data"] for r in rows}
