"""Final-table stack set-ups taken from the user's hand histories (the setups table of mbr.db)."""

from collections import defaultdict
from threading import Lock

from mbr import db
from mbr.pushfold import table_from_hand

_cache = {"count": None, "snapshots": {}}
_lock = Lock()


def setup_of(hand):
    """The set-up of a final-table hand with 2-9 players that fits the push/fold model, or None.

    A set-up holds the blind level and the stacks before posting, in preflop order (first to act ... BB).
    Set-ups where one player is all-in from posting are flagged "forced".
    """
    if not hand.is_final_stage:
        return None
    table = table_from_hand(hand, allow_forced=True)
    if table is None or not 2 <= table.m <= 9:
        return None
    return {"hand_id": hand.hand_id, "level": hand.level, "sb": hand.sb, "bb": hand.bb, "ante": hand.ante,
            "stacks": list(table.stacks), "forced": any(p >= s for p, s in zip(table.posted, table.stacks))}


def snapshots():
    """{players: [set-up]} for every set-up in the database, read again when uploads have added to it."""
    with db.connect() as conn:
        count = conn.execute("SELECT count(*) AS n FROM setups").fetchone()["n"]
        with _lock:
            if count != _cache["count"]:
                out = defaultdict(list)
                for r in conn.execute("SELECT * FROM setups ORDER BY hand_id"):
                    out[r["players"]].append({"hand_id": r["hand_id"], "level": r["level"], "sb": r["sb"],
                                              "bb": r["bb"], "ante": r["ante"], "stacks": r["stacks"],
                                              "forced": r["forced"]})
                _cache.update(count=count, snapshots=dict(out))
            return _cache["snapshots"]


def counts(forced=False):
    """{players: set-ups} without (or, if forced, only) those where a player is all-in from posting."""
    return {n: sum(s["forced"] == forced for s in snaps) for n, snaps in sorted(snapshots().items())}
