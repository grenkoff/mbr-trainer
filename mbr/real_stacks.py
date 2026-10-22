"""Final-table stack set-ups taken from the user's hand histories (hh/hands, not in git)."""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from mbr.hh import load_hands
from mbr.pushfold import table_from_hand

HANDS_DIR = Path(__file__).resolve().parent.parent / "hh" / "hands"


@lru_cache(maxsize=1)
def snapshots():
    """{players: [set-up]} for every final-table hand with 3-9 players that fits the push/fold model.

    A set-up holds the blind level and the stacks before posting, in preflop order (first to act ... BB).
    Set-ups where one player is all-in from posting are flagged "forced".
    """
    if not HANDS_DIR.is_dir():
        return {}
    out = defaultdict(list)
    for hands in load_hands(HANDS_DIR).values():
        for hand in hands:
            if not hand.is_final_stage:
                continue
            table = table_from_hand(hand, allow_forced=True)
            if table is None or not 3 <= table.m <= 9:
                continue
            out[table.m].append({"hand_id": hand.hand_id, "level": hand.level, "sb": hand.sb, "bb": hand.bb,
                                 "ante": hand.ante, "stacks": list(table.stacks),
                                 "forced": any(p >= s for p, s in zip(table.posted, table.stacks))})
    return dict(out)


def counts(forced=False):
    """{players: set-ups} without (or, if forced, only) those where a player is all-in from posting."""
    return {n: sum(s["forced"] == forced for s in snaps) for n, snaps in sorted(snapshots().items())}
