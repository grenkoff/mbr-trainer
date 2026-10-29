"""Extract Hero's preflop decision spots at the final table and tournament structure stats."""

from collections import Counter
from dataclasses import dataclass
from statistics import median

PLACE_PRIZE_BI = {1: 4, 2: 3, 3: 2}  # place prize in buy-ins
STACK_BUCKETS = [(0, 5), (5, 8), (8, 12), (12, 15), (15, 20), (20, 30), (30, 10**9)]


def stack_bucket(bb):
    for lo, hi in STACK_BUCKETS:
        if lo <= bb < hi:
            return f"{lo}-{hi}bb" if hi < 10**9 else f"{lo}+bb"
    return "?"


@dataclass
class Spot:
    tid: str
    hand_id: str
    n_players: int
    position: str
    hero_bb: float  # Hero stack at start of hand, in big blinds
    eff_bb: float  # effective stack vs the relevant opponent(s)
    facing: str  # unopened, limp, raise, allin, allin+call, 3bet ...
    covers: str  # cover / covered / mixed (vs aggressor, or players behind when unopened)
    hero_action: str  # fold, check, call, raise, allin
    cards: str


def hero_preflop_spot(hand):
    """Hero's first voluntary preflop decision in a hand, or None."""
    return preflop_spot(hand, "Hero")


def preflop_spot(hand, name):
    """A player's first voluntary preflop decision in a hand, or None (the Spot's hero_* fields are his)."""
    if name not in hand.stacks:
        return None
    stacks = hand.stacks
    pos = hand.positions()
    pre = [a for a in hand.actions if a.street == "PRE" and a.verb not in ("ante", "sb", "bb", "show")]
    hero_idx = next((i for i, a in enumerate(pre) if a.name == name), None)
    if hero_idx is None:
        return None  # all-in from posting blinds
    before = pre[:hero_idx]
    raises = [a for a in before if a.verb in ("raise", "bet")]
    allins = [a for a in before if a.all_in and a.verb in ("raise", "bet", "call")]
    calls = [a for a in before if a.verb == "call"]
    hero_stack = stacks[name]

    if not raises:
        facing = "limp" if calls else "unopened"
        acted = {a.name for a in before}
        behind = [n for n in hand.players if n != name and n not in acted]
        # players still to act after him (folded players removed)
        folded = {a.name for a in before if a.verb == "fold"}
        opponents = [n for n in hand.players if n != name and n not in folded]
        biggest = max((stacks[n] for n in opponents), default=0)
        eff = min(hero_stack, biggest)
        covered_by = sum(stacks[n] > hero_stack for n in behind)
        covers = "cover-all" if covered_by == 0 else ("covered-by-all" if covered_by == len(behind) else "mixed")
    else:
        aggressor = raises[-1].name
        n_raises = len(raises)
        if allins and allins[-1].verb in ("raise", "bet") and allins[-1].name == aggressor:
            facing = "allin"
            later_calls = [a for a in before[before.index(allins[-1]) + 1:] if a.verb == "call"]
            if later_calls:
                facing = "allin+call"
        else:
            facing = {1: "raise", 2: "3bet"}.get(n_raises, "4bet+")
            if any(a.verb == "call" for a in before[before.index(raises[-1]) + 1:]):
                facing += "+call"
        eff = min(hero_stack, stacks[aggressor])
        covers = "cover" if hero_stack >= stacks[aggressor] else "covered"

    h = pre[hero_idx]
    if h.verb in ("raise", "bet") and h.all_in:
        hero_action = "allin"
    elif h.verb == "call" and facing == "unopened":
        hero_action = "limp"
    else:
        hero_action = h.verb
    return Spot(
        tid=hand.tid, hand_id=hand.hand_id, n_players=len(hand.players),
        position=pos.get(name, "?"), hero_bb=hero_stack / hand.bb, eff_bb=eff / hand.bb,
        facing=facing, covers=covers, hero_action=hero_action,
        cards=hand.hero_cards if name == "Hero" else " ".join(hand.shown_cards().get(name, [])),
    )


def busts(hands):
    """Yield (hand, busted_names) for consecutive final-stage hands of one tournament."""
    ft = [h for h in hands if h.is_final_stage]
    for a, b in zip(ft, ft[1:]):
        gone = [n for n in a.players if n not in b.stacks]
        if gone:
            yield a, gone
    if ft:
        last = ft[-1]
        fs = last.final_stacks()
        gone = [n for n, s in fs.items() if s <= 0]
        if gone:
            yield last, gone


def hero_knockouts(hands):
    """Count players busted in hands where Hero won chips (Hero not among the busted)."""
    kos = 0
    for hand, gone in busts(hands):
        if "Hero" in gone or hand.collected.get("Hero", 0) == 0:
            continue
        # a busted player is Hero's KO if Hero covered them and won a pot they were in
        kos += sum(1 for n in gone if hand.stacks[n] <= hand.stacks["Hero"])
    return kos


def structure_stats(tournaments):
    level_starts = {}  # level -> list of durations in seconds
    ft_start_levels = Counter()
    ft_start_bb = []
    ft_start_hero_bb = []
    rush_minutes = []
    for hands in tournaments.values():
        first_at_level = {}
        for h in hands:
            first_at_level.setdefault(h.level, h.ts)
        levels = sorted(first_at_level)
        for l1, l2 in zip(levels, levels[1:]):
            if l2 == l1 + 1:
                level_starts.setdefault(l1, []).append((first_at_level[l2] - first_at_level[l1]).total_seconds())
        ft = [h for h in hands if h.is_final_stage]
        if ft:
            h0 = ft[0]
            ft_start_levels[(h0.level, h0.sb, h0.bb, h0.ante)] += 1
            ft_start_bb += [s / h0.bb for s in h0.stacks.values()]
            ft_start_hero_bb.append(h0.stacks["Hero"] / h0.bb)
            rush_minutes.append((h0.ts - hands[0].ts).total_seconds() / 60)
    return {
        "level_minutes": {l: median(v) / 60 for l, v in sorted(level_starts.items()) if len(v) >= 5},
        "ft_start_levels": ft_start_levels,
        "ft_start_bb": ft_start_bb,
        "ft_start_hero_bb": ft_start_hero_bb,
        "rush_minutes": rush_minutes,
    }
