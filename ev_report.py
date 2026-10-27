"""All-in adjusted results of Hero's tournaments: chipEV/tour in the Rush stage, $EV at the final table.

A hand where the betting ended before the river with every remaining player's cards shown counts at its
expectation over the boards still to come, whoever played it: at the final table the others' all-ins move
Hero's money too. Money is valued as the "bounty" model does (ICM of 4/3/2 BI plus the bounties still to
come split by chips), with every knockout at the mean bounty of 0.945 BI instead of the envelope drawn.

    .venv/bin/python ev_report.py
"""

import sys
import time
from collections import Counter
from statistics import mean, stdev

from build_library import cpu_temperature
from mbr.hh import load_hands, load_summaries
from mbr.pushfold import MEAN_BOUNTY_BI, Table
from mbr.showdown import rankings, split_pots
from mbr.spots import PLACE_PRIZE_BI

STREET_BOARD = {"PRE": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}
BETTING = ("fold", "check", "call", "bet", "raise")
VALUES = Table(names=[], stacks=[], posted=[], bb=1)  # only its values(): ICM + bounties


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


def tournament(hands):
    """Per tournament: Rush chips (actual, all-in adjusted), final-table value at the start and its change
    (actual, adjusted; the adjusted one also split into Hero's hands and the others'), knockouts."""
    r = Counter()
    ft_started = False
    for hand in hands:
        outs, adjusted = outcomes(hand)
        actual = None if adjusted else outs[0]
        hero_in = not any(a.verb == "fold" and a.name == "Hero" for a in hand.actions)
        if not hand.is_final_stage:
            r["rush_actual"] += hand.final_stacks()["Hero"] - hand.stacks["Hero"]
            r["rush_ev"] += sum(p * s["Hero"] for p, s, _ in outs) - hand.stacks["Hero"]
            r["rush_adjusted"] += adjusted and hero_in
            continue
        if not ft_started:
            ft_started = True
            r["ft"] = 1
            r["v0"] = hero_value(hand, hand.stacks)
        before = hero_value(hand, hand.stacks)
        ev = sum(p * hero_value(hand, s, k) for p, s, k in outs)
        final = hand.final_stacks()
        busted = {x for x in hand.players if final[x] <= 0 and hand.stacks[x] > 0}
        if actual is None:  # the adjusted hand as it really ran out: the full board
            real = outcomes_on_board(hand)
            actual = real if real else (1.0, final, {})
        r["ft_actual"] += hero_value(hand, actual[1], actual[2]) - before
        r["ft_ev"] += ev - before
        r["ft_ev_hero" if hero_in else "ft_ev_others"] += ev - before
        r["ft_actual_hero" if hero_in else "ft_actual_others"] += hero_value(hand, actual[1], actual[2]) - before
        r["kos"] += sum(share for x in busted for w, share in actual[2].get(x, []) if w == "Hero")
        # the run-out on the real board must give the chips of the hand history (an odd chip aside)
        r["check"] = max(r["check"], max(abs(actual[1][n] - final[n]) for n in final))
    return r


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


def cool_down(max_temp=80, resume_temp=75):
    """Wait while the CPU is hotter than max_temp, until it is down to resume_temp (as build_library.py)."""
    t = cpu_temperature()
    if t is None or t < max_temp:
        return
    print(f"  CPU {t:.0f}°C: waiting until {resume_temp}°C", file=sys.stderr, flush=True)
    while (t := cpu_temperature()) is not None and t > resume_temp:
        time.sleep(2)


def stat(values):
    """Mean ± its standard error."""
    m = mean(values)
    se = stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return m, se


def main(hands_dir="hh/hands", summaries_dir="hh/summaries"):
    tours = load_hands(hands_dir)
    summaries = load_summaries(summaries_dir)
    rows = []
    for k, (tid, hands) in enumerate(sorted(tours.items()), 1):
        cool_down()
        r = tournament(hands)
        print(f"  {k}/{len(tours)}", file=sys.stderr, flush=True)
        s = summaries.get(tid)
        r["roi"] = s.received / s.buyin - 1 if s else 0.0
        r["place_prize"] = PLACE_PRIZE_BI.get(s.place, 0) if s else 0
        rows.append(r)
    n = len(rows)
    ft = [r for r in rows if r["ft"]]
    col = lambda key, rs=rows: [r[key] for r in rs]
    fmt = lambda v, unit="", d=3: f"{v[0]:+.{d}f}{unit} ± {v[1]:.{d}f}"
    print(f"Tournaments: {n}, reached the final table: {len(ft)} ({100 * len(ft) / n:.0f}%; average player 50%)")
    bad = sum(r["check"] > 1 for r in rows)
    if bad:
        print(f"  warning: {bad} tournaments where a run-out did not reproduce the chips of the hand history")
    print("\nRush stage (18 -> 9 players), chips per tournament (start 1000, average player -0):")
    print(f"  actual:            {fmt(stat(col('rush_actual')), '', 0)}")
    print(f"  chipEV (all-in adjusted): {fmt(stat(col('rush_ev')), '', 0)}"
          f"   ({sum(col('rush_adjusted'))} of Hero's all-ins adjusted)")
    print("\nFinal table, BI (knockouts at the mean bounty 0.945 BI):")
    print(f"  value at the start:           {stat(col('v0', ft))[0]:.3f} per final table, "
          f"{mean(col('v0')):.3f} per tournament")
    print(f"  actual change:                {fmt(stat(col('ft_actual', ft)))} per final table")
    print(f"  $EV change (all-in adjusted): {fmt(stat(col('ft_ev', ft)))} per final table")
    luck = lambda key: mean(col(f"ft_actual_{key}", ft)) - mean(col(f"ft_ev_{key}", ft))
    print(f"  all-in luck: {luck('hero'):+.3f} in Hero's hands, {luck('others'):+.3f} in the others' (Hero folded)")
    print(f"  knockouts by Hero:            {mean(col('kos', ft)):.2f} per final table")
    print("\nPer tournament, BI (break-even 0):")
    norm = [r["v0"] + r["ft_actual"] - 1 for r in rows]
    ev = [r["v0"] + r["ft_ev"] - 1 for r in rows]
    print(f"  ROI, bounties as drawn:       {fmt(stat(col('roi')))}")
    print(f"  ROI, mean bounties:           {fmt(stat(norm))}")
    print(f"  ROI, mean bounties, final table all-in adjusted: {fmt(stat(ev))}")
    # the Rush chips in money at their average worth: 18 x 0.92 BI of prizes and bounties for 18000 chips
    full = [(1000 + r["rush_ev"]) * 0.92 / 1000 + r["ft_ev"] - 1 for r in rows]
    print(f"  ROI, mean bounties, all-in adjusted throughout (Rush chips at 0.92 mBI): {fmt(stat(full))}")
    check = [r["place_prize"] + r["kos"] * MEAN_BOUNTY_BI - 1 for r in rows]
    print(f"  (place prizes + knockouts x 0.945 - 1 from the summaries: {mean(check):+.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
