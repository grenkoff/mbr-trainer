"""All-in adjusted results of Hero's tournaments: chipEV/tour in the Rush stage, $EV at the final table.

A hand where the betting ended before the river with every remaining player's cards shown counts at its
expectation over the boards still to come, whoever played it: at the final table the others' all-ins move
Hero's money too. Money is valued as the "bounty" model does (ICM of 4/3/2 BI plus the bounties still to
come split by chips), with every knockout at the mean bounty of 0.945 BI instead of the envelope drawn.

Bounties are paid from 9th place on, so a knockout in the last Rush hands pays one too when the final table starts
with fewer than 9 players. Which of Hero's knockouts there were paid is not in the hand history: it is read from the
bounties of the tournament summary, as the number of envelopes that most likely adds up to them.

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
        s = summaries.get(tid)
        r = tournament(hands, s)
        print(f"  {k}/{len(tours)}", file=sys.stderr, flush=True)
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
    kos, rush_kos = sum(col("kos")), sum(col("rush_kos"))
    print(f"\nKnockouts with a bounty: {kos:g} at the final table, {rush_kos:g} at the end of the Rush stage "
          f"(of Hero's {sum(col('rush_candidates'))} there when it ended below 9 players)")
    print(f"  bounties received: {sum(col('bounties')):.2f} BI, {sum(col('bounties')) / (kos + rush_kos):.3f} "
          f"per knockout (mean bounty {MEAN_BOUNTY_BI})")
    odd = [tid for tid, r in zip(sorted(tours), rows) if r["unexplained"]]
    if odd:
        print(f"  warning: bounties that no envelopes of the knockouts add up to: {', '.join(odd)}")
    print("\nPer tournament, BI (break-even 0):")
    # the bounties of the knockouts at the end of the Rush stage: as they happened, at their all-in expectation
    norm = [r["v0"] + r["ft_actual"] + r["rush_kos"] * MEAN_BOUNTY_BI - 1 for r in rows]
    ev = [r["v0"] + r["ft_ev"] + r["rush_kos_ev"] * MEAN_BOUNTY_BI - 1 for r in rows]
    print(f"  ROI, bounties as drawn:       {fmt(stat(col('roi')))}")
    print(f"  ROI, mean bounties:           {fmt(stat(norm))}")
    print(f"  ROI, mean bounties, final table all-in adjusted: {fmt(stat(ev))}")
    # the Rush chips in money at their average worth: 18 x 0.92 BI of prizes and bounties for 18000 chips
    full = [(1000 + r["rush_ev"]) * 0.92 / 1000 + r["ft_ev"] + r["rush_kos_ev"] * MEAN_BOUNTY_BI - 1 for r in rows]
    print(f"  ROI, mean bounties, all-in adjusted throughout (Rush chips at 0.92 mBI): {fmt(stat(full))}")
    check = [r["place_prize"] + (r["kos"] + r["rush_kos"]) * MEAN_BOUNTY_BI - 1 for r in rows]
    print(f"  (place prizes + knockouts x 0.945 - 1 from the summaries: {mean(check):+.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:])
