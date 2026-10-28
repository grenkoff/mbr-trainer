"""All-in adjusted results of Hero's tournaments: chipEV/tour in the Rush stage, $EV at the final table.

The results are computed per tournament by mbr.results (see there how) and kept in the database, so only
tournaments new since the last run are computed.

    .venv/bin/python ev_report.py
"""

import sys
from collections import Counter
from statistics import mean, stdev

from mbr import results
from mbr.pushfold import MEAN_BOUNTY_BI


def stat(values):
    """Mean ± its standard error."""
    m = mean(values)
    se = stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return m, se


def main():
    results.update(results.pending(), log=lambda m: print(f"  {m}", file=sys.stderr, flush=True))
    by_tid = results.load()
    tours = sorted(by_tid)
    rows = [Counter(by_tid[tid]) for tid in tours]
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
    main()
