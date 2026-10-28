"""Print structure stats and the most frequent final-table preflop spots from hand histories."""

import sys
from collections import Counter, defaultdict
from statistics import mean, median

from mbr.db import load_hands, load_summaries
from mbr.spots import PLACE_PRIZE_BI, hero_knockouts, hero_preflop_spot, stack_bucket, structure_stats


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def main(top=40):
    tournaments = load_hands()
    summaries = load_summaries()
    n_hands = sum(map(len, tournaments.values()))
    print(f"Tournaments: {len(tournaments)}, hands: {n_hands}, summaries: {len(summaries)}\n")

    # --- structure ---
    st = structure_stats(tournaments)
    print("Median level duration (min):", ", ".join(f"L{l}:{m:.1f}" for l, m in st["level_minutes"].items()))
    print("Rush stage length, min: median %.1f" % median(st["rush_minutes"]))
    print("Final table starts at level:")
    for (lvl, sb, bb, ante), c in sorted(st["ft_start_levels"].items()):
        print(f"  L{lvl} {sb}/{bb} ante {ante}: {c}")
    bbs = sorted(st["ft_start_bb"])
    q = lambda p: bbs[int(p * (len(bbs) - 1))]
    print(f"Stacks at FT start (bb): p10 {q(.1):.1f}, p25 {q(.25):.1f}, median {q(.5):.1f}, p75 {q(.75):.1f}, p90 {q(.9):.1f}")
    print(f"Hero stack at FT start: median {median(st['ft_start_hero_bb']):.1f}bb\n")

    # --- results & bounties ---
    places = Counter(s.place for s in summaries.values())
    reached_ft = sum(c for p, c in places.items() if p <= 9)
    buyin = next(iter(summaries.values())).buyin
    total_bounty = sum(s.received - PLACE_PRIZE_BI.get(s.place, 0) * buyin for s in summaries.values())
    total_received = sum(s.received for s in summaries.values())
    kos = sum(hero_knockouts(tournaments[tid]) for tid in summaries if tid in tournaments)
    print(f"Reached FT: {reached_ft}/{len(summaries)} ({pct(reached_ft, len(summaries))})")
    print(f"Places 1-3: {places[1]}/{places[2]}/{places[3]}")
    print(f"Total buy-ins ${len(summaries) * buyin:.2f}, received ${total_received:.2f}, "
          f"of which bounties ${total_bounty:.2f}")
    print(f"Hero FT knockouts (from hands): {kos}; bounty per KO ${total_bounty / kos:.3f} "
          f"(expected {0.945 * buyin:.3f})\n" if kos else "")

    # --- spots ---
    spots = [s for hands in tournaments.values() for h in hands if h.is_final_stage
             if (s := hero_preflop_spot(h))]
    print(f"Hero FT preflop decisions: {len(spots)}")
    eff = Counter(stack_bucket(s.eff_bb) for s in spots)
    print("Effective stack distribution:", ", ".join(f"{k} {pct(v, len(spots))}" for k, v in
                                                     sorted(eff.items(), key=lambda kv: float(kv[0].split('-')[0].rstrip('+bb')))))
    print("Facing:", ", ".join(f"{k} {pct(v, len(spots))}" for k, v in Counter(s.facing for s in spots).most_common()))
    print("Players at table:", ", ".join(f"{k}: {pct(v, len(spots))}" for k, v in sorted(Counter(s.n_players for s in spots).items())))
    print()

    def table(title, keyfn, rows=top):
        groups = defaultdict(list)
        for s in spots:
            groups[keyfn(s)].append(s)
        print(f"== {title}")
        print(f"{'spot':<52} {'n':>5} {'share':>6}  actions")
        for key, ss in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:rows]:
            acts = Counter(s.hero_action for s in ss)
            act_str = " ".join(f"{a}:{pct(c, len(ss))}" for a, c in acts.most_common())
            print(f"{' | '.join(map(str, key)):<52} {len(ss):>5} {pct(len(ss), len(spots)):>6}  {act_str}")
        print()

    table("Facing x effective stack", lambda s: (s.facing, stack_bucket(s.eff_bb)))
    table("Facing x cover x effective stack (non-unopened)",
          lambda s: (s.facing, s.covers, stack_bucket(s.eff_bb)) if s.facing != "unopened" else ("unopened",))
    table("Players x position x facing x stack", lambda s: (f"{s.n_players}p", s.position, s.facing, stack_bucket(s.eff_bb)))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:2]))
