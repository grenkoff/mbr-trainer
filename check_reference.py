"""Compare the main solver with the slow reference solver (reference/) on real final-table stacks.

The reference is solved on one sample of deals and both solutions are scored on another one, against
opponents who play the reference ranges. For every decision of the main solver the cost is

    sum over hands of (reference play prob - main play prob) * (EV play - EV fold)

per hand dealt, in buy-ins: what a player loses by using the main solver's range there instead of the
reference one. It is noisy but not biased; the ± is its standard error over blocks of deals. A negative
cost means the main range did better than the reference on the fresh deals.

Decisions reached in well under 1% of hands get few deals per hand class: their reference ranges, and
the costs measured against them, are noise.

    .venv/bin/python check_reference.py --players 3,4 --per-count 2
"""

import argparse
import random
import time

import numpy as np

from mbr import real_stacks
from mbr.pushfold import Solver
from mbr.trainer import build_table, positions
from reference.cards import deals
from reference.solver import ReferenceSolver, Spot


def main_play(main, ref):
    """The main solver's strategies on the reference decisions (it folds where it has no decision:
    the fourth and later players all-in)."""
    play = np.zeros((len(ref.decisions), 169))
    for d in ref.decisions:
        if d.shover < 0:
            play[d.index] = main.push[d.player]
        elif not d.callers:
            play[d.index] = main.call[d.shover, d.player]
        elif len(d.callers) == 1:
            play[d.index] = main.over[d.shover, d.callers[0], d.player]
    return play


def describe(d, pos):
    if d.shover < 0:
        return f"{pos[d.player]} shove"
    line = f"{pos[d.player]} vs {pos[d.shover]}"
    return line + (" + " + " + ".join(pos[c] for c in d.callers) if d.callers else "")


def compare(snap, args, solve_deals, score_deals):
    n = len(snap["stacks"])
    pos = positions(n)
    table = build_table(snap["stacks"], snap["sb"], snap["bb"], snap["ante"], args.model)
    start = time.monotonic()
    main = Solver(table, overcalls=True).solve()
    t_main = time.monotonic() - start
    spot = Spot(stacks=list(table.stacks), posted=list(table.posted), payouts=list(table.payouts),
                bounty=table.bounty, future_bounty=table.future_bounty)
    stacks = " ".join(f"{pos[i]} {s / snap['bb']:.1f}" for i, s in enumerate(snap["stacks"]))
    print(f"\n=== {snap['hand_id']}: {n} players, bb: {stacks}", flush=True)
    with ReferenceSolver(spot, solve_deals, workers=args.workers) as ref:
        start = time.monotonic()
        ref.iterate(args.iterations)
        t_ref = time.monotonic() - start
        ref_play, mp = ref.strategy(), main_play(main, ref)
        blocks = ref.action_values(ref_play, score_deals, blocks=True)
        gain_ref = ref.exploitability(ref_play, score_deals)
        gain_main = ref.exploitability(mp, score_deals)
    print(f"main solver {t_main:.1f}s, reference {args.iterations} iterations {t_ref / 60:.1f} min")

    # cost[block, decision] of the main range against the reference opponents
    cost = np.array([((ref_play - mp) * (cfv[:, 0] - cfv[:, 1])).sum(axis=1) for cfv, _ in blocks])
    mean, se = cost.mean(axis=0), cost.std(axis=0, ddof=1) / np.sqrt(len(cost))
    cc = np.array([6 if r == c else 4 if r < c else 12 for r in range(13) for c in range(13)], dtype=float)
    share = lambda play: 100 * play @ cc / cc.sum()
    print(f"  {'decision':<26} {'main %':>7} {'ref %':>7} {'cost, mBI/hand':>18}")
    for d in ref.decisions:
        a, b = share(mp[d.index]), share(ref_play[d.index])
        if max(a, b) < 0.5:
            continue  # neither solver plays it
        print(f"  {describe(d, pos):<26} {a:>7.1f} {b:>7.1f} {1000 * mean[d.index]:>10.3f} ± {1000 * se[d.index]:.3f}")
    per_player = np.array([[c[[d.index for d in ref.decisions if d.player == x]].sum() for x in range(n)]
                           for c in cost])
    pm, ps = per_player.mean(axis=0), per_player.std(axis=0, ddof=1) / np.sqrt(len(cost))
    print("  cost per player, mBI/hand: " + "  ".join(f"{pos[x]} {1000 * pm[x]:.3f}±{1000 * ps[x]:.3f}"
                                                    for x in range(n)))
    print(f"  exploitability on the scoring deals (noise included), mBI/hand: reference {1000 * gain_ref.max():.3f}, "
          f"main {1000 * gain_main.max():.3f}")
    return pm.max(), ps[pm.argmax()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", default="3,4", help="player counts (3-5)")
    ap.add_argument("--per-count", type=int, default=2, help="set-ups per player count")
    ap.add_argument("--hands", default=None, help="comma-separated hand ids instead of random set-ups")
    ap.add_argument("--deals", type=int, default=4_000_000, help="deals in each sample")
    ap.add_argument("--iterations", type=int, default=200, help="CFR+ iterations")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="bounty", choices=["bounty", "icm"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    snaps = real_stacks.snapshots()
    if not snaps:
        raise SystemExit("no hand histories in hh/hands")
    if args.hands:
        wanted = set(args.hands.split(","))
        chosen = [s for v in snaps.values() for s in v if s["hand_id"] in wanted]
    else:
        rng = random.Random(args.seed)
        chosen = [s for n in map(int, args.players.split(","))
                  for s in rng.sample(snaps[n], min(args.per_count, len(snaps[n])))]
    results = []
    for snap in chosen:
        n = len(snap["stacks"])
        solve_deals = deals(n, args.deals, args.seed, workers=args.workers)
        score_deals = deals(n, args.deals, args.seed + 1000, workers=args.workers)
        results.append((snap["hand_id"], n, *compare(snap, args, solve_deals, score_deals)))
    print("\nlargest cost of one player per spot, mBI/hand:")
    for hand_id, n, c, s in results:
        print(f"  {hand_id} {n}p: {1000 * c:.3f} ± {1000 * s:.3f}")


if __name__ == "__main__":
    main()
