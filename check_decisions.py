"""Check the main solver's EVs decision by decision with a targeted Monte Carlo (reference/targeted.py).

For every decision of the solved spot it deals only hands in which the decision is reached, with exact
card removal (folded players included) and split pots, and computes the exact EV of both actions for
every hand class while everybody plays the main solver's ranges. Rare decisions get as many deals as
frequent ones, unlike in check_reference.py.

Per decision it prints:
- reach: share of hands dealt in which the decision comes up;
- the main range;
- error: mean |main EV gap - exact EV gap| over the hands reaching the decision (gap = EV of shoving or
  calling - EV of folding), with the mean standard error of the exact gaps next to it: an error about
  as big as the noise means there is no measurable error;
- wrong: hands the main range plays (folds) while the exact gap is negative (positive) by more than
  3 standard errors (2000 hands checked per spot make 2 too loose);
- cost: what those hands lose, in thousandths of a buy-in per hand dealt.

    .venv/bin/python check_decisions.py --hand BR1287900949
    .venv/bin/python check_decisions.py 3.2 17.8 94.3 4.7 --decision "SB vs CO + BTN"
    .venv/bin/python check_decisions.py --players 5,6 --per-count 1 --samples 10000
"""

import argparse
import random
import time

import numpy as np

from mbr import real_stacks
from mbr.pushfold import Solver
from mbr.ranges import class_label, combo_counts
from mbr.trainer import build_table, positions
from reference.solver import Spot
from reference.targeted import Profile, TargetedCheck

CC = combo_counts()


def share(play):
    return float(play @ CC / CC.sum())


def decisions(main):
    """(key in main.evaluate(), node for reference.targeted, play probabilities) of every decision."""
    out = [(("push", i), (i, -1, ()), p) for i, p in main.push.items()]
    out += [(("call", i, j), (j, i, ()), p) for (i, j), p in main.call.items()]
    out += [(("over", i, j, k), (k, i, (j,)), p) for (i, j, k), p in main.over.items()]
    return sorted(out, key=lambda d: (d[1][1] if d[1][1] >= 0 else d[1][0], d[1][1] >= 0, d[1][2], d[1][0]))


def describe(node, pos):
    player, shover, callers = node
    if shover < 0:
        return f"{pos[player]} shove"
    return f"{pos[player]} vs {pos[shover]}" + "".join(f" + {pos[c]}" for c in callers)


def rough_reach(main, node):
    """Share of hands reaching the node from the range sizes alone (no card removal)."""
    player, shover, callers = node
    r = 1.0
    for x in range(player):
        if shover < 0 or x < shover:
            r *= 1 - share(main.push[x])
        elif x == shover:
            r *= share(main.push[x])
        else:
            before = [c for c in callers if c < x]
            p = share(main.call[shover, x] if not before else main.over[shover, before[0], x])
            r *= p if x in callers else 1 - p
    return r


def check(snap, args):
    n = len(snap["stacks"])
    pos = positions(n)
    table = build_table(snap["stacks"], snap["sb"], snap["bb"], snap["ante"], args.model)
    main = Solver(table, overcalls=not args.no_overcalls).solve()
    ev = main.evaluate()
    spot = Spot(stacks=list(table.stacks), posted=list(table.posted), payouts=list(table.payouts),
                bounty=table.bounty, future_bounty=table.future_bounty)
    stacks = " ".join(f"{pos[i]} {s / snap['bb']:.1f}" for i, s in enumerate(snap["stacks"]))
    print(f"\n=== {snap['hand_id']}: {n} players, bb: {stacks}", flush=True)
    chosen = [d for d in decisions(main) if args.decision in (None, describe(d[1], pos))]
    if not chosen:
        raise SystemExit("no such decision; names look like " + ", ".join(
            repr(describe(d[1], pos)) for d in decisions(main)[:4]))
    print(f"  {'decision':<24} {'reach %':>8} {'main %':>7} {'error ± noise, BI':>18} {'wrong':>6} "
          f"{'cost, mBI/hand':>14}")
    total_cost = np.zeros(n)
    worst = []
    with TargetedCheck(spot, Profile(main.push, main.call, main.over), workers=args.workers) as tc:
        for key, node, play in chosen:
            if rough_reach(main, node) < args.min_reach:
                continue
            start = time.monotonic()
            r = tc.run(node, args.samples, args.seed)
            gap, se = r["play"] - r["fold"], r["se"]
            gap_main = ev[key][0] - ev[key][1]
            hands = CC * np.nan_to_num(r["reach"])  # combos reaching the node, per class
            reach = hands.sum() / CC.sum()
            if reach <= 0:
                continue
            ok = hands > 0
            err = np.abs(gap_main - gap)
            wrong = ok & (((play > 0.5) & (gap < -3 * se)) | ((play < 0.5) & (gap > 3 * se)))
            loss = np.where(wrong, ((gap > 0) - play) * gap, 0)
            cost = (hands * loss).sum() / CC.sum()
            total_cost[node[0]] += cost
            mean_err = np.average(err[ok], weights=hands[ok])
            noise = np.average(se[ok], weights=hands[ok])
            print(f"  {describe(node, pos):<24} {100 * reach:>8.3f} {100 * share(play):>7.1f} "
                  f"{mean_err:>9.4f} ± {noise:.4f} {wrong.sum():>6} {1000 * cost:>14.3f}"
                  f"   ({time.monotonic() - start:.0f}s)", flush=True)
            listed = wrong.copy()
            if args.all_hands:
                listed |= ok
            elif args.decision:
                listed |= ok & (err > 3 * se)  # errors clearly beyond the noise
            listed = sorted(np.flatnonzero(listed), key=lambda c: (not wrong[c], -err[c]))
            worst += [(describe(node, pos), class_label(c), play[c], gap[c], se[c], gap_main[c], err[c], wrong[c])
                      for c in listed]
    print("  cost per player, mBI/hand: " + "  ".join(f"{pos[x]} {1000 * total_cost[x]:.3f}" for x in range(n)))
    if worst:
        print(f"\n  {'decision':<24} {'hand':<4} {'main %':>7} {'exact gap':>17} {'main gap':>9} {'error':>7}")
        for name, hand, p, g, s, gm, e, w in worst:
            print(f"  {name:<24} {hand:<4} {100 * p:>7.1f} {g:>+9.4f} ± {s:.4f} {gm:>+9.4f} {e:>7.4f}"
                  + ("  wrong" if w else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stacks", nargs="*", type=float, help="stacks in bb, preflop order (... SB BB)")
    ap.add_argument("--ante", type=float, default=0.2, help="ante per player in bb, with stacks (default 0.2)")
    ap.add_argument("--hand", default=None, help="comma-separated hand ids from hh/hands")
    ap.add_argument("--players", default=None, help="random set-ups from hh/hands with these player counts")
    ap.add_argument("--per-count", type=int, default=1, help="set-ups per player count")
    ap.add_argument("--decision", default=None, help='only this decision, as printed, e.g. "SB vs CO + BTN"')
    ap.add_argument("--all-hands", action="store_true", help="list every hand, not only the wrong ones")
    ap.add_argument("--samples", type=int, default=20_000, help="deals per hand class and decision")
    ap.add_argument("--min-reach", type=float, default=1e-4, help="skip decisions reached less often")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="bounty", choices=["bounty", "icm"])
    ap.add_argument("--no-overcalls", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.stacks:
        bb = 1000
        chosen = [{"hand_id": "stacks", "stacks": [round(s * bb) for s in args.stacks], "sb": bb // 2, "bb": bb,
                   "ante": round(args.ante * bb)}]
    else:
        snaps = real_stacks.snapshots()
        if not snaps:
            raise SystemExit("no hand histories in hh/hands")
        if args.hand:
            wanted = set(args.hand.split(","))
            chosen = [s for v in snaps.values() for s in v if s["hand_id"] in wanted]
        elif args.players:
            rng = random.Random(args.seed)
            chosen = [s for n in map(int, args.players.split(","))
                      for s in rng.sample(snaps[n], min(args.per_count, len(snaps[n])))]
        else:
            raise SystemExit("give stacks, --hand or --players")
    for snap in chosen:
        check(snap, args)


if __name__ == "__main__":
    main()
