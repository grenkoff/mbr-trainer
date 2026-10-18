"""Solve one final-table push/fold spot and print the ranges.

Example (5 players, stacks in big blinds in preflop order, last two are SB and BB):
    python solve_spot.py 12 8 20 15 10 --ante 0.2
"""

import argparse

from mbr.pushfold import MEAN_BOUNTY_BI, PAYOUTS_BI, Solver, Table
from mbr.ranges import range_grid, range_percent

POSITIONS = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def positions(m):
    return ["SB", "BB"] if m == 2 else POSITIONS[-m:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stacks", nargs="+", type=float, help="stacks in bb, preflop order (... SB BB)")
    ap.add_argument("--ante", type=float, default=0.2, help="ante per player in bb (default 0.2)")
    ap.add_argument("--bounty", type=float, default=MEAN_BOUNTY_BI, help="bounty in buy-ins (0 = plain ICM)")
    ap.add_argument("--no-future-bounty", action="store_true", help="ignore bounties still to be won later")
    ap.add_argument("--payouts", default=",".join(map(str, PAYOUTS_BI)), help="place prizes in buy-ins")
    ap.add_argument("--hero", default=None, help="only print ranges involving this position")
    args = ap.parse_args()

    bb = 1000
    m = len(args.stacks)
    pos = positions(m)
    stacks = [round(s * bb) for s in args.stacks]
    posted = [round(args.ante * bb)] * m
    posted[-1] += bb
    posted[-2] += bb // 2 if m > 1 else 0
    table = Table(names=pos, stacks=stacks, posted=posted, bb=bb, bounty=args.bounty,
                  payouts=[float(x) for x in args.payouts.split(",")], future_bounty=not args.no_future_bounty)
    s = Solver(table).solve()

    for i in range(m - 1):
        if args.hero in (None, pos[i]):
            print(f"\n{pos[i]} ({args.stacks[i]:g}bb) shove when folded to: {range_percent(s.push[i]):.1f}%")
            print(range_grid(s.push[i]))
        for j in range(i + 1, m):
            if args.hero in (None, pos[j]):
                cover = "covers" if stacks[j] >= stacks[i] else "covered"
                print(f"\n{pos[j]} ({args.stacks[j]:g}bb, {cover}) call vs {pos[i]} shove: {range_percent(s.call[i, j]):.1f}%")
                print(range_grid(s.call[i, j]))


if __name__ == "__main__":
    main()
