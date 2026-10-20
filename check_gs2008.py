"""Check the solver against a published equilibrium.

Ganzfried & Sandholm (AAMAS 2008) give the jam/fold equilibrium of a single hand of a cash game:
3 players with 4500 chips each, blinds 300/600, no ante, chip EV (data/gs2008_single_hand.txt).
This solves the same game with overcalls and compares the ranges hand by hand. It also scores both
solutions in our model: the exploitability of the paper's strategies shows how far our model's
equilibrium is from theirs, independently of which hands differ.

The paper's game leaves out the cards of the players who folded, so the comparison does too
(bunching=False); the last section shows how far they move the equilibrium.

    .venv/bin/python check_gs2008.py
"""

import argparse
from pathlib import Path

import numpy as np

from mbr.exploit import exploitability
from mbr.pushfold import Solver, Table
from mbr.ranges import N_CLASSES, class_label, range_percent

DATA = Path(__file__).resolve().parent / "data" / "gs2008_single_hand.txt"
BB = 600
# paper table -> (decision, share of hands jamming published in the paper's Table 2)
NODES = {
    11: (("push", 0), 0.355),
    12: (("call", 0, 1), 0.234),
    13: (("push", 1), 0.633),
    14: (("call", 0, 2), 0.308),
    15: (("call", 1, 2), 0.466),
    16: (("over", 0, 1, 2), 0.197),
}
NAMES = {11: "BTN jam", 12: "SB call vs BTN", 13: "SB jam", 14: "BB call vs BTN", 15: "BB call vs SB",
         16: "BB overcall"}


class ChipTable(Table):
    """Values are the chips each player ends with, in big blinds (a cash game)."""

    def values(self, stacks_after, ko_winners=(), dead_order=None):
        return np.array(stacks_after, dtype=float) / self.bb


def paper_strategies():
    tables, cur = {}, None
    for line in DATA.read_text().splitlines():
        if line.startswith("table "):
            cur = tables[int(line.split()[1].rstrip(":"))] = []
        elif line.strip() and not line.startswith("#"):
            cur.extend(1.0 if x == "P" else 0.0 if x == "F" else int(x) / 100 for x in line.split())
    return {k: np.array(v) for k, v in tables.items()}


def table():
    return ChipTable(names=["BTN", "SB", "BB"], stacks=[4500] * 3, posted=[0, 300, 600], bb=BB)


def strategy(solver, node):
    name, *key = node
    return {"push": solver.push, "call": solver.call, "over": solver.over}[name][key[0] if name == "push" else
                                                                                 tuple(key)]


def with_strategies(strats, bunching):
    s = Solver(table(), overcalls=True, bunching=bunching)
    for t, (node, _) in NODES.items():
        strategy(s, node)[:] = strats[t]
    return s


def report(title, ours, paper):
    print(f"\n=== {title} ===")
    ev = ours.evaluate()
    print(f"{'decision':<16} {'ours %':>7} {'paper %':>7} {'published':>9}  hands that differ")
    for t, (node, published) in NODES.items():
        a, b = strategy(ours, node), strategy(paper, node)
        play, fold = ev[node][0], np.broadcast_to(ev[node][1], N_CLASSES)
        diff = [(abs(play[h] - fold[h]), h) for h in range(N_CLASSES) if abs(a[h] - b[h]) >= 0.5]
        cells = ", ".join(f"{class_label(h)} {'+' if a[h] > b[h] else '-'}{gap:.3f}" for gap, h in sorted(diff)[::-1])
        print(f"{NAMES[t]:<16} {range_percent(a):>7.1f} {range_percent(b):>7.1f} {100 * published:>9.1f}  "
              f"{cells or '-'}")
    print("  (+ = we play the hand and the paper does not, - = the other way; the number is our EV gap "
          "between play and fold, bb)")
    for label, s in (("our strategies", ours), ("the paper's strategies", paper)):
        g = exploitability(s)
        print(f"exploitability of {label} in our model, bb per hand: "
              + "  ".join(f"{p} {x:.5f}" for p, x in zip(table().names, g)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", type=int, default=2000, help="multiway iterations of the long reference run")
    args = ap.parse_args()
    paper = with_strategies(paper_strategies(), bunching=False)
    report("the paper's game (Rust, default schedule)",
           Solver(table(), overcalls=True, bunching=False).solve(), paper)
    report(f"long run ({args.long} multiway iterations, numpy)",
           Solver(table(), overcalls=True, bunching=False).solve_python(iterations=args.long), paper)
    report("with the folded players' cards, as the trainer plays (Rust)", Solver(table(), overcalls=True).solve(),
           with_strategies(paper_strategies(), bunching=True))


if __name__ == "__main__":
    main()
