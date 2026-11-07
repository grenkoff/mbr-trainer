"""Where Hero loses EV: break the decisions scored by compare.py down by situation.

    .venv/bin/python leaks.py build/decisions_bounty.csv
"""

import argparse
import csv
from collections import defaultdict

RANKS = "AKQJT98765432"  # as in mbr.ranges, without its numpy import


def hand_group(label):
    hi, lo = RANKS.index(label[0]), RANKS.index(label[1])
    if hi == lo:
        return "pair 22-66" if hi >= RANKS.index("6") else ("pair 77-TT" if hi >= RANKS.index("T") else "pair JJ+")
    suited = label.endswith("s")
    if hi == 0:
        return "Ax suited" if suited else "Ax offsuit"
    if hi == 1:
        return "Kx suited" if suited else "Kx offsuit"
    if lo <= RANKS.index("T"):
        return "broadway"
    if suited:
        return "suited connector" if lo - hi <= 2 else "other suited"
    return "other offsuit"


def direction(r):
    if r["action"] == r["solver"]:
        return "ok"
    return "too tight" if r["action"] == "fold" else "too loose"


def breakdown(title, rows, key, top=None, min_loss=0.0):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    items = sorted(groups.items(), key=lambda kv: -sum(r["loss"] for r in kv[1]))
    print(f"\n{title}")
    print(f"  {'situation':<40} {'n':>5} {'miss':>5} {'tight':>5} {'loose':>5} {'loss BI':>8} {'per hand':>8}")
    for k, rs in items[:top]:
        loss = sum(r["loss"] for r in rs)
        if loss < min_loss:
            continue
        tight = sum(direction(r) == "too tight" for r in rs)
        loose = sum(direction(r) == "too loose" for r in rs)
        label = " | ".join(str(x) for x in (k if isinstance(k, tuple) else (k,)) if x != "")
        print(f"  {label:<40} {len(rs):>5} {tight + loose:>5} {tight:>5} {loose:>5} {loss:>8.2f} {loss / len(rs):>8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="build/decisions_bounty.csv")
    ap.add_argument("--tournaments", type=int, default=None, help="tournaments played (default: in the file)")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.csv)) if r["action"] != "other"]
    for r in rows:
        r["loss"] = float(r["ev_loss_bi"])
        r["group"] = hand_group(r["class"])
        r["vs"] = r["covers"] if r["kind"] != "open" else ""
    tournaments = args.tournaments or len({r["tid"] for r in rows})
    total = sum(r["loss"] for r in rows)
    mistakes = [r for r in rows if direction(r) != "ok"]
    print(f"{len(rows)} decisions, {len(mistakes)} mistakes ({100 * len(mistakes) / len(rows):.1f}%), "
          f"EV loss {total:.2f} BI = {total / tournaments:.3f} BI per tournament ({tournaments} tournaments)")

    breakdown("By decision and direction", rows, lambda r: (r["kind"], direction(r)))
    breakdown("By decision and position", rows, lambda r: (r["kind"], r["position"]), top=12)
    breakdown("By decision and effective stack", rows, lambda r: (r["kind"], r["bucket"]))
    breakdown("Calls: who covers whom", [r for r in rows if r["kind"] != "open"],
              lambda r: (r["kind"], r["covers"], r["bucket"]), top=10)
    breakdown("By players at the table", rows, lambda r: f"{r['players']} players")
    breakdown("By hand group", rows, lambda r: (r["group"], direction(r) if direction(r) != "ok" else ""), top=14)
    breakdown("Most expensive situations", rows,
              lambda r: (r["kind"], r["position"], r["bucket"], r["vs"]), top=15)

    print("\nMost expensive single decisions")
    for r in sorted(mistakes, key=lambda r: -r["loss"])[:15]:
        vs = f"vs {r['shover']} {r['shover_bb']}bb" if r.get("shover") else ""
        print(f"  {r['hand_id']}: {r['kind']:<8} {r['players']}p {r['position']:<4} {r['hero_bb']:>5}bb {vs:<16} "
              f"{r['class']:<4} played {r['action']:<4} solver {r['solver']:<4} loss {r['loss']:.3f} BI")


if __name__ == "__main__":
    main()
