"""Compare Hero's final-table push/fold decisions with the solver.

For every final-table hand where Hero is first to act with everyone folded before (shove/fold), faces
a single open shove with everyone in between folded (call/fold), or, with --overcalls, a shove and one
call (overcall/fold), the exact table is rebuilt from the hand, solved, and Hero's action is scored by
the EV difference in buy-ins. `leaks.py` breaks the result down by situation.
"""

import argparse
import csv
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from mbr.db import load_hands
from mbr.pushfold import MEAN_BOUNTY_BI, Solver, table_from_hand
from mbr.ranges import class_label, class_of_cards
from mbr.spots import hero_preflop_spot, stack_bucket
from mbr.trainer import positions

MAX_EFF_BB = 20
MODELS = {
    "bounty": dict(bounty=MEAN_BOUNTY_BI, future_bounty=True),
    "bounty-now": dict(bounty=MEAN_BOUNTY_BI, future_bounty=False),
    "icm": dict(bounty=0.0, future_bounty=False),
}


def candidate(hand, model, overcalls):
    spot = hero_preflop_spot(hand)
    if spot is None or spot.eff_bb > MAX_EFF_BB or not hand.hero_cards:
        return None
    pre = [a for a in hand.actions if a.street == "PRE" and a.verb not in ("ante", "sb", "bb", "show")]
    k = next(i for i, a in enumerate(pre) if a.name == "Hero")
    before = [a for a in pre[:k] if a.verb != "fold"]
    caller = None
    if spot.facing == "unopened":
        kind, shover = "open", None
    elif spot.facing == "allin" and len(before) == 1 and before[0].verb in ("raise", "bet"):
        kind, shover = "call", before[0].name
    elif (overcalls and spot.facing == "allin+call" and len(before) == 2
          and before[0].verb in ("raise", "bet") and before[1].verb == "call"):
        kind, shover, caller = "overcall", before[0].name, before[1].name
    else:
        return None
    table = table_from_hand(hand, **MODELS[model])
    if table is None:
        return None
    return hand, spot, kind, shover, caller, table, overcalls


def _init_worker():
    os.environ["MBR_THREADS"] = "1"  # the pool already uses every core


def score(args):
    hand, spot, kind, shover, caller, table, overcalls = args
    s = Solver(table, overcalls=overcalls).solve()
    h = table.names.index("Hero")
    pos = positions(table.m)
    cls = class_of_cards(hand.hero_cards)
    if kind == "open":
        ev_play, ev_fold = s.ev["push", h]
        freq = s.push[h][cls]
        action = {"allin": "play", "fold": "fold"}.get(spot.hero_action, "other")
    elif kind == "call":
        i = table.names.index(shover)
        ev_play, ev_fold = s.ev["call", i, h]
        freq = s.call[i, h][cls]
        action = {"call": "play", "allin": "play", "fold": "fold"}.get(spot.hero_action, "other")
    else:
        i, j = table.names.index(shover), table.names.index(caller)
        ev_play, ev_fold = s.ev["over", i, j, h]
        freq = s.over[i, j, h][cls]
        action = {"call": "play", "allin": "play", "fold": "fold"}.get(spot.hero_action, "other")
    ev_play = float(ev_play[cls])
    best = "play" if ev_play > ev_fold else "fold"
    loss = 0.0 if action in (best, "other") else abs(ev_play - ev_fold)
    return {
        "tid": hand.tid, "hand_id": hand.hand_id, "kind": kind, "players": table.m,
        "position": spot.position, "eff_bb": round(spot.eff_bb, 1), "bucket": stack_bucket(spot.eff_bb),
        "hero_bb": round(table.stacks[h] / table.bb, 1),
        "shover": pos[table.names.index(shover)] if shover else "",
        "shover_bb": round(table.stacks[table.names.index(shover)] / table.bb, 1) if shover else "",
        "caller": pos[table.names.index(caller)] if caller else "",
        "covers": spot.covers, "cards": hand.hero_cards, "class": class_label(cls),
        "hero_action": spot.hero_action, "action": action, "solver": best, "solver_freq": round(float(freq), 2),
        "ev_play": round(ev_play, 4), "ev_fold": round(ev_fold, 4), "ev_loss_bi": round(loss, 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, default="bounty",
                    help="bounty: ICM + KO bounty + future bounties by chips; bounty-now: ICM + KO bounty; icm: no bounties")
    ap.add_argument("--overcalls", action="store_true", help="solve with overcalls and score overcall spots too")
    ap.add_argument("--workers", type=int, default=None, help="solver processes (default: all cores)")
    args = ap.parse_args()
    out_csv = Path(f"build/decisions_{args.model}{'_overcalls' if args.overcalls else ''}.csv")
    tournaments = load_hands()
    jobs = [c for hands in tournaments.values() for h in hands if h.is_final_stage
            if (c := candidate(h, args.model, args.overcalls))]
    # every scored decision goes to a partial file at once, so an interrupted run resumes where it stopped
    out_csv.parent.mkdir(exist_ok=True)
    partial = out_csv.with_suffix(".partial.csv")
    rows = list(csv.DictReader(partial.open())) if partial.exists() else []
    done = {r["hand_id"] for r in rows}
    todo = [j for j in jobs if j[0].hand_id not in done]
    print(f"Decisions to score: {len(jobs)}, already scored: {len(jobs) - len(todo)}", flush=True)
    with Pool(args.workers, initializer=_init_worker) as pool, partial.open("a", newline="") as f:
        writer = None
        for n, row in enumerate(pool.imap_unordered(score, todo), 1):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not rows:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            rows.append(row)
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    for r in rows:  # rows read back from the partial file are strings
        r["ev_loss_bi"] = float(r["ev_loss_bi"])
        r["solver_freq"] = float(r["solver_freq"])
    order = {j[0].hand_id: k for k, j in enumerate(jobs)}
    rows.sort(key=lambda r: order[r["hand_id"]])
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    partial.unlink()

    scored = [r for r in rows if r["action"] != "other"]
    total_loss = sum(r["ev_loss_bi"] for r in scored)
    errors = [r for r in scored if r["action"] != r["solver"]]
    print(f"Scored push/fold decisions: {len(scored)} (skipped non-all-in raises/limps: {len(rows) - len(scored)})")
    print(f"Mismatches with solver: {len(errors)} ({100 * len(errors) / len(scored):.1f}%), "
          f"total EV loss {total_loss:.2f} BI, {total_loss / len(tournaments):.3f} BI per tournament\n")

    groups = defaultdict(list)
    for r in scored:
        cover = r["covers"] if r["kind"] == "call" else ""
        groups[r["kind"], cover, r["bucket"]].append(r)
    print(f"{'spot':<28} {'n':>4} {'miss':>5} {'too loose':>9} {'too tight':>9} {'EV loss BI':>10}")
    for key, rs in sorted(groups.items(), key=lambda kv: -sum(r["ev_loss_bi"] for r in kv[1])):
        loose = sum(r["action"] == "play" and r["solver"] == "fold" for r in rs)
        tight = sum(r["action"] == "fold" and r["solver"] == "play" for r in rs)
        loss = sum(r["ev_loss_bi"] for r in rs)
        print(f"{' | '.join(k for k in key if k):<28} {len(rs):>4} {loose + tight:>5} {loose:>9} {tight:>9} {loss:>10.2f}")

    print("\nBiggest mistakes:")
    for r in sorted(errors, key=lambda r: -r["ev_loss_bi"])[:20]:
        print(f"  {r['hand_id']}: {r['kind']:<4} {r['players']}p {r['position']:<4} {r['eff_bb']:>5}bb "
              f"{r['covers']:<15} {r['class']:<4} hero {r['hero_action']:<5} solver {r['solver']:<4} "
              f"(freq {r['solver_freq']:.2f}) loss {r['ev_loss_bi']:.3f} BI")
    print(f"\nAll decisions: {out_csv}")


if __name__ == "__main__":
    main()
