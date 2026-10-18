"""Solve trainer spots in advance into the pool, e.g. overnight.

Example:
    .venv/bin/python -m trainer.fill --players 5 --difficulty mix --overcalls --count 300
"""

import argparse
import time

from trainer.pool import SpotPool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, default=5)
    ap.add_argument("--position", default="any")
    ap.add_argument("--kind", default="any", choices=["any", "open", "call", "overcall"])
    ap.add_argument("--bb-min", type=float, default=5)
    ap.add_argument("--bb-max", type=float, default=15)
    ap.add_argument("--difficulty", default="mix", help="natural, easy, medium, hard or mix (the last three)")
    ap.add_argument("--model", default="bounty", choices=["bounty", "bounty-now", "icm"])
    ap.add_argument("--overcalls", action="store_true")
    ap.add_argument("--stacks", default="random", choices=["random", "history"])
    ap.add_argument("--count", type=int, default=100, help="spots to have in stock for each difficulty")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    difficulties = ["easy", "medium", "hard"] if args.difficulty == "mix" else [args.difficulty]
    pool = SpotPool(workers=args.workers)
    keys = []
    for d in difficulties:
        params = {"players": args.players, "position": args.position, "kind": args.kind,
                  "bb_min": args.bb_min, "bb_max": args.bb_max, "difficulty": d, "model": args.model,
                  "overcalls": args.overcalls, "stacks": args.stacks}
        keys.append(pool.want(params, target=args.count))
    start = time.monotonic()
    try:
        while True:
            counts = [pool.count(k) for k in keys]
            print(f"\r{time.monotonic() - start:6.0f}s  " + "  ".join(f"{d}: {c}/{args.count}"
                                                                  for d, c in zip(difficulties, counts)),
                  end="", flush=True)
            if min(counts) >= args.count:
                break
            for k in keys:  # keep the settings wanted while filling
                pool.want(pool.params[k], target=args.count)
            if pool.errors:
                raise SystemExit(f"\n{pool.errors}")
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        pool.shutdown()


if __name__ == "__main__":
    main()
