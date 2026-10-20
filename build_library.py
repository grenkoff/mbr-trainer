"""Solve the final-table set-ups from the hand histories in advance for the trainer (mbr.library).

Only set-ups not in the library yet are solved, so new hand histories just add to it.

    .venv/bin/python build_library.py --players 3            # HRC model: push/fold with FGS depth 1
"""

import argparse
import os
import time
from multiprocessing import Pool

from mbr import library, real_stacks


def _init_worker():
    os.environ["MBR_THREADS"] = "1"  # the pool already uses every core


def _solve(job):
    snap, model = job
    start = time.monotonic()
    library.save(library.solve(snap, model), snap, model)
    return snap["hand_id"], len(snap["stacks"]), time.monotonic() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", default="3", help="player counts, e.g. 3 or 3,4")
    ap.add_argument("--model", default="hrc", choices=sorted(library.MODELS))
    ap.add_argument("--workers", type=int, default=None, help="solver processes (default: all cores)")
    args = ap.parse_args()
    snaps = real_stacks.snapshots()
    done = library.available(args.model)
    jobs = [(s, args.model) for n in map(int, args.players.split(",")) for s in snaps.get(n, [])
            if s["hand_id"] not in done]
    total = sum(len(snaps.get(n, [])) for n in map(int, args.players.split(",")))
    print(f"{args.model}: {total} set-ups, {total - len(jobs)} already solved, {len(jobs)} to solve", flush=True)
    start = time.monotonic()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for k, (hand_id, n, secs) in enumerate(pool.imap_unordered(_solve, jobs), 1):
            elapsed = time.monotonic() - start
            print(f"  {k}/{len(jobs)} {hand_id} {n}p {secs:.0f}s, "
                  f"about {elapsed / k * (len(jobs) - k) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
