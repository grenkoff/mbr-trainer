"""Solve the final-table set-ups from the hand histories in advance for the trainer (mbr.library).

Only set-ups not in the library yet are solved, so new hand histories just add to it.

The solver processes are paused while the CPU is hotter than --max-temp and resumed once it has cooled
to --resume-temp: a long run at full load can overheat a laptop until it switches itself off.

    .venv/bin/python build_library.py --players 3            # HRC model: push/fold with FGS depth 1
    .venv/bin/python build_library.py --players 3 --no-overcalls   # the same, a shove gets one call at most
"""

import argparse
import os
import threading
import time
from multiprocessing import Pool

from mbr import library, real_stacks
from mbr.temperature import cpu_temperature, guard_temperature


def _init_worker():
    os.environ["MBR_THREADS"] = "1"  # the pool already uses every core


def _solve(job):
    snap, model, overcalls = job
    start = time.monotonic()
    library.save(library.solve(snap, model, overcalls), snap, model, overcalls)
    return snap["hand_id"], len(snap["stacks"]), time.monotonic() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", default="3", help="player counts, e.g. 3 or 3,4")
    ap.add_argument("--model", default="hrc", choices=sorted(library.MODELS))
    ap.add_argument("--no-overcalls", dest="overcalls", action="store_false",
                    help="a shove gets one call at most (default: up to three players all-in)")
    ap.add_argument("--workers", type=int, default=None, help="solver processes (default: all cores)")
    ap.add_argument("--max-temp", type=float, default=85, help="pause above this CPU temperature, °C (0: never)")
    ap.add_argument("--resume-temp", type=float, default=75, help="resume at this CPU temperature, °C")
    args = ap.parse_args()
    snaps = real_stacks.snapshots()
    done = library.available(args.model, args.overcalls)
    jobs = [(s, args.model, args.overcalls) for n in map(int, args.players.split(",")) for s in snaps.get(n, [])
            if s["hand_id"] not in done]
    total = sum(len(snaps.get(n, [])) for n in map(int, args.players.split(",")))
    print(f"{library.variant(args.model, args.overcalls)}: {total} set-ups, {total - len(jobs)} already solved, {len(jobs)} to solve", flush=True)
    start = time.monotonic()
    stop = threading.Event()
    with Pool(args.workers, initializer=_init_worker) as pool:
        if args.max_temp and cpu_temperature() is not None:
            def report(paused, t):
                print(f"  {time.strftime('%H:%M:%S')} CPU {t:.0f}°C: "
                      + (f"paused until {args.resume_temp:.0f}°C" if paused else "resumed"), flush=True)
            guard = threading.Thread(target=guard_temperature, daemon=True,
                                     args=(lambda: [p.pid for p in pool._pool], args.max_temp, args.resume_temp, stop,
                                           report))
            guard.start()
        try:
            for k, (hand_id, n, secs) in enumerate(pool.imap_unordered(_solve, jobs), 1):
                elapsed = time.monotonic() - start
                print(f"  {k}/{len(jobs)} {hand_id} {n}p {secs:.0f}s, "
                      f"about {elapsed / k * (len(jobs) - k) / 60:.0f} min left", flush=True)
        finally:
            stop.set()  # the guard resumes the workers it paused
            if args.max_temp and cpu_temperature() is not None:
                guard.join()


if __name__ == "__main__":
    main()
