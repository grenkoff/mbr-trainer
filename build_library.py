"""Solve the final-table set-ups from the hand histories in advance for the trainer (mbr.library).

Only set-ups not in the library yet are solved, so new hand histories just add to it.

The solver processes are paused while the CPU is hotter than --max-temp and resumed once it has cooled
to --resume-temp: a long run at full load can overheat a laptop until it switches itself off.

    .venv/bin/python build_library.py --players 3            # HRC model: push/fold with FGS depth 1
"""

import argparse
import os
import signal
import threading
import time
from multiprocessing import Pool
from pathlib import Path

from mbr import library, real_stacks


def cpu_temperature():
    """The CPU package temperature in °C (coretemp), or None if there is no such sensor."""
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hwmon / "name").read_text().strip() != "coretemp":
                continue
            for label in hwmon.glob("temp*_label"):
                if label.read_text().startswith("Package"):
                    return int(label.with_name(label.name.replace("label", "input")).read_text()) / 1000
        except OSError:
            continue
    return None


def guard_temperature(pool, max_temp, resume_temp, stop):
    """Pause the pool's processes (SIGSTOP) above max_temp, resume them (SIGCONT) at resume_temp."""
    paused = False
    while not stop.is_set():
        t = cpu_temperature()
        if t is not None:
            pids = [p.pid for p in pool._pool]  # the worker processes
            if not paused and t >= max_temp:
                for pid in pids:
                    os.kill(pid, signal.SIGSTOP)
                paused = True
                print(f"  {time.strftime('%H:%M:%S')} CPU {t:.0f}°C: paused until {resume_temp:.0f}°C", flush=True)
            elif paused and t <= resume_temp:
                for pid in pids:
                    os.kill(pid, signal.SIGCONT)
                paused = False
                print(f"  {time.strftime('%H:%M:%S')} CPU {t:.0f}°C: resumed", flush=True)
        stop.wait(2)


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
    ap.add_argument("--max-temp", type=float, default=85, help="pause above this CPU temperature, °C (0: never)")
    ap.add_argument("--resume-temp", type=float, default=75, help="resume at this CPU temperature, °C")
    args = ap.parse_args()
    snaps = real_stacks.snapshots()
    done = library.available(args.model)
    jobs = [(s, args.model) for n in map(int, args.players.split(",")) for s in snaps.get(n, [])
            if s["hand_id"] not in done]
    total = sum(len(snaps.get(n, [])) for n in map(int, args.players.split(",")))
    print(f"{args.model}: {total} set-ups, {total - len(jobs)} already solved, {len(jobs)} to solve", flush=True)
    start = time.monotonic()
    stop = threading.Event()
    with Pool(args.workers, initializer=_init_worker) as pool:
        if args.max_temp and cpu_temperature() is not None:
            threading.Thread(target=guard_temperature, args=(pool, args.max_temp, args.resume_temp, stop),
                             daemon=True).start()
        try:
            for k, (hand_id, n, secs) in enumerate(pool.imap_unordered(_solve, jobs), 1):
                elapsed = time.monotonic() - start
                print(f"  {k}/{len(jobs)} {hand_id} {n}p {secs:.0f}s, "
                      f"about {elapsed / k * (len(jobs) - k) / 60:.0f} min left", flush=True)
        finally:
            stop.set()
            for p in pool._pool:  # never leave a worker stopped
                try:
                    os.kill(p.pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
