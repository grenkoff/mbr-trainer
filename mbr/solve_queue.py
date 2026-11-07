"""Background solving of the queued final-table set-ups (the solve_jobs table of mbr.db) into the library.

The trainer's server runs it: uploads queue their new set-ups (mbr.importer), smaller tables first, and the
worker processes are paused while the CPU is hotter than max_temp, until it is down to resume_temp.
"""

import multiprocessing
import os
import threading
import time
import traceback

from mbr import db
from mbr.temperature import cpu_temperature, guard_temperature

MAX_TEMP, RESUME_TEMP = 80, 75  # °C, the user's thresholds for long runs


def _init_worker():
    os.environ["MBR_THREADS"] = "1"  # one core per worker: the trainer's spot pool uses the others


def _solve(hand_id, name):
    from mbr import library, real_stacks

    snap = next(s for snaps in real_stacks.snapshots().values() for s in snaps if s["hand_id"] == hand_id)
    model, overcalls = library.parse_variant(name)
    start = time.monotonic()
    library.save(library.solve(snap, model, overcalls), snap, model, overcalls)
    return time.monotonic() - start


class SolveQueue:
    def __init__(self, workers=1, max_temp=MAX_TEMP, resume_temp=RESUME_TEMP):
        self.workers = workers
        self.max_temp, self.resume_temp = max_temp, resume_temp
        self.paused = False
        self.running = {}  # (hand_id, variant) -> start time
        self.last_error = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        with db.connect() as conn:  # jobs left running by a server that stopped
            conn.execute("UPDATE solve_jobs SET status = 'queued', started_at = NULL WHERE status = 'running'")
        self.pool = multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker)
        threading.Thread(target=self._loop, daemon=True).start()
        if max_temp and cpu_temperature() is not None:
            threading.Thread(target=guard_temperature, daemon=True,
                             args=(lambda: [p.pid for p in self.pool._pool], max_temp, resume_temp, self.stop,
                                   self._paused)).start()

    def _paused(self, paused, t):
        self.paused = paused

    def notify(self):
        """New jobs were queued."""
        self.wake.set()

    def _claim(self):
        with db.connect() as conn:
            # jobs solved meanwhile by build_library.py
            conn.execute("UPDATE solve_jobs j SET status = 'done', finished_at = now() WHERE status = 'queued' AND"
                         " EXISTS (SELECT 1 FROM solutions s WHERE s.hand_id = j.hand_id AND s.variant = j.variant)")
            return conn.execute(
                "UPDATE solve_jobs SET status = 'running', started_at = now() WHERE (hand_id, variant) = "
                "(SELECT j.hand_id, j.variant FROM solve_jobs j JOIN setups s ON s.hand_id = j.hand_id"
                " WHERE j.status = 'queued' ORDER BY s.players, j.variant LIKE '%%-nooc' DESC, j.queued_at, j.hand_id"
                " LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING hand_id, variant").fetchone()

    def _loop(self):
        while not self.stop.is_set():
            with self.lock:
                free = self.workers - len(self.running)
            job = self._claim() if free > 0 else None
            if job is None:
                self.wake.wait(5)
                self.wake.clear()
                continue
            key = (job["hand_id"], job["variant"])
            with self.lock:
                self.running[key] = time.time()
            self.pool.apply_async(_solve, key, callback=lambda secs, key=key: self._done(key, None),
                                  error_callback=lambda e, key=key: self._done(key, e))

    def _done(self, key, error):
        if error is not None:
            traceback.print_exception(error)
            self.last_error = f"{key[0]} {key[1]}: {error}"
        with db.connect() as conn:
            conn.execute("UPDATE solve_jobs SET status = %s, error = %s, finished_at = now()"
                         " WHERE hand_id = %s AND variant = %s",
                         ("failed" if error else "done", str(error) if error else None, *key))
        with self.lock:
            self.running.pop(key, None)
        self.wake.set()

    def status(self):
        """Jobs by status and player count, the ones running, and whether the workers are paused for heat."""
        with db.connect() as conn:
            rows = conn.execute("SELECT j.status, s.players, count(*) AS n FROM solve_jobs j"
                                " JOIN setups s ON s.hand_id = j.hand_id GROUP BY 1, 2 ORDER BY 2").fetchall()
        by_status = {}
        for r in rows:
            by_status.setdefault(r["status"], {})[r["players"]] = r["n"]
        with self.lock:
            running = [{"hand_id": h, "variant": v, "seconds": round(time.time() - t)}
                       for (h, v), t in self.running.items()]
        return {"jobs": by_status, "running": running, "paused": self.paused, "temperature": cpu_temperature(),
                "max_temp": self.max_temp, "resume_temp": self.resume_temp, "last_error": self.last_error,
                "workers": self.workers, "progress": progress()}


RECENT = 20  # the latest solves a variant's time per set-up is averaged over


def progress():
    """For every player count and variant the library needs: [{players, variant, total, done, failed, seconds}],
    seconds being the wall time per set-up of its latest solves here (pauses for heat included), or None."""
    from mbr import library

    with db.connect() as conn:
        setups = {r["players"]: r["n"] for r in
                  conn.execute("SELECT players, count(*) AS n FROM setups GROUP BY 1")}
        done = {(r["players"], r["variant"]): r["n"] for r in conn.execute(
            "SELECT s.players, x.variant, count(*) AS n FROM solutions x JOIN setups s ON s.hand_id = x.hand_id"
            " GROUP BY 1, 2")}
        failed = {(r["players"], r["variant"]): r["n"] for r in conn.execute(
            "SELECT s.players, j.variant, count(*) AS n FROM solve_jobs j JOIN setups s ON s.hand_id = j.hand_id"
            " WHERE j.status = 'failed' GROUP BY 1, 2")}
        seconds = {(r["players"], r["variant"]): float(r["seconds"]) for r in conn.execute(
            "SELECT players, variant, avg(seconds) AS seconds FROM (SELECT s.players, j.variant,"
            " extract(epoch FROM j.finished_at - j.started_at) AS seconds, row_number() OVER"
            " (PARTITION BY s.players, j.variant ORDER BY j.finished_at DESC) AS k FROM solve_jobs j"
            " JOIN setups s ON s.hand_id = j.hand_id WHERE j.status = 'done' AND j.started_at IS NOT NULL) t"
            " WHERE k <= %s GROUP BY 1, 2", (RECENT,))}
    return [{"players": n, "variant": v, "total": setups[n], "done": done.get((n, v), 0),
             "failed": failed.get((n, v), 0), "seconds": seconds.get((n, v))}
            for n in sorted(setups) for v in library.variants_for(n)]

    def shutdown(self):
        self.stop.set()
        self.wake.set()
        self.pool.terminate()
