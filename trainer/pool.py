"""Background pool of solved trainer spots.

Worker processes keep solving spots for the settings in use, so a request is answered from stock
instead of waiting for the solver. The stock lives in the spots table of mbr.db, which keeps it between runs
and lets `python -m trainer.fill` add to it while the server runs.
"""

import multiprocessing
import os
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from psycopg.types.json import Jsonb

from mbr import db

POOL_VERSION = 6  # bump when the solver or the spot format changes: older stock is then ignored
TARGET = 10  # spots kept ready for each settings in use
WANTED_FOR = 15 * 60  # seconds after the last request that settings count as in use
DEFAULTS = {"players": 5, "position": "any", "kind": "any", "bb_min": 5, "bb_max": 20,
            "difficulty": "medium", "model": "bounty", "overcalls": False, "stacks": "random", "forced": False}


def normalize(params):
    p = {**DEFAULTS, **params}
    return {
        "players": int(p["players"]),
        "position": str(p["position"]),
        "kind": str(p["kind"]),
        "bb_min": float(p["bb_min"]),
        "bb_max": float(p["bb_max"]),
        "difficulty": str(p["difficulty"]),
        "model": str(p["model"]),
        "overcalls": str(p["overcalls"]).lower() in ("1", "true", "yes", "on"),
        "stacks": str(p["stacks"]),
        # set-ups with a player all-in from posting: only history stacks have them
        "forced": p["stacks"] == "history" and str(p["forced"]).lower() in ("1", "true", "yes", "on"),
    }


def key_of(p):
    return (f"{p['players']}p_{p['position']}_{p['kind']}_{p['bb_min']:g}-{p['bb_max']:g}bb_"
            f"{p['difficulty']}_{p['model']}_{'oc' if p['overcalls'] else 'nooc'}"
            + ("_history" if p["stacks"] == "history" else "") + ("_forced" if p["forced"] else ""))


def _init_worker():
    from mbr import equity3, real_stacks
    from mbr.ranges import equity_matrix, pair_weights, tie_matrix

    pair_weights(), equity_matrix(), tie_matrix(), real_stacks.snapshots()
    if equity3.available():
        equity3.order_tensors(), equity3.tie_tensors()


def _solve(params):
    from mbr.trainer import make_spot

    return make_spot(**params)


class SpotPool:
    def __init__(self, workers=None):
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        # one BLAS / solver thread per worker: the workers already use all the cores
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MBR_THREADS"] = "1"
        self.executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_init_worker)
        self.cond = threading.Condition()
        self.params = {}  # key -> normalized settings
        self.wanted = {}  # key -> (time of the last request, target stock)
        self.running = Counter()  # key -> spots being solved
        self.errors = {}  # key -> error message for the next request

    # --- stock in the database --------------------------------------------
    def count(self, key):
        with db.connect() as conn:
            return conn.execute("SELECT count(*) AS n FROM spots WHERE pool_version = %s AND key = %s",
                                (POOL_VERSION, key)).fetchone()["n"]

    def counts(self, keys):
        with db.connect() as conn:
            rows = conn.execute("SELECT key, count(*) AS n FROM spots WHERE pool_version = %s AND key = ANY(%s)"
                                " GROUP BY key", (POOL_VERSION, list(keys))).fetchall()
        return Counter({r["key"]: r["n"] for r in rows})

    def _put(self, key, spot):
        with db.connect() as conn:
            conn.execute("INSERT INTO spots (pool_version, key, spot) VALUES (%s, %s, %s)",
                         (POOL_VERSION, key, Jsonb(spot)))

    def _take(self, key):
        with db.connect() as conn:  # SKIP LOCKED: only one reader wins a spot
            row = conn.execute("DELETE FROM spots WHERE id = (SELECT id FROM spots WHERE pool_version = %s AND key = %s"
                               " ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING spot",
                               (POOL_VERSION, key)).fetchone()
        return row["spot"] if row else None

    # --- scheduling -------------------------------------------------------
    def want(self, params, target=TARGET):
        """Keep `target` spots in stock for these settings; returns the key."""
        p = normalize(params)
        key = key_of(p)
        with self.cond:
            self.params[key] = p
            self.wanted[key] = (time.monotonic(), target)
            self._schedule()
        return key

    def _schedule(self):
        now = time.monotonic()
        for key, (ts, _) in list(self.wanted.items()):
            if now - ts > WANTED_FOR and not self.running[key]:
                del self.wanted[key]
        stock = self.counts(self.wanted) if self.wanted else Counter()
        while sum(self.running.values()) < self.workers:
            short = []
            for key, (ts, target) in self.wanted.items():
                have = stock[key] + self.running[key]
                if have < target:
                    short.append((have, -ts, key))  # emptiest first, then the most recently requested
            if not short:
                return
            self._submit(min(short)[2])

    def _submit(self, key):
        self.running[key] += 1
        future = self.executor.submit(_solve, self.params[key])
        future.add_done_callback(lambda f, key=key: self._done(key, f))

    def _done(self, key, future):
        with self.cond:
            self.running[key] -= 1
            try:
                self._put(key, future.result())
            except ValueError as e:
                self.errors[key] = str(e)
                self.wanted.pop(key, None)
            except Exception:
                traceback.print_exc()
                self.errors[key] = "internal error"
                self.wanted.pop(key, None)
            self._schedule()
            self.cond.notify_all()

    # --- requests ---------------------------------------------------------
    def take(self, params, timeout=900):
        """A solved spot for these settings: from stock, or the next one solved."""
        key = self.want(params)
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                spot = self._take(key)
                if spot is not None:
                    spot["pool_left"] = self.count(key)
                    self._schedule()
                    return spot
                if key in self.errors:
                    raise ValueError(self.errors.pop(key))
                if key not in self.wanted:  # dropped after an error that was already reported
                    self.wanted[key] = (time.monotonic(), TARGET)
                    self._schedule()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("the solver did not finish in time")
                self.cond.wait(min(remaining, 1.0))  # also picks up spots added by trainer.fill

    def status(self):
        with self.cond:
            stock = self.counts(self.wanted)
            return {key: {"stock": stock[key], "solving": self.running[key]} for key in self.wanted}

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
