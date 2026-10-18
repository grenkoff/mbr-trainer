"""Background pool of solved trainer spots.

Worker processes keep solving spots for the settings in use, so a request is answered from stock
instead of waiting for the solver. The stock lives on disk, one JSON file per spot in a directory per
settings, which keeps it between runs and lets `python -m trainer.fill` add to it while the server runs.
"""

import json
import multiprocessing
import os
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

POOL_VERSION = 1  # bump when the solver or the spot format changes: older stock is then ignored
POOL_DIR = Path(__file__).resolve().parent.parent / "build" / "spot_pool" / f"v{POOL_VERSION}"
TARGET = 10  # spots kept ready for each settings in use
WANTED_FOR = 15 * 60  # seconds after the last request that settings count as in use
DEFAULTS = {"players": 5, "position": "any", "kind": "any", "bb_min": 5, "bb_max": 20,
            "difficulty": "medium", "model": "bounty", "overcalls": False}


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
    }


def key_of(p):
    return (f"{p['players']}p_{p['position']}_{p['kind']}_{p['bb_min']:g}-{p['bb_max']:g}bb_"
            f"{p['difficulty']}_{p['model']}_{'oc' if p['overcalls'] else 'nooc'}")


def _init_worker():
    from mbr import equity3
    from mbr.ranges import equity_matrix, pair_weights

    pair_weights(), equity_matrix()
    if equity3.available():
        equity3.order_tensors()


def _solve(params):
    from mbr.trainer import make_spot

    return make_spot(**params)


class SpotPool:
    def __init__(self, workers=None, directory=POOL_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        # one BLAS thread per worker: the workers already use all the cores
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                            initializer=_init_worker)
        self.cond = threading.Condition()
        self.params = {}  # key -> normalized settings
        self.wanted = {}  # key -> (time of the last request, target stock)
        self.running = Counter()  # key -> spots being solved
        self.errors = {}  # key -> error message for the next request

    # --- stock on disk ----------------------------------------------------
    def _files(self, key):
        d = self.dir / key
        return sorted(d.glob("*.json")) if d.is_dir() else []

    def count(self, key):
        return len(self._files(key))

    def _put(self, key, spot):
        d = self.dir / key
        d.mkdir(exist_ok=True)
        name = f"{time.time_ns()}-{os.getpid()}"
        tmp = d / f"{name}.part"
        tmp.write_text(json.dumps(spot))
        tmp.rename(d / f"{name}.json")

    def _take_file(self, key):
        for f in self._files(key):
            claimed = f.with_suffix(".taken")
            try:
                f.rename(claimed)  # atomic: only one reader wins a file
            except FileNotFoundError:
                continue
            spot = json.loads(claimed.read_text())
            claimed.unlink()
            return spot
        return None

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
        while sum(self.running.values()) < self.workers:
            short = []
            for key, (ts, target) in self.wanted.items():
                have = self.count(key) + self.running[key]
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
                spot = self._take_file(key)
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
            return {key: {"stock": self.count(key), "solving": self.running[key]} for key in self.wanted}

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
