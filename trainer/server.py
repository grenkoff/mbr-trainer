"""Local web server: uploads of hand histories, their statistics and the push/fold trainer.

Run:  .venv/bin/python -m trainer [--port 8765]
Then open http://localhost:8765
"""

import argparse
import json
import mimetypes
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from psycopg.types.json import Jsonb

from mbr import db, equity3, importer, library, opponents, real_stacks, results, stats, trainer
from mbr import solve_queue
from mbr.solve_queue import SolveQueue
from trainer.pool import SpotPool

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 512 * 2**20  # bytes
SPOT_PARAMS = {"players", "position", "kind", "bb_min", "bb_max", "difficulty", "model", "overcalls", "stacks",
               "forced"}


def library_counts(forced=False):
    """{model or model-nooc: {players: history set-ups solved in advance}} for the models served from the
    library, with and without overcalls (mbr.library.variant), without (or, if forced, only) those where a
    player is all-in from posting."""
    out = {}
    for model in library.LIBRARY_MODELS:
        for overcalls in (True, False):
            solved = library.available(model, overcalls)
            out[library.variant(model, overcalls)] = {
                n: sum(s["hand_id"] in solved and s["forced"] == forced for s in snaps)
                for n, snaps in sorted(real_stacks.snapshots().items())}
    return out


def uploads_info(queue):
    """What the database holds, the latest uploads and the solve queue."""
    with db.connect() as conn:
        totals = conn.execute(
            "SELECT (SELECT count(*) FROM tournaments) AS tournaments, (SELECT count(*) FROM hands) AS hands,"
            " (SELECT count(*) FROM tournaments WHERE place IS NOT NULL) AS summaries,"
            " (SELECT count(*) FROM setups) AS setups, (SELECT min(played_at) FROM hands) AS first,"
            " (SELECT max(played_at) FROM hands) AS last").fetchone()
        uploads = conn.execute("SELECT name, uploaded_at, hands, summaries FROM uploads ORDER BY id DESC LIMIT 30")
        uploads = [{**u, "uploaded_at": u["uploaded_at"].isoformat()} for u in uploads]
    totals = {k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in totals.items()}
    return {"totals": totals, "uploads": uploads, "queue": queue.status()}


class Handler(BaseHTTPRequestHandler):
    pool = None  # SpotPool, set in main()
    queue = None  # SolveQueue
    results = None  # results.Worker

    def _send(self, code, body, content_type):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json; charset=utf-8")

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            raise ValueError(f"the file is larger than {MAX_UPLOAD // 2**20} MB")
        return self.rfile.read(length)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/upload":
                name = parse_qs(url.query).get("name", ["upload.txt"])[0]
                result = importer.import_file(name, self._body())
                if result["jobs"]:
                    self.queue.notify()
                self.results.notify()
                self._json(200, result)
            elif url.path == "/api/answers":
                records = json.loads(self._body())
                records = records if isinstance(records, list) else [records]
                with db.connect() as conn, conn.cursor() as cur:
                    cur.executemany("INSERT INTO answers (answered_at, record) VALUES (to_timestamp(%s / 1000.0), %s)",
                                    [(r.get("ts") or 0, Jsonb(r)) for r in records])
                self._json(200, {"stored": len(records)})
            else:
                self._send(404, "not found", "text/plain; charset=utf-8")
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "internal error"})

    def do_DELETE(self):
        if urlparse(self.path).path == "/api/answers":
            with db.connect() as conn:
                conn.execute("DELETE FROM answers")
            self._json(200, {})
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/uploads":
            self._json(200, {**uploads_info(self.queue), "results": self.results.status()})
            return
        if url.path == "/api/stats":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                body = stats.summary(results.load(), q.get("from") or None, q.get("to") or None,
                                     float(q["buyin"]) if q.get("buyin") else None)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            self._json(200, {**body, "results": self.results.status()})
            return
        if url.path == "/api/opponents":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                body = stats.opponents_summary(opponents.load(), results.load(), q.get("from") or None,
                                               q.get("to") or None, float(q["buyin"]) if q.get("buyin") else None)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            self._json(200, {**body, "results": self.results.status()})
            return
        if url.path == "/api/decisions":
            # Hero's scored final-table push/fold decisions, only the mistakes unless all=1
            every = parse_qs(url.query).get("all") == ["1"]
            rows = [{**d, "tid": tid, "date": r.get("date"), "buyin": r.get("buyin"), "place": r.get("place")}
                    for tid, r in results.load().items() for d in r.get("decisions", []) if every or d["loss"] > 0]
            progress = solve_queue.progress()
            self._json(200, {"decisions": rows, "results": self.results.status(),
                             "solved": sum(g["done"] for g in progress), "needed": sum(g["total"] for g in progress)})
            return
        if url.path == "/api/hand":
            detail = results.hand_detail(parse_qs(url.query).get("id", [""])[0])
            if detail is None:
                self._json(404, {"error": "no such hand"})
            else:
                self._json(200, detail)
            return
        if url.path == "/api/hand_spot":
            try:
                self._json(200, trainer.hand_spot(parse_qs(url.query).get("id", [""])[0]))
            except ValueError as e:
                self._json(404, {"error": str(e)})
            return
        if url.path == "/api/answers":
            with db.connect() as conn:
                rows = conn.execute("SELECT record FROM answers ORDER BY answered_at, id").fetchall()
            self._json(200, [r["record"] for r in rows])
            return
        if url.path == "/api/info":
            self._json(200, {"overcalls": equity3.available(), "history": real_stacks.counts(),
                             "history_forced": real_stacks.counts(forced=True), "library": library_counts(),
                             "library_forced": library_counts(forced=True), "pool": self.pool.status()})
            return
        if url.path == "/api/spot":
            params = {k: v[0] for k, v in parse_qs(url.query).items() if k in SPOT_PARAMS}
            try:
                self._json(200, self.pool.take(params))
            except (ValueError, TimeoutError) as e:
                self._json(400, {"error": str(e)})
            except Exception:
                traceback.print_exc()
                self._json(500, {"error": "internal error"})
            return
        rel = "index.html" if url.path in ("", "/") else url.path.lstrip("/")
        path = (STATIC / rel).resolve()
        if STATIC not in path.parents or not path.is_file():
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, path.read_bytes(), ctype)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Mystery Battle Royale push/fold trainer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workers", type=int, default=None, help="solver processes (default: CPU cores - 1)")
    ap.add_argument("--solve-workers", type=int, default=1,
                    help="processes solving the uploaded final-table set-ups in the background")
    args = ap.parse_args()
    if not equity3.available():
        print(f"Overcalls are off: {equity3.EQUITY3_FILE} or {equity3.TIES3_FILE.name} is missing (see README)")
    Handler.pool = SpotPool(workers=args.workers)
    Handler.queue = SolveQueue(workers=args.solve_workers)
    Handler.results = results.Worker()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Trainer running at http://localhost:{args.port}  ({Handler.pool.workers} solver processes, "
          f"Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.pool.shutdown()
        Handler.queue.shutdown()
        Handler.results.shutdown()


if __name__ == "__main__":
    main()
