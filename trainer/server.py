"""Local web server for the push/fold trainer.

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

from mbr import equity3, library, real_stacks
from trainer.pool import SpotPool

STATIC = Path(__file__).resolve().parent / "static"
SPOT_PARAMS = {"players", "position", "kind", "bb_min", "bb_max", "difficulty", "model", "overcalls", "stacks"}


def library_counts():
    """{model: {players: history set-ups solved in advance}} for the models served from the library."""
    out = {}
    for model in library.LIBRARY_MODELS:
        solved = library.available(model)
        out[model] = {n: sum(s["hand_id"] in solved for s in snaps)
                      for n, snaps in sorted(real_stacks.snapshots().items())}
    return out


class Handler(BaseHTTPRequestHandler):
    pool = None  # SpotPool, set in main()

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

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/info":
            self._json(200, {"overcalls": equity3.available(), "history": real_stacks.counts(),
                             "library": library_counts(), "pool": self.pool.status()})
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
    args = ap.parse_args()
    if not equity3.available():
        print(f"Overcalls are off: {equity3.EQUITY3_FILE} or {equity3.TIES3_FILE.name} is missing (see README)")
    Handler.pool = SpotPool(workers=args.workers)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Trainer running at http://localhost:{args.port}  ({Handler.pool.workers} solver processes, "
          f"Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.pool.shutdown()


if __name__ == "__main__":
    main()
