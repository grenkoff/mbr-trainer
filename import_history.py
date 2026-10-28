"""Import hand histories and tournament summaries into the database, as the trainer's upload tab does.

    .venv/bin/python import_history.py hh/hands hh/summaries      # files, directories or zip archives
    .venv/bin/python import_history.py --library build/library --pool build/spot_pool   # files kept before

--library moves the solutions of the old build/library/<variant>/<hand_id>.npz files into the database and
--pool the trainer's spots in stock from build/spot_pool/v<version>/<settings>/*.json (both after the hands
whose set-ups they solve).
"""

import argparse
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from mbr import db, importer, library
from trainer.pool import POOL_VERSION


def import_paths(paths):
    files = [f for p in map(Path, paths) for f in (sorted(p.rglob("*")) if p.is_dir() else [p]) if f.is_file()]
    total = {}
    for k, f in enumerate(files, 1):
        r = importer.import_file(f.name, f.read_bytes())
        for key in ("hands", "summaries", "setups", "jobs", "tournaments"):
            total[key] = total.get(key, 0) + r[key]
        total["duplicate"] = total.get("duplicate", 0) + r["duplicate"]
        if k % 50 == 0 or k == len(files):
            print(f"\r  {k}/{len(files)} files", end="", flush=True)
    print(f"\n{len(files)} files: {total.get('hands', 0)} new hands in {total.get('tournaments', 0)} new tournaments, "
          f"{total.get('summaries', 0)} summaries, {total.get('setups', 0)} final-table set-ups, "
          f"{total.get('jobs', 0)} solve jobs queued ({total.get('duplicate', 0)} files uploaded before)")


def import_library(directory):
    with db.connect() as conn:
        setups = {r["hand_id"] for r in conn.execute("SELECT hand_id FROM setups")}
    stored = missing = 0
    for d in sorted(Path(directory).iterdir()):
        for f in sorted(d.glob("*.npz")):
            if f.stem.endswith(".part"):
                continue
            if f.stem not in setups:
                missing += 1
                continue
            library.store(f.stem, d.name, f.read_bytes())
            stored += 1
    with db.connect() as conn:
        done = conn.execute("DELETE FROM solve_jobs j WHERE status = 'queued' AND EXISTS (SELECT 1 FROM solutions s"
                            " WHERE s.hand_id = j.hand_id AND s.variant = j.variant)").rowcount
    print(f"library: {stored} solutions stored, {missing} skipped (set-up not in the database), "
          f"{done} solve jobs no longer needed")


def import_pool(directory):
    d = Path(directory) / f"v{POOL_VERSION}"
    rows = [(key.name, json.loads(f.read_text())) for key in sorted(d.iterdir()) if key.is_dir()
            for f in sorted(key.glob("*.json"))] if d.is_dir() else []
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO spots (pool_version, key, spot) VALUES (%s, %s, %s)",
                            [(POOL_VERSION, key, Jsonb(spot)) for key, spot in rows])
    print(f"pool: {len(rows)} spots of version {POOL_VERSION} stored")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="hand history / summary files, directories or zip archives")
    ap.add_argument("--library", help="the old build/library directory")
    ap.add_argument("--pool", help="the old build/spot_pool directory")
    args = ap.parse_args()
    if args.paths:
        import_paths(args.paths)
    if args.library:
        import_library(args.library)
    if args.pool:
        import_pool(args.pool)


if __name__ == "__main__":
    main()
