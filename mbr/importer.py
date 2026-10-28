"""Import of GG hand history and tournament summary files (plain or zipped) into the database (mbr.db).

Hands and summaries already there are skipped, so the same files can be uploaded again. Every new
final-table set-up goes into the solve queue (mbr.solve_queue) for the variants of mbr.library.variants_for.
"""

import hashlib
import io
import zipfile
from pathlib import PurePath

from mbr import db, library
from mbr.hh import hand_blocks, parse_hand, parse_summary_text
from mbr.real_stacks import setup_of


def texts(name, data):
    """(name, text) of every .txt file in the upload: the file itself, or those in a zip (nested ones too)."""
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if not info.is_dir():
                    yield from texts(f"{name}/{info.filename}", z.read(info))
    elif PurePath(name).suffix.lower() == ".txt":
        yield name, data.decode("utf-8-sig", errors="replace")


def import_file(name, data):
    """Import one uploaded file. Returns what was new: {name, duplicate, files, hands, summaries, setups,
    jobs, tournaments}; a file uploaded before (the same bytes) is skipped as a duplicate."""
    sha = hashlib.sha256(data).hexdigest()
    out = {"name": name, "duplicate": False, "files": 0, "hands": 0, "summaries": 0, "setups": 0, "jobs": 0,
           "tournaments": 0}
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM uploads WHERE sha256 = %s", (sha,)).fetchone():
            return {**out, "duplicate": True}
        touched = set()
        for _, text in texts(name, data):
            out["files"] += 1
            if text.lstrip().startswith("Poker Hand #"):
                _hands(conn, text, out, touched)
            elif (s := parse_summary_text(text)):
                _summary(conn, s, out, touched)
        conn.execute("INSERT INTO uploads (name, sha256, hands, summaries) VALUES (%s, %s, %s, %s)",
                     (name, sha, out["hands"], out["summaries"]))
        # their results are computed again with the new hands or summary
        conn.execute("DELETE FROM tournament_results WHERE tournament_id = ANY(%s)", (list(touched),))
    return out


def _tournament(conn, tid, out):
    cur = conn.execute("INSERT INTO tournaments (id) VALUES (%s) ON CONFLICT DO NOTHING", (tid,))
    out["tournaments"] += cur.rowcount


def _hands(conn, text, out, touched):
    for block in hand_blocks(text):
        hand = parse_hand(block)
        if hand is None:
            continue
        _tournament(conn, hand.tid, out)
        cur = conn.execute("INSERT INTO hands (id, tournament_id, played_at, final_stage, raw) VALUES (%s, %s, %s, %s, %s)"
                           " ON CONFLICT DO NOTHING", (hand.hand_id, hand.tid, hand.ts, hand.is_final_stage, block))
        if not cur.rowcount:
            continue
        out["hands"] += 1
        touched.add(hand.tid)
        if (setup := setup_of(hand)):
            players = len(setup["stacks"])
            conn.execute("INSERT INTO setups (hand_id, players, level, sb, bb, ante, stacks, forced)"
                         " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                         (hand.hand_id, players, setup["level"], setup["sb"], setup["bb"], setup["ante"],
                          setup["stacks"], setup["forced"]))
            out["setups"] += 1
            out["jobs"] += queue_jobs(conn, hand.hand_id, players)


def queue_jobs(conn, hand_id, players):
    """Queue the variants of a set-up not solved yet; returns how many were queued."""
    queued = 0
    for name in library.variants_for(players):
        cur = conn.execute("INSERT INTO solve_jobs (hand_id, variant) SELECT %s, %s WHERE NOT EXISTS"
                           " (SELECT 1 FROM solutions WHERE hand_id = %s AND variant = %s) ON CONFLICT DO NOTHING",
                           (hand_id, name, hand_id, name))
        queued += cur.rowcount
    return queued


def _summary(conn, s, out, touched):
    _tournament(conn, s.tid, out)
    cur = conn.execute("UPDATE tournaments SET buyin = %s, prize_part = %s, rake = %s, bounty_part = %s, place = %s,"
                       " received = %s WHERE id = %s AND place IS NULL",
                       (s.buyin, s.prize_part, s.rake, s.bounty_part, s.place, s.received, s.tid))
    if cur.rowcount:
        out["summaries"] += 1
        touched.add(s.tid)
