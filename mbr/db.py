"""PostgreSQL store (docker-compose.yml): the hand histories uploaded, the final-table set-ups taken from them,
the solver library, the solve queue, the trainer's spot pool and its answers.

The connection string comes from MBR_DATABASE_URL (default: the docker-compose database). The schema is
created on the first connection of a process.
"""

import os
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row

from mbr.hh import Summary, parse_hand

DSN = os.environ.get("MBR_DATABASE_URL", "postgresql://mbr:mbr@localhost:5433/mbr")

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id serial PRIMARY KEY,
    name text NOT NULL,
    sha256 text NOT NULL UNIQUE,  -- a file uploaded again is skipped
    uploaded_at timestamptz NOT NULL DEFAULT now(),
    hands int NOT NULL DEFAULT 0,  -- hands new to the database
    summaries int NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tournaments (
    id text PRIMARY KEY,  -- the GG tournament number
    -- from the tournament summary, null until it is uploaded
    buyin numeric, prize_part numeric, rake numeric, bounty_part numeric, place int, received numeric
);
CREATE TABLE IF NOT EXISTS hands (
    id text PRIMARY KEY,
    tournament_id text NOT NULL REFERENCES tournaments,
    played_at timestamp NOT NULL,
    final_stage boolean NOT NULL,
    raw text NOT NULL  -- parsed again with mbr.hh when read
);
CREATE INDEX IF NOT EXISTS hands_tournament ON hands (tournament_id, played_at, id);
-- the final-table set-ups that fit the push/fold model (mbr.real_stacks)
CREATE TABLE IF NOT EXISTS setups (
    hand_id text PRIMARY KEY REFERENCES hands,
    players int NOT NULL,
    level int NOT NULL, sb int NOT NULL, bb int NOT NULL, ante int NOT NULL,
    stacks int[] NOT NULL,  -- before posting, in preflop order
    forced boolean NOT NULL  -- a player is all-in from posting
);
-- solutions of the set-ups (mbr.library), an npz file each
CREATE TABLE IF NOT EXISTS solutions (
    hand_id text NOT NULL REFERENCES setups,
    variant text NOT NULL,  -- mbr.library.variant(): model, with or without overcalls
    data bytea NOT NULL,
    solved_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (hand_id, variant)
);
CREATE TABLE IF NOT EXISTS solve_jobs (
    hand_id text NOT NULL REFERENCES setups,
    variant text NOT NULL,
    status text NOT NULL DEFAULT 'queued',  -- queued, running, done, failed
    error text,
    queued_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    PRIMARY KEY (hand_id, variant)
);
-- the trainer's solved spots in stock (trainer.pool)
CREATE TABLE IF NOT EXISTS spots (
    id bigserial PRIMARY KEY,
    pool_version int NOT NULL,
    key text NOT NULL,
    spot jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS spots_key ON spots (pool_version, key, id);
-- the trainer's answers, as the page records them
CREATE TABLE IF NOT EXISTS answers (
    id bigserial PRIMARY KEY,
    answered_at timestamptz NOT NULL,
    record jsonb NOT NULL
);
-- per-tournament results (mbr.results), kept until their version changes or the tournament gets new data
CREATE TABLE IF NOT EXISTS tournament_results (
    tournament_id text PRIMARY KEY REFERENCES tournaments,
    version int NOT NULL,
    data jsonb NOT NULL
);
-- the same for the opponents' play (mbr.opponents)
CREATE TABLE IF NOT EXISTS opponent_results (
    tournament_id text PRIMARY KEY REFERENCES tournaments,
    version int NOT NULL,
    data jsonb NOT NULL
);
"""


@lru_cache(maxsize=1)
def _schema():
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(SCHEMA)
    return True


def connect(**kwargs):
    """A new connection with dict rows; as a context manager it commits on exit."""
    _schema()
    return psycopg.connect(DSN, row_factory=dict_row, **kwargs)


def load_hands(tournament_ids=None):
    """{tournament id: [hands in chronological order]}, as mbr.hh.load_hands for a directory."""
    with connect() as conn:
        rows = conn.execute("SELECT tournament_id, raw FROM hands"
                            + (" WHERE tournament_id = ANY(%s)" if tournament_ids is not None else "")
                            + " ORDER BY tournament_id, played_at, id",
                            [list(tournament_ids)] if tournament_ids is not None else None).fetchall()
    out = {}
    for r in rows:
        if (h := parse_hand(r["raw"])):
            out.setdefault(r["tournament_id"], []).append(h)
    return out


def load_summaries():
    """{tournament id: Summary} of the tournaments whose summary was uploaded."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM tournaments WHERE place IS NOT NULL").fetchall()
    return {r["id"]: Summary(tid=r["id"], buyin=float(r["buyin"]), prize_part=float(r["prize_part"]),
                             rake=float(r["rake"]), bounty_part=float(r["bounty_part"]), place=r["place"],
                             received=float(r["received"])) for r in rows}
