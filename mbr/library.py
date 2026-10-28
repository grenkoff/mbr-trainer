"""Solutions of the final-table set-ups from the user's hand histories, solved in advance.

Solving with FGS takes seconds to hours per table, so the trainer reads these instead. The solutions table of
mbr.db holds one npz file per hand and variant with the strategies and the EVs of every decision: variant
<model> with overcalls (at most three players all-in), <model>-nooc without (a shove gets one call at most, in
the next hand of FGS too). The solve queue (mbr.solve_queue) and build_library.py add to it.
"""

import ast
import io

import numpy as np

from mbr import db
from mbr.fgs import MODELS, forced_open, table_for
from mbr.pushfold import Solver

LIBRARY_MODELS = {"hrc"}  # models the trainer only serves from the library
# with overcalls a table takes 20 minutes from 6 players on, days for every player count: up to 5 players only
OVERCALLS_MAX_PLAYERS = 5


def variant(model, overcalls=True):
    """The library directory name of a model with or without overcalls."""
    return model if overcalls else f"{model}-nooc"


def parse_variant(name):
    """(model, overcalls) of a variant name."""
    return (name[:-len("-nooc")], False) if name.endswith("-nooc") else (name, True)


def variants_for(players, model="hrc"):
    """The variants a set-up with this many players is solved in: without overcalls always, with them up to
    OVERCALLS_MAX_PLAYERS."""
    return [variant(model, False)] + ([variant(model, True)] if players <= OVERCALLS_MAX_PLAYERS else [])


def solve(snap, model, overcalls=True):
    """Solve a set-up from mbr.real_stacks with the model's table. The solver's players are in the table's
    order: a player all-in from posting first."""
    table = table_for(snap["stacks"], snap["sb"], snap["bb"], snap["ante"], model, overcalls)
    return Solver(table, overcalls=overcalls, forced_open=forced_open(table)).solve()


def save(solver, snap, model, overcalls=True):
    # not solver.overcalls: heads-up has no overcalls, but goes in both variants
    keys = list(solver.ev)
    buf = io.BytesIO()
    np.savez(buf, keys=np.array([repr(k) for k in keys]),
             strategy=np.array([_strategy(solver, k) for k in keys]),
             ev_play=np.array([solver.ev[k][0] for k in keys]),
             ev_fold=np.array([solver.ev[k][1] for k in keys]))
    store(snap["hand_id"], variant(model, overcalls), buf.getvalue())


def store(hand_id, name, data):
    """Keep the npz bytes of a solution (of variant `name`)."""
    with db.connect() as conn:
        conn.execute("INSERT INTO solutions (hand_id, variant, data) VALUES (%s, %s, %s) ON CONFLICT (hand_id, variant)"
                     " DO UPDATE SET data = EXCLUDED.data, solved_at = now()", (hand_id, name, data))
        # the tournament's decisions are scored against it (mbr.results)
        conn.execute("DELETE FROM tournament_results WHERE tournament_id = (SELECT tournament_id FROM hands WHERE id = %s)",
                     (hand_id,))


def _strategy(solver, key):
    name, *rest = key
    return {"push": solver.push, "call": solver.call, "over": solver.over}[name][rest[0] if name == "push" else tuple(rest)]


def available(model, overcalls=True):
    """Hand ids solved for the model."""
    with db.connect() as conn:
        rows = conn.execute("SELECT hand_id FROM solutions WHERE variant = %s", (variant(model, overcalls),))
        return {r["hand_id"] for r in rows}


def load(snap, model, overcalls=True):
    """A Solver holding the stored strategies and EVs (its table is the plain one: enough to read
    frequencies and lines, not to solve again)."""
    with db.connect() as conn:
        row = conn.execute("SELECT data FROM solutions WHERE hand_id = %s AND variant = %s",
                           (snap["hand_id"], variant(model, overcalls))).fetchone()
    if row is None:
        raise ValueError(f"{snap['hand_id']} is not solved for {variant(model, overcalls)}")
    data = np.load(io.BytesIO(row["data"]))
    plain = "bounty" if MODELS[model].get("bounty") else "icm"
    table = table_for(snap["stacks"], snap["sb"], snap["bb"], snap["ante"], plain)
    table.bunching = False
    solver = Solver(table, overcalls=overcalls, forced_open=forced_open(table))
    solver.ev = {}
    for key_repr, strategy, play, fold in zip(data["keys"], data["strategy"], data["ev_play"], data["ev_fold"]):
        key = ast.literal_eval(str(key_repr))  # tuples like ('call', 0, 2) written by save()
        name, *rest = key
        target = {"push": solver.push, "call": solver.call, "over": solver.over}[name]
        target[rest[0] if name == "push" else tuple(rest)] = strategy
        solver.ev[key] = (play, float(fold))
    return solver
