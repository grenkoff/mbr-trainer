"""Solutions of the final-table set-ups from the user's hand histories, solved in advance.

Solving with FGS takes seconds to hours per table, so the trainer reads these instead. One file per
hand and model in build/library/<model>/<hand_id>.npz holds the strategies and the EVs of every decision,
with overcalls (at most three players all-in); build/library/<model>-nooc/ holds them without (a shove
gets one call at most, in the next hand of FGS too). Build or extend it with build_library.py.
"""

import ast
from pathlib import Path

import numpy as np

from mbr.fgs import MODELS, forced_open, table_for
from mbr.pushfold import Solver

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "build" / "library"
LIBRARY_MODELS = {"hrc"}  # models the trainer only serves from the library


def variant(model, overcalls=True):
    """The library directory name of a model with or without overcalls."""
    return model if overcalls else f"{model}-nooc"


def path(model, hand_id, overcalls=True):
    return LIBRARY_DIR / variant(model, overcalls) / f"{hand_id}.npz"


def solve(snap, model, overcalls=True):
    """Solve a set-up from mbr.real_stacks with the model's table. The solver's players are in the table's
    order: a player all-in from posting first."""
    table = table_for(snap["stacks"], snap["sb"], snap["bb"], snap["ante"], model, overcalls)
    return Solver(table, overcalls=overcalls, forced_open=forced_open(table)).solve()


def save(solver, snap, model, overcalls=True):
    # not solver.overcalls: heads-up has no overcalls, but goes in both variants
    p = path(model, snap["hand_id"], overcalls)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(solver.ev)
    tmp = p.with_suffix(".part.npz")
    np.savez(tmp, keys=np.array([repr(k) for k in keys]),
             strategy=np.array([_strategy(solver, k) for k in keys]),
             ev_play=np.array([solver.ev[k][0] for k in keys]),
             ev_fold=np.array([solver.ev[k][1] for k in keys]))
    tmp.rename(p)


def _strategy(solver, key):
    name, *rest = key
    return {"push": solver.push, "call": solver.call, "over": solver.over}[name][rest[0] if name == "push" else tuple(rest)]


def available(model, overcalls=True):
    """Hand ids solved for the model."""
    d = LIBRARY_DIR / variant(model, overcalls)
    return {f.stem for f in d.glob("*.npz") if not f.stem.endswith(".part")} if d.is_dir() else set()


def load(snap, model, overcalls=True):
    """A Solver holding the stored strategies and EVs (its table is the plain one: enough to read
    frequencies and lines, not to solve again)."""
    data = np.load(path(model, snap["hand_id"], overcalls))
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
