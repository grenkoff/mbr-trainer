"""Random final-table push/fold spots for the trainer, solved and packaged with the answer."""

import random

import numpy as np

from mbr import equity3, library, real_stacks
from mbr.fgs import MODELS, action_order, forced_open, preflop_seats, table_for
from mbr.pushfold import Solver
from mbr.ranges import class_combos, class_label, combo_counts

TOTAL_CHIPS = 18000
# (level, sb, bb, ante) of the Mystery Battle Royale structure from the final stage on
LEVELS = [
    (4, 40, 80, 16), (5, 50, 100, 20), (6, 60, 120, 24), (7, 75, 150, 30), (8, 100, 200, 40),
    (9, 150, 300, 60), (10, 200, 400, 80), (11, 300, 600, 120), (12, 400, 800, 160), (13, 500, 1000, 200),
]
POSITIONS = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
# cost of the mistake |EV(play) - EV(fold)| in buy-ins
DIFFICULTY_BANDS = {"hard": (0.0, 0.03), "medium": (0.03, 0.10), "easy": (0.10, 100.0)}
DIFFICULTIES = ["natural", *DIFFICULTY_BANDS]
MIN_STACK_BB = 1.5
MAX_STACK_BB = 70
AVG_STACK_BB = (4, 30)  # realistic average stack depth for the chosen level


def positions(n):
    return POSITIONS[-n:]


def random_stacks(n, hero_idx, hero_bb, rng):
    """Pick a blind level and stacks summing to TOTAL_CHIPS with Hero at about hero_bb."""
    for _ in range(200):
        level, sb, bb, ante = rng.choice(LEVELS)
        hero = round(hero_bb * bb)
        rest = TOTAL_CHIPS - hero
        if hero < MIN_STACK_BB * bb or rest < (n - 1) * MIN_STACK_BB * bb:
            continue
        avg = TOTAL_CHIPS / n / bb
        if not AVG_STACK_BB[0] <= avg <= AVG_STACK_BB[1]:
            continue
        shares = np.array([rng.gammavariate(1.6, 1.0) for _ in range(n - 1)])
        others = np.floor(shares / shares.sum() * rest).astype(int)
        others[0] += rest - others.sum()
        if others.min() < MIN_STACK_BB * bb or others.max() > MAX_STACK_BB * bb:
            continue
        stacks = list(others)
        stacks.insert(hero_idx, hero)
        return (level, sb, bb, ante), [int(s) for s in stacks]
    raise ValueError("could not generate stacks for these settings")


def build_table(stacks, sb, bb, ante, model, overcalls=True):
    return table_for(stacks, sb, bb, ante, model, overcalls)


def pick_hand(ev_diff, difficulty, rng, strict=True):
    """Pick a hand class whose mistake cost matches the difficulty, or None if strict and there is none.

    "natural" deals every one of the 1326 combos with equal probability, as at the table.
    """
    cc = combo_counts()
    if difficulty == "natural":
        return int(rng.choices(range(len(cc)), weights=list(cc))[0])
    cost = np.abs(ev_diff)
    lo, hi = DIFFICULTY_BANDS[difficulty]
    mask = (cost >= lo) & (cost < hi)
    if not mask.any():
        if strict:
            return None
        gap = np.maximum(np.maximum(lo - cost, cost - hi), 0)  # the hands closest to the band
        mask = gap <= np.sort(gap)[5]
    candidates = np.flatnonzero(mask)
    weights = cc[candidates]
    return int(rng.choices(list(candidates), weights=list(weights))[0])


def pick_cards(cls, rng):
    c1, c2 = rng.choice(class_combos(cls))
    return [c1, c2]


MAX_ATTEMPTS = 8
LINE_ATTEMPTS = 3


def make_spot(players=5, position="any", kind="any", bb_min=5, bb_max=20, difficulty="medium",
              model="bounty", overcalls=False, stacks="random", forced=False, seed=None):
    """Generate spots until one has a hand of the requested difficulty.

    stacks: "random" stacks at a random blind level, or "history" for the blinds and stacks of a real
    final-table hand from the user's hand histories, with Hero in any seat that fits the settings.
    model: see mbr.fgs.MODELS; the history spots of "hrc" (as HoldemResources Calculator: FGS) come from
    the library solved in advance (mbr.library), with or without overcalls.
    forced: history set-ups where a player is all-in from posting are dealt too. He is all-in before
    anybody acts, so Hero calls him (or a caller over him) or folds.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    overcalls = str(overcalls).lower() in ("1", "true", "yes", "on")
    forced = str(forced).lower() in ("1", "true", "yes", "on")
    if overcalls and not equity3.available():
        raise ValueError("overcalls need the 3-way equity tables: build/equity3.f32 and equity3_ties.f32 (see README)")
    if kind == "overcall" and not overcalls:
        raise ValueError("overcall spots need overcalls enabled")
    if not 3 <= int(players) <= len(POSITIONS):
        raise ValueError(f"players must be 3..{len(POSITIONS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty}")
    if stacks not in ("random", "history"):
        raise ValueError(f"unknown stacks source {stacks}")
    if stacks == "history" and not real_stacks.snapshots():
        raise ValueError(f"no hand histories in {real_stacks.HANDS_DIR}")
    rng = random.Random(seed)
    for attempt in range(MAX_ATTEMPTS):
        spot = _make_spot(rng, players, position, kind, bb_min, bb_max, difficulty, model, overcalls, stacks,
                          forced, strict=attempt < MAX_ATTEMPTS - 1)
        if spot:
            return spot


def _kind_fits(kind, i, n):
    return {"open": i < n - 1, "call": i > 0, "overcall": i > 1}[kind]


def pick_line(solver, hero_idx, kind, rng):
    """Choose (shover, caller) before Hero in proportion to how often equilibrium play produces the line."""
    cc = combo_counts()
    freq = lambda strategy: float(strategy @ cc / cc.sum())
    cp = {key: solver._caller_stats(*key)[0] for key in solver.call}
    lines, weights, reach = [], [], 1.0
    for i in range(hero_idx):
        ppush = freq(solver.push[i])
        folds = 1.0  # players after the shover fold before the caller (or Hero)
        for j in range(i + 1, hero_idx):
            if kind == "overcall":
                # and the players between the caller and Hero do not overcall
                over_folds = np.prod([1 - freq(solver.over[i, j, k]) for k in range(j + 1, hero_idx)])
                lines.append((i, j))
                weights.append(reach * ppush * folds * cp[i, j] * over_folds)
            folds *= 1 - cp[i, j]
        if kind == "call":
            lines.append((i, None))
            weights.append(reach * ppush * folds)
        reach *= 1 - ppush
    if sum(weights) <= 0:
        return rng.choice(lines)
    return rng.choices(lines, weights=weights)[0]


def _pick_decision(solver, hero_idx, kind, difficulty, rng, strict):
    """A line before Hero and a hand of the requested difficulty, or None."""
    for _ in range(1 if kind == "open" else LINE_ATTEMPTS):
        shover = caller = None
        if kind == "open":
            ev_play, ev_fold = solver.ev["push", hero_idx]
            freq = solver.push[hero_idx]
        elif kind == "call":
            shover, _ = pick_line(solver, hero_idx, kind, rng)
            ev_play, ev_fold = solver.ev["call", shover, hero_idx]
            freq = solver.call[shover, hero_idx]
        else:
            shover, caller = pick_line(solver, hero_idx, kind, rng)
            ev_play, ev_fold = solver.ev["over", shover, caller, hero_idx]
            freq = solver.over[shover, caller, hero_idx]
        ev_diff = np.asarray(ev_play) - ev_fold
        cls = pick_hand(ev_diff, difficulty, rng, strict)
        if cls is not None:
            return shover, caller, ev_play, ev_fold, ev_diff, freq, cls
    return None


def fits(kind, seat, order, forced, covered=()):
    """Whether Hero in preflop seat `seat` can face `kind`; `order` holds the seats in the solver's order.
    With a player all-in from posting (forced, first in `order`) Hero can only call him, or overcall him
    and a caller; and not call him from a seat in `covered`, where there is nothing to decide."""
    if not forced:
        return _kind_fits(kind, seat, len(order))
    k = order.index(seat)
    return {"open": False, "call": k > 0 and seat not in covered, "overcall": k > 1}[kind]


def covered_seats(stacks, sb, bb, ante, order):
    """The seats whose ante and blind already match the player all-in from posting (first in `order`):
    if everybody before them folds, they just see the showdown."""
    posted = [ante] * len(stacks)
    posted[-2] += sb
    posted[-1] += bb
    return {i for i in range(len(stacks)) if i != order[0] and posted[i] >= stacks[order[0]]}


def history_setup(n, hero_options, bb_min, bb_max, rng, solved=None, forced=False):
    """A real final-table set-up with n players where Hero can sit in a seat that fits the settings
    (among the hand ids in `solved`, if given; with those where a player is all-in from posting if forced)."""
    fitting = []
    for snap in real_stacks.snapshots().get(n, []):
        if solved is not None and snap["hand_id"] not in solved or snap["forced"] and not forced:
            continue
        order = action_order(snap["stacks"], snap["sb"], snap["bb"], snap["ante"])[0]
        options = [(i, k) for i, k in hero_options if bb_min <= snap["stacks"][i] / snap["bb"] <= bb_max
                   and fits(k, i, order, snap["forced"], covered_seats(snap["stacks"], snap["sb"], snap["bb"],
                                                                       snap["ante"], order))]
        if options:
            fitting.append((snap, options))
    if not fitting:
        where = " solved for this model so far" if solved is not None else ""
        raise ValueError(f"no {n}-player hands in the history{where} with a stack of {bb_min:g}-{bb_max:g} bb "
                         "in the chosen position")
    snap, options = rng.choice(fitting)
    return snap, rng.choice(options)


def _make_spot(rng, players, position, kind, bb_min, bb_max, difficulty, model, overcalls, stacks_source,
               forced, strict):
    n = int(players)
    pos = positions(n)
    if kind == "any":
        allowed_kinds = ["open", "call"] + (["overcall"] if overcalls else [])
    else:
        allowed_kinds = [kind]
    # Hero's seats and decisions; whether one can come up depends on the set-up (see fits())
    hero_options = [(i, k) for i, p in enumerate(pos) if position in ("any", p) for k in allowed_kinds]
    if not any(_kind_fits(k, i, n) for i, k in hero_options) and not (forced and stacks_source == "history"):
        raise ValueError(f"position {position} cannot face '{kind}' with {n} players")
    source = {"stacks": stacks_source}
    from_library = model in library.LIBRARY_MODELS and stacks_source == "history"
    if stacks_source == "history":
        solved = library.available(model, overcalls) if from_library else None
        snap, (hero_idx, kind) = history_setup(n, hero_options, float(bb_min), float(bb_max), rng, solved, forced)
        level, sb, bb, ante, stacks = snap["level"], snap["sb"], snap["bb"], snap["ante"], list(snap["stacks"])
        source["hand_id"] = snap["hand_id"]
    else:
        hero_idx, kind = rng.choice([(i, k) for i, k in hero_options if _kind_fits(k, i, n)])
        hero_bb = rng.uniform(float(bb_min), float(bb_max))
        (level, sb, bb, ante), stacks = random_stacks(n, hero_idx, hero_bb, rng)
    if from_library:
        solver = library.load(snap, model, overcalls)
    else:
        table = build_table(stacks, sb, bb, ante, model, overcalls)
        solver = Solver(table, overcalls=overcalls, forced_open=forced_open(table)).solve()
    # the solver's players are the preflop seats in this order: a player all-in from posting first
    order = preflop_seats(solver.t)
    is_forced = forced_open(solver.t)

    # a solved table has more spots: other players in the stack range and other lines before them
    others = [(i, k) for i, k in hero_options
              if (i, k) != (hero_idx, kind) and float(bb_min) <= stacks[i] / bb <= float(bb_max)
              and fits(k, i, order, is_forced, covered_seats(stacks, sb, bb, ante, order))]
    rng.shuffle(others)
    for hero_idx, kind in [(hero_idx, kind)] + others:
        found = _pick_decision(solver, order.index(hero_idx), kind, difficulty, rng, strict)
        if found:
            break
    else:
        return None
    shover, caller, ev_play, ev_fold, ev_diff, freq, cls = found
    acted = order[:order.index(hero_idx)]  # the seats that act before Hero
    shover, caller = (None if x is None else order[x] for x in (shover, caller))

    seats = []
    for i, p in enumerate(pos):
        seats.append({
            "position": p,
            "stack": stacks[i],
            "stack_bb": round(stacks[i] / bb, 1),
            "hero": i == hero_idx,
            "hero_covers": stacks[hero_idx] >= stacks[i] and i != hero_idx,
        })
    actions = []
    for i in acted:
        if i == shover:
            actions.append({"seat": i, "position": pos[i], "action": "allin", "amount_bb": seats[i]["stack_bb"],
                            "forced": is_forced})
        elif i == caller:
            amount = min(stacks[i], stacks[shover])
            actions.append({"seat": i, "position": pos[i], "action": "call", "amount_bb": round(amount / bb, 1),
                            "allin": amount == stacks[i]})
        else:
            actions.append({"seat": i, "position": pos[i], "action": "fold"})
    return {
        "settings": {"players": n, "position": position, "kind": kind, "bb_min": bb_min, "bb_max": bb_max,
                     "difficulty": difficulty, "model": model, "overcalls": overcalls, "stacks": stacks_source,
                     "forced": forced},
        "source": source,
        "level": level, "sb": sb, "bb": bb, "ante": ante,
        "seats": seats,
        "hero": hero_idx,
        "kind": kind,
        "shover": shover,
        "caller": caller,
        "actions": actions,
        "cards": pick_cards(cls, rng),
        "hand": class_label(cls),
        "answer": {
            "hand_class": cls,
            "ev_play": float(ev_play[cls]),
            "ev_fold": float(ev_fold),
            "ev_diff": [round(float(x), 4) for x in ev_diff],
            "freq": [round(float(x), 3) for x in freq],
            "best": "play" if ev_diff[cls] > 0 else "fold",
            "solver_freq": float(freq[cls]),
            "range_pct": float(freq @ combo_counts() / combo_counts().sum() * 100),
        },
    }
