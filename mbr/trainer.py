"""Random final-table push/fold spots for the trainer, solved and packaged with the answer."""

import random

import numpy as np

from mbr.pushfold import MEAN_BOUNTY_BI, Solver, Table
from mbr.ranges import class_combos, class_label, combo_counts

TOTAL_CHIPS = 18000
# (level, sb, bb, ante) of the Mystery Battle Royale structure from the final stage on
LEVELS = [
    (4, 40, 80, 16), (5, 50, 100, 20), (6, 60, 120, 24), (7, 75, 150, 30), (8, 100, 200, 40),
    (9, 150, 300, 60), (10, 200, 400, 80), (11, 300, 600, 120), (12, 400, 800, 160), (13, 500, 1000, 200),
]
POSITIONS = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
MODELS = {
    "bounty": dict(bounty=MEAN_BOUNTY_BI, future_bounty=True),
    "bounty-now": dict(bounty=MEAN_BOUNTY_BI, future_bounty=False),
    "icm": dict(bounty=0.0, future_bounty=False),
}
# cost of the mistake |EV(play) - EV(fold)| in buy-ins
DIFFICULTY_BANDS = {"hard": (0.0, 0.03), "medium": (0.03, 0.10), "easy": (0.10, 100.0)}
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


def build_table(stacks, sb, bb, ante, model):
    n = len(stacks)
    posted = [ante] * n
    posted[-1] += bb
    posted[-2] += sb
    return Table(names=positions(n), stacks=stacks, posted=posted, bb=bb, **MODELS[model])


def pick_hand(ev_diff, difficulty, rng, strict=True):
    """Pick a hand class whose mistake cost matches the difficulty, or None if strict and there is none."""
    cc = combo_counts()
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


def make_spot(players=5, position="any", kind="any", bb_min=5, bb_max=20, difficulty="medium",
              model="bounty", seed=None):
    """Generate spots until one has a hand of the requested difficulty."""
    rng = random.Random(seed)
    for attempt in range(MAX_ATTEMPTS):
        spot = _make_spot(rng, players, position, kind, bb_min, bb_max, difficulty, model,
                          strict=attempt < MAX_ATTEMPTS - 1)
        if spot:
            return spot


def _make_spot(rng, players, position, kind, bb_min, bb_max, difficulty, model, strict):
    n = int(players)
    pos = positions(n)
    allowed_kinds = {"open", "call"} if kind == "any" else {kind}
    hero_options = []
    for i, p in enumerate(pos):
        if position not in ("any", p):
            continue
        for k in allowed_kinds:
            if (k == "open" and i < n - 1) or (k == "call" and i > 0):
                hero_options.append((i, k))
    if not hero_options:
        raise ValueError(f"position {position} cannot face '{kind}' with {n} players")
    hero_idx, kind = rng.choice(hero_options)
    hero_bb = rng.uniform(float(bb_min), float(bb_max))
    (level, sb, bb, ante), stacks = random_stacks(n, hero_idx, hero_bb, rng)
    table = build_table(stacks, sb, bb, ante, model)
    solver = Solver(table).solve()

    if kind == "open":
        shover = None
        ev_play, ev_fold = solver.ev["push", hero_idx]
        freq = solver.push[hero_idx]
    else:
        # choose the shover in proportion to how often equilibrium play produces this line
        cc = combo_counts()
        weights, reach = [], 1.0
        for i in range(hero_idx):
            ppush = float(solver.push[i] @ cc / cc.sum())
            weights.append(reach * ppush)
            reach *= 1 - ppush
        shover = rng.choices(range(hero_idx), weights=weights)[0] if sum(weights) > 0 else rng.randrange(hero_idx)
        ev_play, ev_fold = solver.ev["call", shover, hero_idx]
        freq = solver.call[shover, hero_idx]
    ev_diff = np.asarray(ev_play) - ev_fold
    cls = pick_hand(ev_diff, difficulty, rng, strict)
    if cls is None:
        return None

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
    for i in range(hero_idx):
        if i == shover:
            actions.append({"position": pos[i], "action": "allin", "amount_bb": seats[i]["stack_bb"]})
        else:
            actions.append({"position": pos[i], "action": "fold"})
    return {
        "settings": {"players": n, "position": position, "kind": kind, "bb_min": bb_min, "bb_max": bb_max,
                     "difficulty": difficulty, "model": model},
        "level": level, "sb": sb, "bb": bb, "ante": ante,
        "seats": seats,
        "hero": hero_idx,
        "kind": kind,
        "shover": shover,
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
