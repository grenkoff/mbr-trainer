"""Random final-table push/fold spots for the trainer, solved and packaged with the answer."""

import random

import numpy as np

from mbr import equity3
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


def build_table(stacks, sb, bb, ante, model):
    n = len(stacks)
    posted = [ante] * n
    posted[-1] += bb
    posted[-2] += sb
    return Table(names=positions(n), stacks=stacks, posted=posted, bb=bb, **MODELS[model])


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
              model="bounty", overcalls=False, seed=None):
    """Generate spots until one has a hand of the requested difficulty."""
    overcalls = str(overcalls).lower() in ("1", "true", "yes", "on")
    if overcalls and not equity3.available():
        raise ValueError("overcalls need the 3-way equity table: build/equity3.f32 (see README)")
    if kind == "overcall" and not overcalls:
        raise ValueError("overcall spots need overcalls enabled")
    if not 3 <= int(players) <= len(POSITIONS):
        raise ValueError(f"players must be 3..{len(POSITIONS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty}")
    rng = random.Random(seed)
    for attempt in range(MAX_ATTEMPTS):
        spot = _make_spot(rng, players, position, kind, bb_min, bb_max, difficulty, model, overcalls,
                          strict=attempt < MAX_ATTEMPTS - 1)
        if spot:
            return spot


def _kind_fits(kind, i, n):
    return {"open": i < n - 1, "call": i > 0, "overcall": i > 1}[kind]


def pick_line(solver, hero_idx, kind, rng):
    """Choose (shover, caller) before Hero in proportion to how often equilibrium play produces the line."""
    cc = combo_counts()
    cp = {key: solver._caller_stats(*key)[0] for key in solver.call}
    lines, weights, reach = [], [], 1.0
    for i in range(hero_idx):
        ppush = float(solver.push[i] @ cc / cc.sum())
        folds = 1.0  # players after the shover fold before the caller (or Hero)
        for j in range(i + 1, hero_idx):
            if kind == "overcall":
                lines.append((i, j))
                weights.append(reach * ppush * folds * cp[i, j])
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


def _make_spot(rng, players, position, kind, bb_min, bb_max, difficulty, model, overcalls, strict):
    n = int(players)
    pos = positions(n)
    if kind == "any":
        allowed_kinds = ["open", "call"] + (["overcall"] if overcalls else [])
    else:
        allowed_kinds = [kind]
    hero_options = [(i, k) for i, p in enumerate(pos) if position in ("any", p)
                    for k in allowed_kinds if _kind_fits(k, i, n)]
    if not hero_options:
        raise ValueError(f"position {position} cannot face '{kind}' with {n} players")
    hero_idx, kind = rng.choice(hero_options)
    hero_bb = rng.uniform(float(bb_min), float(bb_max))
    (level, sb, bb, ante), stacks = random_stacks(n, hero_idx, hero_bb, rng)
    table = build_table(stacks, sb, bb, ante, model)
    solver = Solver(table, overcalls=overcalls).solve()

    # a solved table has more spots: other players in the stack range and other lines before them
    others = [(i, k) for i, k in hero_options
              if (i, k) != (hero_idx, kind) and float(bb_min) <= stacks[i] / bb <= float(bb_max)]
    rng.shuffle(others)
    for hero_idx, kind in [(hero_idx, kind)] + others:
        found = _pick_decision(solver, hero_idx, kind, difficulty, rng, strict)
        if found:
            break
    else:
        return None
    shover, caller, ev_play, ev_fold, ev_diff, freq, cls = found

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
        elif i == caller:
            amount = min(stacks[i], stacks[shover])
            actions.append({"position": pos[i], "action": "call", "amount_bb": round(amount / bb, 1),
                            "allin": amount == stacks[i]})
        else:
            actions.append({"position": pos[i], "action": "fold"})
    return {
        "settings": {"players": n, "position": position, "kind": kind, "bb_min": bb_min, "bb_max": bb_max,
                     "difficulty": difficulty, "model": model, "overcalls": overcalls},
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
