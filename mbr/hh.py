"""Parser for GGPoker / PokerOK Mystery Battle Royale hand histories and tournament summaries."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HEADER_RE = re.compile(
    r"^Poker Hand #(?P<hand_id>\S+): Tournament #(?P<tid>\d+), .*? - "
    r"Level(?P<level>\d+)\((?P<sb>[\d,]+)/(?P<bb>[\d,]+)\((?P<ante>[\d,]+)\)\) - "
    r"(?P<ts>\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)"
)
TABLE_RE = re.compile(r"^Table '(?P<table>[^']*)' (?P<max>\d+)-max Seat #(?P<button>\d+) is the button")
SEAT_RE = re.compile(r"^Seat (?P<seat>\d+): (?P<name>\S+) \((?P<stack>[\d,]+) in chips\)")
DEALT_RE = re.compile(r"^Dealt to Hero \[(?P<cards>[^\]]+)\]")
ACTION_RE = re.compile(
    r"^(?P<name>[^:\s]+): (?P<verb>posts the ante|posts small blind|posts big blind|folds|checks|calls|bets|raises|shows)"
    r"(?: (?P<amount>[\d,]+))?(?: to (?P<to>[\d,]+))?(?P<rest>.*)$"
)
COLLECTED_RE = re.compile(r"^(?P<name>\S+) collected (?P<amount>[\d,]+) from pot")
UNCALLED_RE = re.compile(r"^Uncalled bet \((?P<amount>[\d,]+)\) returned to (?P<name>\S+)")
STREET_RE = re.compile(r"^\*\*\* (?P<street>HOLE CARDS|FLOP|TURN|RIVER|SHOWDOWN|SUMMARY)")

SUMMARY_TID_RE = re.compile(r"^Tournament #(?P<tid>\d+), (?P<name>.*?) \$(?P<buyin>[\d.]+),")
SUMMARY_BUYIN_RE = re.compile(r"^Buy-in: \$(?P<prize>[\d.]+)\+\$(?P<rake>[\d.]+)\+\$(?P<bounty>[\d.]+)")
SUMMARY_PLACE_RE = re.compile(r"^(?P<place>\d+)(?:st|nd|rd|th) : Hero, \$(?P<won>[\d.]+)")


def _int(s):
    return int(s.replace(",", "")) if s else 0


@dataclass
class Action:
    street: str
    name: str
    verb: str  # ante, sb, bb, fold, check, call, bet, raise, show
    amount: int = 0  # chips added by this action (for raise: increment)
    to: int = 0  # raise-to total for the street
    all_in: bool = False
    cards: str = ""


@dataclass
class Hand:
    hand_id: str
    tid: str
    level: int
    sb: int
    bb: int
    ante: int
    ts: datetime
    table: str
    max_seats: int
    button: int
    seats: dict  # seat number -> (name, stack)
    hero_cards: str = ""
    actions: list = field(default_factory=list)
    collected: dict = field(default_factory=dict)  # name -> chips won
    uncalled: dict = field(default_factory=dict)
    board: list = field(default_factory=list)  # community cards dealt, e.g. ['Qd', '8d', 'Jc']

    @property
    def stacks(self):
        return {name: stack for name, stack in self.seats.values()}

    @property
    def players(self):
        return [name for name, _ in self.seats.values()]

    @property
    def is_final_stage(self):
        return self.max_seats == 9

    def chips_put_in(self):
        """Total chips each player committed, net of uncalled bets."""
        total = {name: 0 for name in self.players}
        street_put = {}
        street = None
        for a in self.actions:
            if a.street != street:
                street, street_put = a.street, {}
            if a.verb == "ante":
                total[a.name] += a.amount
            elif a.verb in ("sb", "bb", "call", "bet"):
                total[a.name] += a.amount
                street_put[a.name] = street_put.get(a.name, 0) + a.amount
            elif a.verb == "raise":
                add = a.to - street_put.get(a.name, 0)
                total[a.name] += add
                street_put[a.name] = a.to
        for name, amount in self.uncalled.items():
            total[name] -= amount
        return total

    def final_stacks(self):
        put = self.chips_put_in()
        return {n: s - put[n] + self.collected.get(n, 0) for n, s in self.stacks.items()}

    def shown_cards(self):
        """Player name -> the two cards shown at showdown (Hero's own cards too)."""
        out = {a.name: re.findall(r"[2-9TJQKA][cdhs]", a.cards)[:2] for a in self.actions if a.verb == "show"}
        if self.hero_cards and "Hero" in self.stacks:
            out["Hero"] = self.hero_cards.split()
        return {n: c for n, c in out.items() if len(c) == 2}

    def positions(self):
        """Map player name -> position label (BTN, SB, BB, UTG, ..., CO)."""
        order = sorted(self.seats)
        # rotate so that the seat after the button comes first
        start = next((i for i, s in enumerate(order) if s > self.button), 0)
        order = order[start:] + order[:start]
        names = [self.seats[s][0] for s in order]  # action order starting left of button
        sb_poster = next((a.name for a in self.actions if a.verb == "sb"), None)
        bb_poster = next((a.name for a in self.actions if a.verb == "bb"), None)
        pos = {}
        if len(names) == 2:
            for n in names:
                pos[n] = "BB" if n == bb_poster else "SB"
            return pos
        button_name = self.seats.get(self.button, (None,))[0]
        others = [n for n in names if n not in (sb_poster, bb_poster, button_name)]
        labels = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO"][-len(others):] if others else []
        for n, label in zip(others, labels):
            pos[n] = label
        if button_name:
            pos[button_name] = "BTN"
        if sb_poster:
            pos[sb_poster] = "SB"
        if bb_poster:
            pos[bb_poster] = "BB"
        return pos


def parse_hand(block):
    lines = [l.rstrip() for l in block.strip().splitlines()]
    m = HEADER_RE.match(lines[0])
    if not m:
        return None
    t = TABLE_RE.match(lines[1])
    hand = Hand(
        hand_id=m["hand_id"], tid=m["tid"], level=int(m["level"]),
        sb=_int(m["sb"]), bb=_int(m["bb"]), ante=_int(m["ante"]),
        ts=datetime.strptime(m["ts"], "%Y/%m/%d %H:%M:%S"),
        table=t["table"], max_seats=int(t["max"]), button=int(t["button"]), seats={},
    )
    street = "PRE"
    for line in lines[2:]:
        if (s := STREET_RE.match(line)):
            st = s["street"]
            if st == "SUMMARY":
                break
            street = {"HOLE CARDS": "PRE"}.get(st, st)
            if st in ("FLOP", "TURN", "RIVER"):
                hand.board = re.findall(r"[2-9TJQKA][cdhs]", line)
            continue
        if street == "PRE" and (s := SEAT_RE.match(line)) and not hand.actions:
            hand.seats[int(s["seat"])] = (s["name"], _int(s["stack"]))
            continue
        if (d := DEALT_RE.match(line)):
            hand.hero_cards = d["cards"]
            continue
        if (c := COLLECTED_RE.match(line)):
            hand.collected[c["name"]] = hand.collected.get(c["name"], 0) + _int(c["amount"])
            continue
        if (u := UNCALLED_RE.match(line)):
            hand.uncalled[u["name"]] = hand.uncalled.get(u["name"], 0) + _int(u["amount"])
            continue
        if (a := ACTION_RE.match(line)):
            verb = {
                "posts the ante": "ante", "posts small blind": "sb", "posts big blind": "bb",
                "folds": "fold", "checks": "check", "calls": "call", "bets": "bet",
                "raises": "raise", "shows": "show",
            }[a["verb"]]
            act_street = "PRE" if verb in ("ante", "sb", "bb") else street
            hand.actions.append(Action(
                street=act_street, name=a["name"], verb=verb,
                amount=_int(a["amount"]), to=_int(a["to"]),
                all_in="all-in" in (a["rest"] or ""),
                cards=(a["rest"] or "").strip() if verb == "show" else "",
            ))
    return hand


def parse_hands_file(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    hands = [h for b in re.split(r"\n\s*\n(?=Poker Hand #)", text) if b.strip() and (h := parse_hand(b))]
    return hands


def load_hands(directory):
    """Return {tournament_id: [hands in chronological order]}."""
    by_tid = {}
    for path in sorted(Path(directory).glob("*.txt")):
        for h in parse_hands_file(path):
            by_tid.setdefault(h.tid, {})[h.hand_id] = h
    return {tid: sorted(hs.values(), key=lambda h: (h.ts, h.hand_id)) for tid, hs in by_tid.items()}


@dataclass
class Summary:
    tid: str
    buyin: float
    prize_part: float
    rake: float
    bounty_part: float
    place: int
    received: float


def load_summaries(directory):
    out = {}
    for path in sorted(Path(directory).glob("*.txt")):
        text = Path(path).read_text(encoding="utf-8-sig")
        vals = {}
        for line in text.splitlines():
            if (m := SUMMARY_TID_RE.match(line)):
                vals.update(tid=m["tid"], buyin=float(m["buyin"]))
            elif (m := SUMMARY_BUYIN_RE.match(line)):
                vals.update(prize_part=float(m["prize"]), rake=float(m["rake"]), bounty_part=float(m["bounty"]))
            elif (m := SUMMARY_PLACE_RE.match(line)):
                vals.update(place=int(m["place"]), received=float(m["won"]))
        if "tid" in vals and "place" in vals:
            out[vals["tid"]] = Summary(**vals)
    return out
