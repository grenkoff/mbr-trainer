"""The statistics tab: Hero's results, preflop stats and push/fold mistakes, summed up from mbr.results."""

from collections import Counter, defaultdict
from statistics import mean, stdev

from mbr.pushfold import MEAN_BOUNTY_BI
from mbr.results import PREFLOP_KEYS
from mbr.spots import STACK_BUCKETS, stack_bucket

POSITION_ORDER = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
BUCKET_ORDER = [stack_bucket(lo) for lo, _ in STACK_BUCKETS]
WORST = 50  # the costliest mistakes listed


def _stat(values):
    """[mean, its standard error], or None without values."""
    if not values:
        return None
    se = stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return [mean(values), se]


def tournament_row(tid, r):
    """One tournament's results in BI (net of the buy-in) and Rush chips."""
    rush_ko = r.get("rush_kos", 0) * MEAN_BOUNTY_BI
    return {
        "tid": tid, "date": r.get("date"), "buyin": r.get("buyin"), "place": r.get("place"), "ft": bool(r.get("ft")),
        "actual": r.get("roi", 0.0),  # bounties as drawn
        "mean": r.get("v0", 0) + r.get("ft_actual", 0) + rush_ko - 1,  # mean bounties
        "ev": r.get("v0", 0) + r.get("ft_ev", 0) + r.get("rush_kos_ev", 0) * MEAN_BOUNTY_BI - 1,
        # the Rush chips at their average worth too: 18 x 0.92 BI of prizes and bounties for 18000 chips
        "full": (1000 + r.get("rush_ev", 0)) * 0.92 / 1000 + r.get("ft_ev", 0)
                + r.get("rush_kos_ev", 0) * MEAN_BOUNTY_BI - 1,
        "rush_actual": r.get("rush_actual", 0), "rush_ev": r.get("rush_ev", 0),
        "kos": r.get("kos", 0) + r.get("rush_kos", 0), "bounties": r.get("bounties", 0.0),
    }


def summary(results, date_from=None, date_to=None, buyin=None):
    """Everything the statistics tab shows for the tournaments in the filters (dates as YYYY-MM-DD)."""
    picked = _picked(results, date_from, date_to, buyin)
    rows = sorted((tournament_row(tid, r) for tid, r in picked.items()), key=lambda t: (t["date"] or "", t["tid"]))
    ft = [r for tid, r in picked.items() if r.get("ft")]
    places = Counter(t["place"] for t in rows if t["place"])
    kos = sum(t["kos"] for t in rows)
    totals = {
        "tournaments": len(rows), "final_tables": len(ft), "places": dict(sorted(places.items())),
        "itm": sum(n for p, n in places.items() if p <= 3),
        "roi": {k: _stat([t[k] for t in rows]) for k in ("actual", "mean", "ev", "full")},
        "rush": {k: _stat([t[k] for t in rows]) for k in ("rush_actual", "rush_ev")},
        "ft_value": mean(r["v0"] for r in ft) if ft else None,
        "ft_change": {k: _stat([r.get(f"ft_{k}", 0) for r in ft]) for k in ("actual", "ev")},
        "ft_luck": {k: mean(r.get(f"ft_actual_{k}", 0) - r.get(f"ft_ev_{k}", 0) for r in ft) if ft else None
                    for k in ("hero", "others")},
        "kos": kos, "ft_kos": sum(r.get("kos", 0) for r in ft), "rush_kos": sum(r.get("rush_kos", 0) for r in ft),
        "bounties": sum(t["bounties"] for t in rows),
        "bounty_per_ko": sum(t["bounties"] for t in rows) / kos if kos else None,
    }
    return {"tournaments": rows, "totals": totals, "preflop": _preflop(picked.values()),
            "decisions": _decisions(picked.values()), "buyins": sorted({r.get("buyin") for r in results.values()
                                                                        if r.get("buyin") is not None})}


def _preflop(results):
    """Counts by stage and position and by stage and stack bucket, plus each stage's total."""
    by_pos, by_bucket, total = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for r in results:
        for stage, position, bucket, counts in r.get("preflop", []):
            by_pos[stage, position].update(counts)
            by_bucket[stage, bucket].update(counts)
            total[stage].update(counts)
    pick = lambda c: {k: c[k] for k in PREFLOP_KEYS}
    return {stage: {"total": pick(total[stage]),
                    "positions": [[p, pick(by_pos[stage, p])] for p in POSITION_ORDER if (stage, p) in by_pos],
                    "buckets": [[b, pick(by_bucket[stage, b])] for b in BUCKET_ORDER if (stage, b) in by_bucket]}
            for stage in ("rush", "ft") if stage in total}


def _decisions(results):
    """Hero's push/fold decisions against HRC: totals, by decision and position, by decision and stack, worst."""
    ds = [d for r in results for d in r.get("decisions", [])]

    def group(key):
        out = defaultdict(lambda: Counter(decisions=0, mistakes=0, loss=0.0, plays=0, best_plays=0))
        for d in ds:
            g = out[d["kind"], d[key]]
            g["decisions"] += 1
            g["mistakes"] += d["loss"] > 0
            g["loss"] += d["loss"]
            g["plays"] += d["action"] == "play"
            g["best_plays"] += d["best"] == "play"
        order = POSITION_ORDER if key == "position" else BUCKET_ORDER
        return [[kind, k, dict(out[kind, k])] for kind in ("open", "call", "overcall") for k in order
                if (kind, k) in out]

    return {"decisions": len(ds), "mistakes": sum(d["loss"] > 0 for d in ds), "loss": sum(d["loss"] for d in ds),
            "by_position": group("position"), "by_bucket": group("bucket"),
            "worst": sorted((d for d in ds if d["loss"] > 0), key=lambda d: -d["loss"])[:WORST]}


def _picked(results, date_from, date_to, buyin):
    return {tid: r for tid, r in results.items()
            if (not date_from or (r.get("date") or "")[:10] >= date_from)
            and (not date_to or (r.get("date") or "")[:10] <= date_to)
            and (buyin is None or r.get("buyin") == buyin)}


def opponents_summary(theirs, hero, date_from=None, date_to=None, buyin=None):
    """Everything the average opponent tab shows, from mbr.opponents data (and Hero's mbr.results to compare)."""
    picked = _picked(theirs, date_from, date_to, buyin)
    shown = defaultdict(lambda: [0] * 169)
    for r in picked.values():
        for key, counts in r.get("shown", {}).items():
            for cls, n in counts.items():
                shown[key][int(cls)] += n
    return {"tournaments": len(picked), "preflop": _preflop(picked.values()),
            "hero_preflop": _preflop(_picked(hero, date_from, date_to, buyin).values()),
            "decisions": _their_decisions(picked.values()), "shown": dict(shown),
            "buyins": sorted({r.get("buyin") for r in theirs.values() if r.get("buyin") is not None})}


def _their_decisions(results):
    """The opponents' push/fold decisions against HRC by decision and position and by decision and stack:
    how often they play and how much of the hands HRC plays; of their shown hands, those HRC folds. Limps and
    raises short of all-in (other) are counted apart."""
    ds = [d for r in results for d in r.get("decisions", [])]

    def group(key):
        out = defaultdict(lambda: Counter(decisions=0, plays=0, hrc=0.0, shown=0, shown_hrc_folds=0, loss=0.0,
                                          other=0))
        for d in ds:
            for g in (out[d["kind"], d[key]], out[d["kind"], "all"]):
                if d["action"] == "other":
                    g["other"] += 1
                    continue
                g["decisions"] += 1
                g["plays"] += d["action"] == "play"
                g["hrc"] += d["range_pct"] / 100
                if d["action"] == "play" and "best" in d:
                    g["shown"] += 1
                    g["shown_hrc_folds"] += d["best"] == "fold"
                    g["loss"] += d["loss"]
        order = (POSITION_ORDER if key == "position" else BUCKET_ORDER) + ["all"]
        return [[kind, k, dict(out[kind, k])] for kind in ("open", "call", "overcall") for k in order
                if (kind, k) in out]

    return {"decisions": sum(d["action"] != "other" for d in ds), "by_position": group("position"),
            "by_bucket": group("bucket")}
