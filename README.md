# Mystery Battle Royale push/fold tools

Final-table push/fold solver (ICM + mystery bounties), hand-history analysis and a trainer
for GGPoker / PokerOK Mystery Battle Royale.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Trainer

```bash
.venv/bin/python -m trainer          # http://localhost:8765
```

Settings:
- number of players (3–5 for now);
- position;
- decision type (open shove or call vs a shove);
- Hero stack range in bb;
- difficulty, set by the cost of a mistake:
  - easy: more than 0.10 BI;
  - medium: 0.03–0.10 BI;
  - hard: less than 0.03 BI;
- valuation model.

Hotkeys: `F` fold, `A` shove/call, `Space` next.

## Solver

```bash
.venv/bin/python solve_spot.py 12 12 12 8 12 15     # stacks in bb in preflop order, last two are SB and BB
```

The model:
- the first player shoves or folds, and one caller at most (no overcalls yet);
- values are in buy-ins;
- place prizes are 4/3/2 buy-ins with Malmuth-Harville ICM;
- a knockout pays a mystery bounty of 0.945 buy-ins on average;
- optionally, the future bounties are split in proportion to chips.

## Hand history analysis

Put GG hand histories into `hh/hands/` and tournament summaries into `hh/summaries/` (ignored by git).

```bash
python3 report.py                                      # structure stats and most frequent spots
.venv/bin/python compare.py --model bounty             # score Hero's decisions against the solver
```

## Equity table

`data/equity_169.txt` holds Monte Carlo all-in equities for the 169x169 hand classes. To regenerate it:

```bash
gcc -O3 -o build/equity csrc/equity.c && build/equity 50000 > data/equity_169.txt
```
