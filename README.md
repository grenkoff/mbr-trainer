# Mystery Battle Royale push/fold tools

Final-table push/fold solver (ICM + mystery bounties), hand-history analysis and a trainer
for GGPoker / PokerOK Mystery Battle Royale.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# 3-way all-in table for overcalls (19 MB, a couple of minutes on 4 cores)
mkdir -p build
gcc -O3 -march=native -fopenmp -o build/equity3 csrc/equity3.c && build/equity3 1000 build/equity3.f32
# the Rust solver (optional, 2-6 times faster than numpy); without it the numpy version runs
(cd rust && cargo build --release)
```

The solver runs in Rust when `rust/target/release/libmbr_solver.so` exists, unless `MBR_SOLVER=python`;
`MBR_THREADS=N` sets its threads (default: all cores; the trainer's solver processes use one each).

## Trainer

```bash
.venv/bin/python -m trainer          # http://localhost:8765
```

Settings:
- number of players (3–9, several at once: each spot gets one of them);
- position;
- decision type (open shove, call vs a shove, overcall vs a shove and a call);
- Hero stack range in bb;
- stacks: random, or the blinds and stacks of real final-table hands from `hh/hands/`, with Hero in any
  seat whose position and stack fit the settings;
- difficulty:
  - as at the table: every one of the 1326 combos dealt with equal probability;
  - or by the cost of a mistake: easy more than 0.10 BI;
    medium 0.03–0.10 BI, hard less than 0.03 BI;
- valuation model;
- overcalls on/off (needs `build/equity3.f32`).

Hotkeys: `F` fold, `A` shove/call, `Space` next.

Solver processes (CPU cores − 1 by default, `--workers N`) keep about 10 solved spots ready for the
settings in use, so answers come instantly. The stock is kept in `build/spot_pool/` between runs and
can be filled in advance, also while the server runs:

```bash
.venv/bin/python -m trainer.fill --players 5 --difficulty mix --overcalls --stacks history --count 300
```

## Solver

```bash
.venv/bin/python solve_spot.py 12 12 12 8 12 15     # stacks in bb in preflop order, last two are SB and BB
.venv/bin/python solve_spot.py 15 10 22 --overcalls  # with overcalls
```

The model:
- the first player shoves or folds;
- without overcalls one caller at most; with overcalls up to three players all-in, side pots included;
- values are in buy-ins;
- place prizes are 4/3/2 buy-ins with Malmuth-Harville ICM;
- a knockout pays a mystery bounty of 0.945 buy-ins on average;
- optionally, the future bounties are split in proportion to chips.

With overcalls the solver starts from the solution without them and runs 40 more fictitious play
iterations: about 1–2 seconds for 3–5 players.

## Hand history analysis

Put GG hand histories into `hh/hands/` and tournament summaries into `hh/summaries/` (ignored by git).

```bash
python3 report.py                                      # structure stats and most frequent spots
.venv/bin/python compare.py --model bounty             # score Hero's decisions against the solver
.venv/bin/python compare.py --model bounty --overcalls --workers 2   # with overcalls (slower, resumable)
python3 leaks.py build/decisions_bounty_overcalls.csv  # where the EV goes, by situation
```

## Equity tables

`data/equity_169.txt` holds Monte Carlo all-in equities for the 169x169 hand classes. To regenerate it:

```bash
gcc -O3 -o build/equity csrc/equity.c && build/equity 50000 > data/equity_169.txt
```

`build/equity3.f32` (not in git, see Setup) holds the 3-way outcomes for all 169^3 class triples:
card removal weight times the probability of each strength order.
