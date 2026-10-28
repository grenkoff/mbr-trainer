# Mystery Battle Royale push/fold tools

Final-table push/fold solver (ICM + mystery bounties), hand-history analysis and a trainer
for GGPoker / PokerOK Mystery Battle Royale.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# 3-way all-in tables for overcalls (19 + 58 MB, a couple of minutes on 4 cores)
mkdir -p build
gcc -O3 -march=native -fopenmp -o build/equity3 csrc/equity3.c && build/equity3 1000 build/equity3.f32 build/equity3_ties.f32
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
- number of players, 2 to 9;
- position;
- decision type (open shove, call vs a shove, overcall vs a shove and a call);
- Hero stack range in bb;
- stacks: random, or the blinds and stacks of real final-table hands from `hh/hands/`, with Hero in any
  seat whose position and stack fit the settings;
- with history stacks, the hands where a player is all-in from posting his ante and blinds ("auto all-in",
  off by default). He has shoved before anybody acts, so the others call him or fold, the first to act too;
- difficulty:
  - as at the table: every one of the 1326 combos dealt with equal probability;
  - or by the cost of a mistake: easy more than 0.10 BI;
    medium 0.03–0.10 BI, hard less than 0.03 BI;
- valuation model: "as HRC" (the default, see below) or ICM with or without bounties;
- overcalls on/off (needs `build/equity3.f32` and `build/equity3_ties.f32`). The HRC model has them for 2-5
  players only: 6-9 players are solved without, so with overcalls on 2-5 players get them and 6-9 do not.

The HRC model serves history spots from a library solved in advance (`build/library/`), since it
takes about half a minute per 3-player table and one to three minutes per 4-player one:

```bash
.venv/bin/python build_library.py --players 2,3,4,5 --workers 1   # only the set-ups not solved yet: rerun after new hands
.venv/bin/python build_library.py --players 2,3,4,5,6,7,8,9 --workers 1 --no-overcalls   # without overcalls
```

With overcalls up to three players are all-in; without them a shove gets one call at most, in the next hand
of FGS too. Without overcalls a table takes from a few seconds (3-4 players) to three minutes (9 players);
with them five to six minutes for 5 players, and about three and a half times more for every player beyond.

The build pauses its workers while the CPU package is hotter than `--max-temp` (85 °C by default) and resumes
them at `--resume-temp` (75 °C): a laptop running all cores flat out for hours can overheat and shut down.

Hotkeys: `F` fold, `A` shove/call, `Space` next.

Rating: Elo as on chess sites, one for every number of players (as chess sites keep one per time control),
from 1500, with every spot as the opponent. The panel shows the one of the spot on the table. The closer push and fold are in EV,
the stronger the spot: 0.3 BI apart is rated 800, 0.03 BI 1500, 0.003 BI 2200. A big mistake (an easy spot)
therefore loses a lot and a right answer there gains little, while a hard spot gains a lot and costs little.
K is 40 for the first 30 answers (the rating shows a `?`) and 20 after; decisions within 0.005 BI leave it as is.

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
- hands that tie split the pot and the bounty;
- the cards of the players who folded (bunching), to first order. Folds make the hands that leave them
  likely more likely for the players still in. Folded cards are dead for the board, which moves the
  equities (`data/dead_169.f32`);
- values are in buy-ins;
- place prizes are 4/3/2 buy-ins with Malmuth-Harville ICM;
- a knockout pays a mystery bounty of 0.945 buy-ins on average;
- optionally, the future bounties are split in proportion to chips.

### The HRC model

The aim is the ranges HoldemResources Calculator (HRC) shows for push/fold with FGS (`mbr/fgs.py`,
model "hrc"). What HRC does was gathered from its release notes and its developers' forum answers:
- push/fold with at most three players all-in, exact equities;
- the folded players' cards are left out, as in HRC's push/fold (only its Monte Carlo mode has them);
- FGS of depth 1: every outcome of the hand is valued by one more hand. The button moves on, and the
  same blinds and antes are posted. That hand is played as the push/fold equilibrium with ICM after it,
  and each player's value at its start is what the outcome is worth. As in HRC, outcomes where the pot
  is split are valued with ICM;
- a player left with no more than his antes and blinds is all-in from posting in that next hand: he
  always plays, and the others call him or fold. Hands from the history where that happens are solved the
  same way (with two such players they are left out);
- mystery bounties as HRC's KO bounties of 0.945 BI. Future bounties are split by chips, and the
  winner does not get his own bounty: 8 bounties for 9 players. In HRC that means a first prize of
  4 − 0.945 BI with KO bounties of 0.945 BI.

With overcalls the solver starts from the solution without them and runs 40 more fictitious play
iterations: about 0.5–5 seconds for 3–5 players, 25–30 for 9. Later iterations weigh more in the averaged strategies
(weight ~ t³), which converges far faster than the plain mean.

### Checks

```bash
.venv/bin/python check_gs2008.py                  # against the published 3-player equilibrium
.venv/bin/python exploitability.py --per-count 5  # how far from an equilibrium, on real stacks
```

- `check_gs2008.py` solves the game from Ganzfried & Sandholm (AAMAS 2008): one hand of a cash game,
  3 × 4500 chips, blinds 300/600, chip EV. It compares the six ranges with the paper's tables
  (`data/gs2008_single_hand.txt`).
- `exploitability.py` measures how much a player would win per hand by best-responding to the
  solved ranges (`mbr/exploit.py`). Near zero means the ranges are an equilibrium of the model.

### Reference solver

`reference/` is a slow solver of the same game that shares no code with the main one. It drops the
main solver's shortcuts:
- any number of players can be all-in;
- tied hands split the pot and the bounty;
- the cards of all players, folded ones included, come from one deal, so card removal is exact.

It samples a few million complete deals with its own hand evaluator and finds the equilibrium with
CFR+ instead of fictitious play.

```bash
.venv/bin/python check_reference.py --players 3,4 --per-count 2   # ~15 min per 4-player spot
```

The reference is solved on one sample of deals. Both solutions are then scored on another sample,
against opponents who play the reference ranges. For each decision the script prints what the main
solver's range costs, in thousandths of a buy-in per hand, with its standard error. The sampled deals
are cached in `build/reference/`.

A decision reached in well under 1% of hands, such as an overcall behind a very tight shove, gets only
a few dozen sampled deals per hand class. Its reference range is noise, and so are the ranges of the
decisions that follow it, so the cost printed there means little.

### Targeted check of single decisions

```bash
.venv/bin/python check_decisions.py --hand BR1287900949                  # every decision, ~6 min for 4 players
.venv/bin/python check_decisions.py 3.2 17.8 94.3 4.7 --decision "BB vs SB" --samples 100000
```

`check_decisions.py` checks the EVs the main solver reports, one decision at a time, rare ones included
(`reference/targeted.py`). It deals only hands in which the decision is reached:
- the deciding player gets each of the 169 hand classes in turn;
- the players who acted before get hands from their main-solver ranges, all-in or folded;
- the players behind and the board come from the rest of the deck.

Card removal is exact and split pots are split. Everybody plays the main solver's ranges, and the script
computes the exact EV of both actions for every hand class. For each decision it prints the mean error
of the main EV gap next to the sampling noise. It also lists the hands the main range gets wrong by
more than 3 standard errors, and what they cost per hand dealt. It works for any number of players;
with many players it takes long, so lower `--samples` or pick a `--decision`.

On real stacks it found two shortcuts of the main model:
- **Split pots.** The main model used to value a tie as half a win and half a loss. A short stack that
  chops survives, so this is worth more than the average. For a 4.7bb big blind calling a shove the
  model missed 0.01–0.035 BI per hand, the most with weak hands, which tie most often. Ties are now
  split, heads-up and 3-way, and the errors there are down to the sampling noise.
- **Folded players' cards (bunching).** The model used to leave them in the deck. With 6 players its EV
  gaps behind folds were off by 0.01–0.04 BI on average, up to 0.08 BI for some weak suited hands.
  About two thirds of that came from the board, where folded low cards cannot come, and one third from
  the ranges of the players still in. The model now takes both into account to first order. On 3-, 4-
  and 6-player spots every decision is down to the sampling noise, overcalls included.

## Hand history analysis

Put GG hand histories into `hh/hands/` and tournament summaries into `hh/summaries/` (ignored by git).

```bash
python3 report.py                                      # structure stats and most frequent spots
.venv/bin/python compare.py --model bounty             # score Hero's decisions against the solver
.venv/bin/python compare.py --model bounty --overcalls --workers 2   # with overcalls (slower, resumable)
python3 leaks.py build/decisions_bounty_overcalls.csv  # where the EV goes, by situation
.venv/bin/python ev_report.py                          # Rush chipEV/tour and final-table $EV, all-in adjusted
```

`ev_report.py` counts every hand whose betting ended before the river with all the remaining cards shown at
its expectation over the boards still to come (`csrc/showdown.c`, built on first use), and every knockout at
the mean bounty. At the final table money is ICM of 4/3/2 BI plus the bounties to come split by chips, so the
others' all-ins count too: they move Hero's money as well.

Bounties are paid from 9th place on, so when the Rush stage ends below 9 players (several tables bust at once), a
knockout in its last hand may pay one too. Which of Hero's knockouts there were paid is not in the hand history: the
report takes the number whose envelopes most likely add up, with the final-table ones, to the bounties of the
summary. It also lists the tournaments whose bounties no envelopes of their knockouts add up to.

## Equity tables

`data/equity_169.txt` holds Monte Carlo all-in equities for the 169x169 hand classes. To regenerate it:

```bash
gcc -O3 -o build/equity csrc/equity.c && build/equity 50000 > data/equity_169.txt
build/equity 50000 ties > data/ties_169.txt    # the probabilities that two hands tie
gcc -O3 -march=native -fopenmp -o build/dead csrc/dead.c && build/dead 50000 data/dead_169.f32
```

`data/dead_169.f32` holds, for every class pair and card rank, how much one dead card of that rank moves
the heads-up equity and the tie probability. A dead card dealt at random changes nothing, so this is
measured against a random one. The solver adds these changes up over the expected cards of the folded
hands. For 3-way all-ins it uses the part of the effect that can be split by hand
(`mbr.ranges.dead_card_strength`, about 89% of it).

`build/equity3.f32` (not in git, see Setup) holds the 3-way outcomes for all 169^3 class triples:
card removal weight times the probability of each strength order, ties split evenly between the orders
they fit. `build/equity3_ties.f32` holds the ties: two hands tie for best, two tie for last, all three tie.
