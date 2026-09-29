# Kicker Analysis

Forecasting NFL kicker fantasy points, and being honest about how little of it is
forecastable.

The headline is not the model. It is this: **weekly field goal attempts have
negative out-of-sample skill** (R-squared -0.054) and carry 102% of the variance
in kicker scoring, while extra point attempts are genuinely predictable (within-week
Spearman 0.299) and carry 10%. You cannot forecast field goals. You forecast
touchdowns, and the field goals are close to a coin flip the coach can veto.

Everything else follows from that.

## What it does

| | |
|---|---|
| **data** | 14,519 kicker-games, 1999 to 2026 week 3, player grain, from nflverse |
| **target** | that week's actual fantasy points, from the distance buckets |
| **ranking** | ridge on 16 named pre-kickoff features, all 16 non-constant |
| **distribution** | two Poisson count stages convolved through the scoring rule |
| **output** | a weekly board, ranked, with a calibrated `P(15+)` per kicker |

## Does it work

Scored on the decision a manager actually makes: draw two kickers from the week's
starters, start the one the method prefers, measure points gained against starting
either at random. Eight season-forward folds, 2018 to 2025, 141 weeks. Every lift
is the exact expectation over every pair of starters in a week, not a simulation.
The oracle who always starts the better of the two gains 2.46 points a week.

| rank by | lift, pts/wk | share of oracle |
|---|---|---|
| **this model, the ridge that orders the board** | **0.432** | **17.6%** |
| the count model's expected points | 0.382 | 15.5% |
| Vegas implied team total alone | 0.353 | 14.4% |
| kicker's own career average | 0.180 | 7.3% |
| a constant 8.0 | 0.000 | 0.0% |

Against the two things people actually do instead, one family of two tests:

| ridge minus | week-block bootstrap | season-cluster bootstrap | seasons won |
|---|---|---|---|
| Vegas implied total alone | **+0.079 [+0.030, +0.128], p = 0.001** | [+0.008, +0.143], p = 0.028 | 6 of 8, sign test p = 0.29 |
| kicker career average | +0.252 [+0.161, +0.341], p < 0.0001 | [+0.169, +0.348], p < 0.0001 | 8 of 8, p = 0.008 |

The career-average claim holds by every test. The line claim holds in the average
and is thin by season: the ridge loses 2019 and ties 2021, and nearly all of its
edge is 2022 onward, when the line alone got worse. The ridge also beats the count
model's own ranking by +0.050 [+0.005, +0.096], p = 0.026.

Fourteen challengers, each tuned only on seasons before its test season, were run
against the ridge in `notes/model-selection-2026-09-29.md`. None is distinguishable
from it after Holm correction. The closest, the ridge plus a shrunk per-kicker
effect, is +0.006 [-0.016, +0.027], Holm p = 1.00, ahead in 6 of 8 seasons.

Note what the second row costs. The count model's expected points is a statistical
tie with the betting line, +0.029, p = 0.19. All of the edge is the ridge.

Calibration of the distribution, which a betting line cannot give you:

| threshold | predicted | realised |
|---|---|---|
| 15 or more points | 0.0846 | 0.0841 |

Within 6% at every threshold tested.

**Read that honestly.** The gain is real and small. Kicker is the position where
streaming on the implied total already captures most of the available value. This
model is worth having and is not worth agonising over.

## Quick start

```bash
pip install pandas numpy scikit-learn lightgbm pyarrow scipy

python src/kicker_analysis/build_dataset.py     # 14,519 kicker-games, 1999-2026
python src/kicker_analysis/fourth_down.py       # coach aggression, 2015-2026
python src/kicker_analysis/backtest.py          # the numbers above, ridge included
python src/kicker_analysis/predict_week.py      # this week's board
```

Set `OMP_NUM_THREADS=1`. LightGBM thread contention turns a sub-second fit into
three minutes on some machines.

Every module has an assert-based `--self-check` that runs offline in under a
second and needs no data:

```bash
for m in build_dataset features fourth_down model predict_week evaluate; do
  python src/kicker_analysis/$m.py --self-check
done
```

## Layout

```
src/kicker_analysis/
  build_dataset.py   nflverse to one row per kicker per game, with parity checks
  fourth_down.py     coach aggression on 4th down in FG range, per team-game
  features.py        lagged features, all knowable before kickoff
  model.py           RidgeRanker (ordering) + CompoundKickerModel (distribution)
  backtest.py        season-forward evaluation on the decision metric
  predict_week.py    this week's board
  evaluate.py        bootstrap utilities: prob-best, arbitrary-statistic compare
  feat_*.py          three candidate feature families, all REFUTED (see below)
legacy/              the version this replaced, with why each piece was retired
METHOD.md            the full argument, every number, and what was retracted
```

## Things that made the model look good and would have deployed badly

Three independent adversarial reviews ran against this, each required to prove
findings by running code. They were right about a lot.

- **Observed weather is not available on Tuesday.** Temperature and wind ranked
  3rd and 7th by importance. Of 255 unplayed games in the schedule file, 0 carry
  either. Training on them and serving without cost +0.092 MAE; dropping them cost
  0.002. They are replaced by *stadium climate*, the venue's typical wind and
  temperature in that month from prior seasons, which is a fact about the building.
  That turned out to be the single most valuable feature next to the betting line.
- **Retractable roof state is a game-day decision.** `closed` and `open` never
  appear on an unplayed game. That was 15.1% of rows carrying a label that cannot
  exist at forecast time.
- **The fix for that killed two features, and nobody noticed.** The
  retractable flag keyed the building on a `stadium_id` the kicker table does not
  have, fell back to the kicker's team, and since every team has played away under
  a retractable roof, every row was retractable and no row was ever indoors. Both
  flags were constant, so "16 features" were 14. They now come from the game's own
  venue, and the backtest refuses to run if any of the 16 stops varying. It made no
  measurable difference to the ranking (the pre-fix model is +0.003, p = 0.61),
  which is exactly why it hid.
- **The betting line is a closing line.** nflverse `spread_line` and `total_line`
  are the numbers at kickoff, so the backtest scores a board built from them. The
  board is valid for a lineup set close to kickoff; read on Tuesday, with an
  earlier line, expect a little less than the table above.
- **The lift was simulated, and the noise was the size of the effects.** The
  harness drew 300 random pairs a week, which moved pooled lifts by 0.01 to 0.02
  and scored a constant at -0.015. It is now computed exactly, so `backtest.py`,
  the bake-off and the board's published note all agree to the last digit.
- **An exclusion list is not a feature set.** The first version admitted every
  numeric column nobody had remembered to ban. During review, 17 columns were
  merged in and all 17 were silently accepted; three were same-game outcomes, and
  they moved MAE from 3.49 to 2.65 in about five minutes. The feature set is now an
  explicit list of 16 named columns, and all 16 are checked for variance.
- **The weekly board silently re-forecast a week that was already over.** Two
  caches with no expiry: `fetch()` kept `games.csv` forever, so every week-2 game
  still read as unplayed and `upcoming()` returned week 2 while the site header
  said week 3. Underneath it, `data/processed/_v2/kicker_games.parquet` was a
  hand-copied snapshot that **no script regenerated**, so the model trained on
  history that stopped 30 kicker-games short. A stale board and a current one are
  the same file with different numbers, and nothing distinguished them. Fixed at
  the root: live inputs now expire after 6 hours, `_v2` is retired in favour of
  the one path `build_dataset.py` writes, and `predict_week.py` refuses to build
  a week-N board unless the training history reaches week N-1.
- **The headline number could not be reproduced by the command that claimed
  it.** The README said to run `backtest.py` for "the numbers above", and the
  numbers above were the ridge's, but `backtest.py` never fit a ridge: it scored
  the count model, the career mean, the betting line and a constant, and printed
  those. Anyone checking would have got 0.362 where 0.434 was written and
  reasonably concluded the repo was overselling itself. The ridge is now a row
  in that table and in the bootstrap block, so the claim and the command agree.
  Re-run on the refreshed data it lands slightly stronger than published, 0.442
  against 0.434 and p = 0.006 against p = 0.015. Both were later superseded by
  the exact lift and the roof-flag fix: 0.432 and p = 0.001, the figures above.
- **The entire 2021 season was missing.** The upstream nflverse kicking release
  skips it and the loader only backfilled seasons after its cutoff. 561 games, and
  every metric looked healthy without them.

## Things that did not work, kept because a null result is a result

Three feature families were built and each was attacked by a separate reviewer.
All three were refuted, and re-testing under a properly regularised ridge agreed.
This table predates the roof-flag fix and the exact lift, so its baseline is the
14 live columns scored by simulation; the conclusion does not depend on either:

| added to the 16 | MAE | within-week rho | decision lift |
|---|---|---|---|
| nothing | 3.4696 | 0.1732 | 0.4342 |
| red-zone efficiency (+27 cols) | 3.4749 | 0.1517 | 0.3782 |
| drive-level offence (+21) | 3.4801 | 0.1688 | 0.4111 |
| leg strength and stadium (+13) | 3.4743 | 0.1588 | 0.3759 |

They beat their matched noise controls, so they carry information. They are
redundant, and the extra columns cost more in variance than the information is
worth.

Willingness to attempt 50+ was measured separately and is the near miss:
directionally right (+0.021 pts/wk) but **p = 0.27**, because only 34.5% of the
visible team spread is a real team property and the rest is sampling noise on a
median of 10 chances a season. Not added. Worth re-testing with more seasons.

## Method

`METHOD.md` is the full argument, written for someone who will check it. It
includes the two claims that were retracted when the backtest contradicted them.

Data: [nflverse](https://github.com/nflverse/nflverse-data), key-free.
