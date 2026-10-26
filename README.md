# Kicker Analysis

Weekly NFL kicker fantasy forecasts, with season-forward backtests and comparisons
against betting-line and career-average baselines.

In this study, the tested field-goal-attempt model had out-of-sample R-squared
of -0.054, while extra-point-attempt forecasts showed a within-week Spearman
correlation of 0.299. These results describe the tested models and evaluation
period; they do not establish a general limit on forecasting either outcome.

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

The ridge outperformed the career-average baseline under the reported tests.
Its average advantage over the betting-line baseline varied by season: the ridge loses 2019 and ties 2021, and nearly all of its
edge is 2022 onward, when the line alone got worse. The ridge also beats the count
model's own ranking by +0.050 [+0.005, +0.096], p = 0.026.

Fourteen challengers, each tuned only on seasons before its test season, were run
against the ridge in `notes/model-selection-2026-09-29.md`. None is distinguishable
from it after Holm correction. The closest, the ridge plus a shrunk per-kicker
effect, is +0.006 [-0.016, +0.027], Holm p = 1.00, ahead in 6 of 8 seasons.

The count model's advantage over the betting-line baseline was not statistically
distinguishable from zero (+0.029 points per week, p = 0.19). The ridge showed
the stronger decision lift in this evaluation.

Calibration of the distribution, which a betting line cannot give you:

| threshold | predicted | realised |
|---|---|---|
| 15 or more points | 0.0846 | 0.0841 |

Within 6% at every threshold tested.

The estimated gain is modest. In these backtests, implied team total captured
much of the measured ranking value; the ridge added a smaller improvement.

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

## Validation and forecast timing

- Features are explicitly limited to 16 pre-kickoff inputs and checked for variance.
- Stadium climate uses prior seasons. Observed game-day weather and retractable-roof
  state are excluded when unavailable at forecast time.
- Betting-line inputs are closing lines. Reported performance therefore applies
  most directly to decisions made near kickoff; earlier-week performance can differ.
- Decision lift is computed over every pair of starters in each test week.
- Live-input caches expire after six hours. Weekly board generation requires
  training history through the previous week.
- Dataset checks cover season completeness, including the 2021 backfill.

## Additional feature experiments

Three added feature families did not improve the tested baseline. This earlier
experiment used 14 varying baseline features and simulated decision lift; its
metrics are not directly comparable to the current exact-lift results above:

| added to the 16 | MAE | within-week rho | decision lift |
|---|---|---|---|
| nothing | 3.4696 | 0.1732 | 0.4342 |
| red-zone efficiency (+27 cols) | 3.4749 | 0.1517 | 0.3782 |
| drive-level offence (+21) | 3.4801 | 0.1688 | 0.4111 |
| leg strength and stadium (+13) | 3.4743 | 0.1588 | 0.3759 |

These feature sets beat matched noise controls but did not improve the baseline
in this experiment. That suggests limited incremental value under this setup.

Willingness to attempt 50+ was measured separately and is the near miss:
directionally right (+0.021 pts/wk) but **p = 0.27**, because only 34.5% of the
visible team spread is a real team property and the rest is sampling noise on a
median of 10 chances a season. Not added. Worth re-testing with more seasons.

## Method

[METHOD.md](METHOD.md) documents model design, evaluation, and limitations.

Data: [nflverse](https://github.com/nflverse/nflverse-data), key-free.

## More projects

More of my work and contact details are at [twhite.dev](https://twhite.dev).
