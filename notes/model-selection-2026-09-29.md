# Why this model: the kicker ranker bake-off, 2026-09-29

## Rerun of 2026-10-28

Everything after this section is the record of the run of 2026-09-29, and its
numbers are that run's. The bake-off was rerun on 2026-10-28 on the same data
build as `backtest.py`'s tracked CSVs, with play-by-play for every season from
2015. The tracked `reports/model_bakeoff_*.csv` are that rerun. From
`reports/model_bakeoff_summary.csv`:

- The ridge gains 0.427 points a week over a random start at R=2, 17.4% of the
  oracle's 2.458 (0.432 below).
- Ridge minus the implied total alone: +0.074 (week-block 95% CI +0.023 to
  +0.124, p = 0.0036), ahead in 6 of 8 seasons (sign test p = 0.29). It loses
  2019 by 0.126 and 2021 by 0.019.
- The closest challenger is again the ridge plus an empirical-Bayes kicker
  effect: +0.012 (95% CI -0.010 to +0.034, Holm p = 1.00), ahead in 7 of 8
  seasons. It clears the fold rule and fails the significance rule. Nested
  alpha: -0.005. No challenger is distinguishable from the ridge after Holm.

What did not change: no challenger wins, and the ridge stays. Nearly all of the
edge over the line is still 2022 onward: +0.005 in 2018 to 2021, +0.140 in 2022
to 2025 (`model_bakeoff_slices.csv`).

What changed:

- **The line claim is weaker than the text below says.** With each season as
  one observation, the t-test on the eight season differences gives +0.073,
  p = 0.095, 95% CI -0.016 to +0.162 (Wilcoxon p = 0.11;
  `reports/season_level_tests.csv`). The season-cluster bootstrap quoted below
  as support resamples eight clusters by percentile, which is too narrow; on
  the rerun it gives +0.002 to +0.137, p = 0.045, and it is not the test to
  quote. The edge is clear week to week and suggestive, not established,
  season to season.
- **The baseline family is post hoc.** The family of two for the README's
  claims was chosen after the first run (item 3 below). Under the first run's
  single family of 18, the rerun's line comparison would get Holm p = 0.058
  (16 x 0.0036).
- **The fourth-down features are populated in every fold.** On 2026-09-29
  `fourth_down.py` read whichever play-by-play seasons were cached, so the two
  fourth-down features were missing in the early folds and median-imputed
  there. The text below reads that as the features existing only from 2021 and
  2022. `fourth_down.py` now fetches every season from 2015 and stops on a gap,
  and on the rerun `prev_go_rate` keeps its sign in 8 of 8 folds (+0.050 per
  sd).
- **The compound model's probabilities are right on average only.** The
  calibration table below compares pooled rates. Row by row, P(15+) barely
  beats a constant at the base rate: Brier skill +0.001 (95% CI -0.004 to
  +0.006), calibration slope 0.55, AUC 0.55 (`reports/calibration_summary.csv`).

## The run of 2026-09-29

This note was corrected on the day it was written. Its first version had two
dead features in the shipped model, scored its outer folds with a noisy
simulation, and put the challengers and the old baseline claims in one
multiplicity family. The first two are fixed below. The third was handled by
splitting the family after that first run, so the baseline family's Holm p is
post hoc (item 3). Every number from here on is from the corrected run.

## The answer

The board is ordered by a ridge regression on 16 named pre-kickoff features, now
16 that actually vary, because nothing tried here beats it. Fourteen challengers
ran through the same eight season-forward folds (2018 to 2025), each tuned only
on seasons before its test season, and scored on the decision a manager actually
makes: start the better of two kickers drawn from the week, measured as an exact
expectation over every pair rather than a simulation. The best challenger by
lift, the ridge plus a shrunk per-kicker effect, is 0.006 points a week ahead
(95% CI -0.016 to +0.027, Holm p = 1.00) and wins 6 of 8 seasons: it clears the
fold rule and fails the significance rule, so it does not win. No challenger is
distinguishable from the ridge after Holm correction across the 14. Where they
tie, the ridge wins the tiebreaker: the simplest candidate, one fixed
hyperparameter, and coefficients whose main signs hold in 8 of 8 folds.

The size of the prize is the caveat. The ridge beats ranking by the Vegas implied
total alone by **0.079 points a week (95% CI +0.030 to +0.128, p = 0.001)**, about
3% of what an oracle gets. That survives the Holm correction of its own family,
which was set after the first run (p = 0.001), and a season-cluster bootstrap
(CI +0.008 to +0.143, p = 0.028), but the ridge beats the line in only
**6 of 8 seasons**, and a sign test cannot tell 6 of 8
from a coin (p = 0.29). The honest read: a small edge that is consistent in the
average and not yet established season by season.

## What changed between the first and the corrected run

1. **Two of the 16 features were dead.** `is_indoor_certain` was 0 and
   `is_retractable` was 1 on every row. `features.roof_flags` looked for a
   `stadium_id` the kicker table does not carry, fell back to the kicker's TEAM
   as the venue, and since every team has played away under a retractable roof,
   every row was flagged retractable. The scaler zeroed both columns, so the live
   model had 14 features while every document said 16. Fixed in `features.py`:
   the type now comes from the game's own venue, the per-game `roof` field (dome,
   outdoors, or open/closed collapsed to retractable because the setting is a
   game-day decision), and for an unplayed game, whose roof is still blank, from a
   stadium table keyed on that game's `stadium_id` and built from played games
   only. Both flags now have sd 0.36. `features.check_roof` fails the backtest if
   either goes constant again or a dome team's home games stop reading indoors.
   **Effect on the ranking: none measurable.** The pre-fix model is now a
   challenger, and it is 0.003 ahead of the fixed one (p = 0.61). The fix is for
   honesty, not lift.
2. **The outer folds were scored by simulation.** Tuning used the exact pairwise
   lift but the outer score used `backtest.py`'s 300 random draws a week, which
   moved pooled lifts by 0.01 to 0.02: measured, 0.009 on the line alone, 0.018 on
   the career average and 0.015 on a constant, which it scored at -0.015 when the
   exact value is 0. Per fold it was worse: it stretched the pre-fix ridge's 2021
   margin over the line from an exact 0.009 to 0.095. The harness itself is now exact (`backtest._pick_probs`, the closed form for
   any roster size), so `backtest.py`, this bake-off and the board's published
   note all compute the same number. The self-check proves it against brute-force
   enumeration of every roster, with ties.
3. **One Holm family mixed two questions.** The first run put the 14 challengers
   and the four reference rows in one family of 18, so the pre-existing claim that
   the ridge beats the line was penalised for questions asked after it and came
   out at Holm p = 0.081. After that result I split the correction into two
   families: the 14 challengers against the ridge, and the two claims the README
   already made (ridge vs the line, ridge vs a career average). The argument for
   the split is defensible, but it was made after seeing the result, so the
   baseline family's Holm p is post hoc. On this corrected run a single family
   of 18 would have given the line Holm p = 0.016 (16 x 0.0010), so here the
   verdict does not hang on the split; on the first run it did. The compound
   model and the constant are reported unadjusted, as description.
4. **141 weeks are not 141 independent draws.** The ridge's own lift runs from
   0.19 to 0.61 by season, so the week-block bootstrap overstates certainty about
   anything that varies by season. The baseline family now also gets a
   season-cluster bootstrap (whole seasons resampled) and a sign test across the
   eight seasons. That is where the "6 of 8, p = 0.29" above comes from. With
   eight clusters the percentile cluster bootstrap is too narrow; the section
   at the top gives the season-level t-test instead.
5. **"Knowable on Tuesday" was false for the most important feature.** The
   implied total, the total line and closeness come from nflverse closing lines.
   The board is valid for a lineup set close to kickoff, with the line as it stands
   then; read on Tuesday it uses an earlier line and should expect a little less
   than the backtest shows. `features.py` and `model.py` now say so.

## Setup

- **Data.** 5,826 regular-season kicker-games, 2015 to 2026 week 3, built by
  `model.load()`. Test seasons 2018 to 2025 (2026 has three weeks and is excluded
  by the harness's 8-week rule): 4,211 test rows, 141 weeks. Rows through 2025 are
  identical before and after the 2026 week 3 refresh (checked), and no fold trains
  on 2026, so the refresh cannot move any number here.
- **Nesting.** For test season s, each of the last three training seasons (two for
  2018) is used once as validation for a fit on the seasons before it. The config
  with the best mean inner score is refit on every season before s and scored once
  on s. Imputation, scaling, spline knots, pair construction, the Vegas offset and
  the empirical-Bayes variance components are all fit inside the training block.
- **Score.** The exact expected R=2 lift, inner and outer: over every pair of
  kickers in a week, the gain from starting the one the score prefers, ties split
  as a coin flip. Rows are laid out as `backtest.py` lays them out, so the
  reference rows reproduce its numbers exactly: ridge 0.4322, implied 0.3532,
  compound 0.3818, career 0.1797, constant 0.
- **Win rule.** Holm-adjusted p < 0.05 against the shipped ridge at R=2 within the
  challenger family (m = 14, separated from the baselines after the first run;
  item 3 above), a positive difference, and a higher R=2 lift in at
  least 6 of 8 seasons. The fold rule alone is weak (a coin wins 6 of 8 with
  probability 14.5%), which is why both are required.

## Results, pooled over 8 folds

Lift is points per week gained over starting a random one of R kickers; the oracle
gets 2.458 at R=2. `diff`, CI and p are against the shipped ridge at R=2,
week-block bootstrap, 10,000 draws. MAE and calibration slope (y regressed on the
prediction, 1.0 is ideal) only for models whose output is on the points scale.

| model | lift R=2 | share of oracle | lift R=3 | rho | MAE | cal slope | diff vs ridge | 95% CI | p | Holm p | folds won | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ridge, 16 features, alpha 100 (shipped)** | **0.4322** | **17.6%** | **0.6329** | **0.174** | 3.469 | 0.97 | reference | | | | | |
| *challengers, Holm m = 14* | | | | | | | | | | | | |
| ridge + EB kicker effect | 0.4378 | 17.8% | 0.6477 | 0.178 | 3.474 | 0.90 | +0.0056 | [-0.016, +0.027] | 0.61 | 1.00 | 6 | no difference |
| L2 Poisson GLM | 0.4365 | 17.8% | 0.6416 | 0.176 | 3.469 | 1.02 | +0.0043 | [-0.005, +0.014] | 0.38 | 1.00 | 4 | no difference |
| ridge, roof flags dead (as shipped before the fix) | 0.4349 | 17.7% | 0.6358 | 0.176 | 3.467 | 0.98 | +0.0027 | [-0.008, +0.013] | 0.61 | 1.00 | 6 | no difference |
| L2 Tweedie GLM | 0.4343 | 17.7% | 0.6406 | 0.175 | 3.468 | 1.09 | +0.0021 | [-0.011, +0.016] | 0.75 | 1.00 | 5 | no difference |
| monotone depth-2 GBM | 0.4307 | 17.5% | 0.6291 | 0.172 | 3.468 | 0.97 | -0.0015 | [-0.042, +0.038] | 0.96 | 1.00 | 4 | no difference |
| rank blend ridge/implied, nested weight | 0.4294 | 17.5% | 0.6373 | 0.174 | | | -0.0028 | [-0.021, +0.016] | 0.78 | 1.00 | 3 | no difference |
| Vegas-anchored residual ridge | 0.4280 | 17.4% | 0.6348 | 0.173 | 3.469 | 0.96 | -0.0041 | [-0.023, +0.014] | 0.68 | 1.00 | 2 | no difference |
| decomposed ridge (FG + PAT) | 0.4247 | 17.3% | 0.6244 | 0.171 | 3.469 | 1.18 | -0.0075 | [-0.023, +0.009] | 0.37 | 1.00 | 3 | no difference |
| linear pairwise ranker | 0.4209 | 17.1% | 0.6155 | 0.168 | | | -0.0113 | [-0.030, +0.009] | 0.27 | 1.00 | 2 | no difference |
| ridge, nested alpha | 0.4208 | 17.1% | 0.6138 | 0.168 | 3.469 | 1.11 | -0.0114 | [-0.026, +0.003] | 0.13 | 1.00 | 3 | no difference |
| elastic net | 0.4164 | 16.9% | 0.6242 | 0.168 | 3.470 | 1.18 | -0.0158 | [-0.040, +0.008] | 0.19 | 1.00 | 2 | no difference |
| within-week demeaned ridge | 0.4144 | 16.9% | 0.6050 | 0.166 | | | -0.0178 | [-0.034, -0.002] | 0.031 | 0.44 | 3 | no difference |
| spline ridge (GAM-style) | 0.4118 | 16.8% | 0.6067 | 0.167 | 3.470 | 1.05 | -0.0203 | [-0.045, +0.004] | 0.098 | 1.00 | 3 | no difference |
| LightGBM lambdarank | 0.3872 | 15.8% | 0.5626 | 0.156 | | | -0.0449 | [-0.092, +0.001] | 0.057 | 0.74 | 1 | no difference |
| *baselines, Holm m = 2, post hoc (item 3)* | | | | | | | | | | | | |
| Vegas implied total alone | 0.3532 | 14.4% | 0.5137 | 0.143 | | | -0.0790 | [-0.128, -0.030] | 0.0010 | 0.0010 | 2 | ridge better |
| kicker career average | 0.1797 | 7.3% | 0.2940 | 0.071 | 3.571 | 0.29 | -0.2524 | [-0.341, -0.161] | <0.0001 | <0.0001 | 0 | ridge better |
| *descriptive, unadjusted* | | | | | | | | | | | | |
| compound count model E[y] | 0.3818 | 15.5% | 0.5497 | 0.156 | 3.492 | 0.97 | -0.0504 | [-0.096, -0.005] | 0.026 | | 1 | |
| constant (training median) | 0.0000 | 0.0% | 0.0000 | | 3.504 | | -0.4322 | [-0.517, -0.347] | <0.0001 | | 0 | |

Four challengers are nominally ahead at R=2 and none is close to significant.
**Power:** for the linear cousins the CI half-width is about 0.01 to 0.02 points a
week, so any gain from reshaping the model on these features larger than roughly
0.02 (under 1% of the oracle) would have shown. The tree and rank models have CIs
two to four times as wide.

## The ridge against the line, the claim that matters

| test | ridge minus line | interval | p |
|---|---|---|---|
| week-block bootstrap, 141 weeks | +0.079 | [+0.030, +0.128] | 0.001 (Holm, m = 2, post hoc: 0.001) |
| season-cluster bootstrap, 8 seasons | +0.079 | [+0.008, +0.143] | 0.028 |
| sign test across seasons | 6 of 8 | | 0.29 |

Against a career average the same three read +0.252 [+0.161, +0.341] p < 0.0001,
cluster [+0.169, +0.348] p < 0.0001, and 8 of 8 seasons, p = 0.008. That claim is
solid by every test. The line claim is solid in the average and thin by season.

## Per fold, R=2 lift

| model | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| **ridge (shipped)** | **0.412** | **0.406** | **0.536** | **0.389** | **0.187** | **0.379** | **0.606** | **0.546** |
| ridge + EB kicker effect | 0.418 | 0.427 | 0.551 | 0.398 | 0.236 | 0.375 | 0.550 | 0.552 |
| L2 Poisson GLM | 0.408 | 0.405 | 0.556 | 0.397 | 0.205 | 0.372 | 0.598 | 0.554 |
| ridge, roof flags dead (pre-fix) | 0.413 | 0.411 | 0.537 | 0.401 | 0.196 | 0.375 | 0.585 | 0.564 |
| L2 Tweedie GLM | 0.412 | 0.395 | 0.553 | 0.397 | 0.206 | 0.366 | 0.595 | 0.552 |
| monotone depth-2 GBM | 0.342 | 0.517 | 0.514 | 0.395 | 0.166 | 0.320 | 0.640 | 0.557 |
| rank blend ridge/implied | 0.384 | 0.481 | 0.484 | 0.392 | 0.170 | 0.381 | 0.605 | 0.542 |
| Vegas-anchored residual ridge | 0.404 | 0.465 | 0.467 | 0.397 | 0.174 | 0.379 | 0.595 | 0.545 |
| decomposed ridge (FG + PAT) | 0.361 | 0.404 | 0.533 | 0.390 | 0.181 | 0.364 | 0.616 | 0.551 |
| linear pairwise ranker | 0.403 | 0.387 | 0.536 | 0.397 | 0.200 | 0.319 | 0.593 | 0.537 |
| ridge, nested alpha | 0.357 | 0.382 | 0.536 | 0.397 | 0.200 | 0.355 | 0.599 | 0.542 |
| elastic net | 0.367 | 0.464 | 0.458 | 0.376 | 0.154 | 0.379 | 0.593 | 0.542 |
| within-week demeaned ridge | 0.350 | 0.381 | 0.540 | 0.396 | 0.198 | 0.312 | 0.604 | 0.535 |
| spline ridge | 0.363 | 0.458 | 0.527 | 0.339 | 0.192 | 0.299 | 0.568 | 0.554 |
| LightGBM lambdarank | 0.363 | 0.545 | 0.408 | 0.374 | 0.166 | 0.228 | 0.566 | 0.457 |
| compound count model E[y] | 0.373 | 0.475 | 0.510 | 0.379 | 0.182 | 0.234 | 0.479 | 0.434 |
| Vegas implied total alone | 0.362 | 0.516 | 0.425 | 0.392 | 0.145 | 0.200 | 0.470 | 0.329 |
| kicker career average | 0.224 | 0.268 | 0.185 | 0.179 | 0.060 | 0.250 | 0.105 | 0.174 |
| oracle | 2.289 | 2.309 | 2.461 | 2.472 | 2.391 | 2.490 | 2.637 | 2.599 |

The ridge loses to the line in 2019 (by 0.110) and 2021 (by 0.003, a tie). The
pre-fix ridge won 2021 by 0.009, so the old 7 of 8 became 6 of 8 through a tie
changing sides, not through anything the roof fix taught the model. The
season-to-season swing in the ridge's own lift, 0.19 in 2022 to 0.61 in 2024, is
far larger than any gap between candidates, which is why a fold count alone cannot
pick a winner.

## What was tried, and why each lost or tied

1. **Empirical-Bayes kicker effect.** The closest (+0.006, 6 of 8 seasons, Holm
   1.00). The between-kicker residual sd is 0.38 points against 4.30 of week noise,
   so what is left of "the kicker" after the features is small. It is the one
   candidate worth re-testing once 2026 is in, and only as a single pre-registered
   comparison.
2. **L2 Poisson and Tweedie GLMs.** Tie (+0.004, +0.002). The log link is a
   monotone reshaping of the same linear score, so the ranking barely moves. As a
   distribution, Poisson on points is badly wrong in the tail (predicts 2.2% of
   weeks at 15+, realised 8.4%); Tweedie gets 15+ nearly right (1.02) but misses
   the middle thresholds by 6 to 9%, so the compound model remains the better
   probability machine (table below).
3. **The pre-fix ridge.** Ties (+0.003, p = 0.61). The repaired roof flags carry a
   stable retractable coefficient (+0.07 points per sd, same sign 8 of 8) and an
   indoor coefficient that flips sign with era (4 of 8), and together they buy
   nothing on ranking.
4. **Nested alpha.** Ties (-0.011, p = 0.13). The inner loop's choice runs from
   10,000 in 2018 (three training seasons) to 300 through the middle folds and 1 in
   2025. The shipped 100 was read off these same folds, which biases the
   comparison toward the reference, and the nested version still only ties.
5. **Vegas-anchored residual ridge.** Ties (-0.004). Shrinking toward the line
   instead of toward a constant changes nothing, because at alpha 100 the ridge
   already puts its largest weight on the implied total.
6. **Spline ridge.** Loses nominally (-0.020, p = 0.10). The inner loop picked 3
   knots in 7 of 8 folds and heavy shrinkage in most; there is no exploitable
   curvature in the line, wind exposure or recent volume at this sample size.
7. **Learning to rank.** The linear pairwise ranker ties (-0.011): optimising the
   decision's own pairwise loss directly buys nothing over least squares.
   LightGBM lambdarank loses (-0.045) and trails the ridge in 7 of 8 seasons.
8. **Monotone depth-2 GBM.** Ties (-0.002) with the widest interval of any linear
   rival. Honest about the direction of the line and no better than the linear fit.
9. **Nested rank blend.** Ties (-0.003). The weight on the ridge sits between 0.3
   and 0.7 in the first five folds and at 0.9 in the last three: as training data
   grows the inner loop learns to trust the ridge over the raw line.
10. **Cheap extras.** Within-week demeaned ridge: -0.018 (raw p = 0.03, Holm 0.44).
    Elastic net: keeps 15 of 16 columns in the last fold and -0.016. Decomposed
    ridge (separate shrinkage for field-goal and extra-point points): the inner loop
    chose heavy FG shrinkage early (alpha 1,000 to 10,000), as the "field goals have
    no skill" finding predicts, and the sum ties the single ridge (-0.008). Line
    movement was not tested: `games.csv` carries one closing spread and total per
    game, no opening line.

## Calibration, where the model outputs a distribution

| threshold | compound pred / real | ratio | Tweedie GLM ratio | Poisson GLM ratio |
|---|---|---|---|---|
| 8 or more | 0.546 / 0.524 | 1.04 | 0.94 | 1.06 |
| 10 or more | 0.375 / 0.356 | 1.05 | 0.91 | 0.84 |
| 12 or more | 0.224 / 0.218 | 1.03 | 0.91 | 0.58 |
| 15 or more | 0.0846 / 0.0841 | 1.01 | 1.02 | 0.27 |

The ridge is a point forecaster and its mean calibration is fine (slope 0.97).
Rank with the ridge, take probabilities from the compound model.

## Slices

| R=2 lift | weeks 1-4 | weeks 5+ | 2018-2021 | 2022-2025 |
|---|---|---|---|---|
| ridge (shipped) | 0.280 | 0.477 | 0.435 | 0.430 |
| ridge + EB kicker effect | 0.282 | 0.484 | 0.448 | 0.428 |
| L2 Poisson GLM | 0.291 | 0.479 | 0.441 | 0.432 |
| monotone depth-2 GBM | 0.364 | 0.450 | 0.441 | 0.420 |
| Vegas implied total alone | 0.258 | 0.381 | 0.423 | 0.286 |

The ridge's edge over the line lives in weeks 5+ (+0.096), where the lagged
volume features have filled in; in weeks 1 to 4 it is +0.022. And the line alone
got much worse in 2022 to 2025 (0.423 to 0.286) while the ridge did not move, so
nearly all of the edge is recent: +0.012 in 2018 to 2021, +0.143 in 2022 to 2025.
The early-week cells favour the GBM, but that is a slice read after the fact
across 18 models, a lead and not a finding.

## Interpretability and simplicity, the tiebreaker

The shipped ridge, standardised coefficients in points per sd, mean over the 8
folds and how many folds keep that sign:

| feature | mean | same sign | | feature | mean | same sign |
|---|---|---|---|---|---|---|
| implied_total | +0.466 | 8/8 | | k_fantasy_points_l8 | +0.370 | 8/8 |
| opp_implied_total | -0.298 | 8/8 | | climate_wind_exposed | -0.179 | 8/8 |
| total_line | +0.151 | 8/8 | | t_team_score_l8 | -0.133 | 7/8 |
| t_pat_att_l8 | -0.103 | 7/8 | | k_fg_att_l8 | -0.079 | 7/8 |
| is_retractable | +0.068 | 8/8 | | k_fg_pct_career | +0.059 | 8/8 |
| t_fg_share_l8 | -0.054 | 4/4 | | t_fg_att_l8 | -0.028 | 5/8 |
| prev_go_rate | -0.023 | 3/3 | | closeness | +0.017 | 6/8 |
| is_indoor_certain | -0.009 | 4/8 | | is_home | -0.005 | 6/8 |

The largest coefficients keep their sign in 8 of 8 folds and each has a stated
mechanism: the offence's own implied points up, the opponent's down (game script),
wind down, recent kicker output up. The fourth-down features only exist from 2021
and 2022, so they are live in 3 and 4 folds. (That was a gap in this run's
play-by-play cache, not in the data; see the section at the top.) Among the
models that tie, the ridge has the fewest moving parts: one fixed
hyperparameter, no link function, no dispersion or power parameter, no
dependence on the week's slate (the blend and the pairwise ranker both need it),
and one coefficient per named feature.

## Leakage register

Prediction moment: a lineup set close to kickoff, after the closing line.

| risk | class | evidence | severity | effect on this comparison |
|---|---|---|---|---|
| market features are closing lines | timing | nflverse `spread_line`/`total_line` are closing; `predict_week.py` reads the current file | medium | the board is valid near kickoff; read earlier, every Vegas-driven candidate's absolute lift is optimistic, and none is favoured over the ridge |
| shipped alpha = 100 was read off these test folds | tuning on test | METHOD.md says so | low | favours the reference; the nested ridge ties it anyway |
| the 16 features were chosen after three families were rejected on these same folds | selection on test | METHOD.md feature table | low to medium | shared by every candidate, inflates absolute lift, not the ranking |
| fourth-down features absent before 2021/2022 | population mismatch across eras | all-NaN in early folds, dropped by the imputer; a gap in the play-by-play cache, fixed since (top section) | low | slice table shows no era break for the ridge |
| retractable roof open or closed | game-day decision | never set on an unplayed game | none now | both settings map to one "retractable" type; an unplayed game takes its type from a table built from played games only |

All challengers' preprocessing is fit inside the training block, and no test-season
row enters tuning.

## What would be worth trying next

1. **Re-run after 2026 completes.** Both the nested alpha and the blend weight
   drift toward trusting the features as training data grows, and the edge over
   the line is almost all 2022 onward. A ninth season is the cheapest way to move
   the sign test off 6 of 8. It is a rerun, not a rebuild.
2. **The EB kicker effect as one pre-registered test.** It is the only challenger
   that already wins 6 of 8 seasons.
3. **An early-season prior.** Weeks 1 to 4 are where the ridge collapses to the
   line. Carrying prior-season team volume into the first four weeks has a clear
   mechanism; pre-register it as one test.
4. **Opening lines**, only if a key-free historical source turns up. They would
   remove the closing-line optimism above and allow a line-movement feature.
5. **Nothing else on the model side.** Fourteen shapes of model on these features
   land within noise of the ridge. Further gains, if any, are in inputs.

## In plain language

"I rank fantasy kickers with a ridge regression on 16 inputs that exist before
kickoff: mostly the Vegas line, plus the offence's recent volume, the coach's
fourth-down habits and the stadium. I tested 14 alternatives, from count GLMs to
gradient-boosted rankers, each tuned only on seasons before the one it was scored
on, and none beat it, so I kept the simplest. Against just starting the kicker
whose team Vegas expects to score most, it gains about 0.08 points a week. That is
clear week to week, but it only wins 6 of 8 seasons, so I call it a small,
consistent edge and not a big one. The real lesson is that most of a kicker's week
is field goals, which are close to unpredictable, and the betting line already
knows most of what can be known. The first run also had two mistakes of mine in
it: two features were constant because of a join bug, and I had been scoring
with a noisy simulation. Fixing both left the choice of model where it was."

## Reproduce

```
export OMP_NUM_THREADS=1
python src/kicker_analysis/model_bakeoff.py --self-check   # offline, a few seconds
python src/kicker_analysis/model_bakeoff.py --quick        # 3 folds, 1 inner fold, a smoke test
python src/kicker_analysis/model_bakeoff.py                # the full run: 148 s on 2026-10-28
```

Seeds are fixed (bootstrap seed 0, LightGBM `deterministic=True`, one thread) and
the lift is exact, so a rerun on the same inputs reproduces every number.
`reports/data_manifest.json` records the inputs behind the tracked CSVs, which
are the 2026-10-28 rerun, not the run described above. Outputs:
`reports/model_bakeoff_summary.csv` (with a `family` column and the season-cluster
and sign-test columns on the baseline rows), `_folds.csv`, `_params.csv` (every
chosen hyperparameter per fold), `_calibration.csv`, `_slices.csv`,
`_ridge_coefs.csv`, and the ignored `_predictions.parquet`.
