# Why this model: the kicker ranker bake-off, 2026-09-29

## The answer

The board is ordered by a ridge regression on 16 named pre-kickoff features
because nothing tried here beats it. Fourteen challengers were run through the
same eight season-forward folds (2018 to 2025), each tuned only on seasons before
its test season, and scored on the decision a manager actually makes: start the
better of two kickers drawn from the week. Not one has a higher pooled lift than
the ridge's 0.4415 points a week. The closest, a nested blend of the ridge's rank
and the Vegas implied total's rank, sits 0.001 behind (95% CI -0.022 to +0.021).
The count-data GLMs, a Vegas-anchored residual ridge, a pairwise ranker and a
nested alpha all land within 0.01, and the nonlinear models (splines, a monotone
depth-2 GBM, LightGBM lambdarank) lose 0.03 to 0.05. After Holm correction across
18 comparisons no challenger is distinguishable from the ridge, and none beats it
in more than 4 of 8 seasons against the 6 required. Where they tie, the ridge wins
the tiebreaker: it is the simplest candidate, with one fixed hyperparameter and
coefficients whose main signs hold in 8 of 8 folds, and every point estimate says
the extra machinery costs a little rather than earning anything. The caveat is the
size of the prize. The ridge beats ranking by the line alone by 0.080 points a week
(p = 0.005, 7 of 8 seasons), about 3% of what an oracle gets, and that edge is the
one number here that would not survive treating this whole bake-off as a single
family of tests (Holm p = 0.08).

## Setup, fixed before the run

- **Data.** 5,794 regular-season kicker-games, 2015 to 2026, built by `model.load()`.
  Test seasons 2018 to 2025 (2026 has two weeks and is excluded by the harness's
  8-week rule): 4,211 test rows, 141 weeks.
- **Nesting.** For test season s, each of the last three training seasons (two for
  2018) is used once as validation for a fit on the seasons before it. The config
  with the best mean inner score is refit on every season before s and scored once
  on s. Imputation, scaling, spline knots, pair construction, the Vegas offset and
  the empirical-Bayes variance components are all fit inside the training block.
- **Inner score.** The exact expected R=2 lift (every pair in a week, the gain from
  starting the one the score prefers). It is the expectation the harness simulates,
  without the draw noise, so tuning and evaluation aim at the same number. A
  self-check asserts the two agree.
- **Outer score.** The repo's harness, unchanged: `backtest.decision`,
  `per_week_lift`, `rho`, `boot_lift` (paired week-block bootstrap, 10,000 draws).
  Rows are laid out as `backtest.py` lays them out, so the reference rows
  reproduce its published numbers exactly: ridge 0.4415, implied 0.3620, compound
  0.3621, career 0.1613, constant -0.0154.
- **Win rule.** Holm-adjusted p < 0.05 against the shipped ridge at R=2, a
  positive difference, and a higher R=2 lift in at least 6 of 8 seasons. The Holm
  family is every row except the reference, m = 18. The fold rule alone is weak
  (a coin wins 6 of 8 with probability 14.5%), which is why both are required.

## Results, pooled over 8 folds

Lift is points per week gained over starting a random one of R kickers; the oracle
gets 2.440 at R=2. `diff`, CI and p are against the shipped ridge at R=2. MAE and
calibration slope (y regressed on the prediction, 1.0 is ideal) only for models
whose output is on the points scale.

| model | lift R=2 | share of oracle | lift R=3 | rho | MAE | cal slope | diff vs ridge | 95% CI | p | Holm p | folds won | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ridge, 16 features, alpha 100 (shipped)** | **0.4415** | **18.1%** | **0.6405** | **0.176** | 3.467 | 0.98 | reference | | | | | |
| rank blend ridge/implied, nested weight | 0.4403 | 18.0% | 0.6369 | 0.174 | | | -0.0013 | [-0.022, +0.021] | 0.91 | 1.00 | 2 | no difference |
| L2 Poisson GLM | 0.4401 | 18.0% | 0.6475 | 0.176 | 3.467 | 1.01 | -0.0015 | [-0.014, +0.011] | 0.81 | 1.00 | 2 | no difference |
| L2 Tweedie GLM | 0.4381 | 18.0% | 0.6390 | 0.174 | 3.466 | 1.05 | -0.0035 | [-0.018, +0.011] | 0.66 | 1.00 | 3 | no difference |
| ridge, nested alpha | 0.4367 | 17.9% | 0.6342 | 0.172 | 3.468 | 1.08 | -0.0049 | [-0.023, +0.014] | 0.61 | 1.00 | 1 | no difference |
| ridge, roof flags repaired | 0.4364 | 17.9% | 0.6445 | 0.174 | 3.469 | 0.97 | -0.0052 | [-0.018, +0.009] | 0.45 | 1.00 | 2 | no difference |
| Vegas-anchored residual ridge | 0.4350 | 17.8% | 0.6433 | 0.175 | 3.468 | 0.97 | -0.0065 | [-0.027, +0.014] | 0.55 | 1.00 | 4 | no difference |
| linear pairwise ranker | 0.4343 | 17.8% | 0.6389 | 0.172 | | | -0.0073 | [-0.026, +0.012] | 0.47 | 1.00 | 3 | no difference |
| decomposed ridge (FG + PAT) | 0.4331 | 17.8% | 0.6419 | 0.174 | 3.466 | 1.11 | -0.0084 | [-0.027, +0.011] | 0.39 | 1.00 | 2 | no difference |
| within-week demeaned ridge | 0.4258 | 17.5% | 0.6311 | 0.169 | | | -0.0158 | [-0.035, +0.004] | 0.12 | 1.00 | 1 | no difference |
| ridge + EB kicker effect | 0.4234 | 17.4% | 0.6713 | 0.179 | 3.472 | 0.91 | -0.0181 | [-0.045, +0.008] | 0.18 | 1.00 | 3 | no difference |
| elastic net | 0.4230 | 17.3% | 0.6262 | 0.171 | 3.472 | 1.25 | -0.0185 | [-0.050, +0.013] | 0.25 | 1.00 | 2 | no difference |
| monotone depth-2 GBM | 0.4118 | 16.9% | 0.5850 | 0.165 | 3.473 | 0.94 | -0.0297 | [-0.073, +0.014] | 0.18 | 1.00 | 2 | no difference |
| spline ridge (GAM-style) | 0.4107 | 16.8% | 0.5947 | 0.166 | 3.468 | 1.12 | -0.0309 | [-0.059, -0.003] | 0.031 | 0.43 | 3 | no difference |
| LightGBM lambdarank | 0.3907 | 16.0% | 0.5591 | 0.160 | | | -0.0508 | [-0.105, +0.003] | 0.065 | 0.84 | 1 | no difference |
| compound count model E[y] | 0.3621 | 14.8% | 0.5208 | 0.148 | 3.494 | 0.95 | -0.0794 | [-0.129, -0.030] | 0.0016 | 0.026 | 1 | ridge better |
| Vegas implied total alone | 0.3620 | 14.8% | 0.4934 | 0.143 | | | -0.0795 | [-0.137, -0.023] | 0.0054 | 0.081 | 1 | no difference |
| kicker career average | 0.1613 | 6.6% | 0.2877 | 0.071 | 3.571 | 0.29 | -0.2802 | [-0.375, -0.184] | <0.0001 | <0.0001 | 0 | ridge better |
| constant (training median) | -0.0154 | -0.6% | 0.0076 | | 3.504 | | -0.4570 | [-0.551, -0.362] | <0.0001 | <0.0001 | 0 | ridge better |

Every challenger's R=2 point estimate is below the ridge. At R=3, five are
nominally ahead (EB kicker effect +0.031, p = 0.21; Poisson +0.007; roof repair
+0.004; Vegas residual +0.003; decomposed +0.001) and none is close to
significant. **Power:** for the linear cousins the CI half-width is about 0.012 to
0.02 points a week, so any gain from reshaping the model on these features larger
than roughly 0.02 (under 1% of the oracle) would have shown. The tree and rank
models have CIs two to three times as wide and still point down.

## Per fold, R=2 lift

| model | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| **ridge (shipped)** | **0.365** | **0.428** | **0.628** | **0.416** | **0.193** | **0.409** | **0.549** | **0.551** |
| ridge, nested alpha | 0.346 | 0.409 | 0.617 | 0.416 | 0.193 | 0.408 | 0.572 | 0.537 |
| Vegas-anchored residual ridge | 0.399 | 0.486 | 0.514 | 0.375 | 0.191 | 0.410 | 0.571 | 0.539 |
| L2 Poisson GLM | 0.384 | 0.408 | 0.617 | 0.415 | 0.192 | 0.389 | 0.573 | 0.548 |
| L2 Tweedie GLM | 0.371 | 0.413 | 0.629 | 0.409 | 0.192 | 0.384 | 0.537 | 0.575 |
| spline ridge | 0.364 | 0.486 | 0.581 | 0.346 | 0.202 | 0.272 | 0.485 | 0.560 |
| linear pairwise ranker | 0.393 | 0.406 | 0.598 | 0.404 | 0.209 | 0.384 | 0.549 | 0.538 |
| LightGBM lambdarank | 0.357 | 0.498 | 0.437 | 0.396 | 0.146 | 0.294 | 0.535 | 0.469 |
| ridge + EB kicker effect | 0.367 | 0.423 | 0.593 | 0.376 | 0.212 | 0.399 | 0.449 | 0.576 |
| monotone depth-2 GBM | 0.306 | 0.528 | 0.574 | 0.367 | 0.174 | 0.329 | 0.567 | 0.459 |
| rank blend ridge/implied | 0.359 | 0.549 | 0.540 | 0.416 | 0.165 | 0.409 | 0.549 | 0.543 |
| within-week demeaned ridge | 0.338 | 0.406 | 0.609 | 0.395 | 0.200 | 0.381 | 0.548 | 0.533 |
| elastic net | 0.344 | 0.530 | 0.504 | 0.379 | 0.134 | 0.419 | 0.532 | 0.548 |
| decomposed ridge (FG + PAT) | 0.354 | 0.437 | 0.598 | 0.400 | 0.187 | 0.391 | 0.576 | 0.526 |
| ridge, roof flags repaired | 0.362 | 0.417 | 0.620 | 0.403 | 0.179 | 0.397 | 0.564 | 0.556 |
| compound count model E[y] | 0.327 | 0.468 | 0.581 | 0.287 | 0.174 | 0.225 | 0.419 | 0.432 |
| Vegas implied total alone | 0.343 | 0.565 | 0.475 | 0.321 | 0.107 | 0.250 | 0.498 | 0.352 |
| kicker career average | 0.206 | 0.244 | 0.120 | 0.107 | 0.081 | 0.280 | 0.058 | 0.199 |
| constant | 0.023 | 0.040 | -0.014 | -0.009 | -0.041 | 0.010 | -0.051 | -0.076 |

The season-to-season swing in the ridge's own lift (0.19 in 2022 to 0.63 in 2020)
is far larger than any gap between candidates, which is why a fold count alone
cannot pick a winner.

## What was tried, and why each lost or tied

1. **Nested alpha.** Ties (-0.005, p = 0.61). The inner loop's choice drifts from
   10,000 in 2018 (three training seasons) down to 30 by 2023 to 2025: with more
   data the optimum shrinks less, and the shipped 100 sits in the middle. The
   shipped alpha was read off these same folds, which biases the comparison toward
   the reference, and it still only ties, so the tuning does no work.
2. **Vegas-anchored residual ridge.** Ties (-0.007, p = 0.55), the most folds won
   (4 of 8). Shrinking toward the line instead of toward a constant changes
   nothing, because at alpha 100 the ridge already puts its largest weight on the
   implied total.
3. **L2 Poisson and Tweedie GLMs.** Tie (-0.002 and -0.004). The log link is a
   monotone reshaping of the same linear score, so the ranking barely moves. The
   count link does not help ranking. As a distribution, Poisson on points is badly
   wrong in the tail (predicts 2.2% of weeks at 15+, realised 8.4%); Tweedie gets
   15+ right (1.03) but misses the middle thresholds by 6 to 9%, so the compound
   model remains the better probability machine (table below).
4. **Spline ridge.** Loses (-0.031, raw p = 0.03, Holm 0.43). The inner loop picked
   a stiff fit in every fold (alpha 1,000 to 10,000, 3 knots in 7 of 8) and it
   still lost; there is no exploitable curvature in implied total, total line, wind
   exposure or recent volume at this sample size.
5. **Learning to rank.** The linear pairwise ranker ties (-0.007): optimising the
   decision's own pairwise loss directly buys nothing over least squares.
   LightGBM lambdarank loses (-0.051) and trails the ridge in 7 of 8 seasons.
6. **Empirical-Bayes kicker effect.** Ties (-0.018 at R=2, +0.031 at R=3, neither
   significant). The between-kicker residual sd is 0.38 points against 4.3 of
   week noise, so what is left of "the kicker" after the features is small, and
   its sign flips with roster size.
7. **Monotone depth-2 GBM.** Loses (-0.030). Honest about the direction of the
   line and still worse than the linear fit.
8. **Nested rank blend.** Closest of all (-0.001). The weight on the ridge drifts
   from 0.4 in the early folds to 0.9 to 1.0 in the last three: as training data
   grows, the inner loop learns to trust the ridge over the raw line. It converges
   to the shipped model.
9. **Cheap extras, each motivated by METHOD.md or the brief.** Within-week
   demeaned ridge (the decision is within a week, so remove between-week
   variation): -0.016. Elastic net (can fewer columns tie?): keeps 8 of 16 columns
   in the last fold, -0.019, loses 6 of 8 seasons. Decomposed ridge (separate
   shrinkage for field-goal and extra-point points, from the variance
   decomposition): the inner loop did choose heavy FG shrinkage early (alpha 1,000
   to 10,000), as the "field goals have no skill" finding predicts, but the sum
   ties the single ridge (-0.008). Line movement was not tested: `games.csv`
   carries one spread and one total per game (closing), no opening line, and no
   keyed source was added.

## Calibration, where the model outputs a distribution

| threshold | compound pred / real | ratio | Tweedie GLM ratio | Poisson GLM ratio |
|---|---|---|---|---|
| 8 or more | 0.545 / 0.524 | 1.04 | 0.94 | 1.06 |
| 10 or more | 0.374 / 0.356 | 1.05 | 0.91 | 0.84 |
| 12 or more | 0.224 / 0.218 | 1.03 | 0.92 | 0.58 |
| 15 or more | 0.084 / 0.084 | 1.00 | 1.03 | 0.27 |

The ridge is a point forecaster and its mean calibration is fine (slope 0.98, bias
-0.07 points). The split the repo already uses stands: rank with the ridge, take
probabilities from the compound model.

## Slices

| R=2 lift | weeks 1-4 | weeks 5+ | 2018-2021 | 2022-2025 |
|---|---|---|---|---|
| ridge (shipped) | 0.294 | 0.485 | 0.458 | 0.425 |
| rank blend | 0.312 | 0.478 | 0.465 | 0.416 |
| L2 Poisson GLM | 0.316 | 0.477 | 0.455 | 0.426 |
| monotone depth-2 GBM | 0.352 | 0.429 | 0.443 | 0.382 |
| Vegas implied total alone | 0.284 | 0.385 | 0.425 | 0.302 |

Two things the pooled table hides. The ridge's edge over the line lives in weeks
5+ (+0.100), where the lagged volume features have filled in; in weeks 1 to 4 it is
+0.010, a tie. And the line alone got much worse in 2022 to 2025 (0.425 to 0.302)
while the ridge barely moved, so the ridge's edge over the line is larger in the
recent era (+0.123) than the early one (+0.033). The early-week cells favour the
GBM and the GLM, but those are slices read after the fact across 18 models, so
they are leads, not findings.

## Interpretability and simplicity, the tiebreaker

The shipped ridge, standardised coefficients in points per sd, mean over the 8
folds and how many folds keep that sign:

| feature | mean | same sign | | feature | mean | same sign |
|---|---|---|---|---|---|---|
| implied_total | +0.467 | 8/8 | | k_fantasy_points_l8 | +0.367 | 8/8 |
| opp_implied_total | -0.300 | 8/8 | | climate_wind_exposed | -0.215 | 8/8 |
| total_line | +0.149 | 8/8 | | t_team_score_l8 | -0.135 | 7/8 |
| t_pat_att_l8 | -0.102 | 7/8 | | k_fg_att_l8 | -0.079 | 7/8 |
| k_fg_pct_career | +0.061 | 8/8 | | t_fg_share_l8 | -0.054 | 4/4 |
| t_fg_att_l8 | -0.026 | 5/8 | | prev_go_rate | -0.022 | 3/3 |
| closeness | +0.017 | 6/8 | | is_home | -0.005 | 6/8 |
| is_indoor_certain | 0 | dead | | is_retractable | 0 | dead |

The five largest coefficients keep their sign in 8 of 8 folds and each has a
stated mechanism: the offence's
own implied points up, the opponent's down (game script), wind down, recent kicker
output up. The fourth-down features only exist from 2021 and 2022, so they are
live in 3 and 4 folds. Among the models that tie, the ridge has the fewest moving
parts: one fixed hyperparameter, no link function, no dispersion or power
parameter, no dependence on the week's slate (the blend and the pairwise ranker
both need it), and one coefficient per named feature.

## A finding outside the brief: two of the 16 features are dead

`is_indoor_certain` is 0 and `is_retractable` is 1 on every one of the 5,794 rows.
`features.roof_flags` looks for `stadium_id` or `stadium` on the kicker frame,
finds neither, falls back to the kicker's **team** as the venue, and since every
team has played away under a retractable roof, every team is flagged retractable.
The ridge standardises them to zero columns, so the live feature set is 14, not
16, and the docs overstate it. This is not leakage and not train/serve skew (the
board's rows carry no `stadium_id` either, so they get the same constants). Repairing the flags from `games.csv` `stadium_id`
(done inside `model_bakeoff.py` only) makes them real (sd 0.36) and changes
nothing measurable (-0.005, p = 0.45), so the fix is for honesty, not lift. No
existing file was changed.

## Leakage register

Prediction moment: after lines are posted, before kickoff.

| risk | class | evidence | severity | effect on this comparison |
|---|---|---|---|---|
| lines in the backtest are closing lines; the board uses whatever line is in `games.csv` when it runs | future knowledge | nflverse `spread_line`/`total_line` are closing; `predict_week.py` reads the current file | medium | inflates every Vegas-driven candidate's absolute lift; does not favour any challenger over the ridge |
| shipped alpha = 100 was read off these test folds | tuning on test | METHOD.md says so | low | favours the reference; the nested ridge ties it anyway |
| the 16 features were chosen after three families were rejected on these same folds | selection on test | METHOD.md feature table | low to medium | shared by every candidate, inflates absolute lift, not the ranking |
| fourth-down features absent before 2021/2022 | population mismatch across eras | all-NaN in early folds, dropped by the imputer | low | slice table shows no era break for the ridge |
| retractable venue set defined from the whole file | future knowledge | building property, stable | low | moot while the flags are dead |

All challengers' preprocessing is fit inside the training block, and no test-season
row enters tuning.

## What would be worth trying next

1. **Re-run this bake-off after 2026 completes.** Both the nested alpha and the
   blend weight drift toward trusting the features as training data grows, and
   the fourth-down features are only live in 3 to 4 folds. The ridge's edge over
   the line is the thing most likely to grow. It is a rerun, not a rebuild.
2. **Fix or drop the dead roof flags in `features.py`** and correct "16" to the
   real count in the docs. No lift, just honesty.
3. **An early-season prior.** Weeks 1 to 4 are where the ridge collapses to the
   line. Carrying prior-season team volume into the first four weeks is a feature
   idea with a clear mechanism; pre-register it as one test.
4. **Opening lines**, only if a key-free historical source turns up. They would
   both remove the closing-line optimism above and allow a line-movement feature.
5. **Nothing else on the model side.** Fourteen shapes of model on these features
   all land within noise of the ridge or below it. Further gains, if any, are in
   inputs, not in the learner.

## Reproduce

```
set OMP_NUM_THREADS=1
python src/kicker_analysis/model_bakeoff.py --self-check   # offline, a few seconds
python src/kicker_analysis/model_bakeoff.py --quick        # 3 folds, 1 inner fold, about 2 min
python src/kicker_analysis/model_bakeoff.py                # the numbers above, about 5 min
```

Seeds are fixed (harness seed 0, LightGBM `deterministic=True`, one thread).
Outputs: `reports/model_bakeoff_summary.csv`, `_folds.csv`, `_params.csv` (every
chosen hyperparameter per fold), `_calibration.csv`, `_slices.csv`,
`_ridge_coefs.csv`, and the ignored `_predictions.parquet`.
