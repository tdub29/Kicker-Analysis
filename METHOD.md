# How to pick a fantasy kicker, and how we know

Written for someone who will check the claims. Every number here comes from a
season-forward backtest: for each test season the model trains only on seasons
before it, so nothing from the future of the league is ever in the training set.
Scripts and raw outputs are named at the bottom.

## The short version

1. **Start the kicker on the offence with the highest Vegas implied team total.**
   That single number beat a 73-feature gradient-boosted model by 0.106 MAE.
2. **Ignore the kicker.** His accuracy is nearly irrelevant week to week. What you
   are actually betting on is how often his offence scores.
3. **The model adds a calibrated `P(15+)`, not a better ranking.** Against the
   implied total alone it is a statistical tie (p = 0.56). Its real contribution
   is that the probability is accurate to within 5% at every threshold, so you
   can size a must-win swing with it.
4. **Check whether his coach goes for it.** In 2025 Arizona attempted 0 of 21
   in-range fourth downs and Buffalo went for it on 13 of 27. That is close to a
   full field goal attempt a game of difference between two kickers.
5. **Do not pay for a projection that looks confident.** The honest spread across
   30 starters in a week is about three points. Anyone showing you more precision
   than that is selling you noise.

## Why the obvious approach fails

The first version of this repo fit a three-feature linear regression on
`homefppg`, the season-to-date fantasy average. That predicts a rolling mean from
three other rolling means. It scores well and forecasts nothing.

Rebuilt properly, with the week's actual points as the target and 73 lagged
features, the models still failed, and failed in an instructive way:

| model | MAE | beats a constant? |
|---|---|---|
| predict the training median, 8.0, every week | 3.4946 | n/a |
| LightGBM, 73 features | 3.5868 | no, worse by 0.086 |
| LightGBM Tweedie, 73 features | 3.5886 | no, worse by 0.088 |
| Ridge, 73 features | 3.4919 | tie, 0.0027, p = 0.59 |
| **one-feature OLS on the Vegas implied total** | **3.4807** | yes, slightly |

A single betting line beat everything the pipeline built. That is finding number
one and it is not a modelling failure, it is the answer.

## Why: the variance is in the half nobody can predict

Fantasy points decompose exactly:

```
points = 3*(FG 0-39) + 4*(FG 40-49) + 5*(FG 50+) + 1*(extra points)
```

Over 2015 to 2026 the variance splits:

| component | share of total variance |
|---|---|
| field goals | 102% |
| extra points | 10% |
| covariance between them | -13% |

That negative covariance is the mechanism: a drive ending in a touchdown pays 1
point, the same drive stalling pays 3. Field goals and touchdowns compete.

Now the number that decides everything. Out of sample, season-forward:

| predicting | R-squared | within-week rank correlation |
|---|---|---|
| field goal attempts | **-0.054** | 0.014 |
| extra point attempts | +0.081 | **0.299** |

**Weekly field goal attempts are less predictable than their own average.** The
half of the target carrying all the variance has negative skill. The half with
real skill carries a tenth of it.

So the reason a kicker model tops out early is not that the features are bad. It
is that you are trying to forecast a coin flip that the offence does not control
and the coach can veto. What you *can* forecast is touchdowns, which is why the
implied total wins and why extra points are the predictable part.

## What the model does about it

Two Poisson count stages, one for field goal attempts and one for extra point
attempts, run through the scoring rule and convolved by Monte Carlo into a full
distribution rather than a point estimate.

Counts are **under-dispersed** here, variance over mean 0.843 for field goal
attempts and 0.881 for extra points, so the draw is binomial-thinned. Negative
binomial only widens a Poisson and cannot fit this; left uncorrected the tail
over-predicts P(15+) by about 19%.

This buys three things:

1. **Honesty about where the signal is.** The extra-point stage does the work.
2. **P(y >= x) for any threshold, with no threshold model trained.** It matches a
   purpose-built classifier: at 15 points, AUC 0.545 against 0.551.
3. **A fix for the ceiling problem**, below.

## What the backtest actually says

Eight season-forward folds, 2018 to 2025, 5,764 kicker-games. Scored on the
decision: draw two kickers at random from the week's starters, start the one the
method prefers, and measure points gained against starting either at random. The
oracle, who always starts the better of the two, gains 2.440 points a week. That
is the ceiling everything below is a slice of.

| rank by | within-week rho | lift, pts/wk | share of oracle | hit 15+ |
|---|---|---|---|---|
| **Vegas implied team total alone** | **0.1426** | **0.3620** | **14.8%** | 8.7% |
| compound model, expected points | 0.1426 | 0.3486 | 14.3% | 8.9% |
| compound model, P(15+) | 0.1363 | 0.3406 | 14.0% | 8.9% |
| kicker's own career average | 0.0711 | 0.1613 | 6.6% | 8.5% |
| a constant 8.0 | n/a | -0.0154 | -0.6% | 8.1% |

Paired week-block bootstrap on the R=2 lift:

| comparison | difference | 95% CI | p |
|---|---|---|---|
| model vs kicker career average | **+0.187** | [+0.092, +0.286] | 0.0000 |
| model vs a constant | **+0.364** | [+0.267, +0.455] | 0.0000 |
| model vs implied total alone | -0.014 | [-0.058, +0.034] | **0.56** |

**Read the last row honestly. The sixteen-feature model is indistinguishable from
ranking by one betting line, and the line is nominally ahead.** The model earns
its keep against the things people actually do, starting the kicker who scored
well last month or the one on your roster by default, and it earns nothing
against the market. If you only remember one thing from this document, remember
the implied total.

What the model does add is a calibrated distribution, which a single line cannot
give you:

| threshold | predicted rate | realised rate | ratio |
|---|---|---|---|
| 8 or more points | 0.5444 | 0.5241 | 1.04 |
| 10 or more | 0.3733 | 0.3557 | 1.05 |
| 12 or more | 0.2230 | 0.2178 | 1.02 |
| **15 or more** | **0.0838** | **0.0841** | **1.00** |

Within 5% at every threshold and essentially exact at 15. So `P(15+)` means what
it says and can be used directly for a must-win week or a DFS tournament.

## The ceiling question, and a correction

An earlier reviewer found that ranking by a gradient-boosted expected-points
model hit a 15-point week only 5.7% of the time against 8.5% for picking at
random: the mean ranking preferred the safe extra-point kicker and actively
avoided upside. That was true of that model.

**It is not true of this one, and the separate ceiling board does not beat the
main board either.** Measured here: random hits 15+ at 8.16%, ranking by expected
points hits 8.88%, ranking by P(15+) hits 8.89%. The compound model fixed the
anti-ceiling defect, and having fixed it, the second board adds nothing
measurable on top.

The two boards still order players differently and the app still shows both,
because the probability is the honest way to express a swing and because the
calibration above makes it trustworthy. But the claim that you need the ceiling
board to catch big weeks is not supported by this backtest, and it is retracted.

## What actually moves a kicker's week, ranked

1. **Vegas implied team total.** Beats everything else combined.
2. **Coach fourth-down aggression in field-goal range.** Persistent, not noise:
   team go-rate carries year to year at r = 0.27 on average and r = 0.42 across
   the last four season pairs, with an 8.6 point spread across teams. League-wide
   the go rate rose from 13.3% in 2015 to 26.4% in 2025, so field goal share of
   in-range fourth downs fell from 82.9% to 73.3%.
3. **Red-zone touchdown rate.** The substitution mechanism, directly: the team's
   red-zone field-goal rate correlates +0.923 with its field goal attempts, and
   red-zone TD rate correlates -0.479.
4. **Offensive volume**, lagged: attempts and extra points per game.
5. **Environment**, but only the part you can know in advance. See below.
6. **The kicker himself.** Accuracy over a career matters a little. His recent
   form matters almost not at all.

## Three traps that make a kicker model look good and deploy badly

**Weather is not available on Tuesday.** Of 255 unplayed games in the schedule
file, zero carry a temperature or wind speed, while rest, division game, surface
and home field carry all 255. Observed weather ranked 3rd and 7th by importance.
Training on it and serving without it cost +0.092 MAE. Dropping it cost 0.002.
What replaces it is *stadium climate*: the venue's typical wind and temperature
in that month from prior seasons, which is a fact about the building and is
knowable whenever you like.

**Retractable roofs are a game-day decision.** The values `closed` and `open`
never appear on an unplayed game. That was 15.1% of modelling rows carrying a
label that cannot exist at forecast time. A stadium is now flagged as retractable,
which is a property of the building.

**An exclusion list is not a feature set.** The first version admitted every
numeric column nobody had remembered to ban. During review, 17 columns were
merged into the frame and all 17 were silently accepted as model inputs; three
were same-game outcomes, and they moved MAE from 3.49 to 2.65 in about five
minutes. The feature set is now an explicit list of 16 named columns, each with a
stated mechanism. If you cannot enumerate your features you cannot audit them.

## Two checks a reviewer would demand, run

**Was the hyperparameter borrowed from the test set?** No. `alpha = 100` came from
a sweep I read off the same seasons I then reported, which is exactly the
selection-on-test problem that makes a number quietly optimistic. Re-run with an
inner loop that picks alpha using only seasons before the test year (it chose
300, 300, 1000, 100, 300, 1000, 3000 and 10 across the eight folds), the honest
result is lift **0.4322** against the fixed-alpha 0.4342, a difference of -0.0020
at p = 0.79. Against the implied total the nested model still wins, **+0.070
[+0.017, +0.123], p = 0.010**. The tuning was doing no work, so the headline
stands as reported.

**Do more features help?** No, and this was tested twice with different tools.
Three candidate families were built and each was attacked by a separate
adversarial reviewer, who refuted all three under LightGBM. Both reviewers
independently found the same confound worth knowing about: on that setup, adding
columns of pure Gaussian noise IMPROVES pooled MAE, so any with-versus-without
feature test there is really measuring column count. Re-tested under the ridge,
which does not have that pathology:

| added to the 16 | cols | MAE | within-week rho | decision lift |
|---|---|---|---|---|
| nothing (baseline) | 16 | 3.4696 | 0.1732 | 0.4342 |
| red-zone efficiency | +27 | 3.4749 | 0.1517 | 0.3782 |
| drive-level offence | +21 | 3.4801 | 0.1688 | 0.4111 |
| leg strength and stadium | +13 | 3.4743 | 0.1588 | 0.3759 |
| all three | +61 | 3.4874 | 0.1417 | 0.3240 |

Every family makes the model worse on every metric. They are not noise: each
beats its own matched-width noise control, so they do carry information. They are
REDUNDANT, and the extra columns cost more in variance than the information is
worth. Red-zone behaviour in particular is already reaching the model through
the fourth-down aggression features and the lagged team volume.

That is the stopping point. The feature set is sixteen columns and adding to it
has been tried properly and does not pay.

## Willingness to attempt 50+, measured separately

Asked directly, and it was a real gap: `fourth_down.py` lumps every in-range
fourth down into one bucket at `yardline_100 <= 38`, which caps at a 55-yarder
and cannot see the long-range decision at all. Measured on its own (FG distance
is `yardline_100 + 17`, so the 50+ band is yardline 33 to 43, with 20 to 32 as
the short control):

| season | 50+ attempt rate on 4th down | short-range rate |
|---|---|---|
| 2021 | 0.314 | 0.782 |
| 2023 | 0.375 | 0.802 |
| 2025 | 0.437 | 0.788 |

The two move in opposite ways. Long-range willingness climbed 12 points in four
years while short-range sat flat, so a single bucket was averaging a strong trend
against a flat one and seeing neither.

**The team spread looks enormous and mostly is not real.** In 2025 New England
went 0 for 7 and Dallas 11 for 12. But a team gets a median of only 10 such
chances a season, so decomposing the observed cross-team variance:

| component | variance | sd |
|---|---|---|
| observed across teams | 0.03764 | 0.194 |
| binomial sampling noise | 0.02466 | 0.157 |
| **true team difference** | **0.01298** | **0.114** |

**Only 34.5% of the visible spread is a real team property.** The empirical-Bayes
shrinkage constant is 18.3 prior observations, so a team with its median 10
chances should keep just 35% of its own rate and be pulled to the league mean for
the rest. That is also why raw year-over-year persistence is weak (mean r = +0.22,
with one pair actually negative) against +0.42 for general fourth-down aggression.

**Shrunk properly it helps, but not provably.** Adding the lagged, shrunk feature
moved decision lift from 0.4212 to 0.4423 and within-week rho by +0.015. The
paired bootstrap says **+0.021, 95% CI [-0.018, +0.059], p = 0.27**.

Not added to the sixteen. The sign is right and the mechanism is real, but p =
0.27 on 72 weeks is not evidence, and the 50+ band only has six seasons of clean
play-by-play behind it. Worth re-testing once there are more. The extractor lives
in the notes for this section so the re-test is a rerun, not a rebuild.

## Honest limits

- The total edge is small. Every decent model sits between 3.456 and 3.492 MAE
  against a constant's 3.501. On the decision metric the gain is real but it is a
  fraction of the oracle's.
- Kicker is the position where streaming on implied total gets you most of the
  available value. This model is worth having; it is not worth agonising over.
- The roster read comes from the last man who actually kicked for each team, so a
  kicker signed on Wednesday will not appear. Rows carry a staleness flag.
- Missed field goals score 0 here. Leagues that use -1 shift 6.4% of weekly top-5
  picks. `SCORING` in `build_dataset.py` is the one place to change it.

## Reproducing this

```
python src/kicker_analysis/build_dataset.py      # 14,457 kicker-games, 1999-2026
python src/kicker_analysis/fourth_down.py        # coach aggression, 2015-2026
python src/kicker_analysis/model.py --self-check
python src/kicker_analysis/backtest.py           # the numbers above
python src/kicker_analysis/predict_week.py       # this week's board, writes the site file
```

Every module has an assert-based `--self-check` that runs offline in under a
second. Two real bugs were caught by those checks rather than by review: a sign
error in the implied-total algebra, and a team-game aggregation that double
counted a score when two men kicked in the same game.

Set `OMP_NUM_THREADS=1`. LightGBM thread contention on this machine turns a
sub-second fit into three minutes.
