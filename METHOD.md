# How to pick a fantasy kicker, and how I know

Written for someone who will check the claims. Every result here comes from a
tracked CSV in `reports/` or from the printed output of the script that wrote
it, all regenerated on one data build. The few exceptions are historical figures
from code that has since been retired, and each is marked where it appears. The
backtest is season-forward: for each test season the model trains only on
seasons before it, so nothing from the future of the league is ever in the
training set. It covers 2018 to 2025. 2026 is not a backtest fold; it is scored
live, from boards committed before the games. Scripts and the file behind each
section are listed at the bottom.

## The short version

1. **Start the kicker on the offence with the highest Vegas implied team total.**
   On its own it gains 0.353 points a week on the start/sit decision, 83% of
   the 0.427 the full model gains.
2. **Bet on the offence, not the leg.** A kicker's career accuracy is a small
   input. His own recent fantasy output carries real weight in the model, but
   I read that mostly as a record of how much his offence has been scoring.
3. **The model adds a little ranking and a rough upside flag, not a reliable
   probability.** The ridge that orders the board beats the implied total alone
   by 0.074 points a week, about 3% of what an oracle gets (0.074 / 2.458).
   Week to week inside the backtest that is clear (p = 0.0036). With seasons as
   the unit it is not: ahead in 6 of 8, p = 0.095. The count model's `P(15+)`
   is right on average (8.39% predicted, 8.41% happened) but barely tells one
   kicker from another (AUC 0.55). Do not size a bet on it.
4. **Coach aggression is a league trend more than a weekly edge.** Teams go for
   it on in-range fourth downs about twice as often as in 2015 (13.3% then,
   26.4% in 2025), so there are fewer field goals for everyone. A team's own go
   rate is one of the smallest inputs to the board.
5. **Do not pay for a projection that looks confident.** On the 2026 boards the
   ridge's expected points span 2.6 to 3.2 points from the top kicker to the
   bottom one in a week. A single kicker's week has a standard deviation of 4.26
   points (1999 to 2026). Even an oracle that knew the scores in advance gains only 2.458
   points a week on the start/sit decision.

## Why the obvious approach fails

The first version of this repo (2023, in R, later ported to Python) regressed
`homefppg` on three six-week team averages. `homefppg` is the home kickers'
points in that one game. An earlier version of this document called it a
season-to-date average, which was wrong. The regression was fit on every row of
the table it was judged on, and nothing was ever scored out of sample.
`legacy/README.md` has the details.

Rebuilt with the week's actual points as the target, 73 lagged features and
season-forward folds, the models still failed, and failed in an instructive way:

| model | MAE |
|---|---|
| Ridge, 73 features | 3.4919 |
| predict the league mean every week | 3.5008 |
| kicker's own career average | 3.5676 |
| LightGBM, 73 features | 3.5868 |
| LightGBM Tweedie, 73 features | 3.5886 |

This table is the historical record of the retired harness,
`legacy/evaluate_v1.py`, and its output is in `legacy/reports/model_summary.csv`.
It ran 7 folds, 2018 to 2025 without 2021, because the data then lacked that
season: 3,676 test rows against today's 4,211. At the time I also quoted a
constant 8.0 and a one-feature regression on the implied total; their numbers
were never saved to a file, so they are not in the table.

Both gradient-boosted models were worse than predicting the league mean every
week, and the 73-feature ridge beat it by 0.009. MAE ranked the league mean
third of seven, and on the start/sit decision a constant gains nothing at all.
That is why everything below is scored on the decision, not the error.

The current harness tells the same story on the decision metric. The count
model, two gradient-boosted stages on the 16 current features, ties the implied
total alone: +0.009 points a week [-0.034, +0.054], p = 0.66. The betting line
is the benchmark, and most of what a model can know is already in it.

## Why: the variance is in the half I could not predict

Fantasy points decompose exactly:

```
points = 3*(FG 0-39) + 4*(FG 40-49) + 5*(FG 50+) + 1*(extra points)
```

Over 5,730 regular-season kicker-games from 2015 to 2025, the variance of
weekly points (19.47) splits:

| component | variance | share of total |
|---|---|---|
| field goal points | 19.82 | 101.8% |
| extra point points | 2.05 | 10.5% |
| twice the covariance between them | -2.41 | -12.4% |

That negative covariance is the mechanism: a drive that ends in a touchdown pays
1 point, the same drive stalling pays 3. Field goals and touchdowns compete.

Now the numbers that decide everything. Out of sample, season-forward, 4,211
test rows, R-squared against the training-season mean with a week-block 95%
interval:

| predicting | count stage R-squared | within-week rho | ridge R-squared | within-week rho |
|---|---|---|---|---|
| field goal attempts | +0.003 [-0.003, +0.009] | 0.068 | +0.007 [+0.001, +0.013] | 0.084 |
| extra point attempts | +0.126 [+0.112, +0.139] | 0.348 | +0.134 [+0.119, +0.148] | 0.355 |
| total points | +0.022 [+0.012, +0.030] | 0.147 | +0.029 [+0.018, +0.040] | 0.171 |

**Weekly field goal attempts showed close to no out-of-sample skill.** The count
stage cannot be told apart from the training mean, and the ridge is barely above
it. The half of the target that carries the variance showed no measurable skill;
the half with real skill carries about a tenth of it.

This is a retraction. An earlier version said field goal attempts had negative
skill (R-squared -0.054, within-week rho 0.014) and were "less predictable than
their own average". No committed code produced those numbers, and the shipped
models do not reproduce them.

So the reason a kicker model tops out early is not mainly the feature set.
Whether a drive stalls at the 25 or scores, and whether the coach kicks or goes
for it, did not prove forecastable from anything I could build before kickoff.
How often an offence scores touchdowns did. That is why the implied total is
hard to beat and why extra points are the predictable part.

## What the model does about it

Two models, each used for the job it does better. A ridge regression on the 16
named features (alpha 100) orders the board. A count model supplies the
distribution:

- **Means.** Two LightGBM regressors with a Poisson objective (100 trees, 7
  leaves), one for field goal attempts and one for extra point attempts, on the
  same 16 features.
- **Draws.** Each count is drawn as a binomial with mean lambda over ceil(3 x
  lambda) trials. Its variance over mean is 1 - lambda / ceil(3 x lambda): at
  least two thirds, about 0.69 for the lambdas this model predicts. That ratio
  is set by the constant 3, not fitted. The counts are under-dispersed, so a
  plain Poisson draw would be too wide, but the fixed ratio is a choice, not an
  estimate, and it makes the draw somewhat narrower than the data.
- **Makes and distance.** Makes are drawn at the training seasons' league make
  rate, then split into the 3, 4 and 5-point tiers with random weights around
  the league distance mix. That split leans a little toward the longer tiers,
  so each make is worth slightly more than the mix says. Over the test rows the
  count model's expected points run 0.11 high on average.
- **No dependence.** Field goal and extra point counts are drawn independently,
  although they are negatively related (the covariance above) and stay so after
  conditioning on the features (the check is described in `model.py`).
- **Monte Carlo.** 4,000 draws per kicker-game, pushed through the scoring rule.
  It is a simulation, not an analytic convolution.

What this buys: `P(y >= x)` at any threshold with no threshold model trained,
and the floor, median and ceiling on the board. On the board, `exp_points` is
the ridge's prediction; the count model supplies the rest. What it does not buy:
a better ranking (the ridge beats the count model's expected points by +0.065,
p = 0.0070), or sharp probabilities (see the calibration section).

## What the backtest actually says

Eight season-forward folds, 2018 to 2025, 4,211 kicker-games in 141 weeks.
Scored on the decision: draw two kickers at random from the week's starters,
start the one the method prefers, and measure points gained against starting
either at random. Every lift is the exact expectation over every pair in the
week. An earlier version simulated 300 draws a week, which put noise into every
comparison; the headline history below shows how the numbers moved. The oracle,
who always starts the better of the two, gains 2.458 points a week. That is the
ceiling everything below is a slice of.

| rank by | within-week rho | lift, pts/wk | share of oracle | hit 15+ |
|---|---|---|---|---|
| **ridge on the 16 features, which orders the board** | **0.171** | **0.427** | **17.4%** | 9.6% |
| count model, expected points | 0.147 | 0.362 | 14.7% | 9.2% |
| Vegas implied team total alone | 0.143 | 0.353 | 14.4% | 9.0% |
| count model, P(15+) | 0.139 | 0.346 | 14.1% | 9.2% |
| kicker's own career average | 0.071 | 0.180 | 7.3% | 8.7% |
| a constant 8.0 | n/a | 0.000 | 0.0% | 8.4% |

Each comparison is tested two ways. The week-block bootstrap resamples the 141
weeks. The season-level tests treat each of the 8 seasons as one observation:
a t-test on the 8 per-season differences (7 degrees of freedom), a Wilcoxon
signed-rank test and a count of seasons won.

| comparison | difference, pts/wk | week blocks: 95% CI, p | seasons: t 95% CI, p | Wilcoxon p | seasons won |
|---|---|---|---|---|---|
| ridge vs implied total alone | **+0.074** | [+0.024, +0.124], 0.0036 | **[-0.016, +0.162], 0.095** | 0.11 | 6 of 8 |
| ridge vs kicker career average | +0.247 | [+0.156, +0.335], < 0.0001 | [+0.132, +0.362], 0.0014 | 0.0078 | 8 of 8 |
| ridge vs count model expected points | +0.065 | [+0.017, +0.112], 0.0070 | [+0.019, +0.109], 0.012 | 0.016 | 7 of 8 |
| count model vs implied total alone | +0.009 | [-0.034, +0.054], 0.66 | [-0.051, +0.068], 0.75 | 0.74 | 5 of 8 |

The season-level means differ slightly from the pooled differences (+0.073
against +0.074 for the first row) because seasons have 17 or 18 weeks.

**Read the first row honestly.** Inside these 141 weeks the ridge is ahead of
the betting line by a margin the week-block bootstrap separates from zero. But
the weeks of a season share that season's league, and the question a reader
cares about is whether it will be ahead next season. With seasons as the unit
the edge is +0.073 a week, t = 1.93, p = 0.095, and the interval includes zero.
**The edge over the line is suggestive, not established.** The ridge loses 2019
(-0.126) and 2021 (-0.019), and nearly all of its edge comes from 2022 onward,
when the line alone got worse: pooled by week it is +0.005 over the line in
2018 to 2021 and +0.140 in 2022 to 2025. An earlier version of this document
said the edge "survives resampling whole seasons", quoting a percentile
bootstrap over seasons (today [+0.002, +0.137], p = 0.045). With 8 clusters
that method under-covers, and it is not the test to quote.

The count model on the same features ties the line, and the ridge beats it in 7
of 8 seasons. The model clearly earns its keep against what people actually do
instead, such as starting the kicker with the best career: +0.247 a week, ahead
in all 8 seasons. Against the market it may earn a little. If you only remember
one thing from this document, remember the implied total.

**How the multiple-testing family was chosen.** The model bake-off
(`model_bakeoff.py`) treats the two comparisons this repo claimed before the
bake-off existed, ridge versus the line and ridge versus the career average, as
their own Holm family of two. There the line comparison keeps its p of 0.0036.
That split was decided after the first bake-off run. In commit 3c51cf7 all 18
comparisons were one Holm family, and ridge versus the line came out at raw
p = 0.0054, Holm p = 0.081: not significant at 0.05. Later the same day,
commit f04d00e split the family into the 14 challengers and the 2 baseline
claims. The argument for the split, that two claims made before the bake-off
should not pay for 14 questions asked after it, is defensible. It was made
after seeing the result, so the two-family Holm p is a post hoc choice. On
today's numbers one family of 18 puts the line comparison third and gives it
Holm p = 16 x 0.0036 = 0.058.

## Calibration: right on average, weak per kicker

The count model's probabilities, pooled over all 4,211 test rows:

| threshold | predicted | realised | predicted minus realised, 95% CI | Brier skill over the base rate, 95% CI | calibration slope | AUC |
|---|---|---|---|---|---|---|
| 8 or more points | 0.5446 | 0.5241 | +0.021 [+0.006, +0.035] | +0.017 [+0.008, +0.025] | 1.19 | 0.58 |
| 10 or more | 0.3735 | 0.3557 | +0.018 [+0.003, +0.033] | +0.008 [-0.001, +0.017] | 0.81 | 0.56 |
| 12 or more | 0.2232 | 0.2178 | +0.005 [-0.007, +0.018] | +0.006 [-0.001, +0.013] | 0.72 | 0.56 |
| **15 or more** | **0.0839** | **0.0841** | 0.000 [-0.009, +0.008] | **+0.001 [-0.004, +0.006]** | **0.55** | **0.55** |

Brier skill compares the model's Brier score with a constant at the training
seasons' base rate. The calibration slope is from a logistic regression of the
outcome on logit(p); below 1 means the stated chances are too spread out. The
intervals are week-block bootstraps.

The pooled rate at 15 points matches almost exactly. That is calibration in the
large, and a constant at the base rate gets it too. Kicker by kicker, `P(15+)`
barely beats that constant: Brier skill +0.001 with an interval spanning zero,
AUC 0.55, slope 0.55. Split into ten equal bins, predicted rates run from 0.042
to 0.130 while realised rates run only from 0.045 to 0.100, and they do not rise
steadily: the top tenth was predicted at 0.130 and came in at 0.100. At 8 and 10
points the model over-predicts by about two percentage points, and those
intervals exclude zero.

The 15+ match is also partly errors cancelling in the simulator: counts drawn
too narrow, field goals and extra points drawn as if independent, and makes
worth slightly too much. Matching the pooled rate is not hard in any case: a
Tweedie GLM on total points also lands near it (0.0867 against 0.0841).

This is a retraction. An earlier version called `P(15+)` calibrated "within 6%
at every threshold" and said it could be used directly for a must-win week. It
is mean-calibrated only. Read it as a rough flag for upside.

## The ceiling question, and a correction

An earlier version of this repo said that ranking by a gradient-boosted
expected-points model hit a 15-point week less often than picking at random,
because the mean ranking preferred the safe extra-point kicker. That was the
case for a retired model, and it was the reason for a separate ceiling board.

**It is not true of these models, and the separate ceiling board does not beat
the main board either.** Over the 4,211 test rows, picking at random hits 15+ at
8.40%. Ranking by the ridge hits 9.57%, by the count model's expected points
9.20%, and by its `P(15+)` 9.17%. These are point estimates; no interval on them
is in the outputs. Ordering by the probability adds nothing measurable on top of
ordering by the mean.

The board still carries both orderings (`rank_points`, the recommendation, and
`rank_ceiling`), because the probability is an honest way to express a swing.
But the claim that you need the ceiling board to catch big weeks is not
supported by this backtest, and it is retracted.

## What actually moves a kicker's week, ranked

From the shipped ridge's standardised coefficients (points per standard
deviation of each input), averaged over the 8 folds, with the number of folds
that agree in sign (`reports/model_bakeoff_ridge_coefs.csv`). The inputs are
correlated, so a ridge spreads shared signal across them. Read size and sign
consistency, not the order of the small ones.

1. **The market, led by the team's implied total** (+0.465 per sd, 8 of 8
   folds). The opponent's implied total (-0.297, 8 of 8) and the total line
   (+0.151, 8 of 8) sit beside it. The two implied totals add up to the total
   line, so read the three as one block: the more points Vegas expects from the
   kicker's own offence, the better.
2. **The kicker's own recent output** (fantasy points over his last 8 games:
   +0.362, 8 of 8). This is large, and it contradicts the "ignore the kicker"
   an earlier version of this document led with. I read it mostly as team
   volume, not leg: once it is in, the team's own lagged scoring (-0.131, 7 of 8)
   and extra point attempts (-0.100, 7 of 8) enter negatively, which is what
   correlated columns sharing one signal look like. With these features I
   cannot separate the kicker from his offence.
3. **Wind exposure** (-0.186, 8 of 8): the venue's typical wind in that month
   from earlier seasons, zero indoors. It is fourth by size. An early commit
   message called stadium climate the most valuable feature after the betting
   line; the coefficients do not say that.
4. **The rest of lagged volume** is small and mixed: the kicker's own recent
   field goal attempts (-0.074, 7 of 8) and the team's (-0.027, 5 of 8).
5. **The building**: a retractable roof +0.068 (8 of 8), certainly indoors
   -0.011 (4 of 8).
6. **Career accuracy**: +0.061 (8 of 8). Real and small.
7. **Coach fourth-down aggression** is small in the shipped model. The prior
   season's go rate is +0.050 (8 of 8), and the team's recent field-goal share
   of in-range fourth downs is -0.004 (4 of 8). The league trend is real: the go
   rate on in-range fourth downs rose from 13.3% in 2015 to 26.4% in 2025, and
   the field goal share fell from 82.9% to 73.3%. A team's go rate also carries
   over from one season to the next, r = +0.27 averaged over 10 season pairs.
   But as a weekly input it barely moves the board, and its sign is positive,
   not the negative that "more go-for-its, fewer kicks" predicts. An earlier
   version ranked it second; the coefficients do not support that.
8. **Red-zone touchdown rate is a mechanism, not a model input.** It is how
   field goals and touchdowns trade off (the -12.4% covariance above). It is not
   one of the 16 features, and adding the red-zone family to the ridge lowers
   its lift (below). An earlier version ranked it third and quoted two
   correlations that no committed code produces; they are withdrawn.

Game closeness (+0.018, 6 of 8) and home field (-0.004, 6 of 8) add nothing
visible.

## Five traps that make a kicker model look good and deploy badly

Unless a file is named, the figures in this section were measured when each fix
was made, on the data as it stood then. No script regenerates them, and
`features.py` says the same.

**Weather is not available on Tuesday.** Of 255 unplayed games in the schedule
file, zero carry a temperature or wind speed, while rest, division game, surface
and home field carry all 255. Observed weather ranked 3rd and 7th by importance.
Training on it and serving without it cost +0.092 MAE. Dropping it entirely
moved MAE by only 0.002. What replaces it is *stadium climate*: the venue's
typical wind and temperature in that month from prior seasons, which is a fact
about the building and is knowable whenever you like.

**The betting line is a closing line.** The implied total, the total line and
closeness come from nflverse `spread_line` and `total_line`, which are the numbers
at kickoff. So the backtest measures a board built from the closing number, and
the board is valid for a lineup set close to kickoff. Read on Tuesday it uses an
earlier line and should expect a little less than the tables above.

**Retractable roofs are a game-day decision.** The values `closed` and `open`
never appear on an unplayed game. That was 15.1% of modelling rows carrying a
label that cannot exist at forecast time. A stadium is now flagged as
retractable, which is a property of the building. The first version of that fix
was itself broken: it keyed the building on a `stadium_id` the kicker table does
not carry, fell back to the kicker's team, and since every team has played away
under a retractable roof, every row was retractable and none was ever indoors.
Both roof flags were constant, so the model had 14 live features, not 16. The
type now comes from the game's own venue (the per-game roof field, open and
closed collapsed to retractable, and for an unplayed game a stadium table built
from played games), and `features.check_roof` fails the backtest if either flag
goes constant again. Fixing it made no measurable difference to the ranking: the
ridge with the flags dead scores 0.431 against 0.427 for the fixed one, +0.004
[-0.005, +0.014], p = 0.39 (`reports/model_bakeoff_summary.csv`). That is why it
went unnoticed.

**An exclusion list is not a feature set.** The first version admitted every
numeric column nobody had remembered to ban. When 17 extra columns were merged
into the frame, all 17 were silently accepted as model inputs; three were
same-game outcomes, and they moved MAE from 3.49 to 2.65. The feature set is now
an explicit list of 16 named columns, each with a stated mechanism. If you cannot
enumerate your features you cannot check them, and if you do not check that they
vary you cannot count them.

**A fresh clone built a third of the coach data, and nothing failed.**
`fourth_down.py` read whatever play-by-play files were in the local cache. The
published numbers were built from a cache that held every season. A fresh clone
held only the seasons `build_dataset.py` had downloaded for its own use, so the
coach features came from 4 of 12 seasons and were median-imputed everywhere
else. The backtest still ran, because its only check was that each feature
varies, and a partly imputed column still varies. Now `fourth_down.py` downloads
every season from 2015 on and stops if one is missing or has rows for fewer
than 70% of its played team-games. `backtest.py` asserts that each of the 16
features is at least 85% non-missing in every season the folds use (the
prior-season columns are exempt in 2015, which has no prior season). On this
build the lowest is 91.1%.

## Two checks a skeptic would ask for, run

**Was the hyperparameter borrowed from the test set?** Yes, and it does not
matter. `alpha = 100` came from a sweep I read off the same seasons I then
reported, which is exactly the selection-on-test problem that makes a number
quietly optimistic. Re-run with an inner loop that picks alpha using only
seasons before the test year (it chose 10,000, 1,000, 300, 300, 300, 300, 300
and 100 across the eight folds), the nested ridge's lift is 0.422 against the
fixed-alpha 0.427, a difference of -0.005 [-0.019, +0.009], p = 0.48. It is
still 0.069 a week above the line alone and ahead of it in 6 of 8 seasons.

The same bake-off ran thirteen other challengers, from count GLMs to
gradient-boosted and pairwise rankers, each tuned only on the past. None is
distinguishable from the ridge after Holm correction; every challenger's Holm p
is 1.00. The closest is the ridge plus an empirical-Bayes kicker effect, +0.012
[-0.010, +0.034], p = 0.28, ahead in 7 of 8 folds. The furthest behind is a
LightGBM lambdarank, -0.041 [-0.086, +0.003]. The design is in
`notes/model-selection-2026-09-29.md`.

**Do more features help?** No. Three candidate families were built and scored
first under LightGBM. There, the same 27 red-zone columns shuffled at random
lowered pooled MAE on two of three seeds (mean -0.0177;
`experiments/feature_families/feat_redzone_control.log`), so a with-versus-without
test on that setup was partly measuring column count. Re-tested under the
shipped ridge, on the exact lift, with the same folds as the backtest
(`experiments/feature_families/ridge_retest.py`):

| added to the 16 | cols | MAE | within-week rho | lift, pts/wk | lift vs baseline, 95% CI, p |
|---|---|---|---|---|---|
| nothing (baseline) | 0 | 3.4718 | 0.171 | 0.427 | |
| red-zone efficiency | +27 | 3.4765 | 0.149 | 0.370 | -0.057 [-0.085, -0.029], < 0.0001 |
| drive-level offence | +21 | 3.4823 | 0.167 | 0.418 | -0.009 [-0.042, +0.024], 0.57 |
| leg strength and stadium | +13 | 3.4730 | 0.159 | 0.396 | -0.032 [-0.073, +0.009], 0.14 |
| all three | +61 | 3.4853 | 0.141 | 0.347 | -0.080 [-0.132, -0.027], 0.0022 |

Every family makes the ridge worse on every metric. Red-zone lowers the lift in
all 8 test seasons, drive-level in 6 of 8.

Two controls, 50 draws each: the 16 features plus the same number of
standard-normal noise columns, and the 16 plus the family's own columns shuffled
within each season, which keeps their values and correlations and breaks only
the link to the kicker-week. Under this ridge noise is not free: each random
column costs about 0.0015 to 0.0018 points a week of lift, and MAE rises by
0.0055 with 13 noise columns and 0.0310 with 61.

- **On lift, no family beats either control.** Drive-level comes closest: better
  than 48 of 50 noise draws (Monte Carlo p 0.059) and 46 of 50 shuffles (p
  0.098), and it still lowers the lift. Red-zone orders the week worse than
  every one of its 50 shuffled copies. Leg strength and stadium beats 8 of 50
  noise draws and 13 of 50 shuffles; all three, 36 and 14.
- **On MAE**, red-zone (49 of 50 noise draws, p 0.039) and all three (50 of 50,
  p 0.020) beat the noise control, but not their own shuffled rows (40 of 50,
  p 0.22; 33 of 50, p 0.35). Most of that edge comes from the columns' structure,
  not from their link to a kicker's points.

Of the 24 control comparisons, two reach p < 0.05, about what chance gives
(1.2 expected). An earlier version of this section said each family "beats its
own matched-width noise control", so the families carried information and were
merely redundant. The controls do not support that, and it is retracted. Its
table (baseline 0.4342 and so on) also had no committed script behind it. What
stands is the narrower conclusion: none of the three should be added.

That is the stopping point. The feature set is sixteen columns, and adding to it
has been tried and does not pay.

## Willingness to attempt 50+, measured separately

`fourth_down.py` lumps every in-range fourth down into one bucket at
`yardline_100 <= 38`, so it cannot see the long-range decision on its own. That
bucket reaches about a 56-yard attempt: in the play-by-play, kick distance is
`yardline_100 + 18` for 89.7% of field goal attempts (+19 for 8.8%, +17 for
1.5%). I measured the long decision separately in
`experiments/fifty_plus/fifty_plus.py`, which writes `reports/fifty_plus_*.csv`.
The primary bands use the `+17` convention: long is yardline 33 to 43, short
(the control) is 20 to 32. Because the offset is usually 18, that long band is
mostly 51 to 61-yard tries, so the script also runs the `+18` bands (yardline
32 to 42 and 19 to 31). The rate is field goal attempts over fourth-down
decisions, regular season, with `fourth_down.py`'s filters.

| season | 50+ band, FG attempt rate | short band, FG attempt rate |
|---|---|---|
| 2015 | 0.274 | 0.908 |
| 2016 | 0.266 | 0.868 |
| 2017 | 0.243 | 0.888 |
| 2018 | 0.276 | 0.877 |
| 2019 | 0.254 | 0.845 |
| 2020 | 0.296 | 0.848 |
| 2021 | 0.316 | 0.781 |
| 2022 | 0.387 | 0.824 |
| 2023 | 0.369 | 0.801 |
| 2024 | 0.447 | 0.793 |
| 2025 | 0.440 | 0.793 |
| 2026, 7 weeks | 0.477 | 0.793 |

Long-range willingness sat between 0.243 and 0.296 through 2020, then climbed
12 points in four years, from 0.316 in 2021 to 0.440 in 2025. The short rate
fell from 0.908 in 2015 to 0.781 in 2021 and has been flat since. So a single
bucket was averaging two different trends: from 2015 to 2021 they moved in
opposite directions, and since 2021 the long rate has kept rising while the
short rate sat still.

**The team spread looks large, and much of it is not real.** In 2025 New
England went 0 for 7 and Dallas 11 for 12. But a team gets a median of only 9
such chances a season. Decomposing the 2025 cross-team variance:

| component | variance | sd |
|---|---|---|
| observed across teams | 0.05244 | 0.229 |
| binomial sampling noise | 0.02516 | 0.159 |
| **true team difference** | **0.02728** | **0.165** |

In 2025, 52.0% of the visible spread is a real team property, with a 95%
interval from resampling teams of 11.4% to 69.7%. One season's estimate is
unstable: across the 11 complete seasons it runs from -22.7% to 63.8% (median
24.5%), two seasons (2018 and 2019) estimate the true variance at or below zero,
and in 8 of the 11 most of the visible spread is noise. The empirical-Bayes
shrinkage constant is 8.0 prior observations from 2025 alone, so a team with 9
chances keeps 53% of its own rate, and 15.3 pooled over 2015 to 2025, so it
keeps 37%. Raw year-over-year persistence is weak: r = +0.20 averaged over 10
season pairs, one of them negative, against +0.27 for `fourth_down.py`'s go rate.

An earlier version of this section gave 34.5% real, a shrinkage constant of
18.3, a median of 10 chances, persistence of +0.22 against +0.42, and only three
seasons of rates. None of it had committed code, and the figures above replace
it.

**Shrunk, it does not help detectably.** The test feature is the team's
previous-season long rate, shrunk toward the league with the pooled constant,
added to the 16-feature ridge under the backtest's folds. That arm was fixed
before the run. It moves decision lift from 0.427 to 0.429 and within-week rho
by +0.0014. The paired week-block bootstrap gives +0.0023, 95% CI [-0.0074,
+0.0121], p = 0.64. It wins 4 of 8 seasons, season-level t p = 0.69. The
coefficient is positive in 7 of 8 folds, but expanding windows share most of
their training seasons, so that is weak support for the sign. The raw rate
(+0.0031, p = 0.31) and the `+18` bands (+0.0003 and +0.0006) are no better.

Not added to the sixteen. An earlier version called this a near miss: +0.021,
95% CI [-0.018, +0.059], p = 0.27 on 72 weeks, from a run with no committed
code, blamed on "six seasons of clean play-by-play". There are eleven complete
seasons, 2015 to 2025, and this run uses all of them on 141 test weeks. The gain
it finds is about a tenth of the earlier one, and its interval tops out below
it. The protocol and every result are in `experiments/fifty_plus/README.md`.

## Live 2026 record

The backtest stops at 2025. 2026 is scored only from boards that existed before
the games (`score_boards.py`, `reports/live_2026_scorecard.csv`). A game counts
only if the board's exact bytes were committed before its kickoff, and the
scorecard also records when each commit first reached GitHub.

- Weeks 2, 3, 4 and 7 were committed before kickoff. Weeks 3 and 4 are the
  boards as first committed (eb44e17, d964182). A later commit, 24a3c60 on
  2026-10-20, had rewritten them after their games, and that rewrite is
  reverted.
- Weeks 5 and 6 were first committed in 24a3c60, after their games. They are
  shown and not counted.
- Week 4's commit is dated 2026-09-29, but GitHub first received it on
  2026-10-16. On the publicly checkable clock only weeks 2, 3 and 7 count.
- The week 8 board was committed and pushed on 2026-10-28, before the week's
  first game.

| week | kickers scored | board lift | implied-total lift | board minus implied | counted |
|---|---|---|---|---|---|
| 2 | 30 | -0.122 | +0.116 | -0.238 | yes |
| 3 | 32 | -0.240 | -0.456 | +0.216 | yes |
| 4 | 32 | +0.968 | +0.739 | +0.229 | yes, commit clock only |
| 5 | 28 | +1.294 | +0.522 | +0.771 | no |
| 6 | 25 | +0.473 | -0.608 | +1.082 | no |
| 7 | 26 | +0.837 | +0.669 | +0.168 | yes |
| **weeks 2, 3, 4, 7** | **120** | **+0.361** | **+0.267** | **+0.094** | |

Lift is the same R=2 decision lift as the backtest. Two listed kickers who did
not kick in week 7 are left out of that week.

On the four weeks committed before kickoff the board beat ranking by the implied
total by +0.094 a week, t 95% interval [-0.26, +0.45]. On the public-push clock
(weeks 2, 3 and 7) the margin is +0.048, [-0.57, +0.67]. Four weeks cannot tell
the board from the line: the interval is 0.71 points wide, around a backtest
edge of 0.074. The board lost to a random start in weeks 2 and 3. Counting the
after-the-fact weeks 5 and 6 would raise the margin to +0.371, which is exactly
why they are not counted. Mean predicted `P(15+)` over the 120 counted
kicker-games was 9.2%; 10.8% of them scored 15 or more.

## How the headline moved

Four published versions of the headline, the commit behind each, and what
changed:

| commit | ridge lift (share of oracle) | ridge minus implied total | what changed |
|---|---|---|---|
| 58e3d15 | 0.434 (17.8%) | +0.072 [+0.014, +0.130], p = 0.015 | First rebuild. `backtest.py` did not fit the ridge, so this came from a separate run the code could not reproduce. |
| 6220fa2 | 0.4415 (18.1%) | +0.0795 [+0.0234, +0.1369], p = 0.006 | The ridge became a row in `backtest.py`, run on refreshed data (14,487 kicker-games against 14,457). cba5708 then had the board's note read these from the saved backtest instead of carrying typed-in copies. |
| f04d00e | 0.432 (17.6%) | +0.079 [+0.030, +0.128], p = 0.001; season bootstrap p = 0.028 | Exact lift instead of 300 simulated draws a week, and the roof-flag fix, which took the model from 14 live features to 16. |
| 8048153 | 0.427 (17.4%) | +0.074 [+0.024, +0.124], p = 0.0036; season-level t p = 0.095 | Every play-by-play season fetched by the code and checked for coverage, and every CSV regenerated on one data build, with inputs through 2026 week 7. The season-level t-test was added. I have not separated how much of the change each part caused. |

Across the four the lift stayed between 0.427 and 0.442 and the edge over the
line between +0.072 and +0.080. The week-block p moved from 0.015 to 0.001 and
back to 0.0036. A p-value that moves by a factor of 15 with data refreshes and
scoring fixes is a poor basis for a claim, which is one more reason to quote the
season-level test.

## Honest limits

- The market inputs are closing lines. The board is valid for a lineup set close
  to kickoff; the backtest says nothing about a Tuesday read on an opening line.
- The edge is small. Every fitted model in the bake-off sits between 3.468 and
  3.493 MAE, against 3.504 for a constant at the training median and 3.571 for
  a kicker's career average. On the decision metric the ridge gets 17.4% of the
  oracle's lift, and its edge over the line alone is not established season by
  season.
- 2018 to 2025 is not a clean test of the shipped model. The 16 features, alpha
  = 100 and the ridge itself were chosen while looking at these seasons. The
  bake-off tunes its challengers honestly; the shipped model was not chosen that
  way. The live 2026 boards are the out-of-sample test, and four weeks of them
  settle nothing yet.
- `P(15+)` is right on average and weak per kicker.
- Kicker is the position where streaming on implied total gets you most of the
  available value: the line alone gets 83% of the ridge's lift. This model is
  worth having; it is not worth agonising over.
- The roster read comes from the last man who actually kicked for each team, so a
  kicker signed on Wednesday will not appear. Rows carry a staleness flag, and a
  kicker Sleeper's player list puts on another team, or on none, is dropped when
  that list can be fetched.
- Missed field goals score 0 here. `SCORING` in `build_dataset.py` sets the
  target the ridge learns. The count model's simulator keeps its own copy of the
  point values in `model.py` and has no term for a miss, so a league that scores
  a miss at -1 needs both changed.

## Reproducing this

Python 3.11 or newer, and set `OMP_NUM_THREADS=1`: LightGBM thread contention
can turn a one-second fit into minutes.

```
pip install -r requirements.txt
export OMP_NUM_THREADS=1
python scripts/run_pipeline.py              # every pipeline step in order; --bakeoff adds the bake-off
```

Or one step at a time:

```
python src/kicker_analysis/build_dataset.py      # one row per kicker per game: 14,638 on this build, 1999 to 2026 week 7
python src/kicker_analysis/fourth_down.py        # coach aggression, every season 2015 on; fails on a gap
python src/kicker_analysis/backtest.py           # the backtest, calibration, variance and skill tables, 2018-2025
python src/kicker_analysis/model_bakeoff.py      # 14 challengers vs the ridge, coefficients, eras; about 2.5 minutes
python src/kicker_analysis/figures.py            # the charts, drawn from the tracked CSVs
python src/kicker_analysis/score_boards.py       # the live 2026 record
python src/kicker_analysis/predict_week.py       # this week's board; the site file only with --site-out or KICKER_SITE_JS
python experiments/feature_families/ridge_retest.py   # the feature-family re-test; about 2 minutes
python experiments/fifty_plus/fifty_plus.py      # the 50+ study; seconds
pytest -q                                        # every module's --self-check, offline
```

Inputs are fetched live, so a rerun on later data can move the last digit.
`reports/data_manifest.json` records the hash of every input behind the tracked
CSVs.

Where each section's numbers live:

| section | file |
|---|---|
| lift table, comparisons | `reports/backtest_summary.csv`, `reports/season_level_tests.csv`; the week-block intervals are printed by `backtest.py` |
| variance and component skill | `reports/variance_decomposition.csv`, `reports/component_skill.csv` |
| calibration | `reports/calibration_summary.csv`, `reports/calibration_reliability.csv` |
| bake-off, nested alpha, coefficients, eras, Holm | `reports/model_bakeoff_*.csv` |
| coach trend | printed by `fourth_down.py` |
| feature families | `reports/feature_family_ridge_retest.csv`, `experiments/feature_families/ridge_retest.log` |
| 50+ | `reports/fifty_plus_*.csv`, `experiments/fifty_plus/fifty_plus.log` |
| live 2026 | `reports/live_2026_scorecard.csv`, `reports/live_2026_scorecard_weeks.csv` |
| the retired 73-feature table | `legacy/reports/model_summary.csv` |

Most modules carry an assert-based `--self-check` that runs offline on
synthetic data, and `pytest -q` runs all of them with the network blocked. Two
real bugs were caught by those checks before they shipped: a sign error in the
implied-total algebra, and a team-game aggregation that double counted a score
when two men kicked in the same game.
