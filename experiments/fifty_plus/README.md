# Willingness to attempt 50+, measured separately

`fourth_down.py` puts every fourth down at `yardline_100 <= 38` into one bucket,
so it cannot tell the routine 40-yard decision from the long one. This folder
measures the long decision on its own, from play-by-play. It is an experiment,
not part of the pipeline: its conclusion is that the feature does not earn a
place in the sixteen.

## What it measures

- **Bands.** Long band: the attempt would be 50 to 60 yards. Short band (the
  control): 37 to 49 yards. Filters are `fourth_down.py`'s, imported from it:
  regular season, win probability 5% to 95%, not in the last two minutes of a
  half, exactly one of FG / punt / run-or-pass. The rate is FG attempts over
  decisions, and punts stay in the denominator.
- **Yardline conversion, run two ways.** The documented convention, the one
  METHOD.md uses, is `distance = yardline_100 + 17` (yardline 33-43 and
  20-32). In 2015-2026 regular-season play-by-play,
  `kick_distance - yardline_100` is 18 for 89.7% of FG attempts, 19 for 8.8%
  and 17 for 1.5%. So the documented long band is
  mostly 51 to 61 yard attempts, and a typical 50-yarder from the 32 falls in
  the short control. The script runs the documented +17 bands (a
  reproduction) and the offset it measures from the data (+18, yardline 32-42
  and 19-31). Every output row carries `fg_offset`.
- **Variance decomposition**, every season: observed cross-team variance of the
  team long rates, split by the method of moments into binomial sampling noise
  and true team difference. Also reported: `share_real = true / observed` with
  an interval from resampling teams, the empirical-Bayes prior strength `k` (in
  pseudo-observations), and year-over-year persistence of raw team rates beside
  `fourth_down.py`'s `go_rate`. The algebra is in the script's docstring.
- **Decision-lift test.** The feature is the team's previous-season long rate,
  shrunk toward that season's league rate as `(x + k mu) / (n + k)`, with `k`
  pooled over seasons up to that one. It is added to the 16-feature ridge. The
  split is backtest.py's (test seasons 2018-2025, each fit on earlier seasons
  only). Lift is exact (`backtest.decision`) and the paired test is a week-block
  bootstrap (`backtest.boot_lift`). The eight per-season differences also get a
  t-test, a Wilcoxon test and a count of seasons won. The arm under test was
  fixed before the run: shrunk, documented band (`primary = True`). The raw
  rate and the +18 band are run beside it.

Nothing in the feature comes from the season being predicted. It is built from
season s-1 only, on a dense team x season grid, so a missing season stays
missing. The self-check asserts this.

## Results, and how they compare with the earlier write-up

From `reports/fifty_plus_*.csv` and the run's stdout, `fifty_plus.log`. The
figures METHOD.md gave for this section before this script existed had no
committed code behind them (it said the extractor was in `notes/`, which held
none). This script reproduces some of them and not others:

| quantity | earlier METHOD.md | this run (+17 band) |
|---|---|---|
| league long rate 2021 / 2023 / 2025 | 0.314 / 0.375 / 0.437 | 0.316 / 0.369 / 0.440 |
| short rate 2021 / 2023 / 2025 | 0.782 / 0.802 / 0.788 | 0.781 / 0.801 / 0.793 |
| 2025 extremes | NE 0/7, DAL 11/12 | NE 0/7, DAL 11/12 |
| 2025 median chances per team | 10 | 9 |
| 2025 observed variance across teams | 0.03764 | 0.05244 |
| 2025 sampling-noise variance | 0.02466 | 0.02516 |
| 2025 true team variance | 0.01298 | 0.02728 |
| 2025 share of spread that is real | 34.5% | 52.0% [11.4%, 69.7%] |
| shrinkage constant k | 18.3 | 8.0 (2025 alone), 15.3 (pooled 2015-2025) |
| year-over-year r, raw 50+ rate | +0.22, one pair negative | +0.20 over 10 pairs, one negative |
| year-over-year r, fourth-down aggression | +0.42 | +0.27 (`go_rate`) |
| lift gain from the shrunk feature | +0.021 [-0.018, +0.059], p = 0.27 (72 weeks) | +0.0023 [-0.0074, +0.0121], p = 0.64 (141 weeks) |
| within-week rho gain | +0.015 | +0.0014 |

The two lift rows are not the same test window. The earlier figure came from
a separate run on 72 weeks with a base lift of 0.4212, and which seasons it
tested is not recorded. This run tests every season backtest.py tests
(2018-2025, 141 weeks). Its last four test seasons alone (2022-2025, also 72
weeks) give +0.0060, the week-weighted mean of the per-season rows in
`fifty_plus_lift.csv`. That is still under a third of +0.021.

Read plainly:

- The league trend reproduces. Long-range willingness rose from 0.316 in 2021
  to 0.440 in 2025. The short control was flat over those years, but it had
  fallen from 0.908 in 2015 to 0.781 in 2021. "The two move in opposite ways"
  holds from 2021, not over the whole record.
- The 2025 "34.5% real" figure does not reproduce: this run gets 52.0%. One
  season's estimate is also very unstable. Across the 11 complete seasons it
  runs from -22.7% to 63.8% (median 24.5%), and in two seasons (2018, 2019) the
  estimated true variance is below zero. In 8 of the 11 seasons most of the
  visible team spread is noise, but no single percentage describes it.
- **The lift test reaches a weaker conclusion than the earlier write-up.**
  That called the feature a near miss: +0.021, p = 0.27, "the sign is right".
  Here the shrunk feature adds +0.0023 points a week, p = 0.64, and wins 4 of 8
  seasons. Its 95% interval tops out at +0.0121, below the +0.021 reported
  before. The ridge coefficient is positive in 7 of 8 folds. The folds are
  expanding windows that share most of their training seasons, though, so that
  is weak support for the sign. The gain itself is indistinguishable from zero.
  The raw rate (+0.0031, p = 0.31) and the +18 band (+0.0003 and +0.0006) are
  no better. This is not a near miss. The feature stays out. More seasons would
  be testing for a gain about a tenth the size of the one reported before.

## Rerun

From the repo root, with `OMP_NUM_THREADS=1`:

```
python src/kicker_analysis/build_dataset.py     # data/processed/kicker_games.parquet
python src/kicker_analysis/fourth_down.py       # data/processed/fourth_down.parquet, fills the pbp cache
python experiments/fifty_plus/fifty_plus.py     # seconds (7 s on 2026-10-28); writes the three CSVs
python experiments/fifty_plus/fifty_plus.py --self-check   # offline, synthetic data, writes nothing
```

It reads `data/external/pbp_<season>.parquet` for every season from 2015 on and
stops if any season from 2015 to the latest cached one is missing. Options: `--season` picks the season
whose decomposition is printed (default 2025; the CSV has every season), and
`--boot` sets the bootstrap draws (default 10,000, seed 0).

## Outputs

All in `reports/`, with one block of rows per `fg_offset`:

- `fifty_plus_rates.csv`: league (`level = league`, `team = ALL`) and
  team-season rows. For each band: decisions `n`, `fg`, `go`, `punt`, and
  `rate = fg / n`. `weeks` is the number of regular-season weeks in the data
  (2026 is partial).
- `fifty_plus_variance.csv`: one row per season. `var_observed`, `var_noise`,
  `var_true` and their square roots; `share_real` with `share_real_lo/hi`;
  `k_prior` (that season alone; `inf` when the true variance is not positive)
  and `k_pooled_through` (the k used for the next season's feature);
  `own_weight_at_median = median_n / (median_n + k_prior)`; and `yoy_r_long`,
  `yoy_r_short`, `yoy_r_go_rate`, each the correlation across teams with the
  season before.
- `fifty_plus_lift.csv`: per arm, one pooled row (`season = 2018-2025`) and one
  row per test season. Lift at rosters of 2 and 3, within-week rho, the
  difference from the base ridge with its week-block interval and p, the
  season-level t, Wilcoxon and win count, and the standardised ridge
  coefficient on the added column (`coef_std`; on the pooled row, the mean over
  folds).
