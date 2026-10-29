# Experiments

One-off measurements behind a design decision, kept so that a negative result
has its evidence on disk. Nothing here is part of the pipeline: no module in
`src/` imports it and `scripts/run_pipeline.py` does not run it. Scripts and
their raw logs live here. The evidence CSVs they write stay in `reports/`, beside
the rest of the tracked evidence.

## feature_families/

The question: does any of three candidate feature families add to the model?
Each family is built by a module in `src/kicker_analysis/`, and the shipped model
uses none of them.

| family | builder | columns | what it measures, all lagged to before kickoff |
|---|---|---|---|
| red-zone | `feat_redzone.py` | 27 | red-zone trips per game, touchdown and field-goal-attempt rates, the inside-the-10 split, expected stalled trips (trips x (1 - TD rate)), and the same rates for the defence faced |
| drives | `feat_drives.py` | 21 | drives per game, share reaching the opponent 40, share of those that stall without a touchdown, points and penetration yards per drive, three-and-out rate, starting field position |
| leg/stadium | `feat_legstadium.py` | 13 | the kicker's attempt-distance distribution (mean, 90th percentile, 50+ share and its trend), shrunk make rates by distance bucket, and a venue make-over-expected index from prior seasons |

### The current test: `ridge_retest.py`

**What it tests.** Whether each family improves the model that actually orders
the board, `model.RidgeRanker` on the 16 named `model.FEATURES`, on the metric the
README reports. The protocol is `backtest.py`'s: season-forward folds 2018-2025
(8 folds, 4,211 test rows in 141 weeks), the ridge unchanged (median imputation,
standardisation, alpha 100, nothing retuned), and three scores: pooled MAE,
within-week Spearman (`backtest.rho`) and the exact R=2 decision lift
(`backtest.decision`). Arms are the 16 features alone, plus each family, plus all
three. The family tables are rebuilt from the play-by-play cache on every run
and attached with a left join that must neither add nor drop rows. On the test
rows the red-zone and drives columns are fully populated, and the leg/stadium
columns and all three together are 98.2% and 99.6% non-missing (the CSV's
`coverage` column).

**Two controls.** Adding columns moves the metrics even when the columns carry
nothing, so each family is also scored against two matched controls, 50 seeded
draws each:

- *noise*: the 16 features plus the same number of independent standard-normal
  columns (widths 13, 21, 27 and 61).
- *shuffled*: the 16 features plus the family's own columns, their rows permuted
  jointly within each season. This keeps the family's values, the correlations
  among its columns and its missing cells, and breaks only the link to the
  kicker-week. It matters because a family's columns are correlated (red-zone
  and drives carry each quantity over three windows), correlated columns give
  the ridge fewer independent directions to fit noise in, and so beating the
  noise control can come from that structure alone. In this run the shuffled
  control cost less than the noise control of the same width in seven of the
  eight means (lift and MAE for four arms); the exception is leg/stadium's
  lift, 0.4031 shuffled against 0.4055 noise.

"Beats a control" means the family scores better than the control's draws: the
share of draws it beats, and the Monte Carlo p-value
(1 + draws at least as good) / 51. The change from the baseline also gets
`backtest.boot_lift`, the paired week-block bootstrap the README uses.

**Exercises.** `model.RidgeRanker`, `model.FEATURES`, `model.load`;
`backtest.decision`, `backtest.rho`, `backtest.boot_lift`, `backtest._pick_probs`
(in the self-check); `feat_redzone.build`, `feat_drives.build`,
`feat_legstadium.build`.

**Evidence.** `reports/feature_family_ridge_retest.csv` (one row per arm; real
arms have `kind = real`, the matched controls `kind = noise` or
`kind = shuffle` with the mean over draws, its sd, and the min and max of the
lift) and `feature_families/ridge_retest.log` (the run's stdout, including lift
per test season). The baseline row reproduces the
ridge row of `reports/backtest_summary.csv` (lift 0.4272, rho 0.1713), so the
re-test ran on the same data as the tracked backtest.

| added to the 16 | cols | MAE | within-week rho | R=2 lift | lift vs baseline, 95% CI, p | matched noise lift, mean (sd) | noise draws beaten on lift | on MAE |
|---|---|---|---|---|---|---|---|---|
| nothing (baseline) | 0 | 3.4718 | 0.1713 | 0.4272 | | | | |
| red-zone | +27 | 3.4765 | 0.1491 | 0.3697 | -0.0574 [-0.0853, -0.0294], no draw of 10,000 reached 0 | 0.3791 (0.0149) | 13 of 50 | 49 of 50 |
| drives | +21 | 3.4823 | 0.1675 | 0.4178 | -0.0093 [-0.0420, +0.0236], p 0.57 | 0.3902 (0.0166) | 48 of 50 | 19 of 50 |
| leg/stadium | +13 | 3.4730 | 0.1592 | 0.3955 | -0.0316 [-0.0728, +0.0095], p 0.14 | 0.4055 (0.0106) | 8 of 50 | 44 of 50 |
| all three | +61 | 3.4853 | 0.1415 | 0.3469 | -0.0803 [-0.1323, -0.0266], p 0.002 | 0.3375 (0.0195) | 36 of 50 | 50 of 50 |

Against each family's own rows shuffled within season (50 draws each):

| family | shuffled lift, mean (sd) | shuffles beaten on lift | on MAE | on within-week rho |
|---|---|---|---|---|
| red-zone | 0.3948 (0.0129) | 0 of 50 | 40 of 50, p 0.22 | 0 of 50 |
| drives | 0.4025 (0.0140) | 46 of 50, p 0.098 | 6 of 50 | 46 of 50, p 0.098 |
| leg/stadium | 0.4031 (0.0136) | 13 of 50 | 35 of 50, p 0.31 | 15 of 50 |
| all three | 0.3623 (0.0223) | 14 of 50 | 33 of 50, p 0.35 | 18 of 50 |

**Verdict.**

1. *No family helps the ridge.* Every arm is worse than the 16 features alone on
   all three metrics. Red-zone lowers the lift in all 8 test seasons, drives and
   leg/stadium in 6 of 8, all three in 7 of 8. The red-zone and all-three losses
   are outside the bootstrap interval; the drives and leg/stadium losses are not
   distinguishable from zero.
2. *None beats either control on the decision lift.* Pure noise is not free
   here: at alpha 100, every added standard-normal column costs about 0.0015 to
   0.0018 points a week of lift (from -0.0216 for 13 columns to -0.0896 for 61).
   Drives comes closest: better than 48 of 50 noise draws (Monte Carlo p 0.059)
   and 46 of 50 shuffles (p 0.098), and it still lowers the lift by 0.0093.
   Red-zone orders the week worse than every one of its 50 shuffled copies
   (0.3697 against 0.3717 to 0.4319): lined up with the right kicker-weeks, its
   columns cost more ordering skill than the same columns lined up at random.
   Leg/stadium (8 of 50 noise draws, 13 of 50 shuffles) and all three (36 and
   14) beat neither.
3. *The MAE edge over random columns does not survive the shuffled control.*
   Red-zone (49 of 50 noise draws, p 0.039) and all three (50 of 50, p 0.020)
   raise MAE by less than independent random columns of the same width do.
   Against their own rows shuffled within season they do not stand out (40 of
   50, p 0.22; 33 of 50, p 0.35): most of that edge is reproduced by the same
   columns lined up at random, so it is not evidence that they carry
   information about a kicker's points. Leg/stadium and drives beat neither
   control on MAE (44 and 19 of 50 noise draws; 35 and 6 of 50 shuffles).

Of the 24 control comparisons (four arms, three metrics, two controls), two
reach p < 0.05, both on MAE and both against the noise control, where 1.2
would be expected by chance alone. So the earlier claim that each family
"beats its own matched-width noise control" is not supported. The re-test
supports the narrower conclusion that none of the families should be added to
the model.

The ridge table that README.md and METHOD.md carried before this script existed
(baseline 0.4342, red-zone 0.3782, drives 0.4111, leg/stadium 0.3759, all three
0.3240) was not reproducible from any committed script. It also predates the
roof-flag repair and the exact lift. The new lifts differ from it by at most
0.023, and the conclusion that every family is worse than the baseline on every
metric is unchanged.

### The earlier LightGBM runs

These came first and are kept as they ran. All three added a family to the
73-column LightGBM that preceded the ridge (`features.feature_columns()` at the
time; 400 trees, learning rate 0.03, 31 leaves, at least 40 rows per leaf). They
scored pooled MAE and Spearman, not the decision lift, over test seasons
2019-2025. 2021 was then missing from `kicker_games.parquet` (the nflverse kicking
release has no 2021 rows), so each run had 6 folds. `build_dataset.py` now fills
2021 from play-by-play, and the current build admits 69 columns to
`feature_columns()`, not 73. The saved logs and CSVs are from the 6-fold runs,
so a rerun gives 7 folds, a different baseline and different numbers. All four
scripts ran to completion on the build of 2026-10-26 (`feat_legstadium_eval.py`
reports the day-of `wind_outdoor` column as absent rather than stopping on it).

| script | exercises | evidence | result |
|---|---|---|---|
| `feat_redzone_eval.py` | `feat_redzone.FEATURE_COLS`, `COMPOSITE_COLS`, `RAW_PAIR_COLS`; `features.build_features`, `feature_columns` | `reports/feat_redzone_results.csv`, `feat_redzone_eval.log` | Pooled MAE: base 3.6185; + composite 3.6086 (paired t p 0.406); + raw pair 3.6094 (p 0.597); + all 27 columns 3.5708 (p 0.054, Wilcoxon p 0.094, better in 5 of 6 folds). |
| `feat_redzone_control.py` | the same 27 columns, shuffled within each train and test split | `reports/feat_redzone_control.csv`, `feat_redzone_control.log` | Shuffled, the 27 columns moved MAE by -0.0195, -0.0357 and +0.0022 on three seeds (mean -0.0177), against -0.0477 for the real columns. The real columns did better than all three shuffles, but the real-minus-shuffled margin, 0.030, is inside the 0.038 range of the shuffles themselves, and three draws cannot give a Monte Carlo p-value below 0.25, so under LightGBM red-zone was not shown to beat its control. |
| `feat_drives_eval.py` | `feat_drives.FEATURE_COLS`, `JOIN_KEYS` | `feat_drives_eval.out` (stdout only; no CSV) | Pooled MAE 3.6185 to 3.6151 (-0.0034), Spearman 0.0821 to 0.0937. Over 6 folds MAE improved in 4, paired t p 0.857, Wilcoxon p 0.844. No noise control was run. |
| `feat_legstadium_eval.py` | `feat_legstadium.FEATURE_COLS`, `load_attempts`, `stadium_index` | `legstadium_eval_out.txt` | **No result.** The saved output stops at "fold 2024 done". The 2025 fold and the summary never ran, and `reports/feat_legstadium_results.csv`, which the script writes last, was never produced. The ridge re-test above is the only evidence on record for this family. |

### How to rerun

From the repo root, with `OMP_NUM_THREADS=1` set (LightGBM thread contention
turns seconds into minutes):

```
python src/kicker_analysis/build_dataset.py        # data/processed/kicker_games.parquet
python src/kicker_analysis/fourth_down.py          # fourth_down.parquet; fills the pbp cache
python experiments/feature_families/ridge_retest.py               # writes reports/feature_family_ridge_retest.csv
python experiments/feature_families/ridge_retest.py --self-check  # offline, writes nothing
```

`ridge_retest.py` rebuilds the three family tables itself and takes about two
minutes on one thread (131 seconds in the run behind the tracked CSV).
`--draws N` changes the number of draws per control.

The LightGBM scripts read the family tables from `data/processed/`, so build
those first:

```
python src/kicker_analysis/feat_redzone.py         # redzone.parquet
python src/kicker_analysis/feat_drives.py          # drives.parquet
python src/kicker_analysis/feat_legstadium.py      # legstadium.parquet
python experiments/feature_families/feat_redzone_eval.py     # then feat_redzone_control.py
python experiments/feature_families/feat_drives_eval.py
python experiments/feature_families/feat_legstadium_eval.py
```

Two cautions. `feat_redzone_eval.py` caches its frame and per-fold progress in
`reports/_feat_redzone_frame.parquet`, `_feat_redzone_base.json` and
`_feat_redzone_partial.csv`, and reuses them on the next run, so delete them
after rebuilding the data. And a rerun overwrites `reports/feat_redzone_results.csv`
and `reports/feat_redzone_control.csv`, the evidence the saved logs correspond
to.

## fifty_plus/

`fifty_plus.py` measures a staff's willingness to attempt a 50-to-60-yard field
goal, a decision `fourth_down.py` folds into one bucket with every other fourth
down at or inside the opponent's 38. It asks how much of the spread between
teams is a team property rather than sampling noise, and whether the prior
season's rate, shrunk toward the league, adds decision lift to the 16-feature
ridge under the same season-forward protocol as `backtest.py`. It writes
`reports/fifty_plus_rates.csv`, `fifty_plus_variance.csv` and
`fifty_plus_lift.csv`. The folder's own `README.md` gives the protocol, the
results and how to run it, and `fifty_plus.log` is the run's stdout.

The result: it adds nothing detectable. The shrunk prior-season rate moves the
ridge's lift from 0.4272 to 0.4295, +0.0023 points a week (95% CI -0.0074 to
+0.0121, p = 0.64), and wins 4 of 8 test seasons. It is not in the model.
