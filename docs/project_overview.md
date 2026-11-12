# How the repo is organised

The README says what the model does and how well it does it, and `METHOD.md`
makes the argument. This page is the map: what each directory holds, which
module writes which file, and which module reads it. Column definitions are in
[`data_dictionary.md`](data_dictionary.md).

## Directories

| path | what is there |
|---|---|
| `src/kicker_analysis/` | the pipeline, one module per stage (below). Stages run as `python src/kicker_analysis/<module>.py` from the repo root. Every module opens with a docstring explaining why it is built the way it is, and every one of the 13 carries an assert-based `--self-check` that runs offline on synthetic data. |
| `scripts/run_pipeline.py` | runs the pipeline steps in order and stops at the first failure. `--bakeoff` adds the bake-off; `--week` and `--force` are passed to `predict_week.py`; `--self-check` runs every module's self-check instead. |
| `tests/` | `test_self_checks.py`, a pytest wrapper that finds every module with a self-check by reading its source, runs each in a fresh interpreter, and also runs `scripts/run_pipeline.py`'s own checks. `test_repo_hygiene.py` fails on a personal absolute path in any committed text file, and on a relative link or image in `README.md` or `METHOD.md` that does not resolve or could not be committed. `tests/offline/sitecustomize.py` makes any network connection fail, so a check that starts downloading fails. |
| `.github/workflows/ci.yml` | runs `pytest` on Python 3.11, 3.12 and 3.13 on every push and pull request |
| `.github/workflows/weekly-board.yml` | Wednesdays at 15:07 UTC, September to January: builds the dataset, the fourth-down table and the coming week's board (`predict_week.py --max-days-ahead 6`), and commits only a new `reports/week_*_kickers.csv`. It fails rather than commit a change to a board that is already tracked. |
| `reports/` | the tracked evidence: the small CSVs behind the numbers in the README and `METHOD.md`, the input manifest, and the weekly boards as first committed. `reports/reruns/` keeps the week 3, 4 and 7 boards as they were regenerated after their games, apart from the forecasts. Parquet dumps land here too and are gitignored. |
| `experiments/` | one-off experiments (the three refuted feature families, willingness to attempt 50+) with their scripts and saved logs. The CSVs they write are in `reports/`. Not part of the pipeline. |
| `notes/` | dated decision write-ups. `model-selection-2026-09-29.md` is the bake-off behind the ridge; a note at its top gives the 2026-10-28 rerun. |
| `legacy/` | the retired R and Python pipelines, the R workflow's data and the v1 harness's output (`legacy/reports/`), each with the reason it was retired, in `legacy/README.md`. Nothing there runs. |
| `docs/` | this page, `data_dictionary.md`, and `images/`: the README's three charts (light and dark PNGs from `figures.py`) and `fantasy-football-04.webp`, the author's graphic of the inputs, the two models and the 2026-09-29 headline, kept in the repository and no longer shown in the README because its "Beats Vegas totals" and "Calibrated" labels read stronger than the current evidence |
| `data/` | not in the repository. Created on the first run: `data/external/` caches the downloads and `data/processed/` holds the built tables. Both are gitignored, and `build_dataset.py` and `fourth_down.py` rebuild them from scratch. |

`pyproject.toml` installs the package in editable mode (`pip install -e .`);
`requirements.txt` is its one dependency list and `requirements-dev.txt` adds
pytest.

## Modules

| module | job |
|---|---|
| `build_dataset.py` | nflverse and the schedule to one row per kicker per game, with that week's fantasy points as the target. Stops on any release-versus-play-by-play parity difference or a missing or thin season, and writes `reports/data_manifest.json`. |
| `fourth_down.py` | coach decisions on fourth down in field-goal range, per team-game: field-goal share and go rate. Fetches every play-by-play season from 2015 to the current one and stops on a gap. |
| `features.py` | the lagged features, each knowable before kickoff, and the named list of columns that are not |
| `model.py` | the 16-feature list; `RidgeRanker`, which orders the board; `CompoundKickerModel`, two count stages simulated through the scoring rule for floor, ceiling and P(15+); and `load()`, which assembles the modelling frame |
| `backtest.py` | season-forward evaluation, test seasons 2018 to 2025 (`FIRST_TEST`, `LAST_TEST`): exact decision lift, week-block bootstrap, season-level tests, component skill, the variance decomposition and calibration. 2026 is not a fold; it is scored live by `score_boards.py`. |
| `predict_week.py` | the weekly board: pseudo-rows for the coming games, through the same feature builder, then both models |
| `score_boards.py` | scores the published weekly boards against actual points, per game, and records for each game whether the board's bytes were committed, and publicly pushed, before kickoff |
| `model_bakeoff.py` | nested season-forward bake-off of 14 challengers against the ridge, Holm-corrected. It writes only its own report files and never touches the model or the board. |
| `figures.py` | the README's three charts, drawn only from the tracked CSVs in `reports/`, each in a light and a dark version. `--markdown` prints the README's `<picture>` blocks with alt text built from the same CSVs. |
| `evaluate.py` | the earlier evaluation harness, kept for `prob_best` and `bootstrap_compare`. The headline numbers come from `backtest.py`. |
| `feat_drives.py`, `feat_redzone.py`, `feat_legstadium.py` | three candidate feature families (drive shape, red-zone efficiency, leg strength and stadium), all refuted. The model does not use them. |

## How data flows

```
nflverse kicking release ---+
nflverse play-by-play ------+--> build_dataset.py --> data/processed/kicker_games.parquet --+
nflverse games.csv ---------+    (downloads cached in data/external/,                       |
                                  their hashes in reports/data_manifest.json)               |
                                                                                            |
nflverse play-by-play 2015+ --> fourth_down.py --> data/processed/fourth_down.parquet ------+
                                                                                            |
                         features.build_features: the 16 features in model.FEATURES <-------+
                                     |                       |                      |
                                     v                       v                      v
                               backtest.py           model_bakeoff.py        predict_week.py
                    reports/backtest_summary.csv  reports/model_bakeoff_*  reports/week_<season>_<week>_kickers.csv
                    and five more evidence CSVs              |                      |
                                     |                       |                      v
                                     +----> figures.py <-----+               score_boards.py
                                         docs/images/*.png          reports/live_<season>_scorecard*.csv
```

`scripts/run_pipeline.py` runs the steps in this order: `build_dataset`,
`fourth_down`, `backtest`, `model_bakeoff` (only with `--bakeoff`), `figures`,
`score_boards`, `predict_week`. Each reads what an earlier step wrote. Without
`--bakeoff`, `figures.py` draws from the bake-off CSVs already in `reports/`.

`backtest.py` is fixed at test seasons 2018 to 2025, so the tracked numbers do
not move while 2026 is played. 2026 is scored only live, by `score_boards.py`,
from the boards as they were published.

`predict_week.py` refuses to build a board when the training history stops
before the previous week, and a board for week N trains and builds its features
only from games before week N. It never rebuilds an existing board CSV unless
given `--force`; `--out` writes a rebuild somewhere else. The board needs no
backtest run. Only when a site payload is written (`--site-out` or the
`KICKER_SITE_JS` environment variable) does it read
`reports/backtest_predictions.parquet` and `reports/calibration_summary.csv`
for the figures that payload quotes, and it leaves out any sentence whose
source is missing.

`fourth_down.py` requests every play-by-play season from 2015 to the current one
by number and fails if any is missing or thin, so once it has run the cache also
holds every season the `feat_*` builders read. Those builders still read
whatever is in `data/external/`, so run `fourth_down.py` before them.

| file | written by | read by | tracked |
|---|---|---|---|
| `data/external/kicking.parquet`, `pbp_<season>.parquet`, `games.csv` | `build_dataset.py` and `fourth_down.py` (play-by-play 2015 to the current season), as a download cache. `games.csv`, `kicking.parquet` and any play-by-play copy fetched before its season finished (1 March of the following year) are re-fetched once older than six hours, keeping the old copy if the refresh fails. Downloads are atomic, and each leaves a `<name>.fetch.json` record. | play-by-play: `fourth_down.py`, the `feat_*` builders and `experiments/fifty_plus/fifty_plus.py`. `games.csv`: `model.load`, `predict_week.py` (which refreshes it first), `score_boards.py`, `feat_legstadium.py` | no |
| `data/external/players-nfl.json` | `predict_week.py` (Sleeper's player list, through `build_dataset.fetch`, re-fetched after 24 hours) | `predict_week.drop_released`, upcoming slate only | no |
| `reports/data_manifest.json` | `build_dataset.py` and `fourth_down.py`, every normal run | nothing: the record of which cached inputs (name, url, sha256, bytes, fetched_utc, last_modified) produced the tracked tables | yes |
| `data/processed/kicker_games.parquet` | `build_dataset.py` | `model.load` (so `backtest.py`, `model_bakeoff.py` and the experiments), `predict_week.py`, `score_boards.py`, `evaluate.py`, `feat_legstadium.py` | no |
| `data/processed/fourth_down.parquet` | `fourth_down.py` | `model.load`, `predict_week.py`, `evaluate.py` | no |
| `data/processed/drives.parquet`, `redzone.parquet`, `legstadium.parquet` | `feat_drives.py`, `feat_redzone.py`, `feat_legstadium.py` | the LightGBM scripts in `experiments/feature_families/` | no |
| `reports/backtest_summary.csv` | `backtest.py` | `figures.py`; the README's headline table | yes |
| `reports/component_skill.csv`, `variance_decomposition.csv`, `season_level_tests.csv`, `calibration_summary.csv`, `calibration_reliability.csv` | `backtest.py` | `figures.py` (the last three), `predict_week.py` (`calibration_summary.csv`, site payload only); the README and `METHOD.md` | yes |
| `reports/backtest_predictions.parquet` | `backtest.py` | `predict_week.py` (site payload only) | no |
| `reports/model_bakeoff_*.csv` | `model_bakeoff.py` | `figures.py` (summary, folds, slices); `notes/model-selection-2026-09-29.md` | yes |
| `reports/model_bakeoff_predictions.parquet` | `model_bakeoff.py` | nothing | no |
| `reports/week_<season>_<week>_kickers.csv` | `predict_week.py`, which never overwrites one without `--force`; `.github/workflows/weekly-board.yml` commits each new one before kickoff. With `--site-out`, or with `KICKER_SITE_JS` for the upcoming week, it also writes the payload the author's Fantasy Football site loads. | `score_boards.py`; these are the published boards, each as first committed | yes |
| `reports/reruns/week_<season>_<week>_kickers_rerun_<date>.csv` | `predict_week.py`, run again after that week's games (weeks 3 and 4 in 24a3c60, week 7 in db8eba6) | nothing: kept so a rerun cannot stand in for the forecast, and outside the folder `score_boards.py` reads | yes |
| `reports/live_2026_scorecard.csv`, `live_2026_scorecard_weeks.csv` | `score_boards.py` | the README's live record. `score_boards.py` reads the weeks file back, to reuse its recorded clocks with `--offline` or without git history. | yes |
| `reports/feature_family_ridge_retest.csv` | `experiments/feature_families/ridge_retest.py` | `experiments/README.md` | yes |
| `reports/feat_redzone_results.csv`, `feat_redzone_control.csv` | `experiments/feature_families/feat_redzone_eval.py`, `feat_redzone_control.py` | `experiments/README.md` | yes |
| `reports/fifty_plus_rates.csv`, `fifty_plus_variance.csv`, `fifty_plus_lift.csv` | `experiments/fifty_plus/fifty_plus.py` | `experiments/fifty_plus/README.md` | yes |
| `legacy/reports/model_summary.csv`, `season_forward_results.csv`, `feature_importance.csv` | the retired v1 harness, now `legacy/evaluate_v1.py` | nothing: kept as the record of what v1 measured (7 folds; 2021 was missing from the data then) | yes |
| `reports/decision_metrics.csv`, `predictions.parquet` | `evaluate.py` | nothing | no |
| `docs/images/*.png` | `figures.py` | the README | yes |
| `docs/images/fantasy-football-04.webp` | no script here: the author's graphic, added in aebf54f, with the 2026-09-29 run's numbers | nothing: no longer shown in the README | no |
