# How the repo is organised

The README says what the model does and how well it does it, and `METHOD.md`
makes the argument. This page is the map: what each directory holds, which
module writes which file, and which module reads it. Column definitions are in
[`data_dictionary.md`](data_dictionary.md).

## Directories

| path | what is there |
|---|---|
| `src/kicker_analysis/` | the pipeline, one module per stage (below). Stages run as `python src/kicker_analysis/<module>.py` from the repo root. Every module opens with a docstring explaining why it is built the way it is, and most carry an assert-based `--self-check` that runs offline on synthetic data. |
| `scripts/run_pipeline.py` | runs the pipeline stages in order and stops at the first failure; `--self-check` runs every module's self-check instead |
| `tests/` | `test_self_checks.py`, a pytest wrapper that finds every module with a self-check by reading its source and runs it in a fresh interpreter. `tests/offline/sitecustomize.py` makes any network connection fail, so a check that starts downloading fails. |
| `.github/workflows/ci.yml` | runs `pytest` on Python 3.11, 3.12 and 3.13 on every push and pull request |
| `reports/` | the tracked evidence: the small CSVs behind the numbers in the README and `METHOD.md`, and the weekly boards. Parquet dumps land here too and are gitignored. |
| `experiments/` | the scripts that scored the three refuted feature families, with their saved logs and results. Not part of the pipeline. |
| `notes/` | dated decision write-ups. `model-selection-2026-09-29.md` is the bake-off behind the ridge. |
| `legacy/` | the retired R and Python pipelines and the R workflow's data, each with the reason it was retired, in `legacy/README.md`. Nothing there runs. |
| `docs/` | this page, `data_dictionary.md`, and `images/`, the README's graphics |
| `data/` | not in the repository. Created on the first run: `data/external/` caches the downloads and `data/processed/` holds the built tables. Both are gitignored, and `build_dataset.py` rebuilds them from scratch. |

`pyproject.toml` installs the package in editable mode (`pip install -e .`);
`requirements.txt` is its one dependency list and `requirements-dev.txt` adds
pytest.

## Modules

| module | job |
|---|---|
| `build_dataset.py` | nflverse and the schedule to one row per kicker per game, with that week's fantasy points as the target and a release-versus-play-by-play parity check |
| `fourth_down.py` | coach decisions on fourth down in field-goal range, per team-game: field-goal share and go rate |
| `features.py` | the lagged features, each knowable before kickoff, and the named list of columns that are not |
| `model.py` | the 16-feature list; `RidgeRanker`, which orders the board; `CompoundKickerModel`, two count stages simulated through the scoring rule for floor, ceiling and P(15+); and `load()`, which assembles the modelling frame |
| `backtest.py` | season-forward evaluation, test seasons 2018 on: exact decision lift, week-block bootstrap, calibration |
| `predict_week.py` | the weekly board: pseudo-rows for the coming games, through the same feature builder, then both models |
| `model_bakeoff.py` | nested season-forward bake-off of 14 challengers against the ridge, Holm-corrected. It writes only its own report files and never touches the model or the board. |
| `figures.py` | the README's three charts, drawn only from the tracked CSVs in `reports/`, each in a light and a dark version |
| `evaluate.py` | the earlier evaluation harness, kept for `prob_best` and `bootstrap_compare`. The headline numbers come from `backtest.py`. |
| `feat_drives.py`, `feat_redzone.py`, `feat_legstadium.py` | three candidate feature families (drive shape, red-zone efficiency, leg strength and stadium), all refuted. The model does not use them. |

## How data flows

```
nflverse kicking release --+
nflverse play-by-play -----+--> build_dataset.py --> data/processed/kicker_games.parquet --+
habitatring games.csv -----+    (downloads cached in data/external/)                       |
                                                                                            |
data/external/pbp_*.parquet --> fourth_down.py --> data/processed/fourth_down.parquet -----+
                                                                                            |
                         features.build_features: the 16 features in model.FEATURES <------+
                                     |                       |                      |
                                     v                       v                      v
                               backtest.py           model_bakeoff.py        predict_week.py
                    reports/backtest_summary.csv  reports/model_bakeoff_*  reports/week_<season>_<week>_kickers.csv
              reports/backtest_predictions.parquet ----------------------> (read for the board's published note)
                                     |                       |
                                     +----> figures.py <-----+
                                         docs/images/*.png
```

The four pipeline stages run in that order, and `scripts/run_pipeline.py` runs
them so: `build_dataset`, `fourth_down`, `backtest`, `predict_week`. Each reads
what the one before it wrote. `predict_week.py` refuses to build a board when
the training history stops before the previous week, and it reads
`backtest_predictions.parquet` to recompute the edge it quotes, so it needs a
backtest run first.

| file | written by | read by | tracked |
|---|---|---|---|
| `data/external/kicking.parquet`, `pbp_<season>.parquet`, `games.csv` | `build_dataset.py`, as a download cache; `games.csv` and the live season's play-by-play are re-fetched after six hours | play-by-play: `fourth_down.py` and the `feat_*` builders. `games.csv`: `model.load`, `predict_week.py` (which refreshes it first), `feat_legstadium.py` | no |
| `data/processed/kicker_games.parquet` | `build_dataset.py` | `model.load` (so `backtest.py` and `model_bakeoff.py`), `predict_week.py`, `evaluate.py`, `feat_legstadium.py` | no |
| `data/processed/fourth_down.parquet` | `fourth_down.py` | `model.load`, `predict_week.py`, `evaluate.py` | no |
| `data/processed/drives.parquet`, `redzone.parquet`, `legstadium.parquet` | `feat_drives.py`, `feat_redzone.py`, `feat_legstadium.py` | the scripts in `experiments/` | no |
| `reports/backtest_summary.csv` | `backtest.py` | `figures.py`; the README's headline table | yes |
| `reports/backtest_predictions.parquet` | `backtest.py` | `predict_week.py` | no |
| `reports/model_bakeoff_*.csv` | `model_bakeoff.py` | `figures.py`; `notes/model-selection-2026-09-29.md` | yes |
| `reports/week_<season>_<week>_kickers.csv` | `predict_week.py`, which also writes the payload the author's Fantasy Football site loads | nothing: these are the published boards | yes |
| `reports/season_forward_results.csv`, `model_summary.csv`, `feature_importance.csv` | the retired v1 harness, now `legacy/evaluate_v1.py` | nothing: kept as the record of what v1 measured | yes |
| `reports/decision_metrics.csv`, `predictions.parquet` | `evaluate.py` | nothing | no |
| `docs/images/*.png` | `figures.py` | the README | yes |

One gap in that chain, as the code stands: `fourth_down.py` and the `feat_*`
builders read whichever play-by-play seasons are in `data/external/`, but
`build_dataset.py` downloads play-by-play only for the seasons it rebuilds
(2021, 2024 for the parity check, and every season after 2024). On a fresh
clone `fourth_down.parquet` therefore covers 2021 and 2024 onward rather than
2015 onward, and `prev_go_rate` and `t_fg_share_l8` fall back to the training
median for the other seasons.
