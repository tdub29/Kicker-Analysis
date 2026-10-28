# Data dictionary

Every column the current pipeline produces, read off the code that produces it.
In the order they are built:

1. `data/processed/kicker_games.parquet`, written by `build_dataset.py`: the modelling table.
2. The 16 model features, built in memory by `features.build_features` and named in `model.FEATURES`.
3. `reports/week_<season>_<week>_kickers.csv`, written by `predict_week.py`: the weekly board.
4. The evidence CSVs `backtest.py` writes to `reports/`.
5. `reports/live_2026_scorecard.csv` and `live_2026_scorecard_weeks.csv`, written by `score_boards.py`: the published boards scored.

`data/processed/fourth_down.parquet` sits between the first two and is listed
with the features it feeds. The bake-off's `reports/model_bakeoff_*.csv` are
described where they are used, in `notes/model-selection-2026-09-29.md`; the
experiment CSVs (`feature_family_ridge_retest.csv`, `feat_redzone_*.csv`,
`fifty_plus_*.csv`) in `experiments/README.md` and
`experiments/fifty_plus/README.md`. The retired R workflow's home/away table
(`homefppg`, `homecash` and the rest) is described in `legacy/README.md`, next
to the file itself.

## 1. `kicker_games.parquet`

**Grain.** One row per kicker per team-week in which he attempted at least one
field goal or extra point. A kicker with no attempt that week has no row; two
kickers on one team in one week (an injury, an emergency kicker) are two rows.
Regular season and postseason are both kept here. The model keeps regular-season
rows from 2015 on (`model.load`).

**Sources**, all key-free, fetched over HTTPS and cached in `data/external/`.
Each download is atomic and leaves a `<name>.fetch.json` record beside it, and
`reports/data_manifest.json` lists every cached input behind the tracked tables
(name, url, sha256, bytes, fetched_utc, last_modified):

| source | cached as | contributes |
|---|---|---|
| nflverse `player_stats_kicking` release | `kicking.parquet` | the kicking columns for every season it covers. As downloaded on 2026-10-28 that is 1999 to 2024, with no 2021 rows at all, an upstream gap. Re-downloaded when older than six hours. |
| nflverse play-by-play | `pbp_<season>.parquet` | the same kicking columns, rebuilt for every season the release lacks or that is still being played (on 2026-10-28: 2021, 2025 and 2026), and for the parity season, the latest finished season the release carries (`parity_season`, 2024 on 2026-10-28), where `parity_check` compares the rebuild to the release; `build_dataset.py` stops on any difference. `fourth_down.py` fetches every season from 2015 to the current one. A copy fetched before its season finished (1 March of the following year) is re-downloaded when older than six hours. |
| nflverse/nfldata `games.csv` (https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) | `games.csv` | schedule, venue, weather and betting lines. Re-downloaded when older than six hours, because the current season changes every week. |

The build also stops if any season from 1999 to the current one is missing, or
has kicker rows for fewer than 90% of the team-games the schedule says were
played (`MIN_TEAM_GAME_SHARE`; a season still being played is held to all but
its latest week).

**Target.** `fantasy_points`, from `build_dataset.SCORING` and `fantasy_points()`:

```
fantasy_points = 3 x (FG made, 0-39 yards) + 4 x (40-49) + 5 x (50+) + 1 x (PAT made)
```

Missed field goals and missed extra points score 0, not a penalty. It is
computed from the distance buckets rather than from `fg_made`, so two kickers
with the same number of makes are not worth the same.

| column | meaning |
|---|---|
| `season` | NFL season |
| `week` | week of the season, as the source numbers it |
| `season_type` | `REG` or `POST`, from the kicking source |
| `player_id` | nflverse player id, e.g. `00-0032726`. Join on this, not on the name. |
| `player_display_name` | Not one format. Seasons from the release carry its full name (`David Akers`); seasons rebuilt from play-by-play carry its abbreviated `kicker_player_name` (`K.Fairbairn`). The code does not reconcile the two. |
| `team` | the kicker's team. `STL`, `SD` and `OAK` are collapsed to `LA`, `LAC` and `LV` (`RELOCATIONS`), because the two sources disagree about era codes and the mismatch silently dropped 911 kicker-games. |
| `fg_att` | field-goal attempts, whether made, missed or blocked |
| `fg_made` | field goals made |
| `fg_missed` | field goals missed, not counting blocks |
| `fg_blocked` | field goals blocked |
| `fg_long` | longest field goal MADE, in yards; 0 when none was made (`load_kicking` fills nulls with 0) |
| `pat_att` | extra-point attempts, including blocked ones (there is no blocked-PAT column) |
| `pat_made` | extra points made |
| `pat_missed` | extra points missed, not counting blocks |
| `fg_made_0_19`, `fg_made_20_29`, `fg_made_30_39`, `fg_made_40_49`, `fg_made_50_59`, `fg_made_60_` | field goals made by attempt distance in yards; they sum to `fg_made` |
| `game_id` | schedule id, `<season>_<week>_<away>_<home>`, e.g. `2026_07_NYG_HOU` |
| `game_type` | the schedule's game type: `REG`, `WC`, `DIV`, `CON` or `SB` |
| `opponent` | the other team, relocations collapsed the same way |
| `is_home` | 1 if this team is the schedule's home team, which at a neutral site is the designated one |
| `rest` | this team's rest before the game, from games.csv `home_rest` / `away_rest` |
| `roof` | games.csv roof: `dome`, `outdoors`, `open` or `closed`. `open` and `closed` are the game-day setting of a retractable roof, and an unplayed game has none yet. |
| `surface` | games.csv playing surface, as published |
| `temp`, `wind` | weather observed on the day, from games.csv. At a `dome` or `closed` roof a missing value is filled with 68 and 0. Not knowable before kickoff, so banned from the model. |
| `div_game` | 1 for a divisional game |
| `total_line` | the game's betting total (over/under) |
| `spread_line` | the betting spread, quoted from the home side, positive when the home team is favoured |
| `team_spread` | the spread from this team's side: `spread_line` for the home team, its negation for the away team. Positive means this team is favoured by that many points. |
| `implied_total` | `total_line / 2 + team_spread / 2`, the market's forecast of this team's points |
| `opp_implied_total` | `total_line / 2 - team_spread / 2`, the same for the opponent |
| `team_score` | this team's final score. An outcome, so banned from the model. |
| `is_dome` | 1 if `roof` is `dome` or `closed`. Banned: `closed` is decided on game day. The model uses `is_indoor_certain` and `is_retractable` instead. |
| `fantasy_points` | the target, above |

The game-context columns (`game_id` through `is_dome`) are joined from the
schedule on season, week and team, and are null for a kicking row that matches
no scheduled game. `build_dataset.py` prints that count on every build.

## `fourth_down.parquet`

Written by `fourth_down.py` from every play-by-play season from 2015 to the
current one, each downloaded into `data/external/` if absent; the build stops if
a season is missing or has rows for fewer than 70% of its played team-games.
One row per team-game with at least one fourth down that counts as a decision:
at or inside the opponent's 38-yard line, win probability between 5% and 95%,
more than two minutes left in the half, and exactly one of a field-goal
attempt, a punt, or a run or pass. Team-games with none have no row. The code
calls the 38 a 55-yard attempt (38 + 7 + 10). In the play-by-play,
`kick_distance` is `yardline_100 + 18` for 89.7% of regular-season field-goal
attempts from 2015 to 2026 (`experiments/fifty_plus/fifty_plus.log`), so a
kick from the 38 is typically a 56-yarder.

| column | meaning |
|---|---|
| `season`, `week`, `season_type`, `team` | keys, relocations collapsed |
| `fourth_in_range` | fourth downs that count, as above |
| `fourth_fg`, `fourth_go`, `fourth_punt` | how many of them were field-goal attempts, runs or passes, and punts |
| `fourth_mean_togo` | mean yards to go on them |
| `fg_share` | `fourth_fg / fourth_in_range` |
| `go_rate` | `fourth_go / fourth_in_range`. Punts stay in the denominator, because a punt from the 37 also declined the field goal, so the two rates do not sum to 1. |

## 2. The 16 model features

`model.FEATURES` is an inclusion list: these 16 and nothing else go into both
models. `features.build_features` computes many more lagged columns (4-game,
8-game and career windows of each quantity); only these are used.

How every lagged feature is built:

- **The frame.** `kicker_games` joined to `fourth_down` on season, week,
  season type and team, kept to regular-season games from 2015 on. So "career"
  below means regular-season games since 2015, and a window can run across a
  season boundary.
- **The lag.** `shift(1)` is applied to the raw per-game value first and the
  window second, so a row never sees its own game, including a kicker's first
  games. Windows are means with `min_periods=1`: with three earlier games an
  8-game window averages those three, and with none it is null.
- **Team grain.** `t_*` columns are aggregated to one row per team-game before
  lagging (attempts and fantasy points summed across that game's kickers;
  game-level columns such as the score and the fourth-down rates taken once), so a second kicker in the same game cannot become his
  teammate's previous game.
- **Kicker grain.** `k_*` columns lag over the kicker's own games, whichever
  team he kicked for.
- **Missing values.** A null (no history yet) is filled with the training
  median inside each model's pipeline (`SimpleImputer`).
- **The ban list.** `features.SERVE_TIME_UNAVAILABLE` (`temp`, `wind`,
  `wind_outdoor`, `is_dome`, `t_games_prior`, `opp_d_games_prior`) can never
  enter, and `model.py`'s self-check asserts none of them is in `FEATURES`.

| feature | definition | why it is known before kickoff |
|---|---|---|
| `implied_total` | `total_line / 2 + team_spread / 2` | a betting line. Training uses nflverse closing lines, so the backtest describes a lineup set close to kickoff; the board uses the line in games.csv when it was built. |
| `opp_implied_total` | `total_line / 2 - team_spread / 2` | same |
| `total_line` | the game's betting total | same |
| `closeness` | `-abs(team_spread)`: 0 for a pick'em, more negative for a bigger mismatch | same |
| `t_fg_att_l8` | team field-goal attempts per game, mean over its previous 8 games | earlier games only |
| `t_pat_att_l8` | team extra-point attempts per game, previous 8 games | earlier games only |
| `t_team_score_l8` | team points scored per game, previous 8 games | earlier games only |
| `prev_go_rate` | the team's `go_rate` over the whole previous season: fourth-down goes over fourth downs in range, summed across its team-games. Null when the previous season has no data; the team-season grid is made dense before shifting, so a missing year does not hand over the year before it. | a finished season |
| `t_fg_share_l8` | mean of the per-game `fg_share` over the team's previous 8 games. A game with no fourth down in range has no value and is skipped by the mean. | earlier games only |
| `k_fg_pct_career` | the kicker's field goals made over field goals attempted, across all his earlier games in the frame | earlier games only |
| `k_fg_att_l8` | his field-goal attempts per game, previous 8 games | earlier games only |
| `k_fantasy_points_l8` | his fantasy points per game, previous 8 games | earlier games only |
| `is_indoor_certain` | 1 if the venue is a fixed dome. Read from the game's own `roof`; an unplayed game has no `roof` yet, so its type comes from a `stadium_id` table built from played games, plus two neutral sites named by hand (`STADIUM_TYPE_KNOWN`). A venue with no history is scored as outdoors. | a property of the building |
| `is_retractable` | 1 if the venue has a retractable roof (`open` or `closed` on a played game). The day's setting is thrown away because it is decided on game day. | a property of the building |
| `climate_wind_exposed` | `venue_wind_climate x (1 - is_indoor_certain)`. `venue_wind_climate` is the mean observed wind at that venue in that calendar month over PRIOR seasons only (a mean of per-season means; the venue is `stadium_id`, else the stadium name, else the home team), and 0 when the venue has no earlier season in that month. | earlier seasons only |
| `is_home` | 1 if the kicker's team is the schedule's home team | the schedule |

## 3. The weekly board, `reports/week_<season>_<week>_kickers.csv`

One row per kicker per game on the forecast slate (by default the next week's
games not yet played), sorted by `rank_points`. Each kicker's feature row is a pseudo-row with every outcome left
null, built by the same `build_features` call as the training rows, so its
windows can only fill from games already played.

The board mixes the two models. `exp_points` comes from `RidgeRanker` and orders
the board; every distribution column comes from `CompoundKickerModel`'s 4,000
simulated outcomes (`N_SIMS`, seed 0). `predict_week.run` overwrites the count
model's own mean with the ridge forecast, so `exp_points` is not the mean of the
distribution in the columns beside it.

| column | meaning |
|---|---|
| `player_id`, `player_display_name`, `team` | the kicker the board assigns to the team: the last man who kicked for it this season before the board's week (`current_kickers`), minus any kicker Sleeper's player list (cached as `data/external/players-nfl.json`, refreshed daily) puts on a different team or on none (`drop_released`); the roster check is applied to the upcoming slate only, not to a rebuilt past week. Before the season's first game, the last man who kicked for the team in an earlier season. |
| `opponent`, `is_home` | from the schedule, as in `kicker_games` |
| `implied_total`, `total_line`, `team_spread` | the line in games.csv when the board was built; definitions as in `kicker_games` |
| `is_indoor_certain` | as the feature above |
| `game_id` | schedule id |
| `exp_points` | the ridge point forecast, clipped at 0. `rank_points` is ordered by it. |
| `median_points` | median of the simulated outcomes |
| `floor_p20`, `ceiling_p90` | 20th and 90th percentiles of the simulated outcomes |
| `p_ge_8`, `p_ge_10`, `p_ge_12`, `p_ge_15` | share of simulated outcomes at or above 8, 10, 12 and 15 points; multiples of 1/4000 |
| `exp_fg_att`, `exp_pat_att` | the count model's expected field-goal and extra-point attempts: the means of its two count stages, which the simulation draws around |
| `last_seen_week` | the most recent week this season, before the board's week, that the kicker kicked for this team. Empty on a week-1 board. |
| `stale_weeks` | `week - last_seen_week`. 1 means he kicked last week; a bye also reads 2. `predict_week.py` prints every kicker above 1, because the roster read may be out of date. |
| `season`, `week` | the forecast slate |
| `rank_points` | 1 for the highest `exp_points` |
| `rank_ceiling` | rank by `p_ge_15`, highest first; tied kickers share the better rank |

## 4. The evidence CSVs `backtest.py` writes

The test rows are the 4,211 regular-season kicker-games of the eight
season-forward folds, 2018 to 2025 (`backtest.FIRST_TEST`, `LAST_TEST`); each
fold is fit on every season before it. "Week-block" intervals resample whole
weeks, 10,000 draws, seed 0. Lifts are exact expectations over every pair (or
triple) of kickers in a week, not simulations.

**`backtest_summary.csv`**, one row per ranking (`rank by`): the ridge that
orders the board, the count model's expected points (`compound E[y]`) and its
P(y >= 15), the kicker's career mean, the implied total alone, and the training
median (`constant 8.0`).

| column | meaning |
|---|---|
| `rho` | within-week Spearman of the ranking against actual points, averaged over weeks (weeks with fewer than five rows skipped) |
| `mae` | mean absolute error, filled only for `compound E[y]` |
| `lift_r2`, `lift_r3` | points per week gained by starting the ranking's favourite of 2 (or 3) kickers drawn from the week, over starting a random one of them |
| `share_r2`, `share_r3` | that lift as a share of the oracle's, who always starts the higher scorer |
| `ceiling_r2` | how often the R=2 pick scores 15 or more |

**`component_skill.csv`**, one row per target x forecast. Targets: `fg_att`,
`pat_att`, `fg_points`, `pat_points`, `points`. Forecasts: `count_stage` (the
compound model: its two count stages for the attempt targets, and for the
points targets its expected points, split into PAT points, which are its
expected PAT attempts, and the rest), `ridge16` (`RidgeRanker` refit with that
target), `train_mean` (the training seasons' mean, the reference).

| column | meaning |
|---|---|
| `oos_r2` | 1 - SSE / SSE of the training-season mean, so 0 is no better than that mean |
| `r2_lo`, `r2_hi` | week-block 95% interval for `oos_r2` |
| `within_week_spearman` | as `rho` above |
| `n_rows` | test rows |

**`variance_decomposition.csv`**, the variance of weekly points split exactly
into its two halves, over every modelling row from 2015 to 2025: `component`
(`fg_points`, `pat_points`, `two_cov` = twice their covariance, `total`),
`variance` (sample variance) and `share_of_total`.

**`season_level_tests.csv`**, one row per `comparison` (ridge vs implied total,
ridge vs career average, ridge vs compound E[y], compound E[y] vs implied
total). Each test season is one observation: the season's mean weekly
difference in R=2 lift.

| column | meaning |
|---|---|
| `mean_diff` | mean of the eight season differences, seasons weighted equally |
| `t_stat`, `df`, `t_p`, `t_ci_lo`, `t_ci_hi` | one-sample t-test on those differences (df = seasons - 1) and its 95% interval |
| `wilcoxon_p` | Wilcoxon signed-rank test on the same differences |
| `seasons_won`, `n_seasons`, `sign_p` | seasons the first ranking was ahead, out of how many, and the two-sided sign test |
| `cluster_boot_lo`, `cluster_boot_hi`, `cluster_boot_p` | percentile bootstrap resampling whole seasons. Shown for comparison only: with eight clusters it is too narrow, and it is not the test to quote. |

**`calibration_summary.csv`**, one row per `threshold` (8, 10, 12, 15 points),
for the count model's P(y >= threshold):

| column | meaning |
|---|---|
| `pred_mean`, `realised_rate`, `ratio` | mean stated probability, share of rows that reached the threshold, and the first over the second |
| `brier`, `brier_base` | Brier score of the model, and of a constant at the training seasons' base rate for that fold |
| `brier_skill`, `brier_skill_lo`, `brier_skill_hi` | 1 - `brier` / `brier_base` with a week-block 95% interval; 0 is no better than the constant |
| `cal_slope` | slope of a logistic regression of the outcome on logit(p); 1 is ideal, below 1 means the stated chances are too spread out |
| `auc` | probability that a row that reached the threshold was given the higher chance than one that did not; 0.5 is a coin flip |
| `n` | test rows |

**`calibration_reliability.csv`**, ten equal-count bins of the stated
probability per `threshold`, lowest first: `bin`, `n`, `pred_mean`,
`realised_rate`, and `realised_lo` / `realised_hi`, a Wilson 95% interval that
treats rows as independent and so runs somewhat narrow.

`backtest.py` also writes `backtest_predictions.parquet`, every test row with
every ranking's score, which is not tracked.

## 5. The live scorecard, `reports/live_2026_scorecard*.csv`

Written by `score_boards.py`, which scores each published 2026 board against
actual points once its games are final, and records whether each game's board
existed before kickoff. A board's bytes get two clocks: `committed`, the
earliest commit holding exactly those bytes (git dates, which the committer's
machine sets), and `public`, the earliest push to GitHub whose head commit holds
them (GitHub's activity log). A game counts as a forecast on a clock only if
that clock is strictly before its kickoff. Weeks 3 and 4 are on disk as their
first-committed, pre-kickoff versions.

**`live_2026_scorecard.csv`**, one row per `subset` and `week`, plus a pooled row
per subset:

| column | meaning |
|---|---|
| `subset` | `all` board rows, rows `prospective_commit` (committed before kickoff), or rows `prospective_public` (publicly pushed before kickoff) |
| `week` | the week, or `pooled` |
| `n_weeks` | weeks with scored rows |
| `board_rows` | board rows in the subset |
| `not_final`, `did_not_kick` | rows left out: the game is not final yet, or the board's kicker has no kick row that week (inactive, or active with no attempt) |
| `kickers_scored` | rows scored |
| `unlisted_kicked` | kickers who kicked in a scored game but were not on the board |
| `spearman_board`, `spearman_implied` | within-week Spearman of the board's order, and of the implied total, against points |
| `lift_board`, `lift_implied` | exact R=2 lift of the board's order and of the implied total alone, points per week over a random start |
| `oracle_lift`, `share_board`, `share_implied` | the oracle's lift, and each lift as a share of it |
| `board_minus_implied` | `lift_board - lift_implied` |
| `boot_lo`, `boot_hi`, `boot_p` | pooled rows of three or more weeks only: week-block bootstrap of `board_minus_implied` |
| `t_lo`, `t_hi`, `t_p` | pooled rows of three or more weeks only: t interval on the per-week differences (df = weeks - 1). With this few weeks this is the interval to read; the bootstrap can only resample the weeks there are. |
| `pred_p15`, `realised_p15` | mean stated P(15+) of the scored rows, and the share that scored 15 or more |

**`live_2026_scorecard_weeks.csv`**, one row per game on each board: `season`,
`week`, `board` (file name), `board_blob` (git blob id of its bytes),
`first_commit` and `committed_utc`, `public_commit` and `public_utc`,
`game_id`, `kickoff_utc`, `final`, `board_rows` and `scored_rows` (that game's
rows on the board, and how many were scored), and `prospective_commit` /
`prospective_public`. `score_boards.py` reads it back to reuse the recorded
clocks when run with `--offline` or without git history.
