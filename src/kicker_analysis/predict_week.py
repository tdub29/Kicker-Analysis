"""Rank this week's kickers and write the weekly board.

The backtest measures the method. This applies it to games that have not happened
yet, which is a different job with one hard constraint: the feature row for an
unplayed game must be built the same way a training row was, from history only.
That is done by appending a PSEUDO-ROW per kicker per upcoming game, with every
outcome column left null, then running the identical feature builder over history
plus pseudo-rows together. The lagged windows then fill from real prior games and
nothing else, because there is nothing else there to fill from.

A board for week N of season S uses only games strictly before (S, N), for the
model fit and for the lagged features alike. Built before the week starts that
holds by itself, because nothing later exists yet. Rebuilt afterwards it has to
be enforced: without the cutoff, `--week 3` run later in the season fit the
models on week 3's own results, and because the real week-3 row sat next to the
week-3 pseudo-row, the kicker lags counted his week-3 game as his previous one.
The cutoff also leaves out a Thursday result from the week being forecast, so a
board built on Wednesday and one built on Friday train on the same rows. A
rebuilt week is still a reconstruction rather than the board that was
published: games.csv carries closing lines, and Sleeper reports today's rosters,
which is why the roster filter is applied only to the upcoming slate.

The published board is a record, so it is never overwritten by default. If
reports/week_<season>_<week>_kickers.csv (or the `--out` path, when given)
exists, the run stops before training, says so and exits 0 (a configured site
payload is then written from that existing board); `--force` replaces it and
prints the path it replaced, and `--out` writes a rebuild somewhere else without
touching the record.

Two orderings are written, and only the first is the recommendation:

  rank_points     the board: ridge expected points, highest first. This is the
                  ordering the season-forward backtest scores and supports.
  rank_ceiling    the same kickers ordered by the count model's P(y >= 15), an
                  alternative view for a week where only a big score helps. The
                  backtest found it adds nothing measurable: ordered this way the
                  pick hit a 15-point week no more often than the main board did
                  (ceiling_r2 in reports/backtest_summary.csv). The claim that
                  ranking by the mean selects AGAINST the ceiling held for a
                  retired gradient-boosted model, not this one; METHOD.md retracts it.

Who kicks for whom is taken from the most recent game in the data rather than
from a depth chart, and the row carries how many weeks stale that is. A kicker
signed on Wednesday will not appear, which is stated in the output rather than
hidden. Before a season's first game each team's kicker is the last one who
kicked for it in an earlier season, so an offseason signing is missing from a
week-1 board. A kicker Sleeper's player list puts on another team, or on none,
is dropped; that list is cached in data/external/ and refreshed at most daily.

The author's Fantasy Football site reads a JavaScript payload of the board. It is
written only where `--site-out` or the KICKER_SITE_JS environment variable says,
and otherwise skipped with one printed line, so a clone on any machine runs the
same code without needing that site. KICKER_SITE_JS is used only for the
upcoming week; a rebuilt earlier week goes to the site only when `--site-out`
names a path.

    python src/kicker_analysis/predict_week.py
    python src/kicker_analysis/predict_week.py --week 3 --out /tmp/week3.csv
    python src/kicker_analysis/predict_week.py --site-out path/to/site/data/kickers.js
    python src/kicker_analysis/predict_week.py --max-days-ahead 6   # the scheduled run
    python src/kicker_analysis/predict_week.py --self-check
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.build_dataset import (  # noqa: E402
    GAMES_URL, LIVE_MAX_AGE_HOURS, RELOCATIONS, fetch)
from kicker_analysis.features import build_features  # noqa: E402
from kicker_analysis.model import CompoundKickerModel, FEATURES, RidgeRanker  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
# One producer, one path. `_v2` was a hand-copied snapshot that nothing
# regenerated, so the weekly board trained on whatever Sunday the copy was
# taken and looked current: it was 30 kicker-games short of week 2 while
# printing a week-3 header. build_dataset.py owns this file now.
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
REPORTS = ROOT / "reports"
FIRST_SEASON = 2015

# Where the site payload goes. Nothing is resolved at import: a machine-specific
# path computed at module level made the module fail to import anywhere but the
# machine it named, which took the self-check, CI and the pipeline down with it.
SITE_ENV = "KICKER_SITE_JS"

# Sleeper's full NFL player list. Sleeper asks callers to fetch it at most once a
# day, so the cached copy lives a day; a failed refresh keeps the copy on disk.
SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_CACHE = "players-nfl.json"
SLEEPER_MAX_AGE_HOURS = 24.0

# The threshold the site headlines, and the one the calibration sentence reads.
NOTE_THRESHOLD = 15
# Within this relative distance, pooled predicted and realised rates are called
# in agreement in the published note; outside it the note gives the two numbers
# without the word.
AGREE_WITHIN = 0.10


def slate_for(games: pd.DataFrame, week: int | None = None
              ) -> tuple[int, int | None, pd.DataFrame, bool]:
    """(season, week, games to forecast, is_upcoming) for the latest season.

    With no week, the next slate with no result yet, or week None when the
    regular season is over. That is also what a requested week resolves to when
    it is the upcoming one, so a game already played this week cannot reappear
    on the board. Any earlier week is rebuilt whole: every game of it gets a
    pseudo-row, as it did before kickoff. A week after the upcoming one is
    refused: its lags would be missing the upcoming week's games, and the guard
    on the history cannot see that once one game of that week has been played.

    The season is the latest one the schedule lists, not the latest with a
    result, so the week-1 board can be built once the new schedule is out.
    """
    reg = games[games.game_type == "REG"]
    season = int(reg.season.max())
    cur = reg[reg.season == season]
    open_games = cur[cur.home_score.isna()]
    nxt = int(open_games.week.min()) if len(open_games) else None
    if week is None or week == nxt:
        if nxt is None:
            return season, None, open_games, True
        return season, nxt, open_games[open_games.week == nxt].copy(), True
    if nxt is not None and week > nxt:
        raise SystemExit(f"{season} week {week} is after the upcoming week {nxt}; its "
                         f"board can be built once week {nxt} has been played")
    slate = cur[cur.week == week]
    if slate.empty:
        raise SystemExit(f"no {season} regular-season games in week {week}")
    return season, week, slate.copy(), False


def before_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Rows strictly before (season, week): the only rows a board for it may use."""
    return df[(df.season < season) | ((df.season == season) & (df.week < week))]


def schedule_behind(slate: pd.DataFrame, today: pd.Timestamp) -> bool:
    """True if every game on an open slate is dated two or more days before today.

    games.csv is then missing results it should have, and a board built from it
    would forecast games already played under a pre-kickoff name. Two days, not
    one, because gameday is an Eastern date and today is UTC: a late Monday
    kickoff is already Tuesday in UTC before it starts.
    """
    last = pd.to_datetime(slate.gameday, errors="coerce").max()
    return bool(pd.notna(last) and (today - last).days >= 2)


def history_reach(hist: pd.DataFrame, games: pd.DataFrame, season: int,
                  week: int) -> tuple[int, int, int]:
    """(season checked, last week with results in it, week it must reach).

    A board for week N needs the history through week N-1 of its season; a
    week-1 board needs the whole previous regular season, as games.csv lists it.
    """
    if week > 1:
        s, need = season, week - 1
    else:
        prev = games[(games.season == season - 1) & (games.game_type == "REG")]
        s, need = season - 1, int(prev.week.max()) if len(prev) else 0
    done = hist[(hist.season == s) & hist.fantasy_points.notna()]
    return s, int(done.week.max()) if len(done) else 0, need


def current_kickers(hist: pd.DataFrame, season: int) -> pd.DataFrame:
    """One kicker per team, the last man who actually kicked for them.

    Restricted to the current season so a departed kicker from two years ago
    cannot be resurrected onto a roster he no longer holds. Before the season's
    first game (a week-1 board) each team gets its last kicker of an earlier
    season instead, and last_seen_week is left empty: it counts weeks of this
    season, and he has none yet.
    """
    recent = hist[hist.season == season].sort_values(["team", "season", "week"])
    preseason = recent.empty
    if preseason:
        recent = hist.sort_values(["team", "season", "week"])
    last = recent.groupby("team").tail(1)
    out = last[["team", "player_id", "player_display_name", "week"]].rename(
        columns={"week": "last_seen_week"})
    if preseason:
        out["last_seen_week"] = np.nan
    return out


def sleeper_roster() -> pathlib.Path | None:
    """Sleeper's player list in the repo's download cache, or None if unavailable.

    A missing roster is not fatal: the board falls back to game history alone,
    which is how it ran before the roster check existed.
    """
    try:
        return fetch(SLEEPER_URL, SLEEPER_CACHE, max_age_hours=SLEEPER_MAX_AGE_HOURS)
    except Exception as exc:                              # offline, blocked, 5xx
        print(f"could not fetch the Sleeper player list ({str(exc)[:80]}): "
              f"roster read from game history only")
        return None


def drop_released(ks: pd.DataFrame, roster: pathlib.Path | None) -> pd.DataFrame:
    """Drop a kicker Sleeper no longer lists on that team.

    The last man who kicked keeps his team's slot until the next game, so a
    released kicker stayed on the board: Harrison Mevis ranked #2 for the Rams
    a week after they cut him. A name Sleeper does not know is kept.
    """
    if roster is None or not roster.exists():
        return ks
    try:
        raw = json.loads(roster.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"{roster.name} is not valid JSON ({exc}): roster read from game history only")
        return ks
    teams: dict[str, set] = {}
    for p in raw.get("players", raw).values():
        if isinstance(p, dict) and p.get("position") == "K":
            key = f"{(p.get('first_name') or '')[:1]}.{p.get('last_name') or ''}"
            teams.setdefault(key, set()).add({"LAR": "LA"}.get(p.get("team"), p.get("team")))
    gone = ks.apply(lambda r: r.player_display_name in teams
                    and r.team not in teams[r.player_display_name], axis=1)
    if gone.any():
        print(f"not on that roster per Sleeper, dropped: "
              f"{', '.join(ks[gone].player_display_name + ' (' + ks[gone].team + ')')}")
    return ks[~gone]


def pseudo_rows(slate: pd.DataFrame, kickers: pd.DataFrame, season: int,
                week: int, template: pd.DataFrame) -> pd.DataFrame:
    """A feature row per kicker per upcoming game, outcomes left null."""
    rows = []
    for _, g in slate.iterrows():
        for side, opp_side in (("home", "away"), ("away", "home")):
            team = RELOCATIONS.get(g[f"{side}_team"], g[f"{side}_team"])
            opp = RELOCATIONS.get(g[f"{opp_side}_team"], g[f"{opp_side}_team"])
            k = kickers[kickers.team == team]
            if k.empty:
                continue
            k = k.iloc[0]
            margin = g.spread_line if side == "home" else -g.spread_line
            rows.append({
                "season": season, "week": week, "season_type": "REG",
                "player_id": k.player_id, "player_display_name": k.player_display_name,
                "team": team, "opponent": opp, "game_id": g.game_id,
                "is_home": int(side == "home"), "rest": g.get(f"{side}_rest"),
                "roof": g.roof, "surface": g.surface, "div_game": g.div_game,
                "total_line": g.total_line, "team_spread": margin,
                "implied_total": g.total_line / 2.0 + margin / 2.0,
                "opp_implied_total": g.total_line / 2.0 - margin / 2.0,
                "last_seen_week": k.last_seen_week,
            })
    out = pd.DataFrame(rows)
    # Every column the builder expects must exist, null where it is an outcome.
    for c in template.columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[[c for c in template.columns if c in out.columns]
               + [c for c in out.columns if c not in template.columns]]


def assemble(hist: pd.DataFrame, fut: pd.DataFrame, games: pd.DataFrame | None,
             season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, pseudo-row features) for the board for (season, week).

    History is cut BEFORE the feature builder runs, not only before the fit. A
    real week-N row left in would sit next to the same kicker's week-N pseudo-row,
    and the lag could pick it up as his "previous game".
    """
    # Tag the pseudo-rows explicitly. Selecting them back by (season, week) also
    # picked up the Thursday game that had already kicked off, which is how a
    # 15-game slate produced 32 kickers and listed a man whose week was over.
    past = before_week(hist, season, week).assign(_pseudo=0)
    combined = pd.concat([past, fut.drop(columns=["last_seen_week"], errors="ignore")
                          .assign(_pseudo=1)], ignore_index=True)
    feats = build_features(combined, games=games)
    future = feats[feats._pseudo == 1].copy()
    train = feats[(feats._pseudo == 0) & feats.fantasy_points.notna()].copy()
    last = (train.season * 100 + train.week).max() if len(train) else -1
    assert last < season * 100 + week, f"training reaches {last}, board is {season} week {week}"
    return train, future


class BoardExists(FileExistsError):
    """The board for this week is already written and --force was not given."""


def write_board(out: pd.DataFrame, path: pathlib.Path, force: bool = False) -> bool:
    """Write the board; True if it replaced a file. Never replaces one unless forced.

    Not-forced writes open with mode "x", so the refusal holds even if the file
    appears between the early check in run() and this write.
    """
    existed = path.exists()
    if existed and not force:
        raise BoardExists(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if force else "x", newline="", encoding="utf-8") as f:
        out.to_csv(f, index=False)
    return existed


def site_target(flag: str | None, env: dict | None = None) -> pathlib.Path | None:
    """The site payload path: --site-out, else $KICKER_SITE_JS, else None."""
    value = flag or (os.environ if env is None else env).get(SITE_ENV) or None
    return pathlib.Path(value).expanduser() if value else None


def _shown(path: pathlib.Path) -> str:
    """A path relative to the repo when it is inside it, absolute otherwise."""
    p = pathlib.Path(os.path.abspath(path))
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def run(week_override: int | None = None, force: bool = False,
        out_path: pathlib.Path | None = None, site_out: str | None = None,
        max_days_ahead: int | None = None) -> pd.DataFrame:
    # The upcoming slate is decided from this file, so reading a cached copy
    # silently re-forecasts a week that is already over. Refresh it first.
    games = pd.read_csv(fetch(GAMES_URL, "games.csv", max_age_hours=LIVE_MAX_AGE_HOURS),
                        low_memory=False)
    for c in ("home_team", "away_team"):
        games[c] = games[c].replace(RELOCATIONS)
    season, week, slate, live = slate_for(games, week_override)
    if week is None:
        print(f"no unplayed {season} regular-season games: nothing to build")
        return pd.DataFrame()
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    if live and schedule_behind(slate, today):
        raise SystemExit(
            f"games.csv has no result for any {season} week {week} game on the slate "
            f"({', '.join(slate.game_id)}), the last dated "
            f"{pd.to_datetime(slate.gameday, errors='coerce').max().date()}: the schedule "
            f"file is behind, "
            f"and a board built from it would forecast games already played")
    # The market features are lines as they stand when the board is built, and
    # the backtest scores closing lines, so a board built a week early runs on
    # inputs the backtest never scored. A scheduled run passes a bound; by hand
    # there is none.
    first = pd.to_datetime(slate.gameday, errors="coerce").min()
    days = (first - today).days
    if live and max_days_ahead is not None and days > max_days_ahead:
        print(f"the {season} week {week} slate starts {first.date()}, {days} days out; "
              f"--max-days-ahead {max_days_ahead}, so nothing is built yet")
        return pd.DataFrame()

    path = out_path or REPORTS / f"week_{season}_{week}_kickers.csv"
    # A site path from the environment is meant for the live board; a rebuilt past
    # week goes to the site only when --site-out names it on the command line.
    site = site_target(site_out) if (live or site_out) else None
    if path.exists() and not force:
        print(f"{_shown(path)} already exists, so the {season} week {week} board is "
              f"not rebuilt over it. --force replaces it; --out PATH writes a rebuild "
              f"elsewhere.")
        board = pd.read_csv(path, dtype={"player_id": str})
        if site is not None:
            write_site(board, season, week, site)
        return board

    hist = pd.read_parquet(PROC / "kicker_games.parquet")
    fourth = pd.read_parquet(PROC / "fourth_down.parquet")
    hist = hist.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    hist = hist[(hist.season >= FIRST_SEASON) & (hist.season_type == "REG")].copy()

    # The training history must reach the week before the one being forecast.
    # It did not, and nothing said so: `_v2/kicker_games.parquet` was a frozen
    # copy no script rebuilt, so a week-3 board trained through week 1 and
    # looked identical to a correct one. This is the check that would have
    # caught it in one line instead of by reading a parquet by hand. For week 1
    # the week before is the end of the previous season.
    s, through, need = history_reach(hist, games, season, week)
    if through < need:
        raise SystemExit(
            f"training data stops at {s} week {through} but the board for {season} "
            f"week {week} needs it through {s} week {need}. Run build_dataset.py and "
            f"fourth_down.py first; a board built on stale history is "
            f"indistinguishable from a current one.")
    print(f"history complete through {s} week {through}")
    done = hist[(hist.season == season) & hist.fantasy_points.notna()]
    played_this_week = sorted(set(done.loc[done.week == week, "team"]))
    if live and played_this_week:
        print(f"already kicked off this week, excluded from the board: "
              f"{', '.join(played_this_week)}")
    if not live:
        print(f"rebuilding {season} week {week}, which is not the upcoming slate: only "
              f"games before it are used, its lines are as games.csv has them now, and "
              f"Sleeper's roster (today's) is not applied")

    past = before_week(hist, season, week)
    ks = current_kickers(past, season)
    if len(ks) and ks.last_seen_week.isna().all():
        print(f"no {season} games in the history yet: each team's kicker is the last "
              f"one who kicked for it before {season}, so an offseason signing will "
              f"not appear, and stale_weeks is left empty")
    if live:
        ks = drop_released(ks, sleeper_roster())
    fut = pseudo_rows(slate, ks, season, week, past)
    print(f"{season} week {week}: {len(slate)} games, {len(fut)} kickers")

    train, future = assemble(past, fut, games, season, week)
    last = train.sort_values(["season", "week"]).iloc[-1]
    print(f"training on {len(train):,} completed kicker-games through "
          f"{int(last.season)} week {int(last.week)}")

    # Two models, each doing the job it measurably wins. Ridge orders the board
    # (its edge over the implied total and the career average is recomputed from
    # the saved backtest for the site note, never typed in). The count model
    # supplies the distribution; how its pooled rates compare with what happened
    # is in reports/calibration_summary.csv.
    pred = CompoundKickerModel().fit(train).predict(future)
    pred["exp_points"] = RidgeRanker().fit(train).predict(future)
    out = pd.concat([future[["player_id", "player_display_name", "team", "opponent",
                             "is_home", "implied_total", "total_line", "team_spread",
                             "is_indoor_certain", "game_id"]].reset_index(drop=True),
                     pred.reset_index(drop=True)], axis=1)
    out = out.merge(fut[["player_id", "last_seen_week"]].drop_duplicates("player_id"),
                    on="player_id", how="left")
    out["stale_weeks"] = week - out.last_seen_week
    out["season"] = season
    out["week"] = week
    out = out.sort_values("exp_points", ascending=False).reset_index(drop=True)
    out["rank_points"] = np.arange(1, len(out) + 1)
    out["rank_ceiling"] = out.p_ge_15.rank(ascending=False, method="min").astype(int)

    cols = ["rank_points", "rank_ceiling", "player_display_name", "team", "opponent",
            "implied_total", "exp_points", "floor_p20", "ceiling_p90", "p_ge_15",
            "exp_fg_att", "exp_pat_att", "stale_weeks"]
    print("\n" + out[cols].head(20).round(3).to_string(index=False))

    stale = out[out.stale_weeks > 1]
    if len(stale):
        print(f"\n{len(stale)} kicker(s) last kicked more than a week ago, so the roster "
              f"read may be out of date: {', '.join(stale.player_display_name)}")

    replaced = write_board(out, path, force=force)
    print(f"\n{'--force: replaced' if replaced else 'wrote'} {_shown(path)}: "
          f"{len(out)} kickers" + ("; the board it replaced survives only in git "
                                   "history, if it was committed"
                                   if replaced and path.resolve().is_relative_to(ROOT)
                                   else ""))
    if site is not None:
        write_site(out, season, week, site)
    elif live:
        print(f"site payload not written: pass --site-out or set {SITE_ENV} to write it")
    else:
        print("site payload not written: a rebuilt week goes to the site only "
              "with --site-out")
    return out


def calibration_figure(path: pathlib.Path, threshold: int = NOTE_THRESHOLD) -> dict | None:
    """The pooled backtest calibration row at `threshold`, or None if unavailable."""
    if not path.exists():
        return None
    cal = pd.read_csv(path)
    row = cal[pd.to_numeric(cal.threshold, errors="coerce") == threshold]
    if row.empty:
        return None
    r = row.iloc[0]
    return {k: float(r[k]) for k in ("pred_mean", "realised_rate", "ratio", "auc", "n")
            if k in r and pd.notna(r[k])}


def backtest_figures() -> dict:
    """Every number the site note quotes, read from the saved backtest outputs.

    Each source is optional. A clone that has not run backtest.py has neither the
    predictions (they are not tracked) nor, possibly, the calibration table, and
    the note then leaves those sentences out rather than inventing them.
    """
    figs: dict = {}
    preds = REPORTS / "backtest_predictions.parquet"
    if preds.exists():
        from kicker_analysis.backtest import boot_lift, decision
        bt = pd.read_parquet(preds)
        figs["vs_implied"] = boot_lift(bt, "ridge", "implied")
        figs["vs_career"] = boot_lift(bt, "ridge", "career")
        figs["decision"] = decision(bt, "ridge", 2)
    else:
        print(f"{_shown(preds)} missing: the site note omits the measured edge")
    cal = calibration_figure(REPORTS / "calibration_summary.csv")
    if cal is not None:
        figs["calibration"] = cal
    else:
        print("reports/calibration_summary.csv missing or without a "
              f"{NOTE_THRESHOLD}-point row: the site note omits calibration")
    return figs


def site_method() -> str:
    return (f"A ridge on {len(FEATURES)} pre-kickoff features orders the board; two "
            f"LightGBM Poisson-objective models for FG and PAT attempts, drawn "
            f"binomially and pushed through the scoring rule by Monte Carlo, give "
            f"the distribution.")


def site_note(figs: dict) -> str:
    """The published note, built only from the figures given."""
    parts = [f"The board is ordered by a ridge on {len(FEATURES)} pre-kickoff features"]
    vv, vc = figs.get("vs_implied"), figs.get("vs_career")
    if vv and vc:
        def edge(d: float) -> str:
            return "beats" if d > 0 else "trails"
        p = f"p = {vv['p']:.3f}" if vv["p"] >= 0.001 else "p < 0.001"
        parts[0] += (f", which {edge(vv['diff'])} ranking on the Vegas implied total "
                     f"alone by {abs(vv['diff']):.3f} points a week ({p}) "
                     f"and {edge(vc['diff'])} a kicker's own career average by "
                     f"{abs(vc['diff']):.3f}")
    parts[0] += "."
    cal = figs.get("calibration")
    if cal and {"pred_mean", "realised_rate", "ratio"} <= set(cal):
        s = (f"The probabilities come from a separate count model. Pooled over the "
             f"season-forward backtest, its P({NOTE_THRESHOLD}+) averaged "
             f"{cal['pred_mean'] * 100:.1f}% against a realised "
             f"{cal['realised_rate'] * 100:.1f}%")
        s += (", so predicted and realised rates agree in aggregate"
              if abs(cal["ratio"] - 1) <= AGREE_WITHIN else "")
        if "auc" in cal:
            s += (f"; it separates one kicker's big week from another's only weakly "
                  f"(AUC {cal['auc']:.2f}, where 0.5 is a coin flip)")
        parts.append(s + ".")
    else:
        parts.append("The probabilities come from a separate count model.")
    dec = figs.get("decision")
    if dec:
        parts.append(f"Expect a small edge: the oracle who always starts the better of "
                     f"two kickers gains {dec['oracle_lift']:.2f} points a week and this "
                     f"captures {dec['share'] * 100:.1f}% of that.")
    return " ".join(parts)


def write_site(df: pd.DataFrame, season: int, week: int, path: pathlib.Path) -> None:
    """Emit the file the Fantasy Football site loads, as window.KICKER_MODEL."""
    # Headline claims, recomputed from the saved backtest on every run so the
    # published note cannot drift from reports/.
    payload = {
        "season": season, "week": week,
        "method": site_method(),
        "scoring": "FG 0-39 = 3, 40-49 = 4, 50+ = 5, PAT = 1, miss = 0",
        "features": FEATURES,
        "note": site_note(backtest_figures()),
        "kickers": [
            {
                "id": str(r.player_id), "name": r.player_display_name,
                "team": r.team, "opp": r.opponent, "home": int(r.is_home),
                "rankPoints": int(r.rank_points), "rankCeiling": int(r.rank_ceiling),
                "exp": round(float(r.exp_points), 2),
                "floor": round(float(r.floor_p20), 1),
                "ceil": round(float(r.ceiling_p90), 1),
                "pGe15": round(float(r.p_ge_15), 4),
                "pGe10": round(float(r.p_ge_10), 4),
                "impliedTotal": None if pd.isna(r.implied_total) else round(float(r.implied_total), 1),
                "expFgAtt": round(float(r.exp_fg_att), 2),
                "expPatAtt": round(float(r.exp_pat_att), 2),
                "indoor": int(r.is_indoor_certain),
                "staleWeeks": None if pd.isna(r.stale_weeks) else int(r.stale_weeks),
            }
            for r in df.itertuples()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("window.KICKER_MODEL = "
                    + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"\nwrote {path}: {len(payload['kickers'])} kickers, "
          f"{path.stat().st_size / 1024:.0f} KB")


def _import_is_inert() -> list:
    """Re-import this module under an audit hook; return any I/O it did.

    Run in-process, so the dependencies are already loaded and only this
    module's own top level executes. Reading its source or bytecode is the
    import itself and is ignored; anything else is a side effect.
    """
    import importlib
    seen, on = [], [True]
    watched = {"open", "socket.connect", "socket.getaddrinfo", "urllib.Request",
               "os.mkdir", "os.remove", "os.rename", "os.replace", "os.listdir",
               "os.scandir"}

    def hook(event, args):
        if not on[0] or event not in watched:
            return
        if event == "open" and str(args[0]).endswith((".py", ".pyc")):
            return
        if event in ("os.listdir", "os.scandir") and pathlib.Path(
                str(args[0])).resolve() == pathlib.Path(__file__).resolve().parent:
            return                                    # the import system's finder
        seen.append((event, str(args[0])[:120]))

    name = "kicker_analysis.predict_week"
    sys.addaudithook(hook)
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    prev = sys.modules.pop(name, None)
    try:
        importlib.import_module(name)
    finally:
        on[0] = False
        sys.dont_write_bytecode = dont_write
        sys.modules.pop(name, None)
        if prev is not None:
            sys.modules[name] = prev
    return seen


def self_check() -> int:
    """Pseudo-rows, slate selection, the history guard, the training cutoff, the
    no-overwrite rule, the site note and an inert import. All offline, all in
    memory or a temporary directory."""
    import tempfile

    # 1. Pseudo-row construction, where a sign error would invert every ranking.
    tmpl = pd.DataFrame(columns=["season", "week", "season_type", "player_id",
                                 "player_display_name", "team", "opponent", "game_id",
                                 "fantasy_points", "fg_att"])
    slate = pd.DataFrame([{"game_id": "g1", "home_team": "KC", "away_team": "BAL",
                           "spread_line": 3.0, "total_line": 46.0, "roof": "outdoors",
                           "surface": "grass", "div_game": 0, "home_rest": 7,
                           "away_rest": 7}])
    ks = pd.DataFrame([{"team": "KC", "player_id": "k1", "player_display_name": "A.One",
                        "last_seen_week": 2},
                       {"team": "BAL", "player_id": "k2", "player_display_name": "B.Two",
                        "last_seen_week": 2}])
    rows = pseudo_rows(slate, ks, 2026, 3, tmpl).set_index("team")
    assert len(rows) == 2
    # Home favoured by 3 on a 46 total: 24.5 and 21.5, summing to the total.
    assert rows.loc["KC", "implied_total"] == 24.5, rows.loc["KC", "implied_total"]
    assert rows.loc["BAL", "implied_total"] == 21.5
    assert rows.loc["BAL", "team_spread"] == -3.0
    assert rows.loc["KC", "is_home"] == 1 and rows.loc["BAL", "is_home"] == 0
    # Outcomes must be absent, or a pseudo-row could train on itself.
    assert pd.isna(rows.loc["KC", "fantasy_points"])
    assert pd.isna(rows.loc["KC", "fg_att"])
    # A team with no known kicker is skipped rather than given a placeholder.
    rows2 = pseudo_rows(slate, ks[ks.team == "KC"], 2026, 3, tmpl)
    assert len(rows2) == 1 and rows2.iloc[0].team == "KC"

    # Which games a board covers. Week 3: Thursday played, Sunday not yet.
    sched = pd.DataFrame([
        {"season": 2026, "week": 2, "game_type": "REG", "game_id": "a", "home_score": 20.0},
        {"season": 2026, "week": 3, "game_type": "REG", "game_id": "thu", "home_score": 17.0},
        {"season": 2026, "week": 3, "game_type": "REG", "game_id": "sun", "home_score": np.nan},
        {"season": 2026, "week": 4, "game_type": "REG", "game_id": "b", "home_score": np.nan}])
    s, w, sl, live = slate_for(sched)
    assert (s, w, live, sl.game_id.tolist()) == (2026, 3, True, ["sun"])
    # Asking for the upcoming week by number must not bring Thursday back.
    assert slate_for(sched, 3)[2].game_id.tolist() == ["sun"]
    s, w, sl, live = slate_for(sched, 2)
    assert (w, live, sl.game_id.tolist()) == (2, False, ["a"])
    # A week after the upcoming one cannot be built yet: Thursday of week 3 is
    # in, so a history check on weeks alone would let week 4 through.
    try:
        slate_for(sched, 4)
    except SystemExit:
        pass
    else:
        raise AssertionError("a week after the upcoming one was built")
    # Season over: nothing to build. New schedule published: its week 1.
    over = sched.assign(home_score=1.0)
    assert slate_for(over)[1] is None
    nxt = pd.concat([over, pd.DataFrame([{"season": 2027, "week": 1, "game_type": "REG",
                                          "game_id": "c", "home_score": np.nan}])])
    assert slate_for(nxt)[:2] == (2027, 1)
    # A slate whose games are all dated two or more days ago means games.csv
    # is missing results; one dated yesterday may be a late kickoff in UTC.
    today = pd.Timestamp("2026-10-28")
    assert schedule_behind(pd.DataFrame({"gameday": ["2026-10-26"]}), today)
    assert not schedule_behind(pd.DataFrame({"gameday": ["2026-10-27"]}), today)
    assert not schedule_behind(pd.DataFrame({"gameday": ["2026-10-25", "2026-10-29"]}), today)

    # 2. The training cutoff. Real rows exist AT and AFTER the target week, as
    # they do when an old week is rebuilt; none may reach the fit or the lags.
    hist = []
    for season, week, pts in [(2025, 17, 9.0), (2025, 18, 7.0), (2026, 1, 6.0),
                              (2026, 2, 8.0), (2026, 3, 30.0), (2026, 4, 25.0),
                              (2027, 1, 20.0)]:
        for team, opp, pid, sign in (("KC", "BAL", "k1", 1.0), ("BAL", "KC", "k2", -1.0)):
            hist.append({"season": season, "week": week, "season_type": "REG",
                         "player_id": pid, "player_display_name": pid, "team": team,
                         "opponent": opp, "game_id": f"{season}_{week}", "roof": "outdoors",
                         "fantasy_points": pts, "fg_att": pts / 4, "fg_made": pts / 5,
                         "pat_att": 2.0, "fg_long": 40.0, "team_score": 3 * pts,
                         "team_spread": 3.0 * sign, "total_line": 46.0,
                         "implied_total": 23.0 + 1.5 * sign})
    hist = pd.DataFrame(hist)
    cut = before_week(hist, 2026, 3)
    assert set(zip(cut.season, cut.week)) == {(2025, 17), (2025, 18), (2026, 1), (2026, 2)}
    assert set(zip(before_week(hist, 2026, 1).season, before_week(hist, 2026, 1).week)) \
        == {(2025, 17), (2025, 18)}
    fut = pseudo_rows(slate, ks, 2026, 3, hist)
    train, future = assemble(hist, fut, None, 2026, 3)
    assert set(zip(train.season, train.week)) == {(2025, 17), (2025, 18), (2026, 1), (2026, 2)}, \
        sorted(set(zip(train.season, train.week)))
    assert len(future) == 2 and future._pseudo.eq(1).all()
    k1 = future.set_index("player_id").loc["k1"]
    # Last four games before week 3: 9, 7, 6, 8. Week 3's 30 must not be one of them.
    assert k1.k_fantasy_points_l4 == 7.5, k1.k_fantasy_points_l4
    assert k1.k_games_prior == 4, k1.k_games_prior
    assert k1.t_team_score_l8 == 3 * 7.5, k1.t_team_score_l8

    # The history guard: week N needs week N-1; week 1 needs all of last season.
    g2 = pd.DataFrame({"season": [2025] * 3 + [2026] * 2, "week": [16, 17, 18, 1, 2],
                       "game_type": "REG"})
    assert history_reach(hist, g2, 2026, 3) == (2026, 4, 2)
    assert history_reach(hist, g2, 2026, 1) == (2025, 18, 18)
    s, through, need = history_reach(before_week(hist, 2025, 18), g2, 2026, 1)
    assert through < need, "a week-1 board passed on a previous season cut short"
    # Before the season's first game: last season's kicker, last_seen_week empty
    # rather than a negative stale_weeks.
    pre = current_kickers(before_week(hist, 2026, 1), 2026).set_index("team")
    assert pre.loc["KC", "player_id"] == "k1" and pre.last_seen_week.isna().all()
    assert current_kickers(cut, 2026).set_index("team").loc["KC", "last_seen_week"] == 2

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        # 3. The no-overwrite rule.
        p = tmp / "reports" / "week_2026_3_kickers.csv"
        first, second = pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})
        assert write_board(first, p) is False
        before = p.read_bytes()
        try:
            write_board(second, p)
        except BoardExists:
            pass
        else:
            raise AssertionError("an existing board was overwritten without --force")
        assert p.read_bytes() == before, "a refused write changed the board"
        assert write_board(second, p, force=True) is True
        assert pd.read_csv(p).a.tolist() == [2]

        # 4. Sleeper roster: a kicker listed on another team, or on none, is
        # dropped; a name Sleeper does not know is kept; no file changes nothing.
        roster = tmp / SLEEPER_CACHE
        roster.write_text(json.dumps({
            "1": {"position": "K", "first_name": "Ann", "last_name": "One", "team": "BUF"},
            "2": {"position": "K", "first_name": "Ben", "last_name": "Three", "team": None},
            "3": {"position": "QB", "first_name": "Bob", "last_name": "Two", "team": "SF"},
        }), encoding="utf-8")
        ks3 = pd.concat([ks, pd.DataFrame([{"team": "SF", "player_id": "k3",
                                            "player_display_name": "B.Three",
                                            "last_seen_week": 2}])], ignore_index=True)
        kept = drop_released(ks3, roster)
        assert kept.player_display_name.tolist() == ["B.Two"], kept.player_display_name.tolist()
        assert len(drop_released(ks3, None)) == 3
        assert len(drop_released(ks3, tmp / "absent.json")) == 3

        # 5. The site note: numbers come only from the files given, and a missing
        # file drops its sentence instead of breaking the run.
        cal_path = tmp / "calibration_summary.csv"
        pd.DataFrame({"threshold": [10, 15], "pred_mean": [0.37, 0.0838],
                      "realised_rate": [0.36, 0.0841], "ratio": [1.03, 0.996],
                      "brier_skill": [0.01, 0.001], "auc": [0.56, 0.55],
                      "n": [100, 100]}).to_csv(cal_path, index=False)
        cal = calibration_figure(cal_path)
        assert cal["pred_mean"] == 0.0838 and cal["auc"] == 0.55, cal
        assert calibration_figure(tmp / "absent.csv") is None
        note = site_note({"calibration": cal,
                          "vs_implied": {"diff": 0.08, "p": 0.001},
                          "vs_career": {"diff": 0.25, "p": 0.0},
                          "decision": {"oracle_lift": 2.5, "share": 0.17}})
        assert "8.4% against a realised 8.4%" in note and "agree in aggregate" in note, note
        assert "AUC 0.55" in note and "beats ranking on the Vegas" in note, note
        assert "(p = 0.001)" in note, note
        assert "trails" in site_note({"vs_implied": {"diff": -0.01, "p": 0.0},
                                      "vs_career": {"diff": 0.2, "p": 0.0}})
        assert "within" not in note, "the note may not claim a calibration tolerance"
        off = site_note({"calibration": {**cal, "ratio": 1.25}})
        assert "agree" not in off, off
        bare = site_note({})
        assert "calibration" not in bare.lower() and "%" not in bare, bare
        assert "ridge" in site_method() and "Poisson" in site_method()

        # 6. Site path: a flag wins, the environment is next, and neither means
        # no site file at all.
        win = r"C:\Users\someone\site\data\kickers.js"
        assert site_target(None, {}) is None
        assert site_target(None, {SITE_ENV: ""}) is None
        assert site_target(None, {SITE_ENV: win}) == pathlib.Path(win)
        assert site_target(str(tmp / "x.js"), {SITE_ENV: win}) == tmp / "x.js"

    # 7. Importing the module touches neither the filesystem nor the network.
    io = _import_is_inert()
    assert not io, f"import-time side effects: {io}"

    print("predict_week self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the weekly kicker board.")
    ap.add_argument("--self-check", action="store_true",
                    help="run the offline checks and exit")
    ap.add_argument("--week", type=int, default=None,
                    help="week of the latest season; an earlier week is rebuilt "
                         "from games before it only")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing board CSV (it is a record; the "
                         "default refuses)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write the board here instead of "
                         "reports/week_<season>_<week>_kickers.csv (an "
                         "existing file there is kept unless --force)")
    ap.add_argument("--site-out", default=None,
                    help=f"write the site payload here (default: ${SITE_ENV}, "
                         f"used for the upcoming week only; neither set means "
                         f"no site payload)")
    ap.add_argument("--max-days-ahead", type=int, default=None,
                    help="build nothing if the upcoming slate starts more than this "
                         "many days from today (UTC); for scheduled runs")
    a = ap.parse_args()
    if a.self_check:
        sys.exit(self_check())
    if a.week is not None and a.week < 1:
        ap.error("--week must be 1 or more")
    run(a.week, force=a.force, out_path=a.out, site_out=a.site_out,
        max_days_ahead=a.max_days_ahead)
    sys.exit(0)
