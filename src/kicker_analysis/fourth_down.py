"""Coach aggression on fourth down in field-goal range, per team-game.

A kicker's fantasy week is mostly volume, and volume is not his decision. When an
offence reaches fourth down inside the 40, the head coach chooses between a field
goal and going for it, and coaches differ enormously and persistently. An
aggressive staff taxes its own kicker every week. Nothing in the original model
saw this at all.

What counts as a decision, and why each filter is here:

  in range      `yardline_100` <= 38. In nflverse play-by-play a field goal's
                `kick_distance` is its `yardline_100` plus 18 for most attempts
                (the 10-yard end zone and about eight yards of snap and hold;
                experiments/fifty_plus/README.md has the split), so the edge of
                range is typically a 56-yard attempt. Beyond that the field goal
                is not really on the menu, so including those plays measures leg
                strength, not coaching.
  competitive   win probability between 5% and 95%. A four-score game is not a
                choice, and garbage time would otherwise read as aggression.
  not a scramble for the clock
                the last two minutes of either half are excluded. Down eight with
                40 seconds left, going for it is arithmetic rather than
                philosophy, and end-of-half plays are where the freak decisions
                live.
  a real fourth down
                punts stay IN the denominator. A coach who punts from the 37
                also declined the field goal, and dropping punts would make a
                conservative staff look neutral.

Two rates come out, and they are not complements of each other because punts
occupy the difference:

  fg_share      field goal attempts / fourth downs in range. The direct driver of
                the kicker's volume.
  go_rate       run or pass attempts / fourth downs in range. The aggression
                number people quote.

Both are then LAGGED by the caller, because a team's rate in the game you are
predicting is not knowable on Tuesday.

Coverage. Every season from FIRST_SEASON to the current one is requested by
number through build_dataset.pbp_file (same cache, downloaded if absent), and the
table is refused unless each season has a plausible number of team-games.
Building from whatever play-by-play happened to be cached gave a fresh clone
four seasons out of twelve, the ones build_dataset.py needs for itself, and
the two coach features were then median-imputed for every other season without
an error anywhere.

Usage:
    python src/kicker_analysis/fourth_down.py            # writes the table
    python src/kicker_analysis/fourth_down.py --self-check
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.build_dataset import (  # noqa: E402
    coverage_problems, live_season, load_games, pbp_file, write_manifest)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "fourth_down.parquet"

# pbp only carries win probability and clean play typing from 2015 on, and 2015
# is also when the extra point moved back, so it is the natural floor for both.
FIRST_SEASON = 2015
IN_RANGE_YARDLINE = 38     # 38 + 18 = a 56-yard attempt for most kicks
WP_LO, WP_HI = 0.05, 0.95
MIN_HALF_SECONDS = 120

# Team-games with at least one fourth down in range, per team-game played. A
# team-game with none has no row, so even a complete season sits well below 1,
# and a missing season reads 0. On the 2026-10-28 build, removing the first
# three weeks of a finished season put it under 0.70 for every season but 2022
# and 2025, which needed four, and one or two missing weeks passed (checked
# directly against build_dataset.coverage_problems; no committed script runs
# it). This guards against a lost season or a lost month of play-by-play, not a
# lost week.
MIN_TEAM_GAME_SHARE = 0.70

COLS = ["season", "week", "season_type", "game_id", "posteam", "down", "ydstogo",
        "yardline_100", "play_type", "field_goal_attempt", "punt_attempt",
        "rush_attempt", "pass_attempt", "wp", "half_seconds_remaining",
        "score_differential", "field_goal_result", "kick_distance"]

# Same franchise collapse the rest of the pipeline uses.
RELOCATIONS = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def fourth_down_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """The subset of plays that represent a real fourth-down decision in range."""
    d = pbp[(pbp.down == 4) & pbp.posteam.notna()].copy()
    d = d[d.yardline_100 <= IN_RANGE_YARDLINE]
    d = d[d.wp.between(WP_LO, WP_HI)]
    d = d[d.half_seconds_remaining > MIN_HALF_SECONDS]
    # A play must be one of the three real options. Penalties, timeouts and
    # aborted snaps are not decisions and would dilute both rates.
    d["is_fg"] = (d.field_goal_attempt == 1).astype(int)
    d["is_punt"] = (d.punt_attempt == 1).astype(int)
    d["is_go"] = ((d.rush_attempt == 1) | (d.pass_attempt == 1)).astype(int)
    d = d[(d.is_fg + d.is_punt + d.is_go) == 1]
    d["posteam"] = d.posteam.replace(RELOCATIONS)
    return d


def team_game_rates(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: the fourth-down decisions and how they went."""
    d = fourth_down_plays(pbp)
    key = ["season", "week", "season_type", "posteam"]
    out = d.groupby(key, dropna=False).agg(
        fourth_in_range=("is_fg", "size"),
        fourth_fg=("is_fg", "sum"),
        fourth_go=("is_go", "sum"),
        fourth_punt=("is_punt", "sum"),
        # Mean distance to the sticks on those fourth downs. A team that keeps
        # facing 4th and 2 gets a different menu from one facing 4th and 9, so
        # this separates philosophy from situation.
        fourth_mean_togo=("ydstogo", "mean"),
    ).reset_index()
    out = out.rename(columns={"posteam": "team"})
    n = out.fourth_in_range.replace(0, np.nan)
    out["fg_share"] = out.fourth_fg / n
    out["go_rate"] = out.fourth_go / n
    return out


def build(first_season: int = FIRST_SEASON, last_season: int | None = None,
          games: pd.DataFrame | None = None, season_file=pbp_file) -> pd.DataFrame:
    """Team-game rates for every season from `first_season` to the current one.

    `season_file(season)` returns the play-by-play parquet for one season; the
    default fetches it into the shared cache when absent. Raises SystemExit
    naming any season that is missing or thin against the schedule.
    """
    games = load_games() if games is None else games
    last = live_season(games) if last_season is None else last_season
    if last < first_season:
        raise SystemExit(f"no seasons to build: {first_season} to {last}")
    frames, failed = [], {}
    for season in range(first_season, last + 1):
        try:
            frames.append(team_game_rates(pd.read_parquet(season_file(season), columns=COLS)))
        except Exception as exc:                      # offline, 404, unreadable
            failed[season] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if failed:
        raise SystemExit("could not read play-by-play for:\n"
                         + "\n".join(f"  {s}: {e}" for s, e in failed.items())
                         + f"\nevery season from {first_season} to {last} is required; "
                           "nothing was written")
    df = pd.concat(frames, ignore_index=True).sort_values(
        ["season", "week", "team"]).reset_index(drop=True)
    problems = coverage_problems(df, games, first_season, last, MIN_TEAM_GAME_SHARE)
    if problems:
        raise SystemExit("fourth-down table is missing or incomplete for:\n  "
                         + "\n  ".join(problems) + "\nnothing was written")
    return df


def main() -> int:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} team-games, "
          f"{df.season.min()}-{df.season.max()}")
    per = df.groupby("season").size()
    print(f"  team-games per season: {per.min():,} ({per.idxmin()}) "
          f"to {per.max():,} ({per.idxmax()})")
    print(f"  fourth downs in range per team-game: mean "
          f"{df.fourth_in_range.mean():.2f}, max {df.fourth_in_range.max()}")
    print(f"  field goal share: {df.fg_share.mean():.3f}  "
          f"go rate: {df.go_rate.mean():.3f}  punt: {1 - df.fg_share.mean() - df.go_rate.mean():.3f}")
    # The league trend is the sanity check: everyone knows fourth-down aggression
    # has risen, so if this table does not show it, the filters are wrong.
    yr = df.groupby("season").agg(go=("go_rate", "mean"), fg=("fg_share", "mean"))
    print("\n  season  go_rate  fg_share")
    for s, r in yr.iterrows():
        print(f"    {s}   {r.go:.3f}    {r.fg:.3f}")
    m = write_manifest()
    print(f"\nwrote {m.relative_to(ROOT)}")
    return 0


def self_check() -> int:
    """The filters and the two rates, on a hand-built set of fourth downs."""
    base = dict(season=2024, week=1, season_type="REG", game_id="g", posteam="KC",
                down=4, ydstogo=3, yardline_100=30, wp=0.5,
                half_seconds_remaining=600, score_differential=0,
                play_type="field_goal", field_goal_attempt=0, punt_attempt=0,
                rush_attempt=0, pass_attempt=0, field_goal_result=None,
                kick_distance=None)
    rows = [
        {**base, "field_goal_attempt": 1},                       # kept, a FG
        {**base, "rush_attempt": 1},                             # kept, a go
        {**base, "punt_attempt": 1, "yardline_100": 37},         # kept, a punt
        {**base, "field_goal_attempt": 1, "yardline_100": 55},   # dropped, out of range
        {**base, "rush_attempt": 1, "wp": 0.01},                 # dropped, blowout
        {**base, "rush_attempt": 1, "half_seconds_remaining": 40},  # dropped, clock
        {**base, "down": 3, "field_goal_attempt": 1},            # dropped, not 4th
        {**base},                                                # dropped, no option set
        {**base, "field_goal_attempt": 1, "rush_attempt": 1},    # dropped, ambiguous
    ]
    d = pd.DataFrame(rows)
    plays = fourth_down_plays(d)
    assert len(plays) == 3, len(plays)

    r = team_game_rates(d).iloc[0]
    assert r.fourth_in_range == 3, r.fourth_in_range
    assert r.fourth_fg == 1 and r.fourth_go == 1 and r.fourth_punt == 1
    # A punt is a declined field goal, so it stays in the denominator: the two
    # rates are 1/3 each and do NOT sum to one.
    assert abs(r.fg_share - 1 / 3) < 1e-9, r.fg_share
    assert abs(r.go_rate - 1 / 3) < 1e-9, r.go_rate
    assert abs(r.fg_share + r.go_rate - 1.0) > 0.3, "punts must not vanish"

    # Relocated franchises collapse here too, or a team's own history splits in
    # two halfway through the file.
    d2 = pd.DataFrame([{**base, "posteam": "OAK", "field_goal_attempt": 1}])
    assert team_game_rates(d2).iloc[0].team == "LV"

    # build() asks for every season by number, not for whatever is cached, and
    # refuses a season that comes back empty. Three settled seasons of one game,
    # each team with one field goal decision; files live in a temporary directory.
    sched = pd.DataFrame({"season": [2015, 2016, 2017], "week": 1,
                          "home_team": "KC", "away_team": "BAL", "home_score": 20})
    with tempfile.TemporaryDirectory() as tmp:
        requested = []

        def season_file(season: int, empty: tuple = ()) -> pathlib.Path:
            requested.append(season)
            plays = [] if season in empty else [
                {**base, "season": season, "posteam": t, "field_goal_attempt": 1}
                for t in ("KC", "BAL")]
            path = pathlib.Path(tmp) / f"pbp_{season}.parquet"
            pd.DataFrame(plays, columns=COLS).to_parquet(path)
            return path

        df = build(2015, 2017, sched, season_file)
        assert requested == [2015, 2016, 2017], requested
        assert len(df) == 6 and sorted(df.season.unique()) == [2015, 2016, 2017]
        try:
            build(2015, 2017, sched, lambda s: season_file(s, empty=(2016,)))
            raise AssertionError("a season with no team-games must stop the build")
        except SystemExit as exc:
            assert "2016: 0 team-games" in str(exc), exc

        # A season that cannot be fetched is named, not skipped.
        def unreachable(season: int) -> pathlib.Path:
            if season == 2017:
                raise OSError("offline")
            return season_file(season)
        try:
            build(2015, 2017, sched, unreachable)
            raise AssertionError("a season that cannot be fetched must stop the build")
        except SystemExit as exc:
            assert "2017: OSError: offline" in str(exc), exc
    print("fourth_down self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main())
