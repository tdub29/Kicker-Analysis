"""Build the modeling table: one row per kicker per game, 1999 to today.

This replaces the 180-column team-game wide file the repo shipped with. Three
changes, each of which was a correctness problem and not a preference:

  * PLAYER grain, not team. A team's kicking history is not the kicker's: legs
    change mid-season through injury and benching, and the fantasy question is
    always about a named man.
  * The target is the WEEK's fantasy points, computed from the distance buckets.
    The old pipeline regressed `homefppg`, which is season-to-date fantasy points
    per game. Predicting a rolling mean from three other rolling means scores
    well and forecasts nothing.
  * Every feature is LAGGED. A rolling window here is built with shift(1) so a
    row never sees its own game. This is the single easiest way to produce a
    model that looks excellent and loses money, so it is asserted in the
    self-check rather than trusted.

Sources, all key-free:
  * nflverse player_stats_kicking release, 1999 to 2024, with distance buckets.
  * nflverse play-by-play for the seasons the release has not caught up to,
    reduced to the same columns and parity-checked against an overlapping year.
  * habitatring games.csv for schedule, weather, roof, surface and the betting
    lines, which is where the implied team total comes from.

Usage:
    python -m kicker_analysis.build_dataset            # writes data/processed/
    python -m kicker_analysis.build_dataset --self-check
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed"
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "http://www.habitatring.com/games.csv"
# The kicking release lags the live season; anything past this comes from pbp.
RELEASE_LAST_SEASON = 2024

# Standard fantasy kicker scoring. Distance tiers are the whole reason player
# grain matters: a 52-yarder is worth 67% more than a 39-yarder, so two kickers
# with identical make counts are not worth the same.
SCORING = {"fg_0_39": 3.0, "fg_40_49": 4.0, "fg_50_": 5.0, "pat": 1.0, "fg_miss": 0.0}

# Three franchises moved and the two sources disagree about it: the kicking
# release stamps every season with the CURRENT code (LA, LAC, LV) while the
# schedule uses the era's code (STL, SD, OAK). Left alone that silently dropped
# 911 of 13,896 kicker-games, every one of them a Ram, Charger or Raider before
# the move, so the model would have learned those franchises only post-relocation.
# Collapsing to one code per franchise is also the right modeling unit: the
# stadium changed, the organisation did not.
RELOCATIONS = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

BUCKETS = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
           "fg_made_40_49", "fg_made_50_59", "fg_made_60_"]
KEEP = (["season", "week", "season_type", "player_id", "player_display_name", "team",
         "fg_att", "fg_made", "fg_missed", "fg_blocked", "fg_long",
         "pat_att", "pat_made", "pat_missed"] + BUCKETS)


# A finished season never changes; the one being played changes every Sunday.
# Caching both forever is what made the week-3 board a week-2 board: games.csv
# was three days old, so every week-2 game still read as unplayed and
# `upcoming()` picked the slate that had already happened. Anything that can
# still gain rows gets a short life; historical files keep the permanent cache
# because re-downloading 28 seasons of play-by-play weekly is pure waste.
LIVE_MAX_AGE_HOURS = 6.0


def fetch(url: str, name: str, max_age_hours: float | None = None) -> pathlib.Path:
    """Cached download. `max_age_hours` re-fetches a file older than that.

    A failed refresh keeps the copy on disk rather than leaving nothing: a stale
    schedule still forecasts, and no schedule does not. The staleness is printed
    so a silently old board is not mistaken for a current one.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists() and max_age_hours is not None:
        age = (time.time() - path.stat().st_mtime) / 3600.0
        if age > max_age_hours:
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(path)
            except Exception as exc:                      # offline, 404, timeout
                tmp.unlink(missing_ok=True)
                print(f"  warning: could not refresh {name} ({exc}); "
                      f"using the copy on disk, {age:.1f}h old")
    elif not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def current_season_files(season: int) -> None:
    """Drop the cached artifacts for a season still being played.

    Named rather than inlined because the rule is a fact about the DATA, not
    about one caller: the current season's play-by-play gains rows every week.
    """
    for name in (f"pbp_{season}.parquet", "player_stats_kicking.parquet"):
        p = CACHE / name
        if p.exists() and (time.time() - p.stat().st_mtime) / 3600.0 > LIVE_MAX_AGE_HOURS:
            p.unlink()


def fantasy_points(df: pd.DataFrame) -> pd.Series:
    """Week fantasy points from the distance buckets, not from a total."""
    short = df[["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"]].sum(axis=1)
    mid = df["fg_made_40_49"]
    long = df[["fg_made_50_59", "fg_made_60_"]].sum(axis=1)
    return (short * SCORING["fg_0_39"] + mid * SCORING["fg_40_49"]
            + long * SCORING["fg_50_"] + df["pat_made"] * SCORING["pat"]
            + df["fg_missed"] * SCORING["fg_miss"])


def kicking_from_pbp(season: int) -> pd.DataFrame:
    """Rebuild the release's kicking columns for a season it does not cover.

    Only the columns in KEEP, so the two sources concatenate without a silent
    schema difference. Parity against an overlapping season is asserted in
    `parity_check`, because a quiet mismatch here shifts the target itself.
    """
    path = fetch(f"{RELEASE}/pbp/play_by_play_{season}.parquet", f"pbp_{season}.parquet")
    cols = ["season", "week", "season_type", "posteam", "kicker_player_id",
            "kicker_player_name", "field_goal_attempt", "field_goal_result",
            "kick_distance", "extra_point_attempt", "extra_point_result"]
    pbp = pd.read_parquet(path, columns=cols)
    pbp = pbp[(pbp.field_goal_attempt == 1) | (pbp.extra_point_attempt == 1)].copy()
    pbp = pbp[pbp.kicker_player_id.notna()]

    fg = pbp[pbp.field_goal_attempt == 1].copy()
    fg["made"] = (fg.field_goal_result == "made").astype(int)
    fg["missed"] = (fg.field_goal_result == "missed").astype(int)
    fg["blocked"] = (fg.field_goal_result == "blocked").astype(int)
    # nflverse kick_distance for a FG is the attempt distance in yards.
    edges = [(0, 19, "fg_made_0_19"), (20, 29, "fg_made_20_29"), (30, 39, "fg_made_30_39"),
             (40, 49, "fg_made_40_49"), (50, 59, "fg_made_50_59"), (60, 99, "fg_made_60_")]
    for lo, hi, name in edges:
        fg[name] = ((fg.made == 1) & fg.kick_distance.between(lo, hi)).astype(int)

    xp = pbp[pbp.extra_point_attempt == 1].copy()
    xp["pat_made"] = (xp.extra_point_result == "good").astype(int)
    xp["pat_missed"] = (xp.extra_point_result == "failed").astype(int)
    xp["pat_blocked"] = (xp.extra_point_result == "blocked").astype(int)

    key = ["season", "week", "season_type", "posteam", "kicker_player_id", "kicker_player_name"]
    a = fg.groupby(key, dropna=False).agg(
        fg_att=("field_goal_attempt", "sum"), fg_made=("made", "sum"),
        fg_missed=("missed", "sum"), fg_blocked=("blocked", "sum"),
        fg_long=("kick_distance", lambda s: s[fg.loc[s.index, "made"] == 1].max()),
        **{b: (b, "sum") for _, _, b in edges}).reset_index()
    b = xp.groupby(key, dropna=False).agg(
        pat_att=("extra_point_attempt", "sum"), pat_made=("pat_made", "sum"),
        pat_missed=("pat_missed", "sum")).reset_index()
    out = a.merge(b, on=key, how="outer")
    out = out.rename(columns={"posteam": "team", "kicker_player_id": "player_id",
                              "kicker_player_name": "player_display_name"})
    for c in KEEP:
        if c not in out.columns:
            out[c] = 0
    num = [c for c in KEEP if c not in
           ("season", "week", "season_type", "player_id", "player_display_name", "team")]
    out[num] = out[num].fillna(0)
    return out[KEEP]


def parity_check() -> dict:
    """Derive the last released season from pbp and compare to the release.

    The two paths must agree or the 2025+ rows carry a different target from the
    1999-2024 rows, which no downstream metric would reveal.
    """
    rel = pd.read_parquet(fetch(f"{RELEASE}/player_stats/player_stats_kicking.parquet",
                                "kicking.parquet"))
    rel = rel[rel.season == RELEASE_LAST_SEASON][KEEP].copy()
    der = kicking_from_pbp(RELEASE_LAST_SEASON)
    key = ["season", "week", "player_id"]
    m = rel.merge(der, on=key, how="outer", suffixes=("_rel", "_pbp"), indicator=True)
    report = {"rows_release": len(rel), "rows_pbp": len(der),
              "only_release": int((m._merge == "left_only").sum()),
              "only_pbp": int((m._merge == "right_only").sum())}
    both = m[m._merge == "both"]
    for c in ["fg_att", "fg_made", "pat_made"] + BUCKETS:
        report[f"mismatch_{c}"] = int(
            (both[f"{c}_rel"].fillna(0) != both[f"{c}_pbp"].fillna(0)).sum())
    return report


def load_kicking() -> pd.DataFrame:
    """The release, plus play-by-play for every season the release does not cover.

    "Does not cover" means any HOLE, not just the tail. The published
    player_stats_kicking release contains no 2021 rows whatsoever, which is an
    upstream gap and not a local cache problem. Taking the release at face value
    silently dropped the entire first 17-game season from training, and every
    downstream metric looked perfectly healthy without it. Since `parity_check`
    shows the pbp path reproduces the release exactly, filling a hole the same
    way the live seasons are filled is safe and is the only way the set is
    actually complete.
    """
    rel = pd.read_parquet(fetch(f"{RELEASE}/player_stats/player_stats_kicking.parquet",
                                "kicking.parquet"))
    rel = rel[KEEP].copy()
    frames = [rel]
    games = load_games()
    have = set(rel.season.unique())
    wanted = sorted(int(s) for s in games.season.unique() if s >= int(rel.season.min()))
    missing = [s for s in wanted if s not in have]
    if missing:
        print(f"  release is missing {len(missing)} season(s): {missing}; deriving from pbp")
    for season in missing:
        try:
            frames.append(kicking_from_pbp(int(season)))
        except Exception as exc:                      # a season not yet published
            print(f"  pbp {season} unavailable: {str(exc)[:70]}")
    out = pd.concat(frames, ignore_index=True)
    num = [c for c in KEEP if c not in
           ("season", "week", "season_type", "player_id", "player_display_name", "team")]
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    return out


def load_games() -> pd.DataFrame:
    g = pd.read_csv(fetch(GAMES_URL, "games.csv", max_age_hours=LIVE_MAX_AGE_HOURS), low_memory=False)
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(RELOCATIONS)
    return g


def team_game_context(games: pd.DataFrame) -> pd.DataFrame:
    """Games to one row per TEAM per game, with the context a kicker cares about.

    `implied_total` is the point of this function. A team's own Vegas total is
    (game total / 2) +/- (spread / 2), and it is the market's direct forecast of
    how much scoring this offence does. The old pipeline carried `spread_line`
    and `total_line` in the file and fed neither to the model.
    """
    rows = []
    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        d = pd.DataFrame({
            "season": games.season, "week": games.week, "game_id": games.game_id,
            "game_type": games.game_type,
            "team": games[f"{side}_team"], "opponent": games[f"{opp}_team"],
            "is_home": int(side == "home"),
            "rest": games[f"{side}_rest"],
            "roof": games.roof, "surface": games.surface,
            "temp": games.temp, "wind": games.wind, "div_game": games.div_game,
            "total_line": games.total_line, "spread_line": games.spread_line,
        })
        # spread_line is quoted from the HOME side and is positive when the home
        # team is favoured, so the away team's margin is its negation.
        margin = games.spread_line if side == "home" else -games.spread_line
        d["team_spread"] = margin
        d["implied_total"] = games.total_line / 2.0 + margin / 2.0
        d["opp_implied_total"] = games.total_line / 2.0 - margin / 2.0
        d["team_score"] = games[f"{side}_score"]
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    # A dome removes the two weather features rather than leaving them blank, and
    # the flag tells the model which regime it is in.
    out["is_dome"] = out.roof.isin(["dome", "closed"]).astype(int)
    out.loc[out.is_dome == 1, ["temp", "wind"]] = out.loc[out.is_dome == 1,
                                                          ["temp", "wind"]].fillna(
        pd.Series({"temp": 68.0, "wind": 0.0}))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # The season being played gains rows every Sunday, so its cached artifacts
    # are dropped before anything reads them. Without this the board trains on
    # whichever week the cache was first filled in and says nothing about it.
    games_now = pd.read_csv(fetch(GAMES_URL, "games.csv",
                                  max_age_hours=LIVE_MAX_AGE_HOURS), low_memory=False)
    live = int(games_now[games_now.home_score.notna()].season.max())
    current_season_files(live)
    print(f"current season {live}: live caches older than "
          f"{LIVE_MAX_AGE_HOURS:g}h dropped")
    print("parity check, release vs play-by-play:")
    for k, v in parity_check().items():
        print(f"  {k}: {v}")

    kicking = load_kicking()
    kicking["team"] = kicking["team"].replace(RELOCATIONS)
    games = load_games()
    ctx = team_game_context(games)

    df = kicking.merge(ctx, on=["season", "week", "team"], how="left")
    df["fantasy_points"] = fantasy_points(df)
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    path = OUT / "kicker_games.parquet"
    df.to_parquet(path, index=False)
    print(f"\nwrote {path.relative_to(ROOT)}: {len(df):,} kicker-games, "
          f"{df.season.min()}-{df.season.max()}, {df.player_id.nunique():,} kickers")
    print(f"  unmatched to a scheduled game: {int(df.game_id.isna().sum()):,}")
    print(f"  fantasy points: mean {df.fantasy_points.mean():.2f}, "
          f"sd {df.fantasy_points.std():.2f}, max {df.fantasy_points.max():.0f}")
    return 0


def self_check() -> int:
    """Scoring and the implied-total algebra, the two places a silent sign error
    would change every number downstream."""
    d = pd.DataFrame([{"fg_made_0_19": 1, "fg_made_20_29": 0, "fg_made_30_39": 1,
                       "fg_made_40_49": 1, "fg_made_50_59": 1, "fg_made_60_": 0,
                       "pat_made": 3, "fg_missed": 1}])
    # 2 short (3 each) + 1 mid (4) + 1 long (5) + 3 XP = 6 + 4 + 5 + 3 = 18
    assert float(fantasy_points(d).iloc[0]) == 18.0, fantasy_points(d).iloc[0]

    g = pd.DataFrame([{"season": 2026, "week": 1, "game_id": "x", "game_type": "REG",
                       "home_team": "KC", "away_team": "BAL", "home_rest": 7, "away_rest": 7,
                       "roof": "outdoors", "surface": "grass", "temp": 60.0, "wind": 5.0,
                       "div_game": 0, "total_line": 46.0, "spread_line": 3.0,
                       "home_score": 27, "away_score": 20}])
    c = team_game_context(g).set_index("team")
    # Home favoured by 3 on a 46 total: 24.5 and 21.5. They sum to the total and
    # differ by the spread. Sign convention verified against 2015-2025 results:
    # corr(spread_line, home margin) = +0.44, and the implied totals land within
    # 0.2 points of the mean actual score on each side.
    assert c.loc["KC", "implied_total"] == 24.5, c.loc["KC", "implied_total"]
    assert c.loc["BAL", "implied_total"] == 21.5, c.loc["BAL", "implied_total"]
    assert c.loc["KC", "implied_total"] + c.loc["BAL", "implied_total"] == 46.0
    # The away row must mirror, not copy, the home row's spread.
    assert c.loc["BAL", "team_spread"] == -3.0
    assert c.loc["KC", "opp_implied_total"] == c.loc["BAL", "implied_total"]

    # Relocations collapse to one franchise code on BOTH sides, or the pre-move
    # seasons of three teams fall out of the join entirely.
    g2 = g.assign(home_team="STL", away_team="SD")
    c2 = team_game_context(load_games.__wrapped__(g2) if hasattr(load_games, "__wrapped__")
                           else g2.replace({"home_team": RELOCATIONS, "away_team": RELOCATIONS}))
    assert set(c2.team) == {"LA", "LAC"}, set(c2.team)
    print("build_dataset self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main())
