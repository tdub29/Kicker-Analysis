"""Build the modeling table: one row per kicker per game, 1999 to today.

This replaces the 180-column team-game wide file the repo shipped with. Three
changes, each of which was a correctness problem and not a preference:

  * PLAYER grain, not team. A team's kicking history is not the kicker's: legs
    change mid-season through injury and benching, and the fantasy question is
    always about a named man.
  * The target is ONE kicker's fantasy points for the week, computed from the
    distance buckets. The old pipeline's target, `homefppg`, was the week's
    points summed over a team's kickers, and nothing in that pipeline was ever
    scored on a holdout.
  * Every feature is LAGGED. This table puts each game's outcome next to its
    pre-game context, and features.py builds every rolling window from it with
    shift(1), so a row never sees its own game. Leaking the game into its own
    features is the single easiest way to produce a model that looks excellent
    and loses money, so features.py asserts the lag in its self-check rather
    than trusting it.

Sources, all key-free, cached in data/external/ under the names KICKING_CACHE,
GAMES_CACHE and pbp_cache() set below:
  * nflverse player_stats_kicking release (`kicking.parquet`), 1999 on, with
    distance buckets. It lags the live season, and as downloaded in October 2026
    it has no 2021 rows at all.
  * nflverse play-by-play (`pbp_<season>.parquet`) for every season the release
    lacks or that is still being played, reduced to the same columns and
    parity-checked against the latest finished season the release does carry.
  * nflverse/nfldata games.csv, over HTTPS from GitHub (`games.csv`), for
    schedule, weather, roof, surface and the betting lines, which is where the
    implied team total comes from. It is the file nfldata's own documentation
    loads from http://www.habitatring.com/games.csv; the GitHub copy is used
    because it is served over HTTPS and resolves on networks where that host
    does not.

What a normal run refuses to get wrong, failing with a message instead:
  * the play-by-play rebuild must match the release exactly on the parity
    season, or the rebuilt seasons would carry a different target;
  * every season from 1999 to the current one must be present with kicker rows
    for most of its played team-games, because a season that silently drops out
    leaves every downstream metric looking healthy;
  * a download lands under its cache name only once it is complete, so an
    interrupted fetch cannot leave a truncated file that later reads as data.

Each run also writes reports/data_manifest.json: the sha256, size, source URL
and fetch time of every cached input. The inputs are fetched live and nflverse
rebuilds them, so the manifest is the record of which input state produced the
tracked tables.

Usage:
    python src/kicker_analysis/build_dataset.py            # writes data/processed/
    python src/kicker_analysis/build_dataset.py --self-check
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed"
MANIFEST = ROOT / "reports" / "data_manifest.json"

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
KICKING_URL = f"{RELEASE}/player_stats/player_stats_kicking.parquet"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# One name per cached input. Every reader, every refresh rule and the manifest go
# through these, so a file cannot be written under one name and expired under
# another, which is how the kicking release once escaped its refresh entirely.
KICKING_CACHE = "kicking.parquet"
GAMES_CACHE = "games.csv"


def pbp_cache(season: int) -> str:
    return f"pbp_{season}.parquet"


def pbp_url(season: int) -> str:
    return f"{RELEASE}/pbp/play_by_play_{season}.parquet"


# First season of the table, and of both kicking sources.
FIRST_SEASON = 1999

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

# A season counts as complete when it has a kicker row for at least this share of
# the team-games the schedule says were played. A team-game with no field goal
# or extra point attempt has no row, so a complete season can fall short of 1:
# the build of 2026-10-26 measured 0.969 (2000) to 1.000 (2020 and 2026) across
# 1999-2026. A lost season reads 0. On that build, removing the first two weeks
# of any finished season put it under 0.9 and removing one week did not, so this
# catches a lost season or a lost fortnight, not a single lost week.
MIN_TEAM_GAME_SHARE = 0.90


# A finished season never changes; the one being played changes every Sunday.
# Caching both forever is what made the week-3 board a week-2 board: games.csv
# was three days old, so every week-2 game still read as unplayed and
# `upcoming()` picked the slate that had already happened. Anything that can
# still gain rows gets a short life; historical files keep the permanent cache
# because re-downloading 28 seasons of play-by-play weekly is pure waste.
LIVE_MAX_AGE_HOURS = 6.0

# No NFL game is played after February: the Super Bowl falls by mid-month. A
# season's play-by-play fetched on or after 1 March of the following year is
# final; a copy fetched earlier may still be missing games or stat corrections.
SETTLED_MONTH = 3

# Seconds a single socket operation may stall before the download is abandoned.
# Without it a dead connection hangs the build forever rather than failing.
FETCH_TIMEOUT_S = 60

# The opener `fetch` downloads through. A module attribute so the self-check can
# substitute a failing one and prove an interrupted download leaves nothing.
_urlopen = urllib.request.urlopen


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_record(path: pathlib.Path) -> pathlib.Path:
    """Where `fetch` notes the URL, time and Last-Modified of a download."""
    return path.with_name(path.name + ".fetch.json")


def _umask() -> int:
    """The process umask. os.umask only reports it by replacing it, so it is
    set and immediately put back."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _publish(tmp: str, path: pathlib.Path) -> None:
    """Move a finished temporary file into place under its final name.

    mkstemp creates files readable by their owner only. Without the chmod every
    cached input and the manifest would carry that mode, unlike every other file
    the pipeline writes, so they get the permissions an ordinary open() would give.
    """
    os.chmod(tmp, 0o666 & ~_umask())
    os.replace(tmp, path)


def _write_atomic(path: pathlib.Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _publish(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def _complete(path: pathlib.Path) -> bool:
    """False for a parquet file cut short.

    A parquet file ends with the 4-byte magic `PAR1`, so a truncated one is
    recognisable without reading it. `_download` is atomic, but a cache filled
    by an older, non-atomic version of this module can still hold one.
    """
    if path.suffix != ".parquet":
        return True
    if path.stat().st_size < 8:
        return False
    with open(path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return f.read(4) == b"PAR1"


def _download(url: str, path: pathlib.Path) -> None:
    """Download `url` to `path` atomically, then record where it came from.

    The bytes go to a temporary file in the same directory and are moved into
    place with os.replace only once the body is complete and matches the
    server's Content-Length. An interrupted, stalled or short download raises and
    leaves the cache exactly as it was.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    digest, n = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as out, _urlopen(url, timeout=FETCH_TIMEOUT_S) as resp:
            expected = resp.headers.get("Content-Length")
            last_modified = resp.headers.get("Last-Modified")
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                n += len(chunk)
        if expected is not None and n != int(expected):
            raise OSError(f"{url}: received {n:,} of {int(expected):,} bytes")
        _publish(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise
    record = {"url": url, "fetched_utc": _utc(path.stat().st_mtime),
              "last_modified": last_modified, "sha256": digest.hexdigest(), "bytes": n}
    _write_atomic(_fetch_record(path), (json.dumps(record, indent=2) + "\n").encode())


def fetch(url: str, name: str, max_age_hours: float | None = None) -> pathlib.Path:
    """Cached download. `max_age_hours` re-fetches a file older than that.

    A failed refresh keeps the copy on disk rather than leaving nothing: a stale
    schedule still forecasts, and no schedule does not. The staleness is printed
    so a silently old board is not mistaken for a current one. A file that is not
    on disk at all, or is a truncated parquet, must download or the call raises.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists() and not _complete(path):
        print(f"  {name} in the cache is truncated; downloading it again")
        _download(url, path)
    elif not path.exists():
        _download(url, path)
    elif max_age_hours is not None:
        age = (time.time() - path.stat().st_mtime) / 3600.0
        if age > max_age_hours:
            try:
                _download(url, path)
            except Exception as exc:                      # offline, 404, timeout
                print(f"  warning: could not refresh {name} ({exc}); "
                      f"using the copy on disk, {age:.1f}h old")
    return path


def season_settled(season: int, when: float) -> bool:
    """True if a copy written at `when` (epoch seconds) post-dates every game of `season`."""
    return when >= datetime(season + 1, SETTLED_MONTH, 1, tzinfo=timezone.utc).timestamp()


def pbp_file(season: int) -> pathlib.Path:
    """Cached play-by-play for one season, downloaded if absent.

    A copy fetched before its season settled is provisional and is refreshed
    once older than LIVE_MAX_AGE_HOURS. The rule keys on when the copy was taken,
    not on which season is current, so a copy taken in December is still
    refreshed after the calendar moves on, and a failed refresh keeps the old
    copy instead of leaving none.
    """
    path = CACHE / pbp_cache(season)
    provisional = not (path.exists() and season_settled(season, path.stat().st_mtime))
    return fetch(pbp_url(season), pbp_cache(season),
                 max_age_hours=LIVE_MAX_AGE_HOURS if provisional else None)


def load_release() -> pd.DataFrame:
    """The kicking release. Refreshed like the live files: nflverse rebuilds it
    on its own schedule, and it is a quarter of a megabyte."""
    return pd.read_parquet(fetch(KICKING_URL, KICKING_CACHE, max_age_hours=LIVE_MAX_AGE_HOURS))


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
    schema difference. Parity against an overlapping season is enforced by
    `main`, because a quiet mismatch here shifts the target itself.
    """
    path = pbp_file(season)
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


def parity_season(rel: pd.DataFrame, now: float | None = None) -> int:
    """The latest season the release carries that is also finished.

    Derived from the release rather than written down, so the check follows the
    release when nflverse extends it, and never lands on a season still being
    played, where the two sources can legitimately differ by a night's update.
    """
    now = time.time() if now is None else now
    done = [int(s) for s in rel.season.unique() if season_settled(int(s), now)]
    if not done:
        raise SystemExit("the kicking release holds no finished season to check "
                         "the play-by-play rebuild against")
    return max(done)


def parity_check(rel: pd.DataFrame | None = None) -> dict:
    """Rebuild the parity season from pbp and compare it to the release.

    The two paths must agree or the seasons rebuilt from play-by-play carry a
    different target from the released ones, which no downstream metric would
    reveal. `main` refuses to build on any difference (`parity_failures`).
    """
    rel = load_release() if rel is None else rel
    season = parity_season(rel)
    rel = rel[rel.season == season][KEEP].copy()
    der = kicking_from_pbp(season)
    key = ["season", "week", "player_id"]
    m = rel.merge(der, on=key, how="outer", suffixes=("_rel", "_pbp"), indicator=True)
    report = {"season": season, "rows_release": len(rel), "rows_pbp": len(der),
              "only_release": int((m._merge == "left_only").sum()),
              "only_pbp": int((m._merge == "right_only").sum())}
    both = m[m._merge == "both"]
    for c in ["fg_att", "fg_made", "pat_made"] + BUCKETS:
        report[f"mismatch_{c}"] = int(
            (both[f"{c}_rel"].fillna(0) != both[f"{c}_pbp"].fillna(0)).sum())
    return report


def parity_failures(report: dict) -> list[str]:
    """The entries of a parity report that are not clean, by name."""
    bad = [k for k, v in report.items()
           if k.startswith(("only_", "mismatch_")) and v != 0]
    if report["rows_release"] == 0:
        bad.append("rows_release")
    if report["rows_release"] != report["rows_pbp"]:
        bad.append("rows_pbp")
    return bad


def live_season(games: pd.DataFrame) -> int:
    """The latest season with at least one game played."""
    played = games[games.home_score.notna()]
    if played.empty:
        raise SystemExit("the schedule has no played games; games.csv is not usable")
    return int(played.season.max())


def coverage_problems(have: pd.DataFrame, games: pd.DataFrame, first: int, last: int,
                      min_share: float, now: float | None = None) -> list[str]:
    """Seasons in [first, last] that are missing or too thin to be complete.

    `have` needs season, week and team columns. Its distinct team-weeks are
    compared with the team-games the schedule says were played. A season still
    being played is held to all but its latest played week, because the
    play-by-play and the schedule are published separately and the play-by-play
    can trail the latest scores; a lag longer than a week is still caught (and
    predict_week.py separately refuses a board whose history stops short of the
    week before the slate).
    """
    now = time.time() if now is None else now
    played = games[games.home_score.notna()]
    got = have[["season", "week", "team"]].drop_duplicates().groupby("season").size()
    problems = []
    for season in range(first, last + 1):
        g = played[played.season == season]
        if not season_settled(season, now) and len(g):
            g = g[g.week < g.week.max()]
        expected, n = 2 * len(g), int(got.get(season, 0))
        if expected and n < min_share * expected:
            problems.append(f"{season}: {n:,} team-games, under {min_share:.0%} of the "
                            f"{expected:,} the schedule says were played")
    return problems


def load_kicking(games: pd.DataFrame | None = None,
                 rel: pd.DataFrame | None = None) -> pd.DataFrame:
    """The release, plus play-by-play for every season the release does not cover.

    "Does not cover" means any HOLE, not just the tail. The published
    player_stats_kicking release contains no 2021 rows whatsoever, which is an
    upstream gap and not a local cache problem. Taking the release at face value
    silently dropped the entire first 17-game season from training, and every
    downstream metric looked perfectly healthy without it. `main` builds only
    when `parity_check` finds the pbp path reproducing the release exactly, so
    filling a hole the same way the live seasons are filled is safe and is the
    only way the set is actually complete. A season still being played also
    comes from play-by-play, which nflverse updates through the season while
    the release lags it.

    A season that cannot be rebuilt, or that comes back thin, stops the build:
    the same silent loss of 2021 would otherwise return the first time a
    download failed.
    """
    games = load_games() if games is None else games
    rel = load_release() if rel is None else rel
    last, now = live_season(games), time.time()
    have = {int(s) for s in rel.season.unique()}
    from_pbp = [s for s in range(FIRST_SEASON, last + 1)
                if s not in have or not season_settled(s, now)]
    if from_pbp:
        print(f"  play-by-play rebuild for {len(from_pbp)} season(s) the release lacks "
              f"or has not settled: {from_pbp}")
    frames = [rel[~rel.season.isin(from_pbp)][KEEP].copy()]
    failed = {}
    for season in from_pbp:
        try:
            frames.append(kicking_from_pbp(season))
        except Exception as exc:
            failed[season] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if failed:
        raise SystemExit("could not rebuild these seasons from play-by-play:\n"
                         + "\n".join(f"  {s}: {e}" for s, e in failed.items())
                         + f"\nevery season from {FIRST_SEASON} to {last} is required; "
                           "nothing was written")
    out = pd.concat(frames, ignore_index=True)
    num = [c for c in KEEP if c not in
           ("season", "week", "season_type", "player_id", "player_display_name", "team")]
    out[num] = out[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    problems = coverage_problems(out, games, FIRST_SEASON, last, MIN_TEAM_GAME_SHARE, now)
    if problems:
        raise SystemExit("kicker data is missing or incomplete for:\n  "
                         + "\n  ".join(problems) + "\nnothing was written")
    return out


def normalise_games(g: pd.DataFrame) -> pd.DataFrame:
    """One franchise code per team on both sides of the schedule."""
    g = g.copy()
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(RELOCATIONS)
    return g


def load_games() -> pd.DataFrame:
    return normalise_games(pd.read_csv(
        fetch(GAMES_URL, GAMES_CACHE, max_age_hours=LIVE_MAX_AGE_HOURS), low_memory=False))


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


def _input_url(name: str) -> str | None:
    """The source URL of a cache file, or None if the name is not a pipeline input."""
    if name == GAMES_CACHE:
        return GAMES_URL
    if name == KICKING_CACHE:
        return KICKING_URL
    m = re.fullmatch(r"pbp_(\d{4})\.parquet", name)
    return pbp_url(int(m.group(1))) if m else None


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


MANIFEST_ABOUT = (
    "Inputs cached in data/external/ when the tracked tables were last built. "
    "fetched_utc is when the bytes were downloaded: from the fetch record, or the "
    "file's modification time for a file placed in the cache by hand. "
    "last_modified is the server's Last-Modified header, null when none was recorded.")


def manifest_entries(cache: pathlib.Path | None = None) -> list[dict]:
    """One entry per cached input: name, url, sha256, bytes, fetched_utc, last_modified.

    The fetch record is trusted only when its sha256 matches the file, so a file
    replaced by hand is reported by what is actually on disk.
    """
    cache = CACHE if cache is None else cache
    entries = []
    for path in sorted(cache.iterdir()) if cache.exists() else []:
        url = _input_url(path.name)
        if url is None or not path.is_file():
            continue
        sha = _sha256(path)
        try:
            rec = json.loads(_fetch_record(path).read_text())
        except (OSError, ValueError):
            rec = {}
        if rec.get("sha256") != sha:
            rec = {}
        entries.append({"name": path.name, "url": rec.get("url", url), "sha256": sha,
                        "bytes": path.stat().st_size,
                        "fetched_utc": rec.get("fetched_utc", _utc(path.stat().st_mtime)),
                        "last_modified": rec.get("last_modified")})
    return entries


def write_manifest(path: pathlib.Path | None = None,
                   cache: pathlib.Path | None = None) -> pathlib.Path:
    """Write the input manifest. Identical inputs give identical bytes: entries
    are sorted by name and nothing in it records when the manifest was written."""
    path = MANIFEST if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"about": MANIFEST_ABOUT, "inputs": manifest_entries(cache)}
    _write_atomic(path, (json.dumps(doc, indent=2) + "\n").encode())
    return path


def main() -> int:
    games = load_games()
    live = live_season(games)
    print(f"schedule {GAMES_URL}\n  {int(games.season.min())}-{int(games.season.max())}, "
          f"games played through {live}")

    rel = load_release()
    report = parity_check(rel)
    print(f"parity check, release vs play-by-play, {report['season']}:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    bad = parity_failures(report)
    if bad:
        raise SystemExit(
            f"parity check failed for {report['season']}: "
            + ", ".join(f"{k}={report[k]}" for k in bad)
            + ". The play-by-play rebuild no longer reproduces the release, so seasons "
              "rebuilt from play-by-play would carry a different target; nothing was written.")

    kicking = load_kicking(games, rel)
    kicking["team"] = kicking["team"].replace(RELOCATIONS)
    ctx = team_game_context(games)

    df = kicking.merge(ctx, on=["season", "week", "team"], how="left")
    df["fantasy_points"] = fantasy_points(df)
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "kicker_games.parquet"
    df.to_parquet(path, index=False)
    print(f"\nwrote {path.relative_to(ROOT)}: {len(df):,} kicker-games, "
          f"{df.season.min()}-{df.season.max()}, {df.player_id.nunique():,} kickers")
    print(f"  unmatched to a scheduled game: {int(df.game_id.isna().sum()):,}")
    print(f"  fantasy points: mean {df.fantasy_points.mean():.2f}, "
          f"sd {df.fantasy_points.std():.2f}, max {df.fantasy_points.max():.0f}")
    m = write_manifest()
    print(f"wrote {m.relative_to(ROOT)}: {len(json.loads(m.read_text())['inputs'])} inputs")
    return 0


class _FakeResponse:
    """Stands in for an HTTP response in the self-check: `body` bytes, then
    optionally an exception, as a dropped connection would raise."""

    def __init__(self, body: bytes, length: int | None = None, fail: bool = False):
        self.body, self.fail = body, fail
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def read(self, n: int = -1) -> bytes:
        if self.body:
            out, self.body = self.body, b""
            return out
        if self.fail:
            raise ConnectionResetError("connection dropped mid-download")
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def self_check() -> int:
    """Scoring, the implied-total algebra, and the guards around the inputs.

    The first two are the places a silent sign error would change every number
    downstream. The rest prove, offline and inside a temporary directory, that a
    download is atomic, that a failed refresh keeps the old copy, that a missing
    or thin season and a parity difference are refused, and that the manifest is
    deterministic.
    """
    global CACHE, _urlopen
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
    c2 = team_game_context(normalise_games(g.assign(home_team="STL", away_team="SD")))
    assert set(c2.team) == {"LA", "LAC"}, set(c2.team)

    # Every input has one cache name and one HTTPS source, and the manifest maps
    # names back to the same URLs the fetches use.
    for url in (GAMES_URL, KICKING_URL, pbp_url(2021)):
        assert url.startswith("https://"), url
    assert (KICKING_CACHE, GAMES_CACHE, pbp_cache(2021)) == (
        "kicking.parquet", "games.csv", "pbp_2021.parquet")
    assert _input_url(KICKING_CACHE) == KICKING_URL
    assert _input_url(GAMES_CACHE) == GAMES_URL
    assert _input_url(pbp_cache(2021)) == pbp_url(2021)
    assert _input_url("pbp_2021.parquet.fetch.json") is None
    assert _input_url(".pbp_2021.parquet.x.part") is None

    # A season settles on 1 March of the next year, not at New Year (playoffs).
    ts = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()  # noqa: E731
    assert season_settled(2025, ts(2026, 3, 1))
    assert not season_settled(2025, ts(2026, 2, 28, 23))
    assert not season_settled(2026, ts(2026, 10, 26))

    # Coverage: two settled seasons, two weeks of one game each.
    sched = pd.DataFrame({"season": [2020, 2020, 2021, 2021], "week": [1, 2, 1, 2],
                          "home_team": "KC", "away_team": "BAL",
                          "home_score": [20, 21, 22, 23]})
    full = pd.DataFrame([{"season": s, "week": w, "team": t}
                         for s in (2020, 2021) for w in (1, 2) for t in ("KC", "BAL")])
    now = ts(2026, 10, 26)
    assert coverage_problems(full, sched, 2020, 2021, 0.9, now) == []
    hole = coverage_problems(full[full.season != 2021], sched, 2020, 2021, 0.9, now)
    assert len(hole) == 1 and hole[0].startswith("2021: 0 team-games"), hole
    thin = coverage_problems(full[~((full.season == 2020) & (full.week == 2))],
                             sched, 2020, 2021, 0.9, now)
    assert len(thin) == 1 and thin[0].startswith("2020: 2 team-games"), thin
    # An open season is allowed to trail by its latest played week, no more.
    assert coverage_problems(full[full.week == 1], sched, 2021, 2021, 0.9,
                             ts(2021, 10, 1)) == []
    assert coverage_problems(full[full.week == 1], sched, 2021, 2021, 0.9, now) != []

    # A clean parity report passes; any difference is named.
    clean = {"season": 2024, "rows_release": 567, "rows_pbp": 567, "only_release": 0,
             "only_pbp": 0, **{f"mismatch_{c}": 0 for c in ["fg_att", "fg_made"] + BUCKETS}}
    assert parity_failures(clean) == []
    assert parity_failures({**clean, "mismatch_fg_made_50_59": 2}) == ["mismatch_fg_made_50_59"]
    assert set(parity_failures({**clean, "rows_pbp": 560, "only_release": 7})) == {
        "only_release", "rows_pbp"}
    assert parity_season(pd.DataFrame({"season": [2023, 2024, 2026]}), now) == 2024

    saved = CACHE, _urlopen
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            CACHE = tmp / "cache"
            src = tmp / "src.csv"
            src.write_bytes(b"season,week\n2026,1\n")

            # A completed download lands whole, with a record of where it came from.
            p = fetch(src.as_uri(), GAMES_CACHE)
            assert p.read_bytes() == src.read_bytes()
            mode = 0o666 & ~_umask()
            for f in (p, _fetch_record(p)):
                assert f.stat().st_mode & 0o777 == mode, (f.name, oct(f.stat().st_mode))
            rec = json.loads(_fetch_record(p).read_text())
            assert rec["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
            assert rec["bytes"] == src.stat().st_size and rec["last_modified"]

            # A dropped connection or a short body leaves no file under the
            # cache name and no temporary file behind.
            for resp in (_FakeResponse(b"PAR1 half a file", fail=True),
                         _FakeResponse(b"PAR1 half a file", length=10_000)):
                _urlopen = lambda url, timeout, r=resp: r  # noqa: E731
                try:
                    fetch("https://example.invalid/x.parquet", pbp_cache(2015))
                    raise AssertionError("a broken download must raise")
                except OSError:
                    pass
                assert not (CACHE / pbp_cache(2015)).exists()
                assert not list(CACHE.glob("*.part")), list(CACHE.glob("*.part"))

            # A failed refresh keeps the old copy rather than leaving nothing.
            old = time.time() - 48 * 3600
            os.utime(p, (old, old))
            _urlopen = lambda url, timeout: _FakeResponse(b"x", fail=True)  # noqa: E731
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                kept = fetch("https://example.invalid/games.csv", GAMES_CACHE, max_age_hours=6)
            assert kept.read_bytes() == src.read_bytes()
            assert "could not refresh games.csv" in said.getvalue(), said.getvalue()

            # A truncated parquet already in the cache is fetched again.
            good = tmp / "good.parquet"
            pd.DataFrame({"a": [1, 2]}).to_parquet(good)
            (CACHE / pbp_cache(2016)).write_bytes(good.read_bytes()[:-4])
            assert not _complete(CACHE / pbp_cache(2016))
            _urlopen = saved[1]
            with contextlib.redirect_stdout(io.StringIO()):
                fetch(good.as_uri(), pbp_cache(2016))
            assert _complete(CACHE / pbp_cache(2016))
            assert pd.read_parquet(CACHE / pbp_cache(2016)).a.tolist() == [1, 2]

            # The manifest: one entry per input, contract keys, record used only
            # when it matches the bytes, identical output for identical inputs.
            (CACHE / KICKING_CACHE).write_bytes(good.read_bytes())   # placed by hand
            (CACHE / pbp_cache(2016)).write_bytes(b"PAR1 replaced PAR1")  # record now stale
            (CACHE / "notes.txt").write_text("not an input")
            e = {x["name"]: x for x in manifest_entries()}
            assert list(e) == [GAMES_CACHE, KICKING_CACHE, pbp_cache(2016)], list(e)
            assert all(list(x) == ["name", "url", "sha256", "bytes", "fetched_utc",
                                   "last_modified"] for x in e.values())
            assert e[GAMES_CACHE]["url"] == src.as_uri() and e[GAMES_CACHE]["last_modified"]
            for name, url in ((KICKING_CACHE, KICKING_URL), (pbp_cache(2016), pbp_url(2016))):
                assert e[name]["url"] == url and e[name]["last_modified"] is None, e[name]
                assert e[name]["fetched_utc"] == _utc((CACHE / name).stat().st_mtime)
            m1 = write_manifest(tmp / "m1.json").read_bytes()
            m2 = write_manifest(tmp / "m2.json").read_bytes()
            assert m1 == m2 and json.loads(m1)["inputs"][0]["name"] == GAMES_CACHE
            assert (tmp / "m1.json").stat().st_mode & 0o777 == mode

            # load_kicking end to end on one played game in FIRST_SEASON: a
            # release hole is filled from the season's play-by-play, a thin
            # season is refused, and a season that cannot be fetched is named.
            sched = pd.DataFrame({"season": [FIRST_SEASON], "week": [1],
                                  "home_team": ["KC"], "away_team": ["BAL"],
                                  "home_score": [20], "away_score": [17]})
            rel = pd.DataFrame([{**{c: 0 for c in KEEP}, "season": FIRST_SEASON, "week": 1,
                                 "season_type": "REG", "player_id": f"k{t}",
                                 "player_display_name": f"K {t}", "team": t, "fg_att": 1,
                                 "fg_made": 1, "fg_made_30_39": 1, "fg_long": 35}
                                for t in ("KC", "BAL")])
            assert len(load_kicking(sched, rel)) == 2
            try:
                load_kicking(sched, rel[rel.team == "KC"])
                raise AssertionError("a season with half its team-games must stop the build")
            except SystemExit as exc:
                assert f"{FIRST_SEASON}: 1 team-games" in str(exc), exc
            hole = rel.assign(season=FIRST_SEASON + 1)   # the release lacks the season

            def offline(url, timeout):
                raise OSError("offline")
            _urlopen = offline
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    load_kicking(sched, hole)
                raise AssertionError("a season that cannot be fetched must stop the build")
            except SystemExit as exc:
                assert f"{FIRST_SEASON}: OSError: offline" in str(exc), exc
            pd.DataFrame({
                "season": FIRST_SEASON, "week": 1, "season_type": "REG",
                "posteam": ["KC", "BAL"], "kicker_player_id": ["kKC", "kBAL"],
                "kicker_player_name": ["K KC", "K BAL"], "field_goal_attempt": 1,
                "field_goal_result": "made", "kick_distance": 35.0,
                "extra_point_attempt": 0, "extra_point_result": None,
            }).to_parquet(CACHE / pbp_cache(FIRST_SEASON))
            with contextlib.redirect_stdout(io.StringIO()):
                filled = load_kicking(sched, hole)
            got = filled[filled.season == FIRST_SEASON].sort_values("team")
            assert got.team.tolist() == ["BAL", "KC"] and got.fg_made_30_39.tolist() == [1, 1]
    finally:
        CACHE, _urlopen = saved
    print("build_dataset self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main())
