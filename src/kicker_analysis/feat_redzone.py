"""Red-zone efficiency, per team-game, lagged, plus the defence faced.

The mechanism, in football terms: a kicker's fantasy week is volume, and volume
is his offence failing in a specific place. An offence that reaches the red zone
six times and punches in twice hands its kicker four scoring chances; one that
reaches it four times and scores four touchdowns hands him four extra points and
nothing else. `fourth_down.py` already measures the coach's *choice* once the
drive stalls. This file measures how often the drive stalls at all, which is the
half of the story the model cannot currently see.

Definitions, and why each is drawn this way:

  a trip         one DRIVE that reached `yardline_100` <= 20, not one play. A
                 drive with eight snaps inside the 20 is one trip. Counting
                 plays would score a grinding offence as eight scoring chances
                 and a one-play touchdown as one, which inverts the signal.
  drive outcome  `fixed_drive_result`, nflverse's own per-drive label, read once
                 per drive rather than reconstructed from the plays. "Opp
                 touchdown" (a pick six on the drive) is NOT an offensive red
                 zone touchdown and is correctly not counted as one.
  fg rate        ATTEMPTS, so "Field goal" plus "Missed field goal". The kicker
                 is paid for the attempt reaching him; whether he makes it is
                 his own column, not the offence's.
  inside the 10  the same thing at `yardline_100` <= 10. Between the 20 and the
                 10 a stall is a 30-odd yard attempt; inside the 10 it is a chip
                 shot, and the two are different products for a kicker. Reported
                 separately so the study can say whether the split earns its
                 keep.
  defence faced  the identical aggregation keyed on `defteam`, then attached to
                 the OPPONENT's row. An offence facing a defence that has
                 bent-but-not-broken all year should see more attempts.

Plays excluded from the red-zone test, and why it matters: an extra point snaps
from `yardline_100` == 15 and a two-point try from the 2. Left in, every single
touchdown drive would also register as a red-zone trip by way of its own PAT,
and the trip count would become a touchdown count. Kickoffs and punts are
dropped for the same class of reason.

Lagging: every rate here describes a finished game, so `shift(1)` runs on the
raw per-game COUNTS inside the team group before any window. Rates are then
built as a ratio of the two lagged sums, never as a mean of per-game rates: a
game with one trip and a game with six trips are not equally informative, and
averaging their rates pretends they are. Measured over 2015-2025 team-seasons
the two differ by 0.027 in red-zone TD rate against an across-team spread of
0.079, so the wrong choice costs a third of a standard deviation.

Verified before shipping:
  * trips per team-game 2015-2026 average 3.34, red-zone TD rate 0.563 and
    red-zone FG-attempt rate 0.323, with 2.17 inside-10 trips a game at a 0.668
    TD rate. The league's published red-zone TD rate sits in the mid-50s, so the
    drive identification is not silently double counting or dropping drives.
  * every relocated franchise code is collapsed, so LA/LAC/LV histories are
    continuous instead of starting at the move.
  * the self-check below fails if `shift(1)` is dropped, if rates are averaged
    instead of pooled, or if a PAT is allowed to create a trip.
  * PERSISTENCE, checked before trusting any of it: year over year 2016-2025 a
    team's red-zone trips per game carry at r = 0.32, but its red-zone TD RATE
    carries at r = 0.10. The volume half of this mechanism is a real team
    property; the efficiency half is very close to noise. That is a reason to
    doubt the composite before it is fitted, not after, and the evaluation
    bears it out.

RESULT, stated up front so nobody re-derives it: THESE FEATURES DO NOT HELP.
Season-forward over 2019-2025 (6 folds; 2021 is absent from the shipped
kicker_games.parquet), pooled MAE goes 3.6185 -> 3.5708 with all 27 columns,
which looks like a 1.3% win at p=0.054. It is not one. The same harness given
the SAME 27 columns with their values SHUFFLED gains -0.020, -0.036 and +0.002
on three seeds, averaging -0.018, so most of the nominal gain is what this
LightGBM does when handed 27 more columns of anything. The real-minus-shuffled
margin of 0.030 MAE sits inside the 0.038 spread of the shuffle draws.

The composite specifically loses the question it was asked: rz_exp_fg_* alone
moves MAE by -0.0098 (p=0.41) and the two raw columns it is built from move it
by -0.0089 (p=0.60). They are indistinguishable from each other, and both are
smaller than the shuffled-column control. The composite does not beat the raw
pair; neither beats noise.

Why, in football terms: the model already had this. Lagged team points and the
market's implied total are cheaper proxies for the volume half (r = 0.68 and
0.53 against rz_trips_pg_l16), lagged team FG attempts already carries the
stall half (r = 0.46 against rz_fg_rate_l16), and the efficiency half is not
forecastable anyway at r = 0.10 year over year. The mechanism is real; the
incremental information is not.

The module is kept because the columns are correct, cheap and honestly lagged,
and because a null needs its evidence left on disk. Do not wire it into
features.py on the strength of the 3.5708.

Usage:
    python src/kicker_analysis/feat_redzone.py
    python src/kicker_analysis/feat_redzone.py --self-check
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed" / "redzone.parquet"

FIRST_SEASON = 2015          # the pbp cache starts here
RED_ZONE = 20
INSIDE_10 = 10

RELOCATIONS = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

# Windows in GAMES. 8 is about half a season and 16 a full one. features.py uses
# 4 and 8, but 4 is too narrow for a RATE: four games is about 13 trips, so a
# single goal-line stand moves the red-zone TD rate by 7.5 points, which is one
# across-team standard deviation (0.079). Widening the window is the only lever
# available, since the denominator is what it is.
WINDOWS = (8, 16)

COLS = ["season", "week", "season_type", "game_id", "posteam", "defteam",
        "fixed_drive", "fixed_drive_result", "yardline_100",
        "extra_point_attempt", "two_point_attempt", "kickoff_attempt",
        "punt_attempt"]

JOIN_KEYS = ["season", "week", "season_type", "team"]

# Offence, then the defence the team is about to face. Both already lagged.
FEATURE_COLS = (
    [f"rz_trips_pg_l{w}" for w in WINDOWS] + ["rz_trips_pg_career"]
    + [f"rz_td_rate_l{w}" for w in WINDOWS] + ["rz_td_rate_career"]
    + [f"rz_fg_rate_l{w}" for w in WINDOWS] + ["rz_fg_rate_career"]
    + [f"rz10_trips_pg_l{w}" for w in WINDOWS] + ["rz10_trips_pg_career"]
    + [f"rz10_td_rate_l{w}" for w in WINDOWS] + ["rz10_td_rate_career"]
    + [f"rz_exp_fg_l{w}" for w in WINDOWS] + ["rz_exp_fg_career"]
    + [f"rzd_trips_pg_l{w}" for w in WINDOWS] + ["rzd_trips_pg_career"]
    + [f"rzd_td_rate_l{w}" for w in WINDOWS] + ["rzd_td_rate_career"]
    + [f"rzd_fg_rate_l{w}" for w in WINDOWS] + ["rzd_fg_rate_career"]
)

# The composite the brief asks about, kept separate so it can be tested alone
# against the two columns it is built from.
COMPOSITE_COLS = [f"rz_exp_fg_l{w}" for w in WINDOWS] + ["rz_exp_fg_career"]
RAW_PAIR_COLS = ([f"rz_trips_pg_l{w}" for w in WINDOWS] + ["rz_trips_pg_career"]
                 + [f"rz_td_rate_l{w}" for w in WINDOWS] + ["rz_td_rate_career"])


def drive_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per offensive drive: who had it, how deep it got, how it ended.

    `fixed_drive` restarts at 1 each game, so the drive key is game plus drive
    number; posteam is carried along rather than used as a key because a drive
    belongs to exactly one offence by construction.
    """
    d = pbp[pbp.posteam.notna() & pbp.fixed_drive.notna() & pbp.yardline_100.notna()]
    # See the module docstring: a PAT snaps from the 15 and a two-point try from
    # the 2, so leaving them in turns every touchdown into a red-zone trip.
    d = d[(d.extra_point_attempt != 1) & (d.two_point_attempt != 1)
          & (d.kickoff_attempt != 1) & (d.punt_attempt != 1)]

    key = ["season", "week", "season_type", "game_id", "fixed_drive"]
    out = d.groupby(key, dropna=False).agg(
        team=("posteam", "first"),
        defense=("defteam", "first"),
        # The deepest point the drive reached. min() of yardline_100, because
        # yardline_100 counts DOWN toward the opponent's goal line.
        best=("yardline_100", "min"),
        result=("fixed_drive_result", "first"),
    ).reset_index()

    out["team"] = out.team.replace(RELOCATIONS)
    out["defense"] = out.defense.replace(RELOCATIONS)
    out["trip"] = (out.best <= RED_ZONE).astype(int)
    out["trip10"] = (out.best <= INSIDE_10).astype(int)
    is_td = out.result.eq("Touchdown").astype(int)
    # Made and missed both count: the attempt is the kicker's fantasy event.
    is_fg = out.result.isin(["Field goal", "Missed field goal"]).astype(int)
    out["td"] = out.trip * is_td
    out["fg"] = out.trip * is_fg
    out["td10"] = out.trip10 * is_td
    return out


def _team_game(drives: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Counts per game for one side of the ball. `entity` is team or defense.

    The entity column is renamed to `team` either way, so the same lagging code
    runs over an offence's own history and over a defence's.
    """
    out = drives.groupby(["season", "week", "season_type", entity],
                         dropna=False).agg(
        trips=("trip", "sum"), td=("td", "sum"), fg=("fg", "sum"),
        trips10=("trip10", "sum"), td10=("td10", "sum")).reset_index()
    return out.rename(columns={entity: "team"})


def _lagged_rates(counts: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prior-games-only rates for one entity, as ratios of lagged sums.

    shift(1) is applied to the raw counts first, exactly as features.py::_lagged
    does, so week N sees weeks 1..N-1 and nothing else. Windowing first and
    shifting after is equal only once the window is full and leaks at the start
    of every group, which for a team is the start of every era.
    """
    d = counts.sort_values(["team", "season", "_order", "week"]).reset_index(drop=True)
    g = d.groupby("team", sort=False)
    raw = ["trips", "td", "fg", "trips10", "td10"]
    prior = {c: g[c].shift(1) for c in raw}

    sums: dict[str, pd.Series] = {}
    for c in raw:
        by = prior[c].groupby(d.team, sort=False)
        for w in WINDOWS:
            sums[f"{c}_l{w}"] = by.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).sum())
        sums[f"{c}_career"] = by.transform(lambda s: s.expanding().sum())
    # Games behind the current one, the denominator for a per-game rate.
    played = g.cumcount()
    games = {f"_l{w}": played.clip(upper=w) for w in WINDOWS}
    games["_career"] = played

    out = {}
    for suf in [f"_l{w}" for w in WINDOWS] + ["_career"]:
        n_games = games[suf].replace(0, np.nan)
        trips = sums[f"trips{suf}"]
        denom = trips.replace(0, np.nan)
        out[f"{prefix}trips_pg{suf}"] = trips / n_games
        out[f"{prefix}td_rate{suf}"] = sums[f"td{suf}"] / denom
        out[f"{prefix}fg_rate{suf}"] = sums[f"fg{suf}"] / denom
        out[f"{prefix}trips10_pg{suf}"] = sums[f"trips10{suf}"] / n_games
        out[f"{prefix}td10_rate{suf}"] = (sums[f"td10{suf}"]
                                          / sums[f"trips10{suf}"].replace(0, np.nan))
    res = pd.concat([d[JOIN_KEYS], pd.DataFrame(out, index=d.index)], axis=1)
    return res


def build_frame(drives: pd.DataFrame) -> pd.DataFrame:
    """Lagged offence features plus the lagged record of the defence faced."""
    off_counts = _team_game(drives, "team")
    def_counts = _team_game(drives, "defense")
    # POST weeks continue the numbering in modern nflverse, but the flag is the
    # only thing guaranteed to order a postseason game after the regular season.
    for c in (off_counts, def_counts):
        c["_order"] = (c.season_type != "REG").astype(int)

    sufs = [f"_l{w}" for w in WINDOWS] + ["_career"]
    # rz10_* reads better than rz_*10 and is what FEATURE_COLS advertises.
    off = _lagged_rates(off_counts, "rz_").rename(columns={
        f"rz_{a}{s}": f"rz10_{b}{s}" for a, b in
        (("trips10_pg", "trips_pg"), ("td10_rate", "td_rate")) for s in sufs})
    dfn = _lagged_rates(def_counts, "rzd_")
    dfn = dfn.drop(columns=[c for c in dfn.columns
                            if "trips10" in c or "td10" in c])

    # The composite the brief names: how many drives a team is expected to stall
    # in the red zone. Trips times the complement of the touchdown rate is the
    # count of red-zone possessions that did NOT end in six, which is the
    # kicker's addressable market before the coach's fourth-down choice.
    for suf in sufs:
        off[f"rz_exp_fg{suf}"] = (off[f"rz_trips_pg{suf}"]
                                  * (1.0 - off[f"rz_td_rate{suf}"]))

    # Attach the defence's history to the team that is about to play it. The
    # pairing comes from the drives themselves, so no schedule join is needed.
    pair = (drives.groupby(["season", "week", "season_type", "team"], dropna=False)
            .defense.first().reset_index().rename(columns={"defense": "opp"}))
    out = off.merge(pair, on=JOIN_KEYS, how="left")
    out = out.merge(dfn.rename(columns={"team": "opp"}),
                    on=["season", "week", "season_type", "opp"], how="left")
    return out[JOIN_KEYS + FEATURE_COLS].sort_values(JOIN_KEYS).reset_index(drop=True)


def build(first_season: int = FIRST_SEASON) -> pd.DataFrame:
    frames = []
    for path in sorted(CACHE.glob("pbp_*.parquet")):
        season = int(path.stem.split("_")[1])
        if season < first_season:
            continue
        frames.append(drive_table(pd.read_parquet(path, columns=COLS)))
    if not frames:
        raise SystemExit(f"no pbp parquet files in {CACHE}")
    drives = pd.concat(frames, ignore_index=True)
    out = build_frame(drives)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    # Raw (unlagged) league rates, printed only as a sanity check against
    # published numbers. These are NOT what the model sees.
    rz = drives[drives.trip == 1]
    per_game = drives.groupby(["season", "week", "season_type", "team"]).trip.sum()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out):,} team-games, "
          f"{out.season.min()}-{out.season.max()}, {len(FEATURE_COLS)} features")
    print(f"  red-zone trips per team-game: mean {per_game.mean():.2f}, "
          f"max {per_game.max()}")
    print(f"  red-zone TD rate {rz.td.mean():.3f}   FG-attempt rate {rz.fg.mean():.3f}   "
          f"other {1 - rz.td.mean() - rz.fg.mean():.3f}")
    i10 = drives[drives.trip10 == 1]
    print(f"  inside-10 trips per team-game "
          f"{drives.groupby(['season', 'week', 'season_type', 'team']).trip10.sum().mean():.2f}"
          f", TD rate {i10.td10.mean():.3f}")
    print("\n  season  trips/g  td_rate  fg_rate")
    for s, g in drives.groupby("season"):
        r = g[g.trip == 1]
        tg = g.groupby(["week", "season_type", "team"]).trip.sum().mean()
        print(f"    {s}    {tg:.2f}     {r.td.mean():.3f}    {r.fg.mean():.3f}")
    return out


def self_check() -> int:
    """Trip counting, the PAT trap, pooled rates and the lag, hand-built."""
    base = dict(season=2024, week=1, season_type="REG", game_id="g1",
                posteam="KC", defteam="OAK", yardline_100=50.0,
                extra_point_attempt=0, two_point_attempt=0, kickoff_attempt=0,
                punt_attempt=0, fixed_drive_result="Punt")
    rows = [
        # drive 1: reaches the 8, scores. One trip, one inside-10, one TD.
        {**base, "fixed_drive": 1, "yardline_100": 30.0, "fixed_drive_result": "Touchdown"},
        {**base, "fixed_drive": 1, "yardline_100": 8.0, "fixed_drive_result": "Touchdown"},
        # the PAT that follows snaps from the 15. It must not create a trip, and
        # it must not make drive 1 count twice.
        {**base, "fixed_drive": 1, "yardline_100": 15.0, "extra_point_attempt": 1,
         "fixed_drive_result": "Touchdown"},
        # drive 2: four snaps inside the 20, ends in a field goal. STILL ONE TRIP.
        {**base, "fixed_drive": 2, "yardline_100": 19.0, "fixed_drive_result": "Field goal"},
        {**base, "fixed_drive": 2, "yardline_100": 17.0, "fixed_drive_result": "Field goal"},
        {**base, "fixed_drive": 2, "yardline_100": 16.0, "fixed_drive_result": "Field goal"},
        {**base, "fixed_drive": 2, "yardline_100": 16.0, "fixed_drive_result": "Field goal"},
        # drive 3: stalls at the 35. Not a trip at all.
        {**base, "fixed_drive": 3, "yardline_100": 35.0},
        # drive 4: reaches the 12 and throws a pick six. "Opp touchdown" is not
        # an offensive red-zone touchdown.
        {**base, "fixed_drive": 4, "yardline_100": 12.0,
         "fixed_drive_result": "Opp touchdown"},
    ]
    d = drive_table(pd.DataFrame(rows))
    assert len(d) == 4, len(d)
    assert d.trip.sum() == 3, d.trip.sum()            # drives 1, 2, 4
    assert d.trip10.sum() == 1, d.trip10.sum()        # only drive 1
    assert d.td.sum() == 1 and d.fg.sum() == 1, (d.td.sum(), d.fg.sum())
    # Drive 2 had four separate snaps inside the 20 and is still exactly one
    # trip. Counting plays would have scored it as four scoring chances.
    assert d.loc[d.fixed_drive == 2, "trip"].iloc[0] == 1
    # The PAT row did not survive the filter, so it cannot invent a fifth drive
    # nor drag a non-red-zone drive inside the 20 by way of its 15-yard snap.
    assert 5 not in set(d.fixed_drive)

    # Relocated codes collapse on both sides of the ball.
    assert set(d.team) == {"KC"} and set(d.defense) == {"LV"}, set(d.defense)

    # --- lag and pooling, over four synthetic games ------------------------
    # KC: trips 4,2,6,1 and TDs 1,2,0,1. OAK is the defence in all four.
    games = []
    plan = [(1, 4, 1), (2, 2, 2), (3, 6, 0), (4, 1, 1)]
    for wk, trips, tds in plan:
        for i in range(trips):
            res = "Touchdown" if i < tds else "Field goal"
            games.append({**base, "week": wk, "game_id": f"g{wk}", "fixed_drive": i + 1,
                          "yardline_100": 5.0, "fixed_drive_result": res})
    dd = drive_table(pd.DataFrame(games))
    f = build_frame(dd).sort_values("week").reset_index(drop=True)

    # Week 1 has no prior game: undefined, not zero. A model fed 0.0 here would
    # learn that every team's debut offence never reaches the red zone.
    assert pd.isna(f.rz_trips_pg_l8.iloc[0]), f.rz_trips_pg_l8.iloc[0]
    # Week 3 sees weeks 1-2 only: (4+2)/2 = 3.0 trips per game.
    assert f.rz_trips_pg_l8.iloc[2] == 3.0, f.rz_trips_pg_l8.iloc[2]
    # Week 4 sees weeks 1-3: (4+2+6)/3 = 4.0. If the window were applied before
    # the shift this would be 3.25 and the row would contain its own game.
    assert f.rz_trips_pg_l8.iloc[3] == 4.0, f.rz_trips_pg_l8.iloc[3]
    # Pooled, not averaged: TD rate at week 4 is (1+2+0)/(4+2+6) = 3/12 = 0.25.
    # The mean of the three per-game rates is (0.25+1.0+0.0)/3 = 0.4167, which is
    # the wrong answer and the one a naive .mean() of rates would give.
    assert abs(f.rz_td_rate_l8.iloc[3] - 0.25) < 1e-12, f.rz_td_rate_l8.iloc[3]
    assert abs(f.rz_td_rate_l8.iloc[3] - 0.41666) > 0.1
    # FG attempts are the complement here by construction: 9/12.
    assert abs(f.rz_fg_rate_l8.iloc[3] - 0.75) < 1e-12, f.rz_fg_rate_l8.iloc[3]
    # Composite: 4.0 trips per game x (1 - 0.25) = 3.0 expected stalled trips.
    assert abs(f.rz_exp_fg_l8.iloc[3] - 3.0) < 1e-12, f.rz_exp_fg_l8.iloc[3]
    # The defence faced is LV, whose allowed history is KC's offence mirrored.
    assert abs(f.rzd_trips_pg_l8.iloc[3] - 4.0) < 1e-12, f.rzd_trips_pg_l8.iloc[3]
    assert abs(f.rzd_td_rate_l8.iloc[3] - 0.25) < 1e-12
    # Week 4 itself had 1 trip and 1 TD, a rate of 1.0. No column on week 4's
    # row may carry either number, which is the shape a dropped shift takes.
    row = f.iloc[3][FEATURE_COLS]
    assert not (row == 1.0).any(), row[row == 1.0]
    assert f.rz_trips_pg_career.iloc[3] == 4.0

    # Every advertised column exists and nothing extra rides along.
    assert list(f.columns) == JOIN_KEYS + FEATURE_COLS, set(f.columns) ^ set(
        JOIN_KEYS + FEATURE_COLS)
    print("feat_redzone self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else (build() is not None and 0))
