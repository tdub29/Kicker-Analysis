from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import normalize_team_codes, normalize_team_codes_forward, safe_divide, select_numeric_columns

logger = logging.getLogger(__name__)


@dataclass
class WindowSpec:
    min_week: int
    max_week: int
    window_weeks: int


def build_games_played(schedule: pd.DataFrame, week: int, window_weeks: int) -> pd.DataFrame:
    window = schedule[(schedule["week"] < week) & (schedule["week"] > week - (window_weeks + 1))]
    home = window["home_team"].value_counts().rename_axis("team").reset_index(name="home_games")
    away = window["away_team"].value_counts().rename_axis("team").reset_index(name="away_games")
    merged = pd.merge(home, away, on="team", how="outer").fillna(0)
    merged["G"] = merged["home_games"] + merged["away_games"]
    return merged[["team", "G"]]


def aggregate_kicking_window(weekly: pd.DataFrame, week: int, window_weeks: int) -> pd.DataFrame:
    window = weekly[(weekly["week"] < week) & (weekly["week"] > week - (window_weeks + 1))]
    numeric_stats = select_numeric_columns(window, exclude={"week", "season"})
    grouped = pd.concat([window[["team"]], numeric_stats], axis=1).groupby("team", as_index=False).sum()
    return grouped


def add_opponent_team(schedule_week: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    home = schedule_week[["week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "oppteam"}
    )
    away = schedule_week[["week", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "oppteam"}
    )
    matchups = pd.concat([home, away], ignore_index=True)
    return team_df.merge(matchups, on="team", how="left")


def compute_fg_allowed(pbp: pd.DataFrame, week: int, window_weeks: int, down_filter: int | None = None) -> pd.DataFrame:
    fga = pbp[(pbp["field_goal_attempt"] == 1)].copy()
    if down_filter is not None:
        fga = fga[fga["down"] == down_filter]
    fga = fga[(fga["week"] < week) & (fga["week"] > week - (window_weeks + 1))]
    fga["field_goal_result"] = np.where(fga["field_goal_result"] == "made", 1, 0)
    def_team = (
        fga.groupby("defteam", as_index=False)[["field_goal_attempt", "field_goal_result"]]
        .sum()
        .rename(columns={"defteam": "team"})
    )
    off_team = (
        fga.groupby("posteam", as_index=False)[["field_goal_attempt", "field_goal_result"]]
        .sum()
        .rename(columns={"posteam": "team"})
    )
    return def_team, off_team


def compute_fantasy_points(weekly: pd.DataFrame) -> pd.DataFrame:
    data = weekly.copy()
    if {"fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_"}.issubset(
        data.columns
    ):
        data["fantasy_points"] = (
            data["pat_made"].fillna(0)
            + data["fg_made_0_19"].fillna(0) * 3
            + data["fg_made_20_29"].fillna(0) * 3
            + data["fg_made_30_39"].fillna(0) * 3
            + data["fg_made_40_49"].fillna(0) * 4
            + data["fg_made_50_59"].fillna(0) * 5
            + data["fg_made_60_"].fillna(0) * 5
        )
    else:
        data["fantasy_points"] = data["pat_made"].fillna(0) + data["fg_made"].fillna(0) * 3
    return data[["season", "week", "team", "fantasy_points"]]


def build_week_features(
    weekly: pd.DataFrame,
    schedule: pd.DataFrame,
    pbp: pd.DataFrame,
    week: int,
    season: int,
    window_weeks: int,
) -> pd.DataFrame:
    schedule = normalize_team_codes(schedule, "home_team", season)
    schedule = normalize_team_codes(schedule, "away_team", season)

    weekly = normalize_team_codes(weekly, "team", season)
    pbp = normalize_team_codes(pbp, "defteam", season)
    pbp = normalize_team_codes(pbp, "posteam", season)

    games_played = build_games_played(schedule, week, window_weeks)
    kicking = aggregate_kicking_window(weekly, week, window_weeks)
    kicking = kicking.merge(games_played, on="team", how="left")
    kicking["attpg"] = safe_divide(kicking.get("fg_att", pd.Series(0, index=kicking.index)), kicking["G"])

    schedule_week = schedule[schedule["week"] == week]
    kicking = add_opponent_team(schedule_week, kicking)

    def_team, _ = compute_fg_allowed(pbp, week, window_weeks)
    def_team = def_team.merge(games_played, on="team", how="left")
    def_team["pweek"] = week
    kicking = kicking.merge(def_team[["team", "field_goal_attempt", "G"]], left_on="oppteam", right_on="team", how="left")
    kicking = kicking.rename(columns={"field_goal_attempt": "oppfga", "G_y": "OG"}).drop(columns=["team_y"], errors="ignore")
    kicking = kicking.rename(columns={"team_x": "team"})
    kicking["oppfgapg"] = safe_divide(kicking["oppfga"], kicking["OG"])
    kicking["projfga"] = (kicking["oppfgapg"] + kicking["attpg"]) / 2
    kicking["pweek"] = week
    kicking["season"] = season

    def_fourth, off_fourth = compute_fg_allowed(pbp, week, window_weeks, down_filter=4)
    def_fourth["pweek"] = week
    off_fourth["pweek"] = week

    return kicking, def_fourth, off_fourth


def merge_home_away_features(
    schedule: pd.DataFrame,
    kickers: pd.DataFrame,
    def_fourth: pd.DataFrame,
    off_fourth: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    schedule = normalize_team_codes(schedule, "home_team", season)
    schedule = normalize_team_codes(schedule, "away_team", season)

    step1 = schedule.merge(kickers, left_on=["week", "home_team"], right_on=["pweek", "team"], how="left")
    step2 = step1.merge(kickers, left_on=["week", "away_team"], right_on=["pweek", "team"], how="left", suffixes=("", "_away"))

    home_cols = [col for col in step2.columns if col.endswith("_away")]
    rename_map = {col: f"away.{col.replace('_away', '')}" for col in home_cols}
    step2 = step2.rename(columns=rename_map)

    base_cols = [col for col in step2.columns if col not in rename_map.values() and col not in rename_map.keys()]
    step2 = step2[base_cols + list(rename_map.values())]

    step3 = step2.merge(def_fourth, left_on=["week", "away_team"], right_on=["pweek", "team"], how="left")
    step4 = step3.merge(def_fourth, left_on=["week", "home_team"], right_on=["pweek", "team"], how="left", suffixes=("", "_home_def"))

    step4 = step4.rename(columns={
        "field_goal_attempt": "away.adjfield_goal_attempt",
        "field_goal_result": "away.adjfield_goal_result",
        "field_goal_attempt_home_def": "home.adjfield_goal_attempt",
        "field_goal_result_home_def": "home.adjfield_goal_result",
    })

    step5 = step4.merge(off_fourth, left_on=["week", "away_team"], right_on=["pweek", "team"], how="left")
    step6 = step5.merge(off_fourth, left_on=["week", "home_team"], right_on=["pweek", "team"], how="left", suffixes=("", "_home_off"))

    step6 = step6.rename(columns={
        "field_goal_attempt": "away.adjfield_goal_attempt_for",
        "field_goal_result": "away.adjfield_goal_result_for",
        "field_goal_attempt_home_off": "home.adjfield_goal_attempt_for",
        "field_goal_result_home_off": "home.adjfield_goal_result_for",
    })

    return step6


def add_actual_stats(kickanalysis: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    actual = weekly[["team", "week", "season", "fg_att", "fg_made", "pat_made"]].copy()
    actual = actual.rename(
        columns={
            "fg_att": "fg_att",
            "fg_made": "fg_made",
            "pat_made": "pat_made",
        }
    )

    away = kickanalysis.merge(
        actual,
        left_on=["week", "away_team", "season"],
        right_on=["week", "team", "season"],
        how="left",
    ).rename(columns={
        "fg_att": "actualaway.fg_att",
        "fg_made": "actualaway.fg_made",
        "pat_made": "actualaway.pat_made",
    })

    home = away.merge(
        actual,
        left_on=["week", "home_team", "season"],
        right_on=["week", "team", "season"],
        how="left",
        suffixes=("", "_home"),
    ).rename(columns={
        "fg_att": "actualhome.fg_att",
        "fg_made": "actualhome.fg_made",
        "pat_made": "actualhome.pat_made",
    })

    return home


def add_games_played(kickanalysis: pd.DataFrame, schedule: pd.DataFrame, window_weeks: int) -> pd.DataFrame:
    rows = []
    for season in sorted(kickanalysis["season"].unique()):
        season_schedule = schedule[schedule["season"] == season]
        for week in sorted(kickanalysis[kickanalysis["season"] == season]["week"].unique()):
            gp = build_games_played(season_schedule, week, window_weeks)
            gp["week"] = week
            gp["season"] = season
            rows.append(gp)
    games = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    away = kickanalysis.merge(
        games,
        left_on=["week", "season", "away_team"],
        right_on=["week", "season", "team"],
        how="left",
    ).rename(columns={"G": "awaygamesplayed"})

    home = away.merge(
        games,
        left_on=["week", "season", "home_team"],
        right_on=["week", "season", "team"],
        how="left",
        suffixes=("", "_home"),
    ).rename(columns={"G": "homegamesplayed"})

    return home


def compute_expected_metrics(kickanalysis: pd.DataFrame) -> pd.DataFrame:
    analysis = kickanalysis.copy()
    analysis["adjhome.attpg"] = safe_divide(analysis["home.adjfield_goal_attempt"], analysis["homegamesplayed"])
    analysis["adjaway.attpg"] = safe_divide(analysis["away.adjfield_goal_attempt"], analysis["awaygamesplayed"])
    analysis["adjhome.attpgagainst"] = safe_divide(
        analysis["home.adjfield_goal_attempt_for"], analysis["homegamesplayed"]
    )
    analysis["adjaway.attpgallowed"] = safe_divide(
        analysis["away.adjfield_goal_attempt_for"], analysis["awaygamesplayed"]
    )
    analysis["homeexpatt"] = (analysis["adjhome.attpg"] + analysis["adjaway.attpgallowed"]) / 2
    analysis["awayexpatt"] = (analysis["adjaway.attpg"] + analysis["adjhome.attpgagainst"]) / 2
    analysis["homecash"] = np.where(analysis["actualhome.fg_made"] > 1.5, 1, -1)
    analysis["awaycash"] = np.where(analysis["actualaway.fg_made"] > 1.5, 1, -1)
    analysis["homexppg"] = safe_divide(analysis["home.pat_made"], analysis["homegamesplayed"])
    analysis["awayxppg"] = safe_divide(analysis["away.pat_made"], analysis["awaygamesplayed"])
    analysis["homekick_pct"] = safe_divide(
        analysis["home.fg_made"], analysis["home.fg_made"] + analysis["home.fg_missed"]
    )
    analysis["awaykick_pct"] = safe_divide(
        analysis["away.fg_made"], analysis["away.fg_made"] + analysis["away.fg_missed"]
    )
    analysis["homeexfppg"] = analysis["homeexpatt"] * 3.8 * analysis["homekick_pct"] + analysis["homexppg"]
    analysis["awayexfppg"] = analysis["awayexpatt"] * 3.8 * analysis["awaykick_pct"] + analysis["awayxppg"]
    analysis["adjhomeexpatt"] = 0.35 * analysis["adjhome.attpg"] + 0.65 * analysis["adjaway.attpgallowed"]
    analysis["adjawayexpatt"] = 0.35 * analysis["adjaway.attpg"] + 0.65 * analysis["adjhome.attpgagainst"]
    analysis["homeregressfp"] = (
        3.96760
        + 0.40388 * analysis["homexppg"]
        + 1.11950 * analysis["adjhomeexpatt"]
        + 2.24413 * analysis["homekick_pct"]
        - 0.66712 * analysis["adjhomeexpatt"] * analysis["homekick_pct"]
    )
    analysis["awayregressfp"] = (
        3.96760
        + 0.40388 * analysis["awayxppg"]
        + 1.11950 * analysis["adjawayexpatt"]
        + 2.24413 * analysis["awaykick_pct"]
        - 0.66712 * analysis["adjawayexpatt"] * analysis["awaykick_pct"]
    )
    return analysis


def add_fantasy_points(analysis: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    fantasy = compute_fantasy_points(weekly)
    merged = analysis.merge(
        fantasy,
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).rename(columns={"fantasy_points": "homefppg"})

    merged = merged.merge(
        fantasy,
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=("", "_away"),
    ).rename(columns={"fantasy_points": "awayfppg"})

    merged["homefppg"] = merged["homefppg"].fillna(0)
    merged["awayfppg"] = merged["awayfppg"].fillna(0)
    return merged
