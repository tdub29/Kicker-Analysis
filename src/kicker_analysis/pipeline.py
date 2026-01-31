from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .data_ingestion import get_season_data
from .feature_engineering import (
    add_actual_stats,
    add_fantasy_points,
    add_games_played,
    build_week_features,
    compute_expected_metrics,
    merge_home_away_features,
)
from .utils import ensure_dir, normalize_team_codes_forward

logger = logging.getLogger(__name__)


def build_kicker_analysis(config: ProjectConfig) -> pd.DataFrame:
    rows = []
    weekly_rows = []
    schedule_rows = []
    for season in config.seasons:
        logger.info("Processing season %s", season)
        season_data = get_season_data(season)
        weekly = season_data["weekly"].copy()
        schedules = season_data["schedules"].copy()
        pbp = season_data["pbp"].copy()

        weekly_rows.append(weekly)
        schedule_rows.append(schedules)

        for week in range(config.min_week, config.max_week + 1):
            kickers, def_fourth, off_fourth = build_week_features(
                weekly=weekly,
                schedule=schedules,
                pbp=pbp,
                week=week,
                season=season,
                window_weeks=config.window_weeks,
            )
            schedule_week = schedules[schedules["week"] == week]
            analysis = merge_home_away_features(
                schedule_week,
                kickers,
                def_fourth,
                off_fourth,
                season,
            )
            analysis = analysis[analysis["week"] > config.window_weeks]
            rows.append(analysis)

    kickanalysis = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if kickanalysis.empty:
        return kickanalysis

    weekly_all = pd.concat(weekly_rows, ignore_index=True)
    schedules_all = pd.concat(schedule_rows, ignore_index=True)

    kickanalysis = add_actual_stats(kickanalysis, weekly_all)
    kickanalysis = add_games_played(kickanalysis, schedules_all, config.window_weeks)
    kickanalysis = add_fantasy_points(kickanalysis, weekly_all)
    kickanalysis = compute_expected_metrics(kickanalysis)

    kickanalysis = normalize_team_codes_forward(kickanalysis, "home_team", max(config.seasons))
    kickanalysis = normalize_team_codes_forward(kickanalysis, "away_team", max(config.seasons))
    return kickanalysis


def run_pipeline(config: ProjectConfig, output_path: Path | None = None) -> Path:
    logger.info("Building kicker analysis dataset.")
    analysis = build_kicker_analysis(config)
    ensure_dir(config.processed_dir)
    output_path = output_path or (config.processed_dir / "kicker_analysis.parquet")
    analysis.to_parquet(output_path, index=False)
    logger.info("Saved dataset to %s", output_path)
    return output_path
