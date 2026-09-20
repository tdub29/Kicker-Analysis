from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import pandas as pd

try:
    import nfl_data_py as nfl
except ImportError as exc:  # pragma: no cover - helpful error for users
    raise ImportError(
        "Missing dependency nfl_data_py. Install with `pip install nfl_data_py`."
    ) from exc

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def load_weekly_kicking(seasons: tuple[int, ...]) -> pd.DataFrame:
    logger.info("Loading weekly stats for seasons: %s", seasons)
    weekly = nfl.import_weekly_data(list(seasons))
    weekly = weekly[weekly["position"] == "K"].copy()
    weekly = weekly.rename(columns={"season": "season", "week": "week", "recent_team": "team"})
    return weekly


@lru_cache(maxsize=None)
def load_schedules(seasons: tuple[int, ...]) -> pd.DataFrame:
    logger.info("Loading schedules for seasons: %s", seasons)
    return nfl.import_schedules(list(seasons))


@lru_cache(maxsize=None)
def load_pbp(seasons: tuple[int, ...]) -> pd.DataFrame:
    logger.info("Loading play-by-play data for seasons: %s", seasons)
    return nfl.import_pbp_data(list(seasons), downcast=True)


def get_season_data(season: int) -> dict[str, pd.DataFrame]:
    seasons = (season,)
    return {
        "weekly": load_weekly_kicking(seasons),
        "schedules": load_schedules(seasons),
        "pbp": load_pbp(seasons),
    }
