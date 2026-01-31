from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

TEAM_RENAMES_BY_SEASON = {
    "LAC": {"before": 2016, "from": "LAC", "to": "SD"},
    "LA": {"before": 2017, "from": "LA", "to": "STL"},
    "LV": {"before": 2020, "from": "LV", "to": "OAK"},
}

TEAM_RENAMES_AFTER = {
    "SD": {"after": 2016, "from": "SD", "to": "LAC"},
    "STL": {"after": 2017, "from": "STL", "to": "LA"},
    "OAK": {"after": 2020, "from": "OAK", "to": "LV"},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_team_codes(df: pd.DataFrame, column: str, season: int) -> pd.DataFrame:
    """Normalize team codes to match historical franchise names for a season."""
    updated = df.copy()
    for _, rule in TEAM_RENAMES_BY_SEASON.items():
        if season < rule["before"]:
            updated[column] = updated[column].replace(rule["from"], rule["to"])
    return updated


def normalize_team_codes_forward(df: pd.DataFrame, column: str, season: int) -> pd.DataFrame:
    """Normalize historical team codes forward to modern names for a season."""
    updated = df.copy()
    for _, rule in TEAM_RENAMES_AFTER.items():
        if season > rule["after"]:
            updated[column] = updated[column].replace(rule["from"], rule["to"])
    return updated


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator).where(denominator != 0, 0)


def select_numeric_columns(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    numeric_cols = [col for col in df.select_dtypes(include="number").columns if col not in exclude]
    return df[numeric_cols]
