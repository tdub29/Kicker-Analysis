"""YAML config loader for the retired pipeline. Moved here from src/kicker_analysis/.

Only `legacy/pipeline.py` and the old `scripts/run_pipeline.py` ever called it,
with `legacy/configs/base.yaml` (seasons 2005 to 2022). The current modules do
not read a config file: each states its own window as a named constant next to
the reason for it (`FIRST_SEASON = 2015` in fourth_down.py and model.py,
`FIRST_TEST = 2018` in backtest.py), so a season range cannot change without
the code that depends on it being in the same diff.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    seasons: List[int]
    min_week: int
    max_week: int
    window_weeks: int


DEFAULT_CONFIG_NAME = "base.yaml"


def load_config(config_path: Path | str, root_dir: Path | None = None) -> ProjectConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    root = Path(root_dir) if root_dir else Path(raw_config.get("root_dir", ".")).resolve()
    data_dir = root / raw_config.get("data_dir", "data")

    seasons = raw_config.get("seasons", [])
    if isinstance(seasons, dict) and "start" in seasons and "end" in seasons:
        seasons = list(range(int(seasons["start"]), int(seasons["end"]) + 1))
    elif isinstance(seasons, Iterable):
        seasons = [int(season) for season in seasons]
    else:
        raise ValueError("Config seasons must be a list or a start/end mapping.")

    return ProjectConfig(
        root_dir=root,
        data_dir=data_dir,
        raw_dir=data_dir / raw_config.get("raw_dir", "raw"),
        interim_dir=data_dir / raw_config.get("interim_dir", "interim"),
        processed_dir=data_dir / raw_config.get("processed_dir", "processed"),
        outputs_dir=data_dir / raw_config.get("outputs_dir", "outputs"),
        seasons=seasons,
        min_week=int(raw_config.get("min_week", 7)),
        max_week=int(raw_config.get("max_week", 20)),
        window_weeks=int(raw_config.get("window_weeks", 6)),
    )
