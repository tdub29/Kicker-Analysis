#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from kicker_analysis import load_config, run_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kicker analysis pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to configuration YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for processed dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config, args.output)


if __name__ == "__main__":
    main()
