# Project Structure & Collaboration Guide

## Why this structure
This project follows a production-quality data science layout designed for reproducibility, clarity, and long-term collaboration.
Each top-level directory has a defined purpose, ownership, and workflow.

### Top-level directories (with purpose)
- **configs/**: Centralized configuration for data sources, season ranges, and pipeline settings. Keeps parameters versioned and reproducible.
- **data/**: All data artifacts, separated by lifecycle stage (`raw`, `interim`, `processed`, `outputs`).
- **docs/**: Project documentation, narrative decisions, and data dictionaries.
- **notebooks/**: Exploratory analysis and ad-hoc investigations. Not treated as production code.
- **reports/**: Figures, charts, or presentation-ready assets generated from analyses.
- **scripts/**: Entry points for CLI workflows (pipeline runs, data refreshes).
- **src/**: Source code for repeatable data ingestion, feature engineering, modeling, evaluation, and deployment utilities.
- **tests/**: Automated tests and data validations.

## Naming conventions
- **Python modules**: `snake_case.py` with descriptive names (e.g., `feature_engineering.py`).
- **Datasets**: Use `lower_snake_case` (e.g., `kicker_analysis.parquet`).
- **Config files**: `*.yaml`, with `base.yaml` as a default baseline.
- **Reports**: Prefix with date and brief title (e.g., `2024-09-10-week20-summary.pdf`).

## README recommendations
A strong README should contain:
1. **Project overview** — problem statement, data sources, and intended audience.
2. **Quickstart** — minimal steps to run the pipeline.
3. **Directory map** — link to this document with a short summary.
4. **Configuration** — how to customize seasons/weeks/data paths.
5. **Reproducibility** — requirements, pinned versions, and data notes.
6. **Contributing** — style guidelines, branch strategy, and testing.

## Configuration management
- Use YAML in `configs/` for any environment-specific settings.
- Keep defaults in `configs/base.yaml`, then allow overrides via CLI or environment variables.
- Store secrets (if introduced later) outside of git and reference them in a separate `configs/local.yaml`.

## Reproducibility & collaboration
- Keep raw data immutable in `data/raw`.
- Only write derived data to `data/interim` or `data/processed`.
- Capture dataset creation logic in `src/` and run it via `scripts/`.
- Keep notebooks clean and reproducible; export core logic to `src/` when it becomes stable.
