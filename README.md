# Kicker Analysis: NFL Fantasy Forecasting

This project builds an analytics pipeline to forecast NFL kicker outcomes, including expected field goal attempts, fantasy points, and market-style thresholds (e.g., 1.5 made field goals). The workflow merges rolling team performance, opponent defensive context, and schedule information to produce a dataset for weekly decision-making and model development.

The repo includes a Python replica of the original R workflow with modular ingestion, feature engineering, modeling, and evaluation.

## Project goals
- Build a rolling-window dataset for kicker performance across seasons.
- Generate opponent-adjusted expectations for weekly attempt volume and fantasy output.
- Create a clean dataset for modeling, evaluation, and downstream deployment.
- Provide a structure that supports ongoing contributions and extension.

## Quickstart
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/base.yaml
```
The processed dataset is saved to `data/processed/kicker_analysis.parquet` by default.

## Folder structure
```
configs/        # Versioned pipeline configuration
 data/          # Raw, interim, processed, and output artifacts
 docs/          # Audience-facing documentation and data dictionary
 notebooks/     # Exploratory analysis (kept separate from pipeline code)
 reports/       # Generated charts and summaries
 scripts/       # CLI entry points
 src/           # Reusable pipeline code (Python + legacy R)
 tests/         # Validation and unit tests
```
See `docs/project_overview.md` for full justifications and conventions.【F:docs/project_overview.md†L1-L52】

## Data sources
- **nfl_data_py** (Python): used for weekly stats, schedules, and play-by-play data.
- **nflfastR** (R): legacy workflow stored for reference and parity checks.

## Configuration
All pipeline parameters live in `configs/base.yaml`. Customize seasons, week ranges, and output directories there for reproducibility.

## Python pipeline (replica of R workflow)
The Python modules mirror the original R logic:
- **Data ingestion**: `src/kicker_analysis/data_ingestion.py`
- **Feature engineering**: `src/kicker_analysis/feature_engineering.py`
- **Modeling and evaluation**: `src/kicker_analysis/modeling.py`, `src/kicker_analysis/evaluation.py`
- **Pipeline orchestration**: `src/kicker_analysis/pipeline.py`

## Naming conventions
- Files: `snake_case.py`
- Datasets: lowercase, `snake_case` with format or scope in the name (e.g., `kicker_analysis.parquet`)
- Configs: YAML files, `base.yaml` as defaults

## How to contribute
- Keep new feature logic in `src/` and wire it through `scripts/`.
- Add or update tests in `tests/` when you change pipeline logic.
- Use configuration files rather than hard-coded parameters.

## Legacy R workflow
The original R script has been preserved at `src/r/kicker_pipeline.R` for reference and cross-validation with the new Python implementation.
