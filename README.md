# Weekly NFL kicker forecasts

This project prepares kicker outcomes and pregame features, then compares small
forecast models chronologically. The app currently uses a transparent market
reference. No practical advantage on real available-player lineup decisions has
been established.

## Current model decision

**Simplify the recommendation; retain richer models as experimental research.**
The reference regresses selected-league kicker points on implied team total.
A separate pooled opportunity/conversion GLM supplies experimental ranges and
threshold probabilities. Its mean is labeled separately from reference points.
The ridge and coach/range alternatives do not control the production ordering.

The corrected research includes full team-game participation records, verified
placekicker zero-opportunity outcomes, missed/blocked scoring, PAT conversion,
attempted-distance accounting, train-only count dispersion and probability
scores. Kickoff specialists and punters are not placekicker zeroes. Missing
weather stays unknown. Historical free-agent availability remains unavailable.

Read [method](METHOD.md), [research charter](notes/kicker-decision-charter-20261115.md)
and [frozen candidate protocol](notes/kicker-rebuild-protocol-20261115.md).

## Run

Install an editable checkout: `python -m pip install -e ".[dev]"`.

1. `python src/kicker_analysis/prepare.py --refresh` builds corrected inputs in
   ignored `data/processed/rebuilt/` and records quality, provenance, coaching
   choice menus, exact long attempts and trend/boundary sensitivity tables.
2. `python src/kicker_analysis/predict_week.py --scoring-settings league.json
   --site-out path/to/site/data/kickers.js` writes a uniquely identified new
   board and the explicitly selected app payload. `league.json` is a Sleeper
   league response or scoring-settings object. Unsupported active kicker scoring
   fails explicitly. Original `reports/week_*` forecasts remain unchanged.
3. `python -m kicker_analysis.evaluate_rebuild --processed-dir data/processed/rebuilt
   --scoring-settings league.json --out-dir data/evaluations/new-run` runs the
   finite exploratory 2018-2025 comparison. Use a new directory per run.
4. `python -m pytest -q` runs repository checks. Managed local instruction files
   contain machine-specific paths and trigger the existing hygiene assertion;
   report that failure separately from model correctness checks.

Set `OMP_NUM_THREADS=1` to avoid LightGBM thread contention. No paid data or
new service is required. Historical input data comes from nflverse and PFR snap
counts, with exact source and derived-input identities retained per run.

## Limitations

Historical seasons 2018-2025 were repeatedly inspected and are development data.
Random observed-kicker pairs are diagnostic, not actual waiver choices. Closing
lines evaluate near-kickoff context, not an earlier weekly market snapshot.
Special-teams participation is postgame evidence, not archived starter availability.
Unresolved roles and missing games remain explicit. Coach effects are conditional
associations, and yardline plus17 is an opportunity-distance proxy. A new
prospective forecast/policy record must establish any claimed lineup value.
