# Kicker recommendation: simplify, 2026-11-15

Use a transparent implied-team-total reference for production expected points.
Keep corrected ridge, pooled/count, partial-pooling and coach/range models as
experimental research. This is a conservative product decision, not proof that
the reference predicts better than every richer model.

## Corrected population and evaluation

The historical 2018-2025 denominator is 4,254 played team-games. The conditional
placekicker cohort contains 4,234 team-games, including 39 evidence-classified
active zero-attempt games. Twenty team-games remain outside that cohort: seven
emergency/unknown-role, seven special-teams-only with unresolved placekicker
role, and six without verified kicker identity/participation. Those cases remain
in the ledger. Current roster status was not used as historical availability.

Independent review caught and corrected an earlier preparation error: K-labeled
special-teams snaps included punters and kickoff specialists. The first research
run is withdrawn. The corrected rule requires prior placekicking role and no
competing regular attempter for a zero row; an observed primary placekicker is
not excluded just because a kickoff specialist also played.

The fixed ladder uses eight expanding outer seasons, two chronological inner
seasons, matched selected-league scoring and training-only transformations.
The prior empirical outcome distribution is the full-distribution reference.
The binary-event reference uses training base rates. Exact run inputs and source
identities were captured before fitting and checked unchanged afterward.

Results are in [summary](kicker-rebuild-summary-20261115.csv),
[season comparisons](kicker-rebuild-comparisons-20261115.csv) and
[reproduction evidence](kicker-rebuild-evidence-20261115.json).

| Method | MAE, points | Observed-pair gain over market, points | CRPS |
|---|---:|---:|---:|
| Market reference | 3.619 | reference | not a distribution |
| Corrected ridge | 3.600 | 0.084 | not a distribution |
| Pooled GLM | 3.614 | 0.056 | 2.537 |
| Partially pooled conversion GLM | 3.614 | 0.068 | 2.534 |
| Coach/range GLM | 3.615 | 0.063 | 2.532 |
| Training empirical distribution | not a point forecast here | not a ranking policy | 2.568 |

The ridge's season-level 95% interval for the observed-pair gain is -0.006 to
0.175 points. These intervals and p-values are exploratory after repeated
inspection of 2018-2025; they do not remove adaptive selection bias. Coach/range
features improve descriptive distribution scores slightly, but do not improve
MAE over the pooled candidate. No conditional-calibration or practical-policy
superiority claim follows from that small difference.

Historical feasible rosters/free-agent sets are unavailable. Random observed
pairs cannot test the charter's 0.25-point net gain criterion on real decisions.
That criterion is untested, not numerically failed. Do not invent historical
availability or call this a validated waiver recommendation.

## Coach and long-range data

Competitive fourth-down FG/go/punt menus preserve all three choices and their
support. Rates adjust for field position, yards to gain and score state, with
prior-week shrinkage by coach identity. Season/coach/distance trend tables and
prior 16-game counts are retained. Actual 55+ and 55-59 attempts/makes remain
separate from the yardline-plus-17 opportunity-distance proxy. Boundary
sensitivity is reported instead of conditioning declined opportunities on a
future attempted distance. Recency and context effects remain research.

## Product behavior and remaining limits

Reference points, distribution mean and research ridge mean are separate.
Floor, ceiling and 15+ probabilities are labeled experimental. Differences below
0.5 reference points are practical ties under a declared judgment. Season,
requested week, normalized team and scoring must match before the app uses a
forecast. Unsupported yard/miss-distance scoring stays explicitly unavailable.
No legacy board is silently exported as a corrected current forecast.

Forecasts bind model/source/input/scoring/cutoff/board identities and receive new
archive paths. Original pregame forecasts and old research reports remain intact.
Inferred current kickers still require an availability check; prior participation
is not a verified pregame depth chart. Missing weather stays unknown; historical
forecast-weather vintages are absent. Authenticated live UI evidence is separate
from source and deployed artifact acceptance.

Prospective superiority requires new frozen pre-lock forecasts, actual feasible
choices and dependence-aware evidence under the charter. No model refinement
today manufactures that future record.
