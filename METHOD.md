# Kicker forecasting method

## Decision and target

Forecast selected-league kicker points before lock, then compare the rostered
kicker with actually available alternatives. Historical roster/free-agent
snapshots are incomplete, so historical policy utility is unavailable. A random
pair diagnostic cannot satisfy the charter's real-policy promotion criterion.

Labels retain six made-distance buckets, attempt buckets, FG/PAT makes, misses
and blocks. Sleeper counts blocked FG/PAT as misses; configured deductions apply.
Aborted/safety mappings and unsupported distance/yard scoring fail explicitly.
The same scoring function constructs targets, historical point lags, simulation
and evaluation. Original forecasts are immutable records.

## Population and data quality

The denominator is played team-games. Raw PBP establishes attempted kicks.
Special-teams snaps establish activity only. A zero-attempt placekicker requires
consistent player position, prior career placekicking role and no other
regular placekicker taking that game's kicks. Punters and kickoff specialists
remain separate. This conditional research population does not establish who
was known to be available before kickoff. Unresolved games remain in the ledger.
Corrected files are stored in immutable generations. Consumers validate the
completed generation's hashes before fitting; an interrupted build cannot
publish a stable mixture of old and new tables.

Weather uses only prior-season venue climate where observed. Unknown wind is
not calm. Indoor certainty and retractable-roof type differ. Allianz Arena's
open pitch overrides the erroneous Munich dome label. Actual forecast weather
needs a historical forecast vintage before claiming pregame value.

## Coach decisions and long kicks

Real fourth-down FG/go/punt choices all remain in the denominator. The competitive
cohort uses WP .05-.95 and more than120 seconds remaining in the half. Context
strata include field position, yards to gain and score state. Prior-week coach
residuals shrink toward league context rates; support stays visible. Prior16-game
counts and season/coach/distance trend tables are retained for further research.
They are not automatically admitted as new winning predictors.

The long-kick menu uses yardline plus17 up to64 yards as a proxy for a feasible
kick, not actual attempted distance. Report its boundary sensitivity. Exact55+
and55-59 attempts/makes are separate facts. Do not condition declined-opportunity
denominators on the distance of a kick that never happened. End-of-half and
noncompetitive attempts remain raw outcomes but do not masquerade as the same
coaching decision context.

## Models

Production reference: linear regression from implied team total to selected-league
points. It is a transparent reference, not a demonstrated optimal policy. Research
comparators include corrected ridge, boosted opportunity models, minimal Poisson
GLMs, partially pooled conversion and a predeclared coach/range ablation.

The pooled distribution separates total opportunities, FG/PAT allocation,
attempted distance and conversion. A held-out final training season estimates
count dispersion. A binomial total supports underdispersion; negative binomial
supports overdispersion. Conditional binomial FG/PAT allocation is a simplification
whose residual dependence is reported. Distance makes and PAT makes receive
actual scoring. All fitting, scaling, imputation and shrinkage selection occur
inside chronological training. Sparse kicker conversion shrinks toward
training-population rates in research candidates.

Reference mean, experimental distribution mean and research ridge mean are
separate outputs. Quantiles and tail probabilities concern only the distribution.
Simulation is not a source of predictive skill. Brier uses mean squared binary
probability error, range0-1. CRPS is compared with a same-fold training empirical
outcome distribution; reliability by season is diagnostic with limited clusters.

## Evidence and claims

The fixed ladder and inner tuning rules are in the research protocol. Outer
seasons2018-2025 are expanding-window development comparisons, not untouched
confirmation. Report all candidates and losses. Season-level uncertainty remains
exploratory after adaptive reuse; small p-values cannot erase selection history.

The product is simplified conservatively because real-feasible-policy improvement
has not been demonstrated. No historical pair score numerically tests the
0.25-point real-policy promotion bar. Forecast gaps below0.5 points are practical
ties under an explicit design judgment, not statistical confidence.

Prospective confirmation requires immutable pre-lock inputs, scoring, model and
board identities plus actual feasible alternatives. At least17 observed weeks is
a collection minimum, not proof of independent information. Preserve every
published wrong-starter prediction in operational grading, then append corrections
and postmortems without revising the original forecast.
