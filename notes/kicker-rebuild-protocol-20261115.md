# Kicker rebuild executable research protocol

Frozen before new model fits on 2026-11-15. Trevor resumed implementation after
the book scout repair and specifically requested coach go/kick tendencies and
55+ yard willingness. All 2018-2025 comparisons remain exploratory.

## Data and target

Raw PBP 2015 onward supplies FG/PAT outcomes, six attempted-distance buckets,
55+ and 55-59 attempts/makes, fourth-down decisions and game context. PFR special
teams snaps plus stable PFR/GSIS crosswalk restore verified active-zero kicker
participants. Current player status/team is never historical availability.
Model fitting uses unique regular-K participants; multiple-K games, emergencies
and unresolved games stay in the ledger and team lags. This conditional cohort
is not a reconstruction of historical pregame starter or free-agent availability.

Selected-league scoring is applied before any point lag. Primary research uses
Phi Kap's snapshotted scoring. Unsupported outcome/scoring combinations fail,
rather than silently awarding zero. Preserve original forecasts and raw input
identities. Label resolution is the recorded retrieved official outcome snapshot;
later corrections append a new version.

Historical market inputs are closing lines. They evaluate near-kickoff research,
not Tuesday recommendations. No forecast-weather vintages are supplied. Missing
historical climate remains unknown, with training-fold imputation and an
indicator. Weather-free models form the primary interpretable references.

## Finite candidate grid

References: prior-training mean; linear points regression on implied team total;
ranking by implied total; selected-league incumbent projection where same-cutoff
data exists. Historical free-agent policy value is unavailable without archived
feasible choices. Do not manufacture those cases from random pairs.

1. Corrected existing ridge, alpha100, with explicit unknown-wind indicator.
2. Corrected existing boosted count model as a research comparator.
3. Poisson opportunity GLMs, alpha1, on implied team total, opponent implied
   total, home and known venue type; pooled attempted-distance mix and pooled
   distance-conditional conversion. Counts are FG/PAT attempts per game, no
   exposure offset because every row is one complete game for the declared cohort.
4. Same GLMs with partially pooled kicker conversion in each distance bucket
   and PAT conversion. Beta-binomial empirical priors use training population
   rates; prior strength20 or100 is selected on inner chronological seasons only.
5. One coach/range ablation of candidate4: add context-adjusted prior coach
   FG/go deviations, prior long-kick menu rate and its support, and kicker prior
   55+ attempt/make support. A regularized multinomial distance model may allocate
   attempted distances from the same pregame features. Compare this bundle as
   one candidate, not a search over every feature subset.

All medians/scaling and conversion priors fit in training. Use last two eligible
training seasons as chronological inner folds; select by equal-season MAE as
development loss when feasible-policy records are absent. Refit on all prior
seasons, then score each outer season2018-2025. No 2026 result enters that grid.
No new family or tuning dimension without an appended hypothesis before fitting.

## Distribution and diagnostics

Separate opportunity totals from FG/PAT allocation, attempted distance and
conversion. Estimate total-opportunity dispersion on a held-out final training
season. Underdispersed totals use a mean-preserving binomial; overdispersed totals
use negative binomial; unit dispersion uses Poisson. Conditional binomial FG/PAT
allocation induces dependence, whose adequacy is checked against residual
covariance; this is an explicit simplifying assumption, not a full drive model.
Report raw fitted dispersion and any numerical bounds. Do not tune to held-out
tail agreement. Posterior conversion draws distinguish parameter uncertainty
from ordinary opportunity variation; fixed seed0,4000 draws for development.

Every made kick belongs to one attempt and distance bucket. Score PAT makes,
not PAT attempts. Export exact distribution mean separately from Monte Carlo
mean; probabilities, intervals and CRPS concern that distribution. A separate
direct mean must be labeled. Compare binary threshold Brier, discrete CRPS,
P20-P90 coverage/width, MAE and calibration against training-only event rates.

## Coach and range preparation

Main comparable menu: real fourth downs, WP .05-.95, more than120 half-seconds,
implied kick distances17-64 yards. FG/go/punt remain in the denominator. Distance
bands <=39,40-49,50-54,55-59,60-64; yards to gain1,2-3,4-6,7+; score state
trailing/tied/leading. League context expectations and coach residuals exclude
the forecast's entire week. Coach deviations shrink with20 equivalent prior
decisions; support is always retained. Coaching changes reset the coach identity,
not the franchise's old rate. Assignments are reconstructed, not vintage-verified.

Retain all long attempts and their makes/misses separately from the competitive
choice cohort. End-of-half, desperation, weather and field position can change
the menu, so make percentage or longest made FG cannot stand for willingness.
No-opportunity games contribute zero counts and no denominator to a rate, not
an invented preference. Review evidence that whole-history coach rates miss
era/recency changes before treating the ablation as a causal coach effect.

## Decision and stop

Keep the charter's .5 expected-point practical switch rule and 0/.25/.5-point
cost sensitivity as judgments. No confident switch from a small noisy gap.
Apply the .25-point net promotion bar only to real feasible decisions; random
pairs cannot satisfy it. If historical availability is absent or richer models
fail the bar, simplify the product recommendation and retain research forecasts
as experimental. Sealed prospective boards and actual policy snapshots must
provide future evidence; no historical retuning establishes superiority.
