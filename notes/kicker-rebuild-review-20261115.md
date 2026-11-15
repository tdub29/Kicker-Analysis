# Independent final kicker review, 2026-11-15

## Verdict: ACCEPT for scoped code and exploratory research

The original five core findings and two producer identity findings now have
concrete fixes or appropriately restricted
claims. The corrected historical comparison is acceptable as exploratory research
on the declared conditional population. The conservative SIMPLIFY disposition is
supported as a product decision under missing real-policy evidence, not a claim
that a .25-point feasible-policy test was run or failed.

The serving/source scope is now included. Shared JavaScript lookup correctly
checks season, requested week, normalized team and selected league scoring. All
kicker-point consumers inspected route through it, including the standalone
board, and there is no old projection cache. Market reference, experimental
distribution mean and research ridge are labeled separately. The explicit
cross-repo prepare/site-output/scoring route is present.

The producer/export review found two further issues. Both are now resolved in
the exact source reread below; this section preserves the review trail:

1. `predict_week.run` reads schedule/history and builds training/future rows
   before freezing `frozen_source`/`frozen_inputs`. A preparation or source edit
   between those reads and the hash can make the retained identity describe
   bytes different from those actually used. End-of-fit equality cannot detect
   that gap. Freeze input/source identities before those reads, or hash the exact
   bytes read, then retain the existing after-fit check. The final fix freezes
   source identity at run entry, captures schedule/prepared/roster bytes into
   memory, parses those exact bytes and binds their digests. After-fit checks
   compare on-disk bytes with those actually consumed. The roster is captured
   after retrieval and before parsing. Accepted.
2. `write_site` checks only presence/constancy of identity columns. It accepts a
   `scoring_spec` changed to a different league while keeping the old
   `scoring_id` and point predictions. It emits the changed scoring object, so
   the JavaScript gate can treat the wrong-scored predictions as matching the
   new league. Validate the scoring-spec digest against `scoring_id`. The
   `board_id` is also computed before final sort/rank columns and is never
   checked against record contents on CSV reuse/export. The final fix computes
   board identity after final sorting/ranking using CSV content excluding the
   identity column. Export recomputes the content digest, reconstructs the
   scoring object and checks its scoring digest. CSV reuse uses round-trip float
   parsing. The existing export fixture now rejects changed points and changed
   scoring. Accepted.

The final source read confirms these mechanisms are present, not merely stated
in a response. No model rerun was needed for these producer consistency fixes.
Final current-board generation and guarded publication remain the parent's
delivery responsibility. No live artifact or authenticated UI acceptance is
granted by this code review.

## Accepted dispositions from the first core pass

- Role logic now vetoes a stable P identity, requires prior same-team
  placekicking role for a verified zero, and retains other K special-team
  participants as unresolved rather than excluding the real placekicker.
  Badgley's 2020 cohort is restored to16 rows. The new role fixture covers a
  mislabeled punter and kickoff specialist alongside the real placekicker.
- The long menu is explicitly a yardline-plus17 proxy. Actual55+ and55-59
  attempts/makes remain separate; coach/distance trends, prior16-game support
  and37/38/39-yardline boundary sensitivity are retained. This does not identify
  causal coach preference or a literal exact-distance opportunity denominator.
- Raw implied team total now has unavailable fantasy-point MAE/RMSE. Its
  ranking diagnostic remains valid.
- Exact prior-training empirical-distribution CRPS now supplies a same-fold
  naive reference. Per-season probability bins are present. These provide
  descriptive diagnostics, not proof of fully conditional calibration or
  a practical ceiling strategy.
- `evaluation-final/run-inputs.json` freezes derived-table, schedule, source,
  protocol and scoring identities before fits. Its recorded source and protocol
  identities match the final evaluation record. The previous run is withdrawn.
- Current model/docs distinguish rebuilt research from old simulator claims.
  No old tail calibration is asserted for the new law.

## Corrected numerical evidence

Inspected4234 outer-test records across2018-2025, including39 verified-zero
outcomes. Historical available-player policy value remains unavailable. Six
historical unresolved team-games are not fabricated as zero.

Equal-season observed random-pair gains over the market reference:
ridge+0.08429, compound+0.01047, pooled GLM+0.05638, partially pooled GLM+0.06831,
coach/range bundle+0.06295 points. Ridge's unadjusted season t interval is
[-0.00647,+0.17506]. These are adaptive-history development descriptions, not
prospective superiority or actual feasible-lineup policy gains.

The naive empirical CRPS is2.56827. Candidate CRPS spans2.53153-2.53693, about
1.2%-1.4% descriptive improvement over that naive distribution. The coach/range
bundle's MAE3.61475 is slightly worse than the partially pooled candidate's
3.61426, while CRPS2.53153 is slightly better than2.53446. This bundled candidate
also changes distance allocation and tunes prior strength, so the comparison
does not isolate a causal coach feature effect. No numerical .25-point
real-policy pass/fail conclusion is supported.

## Limits and checks

Source review plus bounded existing-artifact reads; no model refits, browser
checks, broad product suite or deployment by reviewer. First-pass narrow checks
already verified raw FG/PAT accounting and chronological coach features. Final
pass inspected the new small tests and identity/scoring helper, without
duplicating the parent's reported tests. The final targeted reread included
source-at-entry capture, in-memory input parsing, after-fit consistency,
scoring/content digest validation and the two new rejection cases. No clean-CI
claim: managed-instruction
personal-path hygiene remains a reported existing failure.

Participation is reconstructed, not pregame starter availability. Published
wrong-starter records must stay in future operational grading. Aborted/safety
and unsupported scoring remain explicit limitations. Weather has no historical
forecast vintages. Small threshold probabilities retain Monte Carlo and model
uncertainty; displayed experimental labels are necessary. Original forecast
archives and unrelated dirty source/data must remain preserved. Authenticated
live UI and deployment are outside this review's verdict. Any material later
changes to reviewed model, scoring, population, serving or identity logic need
their own disposition; this acceptance is not permission to expand model scope.
