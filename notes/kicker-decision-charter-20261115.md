# Kicker decision research charter, 2026-11-15

Status: revision 2 after the MAX book-workflow repair. Model edits are paused by Trevor. Earlier scoring/simulator edits remain unaccepted; no new model edits, fits, backtests or publication are authorized by this scout repair. Prior 2018-2025 results are development evidence, not untouched confirmation. This charter does not assert a predictive edge.

## Method and MAX disposition
`predictive-modeling` supplies the sequence: Provost framing, Huyen data/features/evaluation, then domain-specific development and impact evaluation. `sports-modeling-doctrine` supplements that sequence. The earlier dismissal of the router as over-routing is withdrawn. Full local book workflows, not just frontmatter, were read during the repaired scout.

The fresh `kicker-decision-forecasting` verifier failed its first primary-source fetch with `Unknown tool`. No retry or source-clearance claim follows. The draft is CUT, not installed, and is not an implementation authority. Existing book/domain skills cover the methodological requirements below. Platform semantics, historical availability and forecast-weather vintages remain primary-evidence tasks; a new skill cannot manufacture them. Keep the draft and incomplete review only as rejected research history.

The repaired body/reference search, full-read inventory and source limitations are in `.agents/artifacts/kicker-redesign-20261115/scout/book-frameworks-rescout.md` and `fresh-source-verification.md`. No model work resumes in this revision.

## Question and action
For a league that starts K, does a forecast help choose between the rostered kicker and actually available replacements, under that league's scoring, better than keeping the current kicker, sourced Sleeper points, or highest implied team total?

Row: one identified pregame kicker/team/game/scoring-policy forecast. Primary decision row: one league/week/rostered-kicker/available-pool snapshot. Two team rows in a game remain grouped. Preserve published wrong-starter forecasts in operational evaluation.

Decision time: a timestamped snapshot before the first relevant player's lock. Historical closing-line experiments concern near-kickoff decisions only. No historical forecast-weather, ownership or roster availability claims without as-of snapshots. Historical reconstruction and prospective publication are separate clocks.

Target: realized selected-league kicker points from raw event outcomes, including verified active zero-opportunity games. Inactive/replacement/emergency/unknown cases stay explicit. Current rosters never prove historical availability. Raw event categories remain available; blocked FG/PAT count as missed in Sleeper (official support scoring-options page). Aborted/safety cases require explicit disposition and cannot silently become misses.

### Provost label contract and Huyen sampling contract
Before the next model edit, record the target contract: stable kicker/team/game identity, scoring version, decision timestamp, game outcome window, label source, label-resolution timestamp and correction policy. Keep raw categories and later stat corrections; do not revise an original forecast to match a corrected label.

Build two explicit populations. The forecast population concerns identified pregame kickers and complete scored outcomes. The policy population concerns league decisions with timestamped roster/free-agent choices. Do not infer either population from the presence of an attempted kick. Adjudicate active-zero, inactive, emergency/replacement and unknown cases with evidence and a reason. Unresolved is not zero or an automatic exclusion.

Report conditional forecast research on the declared verified-starter cohort separately from operational accuracy on all published forecasts. Published wrong-starter recommendations remain operational errors. Report excluded/unresolved counts and denominators by season. Do not rebalance to an artificial made/missed/zero distribution. If historical feasible-player snapshots cannot be recovered, historical policy-value claims are unavailable; prospective capture is required.

### McGilvray quality baseline and upstream prevention
Before fitting, create one compact rule catalog with population/denominator, pass/fail definition, source lineage, owner, observed defect count and decision impact. Prior audit counts are leads to reconcile, not new measurements. The PC agent owns acquisition and repair; unavailable evidence stays an explicit limitation rather than a request for invented labels.

| Critical defect | Root-cause work and prevention point | Acceptance requirement |
|---|---|---|
| Missing kicker games | Trace schedule/starter evidence through attempt-maker filtering and joins. Classify missing records before insertion or exclusion. | Full intended team-game denominator; classified zero/inactive/replacement/unknown counts and evidence. |
| Wrong fantasy labels | Trace raw FG/PAT result fields through aggregation and selected-league scoring. Preserve blocked/aborted/safety distinctions. | One verified scoring contract for labels, lags, simulations and grading; unresolved mappings fail explicitly. |
| Missing wind treated as calm | Trace station/venue join, source missingness and downstream imputation. Preserve unknown separately from measured zero. | Missingness by source/season/venue; train-only imputation or unavailable output; historical forecast claims require forecast vintage. |
| Incorrect venue/roof | Trace authoritative game location and neutral-site mappings before feature generation. | Versioned venue evidence and consistent shared transformations; no one-player ranking patch. |
| Unavailable or stale policy inputs | Trace publication, retrieval and effective timestamps for quotes, projections, eligibility and scoring. | Backward as-of joins, join cardinality/unmatched counts and declared unavailable-baseline handling. |
| Conflicting published outputs | Trace direct-point ordering, simulated tails and final payload identities. | Explicit output semantics and immutable forecast/model/input/cutoff identity; no silent archive overwrite. |

Correction and prevention are separate deliverables. Confirm where each defect originates before repairing its propagation. A final CSV patch is not a root-cause remedy. Prioritize scoring/population harm before adding features.

## Locked reference policies and candidate ladder
1. Retain rostered kicker; constant prior-training mean; training-only mean versus implied-total regression.
2. Highest implied team total and sourced Sleeper projections published by the same decision cutoff, on identical available players. Current responses cannot reconstruct historical quotes, settings or ownership.
3. A small interpretable opportunity/conversion GLM candidate, with complete pooling as its reference and justified partial pooling as a candidate; corrected ridge and the corrected existing simulator remain comparators. New features or model families require a dated hypothesis and fixed, small candidate grid. Read prior failed red-zone, drive and leg experiments first. No McPherson rank target.
All policies share target, eligibility, missing-data fallback and cutoff. Missing market/projection policies retain current kicker; report coverage and matched comparisons, never silently drop their hard cases.

### Tetlock outside view before matchup adjustment
Record training-only opportunity, conversion and score base rates with their comparison population before considering the individual matchup. Name one plausible alternative reference class in advance, and choose it inside training only. Quantify subsequent offense/opponent/weather adjustments against that reference. Opponent rank 31 or a hot offense is evidence to assess with exposure, schedule strength and uncertainty, not a multiplier or required ordering. Do not select the reference class after seeing McPherson's position.

### Statistical-modeling candidate contract
Use count likelihoods for opportunities and binomial conversion for makes conditional on attempts. Points remain the sum under selected scoring; a direct continuous point model is a distinct comparator. Sparse kicker/team effects require pooled or partially pooled estimates, not unrestricted small-sample coefficients.

Start with pooled rates and a minimal legal-feature Poisson opportunity GLM. Compare a partially pooled version only through a predeclared grouping/shrinkage specification. Diagnose dispersion, zeros and FG/PAT dependence on training data. Negative binomial is a candidate for supported overdispersion, not a remedy for underdispersion; hurdle/zero inflation needs evidence of a separate zero process. A Gaussian mixed model is not a count GLM. Retain complete pooling if it wins.

Before evaluation, freeze the exact response, feature formula, exposure/offset if applicable, groups, shrinkage estimator, any priors and finite tuning grid. This methodological ladder is not that executable specification; no broad family search is authorized. Existing dependencies and simple penalized/empirical-Bayes approaches come first. A Bayesian alternative needs an actual advantage plus prior sensitivity, convergence and posterior predictive evidence.

Huyen feature gate: diagnose missingness and source availability before deletion/imputation. Split chronologically first; learn all imputers/transforms and feature selection inside training. Compare feature coverage/ranges by fold, including early-season and replacement kickers. An observed-weather historical analysis must be labeled retrospective; it cannot establish real pregame weather value. Controlled perturbation/directional diagnostics do not force conditional coefficient signs or a desired rank.

## Metrics and practical bar
Primary: paired realized points gained per feasible league decision versus each usable incumbent, equal weight per week, then equal season weight. Report transaction counts and a predeclared 0.25-point cost sensitivity per change, plus 0 and 0.5; these are modeling judgments, not a FAAB conversion. A change must exceed 0.5 estimated points before costs. Smaller differences retain the current kicker. This practical threshold is a design judgment, not demonstrated statistical confidence. Define uncertainty in the paired expected utility gain before tuning any further abstention rule; weekly outcome variation and Monte Carlo error are different quantities.

Provost decision evaluation: report overall policy value, matched-coverage value, switch frequency and disagreement cases. Compare all policies on the same feasible set and cutoff, including cases where they agree. Hindsight-best realized utility is an oracle ceiling; loss to it is diagnostic realized opportunity loss, not loss to an optimal expected-value policy. Expected-points utility does not establish matchup-win value; that objective would need other lineup distributions and dependence assumptions.

Secondary: MAE/RMSE, discrete CRPS, threshold Brier scores, P20-to-P90 interval coverage and width, zero-opportunity accuracy and wrong-starter frequency, by season and declared slices. P20-to-P90 is a nominal 70% interval; discrete outcomes can prevent exact nominal coverage. Tail agreement alone does not certify individual probabilities. Report both direct-point and distribution means until a coherent candidate is accepted.

Tetlock scorekeeping: binary threshold Brier is `mean((p - I(Y >= t))^2)`, range 0-1; constant 0.5 forecasts score 0.25. Do not import the two-class summed convention's 0-2 scale or coin-flip 0.5. Compare calibration and resolution separately against training-only event-rate forecasts on the same questions. Freeze event thresholds, score resolution, input/scoring identities and the board before lock; append corrections. Show uncertainty/ties, not six-decimal implied certainty. Postmortem successes and failures, including wins caused by offsetting model defects.

## Validation and acceptance
Historical: outer expanding seasons 2018-2025; inner chronological folds alone choose configuration and imputation. These previously inspected seasons support development comparisons only. Bootstrap paired decision differences by season and report independent cluster count and equal-season intervals. Random observed pairs remain a secondary diagnostic, never the primary product claim.

Aronson selection-bias register: record the full adaptive search, not just the final grid. Include old red-zone/drive/leg failures, weather/scoring repairs, feature windows, calibrations, distribution choices, thresholds and reported slices. Record unknown historical trial counts as unknown. Every inspected 2018-2025 result is exploratory; nested reruns cannot undo previous human selection. The chosen winner's unadjusted interval or p-value does not establish superiority. Family-level inference needs a reconstructible search family and dependence-respecting comparison; otherwise keep historical inference exploratory. Do not transfer financial detrending or unrestricted permutations to dependent kicker/game/week data.
Practical promotion bar: corrected candidate gains at least 0.25 realized points per eligible decision net of the stated cost against strongest usable simple reference, wins a majority of outer seasons, has no unresolved leakage/population/scoring errors or worse distribution calibration, and has independent technical review. Historical tuning cannot establish this as prospective.
Prospective confirmation: freeze recipe and publish immutable exact input/scoring/cutoff/payload identity before lock. Record every eligible decision over at least 17 distinct NFL weeks; no stopping when results first look favorable. Seventeen weeks is a minimum collection rule, not proof of adequate independent information. Freeze the interval/resampling design after the actual dependence structure and available seasons are known, before prospective scoring starts. Require a positive lower 95% bound against the strongest reference for a superiority claim; inadequate independent clusters mean inconclusive. Do not claim season generalization from a single season. Keep/Simplify/Retire may honestly choose a transparent baseline while research remains experimental.

## Order after the scout repair
1. Keep model edits paused. MAX draft is cut; no unverified guidance is installed.
2. Before resumed implementation, finish target/gray-zone contracts, the DQ cause/prevention catalog and source-availability register. Reconcile raw-to-processed scoring/population defects first.
3. Freeze executable baseline/candidate specifications and the complete experiment ledger before new fits. Apply comparable development effort and preserve losing runs.
4. Run authorized chronological development comparisons and feasible-policy evaluation only where availability evidence supports it. Choose keep/simplify/retire without forcing novelty or a preferred player order.
5. Complete correctness/serving review, then freeze prospective scorekeeping. No superiority claim before prospective evidence meets the declared bar.

## Book-workflow trace
Paths below are in the workspace's loaded `.claude/skills/` tree. These are full local workflow reads, not claims of reading every original book in full.

- Provost: `references/provost-target-label-design.md`, `provost-expected-value-evaluation.md`, `provost-leakage-audit.md` in `provost-data-science-for-business`: target/gray-zone contract, same-cutoff feasible utility and leakage register.
- Huyen: `references/training-data.md`, `feature-engineering.md`, `offline-evaluation.md` in `huyen-ml-systems-design`: natural population, missingness, train-only transforms, baselines, slices and policy disagreements.
- McGilvray: `references/dq-assessment-baseline.md`, `dq-root-cause-remediation.md` in `mcgilvray-data-quality-project`: measured defect denominators, cause confirmation, correction plus prevention.
- Tetlock: `references/tetlock-outside-then-inside-view.md`, `tetlock-calibration-brier-granularity.md`, `tetlock-keep-score-postmortem.md` in `tetlock-superforecasting`: reference classes, explicit scoring convention, frozen predictions and symmetric postmortems.
- Statistical modeling: `statistical-modeling/SKILL.md`, `references/sports_glm_guide.md`, `bayesian_statistics.md`, `diagnostics_checklist.md`: likelihood support, sparse-group pooling, dispersion and dependence diagnostics.
- Aronson: `evidence-based-technical-analysis/SKILL.md`: adaptive-search inventory, benchmark-relative comparisons and no unadjusted historical winner claim.

## Required repairs and evidence
- Shared explicit scoring for labels, point lags, simulations and grading, including misses and blocked attempts.
- Evidence-classified complete team-game participation ledger; verified zero rows and emergency/wrong-starter cases; known unresolved coverage quantified.
- PAT conversion, categorical attempt distance/makes, training-only dispersion/dependence diagnostics and held-out distribution scores.
- Missing wind preserved as unknown; candidate imputation learned inside folds; authoritative neutral-site/roof identity.
- Serving rejects wrong season/week/team/scoring, explicitly distinguishes ridge and simulation, and explains material ranking reasons.
- Explicit cross-repo output path and immutable version/input/cutoff identity; original pregame archives untouched.
- Corrected matched baseline evaluation, realistic available-player decisions, technical independent review, decision record and Muse closure only after full evidence.

## Stop and non-goals
No forced intuitive ranking, coefficient-sign rule, invented novel effect, archive rewrite, current weather substituted for historical forecast, fabricated participation/ownership or superiority claim from selected seasons. No unbounded hyperparameter search, new service or unrelated app change. Keep goal open while required evidence or repairs remain incomplete. Human-only evidence needs owner/exact input/next action in the checkpoint, not assumed approval.
