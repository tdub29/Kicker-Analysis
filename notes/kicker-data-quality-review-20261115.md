# Additional kicker data-quality review, 2026-11-15

Three independent readers examined outcomes/participation, coach/range data and
feature timing/joins. The review found concrete additional defects rather than
new reasons to expand the model search.

## Corrections

- Roster filtering treated initial/surname and full names differently. All 56
  originally inserted zero rows used full names. A released `E.McPherson` was
  dropped while `Evan McPherson` survived. The shared lookup now uses stable GSIS
  identity where present and one normalized-name fallback; both formats follow
  the same team/release rule.
- Latest-same-team history rejected six veteran zero games after signings or
  temporary injury replacements. Prior career placekicking plus current K
  participation and no competing regular attempter resolves those cases. Punter,
  kickoff-specialist and ambiguous-role vetoes remain. The 2018-2025 conditional
  cohort is now 4,240 of 4,254 team-games, with 45 verified zero games. Fourteen
  remain explicitly outside the cohort; historical pregame availability is not
  inferred from those postgame role records.
- Seven botched FG/punt formations were mislabeled as voluntary go choices.
  All aborted snaps are now excluded from the declared clean-choice cohort;
  offensive aborted snaps are not relabeled as kicks. One genuine blocked FG
  followed by a recovery pass was dropped by mutually exclusive event flags.
  Actual kick flags now take precedence over recovery pass/rush flags.
- The narrow fourth-down table omitted 1,060 verified zero-choice team-games.
  It now preserves their zero counts and undefined rates so count lags do not
  become conditional on having a choice.
- Live snap-count and player-ID files inherited permanent caching. Unsettled
  snap seasons now use the existing live expiry policy; identity data refreshes
  daily. Coverage and source-fetch timing distinguish current data from missing
  or stale participation evidence.
- Climate support counted missing weather as observations. Prior game counts
  and nonmissing temperature/wind counts are now separate. Phoenix November's
  32 prior games with zero weather measurements no longer imply 32 observations.
  Prior-season mean weighting was preserved.
- Sequential prepared-file writes could leave stable files from mixed builds.
  Preparation now writes an immutable generation and atomically publishes its
  completion manifest last. Consumers validate exact generation hashes and
  consume those bytes. Earlier complete generations survive interrupted builds.

## Evidence and limits

All 25,911 cached regular-season FG/PAT events reconciled to processed attempt
totals in the review. No new scoring arithmetic defect, lag alignment mismatch,
duplicate join expansion or reversed game chronology was found. One source
disagreement was retained: Austin Seibert in 2019 week 15 has 3 PATs in PBP and 4 in the
kicking release; the play descriptions and score support the current PBP count of 3.
No target was changed merely to force source parity.

Fixtures cover role/roster formats, veteran transfer, aborted/recovery plays,
zero-decision counts, missing-weather support and generation integrity. Eleven
focused checks and the fourth-down self-check passed. Corrected chronological
research was rerun under the same finite candidate grid; results are in the
updated rebuild summary and evidence. Earlier runs remain superseded history.

Historical roster/free-agent and forecast-weather vintages remain unavailable.
The conservative simplify decision does not claim a demonstrated lineup edge.
Authenticated live UI remains separate from source/artifact review.

Independent consolidated review accepted these corrections. Every evaluated data
component matches the current prepared data. Later quality and source-manifest
metadata differ, so the frozen run retains its original generation identities;
this does not claim the whole current snapshot was consumed by that run.
