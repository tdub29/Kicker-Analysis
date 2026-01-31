# Data Dictionary (Kicker Analysis)

This dictionary describes the core fields produced by the Python pipeline in `data/processed/kicker_analysis.parquet`.
It mirrors the key derived metrics from the legacy R workflow.

## Identifiers
- **season**: NFL season year.
- **week**: NFL week number.
- **home_team**: Home team abbreviation (modernized).
- **away_team**: Away team abbreviation (modernized).

## Historical team form (rolling window)
- **home.fg_att / away.fg_att**: Field goal attempts over the rolling window.
- **home.fg_made / away.fg_made**: Field goals made over the rolling window.
- **home.pat_made / away.pat_made**: Extra points made over the rolling window.
- **homegamesplayed / awaygamesplayed**: Games played over the rolling window.

## Opponent-adjusted defense/offense (4th down focused)
- **home.adjfield_goal_attempt**: Opponent field goal attempts allowed on 4th downs.
- **away.adjfield_goal_attempt**: Opponent field goal attempts allowed on 4th downs.
- **home.adjfield_goal_attempt_for**: Team field goal attempts on 4th downs.
- **away.adjfield_goal_attempt_for**: Team field goal attempts on 4th downs.

## Expected attempt & fantasy metrics
- **homeexpatt / awayexpatt**: Expected attempts using offense/defense blend.
- **homekick_pct / awaykick_pct**: Rolling field goal accuracy.
- **homeexfppg / awayexfppg**: Expected fantasy points per game.
- **homeregressfp / awayregressfp**: Regression estimate based on expected attempts and accuracy.
- **homefppg / awayfppg**: Actual fantasy points derived from weekly kicker stats.

## Actual outcomes
- **actualhome.fg_att / actualaway.fg_att**: Actual field goal attempts in the target week.
- **actualhome.fg_made / actualaway.fg_made**: Actual field goals made in the target week.
- **actualhome.pat_made / actualaway.pat_made**: Actual PATs made in the target week.
- **homecash / awaycash**: Binary indicator for hitting 1.5+ field goals (1 hit, -1 miss).
