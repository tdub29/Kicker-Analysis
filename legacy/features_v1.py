"""Lagged features for the kicker model.

The one rule this file exists to enforce: a row may only see games that finished
BEFORE its own kickoff. Every history feature is built with `shift(1)` inside the
group before any window is applied, so week N's row is a function of weeks 1..N-1
and nothing else. The old pipeline had no such guarantee, and its target was a
rolling average of the very thing it was predicting.

Features come in four families:

  kicker form   what this leg has done lately, at three horizons. Four games is
                noisy and a career mean is stale, so both are offered and the
                importance study decides which earns its place.
  team offence  how often this offence stalls in range. A kicker's volume is an
                offence's failure mode, which is why team context beats personal
                accuracy for predicting ATTEMPTS.
  opponent      what the defence has allowed. Thin signal historically, included
                so the study can say so with a number instead of an opinion.
  game context  known before kickoff and needs no lag: the betting market, the
                roof, the weather, rest, home field.

`implied_total` is the headline addition. It is the market's own forecast of this
offence's scoring, it is available on Tuesday, and the previous model ignored it
while carrying the raw line in the file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Horizons in games. 4 is "current form", 8 is roughly half a season, and the
# expanding mean is everything the man has ever done.
WINDOWS = (4, 8)

# Outcome columns that describe a finished game. Every one of these must be
# lagged before use; naming them in one place is what makes that checkable.
OUTCOME_COLS = ["fantasy_points", "fg_att", "fg_made", "fg_missed", "pat_att",
                "pat_made", "fg_long", "team_score",
                "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
                "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
                # Fourth-down behaviour in THIS game is an outcome like any
                # other. Only the lagged versions may reach the model.
                "fourth_in_range", "fourth_fg", "fourth_go", "fourth_punt",
                "fourth_mean_togo", "fg_share", "go_rate"]

# Coach aggression on fourth down in field-goal range, from fourth_down.py.
# Measured 2015-2025: a team's go rate carries year to year at r = 0.27 on
# average and r = 0.42 over the last four pairs, with an 8.6 point spread across
# teams in a season. In 2025 Arizona went for it on 0 of 21 in-range fourth
# downs and Buffalo on 13 of 27, which is most of a field goal a game between
# them. It is a persistent property of a staff, so it is predictive, not noise.
FOURTH_COLS = ["fourth_in_range", "fourth_fg", "fourth_go", "fourth_mean_togo",
               "fg_share", "go_rate"]

# Known before kickoff, so they enter the model as-is.
PREGAME_COLS = ["is_home", "is_dome", "temp", "wind", "div_game", "rest",
                "total_line", "team_spread", "implied_total", "opp_implied_total"]


def _lagged(df: pd.DataFrame, group: str, cols: list[str], prefix: str) -> pd.DataFrame:
    """Rolling and expanding means of `cols` over games strictly before this one.

    shift(1) happens FIRST, on the raw column, so no window ever touches the
    current row. Doing it the other way round (window then shift) is the same
    thing only when the window is complete, and silently leaks at the start of
    every group, which is exactly where a new kicker lives.
    """
    out = {}
    g = df.groupby(group, sort=False)
    for c in cols:
        prior = g[c].shift(1)
        by = prior.groupby(df[group], sort=False)
        for w in WINDOWS:
            out[f"{prefix}_{c}_l{w}"] = by.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean())
        out[f"{prefix}_{c}_career"] = by.transform(lambda s: s.expanding().mean())
    out[f"{prefix}_games_prior"] = g.cumcount()
    return pd.DataFrame(out, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per kicker-game with its pre-kickoff feature vector."""
    d = df.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    d["game_order"] = np.arange(len(d))

    # A kicker's own history, across teams and seasons. It follows the man.
    kicker = _lagged(d.sort_values(["player_id", "season", "week"]),
                     "player_id", ["fantasy_points", "fg_att", "fg_made",
                                   "pat_att", "fg_long"], "k")
    # Accuracy as a rate rather than two counts, since the counts are collinear
    # with volume and the rate is the part that is about the leg.
    kicker["k_fg_pct_career"] = (kicker["k_fg_made_career"]
                                 / kicker["k_fg_att_career"].replace(0, np.nan))
    kicker["k_fg_pct_l8"] = (kicker["k_fg_made_l8"]
                             / kicker["k_fg_att_l8"].replace(0, np.nan))

    team_cols = ["fg_att", "pat_att", "team_score", "fantasy_points"]
    team_cols += [c for c in FOURTH_COLS if c in d.columns]
    team = _lagged(d.sort_values(["team", "season", "week"]), "team", team_cols, "t")

    # A prior-SEASON rate as well as a rolling one. Four in-range fourth downs
    # into a year the rolling window is one coin flip wide, while last season is
    # a full sample of the same staff. Shifted by season so the current year
    # never contributes to its own feature.
    if "go_rate" in d.columns:
        season_rate = (d.groupby(["team", "season"])
                       .agg(n=("fourth_in_range", "sum"), g=("fourth_go", "sum"),
                            f=("fourth_fg", "sum")).reset_index())
        season_rate["prev_go_rate"] = (season_rate.g / season_rate.n.replace(0, np.nan))
        season_rate["prev_fg_share"] = (season_rate.f / season_rate.n.replace(0, np.nan))
        season_rate["prev_fourth_n"] = season_rate.n
        season_rate = season_rate.sort_values(["team", "season"])
        for c in ("prev_go_rate", "prev_fg_share", "prev_fourth_n"):
            season_rate[c] = season_rate.groupby("team")[c].shift(1)
        d = d.merge(season_rate[["team", "season", "prev_go_rate", "prev_fg_share",
                                 "prev_fourth_n"]], on=["team", "season"], how="left")

    # Opponent defence, keyed on the defence's own game history. Built by
    # relabelling each row under the team it was played AGAINST.
    opp_src = d.rename(columns={"team": "_off", "opponent": "team"})
    opp = _lagged(opp_src.sort_values(["team", "season", "week"]),
                  "team", ["fg_att", "team_score"], "d")
    opp = opp.add_prefix("opp_")

    feats = pd.concat([d, kicker.reindex(d.index), team.reindex(d.index),
                       opp.reindex(d.index)], axis=1)

    # Two interactions with a reason to exist rather than a search over pairs.
    # A close, high-scoring game is the field-goal machine: plenty of drives and
    # neither side sitting on a lead.
    feats["closeness"] = -feats["team_spread"].abs()
    feats["close_high_total"] = feats["closeness"] * feats["total_line"]
    # Wind matters outdoors and is defined as zero in a dome, so the product
    # carries "exposed to weather" rather than the model inferring it.
    feats["wind_outdoor"] = feats["wind"].fillna(0) * (1 - feats["is_dome"])
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric model inputs. Excludes the target, ids, and anything unlagged."""
    banned = set(OUTCOME_COLS) | {
        "season", "week", "game_order", "fg_blocked", "pat_missed", "fg_pct"}
    cols = []
    for c in df.columns:
        if c in banned or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return sorted(cols)


def self_check() -> int:
    """The leakage guarantee, stated as a test that fails if shift(1) is dropped."""
    d = pd.DataFrame({
        "player_id": ["a"] * 5 + ["b"] * 3,
        "team": ["KC"] * 5 + ["BAL"] * 3,
        "opponent": ["BAL"] * 5 + ["KC"] * 3,
        "season": [2024] * 8, "week": [1, 2, 3, 4, 5, 1, 2, 3],
        "fantasy_points": [10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0],
        "fg_att": [1, 2, 3, 4, 5, 1, 1, 1], "fg_made": [1, 2, 3, 4, 5, 1, 1, 1],
        "pat_att": [1] * 8, "pat_made": [1] * 8, "fg_long": [40] * 8,
        "fg_missed": [0] * 8, "team_score": [20] * 8,
        "fg_made_0_19": [0] * 8, "fg_made_20_29": [0] * 8, "fg_made_30_39": [1] * 8,
        "fg_made_40_49": [0] * 8, "fg_made_50_59": [0] * 8, "fg_made_60_": [0] * 8,
        "is_home": [1] * 8, "is_dome": [0] * 8, "temp": [60.0] * 8, "wind": [5.0] * 8,
        "div_game": [0] * 8, "rest": [7] * 8, "total_line": [45.0] * 8,
        "team_spread": [-3.0] * 8, "implied_total": [21.0] * 8,
        "opp_implied_total": [24.0] * 8,
    })
    f = build_features(d).sort_values(["player_id", "week"]).reset_index(drop=True)
    a = f[f.player_id == "a"]

    # Game 1 has no prior game, so every history feature is undefined, not zero.
    assert pd.isna(a.k_fantasy_points_career.iloc[0]), a.k_fantasy_points_career.iloc[0]
    assert a.k_games_prior.iloc[0] == 0
    # Game 3's career mean is the mean of games 1 and 2 only: (10+20)/2 = 15.
    assert a.k_fantasy_points_career.iloc[2] == 15.0, a.k_fantasy_points_career.iloc[2]
    # Game 5's 4-game window is games 1-4: (10+20+30+40)/4 = 25. If the shift were
    # applied after the window it would be 30 and the model would see its own game.
    assert a.k_fantasy_points_l4.iloc[4] == 25.0, a.k_fantasy_points_l4.iloc[4]
    # The last row must never equal its own value under any history column.
    hist = [c for c in f.columns if c.startswith(("k_", "t_", "opp_"))]
    assert not (a[hist].iloc[-1] == 50.0).any(), "a history column leaked the target"
    # Groups are independent: kicker b's first game knows nothing about a.
    b = f[f.player_id == "b"]
    assert pd.isna(b.k_fantasy_points_career.iloc[0])
    # The target itself must never be offered as an input.
    assert "fantasy_points" not in feature_columns(f)
    assert "implied_total" in feature_columns(f)
    print("features self-check ok")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_check())
