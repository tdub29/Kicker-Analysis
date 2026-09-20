"""Features, rebuilt so that everything is knowable on Tuesday.

Version one passed a leakage self-check and still could not be deployed. The
self-check proved the rolling windows were lagged correctly, which they were, and
that is not the same question as whether a column EXISTS before kickoff. Three
independent audits found the difference, and the numbers below are theirs.

What changed and why:

  temp and wind are gone.
      They were the 3rd and 7th most important features. They are also post-game
      observations: of 255 unplayed games in the schedule file, 0 carry a
      temperature or a wind speed, while rest, div_game, surface and home field
      carry 255 of 255. Training on observed weather and serving NaN cost
      +0.092 MAE against training and serving the same way. Dropping them
      entirely cost -0.002. The signal is real but thin, and it was being paid
      for with a column that does not exist at forecast time.
  stadium climate replaces them.
      A venue's typical wind and temperature in a given month IS knowable on
      Tuesday, computed from PRIOR seasons only. Soldier Field in December is a
      fact about the stadium, not about Sunday. This keeps the honest part of the
      weather signal and discards the part that was cheating.
  roof is collapsed to what is known in advance.
      `closed` and `open` are game-day decisions: across the whole file neither
      value ever appears on an unplayed game, so 792 of 5,229 modelling rows
      (15.1%) carried a label that cannot exist at forecast time. A retractable
      roof is now flagged as retractable, which is a property of the building.
      The old `is_dome` also silently scored a NaN roof as outdoors.
  the two calendar clocks are gone.
      `t_games_prior` and `opp_d_games_prior` correlate with `season` at 0.992.
      `season` is deliberately banned as a feature and these reintroduced it: in
      the 2025 fold, 80% and 86% of test rows sat above the training maximum, so
      the model was extrapolating a trend off the end of a counter. `k_games_prior`
      stays: it correlates 0.52 and is a real fact about a kicker's experience.
  team features are aggregated at TEAM-GAME grain before lagging.
      Nine team-games have two kicker rows, usually a punter or a position player
      kicking in an emergency. Sorted by team then week, those two rows are
      adjacent, so shift(1) landed inside the same game: Dallas week 7 2017 shows
      one row with the correct prior-four mean of 26.50 and the other with 32.25,
      which includes that day's 40 points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (4, 8)

OUTCOME_COLS = ["fantasy_points", "fg_att", "fg_made", "fg_missed", "pat_att",
                "pat_made", "fg_long", "team_score",
                "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
                "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
                "fourth_in_range", "fourth_fg", "fourth_go", "fourth_punt",
                "fourth_mean_togo", "fg_share", "go_rate"]

# Observed on the day, never available in advance. Named here so the ban is a
# fact in the code rather than a habit, and so feature_columns can assert it.
SERVE_TIME_UNAVAILABLE = ["temp", "wind", "wind_outdoor", "is_dome",
                          "t_games_prior", "opp_d_games_prior"]

FOURTH_COLS = ["fourth_in_range", "fourth_fg", "fourth_go", "fourth_mean_togo",
               "fg_share", "go_rate"]

# Venues whose roof can be shut on the day. Derived from the data rather than
# hardcoded, but the list is stable and worth naming for the reader.
RETRACTABLE_ROOFS = {"closed", "open"}
FIXED_DOME = {"dome"}


def _lagged(df: pd.DataFrame, group: str, cols: list[str], prefix: str,
            counter: bool = True) -> pd.DataFrame:
    """Rolling and expanding means over games strictly before this one.

    shift(1) is applied to the raw column FIRST, then the window. Windowing then
    shifting is equivalent only once the window is full, and leaks at the start of
    every group, which is exactly where a rookie kicker lives.
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
    if counter:
        out[f"{prefix}_games_prior"] = g.cumcount()
    return pd.DataFrame(out, index=df.index)


def stadium_climate(games: pd.DataFrame) -> pd.DataFrame:
    """Typical weather at a venue in a month, from PRIOR seasons only.

    This is the legitimate half of the weather signal. Nobody knows Sunday's gust
    on Tuesday, but everybody knows Soldier Field in December is windy, and that
    is a property of the building and the calendar rather than of the game. The
    expanding mean is shifted by season so a venue's own current year never
    contributes to its own climate estimate.
    """
    g = games.copy()
    g["month"] = pd.to_datetime(g.gameday, errors="coerce").dt.month
    g["venue"] = g.stadium_id.fillna(g.stadium).fillna(g.home_team)
    keep = g[["season", "month", "venue", "temp", "wind"]].dropna(subset=["venue", "month"])
    per = (keep.groupby(["venue", "month", "season"])
               .agg(temp=("temp", "mean"), wind=("wind", "mean"), n=("temp", "size"))
               .reset_index().sort_values(["venue", "month", "season"]))
    grp = per.groupby(["venue", "month"], sort=False)
    for src, dst in (("temp", "venue_temp_climate"), ("wind", "venue_wind_climate")):
        per[dst] = grp[src].transform(lambda s: s.shift(1).expanding().mean())
    per["venue_climate_n"] = grp["n"].transform(lambda s: s.shift(1).expanding().sum())
    return per[["venue", "month", "season", "venue_temp_climate",
                "venue_wind_climate", "venue_climate_n"]]


def roof_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Roof, as known before the game rather than as it turned out.

    A retractable stadium is identified by ever having reported open or closed.
    That identity is a property of the building, so it is available in advance,
    whereas the day's setting is not.
    """
    d = df.copy()
    venue = d.get("stadium_id")
    if venue is None:
        venue = d.get("stadium")
    d["_venue"] = venue.fillna(d.team) if venue is not None else d.team
    retract = set(d.loc[d.roof.isin(RETRACTABLE_ROOFS), "_venue"].dropna().unique())
    d["is_fixed_dome"] = d.roof.isin(FIXED_DOME).astype(int)
    d["is_retractable"] = d["_venue"].isin(retract).astype(int)
    # Indoors-for-certain means a fixed dome. A retractable is its own state,
    # because the model should learn that those games are usually but not always
    # sheltered rather than being told the wrong answer either way.
    d["is_indoor_certain"] = ((d.is_fixed_dome == 1) & (d.is_retractable == 0)).astype(int)
    return d.drop(columns=["_venue"])


def build_features(df: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    d = roof_flags(d)

    kicker = _lagged(d.sort_values(["player_id", "season", "week"]),
                     "player_id", ["fantasy_points", "fg_att", "fg_made",
                                   "pat_att", "fg_long"], "k")
    kicker["k_fg_pct_career"] = (kicker["k_fg_made_career"]
                                 / kicker["k_fg_att_career"].replace(0, np.nan))
    kicker["k_fg_pct_l8"] = (kicker["k_fg_made_l8"]
                             / kicker["k_fg_att_l8"].replace(0, np.nan))

    # TEAM-GAME grain, deduplicated, so a second kicker in the same game cannot
    # become his own teammate's "previous game".
    # Two different kinds of column, and summing both is wrong. Kicker columns are
    # additive across the men who kicked: two kickers in one game did attempt the
    # sum of their attempts. Game columns already describe the whole team-game and
    # summing them double-counts, which turned a 40-point afternoon into 80.
    tsum = ["fg_att", "pat_att", "fantasy_points"]
    tfirst = ["team_score"] + [c for c in FOURTH_COLS if c in d.columns]
    tg = (d.groupby(["team", "season", "week"], as_index=False)
            .agg({**{c: "sum" for c in tsum}, **{c: "first" for c in tfirst}})
            .sort_values(["team", "season", "week"]))
    tcols = tsum + tfirst
    tlag = _lagged(tg, "team", tcols, "t", counter=False)
    team = pd.concat([tg[["team", "season", "week"]], tlag], axis=1)

    # Opponent defence, same grain and the same split, keyed by the team played
    # against. FG attempts allowed sum across kickers; points allowed do not.
    og = (d.groupby(["opponent", "season", "week"], as_index=False)
            .agg({"fg_att": "sum", "team_score": "first"})
            .rename(columns={"opponent": "team"})
            .sort_values(["team", "season", "week"]))
    olag = _lagged(og, "team", ["fg_att", "team_score"], "opp_d", counter=False)
    opp = pd.concat([og[["team", "season", "week"]], olag], axis=1).rename(
        columns={"team": "opponent"})

    feats = (d.merge(team, on=["team", "season", "week"], how="left")
              .merge(opp, on=["opponent", "season", "week"], how="left"))
    feats = pd.concat([feats, kicker.reindex(feats.index)], axis=1)

    # Prior-season fourth-down rate, from TEAM-GAME rows so the denominator is the
    # number of fourth downs the team faced and not the number of kicker rows.
    # At kicker grain 26 of 347 team-seasons had the wrong count.
    if "go_rate" in d.columns:
        tgf = d.drop_duplicates(["team", "season", "week"])[
            ["team", "season", "week", "fourth_in_range", "fourth_fg", "fourth_go"]]
        sr = (tgf.groupby(["team", "season"], as_index=False)
                 .agg(n=("fourth_in_range", "sum"), g=("fourth_go", "sum"),
                      f=("fourth_fg", "sum")))
        sr["prev_go_rate"] = sr.g / sr.n.replace(0, np.nan)
        sr["prev_fg_share"] = sr.f / sr.n.replace(0, np.nan)
        sr["prev_fourth_n"] = sr.n
        # Reindex onto a dense team x season grid before shifting, so a missing
        # season shifts to NaN instead of quietly handing over the year before it.
        # With 2021 absent upstream, a row shift gave 2022 teams their 2020 rate.
        grid = pd.MultiIndex.from_product(
            [sorted(sr.team.unique()), range(int(sr.season.min()), int(sr.season.max()) + 1)],
            names=["team", "season"])
        sr = sr.set_index(["team", "season"]).reindex(grid).reset_index()
        for c in ("prev_go_rate", "prev_fg_share", "prev_fourth_n"):
            sr[c] = sr.groupby("team")[c].shift(1)
        feats = feats.merge(sr[["team", "season", "prev_go_rate", "prev_fg_share",
                                "prev_fourth_n"]], on=["team", "season"], how="left")

    if games is not None:
        clim = stadium_climate(games)
        gg = games.copy()
        gg["month"] = pd.to_datetime(gg.gameday, errors="coerce").dt.month
        gg["venue"] = gg.stadium_id.fillna(gg.stadium).fillna(gg.home_team)
        gmap = gg[["game_id", "venue", "month"]]
        feats = feats.merge(gmap, on="game_id", how="left").merge(
            clim, on=["venue", "month", "season"], how="left").drop(columns=["venue", "month"])

    feats["closeness"] = -feats["team_spread"].abs()
    feats["close_high_total"] = feats["closeness"] * feats["total_line"]
    # Exposure to weather, using only what is knowable: an outdoor venue in a
    # month that is historically windy.
    if "venue_wind_climate" in feats.columns:
        feats["climate_wind_exposed"] = (feats["venue_wind_climate"].fillna(0)
                                         * (1 - feats["is_indoor_certain"]))
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    banned = set(OUTCOME_COLS) | set(SERVE_TIME_UNAVAILABLE) | {
        "season", "week", "game_order", "fg_blocked", "pat_missed", "fg_pct"}
    return sorted(c for c in df.columns
                  if c not in banned and pd.api.types.is_numeric_dtype(df[c]))


def self_check() -> int:
    d = pd.DataFrame({
        "player_id": ["a"] * 5 + ["b"] * 3,
        "team": ["KC"] * 5 + ["BAL"] * 3,
        "opponent": ["BAL"] * 5 + ["KC"] * 3,
        "season": [2024] * 8, "week": [1, 2, 3, 4, 5, 1, 2, 3],
        "game_id": [f"g{i}" for i in range(8)],
        "fantasy_points": [10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0],
        "fg_att": [1, 2, 3, 4, 5, 1, 1, 1], "fg_made": [1, 2, 3, 4, 5, 1, 1, 1],
        "pat_att": [1] * 8, "pat_made": [1] * 8, "fg_long": [40] * 8,
        "fg_missed": [0] * 8, "team_score": [20] * 8,
        "roof": ["outdoors"] * 4 + ["closed"] + ["dome"] * 3,
        "stadium_id": ["S1"] * 5 + ["S2"] * 3,
        "is_home": [1] * 8, "temp": [60.0] * 8, "wind": [5.0] * 8,
        "div_game": [0] * 8, "rest": [7] * 8, "total_line": [45.0] * 8,
        "team_spread": [-3.0] * 8, "implied_total": [21.0] * 8,
        "opp_implied_total": [24.0] * 8,
    })
    f = build_features(d)
    a = f[f.player_id == "a"].sort_values("week")

    assert pd.isna(a.k_fantasy_points_career.iloc[0])
    assert a.k_fantasy_points_career.iloc[2] == 15.0
    assert a.k_fantasy_points_l4.iloc[4] == 25.0

    cols = feature_columns(f)
    # The three serve-time columns must be unreachable, not merely unused.
    for banned in ("temp", "wind", "is_dome", "t_games_prior", "opp_d_games_prior"):
        assert banned not in cols, banned
    assert "fantasy_points" not in cols
    assert "implied_total" in cols
    # A retractable venue is flagged by the building, not by Sunday's setting, so
    # every S1 row carries it even though only one of them reported `closed`.
    s1 = f[f.stadium_id == "S1"]
    assert (s1.is_retractable == 1).all(), s1.is_retractable.tolist()
    assert (s1.is_indoor_certain == 0).all()
    s2 = f[f.stadium_id == "S2"]
    assert (s2.is_indoor_certain == 1).all()

    # Two kickers in ONE team-game must not become each other's previous game.
    dup = pd.DataFrame({
        "player_id": ["x", "y", "x"], "team": ["DAL"] * 3, "opponent": ["NYG"] * 3,
        "season": [2024] * 3, "week": [1, 1, 2], "game_id": ["g1", "g1", "g2"],
        "fantasy_points": [9.0, 3.0, 5.0], "fg_att": [3, 1, 2], "fg_made": [3, 1, 2],
        "pat_att": [1, 0, 1], "pat_made": [1, 0, 1], "fg_long": [50, 30, 40],
        "fg_missed": [0, 0, 0], "team_score": [40.0, 40.0, 17.0],
        "roof": ["outdoors"] * 3, "stadium_id": ["S9"] * 3, "is_home": [1] * 3,
        "temp": [60.0] * 3, "wind": [5.0] * 3, "div_game": [0] * 3, "rest": [7] * 3,
        "total_line": [45.0] * 3, "team_spread": [-3.0] * 3,
        "implied_total": [21.0] * 3, "opp_implied_total": [24.0] * 3,
    })
    g = build_features(dup)
    wk1 = g[g.week == 1]
    # Neither week-1 row may see week 1's own team score of 40.
    assert wk1.t_team_score_l4.isna().all(), wk1.t_team_score_l4.tolist()
    # Week 2 sees week 1 exactly once, at team-game grain, not twice.
    assert float(g[g.week == 2].t_team_score_l4.iloc[0]) == 40.0
    # Team fantasy points for week 1 are the SUM of both kickers, seen in week 2.
    assert float(g[g.week == 2].t_fantasy_points_l4.iloc[0]) == 12.0
    print("features self-check ok")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_check())
