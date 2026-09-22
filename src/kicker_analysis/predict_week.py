"""Rank this week's kickers and write the file the Fantasy Football site reads.

The backtest measures the method. This applies it to games that have not happened
yet, which is a different job with one hard constraint: the feature row for an
unplayed game must be built the same way a training row was, from history only.
That is done by appending a PSEUDO-ROW per kicker per upcoming game, with every
outcome column left null, then running the identical feature builder over history
plus pseudo-rows together. The lagged windows then fill from real prior games and
nothing else, because there is nothing else there to fill from.

Two rankings come out and they are not the same list:

  by expected points     the safe start. Highest mean.
  by P(y >= 15)          the ceiling swing. Use it when a floor does not help,
                         because ranking by the mean actively selects AGAINST
                         the ceiling: it prefers the extra-point-heavy kicker on
                         a good offence and hits a 15-point week less often than
                         picking at random.

Who kicks for whom is taken from the most recent game in the data rather than
from a depth chart, and the row carries how many days stale that is. A kicker
signed on Wednesday will not appear, which is stated in the output rather than
hidden.

    python src/kicker_analysis/predict_week.py
    python src/kicker_analysis/predict_week.py --week 3
    python src/kicker_analysis/predict_week.py --self-check
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.build_dataset import (  # noqa: E402
    GAMES_URL, LIVE_MAX_AGE_HOURS, RELOCATIONS, fetch)
from kicker_analysis.features import build_features  # noqa: E402
from kicker_analysis.model import CompoundKickerModel, FEATURES, RidgeRanker  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
# One producer, one path. `_v2` was a hand-copied snapshot that nothing
# regenerated, so the weekly board trained on whatever Sunday the copy was
# taken and looked current: it was 30 kicker-games short of week 2 while
# printing a week-3 header. build_dataset.py owns this file now.
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
REPORTS = ROOT / "reports"
SITE = pathlib.Path(
    r"C:\Users\TrevorWhite\Downloads\Big Projects\Fantasy Football\site\data\kickers.js")
FIRST_SEASON = 2015


def upcoming(games: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    """The next slate with no result yet, in the current season."""
    reg = games[(games.game_type == "REG")].copy()
    season = int(reg[reg.home_score.notna()].season.max())
    cur = reg[reg.season == season]
    open_games = cur[cur.home_score.isna()]
    if open_games.empty:
        raise SystemExit(f"no unplayed {season} games left")
    week = int(open_games.week.min())
    return season, week, open_games[open_games.week == week].copy()


def current_kickers(hist: pd.DataFrame, season: int) -> pd.DataFrame:
    """One kicker per team, the last man who actually kicked for them.

    Restricted to the current season so a departed kicker from two years ago
    cannot be resurrected onto a roster he no longer holds.
    """
    recent = hist[hist.season == season].sort_values(["team", "season", "week"])
    if recent.empty:
        recent = hist.sort_values(["team", "season", "week"])
    last = recent.groupby("team").tail(1)
    return last[["team", "player_id", "player_display_name", "week"]].rename(
        columns={"week": "last_seen_week"})


def pseudo_rows(slate: pd.DataFrame, kickers: pd.DataFrame, season: int,
                week: int, template: pd.DataFrame) -> pd.DataFrame:
    """A feature row per kicker per upcoming game, outcomes left null."""
    rows = []
    for _, g in slate.iterrows():
        for side, opp_side in (("home", "away"), ("away", "home")):
            team = RELOCATIONS.get(g[f"{side}_team"], g[f"{side}_team"])
            opp = RELOCATIONS.get(g[f"{opp_side}_team"], g[f"{opp_side}_team"])
            k = kickers[kickers.team == team]
            if k.empty:
                continue
            k = k.iloc[0]
            margin = g.spread_line if side == "home" else -g.spread_line
            rows.append({
                "season": season, "week": week, "season_type": "REG",
                "player_id": k.player_id, "player_display_name": k.player_display_name,
                "team": team, "opponent": opp, "game_id": g.game_id,
                "is_home": int(side == "home"), "rest": g.get(f"{side}_rest"),
                "roof": g.roof, "surface": g.surface, "div_game": g.div_game,
                "total_line": g.total_line, "team_spread": margin,
                "implied_total": g.total_line / 2.0 + margin / 2.0,
                "opp_implied_total": g.total_line / 2.0 - margin / 2.0,
                "last_seen_week": k.last_seen_week,
            })
    out = pd.DataFrame(rows)
    # Every column the builder expects must exist, null where it is an outcome.
    for c in template.columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[[c for c in template.columns if c in out.columns]
               + [c for c in out.columns if c not in template.columns]]


def run(week_override: int | None = None) -> pd.DataFrame:
    # The upcoming slate is decided from this file, so reading a cached copy
    # silently re-forecasts a week that is already over. Refresh it first.
    games = pd.read_csv(fetch(GAMES_URL, "games.csv", max_age_hours=LIVE_MAX_AGE_HOURS),
                        low_memory=False)
    for c in ("home_team", "away_team"):
        games[c] = games[c].replace(RELOCATIONS)
    hist = pd.read_parquet(PROC / "kicker_games.parquet")
    fourth = pd.read_parquet(ROOT / "data" / "processed" / "fourth_down.parquet")
    hist = hist.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    hist = hist[(hist.season >= FIRST_SEASON) & (hist.season_type == "REG")].copy()

    season, week, slate = upcoming(games)
    if week_override:
        week = week_override
        slate = games[(games.season == season) & (games.week == week)
                      & (games.game_type == "REG")].copy()
    # The training history must reach the week before the one being forecast.
    # It did not, and nothing said so: `_v2/kicker_games.parquet` was a frozen
    # copy no script rebuilt, so a week-3 board trained through week 1 and
    # looked identical to a correct one. This is the check that would have
    # caught it in one line instead of by reading a parquet by hand.
    done = hist[(hist.season == season) & hist.fantasy_points.notna()]
    through = int(done.week.max()) if len(done) else 0
    if through < week - 1:
        raise SystemExit(
            f"training data stops at {season} week {through} but the board is for "
            f"week {week}. Run build_dataset.py and fourth_down.py first; a board "
            f"built on stale history is indistinguishable from a current one.")
    print(f"history complete through {season} week {through}")

    ks = current_kickers(hist, season)
    fut = pseudo_rows(slate, ks, season, week, hist)
    print(f"{season} week {week}: {len(slate)} games, {len(fut)} kickers")

    # Tag the pseudo-rows explicitly. Selecting them back by (season, week) also
    # picked up the Thursday game that had already kicked off, which is how a
    # 15-game slate produced 32 kickers and listed a man whose week was over.
    hist = hist.assign(_pseudo=0)
    fut = fut.assign(_pseudo=1)
    combined = pd.concat([hist, fut.drop(columns=["last_seen_week"])], ignore_index=True)
    feats = build_features(combined, games=games)
    future = feats[feats._pseudo == 1].copy()
    # Train on everything with a real result, including games already played in
    # the current week. A Thursday result is legitimate input to Sunday.
    train = feats[(feats._pseudo == 0) & feats.fantasy_points.notna()].copy()
    played_this_week = set(feats.loc[(feats._pseudo == 0) & (feats.season == season)
                                     & (feats.week == week), "team"])
    if played_this_week:
        print(f"already kicked off this week, excluded from the board: "
              f"{', '.join(sorted(played_this_week))}")
    print(f"training on {len(train):,} completed kicker-games through "
          f"{int(train.season.max())} week {int(train[train.season == train.season.max()].week.max())}")

    # Two models, each doing the job it measurably wins. Ridge orders the board
    # (+0.072 pts/wk over the implied total, p = 0.015; +0.086 over the count
    # model, p = 0.002). The count model supplies the distribution, where it is
    # calibrated to within 5% at every threshold.
    pred = CompoundKickerModel().fit(train).predict(future)
    pred["exp_points"] = RidgeRanker().fit(train).predict(future)
    out = pd.concat([future[["player_id", "player_display_name", "team", "opponent",
                             "is_home", "implied_total", "total_line", "team_spread",
                             "is_indoor_certain", "game_id"]].reset_index(drop=True),
                     pred.reset_index(drop=True)], axis=1)
    out = out.merge(fut[["player_id", "last_seen_week"]].drop_duplicates("player_id"),
                    on="player_id", how="left")
    out["stale_weeks"] = week - out.last_seen_week
    out["season"] = season
    out["week"] = week
    out = out.sort_values("exp_points", ascending=False).reset_index(drop=True)
    out["rank_points"] = np.arange(1, len(out) + 1)
    out["rank_ceiling"] = out.p_ge_15.rank(ascending=False, method="min").astype(int)

    cols = ["rank_points", "rank_ceiling", "player_display_name", "team", "opponent",
            "implied_total", "exp_points", "floor_p20", "ceiling_p90", "p_ge_15",
            "exp_fg_att", "exp_pat_att", "stale_weeks"]
    print("\n" + out[cols].head(20).round(3).to_string(index=False))

    stale = out[out.stale_weeks > 1]
    if len(stale):
        print(f"\n{len(stale)} kicker(s) last kicked more than a week ago, so the roster "
              f"read may be out of date: {', '.join(stale.player_display_name)}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / f"week_{season}_{week}_kickers.csv", index=False)
    write_site(out, season, week)
    return out


def write_site(df: pd.DataFrame, season: int, week: int) -> None:
    """Emit the file the Fantasy Football site loads, as window.KICKER_MODEL."""
    payload = {
        "season": season, "week": week,
        "method": "two-stage Poisson counts (FG attempts, PAT attempts) convolved "
                  "through the scoring rule into a full predictive distribution",
        "scoring": "FG 0-39 = 3, 40-49 = 4, 50+ = 5, PAT = 1, miss = 0",
        "features": FEATURES,
        "note": "The board is ordered by a ridge on 16 pre-kickoff features, which "
                "beats ranking on the Vegas implied total alone by 0.072 points a week "
                "(p = 0.015) and beats a kicker's own career average by 0.273. The "
                "probabilities come from a separate count model whose P(15+) is "
                "calibrated to within half a percent. Expect a small edge: the oracle "
                "who always starts the better of two kickers gains 2.44 points a week "
                "and this captures 17.8% of that.",
        "kickers": [
            {
                "id": str(r.player_id), "name": r.player_display_name,
                "team": r.team, "opp": r.opponent, "home": int(r.is_home),
                "rankPoints": int(r.rank_points), "rankCeiling": int(r.rank_ceiling),
                "exp": round(float(r.exp_points), 2),
                "floor": round(float(r.floor_p20), 1),
                "ceil": round(float(r.ceiling_p90), 1),
                "pGe15": round(float(r.p_ge_15), 4),
                "pGe10": round(float(r.p_ge_10), 4),
                "impliedTotal": None if pd.isna(r.implied_total) else round(float(r.implied_total), 1),
                "expFgAtt": round(float(r.exp_fg_att), 2),
                "expPatAtt": round(float(r.exp_pat_att), 2),
                "indoor": int(r.is_indoor_certain),
                "staleWeeks": None if pd.isna(r.stale_weeks) else int(r.stale_weeks),
            }
            for r in df.itertuples()
        ],
    }
    SITE.parent.mkdir(parents=True, exist_ok=True)
    SITE.write_text("window.KICKER_MODEL = "
                    + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"\nwrote {SITE}: {len(payload['kickers'])} kickers, "
          f"{SITE.stat().st_size / 1024:.0f} KB")


def self_check() -> int:
    """Pseudo-row construction, where a sign error would invert every ranking."""
    tmpl = pd.DataFrame(columns=["season", "week", "season_type", "player_id",
                                 "player_display_name", "team", "opponent", "game_id",
                                 "fantasy_points", "fg_att"])
    slate = pd.DataFrame([{"game_id": "g1", "home_team": "KC", "away_team": "BAL",
                           "spread_line": 3.0, "total_line": 46.0, "roof": "outdoors",
                           "surface": "grass", "div_game": 0, "home_rest": 7,
                           "away_rest": 7}])
    ks = pd.DataFrame([{"team": "KC", "player_id": "k1", "player_display_name": "A",
                        "last_seen_week": 1},
                       {"team": "BAL", "player_id": "k2", "player_display_name": "B",
                        "last_seen_week": 1}])
    rows = pseudo_rows(slate, ks, 2026, 2, tmpl).set_index("team")
    assert len(rows) == 2
    # Home favoured by 3 on a 46 total: 24.5 and 21.5, summing to the total.
    assert rows.loc["KC", "implied_total"] == 24.5, rows.loc["KC", "implied_total"]
    assert rows.loc["BAL", "implied_total"] == 21.5
    assert rows.loc["BAL", "team_spread"] == -3.0
    assert rows.loc["KC", "is_home"] == 1 and rows.loc["BAL", "is_home"] == 0
    # Outcomes must be absent, or a pseudo-row could train on itself.
    assert pd.isna(rows.loc["KC", "fantasy_points"])
    assert pd.isna(rows.loc["KC", "fg_att"])
    # A team with no known kicker is skipped rather than given a placeholder.
    rows2 = pseudo_rows(slate, ks[ks.team == "KC"], 2026, 2, tmpl)
    assert len(rows2) == 1 and rows2.iloc[0].team == "KC"
    print("predict_week self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--week", type=int, default=None)
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else (run(a.week), 0)[1])
