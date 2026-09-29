"""Season-forward backtest of the compound model against the baselines that beat v1.

Scored on the decision, not on the residual. The first harness sorted by MAE and
crowned a model that a constant 8.0 beat; MAE on this target is dominated by
irreducible noise (sd 4.4) and compresses real differences into the fourth
decimal. What a manager actually does is pick one kicker from the two or three he
can start, so that is what is measured:

  lift_rN    points per week gained by starting the model's favourite of N
             kickers drawn at random, against starting a random one of the N.
  share      that lift as a fraction of the oracle's, so the ceiling is visible.
  ceiling    how often the pick returns a 15-point week. Ranking by expected
             points makes this WORSE than random, which is the finding that
             justifies carrying a distribution at all.
  rho        within-week Spearman, the ordering the lift comes from.
  MAE        kept, demoted, never sorted on.

Every one of those is computed EXACTLY, not simulated. The first harness drew
300 random rosters a week, which added noise to every comparison and even gave a
constant a lift of -0.015. The expectation over every R-subset of the week has a
closed form (see _pick_probs), so there is nothing to simulate.

Intervals are a week-block bootstrap. Weeks are the unit because rows inside one
share a slate, a weather system and a set of opponents.

    python src/kicker_analysis/backtest.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.features import check_roof  # noqa: E402
from kicker_analysis.model import CompoundKickerModel, RidgeRanker, FEATURES, load  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIRST_TEST = 2018
MIN_WEEKS = 8
SEED = 0


def _pick_probs(s: np.ndarray, roster: int) -> np.ndarray:
    """P(row i is the pick) when `roster` rows are drawn uniformly from the week
    and the one with the highest score is started, ties split evenly (which is
    what argmax over a randomly ordered draw does). Row i is picked when the
    other roster-1 are all at or below it, and it wins a t-way tie 1/(t+1)."""
    lower = (s[None, :] < s[:, None]).sum(axis=1)
    tied = (s[None, :] == s[:, None]).sum(axis=1) - 1
    p = sum(comb(tied, t) * comb(lower, roster - 1 - t) / (t + 1) for t in range(roster))
    return p / comb(len(s), roster)


def per_week_lift(df: pd.DataFrame, col: str, roster: int, seed: int = SEED):
    """Per-WEEK lift, ceiling rate, oracle lift and random ceiling rate, exact.

    Returned per week rather than averaged, because the week is the resampling
    unit and the statistic is a mean over weeks. `seed` is kept for callers and
    unused: nothing here is random any more.
    """
    lift, ceil, orac, rndc = [], [], [], []
    # Four columns only: iterating groups of a wide frame is 10x slower.
    for _, g in df[["season", "week", "y", col]].groupby(["season", "week"], sort=True):
        if len(g) < roster:
            continue
        y, s = g.y.to_numpy(float), g[col].to_numpy(float)
        assert np.isfinite(s).all(), f"{col} has a missing score"
        p, o = _pick_probs(s, roster), _pick_probs(y, roster)
        lift.append(p @ y - y.mean())
        ceil.append(p @ (y >= 15))
        orac.append(o @ y - y.mean())
        rndc.append((y >= 15).mean())
    return (np.array(lift), np.array(ceil), np.array(orac), np.array(rndc))


def decision(df: pd.DataFrame, col: str, roster: int, seed: int = SEED) -> dict:
    lift, ceil, orac, rndc = per_week_lift(df, col, roster, seed)
    omax = float(orac.mean()) if len(orac) else np.nan
    l = float(lift.mean()) if len(lift) else np.nan
    return {"lift": l, "share": l / omax if omax else np.nan,
            "ceiling": float(ceil.mean()), "ceiling_random": float(rndc.mean()),
            "oracle_lift": omax}


def rho(df: pd.DataFrame, col: str) -> float:
    vals = []
    for _, g in df.groupby(["season", "week"]):
        if len(g) < 5 or g[col].nunique() < 2 or g.y.nunique() < 2:
            continue
        r = spearmanr(g.y, g[col]).statistic
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def boot_lift(df: pd.DataFrame, a: str, b: str, roster: int = 2, n: int = 10000) -> dict:
    """Paired week-block bootstrap of the difference in mean weekly lift.

    Paired on the WEEK: both rankings are evaluated on the same weeks, so the
    comparison is not polluted by which weeks happened to be easy.
    """
    la, _, _, _ = per_week_lift(df, a, roster)
    lb, _, _, _ = per_week_lift(df, b, roster)
    assert len(la) == len(lb), (len(la), len(lb))
    d = la - lb
    rng = np.random.default_rng(SEED)
    k = len(d)
    draws = d[rng.integers(0, k, size=(n, k))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "weeks": k,
            "p": float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))}


def main() -> int:
    df = load(processed=ROOT / "data" / "processed")
    seasons = sorted(df.season.unique())
    tests = [s for s in seasons if s >= FIRST_TEST
             and df[df.season == s].week.nunique() >= MIN_WEEKS]
    check_roof(df)
    live = [f for f in FEATURES if df[f].std() > 0]
    assert len(live) == len(FEATURES), sorted(set(FEATURES) - set(live))
    print(f"{len(df):,} kicker-games, {len(FEATURES)} named features, "
          f"{len(live)} non-constant")
    print(f"test seasons: {[int(s) for s in tests]}\n")

    out = []
    for s in tests:
        tr, te = df[df.season < s], df[df.season == s]
        pred = CompoundKickerModel().fit(tr).predict(te)
        # The ridge is what ORDERS the published board, so it has to be in the
        # table that the README points at. It was missing, which meant the
        # headline lift could not be reproduced by the command the README gives.
        ridge = RidgeRanker().fit(tr).predict(te)
        med = float(tr.fantasy_points.median())
        block = pd.DataFrame({
            "season": te.season.to_numpy(), "week": te.week.to_numpy(),
            "player_id": te.player_id.to_numpy(),
            "player": te.player_display_name.to_numpy(), "team": te.team.to_numpy(),
            "y": te.fantasy_points.to_numpy(),
            "constant": med,
            "career": te.k_fantasy_points_career.fillna(med).to_numpy(),
            "implied": te.implied_total.fillna(te.implied_total.median()).to_numpy(),
            "ridge": np.asarray(ridge, dtype=float),
        })
        for c in pred.columns:
            block[c] = pred[c].to_numpy()
        out.append(block)
    P = pd.concat(out, ignore_index=True)

    rows = []
    for name, col in [("ridge (the board)", "ridge"),
                      ("compound E[y]", "exp_points"), ("compound P(y>=15)", "p_ge_15"),
                      ("kicker career mean", "career"), ("implied total only", "implied"),
                      ("constant 8.0", "constant")]:
        r = {"rank by": name, "rho": rho(P, col),
             "mae": float(np.abs(P.y - P.exp_points).mean()) if col == "exp_points" else np.nan}
        for R in (2, 3):
            d = decision(P, col, R)
            r[f"lift_r{R}"] = d["lift"]
            r[f"share_r{R}"] = d["share"]
            if R == 2:
                r["ceiling_r2"] = d["ceiling"]
        rows.append(r)
    t = pd.DataFrame(rows)
    print("Ranked by what a manager actually gets, points per week over random:\n")
    print(t.round(4).to_string(index=False))
    o = decision(P, "exp_points", 2)
    print(f"\noracle lift R=2 = {o['oracle_lift']:.3f} pts/wk (the ceiling)")
    print(f"random ceiling rate R=2 = {o['ceiling_random']:.4f} "
          f"(a pick must BEAT this to be worth making for upside)")
    print(f"MAE, compound vs constant: {np.abs(P.y - P.exp_points).mean():.4f} "
          f"vs {np.abs(P.y - P.constant).mean():.4f}")

    print("\nWeek-block bootstrap, R=2 lift:")
    for a, b in [("ridge", "implied"), ("ridge", "career"), ("ridge", "exp_points"),
                 ("exp_points", "career"), ("exp_points", "constant"),
                 ("exp_points", "implied")]:
        c = boot_lift(P, a, b)
        print(f"  {a} vs {b:10} {c['diff']:+.4f}  95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  p={c['p']:.4f}")

    # Calibration: does P(y>=x) mean what it says?
    print("\nCalibration of the predictive law:")
    for thr in (8, 10, 12, 15):
        pred_rate = P[f"p_ge_{thr}"].mean()
        real_rate = (P.y >= thr).mean()
        print(f"  P(y>={thr:2}) predicted {pred_rate:.4f}  realised {real_rate:.4f}  "
              f"ratio {pred_rate / real_rate:.3f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    # predict_week reads this file, possibly while this runs: write, then swap.
    tmp = REPORTS / "backtest_predictions.parquet.tmp"
    P.to_parquet(tmp, index=False)
    os.replace(tmp, REPORTS / "backtest_predictions.parquet")
    t.to_csv(REPORTS / "backtest_summary.csv", index=False)
    print(f"\nwrote reports/backtest_summary.csv and backtest_predictions.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
