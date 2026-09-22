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

Intervals are a week-block bootstrap. Weeks are the unit because rows inside one
share a slate, a weather system and a set of opponents.

    python src/kicker_analysis/backtest.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.model import CompoundKickerModel, RidgeRanker, FEATURES, load  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIRST_TEST = 2018
MIN_WEEKS = 8
N_DRAWS = 300
SEED = 0


def per_week_lift(df: pd.DataFrame, col: str, roster: int, seed: int = SEED):
    """Per-WEEK lift, ceiling rate and oracle lift, computed once.

    Returned per week rather than averaged, because the week is the resampling
    unit and the statistic is a mean over weeks. Bootstrapping this array is
    identical to re-running the simulation on a resampled set of weeks, and it is
    three orders of magnitude cheaper: the old version re-ran 300 draws across
    every week inside every one of 400 resamples, for each of three comparisons.
    """
    rng = np.random.default_rng(seed)
    lift, ceil, orac, rndc = [], [], [], []
    for _, g in df.groupby(["season", "week"], sort=True):
        if len(g) < roster:
            continue
        y, p = g.y.to_numpy(), g[col].to_numpy()
        n = len(g)
        idx = np.array([rng.choice(n, roster, replace=False) for _ in range(N_DRAWS)])
        picks, preds = y[idx], p[idx]
        chosen = picks[np.arange(N_DRAWS), preds.argmax(axis=1)]
        lift.append(chosen.mean() - picks.mean())
        ceil.append((chosen >= 15).mean())
        rndc.append((picks >= 15).mean())
        orac.append(picks.max(axis=1).mean() - picks.mean())
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


def boot_lift(df: pd.DataFrame, a: str, b: str, roster: int = 2, n: int = 4000) -> dict:
    """Paired week-block bootstrap of the difference in mean weekly lift.

    Paired on the WEEK: both rankings are evaluated on the same weeks and the
    same random opponent draws, so the comparison is not polluted by which weeks
    happened to be easy.
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
    print(f"{len(df):,} kicker-games, {len(FEATURES)} named features")
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
    P.to_parquet(REPORTS / "backtest_predictions.parquet", index=False)
    t.to_csv(REPORTS / "backtest_summary.csv", index=False)
    print(f"\nwrote reports/backtest_summary.csv and backtest_predictions.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
