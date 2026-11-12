"""Season-forward backtest: the one command behind the headline numbers.

For every test season from 2018 the models are fit on earlier seasons only and
scored on that season, so nothing from the league's future is ever in a fit.

Scored on the decision, not on the residual. MAE on this target is dominated by
irreducible week-to-week noise and squeezes real differences into the second
decimal, so it is kept and never sorted on. What a manager actually does is
start one kicker out of the two or three on hand, so that is what is measured
(reports/backtest_summary.csv):

  lift_rN    points per week gained by starting the method's favourite of N
             kickers drawn at random from the week, against starting a random
             one of the N.
  share      that lift as a fraction of the oracle's, so the ceiling is visible.
  ceiling    how often the pick returns a 15-point week. A random pick does it
             8.4% of the time and every non-constant ranking here does better,
             the ridge most (9.6%). Ranking by the compound model's expected
             points does not push it below random, and ranking by its P(y>=15)
             instead does not raise it (9.2% either way), so a separate
             ceiling board is not justified by this backtest.
  rho        within-week Spearman, the ordering the lift comes from.
  MAE        the compound model's, for reference.

Every lift is computed EXACTLY. The expectation over every N-subset of a week
has a closed form (_pick_probs), so no comparison carries simulation noise and
a constant scores zero up to floating-point rounding.

Intervals are a week-block bootstrap. Weeks are the unit because rows inside
one share a slate, a weather system and a set of opponents.

The same run writes the evidence for the other claims the documents make:

  component_skill.csv
      Fantasy points are FG points plus PAT points, and the design argument is
      about which half can be forecast. For FG attempts, PAT attempts, FG
      points, PAT points and total points: out-of-sample R-squared against the
      TRAINING seasons' mean (1 - SSE / SSE of that mean, so 0 is "no better than
      the average you already had" and below 0 is worse), with a week-block
      bootstrap interval, and the within-week Spearman (rho below: each week's
      Spearman, averaged over weeks). Three forecasts per target: the shipped
      compound model (count_stage: its LightGBM count stages for the attempt
      targets, the means of its law for the points targets, see
      fold_components), the ridge that orders the board refit on that target
      (ridge16), and the training mean itself (R-squared 0 by construction,
      the reference row).
  variance_decomposition.csv
      var(points) = var(FG points) + var(PAT points) + 2 cov, over every
      modelling row (regular season, 2015 through LAST_TEST).
  season_level_tests.csv
      Each test season (eight in the tracked run) is one observation of the
      edge, however many weeks it contains. Per-season mean differences in
      R=2 lift get a one-sample t-test (df = seasons - 1) with its interval, a
      Wilcoxon signed-rank test and a sign test. The percentile season-cluster
      bootstrap is reported beside them for transparency only: with eight
      clusters it is too narrow, and is not the test to quote.
  calibration_summary.csv, calibration_reliability.csv
      Matching the average rate is necessary and nowhere near sufficient: a
      constant at the training base rate does it too. So for P(y >= 8, 10, 12,
      15): Brier score against that constant (Brier skill, week-block interval),
      the logistic calibration slope (1 is ideal; below 1 means the predictions
      are spread wider than the outcomes support), AUC, and a ten-bin
      reliability table with Wilson intervals.

What the run behind the tracked CSVs says, in brief. FG attempts are close to
unforecastable from these features: out-of-sample R-squared +0.003 for the
count stage (its interval includes 0) and +0.007 for ridge16, within-week
Spearman 0.07 and 0.08. PAT attempts carry real skill, R-squared 0.13 and
Spearman 0.35 for both. FG points carry 102% of the variance of weekly points,
PAT points 10%, and twice their covariance -12%. Averaged by season, the
ridge's edge over the implied total is +0.073 points a week with t-test p 0.095
on 7 df. P(y>=15) matches the realised rate on average, but its Brier skill
against the training base rate is 0.001 (interval includes 0), its AUC 0.55
and its calibration slope 0.55.

Before anything is fit, the inputs are checked: the roof flags vary
(features.check_roof), all 16 features vary, and every feature is at least 85%
non-missing in every season the folds use, prev_* features excepted in the
first (check_coverage).

    python src/kicker_analysis/backtest.py               # full run, writes reports/
    python src/kicker_analysis/backtest.py --self-check  # offline, toy data, writes nothing
"""
from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb, expit, logit
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.features import check_roof  # noqa: E402
from kicker_analysis.model import (CompoundKickerModel, RidgeRanker, FEATURES,  # noqa: E402
                                   N_SIMS, PTS_LONG, PTS_MID, PTS_PAT, PTS_SHORT, load)

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIRST_TEST = 2018
# The backtest is frozen at 2018-2025. 2026 is scored prospectively, from the
# boards published before each week (score_boards.py), so it never becomes a
# fold here and the tracked numbers do not move while the season is played.
LAST_TEST = 2025
MIN_WEEKS = 8
SEED = 0
N_BOOT = 10000
THRESHOLDS = (8, 10, 12, 15)
N_BINS = 10
# Lowest share of non-missing rows any feature may have in any season the folds
# use. Thin lags in week 1 of the first season and rookie kickers keep the true
# value below 1 (the lowest on the full 2015-2025 build is 0.911, career FG% in
# 2015); a season missing from an upstream file puts it near 0.
COVERAGE_FLOOR = 0.85
TARGETS = ("fg_att", "pat_att", "fg_points", "pat_points", "points")
SKILL_MODELS = ("count_stage", "ridge16", "train_mean")
COMPARISONS = [("ridge vs implied total", "ridge", "implied"),
               ("ridge vs career average", "ridge", "career"),
               ("ridge vs compound E[y]", "ridge", "exp_points"),
               ("compound E[y] vs implied total", "exp_points", "implied")]
Z95 = float(stats.norm.ppf(0.975))


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
    unused: nothing here is random.
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
    """Mean over weeks of the within-week Spearman between y and `col`."""
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


# ------------------------------------------------------------------ input guard

def check_coverage(df: pd.DataFrame, features: list[str] | None = None,
                   floor: float = COVERAGE_FLOOR) -> pd.DataFrame:
    """Every feature present, BEFORE imputation, in every season of `df`.

    Both models median-impute inside their fit, so a hole in an input is filled
    without a word, and the std > 0 check cannot see it because the seasons that
    are present still vary. fourth_down.parquet is built from whatever
    play-by-play is cached; built from a partial cache it leaves prev_go_rate
    and t_fg_share_l8 empty for whole seasons, and the backtest then runs clean
    and reports a different headline: with only the 2021 and 2024-2026
    play-by-play cached and every other input the same, the ridge's R=2 lift
    reads 0.4358 instead of 0.4272. This turns that into a failure.

    One gap is legitimate: a prev_* feature describes the season before, so it
    is empty in the first season of the frame and is exempt there. Seasons must
    also be contiguous, since a season missing outright is the same failure one
    level up. Returns the season x feature coverage table.
    """
    features = list(FEATURES if features is None else features)
    seasons = sorted(int(s) for s in df.season.unique())
    gap = sorted(set(range(seasons[0], seasons[-1] + 1)) - set(seasons))
    assert not gap, f"seasons missing from the modelling frame: {gap}"
    cov = df[features].notna().groupby(df.season.to_numpy()).mean()
    bad = [f"{f} {s}: {cov.at[s, f]:.2f}" for f in features for s in seasons
           if not (f.startswith("prev_") and s == seasons[0]) and cov.at[s, f] < floor]
    assert not bad, (f"feature coverage below {floor:.0%} before imputation, so an input "
                     f"was built from partial data: " + "; ".join(bad))
    return cov


# ------------------------------------------------------------- skill and parts

def _blocks(season, week) -> np.ndarray:
    """Week-block id per row, 0..k-1 in (season, week) order."""
    key = pd.DataFrame({"s": np.asarray(season), "w": np.asarray(week)})
    return key.groupby(["s", "w"], sort=True).ngroup().to_numpy()


def skill_score(blocks: np.ndarray, loss: np.ndarray, loss_ref: np.ndarray,
                n: int = N_BOOT) -> dict:
    """1 - sum(loss) / sum(loss_ref), with a week-block bootstrap interval.

    A skill score against a reference forecast. With squared error and the
    training-season mean as the reference it is out-of-sample R-squared; with
    the Brier loss and the training-season base rate it is Brier skill. Zero is
    the reference, negative is worse than it. The ratio is of totals, so weeks
    are resampled whole and the ratio is recomputed per draw.
    """
    lm = np.bincount(blocks, weights=np.asarray(loss, float))
    lr = np.bincount(blocks, weights=np.asarray(loss_ref, float))
    k = len(lm)
    pick = np.random.default_rng(SEED).integers(0, k, size=(n, k))
    draws = 1 - lm[pick].sum(axis=1) / lr[pick].sum(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"skill": float(1 - lm.sum() / lr.sum()), "lo": float(lo), "hi": float(hi)}


def block_mean(blocks: np.ndarray, x: np.ndarray, n: int = N_BOOT) -> dict:
    """Mean of x over rows, with a week-block bootstrap interval."""
    sx = np.bincount(blocks, weights=np.asarray(x, float))
    cnt = np.bincount(blocks).astype(float)
    pick = np.random.default_rng(SEED).integers(0, len(sx), size=(n, len(sx)))
    draws = sx[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(sx.sum() / cnt.sum()), "lo": float(lo), "hi": float(hi)}


def component_points(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """FG points and PAT points per row, by the scoring rule. A miss scores 0."""
    fg = (PTS_SHORT * frame[["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"]].sum(axis=1)
          + PTS_MID * frame["fg_made_40_49"]
          + PTS_LONG * frame[["fg_made_50_59", "fg_made_60_"]].sum(axis=1))
    return fg.to_numpy(float), (PTS_PAT * frame["pat_made"]).to_numpy(float)


def fold_components(tr: pd.DataFrame, te: pd.DataFrame, pred: pd.DataFrame,
                    ridge: np.ndarray) -> pd.DataFrame:
    """One fold's realised components and each forecast of them, row-aligned with te.

    count_stage is the shipped compound model. Its fg_att and pat_att are the
    two LightGBM count stages; its points are the law's mean E[y]; its PAT points
    are its expected PAT attempts, because the law scores every PAT attempt as a
    point; its FG points are E[y] less that, so they carry the Monte Carlo noise
    of E[y]. ridge16 is RidgeRanker, the board's ordering model, refit with the
    component as its target (for points it IS the board's ridge).
    """
    fg_tr, pat_tr = component_points(tr)
    fg_te, pat_te = component_points(te)
    train = {"fg_att": tr.fg_att.to_numpy(float), "pat_att": tr.pat_att.to_numpy(float),
             "fg_points": fg_tr, "pat_points": pat_tr,
             "points": tr.fantasy_points.to_numpy(float)}
    test = {"fg_att": te.fg_att.to_numpy(float), "pat_att": te.pat_att.to_numpy(float),
            "fg_points": fg_te, "pat_points": pat_te,
            "points": te.fantasy_points.to_numpy(float)}
    exp_pat = pred.exp_pat_att.to_numpy(float)
    count = {"fg_att": pred.exp_fg_att.to_numpy(float), "pat_att": exp_pat,
             "fg_points": pred.exp_points.to_numpy(float) - PTS_PAT * exp_pat,
             "pat_points": PTS_PAT * exp_pat, "points": pred.exp_points.to_numpy(float)}
    c = pd.DataFrame({"season": te.season.to_numpy(), "week": te.week.to_numpy()})
    for t in TARGETS:
        c[t] = test[t]
        c[f"{t}|train_mean"] = float(train[t].mean())
        c[f"{t}|count_stage"] = count[t]
        c[f"{t}|ridge16"] = (np.asarray(ridge, float) if t == "points" else
                             RidgeRanker().fit(tr.assign(fantasy_points=train[t])).predict(te))
    return c


def component_skill(C: pd.DataFrame, n: int = N_BOOT) -> pd.DataFrame:
    """component_skill.csv: one row per target x forecast, pooled over every fold."""
    blocks = _blocks(C.season, C.week)
    rows = []
    for t in TARGETS:
        y = C[t].to_numpy(float)
        ref = C[f"{t}|train_mean"].to_numpy(float)
        for m in SKILL_MODELS:
            f = C[f"{t}|{m}"].to_numpy(float)
            s = skill_score(blocks, (y - f) ** 2, (y - ref) ** 2, n)
            frame = pd.DataFrame({"season": C.season, "week": C.week, "y": y, "f": f})
            rows.append({"target": t, "model": m, "oos_r2": s["skill"], "r2_lo": s["lo"],
                         "r2_hi": s["hi"], "within_week_spearman": rho(frame, "f"),
                         "n_rows": len(y)})
    return pd.DataFrame(rows)


def variance_decomposition(fg, pat, total) -> pd.DataFrame:
    """variance_decomposition.csv: var(total) split exactly into its two halves.

    Sample variances (ddof 1). Asserts the scoring identity first, so a change
    to the scoring rule that breaks points = FG points + PAT points fails here
    rather than producing a decomposition that does not add up.
    """
    fg, pat, total = (np.asarray(x, float) for x in (fg, pat, total))
    assert np.allclose(fg + pat, total), "points are not FG points plus PAT points"
    v = {"fg_points": float(np.var(fg, ddof=1)), "pat_points": float(np.var(pat, ddof=1)),
         "two_cov": float(2 * np.cov(fg, pat, ddof=1)[0, 1]),
         "total": float(np.var(total, ddof=1))}
    parts = v["fg_points"] + v["pat_points"] + v["two_cov"]
    assert abs(parts - v["total"]) < 1e-9 * max(1.0, v["total"]), (parts, v["total"])
    return pd.DataFrame({"component": list(v), "variance": list(v.values()),
                         "share_of_total": [x / v["total"] for x in v.values()]})


# ------------------------------------------------------------- season inference

def weekly_lifts(P: pd.DataFrame, cols: list[str], roster: int = 2) -> pd.DataFrame:
    """Exact per-week lift for each column, labelled by season, in per_week_lift order."""
    out = []
    for s in sorted(P.season.unique()):
        g = P[P.season == s]
        out.append(pd.DataFrame({"season": s, **{c: per_week_lift(g, c, roster)[0] for c in cols}}))
    return pd.concat(out, ignore_index=True)


def season_tests(wk: pd.DataFrame, a: str, b: str, n: int = N_BOOT) -> dict:
    """Season-level inference on the R=2 lift difference a - b.

    The unit is the season mean of the weekly difference. t-test (df = seasons
    - 1) and its 95% interval, Wilcoxon signed-rank (exact at this n when no two
    seasons tie), and the two-sided sign test with ties dropped, all on those
    season means. The cluster bootstrap resamples whole seasons, recomputes the
    pooled weekly mean (weeks weighted equally, so its centre can differ a
    little from mean_diff, which weights seasons equally), and takes
    percentiles; it is here so the reader can see how much narrower it is.
    """
    dw = (wk[a] - wk[b]).to_numpy(float)
    by = pd.Series(dw).groupby(wk.season.to_numpy())
    d = by.mean().to_numpy()
    k = len(d)
    se = d.std(ddof=1) / np.sqrt(k)
    t = d.mean() / se
    q = stats.t.ppf(0.975, k - 1)
    wins, nonzero = int((d > 0).sum()), int((d != 0).sum())
    sums, cnt = by.sum().to_numpy(), by.size().to_numpy()
    pick = np.random.default_rng(SEED).integers(0, k, size=(n, k))
    draws = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "t_stat": float(t), "df": k - 1,
            "t_p": float(2 * stats.t.sf(abs(t), k - 1)),
            "t_ci_lo": float(d.mean() - q * se), "t_ci_hi": float(d.mean() + q * se),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
            "seasons_won": wins, "n_seasons": k,
            "sign_p": float(stats.binomtest(wins, nonzero).pvalue) if nonzero else 1.0,
            "cluster_boot_lo": float(lo), "cluster_boot_hi": float(hi),
            "cluster_boot_p": float(min(1.0, 2 * min((draws <= 0).mean(),
                                                     (draws >= 0).mean())))}


def season_level_tests(P: pd.DataFrame, n: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """season_level_tests.csv, plus the per-season differences behind it."""
    cols = sorted({c for _, a, b in COMPARISONS for c in (a, b)})
    wk = weekly_lifts(P, cols)
    rows, per = [], {}
    for name, a, b in COMPARISONS:
        rows.append({"comparison": name, **season_tests(wk, a, b, n)})
        per[name] = (wk[a] - wk[b]).groupby(wk.season).mean()
    return pd.DataFrame(rows), pd.DataFrame(per)


# ----------------------------------------------------------------- calibration

def wilson(k: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n. Treats rows as independent."""
    if n == 0:
        return np.nan, np.nan
    p, z2 = k / n, z * z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, c - h), min(1.0, c + h)


def auc(p: np.ndarray, o: np.ndarray) -> float:
    """Mann-Whitney AUC: P(a positive outranks a negative), ties counted half."""
    p, o = np.asarray(p, float), np.asarray(o, bool)
    r = stats.rankdata(p)
    n1, n0 = int(o.sum()), int((~o).sum())
    return float((r[o].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_slope(p: np.ndarray, o: np.ndarray, eps: float = 0.5 / N_SIMS) -> float:
    """Slope of a logistic regression of the outcome on logit(p), intercept free.

    1 means the predictions are as spread as they should be. Below 1 they are
    too extreme: the high ones too high and the low ones too low. p is a share
    of Monte Carlo draws, so it is clipped half a draw inside 0 and 1 before
    the logit. Fit by Newton-Raphson from the identity (intercept 0, slope 1).
    NaN when there is no finite fit (the outcome separates perfectly on p).
    """
    x = logit(np.clip(np.asarray(p, float), eps, 1 - eps))
    o = np.asarray(o, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(100):
        mu = expit(X @ beta)
        try:
            step = np.linalg.solve(X.T @ (X * (mu * (1 - mu))[:, None]), X.T @ (o - mu))
        except np.linalg.LinAlgError:
            return np.nan
        beta += step
        if np.abs(step).max() < 1e-10:
            return float(beta[1])
    return np.nan


def reliability(p: np.ndarray, o: np.ndarray, bins: int = N_BINS) -> pd.DataFrame:
    """Equal-count bins of predicted probability, lowest first.

    Rows are sorted by p and cut into `bins` runs of equal size (sizes differ by
    at most one); rows tied on p are kept in their frame order, so a tie can
    straddle a cut. The interval is Wilson's, which ignores that rows in one
    week are correlated, so it is somewhat narrow.
    """
    p, o = np.asarray(p, float), np.asarray(o, float)
    which = np.empty(len(p), int)
    which[np.argsort(p, kind="stable")] = np.arange(len(p)) * bins // len(p)
    rows = []
    for i in range(bins):
        m = which == i
        k, nn = float(o[m].sum()), int(m.sum())
        lo, hi = wilson(k, nn)
        rows.append({"bin": i + 1, "n": nn, "pred_mean": float(p[m].mean()),
                     "realised_rate": k / nn, "realised_lo": lo, "realised_hi": hi})
    return pd.DataFrame(rows)


def calibration(P: pd.DataFrame, base_rate: dict, n: int = N_BOOT,
                thresholds=THRESHOLDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """calibration_summary.csv and calibration_reliability.csv.

    base_rate[season][thr] is the share of TRAINING rows at or above thr for the
    fold that tested `season`: the constant forecast a model has to beat.
    """
    blocks = _blocks(P.season, P.week)
    y = P.y.to_numpy(float)
    summ, rel = [], []
    for thr in thresholds:
        p = P[f"p_ge_{thr}"].to_numpy(float)
        o = (y >= thr).astype(float)
        b = np.array([base_rate[int(s)][thr] for s in P.season], float)
        bs = skill_score(blocks, (p - o) ** 2, (b - o) ** 2, n)
        summ.append({"threshold": thr, "pred_mean": float(p.mean()),
                     "realised_rate": float(o.mean()), "ratio": float(p.mean() / o.mean()),
                     "brier": float(((p - o) ** 2).mean()),
                     "brier_base": float(((b - o) ** 2).mean()),
                     "brier_skill": bs["skill"], "brier_skill_lo": bs["lo"],
                     "brier_skill_hi": bs["hi"], "cal_slope": calibration_slope(p, o),
                     "auc": auc(p, o), "n": len(p)})
        rel.append(reliability(p, o).assign(threshold=thr))
    rel = pd.concat(rel, ignore_index=True)[["threshold", "bin", "n", "pred_mean",
                                             "realised_rate", "realised_lo", "realised_hi"]]
    return pd.DataFrame(summ), rel


# ------------------------------------------------------------------------ run

def main() -> int:
    df = load(processed=ROOT / "data" / "processed")
    seasons = sorted(df.season.unique())
    tests = [s for s in seasons if FIRST_TEST <= s <= LAST_TEST
             and df[df.season == s].week.nunique() >= MIN_WEEKS]
    check_roof(df)
    live = [f for f in FEATURES if df[f].std() > 0]
    assert len(live) == len(FEATURES), sorted(set(FEATURES) - set(live))
    # Coverage over the seasons the folds train or test on. A season still in
    # progress and not yet a test season is never used, so it is not held to it.
    cov = check_coverage(df[df.season <= max(tests)])
    held = cov.copy()
    held.loc[cov.index.min(), [f for f in cov.columns if f.startswith("prev_")]] = np.nan
    print(f"{len(df):,} kicker-games, {len(FEATURES)} named features, "
          f"{len(live)} non-constant, each at least {held.min().min():.1%} non-missing "
          f"in every season {int(seasons[0])}-{int(max(tests))} "
          f"(prev_* exempt in {int(seasons[0])})")
    print(f"test seasons: {[int(s) for s in tests]}\n")

    out, parts, base_rate = [], [], {}
    for s in tests:
        tr, te = df[df.season < s], df[df.season == s]
        pred = CompoundKickerModel().fit(tr).predict(te)
        # The ridge is what ORDERS the published board, so it is the headline row.
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
        parts.append(fold_components(tr, te, pred, ridge))
        base_rate[int(s)] = {thr: float((tr.fantasy_points >= thr).mean()) for thr in THRESHOLDS}
    P = pd.concat(out, ignore_index=True)
    C = pd.concat(parts, ignore_index=True)

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

    seas, per_season = season_level_tests(P)
    print(f"\nSeason-level tests, R=2 lift, {len(per_season)} seasons as the unit "
          f"(per-season differences, then t / Wilcoxon / sign; the percentile cluster "
          f"bootstrap is shown for comparison and under-covers at this many clusters):")
    print(per_season.round(3).T.to_string())
    print()
    print(seas.round(4).to_string(index=False))

    skill = component_skill(C)
    print(f"\nComponent skill, {len(C):,} test rows. oos_r2 is 1 - SSE / SSE of the "
          f"training-season mean, with a week-block 95% interval:")
    print(skill.round(4).to_string(index=False))

    hist = df[df.season <= LAST_TEST]
    fg_all, pat_all = component_points(hist)
    vd = variance_decomposition(fg_all, pat_all, hist.fantasy_points)
    print(f"\nVariance of weekly points, {len(hist):,} kicker-games "
          f"{int(hist.season.min())}-{int(hist.season.max())}:")
    print(vd.round(4).to_string(index=False))

    cal, rel = calibration(P, base_rate)
    print("\nCalibration of the predictive law. brier_base is a constant at the "
          "training seasons' base rate; brier_skill = 1 - brier / brier_base:")
    print(cal.round(4).to_string(index=False))
    blocks = _blocks(P.season, P.week)
    for thr in THRESHOLDS:
        bias = block_mean(blocks, P[f"p_ge_{thr}"] - (P.y >= thr).astype(float))
        print(f"  P(y>={thr:2}) predicted minus realised {bias['mean']:+.4f}, "
              f"week-block 95% CI [{bias['lo']:+.4f}, {bias['hi']:+.4f}]")
    print("\nReliability of P(y>=15), ten equal-count bins:")
    print(rel[rel.threshold == 15].round(4).to_string(index=False))

    REPORTS.mkdir(parents=True, exist_ok=True)
    # predict_week reads this file, possibly while this runs: write, then swap.
    tmp = REPORTS / "backtest_predictions.parquet.tmp"
    P.to_parquet(tmp, index=False)
    os.replace(tmp, REPORTS / "backtest_predictions.parquet")
    t.to_csv(REPORTS / "backtest_summary.csv", index=False)
    skill.to_csv(REPORTS / "component_skill.csv", index=False)
    vd.to_csv(REPORTS / "variance_decomposition.csv", index=False)
    seas.to_csv(REPORTS / "season_level_tests.csv", index=False)
    cal.to_csv(REPORTS / "calibration_summary.csv", index=False)
    rel.to_csv(REPORTS / "calibration_reliability.csv", index=False)
    print("\nwrote reports/backtest_summary.csv, component_skill.csv, "
          "variance_decomposition.csv, season_level_tests.csv, calibration_summary.csv, "
          "calibration_reliability.csv and backtest_predictions.parquet")
    return 0


# ------------------------------------------------------------------ self-check

def _raises(fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except AssertionError:
        return True
    return False


def self_check() -> int:
    """Every statistic above on toy data with a known answer. No data, no writes."""
    rng = np.random.default_rng(1)

    # The closed form against brute-force enumeration of every roster, with
    # ties split evenly among the tied leaders.
    for roster in (2, 3, 4):
        for _ in range(20):
            s = rng.integers(0, 4, size=7).astype(float)
            subsets = list(itertools.combinations(range(len(s)), roster))
            brute = np.zeros(len(s))
            for sub in map(np.array, subsets):
                top = sub[s[sub] == s[sub].max()]
                brute[top] += 1.0 / len(top)
            got = _pick_probs(s, roster)
            assert np.allclose(got, brute / len(subsets)), (roster, s, got)
            assert abs(got.sum() - 1) < 1e-12

    # Toy weeks with integer points, so y itself has ties.
    T = pd.DataFrame({"season": 2020, "week": np.repeat(np.arange(1, 13), 10),
                      "y": rng.integers(0, 20, 120).astype(float)})
    T["perfect"], T["reverse"], T["constant"] = T.y, -T.y, 8.0
    T["noise"] = rng.normal(size=len(T))
    T["noisy"] = T.y + rng.normal(0, 5, len(T))
    for R in (2, 3):
        orc = decision(T, "perfect", R)
        worst = decision(T, "reverse", R)["lift"]
        assert orc["oracle_lift"] > 0 and abs(orc["lift"] - orc["oracle_lift"]) < 1e-12
        assert abs(orc["share"] - 1) < 1e-12
        assert abs(decision(T, "constant", R)["lift"]) < 1e-12
        for c in ("noise", "noisy", "reverse"):
            lift = decision(T, c, R)["lift"]
            assert worst - 1e-12 <= lift <= orc["oracle_lift"] + 1e-12, (c, R, lift)
        # At R=2 picking the worse of a pair is the oracle's exact mirror, so
        # every ranking lies in [-oracle, +oracle]. At R=3 the floor is the
        # reversed ranking, which can sit below -oracle when y is skewed.
        if R == 2:
            assert abs(worst + orc["oracle_lift"]) < 1e-12
    assert decision(T, "noisy", 2)["lift"] > decision(T, "noise", 2)["lift"]

    same = boot_lift(T, "noisy", "noisy", 2, n=500)
    assert same["diff"] == same["lo"] == same["hi"] == 0.0 and same["p"] == 1.0, same
    best = boot_lift(T, "perfect", "constant", 2, n=500)
    assert abs(best["diff"] - decision(T, "perfect", 2)["lift"]) < 1e-12
    assert 0 < best["lo"] <= best["diff"] <= best["hi"] and best["p"] == 0.0, best

    # Coverage: a prev_* feature empty in the first season is fine; a feature
    # half missing in a later season fails; so does a season missing outright.
    cv = pd.DataFrame({"season": np.repeat([2015, 2016, 2017], 20),
                       "prev_x": np.r_[np.full(20, np.nan), np.ones(40)],
                       "lag_y": np.ones(60)})
    assert check_coverage(cv, ["prev_x", "lag_y"]).loc[2015, "prev_x"] == 0.0
    assert _raises(check_coverage, cv.assign(lag_y=np.r_[np.ones(30), np.full(10, np.nan),
                                                         np.ones(20)]), ["prev_x", "lag_y"])
    assert _raises(check_coverage, cv.assign(prev_x=1.0, lag_y=np.r_[np.full(20, np.nan),
                                                                     np.ones(40)]),
                   ["prev_x", "lag_y"])
    assert _raises(check_coverage, cv[cv.season != 2016], ["prev_x", "lag_y"])

    # Skill score: y = 1..4 against a reference of 2.5 has SSE 5; the forecast
    # (1.5, 2, 3, 3.5) has SSE 0.5, so R-squared is 0.9 exactly.
    y, ref = np.array([1.0, 2, 3, 4]), np.full(4, 2.5)
    blk = np.array([0, 0, 1, 1])
    sk = skill_score(blk, (y - np.array([1.5, 2, 3, 3.5])) ** 2, (y - ref) ** 2, n=200)
    assert abs(sk["skill"] - 0.9) < 1e-12 and sk["lo"] <= 0.9 <= sk["hi"], sk
    assert skill_score(blk, (y - ref) ** 2, (y - ref) ** 2, n=200) == {
        "skill": 0.0, "lo": 0.0, "hi": 0.0}
    assert skill_score(blk, np.zeros(4), (y - ref) ** 2, n=200)["skill"] == 1.0
    bm = block_mean(np.array([0, 0, 1, 2]), np.array([1.0, 3, 2, 6]), n=500)
    assert bm["mean"] == 3.0 and 2.0 <= bm["lo"] <= bm["hi"] <= 6.0, bm
    assert block_mean(blk, np.full(4, 0.25), n=200) == {"mean": 0.25, "lo": 0.25, "hi": 0.25}

    # Component skill end to end, on the contract's columns. The training mean
    # is the reference, so its own row is exactly 0 and has no ordering.
    C = pd.DataFrame({"season": 2020, "week": np.repeat(np.arange(6), 8)})
    for tname in TARGETS:
        C[tname] = rng.poisson(3, len(C)).astype(float)
        C[f"{tname}|train_mean"] = 3.0
        C[f"{tname}|count_stage"] = C[tname] + rng.normal(0, 0.5, len(C))
        C[f"{tname}|ridge16"] = 3.0 + rng.normal(0, 1, len(C))
    cs = component_skill(C, n=200).set_index(["target", "model"])
    assert list(cs.reset_index().columns) == ["target", "model", "oos_r2", "r2_lo", "r2_hi",
                                              "within_week_spearman", "n_rows"]
    assert (cs.xs("train_mean", level="model").oos_r2 == 0).all()
    assert cs.xs("train_mean", level="model").within_week_spearman.isna().all()
    assert (cs.xs("count_stage", level="model").oos_r2 > 0.5).all()
    assert (cs.xs("ridge16", level="model").oos_r2 < 0).all()

    # Scoring split: one make in each distance bucket, then a row of PATs only.
    # Each bucket lands in its own tier, and a 60+ make is a long one.
    buckets = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49",
               "fg_made_50_59", "fg_made_60_"]
    toy = pd.DataFrame(np.eye(7, 6, dtype=int), columns=buckets).assign(pat_made=[0] * 6 + [3])
    fgp, patp = component_points(toy)
    assert fgp.tolist() == [PTS_SHORT] * 3 + [PTS_MID] + [PTS_LONG] * 2 + [0.0], fgp
    assert patp.tolist() == [0.0] * 6 + [3 * PTS_PAT], patp

    # Variance decomposition: the parts add up exactly, the covariance carries
    # its sign, and a total that is not FG + PAT is refused.
    fg, pat = np.array([3.0, 0, 6, 3, 0, 9]), np.array([1.0, 4, 0, 2, 3, 0])
    vd = variance_decomposition(fg, pat, fg + pat).set_index("component")
    assert abs(vd.share_of_total.drop("total").sum() - 1) < 1e-12
    assert abs(vd.at["two_cov", "variance"] - 2 * np.cov(fg, pat)[0, 1]) < 1e-12
    assert vd.at["two_cov", "variance"] < 0 and vd.at["fg_points", "variance"] == np.var(fg, ddof=1)
    assert _raises(variance_decomposition, fg, pat, fg + pat + 1)

    # Season-level tests on eight season differences, two of them losses and
    # one of those the smallest in size, ten weeks each. Exact answers by
    # hand: Wilcoxon W- = 1 + 4 = 5, and 10 of the 256 sign patterns have
    # W <= 5, so p = 20/256; the sign test is 6 of 8, p = 2 * 37/256.
    dd = np.array([0.0503, -0.1103, 0.1106, -0.0027, 0.042, 0.1791, 0.1358, 0.2167])
    wk = pd.DataFrame({"season": np.repeat(np.arange(2018, 2026), 10),
                       "a": np.repeat(dd, 10), "b": 0.0})
    st = season_tests(wk, "a", "b", n=2000)
    tt = stats.ttest_1samp(dd, 0.0)
    assert st["df"] == 7 and st["n_seasons"] == 8 and st["seasons_won"] == 6
    assert abs(st["mean_diff"] - dd.mean()) < 1e-12
    assert abs(st["t_stat"] - tt.statistic) < 1e-9 and abs(st["t_p"] - tt.pvalue) < 1e-9
    assert abs(st["wilcoxon_p"] - 20 / 256) < 1e-12, st["wilcoxon_p"]
    assert abs(st["sign_p"] - 74 / 256) < 1e-12, st["sign_p"]
    ci = tt.confidence_interval(0.95)
    assert abs(st["t_ci_lo"] - ci.low) < 1e-9 and abs(st["t_ci_hi"] - ci.high) < 1e-9, (st, ci)
    assert st["t_ci_lo"] < 0 < st["t_ci_hi"] and st["t_p"] > 0.05
    # The percentile bootstrap over eight clusters is the narrower interval,
    # which is why it is not the one to quote.
    assert dd.min() <= st["cluster_boot_lo"] < st["cluster_boot_hi"] <= dd.max()
    assert st["cluster_boot_hi"] - st["cluster_boot_lo"] < st["t_ci_hi"] - st["t_ci_lo"]
    one = season_tests(wk.assign(a=np.abs(wk.a)), "a", "b", n=500)
    assert one["seasons_won"] == 8 and abs(one["sign_p"] - 2 / 256) < 1e-12
    assert abs(one["wilcoxon_p"] - 2 / 256) < 1e-12 and one["cluster_boot_p"] == 0.0

    # Calibration pieces. AUC against counting every positive-negative pair.
    p = rng.integers(0, 5, 60) / 4.0
    o = rng.random(60) < np.clip(p, 0.1, 0.9)
    pairs = [(a > b) + 0.5 * (a == b) for a in p[o] for b in p[~o]]
    assert abs(auc(p, o) - np.mean(pairs)) < 1e-12
    # Outcomes drawn from p itself give slope 1; doubling every logit makes
    # the forecasts twice as extreme, so the slope halves.
    q = expit(rng.normal(-1, 1, 20000))
    oq = (rng.random(20000) < q).astype(float)
    assert abs(calibration_slope(q, oq) - 1) < 0.1
    assert abs(calibration_slope(expit(2 * logit(q)), oq) - 0.5) < 0.05
    # Wilson: 0 of n runs from 0 to z^2 / (n + z^2), and k of n mirrors n - k.
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - Z95 ** 2 / (10 + Z95 ** 2)) < 1e-12
    assert abs(wilson(3, 10)[0] - (1 - wilson(7, 10)[1])) < 1e-12
    rb = reliability(q[:1003], oq[:1003])
    assert rb.n.sum() == 1003 and set(rb.n) <= {100, 101}
    assert rb.pred_mean.is_monotonic_increasing
    assert ((rb.realised_lo <= rb.realised_rate) & (rb.realised_rate <= rb.realised_hi)).all()

    # Calibration end to end: a forecast equal to the training base rate has
    # Brier skill 0 exactly; the outcome itself has skill 1 and AUC 1.
    Pc = pd.DataFrame({"season": np.repeat([2020, 2021], 60),
                       "week": np.tile(np.repeat(np.arange(6), 10), 2),
                       "y": rng.integers(0, 20, 120).astype(float)})
    base = {2020: {thr: 0.3 for thr in THRESHOLDS}, 2021: {thr: 0.4 for thr in THRESHOLDS}}
    for thr in THRESHOLDS:
        Pc[f"p_ge_{thr}"] = np.where(Pc.season == 2020, 0.3, 0.4)
    cal, rel = calibration(Pc, base, n=200)
    assert list(cal.columns) == ["threshold", "pred_mean", "realised_rate", "ratio", "brier",
                                 "brier_base", "brier_skill", "brier_skill_lo",
                                 "brier_skill_hi", "cal_slope", "auc", "n"]
    assert list(rel.columns) == ["threshold", "bin", "n", "pred_mean", "realised_rate",
                                 "realised_lo", "realised_hi"]
    assert (cal.brier_skill == 0).all() and len(rel) == len(THRESHOLDS) * N_BINS
    for thr in THRESHOLDS:
        Pc[f"p_ge_{thr}"] = (Pc.y >= thr).astype(float)
    cal, _ = calibration(Pc, base, n=200)
    assert (cal.brier_skill == 1).all() and (cal.auc == 1).all()

    # Season-level output carries the contract's columns, one row per comparison.
    Ps = pd.DataFrame({"season": np.repeat(np.arange(2018, 2022), 30),
                       "week": np.tile(np.repeat(np.arange(3), 10), 4),
                       "y": rng.integers(0, 20, 120).astype(float)})
    for c in ("ridge", "implied", "career", "exp_points"):
        Ps[c] = Ps.y + rng.normal(0, 8, len(Ps))
    sl, per = season_level_tests(Ps, n=200)
    assert list(sl.columns) == ["comparison", "mean_diff", "t_stat", "df", "t_p", "t_ci_lo",
                                "t_ci_hi", "wilcoxon_p", "seasons_won", "n_seasons",
                                "sign_p", "cluster_boot_lo", "cluster_boot_hi",
                                "cluster_boot_p"]
    assert list(sl.comparison) == [c[0] for c in COMPARISONS] and per.shape == (4, 4)
    print("backtest self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Season-forward backtest of the kicker models.")
    ap.add_argument("--self-check", action="store_true",
                    help="offline checks on toy data; reads no data and writes nothing")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main())
