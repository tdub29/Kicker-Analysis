"""Model-selection bake-off for the kicker ranker: does anything beat the ridge?

Every candidate is tuned ONLY on seasons before its test season. The inner loop
is itself season-forward: for test season s, each of the last (up to) three
training seasons serves once as validation for a fit on the seasons before it,
the configuration with the best mean inner score is refit on every season
before s, and it is scored once on s. Every preprocessing step (median
imputation, scaling, spline knots, pair construction, the empirical-Bayes
variance components, the Vegas offset) is fit inside the training block.

The inner AND outer score is the EXACT expected R=2 decision lift: over every
pair of kickers in a week, the gain from starting the one the score prefers.
backtest.per_week_lift computes it in closed form for any roster size, so tuning
and evaluation aim at the same number and no comparison carries draw noise. (The
first run scored the outer folds with a 300-draw simulation; a reviewer caught it.)

Evaluation is the repo's harness: backtest.decision and backtest.per_week_lift
for the lift, backtest.rho for within-week Spearman, backtest.boot_lift for the
paired week-block bootstrap against the shipped ridge. Rows are laid out in the
order backtest.py uses, so the reference rows reproduce its numbers exactly.

Two multiplicity families, fixed before the run:
  challengers  the 14 candidates vs the shipped ridge, Holm m = 14. A challenger
               wins only if Holm p < 0.05, the difference is positive, AND it
               beats the ridge in at least 6 of the 8 test seasons.
  baselines    the two claims the README already made, ridge vs the Vegas
               implied total and ridge vs the kicker's career average, Holm
               m = 2. Each also gets a season-cluster bootstrap (whole seasons
               resampled, 8 clusters) and a two-sided sign test across seasons,
               because 141 weeks are not 141 independent draws when the
               ridge's own lift swings 0.19 to 0.63 by season.
The compound model and the constant are reported unadjusted, as description.

This file only reads. It does not touch the production model or the board.

    python src/kicker_analysis/model_bakeoff.py            # full run, 8 folds
    python src/kicker_analysis/model_bakeoff.py --quick    # 3 folds, 1 inner fold, smoke test
    python src/kicker_analysis/model_bakeoff.py --self-check
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.backtest import (FIRST_TEST, MIN_WEEKS, boot_lift,  # noqa: E402
                                      decision, per_week_lift, rho)
from kicker_analysis.features import check_roof  # noqa: E402
from kicker_analysis.model import (CompoundKickerModel, FEATURES,  # noqa: E402
                                   RidgeRanker, load)

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SEED = 0
KEY = ["season", "week"]
THRESHOLDS = (8, 10, 12, 15)

# Grids run from most to least regularised, so an inner tie goes to the simpler fit.
ALPHAS = [10000, 3000, 1000, 300, 100, 30, 10, 3, 1]
GLM_ALPHAS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.001, 0.0001]
# The nonlinearity test: the four inputs a kicker's volume plausibly bends on.
SPLINE_COLS = ["implied_total", "total_line", "climate_wind_exposed",
               "t_fg_att_l8", "t_pat_att_l8"]


# ---------------------------------------------------------------- shared pieces
def _prep(train: pd.DataFrame):
    """Median impute + standardise, fit on the training block only."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler()).fit(train[FEATURES])


def exact_lift(season, week, y, score) -> float:
    """Exact expected R=2 lift, the harness's own closed form, as one number."""
    d = pd.DataFrame({"season": np.asarray(season), "week": np.asarray(week),
                      "y": np.asarray(y, float), "s": np.asarray(score, float)})
    return float(per_week_lift(d, "s", 2)[0].mean())


def season_cluster(wk: pd.DataFrame, a: str, b: str, n: int) -> dict:
    """Resample whole SEASONS with replacement; the statistic is the pooled
    weekly mean of a - b over the resampled weeks, the same statistic as the
    week-block bootstrap, with the dependence inside a season kept intact."""
    d = (wk[a] - wk[b]).to_numpy()
    by = wk.groupby("season").indices
    sums = np.array([d[i].sum() for i in by.values()])
    cnt = np.array([len(i) for i in by.values()])
    pick = np.random.default_rng(SEED).integers(0, len(sums), size=(n, len(sums)))
    draws = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"season_lo": float(lo), "season_hi": float(hi),
            "season_p": float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))}


def sign_test(fa: pd.Series, fb: pd.Series) -> dict:
    """Two-sided exact sign test on per-season lifts; ties dropped."""
    from scipy.stats import binomtest
    d = (fa - fb).to_numpy()
    wins, k = int((d > 0).sum()), int((d != 0).sum())
    return {"seasons_won": wins, "sign_p": float(binomtest(wins, k).pvalue) if k else 1.0}


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj
    return out


def eb_effects(ids, res):
    """One-way random-effects shrinkage of per-kicker mean residuals.

    tau2 is the ANOVA method-of-moments between-kicker variance, sigma2 the
    pooled within-kicker variance. A kicker with n games keeps
    n*tau2 / (n*tau2 + sigma2) of his own mean residual.
    """
    d = pd.DataFrame({"k": ids, "r": res})
    g = d.groupby("k").r.agg(["mean", "size"])
    N, K = len(d), len(g)
    sigma2 = float(((d.r - d.k.map(g["mean"])) ** 2).sum() / (N - K))
    grand = float(d.r.mean())
    ms_between = float((g["size"] * (g["mean"] - grand) ** 2).sum() / (K - 1))
    n0 = (N - float((g["size"] ** 2).sum()) / N) / (K - 1)
    tau2 = max(0.0, (ms_between - sigma2) / n0)
    shrink = g["size"] * tau2 / (g["size"] * tau2 + sigma2)
    return (shrink * (g["mean"] - grand)).to_dict(), tau2, sigma2


# ---------------------------------------------------------------- candidates
def f_ridge(tr, te, alpha=100.0):
    return RidgeRanker(alpha=alpha).fit(tr).predict(te)


def f_vegas_resid(tr, te, alpha):
    """Offset = OLS on the implied total; ridge only explains what the line misses.
    Shrinkage now pulls toward the betting line instead of toward a constant."""
    from sklearn.linear_model import Ridge
    fill = float(tr.implied_total.median())
    b = np.polyfit(tr.implied_total.fillna(fill), tr.fantasy_points, 1)
    base_tr = np.polyval(b, tr.implied_total.fillna(fill))
    base_te = np.polyval(b, te.implied_total.fillna(fill))
    p = _prep(tr)
    m = Ridge(alpha=alpha).fit(p.transform(tr[FEATURES]), tr.fantasy_points - base_tr)
    return base_te + m.predict(p.transform(te[FEATURES]))


def _glm(tr, alpha, power):
    from sklearn.linear_model import TweedieRegressor
    p = _prep(tr)
    m = TweedieRegressor(power=power, link="log", alpha=alpha, max_iter=3000)
    return p, m.fit(p.transform(tr[FEATURES]), tr.fantasy_points)


def f_glm(tr, te, alpha, power):
    """power 1 is the Poisson GLM; 1 < power < 2 is compound Poisson-gamma."""
    p, m = _glm(tr, alpha, power)
    return m.predict(p.transform(te[FEATURES]))


def glm_tail(tr, te, alpha, power, sims=2000) -> dict:
    """P(y >= thr) implied by the fitted GLM's own distribution."""
    from scipy.stats import poisson
    p, m = _glm(tr, alpha, power)
    mu_te = m.predict(p.transform(te[FEATURES]))
    if power == 1.0:
        return {t: poisson.sf(t - 1, mu_te) for t in THRESHOLDS}
    mu_tr = m.predict(p.transform(tr[FEATURES]))
    phi = float(((tr.fantasy_points - mu_tr) ** 2 / mu_tr ** power).sum()
                / (len(tr) - len(FEATURES) - 1))
    rng = np.random.default_rng(SEED)
    lam = mu_te ** (2 - power) / (phi * (2 - power))
    shape = (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu_te ** (power - 1)
    n = rng.poisson(np.repeat(lam[:, None], sims, axis=1))
    draw = np.where(n > 0, rng.gamma(np.maximum(n * shape, 1e-12), scale[:, None]), 0.0)
    # The target is integer points, the law is continuous: half-point correction.
    return {t: (draw >= t - 0.5).mean(axis=1) for t in THRESHOLDS}


def f_spline(tr, te, alpha, knots):
    """Cubic B-splines (linear beyond the boundary knots, the natural-spline
    tail) on five inputs, linear terms on the other eleven, one ridge on all."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    lin = [c for c in FEATURES if c not in SPLINE_COLS]
    ct = ColumnTransformer([
        ("spl", make_pipeline(SimpleImputer(strategy="median"),
                              SplineTransformer(n_knots=knots, degree=3,
                                                extrapolation="linear")), SPLINE_COLS),
        ("lin", SimpleImputer(strategy="median"), lin)])
    m = make_pipeline(ct, StandardScaler(), Ridge(alpha=alpha))
    return m.fit(tr[FEATURES], tr.fantasy_points).predict(te[FEATURES])


def f_pairwise(tr, te, C):
    """Linear pairwise ranker: logistic on within-week feature differences,
    each pair weighted by |y_i - y_j| so the loss is the decision's own stake."""
    from sklearn.linear_model import LogisticRegression
    p = _prep(tr)
    X = p.transform(tr[FEATURES])
    y = tr.fantasy_points.to_numpy(float)
    rng = np.random.default_rng(SEED)
    D, L, W = [], [], []
    for idx in tr.groupby(KEY, sort=True).indices.values():
        i, j = np.triu_indices(len(idx), 1)
        a, b = idx[i], idx[j]
        dy = y[a] - y[b]
        keep = dy != 0
        a, b, dy = a[keep], b[keep], dy[keep]
        flip = rng.random(len(a)) < 0.5            # balance the orientation
        a, b = np.where(flip, b, a), np.where(flip, a, b)
        D.append(X[a] - X[b])
        L.append((y[a] > y[b]).astype(int))
        W.append(np.abs(dy))
    w = np.concatenate(W)
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000)
    m.fit(np.vstack(D), np.concatenate(L), sample_weight=w / w.mean())
    return p.transform(te[FEATURES]) @ m.coef_.ravel()


def f_lambdarank(tr, te, leaves, n_est):
    """LightGBM lambdarank grouped by week, gain linear in points, shallow."""
    from lightgbm import LGBMRanker
    t = tr.sort_values(KEY, kind="mergesort")
    groups = t.groupby(KEY, sort=True).size().to_numpy()
    m = LGBMRanker(objective="lambdarank", n_estimators=n_est, num_leaves=leaves,
                   max_depth=2 if leaves <= 3 else 3, learning_rate=0.03,
                   min_child_samples=100, reg_lambda=10.0, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8,
                   label_gain=list(range(64)), lambdarank_truncation_level=64,
                   random_state=SEED, n_jobs=1, deterministic=True,
                   force_row_wise=True, verbose=-1)
    m.fit(t[FEATURES], t.fantasy_points.astype(int).clip(0, 63), group=groups)
    return m.predict(te[FEATURES])


def f_eb(tr, te):
    """Shipped ridge plus a shrunk per-kicker residual from prior seasons only.
    No tuning: the shrinkage is estimated, not searched."""
    r = RidgeRanker().fit(tr)
    res = tr.fantasy_points.to_numpy() - r.m_.predict(tr[FEATURES])
    eff, _, _ = eb_effects(tr.player_id.to_numpy(), res)
    return r.m_.predict(te[FEATURES]) + te.player_id.map(eff).fillna(0.0).to_numpy()


def f_mono(tr, te, n_est, mcs):
    """Depth-2 GBM, heavy regularisation, monotone increasing in implied total."""
    from lightgbm import LGBMRegressor
    mono = [1 if f == "implied_total" else 0 for f in FEATURES]
    m = LGBMRegressor(objective="regression", n_estimators=n_est, max_depth=2,
                      num_leaves=3, learning_rate=0.03, min_child_samples=mcs,
                      reg_lambda=10.0, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, monotone_constraints=mono,
                      random_state=SEED, n_jobs=1, deterministic=True,
                      force_row_wise=True, verbose=-1)
    return m.fit(tr[FEATURES], tr.fantasy_points).predict(te[FEATURES])


def f_blend(tr, te, w):
    """w * within-week rank of the shipped ridge + (1-w) * rank of the line."""
    r = RidgeRanker().fit(tr).predict(te)
    d = pd.DataFrame({"season": te.season.to_numpy(), "week": te.week.to_numpy(),
                      "r": r, "v": te.implied_total.to_numpy()})
    g = d.groupby(KEY)
    return (w * g.r.rank(pct=True) + (1 - w) * g.v.rank(pct=True)).to_numpy()


def f_within(tr, te, alpha):
    """Ridge on week-demeaned features and target. The decision is within a
    week, so between-week variation is noise to a ranker; this removes it."""
    from sklearn.linear_model import Ridge
    p = _prep(tr)
    X = pd.DataFrame(p.transform(tr[FEATURES]), index=tr.index)
    keys = [tr.season, tr.week]
    Xd = X - X.groupby(keys).transform("mean")
    yd = tr.fantasy_points - tr.fantasy_points.groupby(keys).transform("mean")
    m = Ridge(alpha=alpha, fit_intercept=False).fit(Xd.to_numpy(), yd.to_numpy())
    return p.transform(te[FEATURES]) @ m.coef_


def f_enet(tr, te, alpha, l1):
    """Sparse linear fit: can fewer than 16 columns tie the ridge?"""
    from sklearn.linear_model import ElasticNet
    p = _prep(tr)
    m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000)
    return m.fit(p.transform(tr[FEATURES]), tr.fantasy_points).predict(p.transform(te[FEATURES]))


def f_decomp(tr, te, a_fg, a_pat):
    """METHOD.md's decomposition as a model: one ridge for field-goal points
    (no skill, so it should want heavy shrinkage), one for extra points (real
    skill, lighter shrinkage), summed. Still linear, different penalty per half."""
    fg = RidgeRanker(alpha=a_fg).fit(tr.assign(fantasy_points=tr.fantasy_points - tr.pat_made))
    pat = RidgeRanker(alpha=a_pat).fit(tr.assign(fantasy_points=tr.pat_made))
    return fg.m_.predict(te[FEATURES]) + pat.m_.predict(te[FEATURES])


# The roof flags were dead until 2026-09-29 (features.roof_flags keyed the
# building on the kicker's team, so both flags were constant). They are repaired
# in features.py, so the shipped ridge now carries them. This challenger IS the
# ridge as it shipped before the fix: without the two flags, which a scaler
# zeroed anyway, and with the wind exposure that never saw a dome.
PREFIX_FEATURES = ([f for f in FEATURES if f not in ("is_indoor_certain", "is_retractable",
                                                    "climate_wind_exposed")]
                   + ["climate_wind_prefix"])


def f_prefix_roof(tr, te):
    return RidgeRanker(features=PREFIX_FEATURES).fit(tr).predict(te)


CANDS = {
    "ridge_nested": (f_ridge, [{"alpha": a} for a in ALPHAS]),
    "vegas_resid": (f_vegas_resid, [{"alpha": a} for a in ALPHAS]),
    "poisson_glm": (f_glm, [{"alpha": a, "power": 1.0} for a in GLM_ALPHAS]),
    "tweedie_glm": (f_glm, [{"alpha": a, "power": pw} for pw in (1.2, 1.5, 1.8)
                            for a in GLM_ALPHAS]),
    "spline_ridge": (f_spline, [{"alpha": a, "knots": k} for k in (3, 5) for a in ALPHAS]),
    "pairwise_linear": (f_pairwise, [{"C": c} for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]),
    "lambdarank_gbm": (f_lambdarank, [{"leaves": lv, "n_est": n} for lv in (3, 7)
                                      for n in (100, 300)]),
    "eb_kicker": (f_eb, [{}]),
    "mono_gbm": (f_mono, [{"n_est": n, "mcs": m} for m in (200, 50) for n in (100, 300, 600)]),
    "blend_rank": (f_blend, [{"w": float(w)} for w in np.round(np.linspace(1, 0, 11), 1)]),
    "within_week_ridge": (f_within, [{"alpha": a} for a in ALPHAS]),
    "elastic_net": (f_enet, [{"alpha": a, "l1": l1} for l1 in (1.0, 0.5)
                             for a in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003)]),
    "decomposed_ridge": (f_decomp, [{"a_fg": f, "a_pat": p} for f in ALPHAS for p in ALPHAS]),
    "ridge_prefix_roof": (f_prefix_roof, [{}]),
}

LABEL = {
    "ridge": "ridge, 16 features, alpha 100 (shipped)",
    "ridge_nested": "ridge, nested alpha",
    "vegas_resid": "Vegas-anchored residual ridge",
    "poisson_glm": "L2 Poisson GLM",
    "tweedie_glm": "L2 Tweedie GLM",
    "spline_ridge": "spline ridge (GAM-style)",
    "pairwise_linear": "linear pairwise ranker",
    "lambdarank_gbm": "LightGBM lambdarank",
    "eb_kicker": "ridge + EB kicker effect",
    "mono_gbm": "monotone depth-2 GBM",
    "blend_rank": "rank blend ridge/implied",
    "within_week_ridge": "within-week demeaned ridge",
    "elastic_net": "elastic net",
    "decomposed_ridge": "decomposed ridge (FG + PAT)",
    "ridge_prefix_roof": "ridge, roof flags dead (as shipped before the fix)",
    "exp_points": "compound count model E[y]",
    "implied": "Vegas implied total alone",
    "career": "kicker career average",
    "constant": "constant (training median)",
}
FAMILY = {**{c: "challengers" for c in CANDS}, "implied": "baselines", "career": "baselines"}
# Columns whose output is on the points scale, so MAE and calibration mean something.
POINTS_SCALE = {"ridge", "ridge_nested", "vegas_resid", "poisson_glm", "tweedie_glm",
                "spline_ridge", "eb_kicker", "mono_gbm", "elastic_net",
                "decomposed_ridge", "ridge_prefix_roof", "exp_points", "career", "constant"}


def inner_splits(train: pd.DataFrame, n_inner: int):
    seasons = sorted(train.season.unique())
    return [(train[train.season < v], train[train.season == v])
            for v in seasons[1:][-n_inner:]]


def tune(fit, grid, train, n_inner):
    if len(grid) == 1:
        return grid[0]
    splits = inner_splits(train, n_inner)
    scores = [np.mean([exact_lift(va.season, va.week, va.fantasy_points,
                                  fit(tr, va, **g)) for tr, va in splits]) for g in grid]
    return grid[int(np.argmax(scores))]


# ---------------------------------------------------------------- the run
def run(quick: bool) -> int:
    t0 = time.time()
    df = load()
    check_roof(df)
    df["climate_wind_prefix"] = df.venue_wind_climate.fillna(0)
    live = [f for f in FEATURES if df[f].std() > 0]
    assert len(live) == len(FEATURES), sorted(set(FEATURES) - set(live))
    print("  roof flags sd: " + ", ".join(f"{c} {df[c].std():.3f}"
                                         for c in ("is_indoor_certain", "is_retractable")))
    seasons = sorted(df.season.unique())
    tests = [s for s in seasons if s >= FIRST_TEST and df[df.season == s].week.nunique() >= MIN_WEEKS]
    n_inner, n_boot = (1, 1000) if quick else (3, 10000)
    if quick:
        tests = tests[-3:]
    need = math.ceil(0.75 * len(tests))          # 6 of 8 in the full run
    print(f"{len(df):,} kicker-games, {len(live)} non-constant features, test seasons "
          f"{[int(s) for s in tests]}, inner folds {n_inner}, bootstrap {n_boot}\n")

    blocks, params = [], []
    for s in tests:
        tr, te = df[df.season < s], df[df.season == s]
        med = float(tr.fantasy_points.median())
        block = pd.DataFrame({
            "season": te.season.to_numpy(), "week": te.week.to_numpy(),
            "player_id": te.player_id.to_numpy(), "y": te.fantasy_points.to_numpy(),
            "constant": med,
            "career": te.k_fantasy_points_career.fillna(med).to_numpy(),
            "implied": te.implied_total.fillna(te.implied_total.median()).to_numpy(),
            "ridge": np.asarray(RidgeRanker().fit(tr).predict(te), dtype=float)})
        comp = CompoundKickerModel().fit(tr).predict(te)
        block["exp_points"] = comp.exp_points.to_numpy()
        for t in THRESHOLDS:
            block[f"compound_p{t}"] = comp[f"p_ge_{t}"].to_numpy()
        for name, (fit, grid) in CANDS.items():
            g = tune(fit, grid, tr, n_inner)
            block[name] = np.asarray(fit(tr, te, **g), dtype=float)
            params.append({"season": int(s), "model": name, **g})
            if name in ("poisson_glm", "tweedie_glm"):
                for t, v in glm_tail(tr, te, **g).items():
                    block[f"{name}_p{t}"] = v
        blocks.append(block)
        print(f"  fold {int(s)} done, {time.time() - t0:.0f}s")
    P = pd.concat(blocks, ignore_index=True)
    PAR = pd.DataFrame(params)

    cols = ["ridge"] + list(CANDS) + ["exp_points", "implied", "career", "constant"]

    # Per-week R=2 lift, keyed the way per_week_lift iterates, for the fold table.
    wk = pd.DataFrame([k for k, g in P.groupby(KEY, sort=True) if len(g) >= 2], columns=KEY)
    for c in cols:
        wk[c] = per_week_lift(P, c, 2)[0]
    folds = wk.groupby("season")[cols].mean()

    rows = []
    for c in cols:
        d2, d3 = decision(P, c, 2), decision(P, c, 3)
        r = {"key": c, "model": LABEL[c], "lift_r2": d2["lift"], "share_r2": d2["share"],
             "lift_r3": d3["lift"], "rho": rho(P, c), "hit15_r2": d2["ceiling"]}
        if c in POINTS_SCALE:
            r["mae"] = float(np.abs(P.y - P[c]).mean())
            if P[c].std() > 0:
                r["cal_slope"] = float(np.polyfit(P[c], P.y, 1)[0])
                r["bias"] = float((P[c] - P.y).mean())
        if c != "ridge":
            b2, b3 = boot_lift(P, c, "ridge", 2, n_boot), boot_lift(P, c, "ridge", 3, n_boot)
            r.update({"family": FAMILY.get(c, "descriptive"),
                      "diff_r2": b2["diff"], "lo_r2": b2["lo"], "hi_r2": b2["hi"],
                      "p_r2": b2["p"], "diff_r3": b3["diff"], "p_r3": b3["p"],
                      "folds_won": int((folds[c] > folds["ridge"]).sum())})
            if r["family"] == "baselines":
                # Oriented as the README states them: ridge minus the baseline.
                r.update(season_cluster(wk, "ridge", c, n_boot))
                r.update(sign_test(folds["ridge"], folds[c]))
        else:
            r["family"] = "reference"
        rows.append(r)
    T = pd.DataFrame(rows)
    for fam in ("challengers", "baselines"):
        m = T.family == fam
        T.loc[m, "holm_p"] = holm(T.loc[m, "p_r2"])

    def verdict(r):
        if r.key == "ridge":
            return "reference"
        if r.family == "descriptive":
            return "unadjusted"
        if r.holm_p < 0.05 and r.diff_r2 > 0 and r.folds_won >= need:
            return "WINS"
        if r.holm_p < 0.05 and r.diff_r2 < 0:
            return "ridge better"
        return "no difference"
    T["verdict"] = T.apply(verdict, axis=1)
    rank = {"reference": 0, "challengers": 1, "baselines": 2, "descriptive": 3}
    T = (T.assign(_f=T.family.map(rank)).sort_values(["_f", "lift_r2"], ascending=[True, False])
          .drop(columns="_f"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["model", "lift_r2", "share_r2", "lift_r3", "rho", "mae", "cal_slope",
            "diff_r2", "lo_r2", "hi_r2", "p_r2", "holm_p", "folds_won", "diff_r3",
            "p_r3", "verdict"]
    print(f"\nPooled, {len(tests)} season-forward folds, {len(wk)} weeks, exact lift. "
          f"Win rule: Holm p < 0.05, diff > 0, and >= {need} of {len(tests)} folds.\n")
    print(T[["family"] + show].round(4).to_string(index=False))
    o = decision(P, "ridge", 2)
    fam_m = T.family.value_counts()
    print(f"\noracle lift R=2 = {o['oracle_lift']:.4f}; ridge share {o['share']:.4f}; "
          f"Holm m = {fam_m.get('challengers', 0)} challengers, {fam_m.get('baselines', 0)} baselines")
    print("\nBaseline family, ridge minus baseline, R=2:")
    for _, r in T[T.family == "baselines"].iterrows():
        print(f"  vs {r.model:24} {-r.diff_r2:+.4f}  week-block [{-r.hi_r2:+.4f}, {-r.lo_r2:+.4f}] "
              f"p={r.p_r2:.4f} Holm {r.holm_p:.4f} | season-cluster [{r.season_lo:+.4f}, "
              f"{r.season_hi:+.4f}] p={r.season_p:.4f} | sign test {int(r.seasons_won)}/"
              f"{len(tests)} seasons p={r.sign_p:.4f}")

    print("\nPer-fold R=2 lift:")
    print(folds.T.rename(index=LABEL).round(3).to_string())

    print("\nChosen hyperparameters per fold (inner loop, training seasons only):")
    for name in CANDS:
        g = PAR[PAR.model == name].drop(columns="model").set_index("season")
        g = g.dropna(axis=1, how="all")
        if g.shape[1]:
            print(f"  {name:18} " + "; ".join(
                f"{int(s)}: " + ",".join(f"{k}={v:g}" for k, v in row.items())
                for s, row in g.iterrows()))

    cal = []
    for name, pref in (("compound count model", "compound_p"),
                       ("Poisson GLM", "poisson_glm_p"), ("Tweedie GLM", "tweedie_glm_p")):
        for t in THRESHOLDS:
            pr, rl = float(P[f"{pref}{t}"].mean()), float((P.y >= t).mean())
            cal.append({"model": name, "threshold": t, "predicted": pr, "realised": rl,
                        "ratio": pr / rl})
    CAL = pd.DataFrame(cal)
    print("\nCalibration of the predictive law, P(y >= x):")
    print(CAL.round(4).to_string(index=False))

    wk["slice_week"] = np.where(wk.week <= 4, "weeks 1-4", "weeks 5+")
    wk["slice_era"] = np.where(wk.season <= 2021, "2018-2021", "2022-2025")
    sl = pd.concat([wk.groupby("slice_week")[cols].mean(), wk.groupby("slice_era")[cols].mean()])
    print("\nSlices, R=2 lift:")
    print(sl.T.rename(index=LABEL).round(3).to_string())

    # Interpretability: the shipped ridge's coefficients, per fold, standardised.
    coefs = {}
    for s in tests:
        r = RidgeRanker().fit(df[df.season < s])
        names = r.m_[0].get_feature_names_out()
        coefs[int(s)] = pd.Series(r.m_[-1].coef_, index=names)
    C = pd.DataFrame(coefs).reindex(FEATURES)
    C["mean"] = C.mean(axis=1)
    C["same_sign_folds"] = (np.sign(C[list(coefs)]).eq(np.sign(C["mean"]), axis=0)
                            & C[list(coefs)].notna()).sum(axis=1)
    print("\nShipped ridge, standardised coefficients (points per sd):")
    print(C.round(3).to_string())

    last = df[df.season < tests[-1]]
    r = RidgeRanker().fit(last)
    _, tau2, sigma2 = eb_effects(last.player_id.to_numpy(),
                                 last.fantasy_points.to_numpy() - r.m_.predict(last[FEATURES]))
    g = PAR[(PAR.model == "elastic_net") & (PAR.season == tests[-1])].iloc[0]
    from sklearn.linear_model import ElasticNet
    p = _prep(last)
    nz = int((ElasticNet(alpha=g.alpha, l1_ratio=g.l1, max_iter=20000)
              .fit(p.transform(last[FEATURES]), last.fantasy_points).coef_ != 0).sum())
    print(f"\nEB kicker effect, fit on seasons < {int(tests[-1])}: tau = {math.sqrt(tau2):.3f} "
          f"pts, sigma = {math.sqrt(sigma2):.3f}; elastic net keeps {nz} of {len(FEATURES)} columns")
    print(f"\nelapsed {time.time() - t0:.0f}s")

    if not quick:
        REPORTS.mkdir(parents=True, exist_ok=True)
        T.to_csv(REPORTS / "model_bakeoff_summary.csv", index=False)
        folds.T.to_csv(REPORTS / "model_bakeoff_folds.csv")
        PAR.to_csv(REPORTS / "model_bakeoff_params.csv", index=False)
        CAL.to_csv(REPORTS / "model_bakeoff_calibration.csv", index=False)
        sl.T.to_csv(REPORTS / "model_bakeoff_slices.csv")
        C.to_csv(REPORTS / "model_bakeoff_ridge_coefs.csv")
        P.to_parquet(REPORTS / "model_bakeoff_predictions.parquet", index=False)
        print("wrote reports/model_bakeoff_*.csv")
    return 0


def self_check() -> int:
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"season": 2020, "week": np.repeat(np.arange(40), 12),
                      "y": rng.poisson(8, 480).astype(float)})
    d["s"] = d.y + rng.normal(0, 4, len(d))
    # The closed form equals the pairwise-sign formula at R=2 ...
    ex = exact_lift(d.season, d.week, d.y, d.s)
    pw = []
    for _, g in d.groupby("week"):
        yy, ss = g.y.to_numpy(), g.s.to_numpy()
        n = len(g)
        pw.append((np.sign(ss[:, None] - ss[None, :]) * (yy[:, None] - yy[None, :])).sum()
                  / (2 * n * (n - 1)))
    assert abs(ex - np.mean(pw)) < 1e-9, (ex, np.mean(pw))
    # ... and brute-force enumeration of every roster at R=2 and R=3, with ties
    # split evenly the way argmax over a random draw splits them.
    from itertools import combinations
    yy, ss = rng.integers(0, 20, 9).astype(float), rng.integers(0, 4, 9).astype(float)
    w = pd.DataFrame({"season": 1, "week": 1, "y": yy, "s": ss})
    for R in (2, 3):
        bf = np.mean([yy[[i for i in c if ss[i] == ss[list(c)].max()]].mean() - yy[list(c)].mean()
                      for c in combinations(range(9), R)])
        assert abs(per_week_lift(w, "s", R)[0][0] - bf) < 1e-9, (R, bf)
    # A constant has no lift at all, exactly (the simulation gave it -0.015).
    assert np.abs(per_week_lift(d.assign(c=1.0), "c", 2)[0]).max() < 1e-12
    # Season-cluster bootstrap and sign test on a known shape.
    wk = pd.DataFrame({"season": np.repeat(np.arange(8), 10), "a": 1.0, "b": 0.0})
    sc = season_cluster(wk, "a", "b", 500)
    assert sc["season_lo"] == sc["season_hi"] == 1.0 and sc["season_p"] == 0.0
    st = sign_test(pd.Series([1.0] * 7 + [0.0]), pd.Series([0.5] * 8))
    assert st["seasons_won"] == 7 and abs(st["sign_p"] - 18 / 256) < 1e-12, st
    assert exact_lift([1] * 4, [1] * 4, [1.0, 5.0, 2.0, 9.0], [0.0] * 4) == 0.0
    assert exact_lift([1, 1], [1, 1], [2.0, 6.0], [0.0, 1.0]) == 2.0
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    # EB: a real +-3 kicker effect under small noise survives shrinkage nearly whole;
    # pure noise shrinks to nothing.
    ids = np.repeat(["a", "b"], 200)
    eff, tau2, _ = eb_effects(ids, np.where(ids == "a", 3.0, -3.0) + rng.normal(0, 1, 400))
    assert abs(eff["a"] - 3) < 0.3 and abs(eff["b"] + 3) < 0.3, eff
    ids = np.repeat(list("abcdefghij"), 20)
    _, tau2, _ = eb_effects(ids, rng.normal(0, 1, 200))
    assert tau2 < 0.05, tau2
    print("model_bakeoff self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 folds, 1 inner fold, 1000 bootstrap draws")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else run(a.quick))
