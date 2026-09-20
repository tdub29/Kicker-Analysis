"""Season-forward evaluation: which model, and which features actually work.

Every number here is out of sample in the only way that matters for a weekly
forecast. For each test season S the model trains on seasons strictly before S
and predicts S, so nothing from the future of the league is ever in the training
set. A random train/test split on this data would put week 12 in the training
set and week 3 in the test set for the same kicker and the same season, which
inflates every metric and is how the repo's original evaluation would have gone
had it run one.

Three things are reported, because they answer different questions:

  MAE / RMSE     how close the points prediction is. The honest accuracy number,
                 and the one a naive baseline is hardest to beat on, because
                 kicker weeks are mostly noise.
  Spearman       how well the ORDER is called. This is the fantasy question:
                 start the right man, do not forecast his exact total.
  top-5 hit      of the five kickers the model likes most in a week, how many
                 actually finish top five that week. The decision, scored.

Baselines exist so an improvement has to be earned. Beating "predict the league
average every time" is not an achievement; beating a kicker's own career average
is the real bar, and it is a surprisingly high one.

Usage:
    python src/kicker_analysis/evaluate.py
    python src/kicker_analysis/evaluate.py --self-check
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.features import build_features, feature_columns  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# 2015 is the floor: the extra point moved to the 33 that year, which changed
# kicker scoring outright, and it is also where the fourth-down features start.
FIRST_SEASON = 2015
# Leave enough history to train on before the first test year.
FIRST_TEST_SEASON = 2018


def load() -> pd.DataFrame:
    games = pd.read_parquet(PROC / "kicker_games.parquet")
    fourth = pd.read_parquet(PROC / "fourth_down.parquet")
    df = games.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    df = df[(df.season >= FIRST_SEASON) & (df.season_type == "REG")].copy()
    return build_features(df)


def metrics(y, p, weeks=None) -> dict:
    out = {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "spearman": float(spearmanr(y, p).statistic),
    }
    if weeks is not None:
        hits, n = 0, 0
        frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "w": np.asarray(weeks)})
        for _, g in frame.groupby("w"):
            if len(g) < 10:
                continue
            picked = set(g.nlargest(5, "p").index)
            actual = set(g.nlargest(5, "y").index)
            hits += len(picked & actual)
            n += 5
        out["top5_hit"] = hits / n if n else float("nan")
    return out


def model_specs():
    """Each entry returns fitted predictions. Kept as closures so the harness
    stays one loop and adding a family is one more line, not a new branch."""
    from lightgbm import LGBMRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def ridge(Xtr, ytr, Xte, _):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          Ridge(alpha=10.0))
        m.fit(Xtr, ytr)
        return m.predict(Xte), m

    def gbm(Xtr, ytr, Xte, _):
        m = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=1.0, verbose=-1,
                          random_state=0)
        m.fit(Xtr, ytr)
        return m.predict(Xte), m

    def gbm_tweedie(Xtr, ytr, Xte, _):
        # Fantasy points are non-negative and right-skewed with a spike at low
        # values. Squared error assumes neither, so Tweedie is the distribution
        # actually implied by the target rather than the default.
        m = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                          n_estimators=500, learning_rate=0.03, num_leaves=31,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=1.0, verbose=-1,
                          random_state=0)
        m.fit(Xtr, ytr)
        return m.predict(Xte), m

    return {"ridge": ridge, "gbm": gbm, "gbm_tweedie": gbm_tweedie}


def baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """What has to be beaten before any model counts as working."""
    gmean = train.fantasy_points.mean()
    out = {"naive_league_mean": np.full(len(test), gmean)}
    # A kicker's own career mean to date, which is already a lagged feature.
    own = test["k_fantasy_points_career"].fillna(gmean).to_numpy()
    out["naive_kicker_career"] = own
    out["naive_kicker_last8"] = test["k_fantasy_points_l8"].fillna(gmean).to_numpy()
    # The shape of the original repo model: three rolling volume/accuracy terms,
    # ordinary least squares, no market and no context.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    cols = ["k_fg_att_l8", "k_fg_pct_career", "k_pat_att_l8"]
    m = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    m.fit(train[cols], train.fantasy_points)
    out["old_model_shape"] = m.predict(test[cols])
    return out


def run() -> dict:
    df = load()
    feats = feature_columns(df)
    target = "fantasy_points"
    seasons = sorted(s for s in df.season.unique() if s >= FIRST_TEST_SEASON)
    print(f"{len(df):,} kicker-games {df.season.min()}-{df.season.max()}, "
          f"{len(feats)} features, testing {seasons[0]}-{seasons[-1]}\n")

    specs = model_specs()
    rows, preds = [], []
    for s in seasons:
        train = df[df.season < s]
        test = df[df.season == s]
        if len(test) < 50:
            continue
        Xtr, ytr = train[feats], train[target]
        Xte, yte = test[feats], test[target]
        for name, p in baselines(train, test).items():
            rows.append({"season": s, "model": name, "n": len(test),
                         **metrics(yte, p, test.week)})
        for name, fn in specs.items():
            p, m = fn(Xtr, ytr, Xte, feats)
            p = np.clip(p, 0, None)          # a kicker cannot score negative here
            rows.append({"season": s, "model": name, "n": len(test),
                         **metrics(yte, p, test.week)})
            preds.append(pd.DataFrame({"season": s, "model": name,
                                       "y": yte.to_numpy(), "p": p}))
    res = pd.DataFrame(rows)

    print("Pooled over every test season, weighted by games:")
    agg = (res.assign(w=lambda d: d.n)
             .groupby("model")
             .apply(lambda g: pd.Series({
                 "mae": np.average(g.mae, weights=g.w),
                 "rmse": np.average(g.rmse, weights=g.w),
                 "spearman": np.average(g.spearman, weights=g.w),
                 "top5_hit": np.average(g.top5_hit, weights=g.w),
                 "games": int(g.n.sum())}), include_groups=False)
             .sort_values("mae"))
    print(agg.round(4).to_string())

    # Feature importance on the hardest, most recent split.
    last = seasons[-1]
    train, test = df[df.season < last], df[df.season == last]
    from lightgbm import LGBMRegressor
    m = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                      min_child_samples=40, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=0)
    m.fit(train[feats], train[target])
    gain = pd.Series(m.booster_.feature_importance("gain"), index=feats)
    perm = permutation_importance(m, test[feats], test[target], n_repeats=10,
                                  random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"gain": gain / gain.sum(),
                        "perm_mae_delta": perm.importances_mean}).sort_values(
        "perm_mae_delta", ascending=False)
    print(f"\nTop 20 features by permutation importance, test season {last}")
    print("(perm_mae_delta = MAE added when the column is shuffled; <=0 means it earns nothing)")
    print(imp.head(20).round(5).to_string())
    print(f"\nFeatures that earn nothing (perm <= 0): "
          f"{int((imp.perm_mae_delta <= 0).sum())} of {len(imp)}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS / "season_forward_results.csv", index=False)
    imp.to_csv(REPORTS / "feature_importance.csv")
    agg.to_csv(REPORTS / "model_summary.csv")
    print(f"\nwrote {REPORTS.relative_to(ROOT)}/ season_forward_results.csv, "
          f"feature_importance.csv, model_summary.csv")
    return {"summary": agg, "by_season": res, "importance": imp}


def self_check() -> int:
    """The two metrics that are easy to get subtly wrong."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(metrics(y, y)["mae"]) < 1e-12
    assert abs(metrics(y, y)["spearman"] - 1.0) < 1e-12
    # A perfectly reversed prediction is a perfect NEGATIVE rank correlation, and
    # its MAE is not zero. A metric that reported 1.0 here would hide a sign flip.
    assert abs(metrics(y, y[::-1])["spearman"] + 1.0) < 1e-12

    # top-5 hit rate: 12 players in one week, prediction is the truth reversed,
    # so the five it likes are the five worst and nothing overlaps.
    y2 = np.arange(12, dtype=float)
    w2 = np.zeros(12)
    assert metrics(y2, y2, w2)["top5_hit"] == 1.0
    assert metrics(y2, -y2, w2)["top5_hit"] == 0.0
    # A week with fewer than 10 scoring kickers is skipped rather than scored on
    # a field too small for a top five to mean anything.
    assert np.isnan(metrics(y2[:6], y2[:6], w2[:6])["top5_hit"])
    print("evaluate self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        sys.exit(self_check())
    run()
