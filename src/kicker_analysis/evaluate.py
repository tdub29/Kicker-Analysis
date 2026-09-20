"""Season-forward evaluation, rebuilt around the decision instead of the residual.

Superseded for the headline numbers by backtest.py, which is what the shipped numbers come from. This file
is kept because it carries two things backtest.py does not: `prob_best`, the
bootstrap probability that a given model wins on a redraw, and `bootstrap_compare`
over an arbitrary statistic. Both are worth having when a new candidate model
appears. Read the numbers from backtest.py.

The first version of this harness sorted a table by MAE and read a winner off the
top. An independent review showed that was measuring almost nothing:

  * Predicting the training MEDIAN (8.0 points) every single week scores a pooled
    MAE of 3.4946. The best real model beat that by 0.0027, which is one sixth of
    a bootstrap standard error. Both gradient-boosted variants were WORSE than the
    constant.
  * On the decision that actually matters, the same models separate cleanly.
    Within-week rank correlation runs 0.138 for ridge against 0.074 for a kicker's
    own career average, p <= 0.0006.
  * The league-mean baseline, which MAE ranked third of seven, gains -0.011 points
    a week when you actually start the kicker it likes. It is worthless, and MAE
    could not see that.

The reason is structural, not a tuning problem. A kicker week is mostly noise: the
target has sd 4.25 and almost all of it is irreducible. Squared or absolute error
is dominated by that noise floor, so every model converges toward the same
unconditional centre and the metric compresses real differences into the fourth
decimal. Ordering is a different question, and it is the only one a fantasy
manager asks: given the two or three kickers I can start, which one.

So this harness reports, in priority order:

  1. decision lift    points per week gained by starting the model's favourite
                      among R rostered kickers, against picking at random. This IS
                      the decision. The oracle value is reported alongside so the
                      share of available value is visible rather than implied.
  2. within-week rho  Spearman inside a week, the ordering the lift depends on.
  3. MAE and RMSE     kept as a calibration sanity check, demoted, and never the
                      column anything is sorted by.

Every comparison carries a week-block bootstrap, because 7 folds cannot resolve
anything: the smallest two-sided Wilcoxon p attainable at n=7 is 0.0156, so a
fold-level test is underpowered by construction. Model choice is reported as a
bootstrap probability of being best, not as a rank, because with six candidates
the expected gain from picking the best of six equally good models is roughly
0.021 MAE, which was two to eight times the winner's actual margin.

Usage:
    python src/kicker_analysis/evaluate2.py
    python src/kicker_analysis/evaluate2.py --self-check
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.features import build_features, feature_columns  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FIRST_SEASON = 2015          # the extra point moved to the 33, and pbp wp starts here
FIRST_TEST_SEASON = 2018
# A season enters the pooled table only with enough distinct weeks to be an
# estimate rather than an anecdote. The old guard was `len(test) < 50` rows,
# a magic number unrelated to stability: 32 more rows would have flipped a
# two-week 2026 into a games-weighted average of full seasons.
MIN_TEST_WEEKS = 8
# Roster depth for the decision simulation. Most managers carry one kicker and
# stream; 2 and 3 bracket the realistic choice set.
ROSTER_SIZES = (2, 3)
N_DRAWS = 400                # random opponent rosters drawn per week
BOOTSTRAP = 4000
RNG_SEED = 0


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def within_week_rho(df: pd.DataFrame) -> float:
    """Mean Spearman inside a week, over weeks where it is defined.

    Pooling rank correlation across a whole season instead would mostly measure
    the gap between good and bad kickers season-long, which nobody has to decide.
    The weekly question is the one a manager faces. A week where the prediction is
    constant yields NaN from scipy and is skipped rather than propagated, which is
    how a constant baseline used to put a NaN in the shipped table.
    """
    out = []
    for _, g in df.groupby(["season", "week"]):
        if len(g) < 5 or g.p.nunique() < 2 or g.y.nunique() < 2:
            continue
        r = spearmanr(g.y, g.p).statistic
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else float("nan")


def decision_lift(df: pd.DataFrame, roster: int, n_draws: int = N_DRAWS,
                  seed: int = RNG_SEED) -> dict:
    """Points per week gained by starting the model's favourite of R kickers.

    Each week, draw R kickers at random from those who played, start the one the
    model ranks highest, and compare to starting a random one of the R. The oracle
    starts the one who actually scored most. Random is the floor and oracle is the
    ceiling, so the share between them says how much of the available value the
    model captures, which a raw point total cannot.

    Ties in the prediction are broken by the draw order rather than by player id.
    A constant predictor therefore scores random by construction, which is correct
    and is exactly what the old top-5 metric got wrong: it returned the
    alphabetically first five ids and reported that as a hit rate.
    """
    rng = np.random.default_rng(seed)
    model, rand, oracle, weeks = [], [], [], 0
    for _, g in df.groupby(["season", "week"]):
        if len(g) < roster:
            continue
        y = g.y.to_numpy()
        p = g.p.to_numpy()
        n = len(g)
        weeks += 1
        idx = np.array([rng.choice(n, size=roster, replace=False) for _ in range(n_draws)])
        picks = y[idx]
        preds = p[idx]
        # argmax over the draw picks the first maximum, and the draw order is
        # already random, so tied predictions resolve to a random member.
        model.append(picks[np.arange(n_draws), preds.argmax(axis=1)].mean())
        rand.append(picks.mean())
        oracle.append(picks.max(axis=1).mean())
    if not weeks:
        return {"lift": float("nan"), "oracle_lift": float("nan"), "share": float("nan"),
                "weeks": 0}
    lift = float(np.mean(model) - np.mean(rand))
    omax = float(np.mean(oracle) - np.mean(rand))
    return {"lift": lift, "oracle_lift": omax,
            "share": lift / omax if omax else float("nan"), "weeks": weeks,
            "per_week": np.array(model) - np.array(rand)}


def basic(df: pd.DataFrame) -> dict:
    """MAE and RMSE pooled over ROWS.

    Averaging per-season RMSEs, weighted or not, is not the pooled RMSE: the
    square root does not commute with the mean. The old harness did that and
    understated RMSE by 0.004 to 0.008.
    """
    return {"mae": float(mean_absolute_error(df.y, df.p)),
            "rmse": float(np.sqrt(np.mean((df.y - df.p) ** 2)))}


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def week_blocks(df: pd.DataFrame) -> list[np.ndarray]:
    """Row indices grouped by week. Weeks are the resampling unit because rows
    inside a week share a weather system, a slate and a set of opponents, so
    treating them as independent would understate every interval."""
    return [g.index.to_numpy() for _, g in df.groupby(["season", "week"])]


def bootstrap_compare(preds: dict, blocks: list[np.ndarray], stat, a: str, b: str,
                      n: int = BOOTSTRAP, seed: int = RNG_SEED) -> dict:
    """Paired week-block bootstrap of stat(a) - stat(b)."""
    rng = np.random.default_rng(seed)
    obs = stat(preds[a]) - stat(preds[b])
    diffs = np.empty(n)
    nb = len(blocks)
    for i in range(n):
        pick = rng.integers(0, nb, nb)
        rows = np.concatenate([blocks[j] for j in pick])
        diffs[i] = stat(preds[a].loc[rows]) - stat(preds[b].loc[rows])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p by the usual sign-reversal convention.
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(obs), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p))}


def prob_best(preds: dict, blocks: list[np.ndarray], stat, higher_is_better: bool,
              n: int = 2000, seed: int = RNG_SEED) -> dict:
    """How often each model wins on a bootstrap redraw.

    A pooled table names one winner. This says whether that name would survive
    another season of the same league, which is the honest form of the claim.
    """
    rng = np.random.default_rng(seed)
    names = list(preds)
    wins = {k: 0 for k in names}
    nb = len(blocks)
    for _ in range(n):
        pick = rng.integers(0, nb, nb)
        rows = np.concatenate([blocks[j] for j in pick])
        vals = {k: stat(preds[k].loc[rows]) for k in names}
        best = (max if higher_is_better else min)(vals, key=vals.get)
        wins[best] += 1
    return {k: v / n for k, v in wins.items()}


# --------------------------------------------------------------------------
# data and models
# --------------------------------------------------------------------------
def load() -> pd.DataFrame:
    games = pd.read_parquet(PROC / "kicker_games.parquet")
    fourth = pd.read_parquet(PROC / "fourth_down.parquet")
    df = games.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    df = df[(df.season >= FIRST_SEASON) & (df.season_type == "REG")].copy()
    return build_features(df)


def model_specs():
    from lightgbm import LGBMRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def ridge(Xtr, ytr, Xte):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    def gbm(Xtr, ytr, Xte):
        m = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=0)
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    return {"ridge": ridge, "gbm": gbm}


def baseline_preds(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """The floor. Nothing counts as working until it clears these.

    `constant_median` is the one the first version was missing and the one that
    matters: MAE is minimised by the median, so the best possible constant is the
    median and not the mean. It beat both boosted models.
    """
    med = float(train.fantasy_points.median())
    mean = float(train.fantasy_points.mean())
    return {
        "constant_median": np.full(len(test), med),
        "constant_mean": np.full(len(test), mean),
        "kicker_career": test["k_fantasy_points_career"].fillna(mean).to_numpy(),
        "kicker_last8": test["k_fantasy_points_l8"].fillna(mean).to_numpy(),
        "implied_total_only": None,     # filled below, needs a fit
    }


def run() -> dict:
    df = load()
    feats = feature_columns(df)
    seasons = sorted(df.season.unique())
    test_seasons = [s for s in seasons if s >= FIRST_TEST_SEASON
                    and df[df.season == s].week.nunique() >= MIN_TEST_WEEKS]
    skipped = [int(s) for s in seasons if s >= FIRST_TEST_SEASON and s not in test_seasons]
    print(f"{len(df):,} kicker-games {df.season.min()}-{df.season.max()}, {len(feats)} features")
    print(f"test seasons scored: {[int(s) for s in test_seasons]}")
    if skipped:
        print(f"skipped, under {MIN_TEST_WEEKS} weeks: {skipped}")

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    specs = model_specs()
    frames = {}
    for s in test_seasons:
        train, test = df[df.season < s], df[df.season == s]
        base = baseline_preds(train, test)
        # A single-feature market model, so the value of everything else is
        # measured against the line rather than against nothing.
        mk = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
        mk.fit(train[["implied_total"]], train.fantasy_points)
        base["implied_total_only"] = mk.predict(test[["implied_total"]])
        preds = dict(base)
        for name, fn in specs.items():
            preds[name] = np.clip(fn(train[feats], train.fantasy_points, test[feats]), 0, None)
        for name, p in preds.items():
            frames.setdefault(name, []).append(pd.DataFrame(
                {"season": test.season.to_numpy(), "week": test.week.to_numpy(),
                 "player_id": test.player_id.to_numpy(),
                 "y": test.fantasy_points.to_numpy(), "p": np.asarray(p)}))

    preds = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    blocks = week_blocks(next(iter(preds.values())))

    rows = []
    for name, d in preds.items():
        r = {"model": name, **basic(d), "within_week_rho": within_week_rho(d)}
        for R in ROSTER_SIZES:
            dl = decision_lift(d, R)
            r[f"lift_r{R}"] = dl["lift"]
            r[f"share_r{R}"] = dl["share"]
        rows.append(r)
    table = pd.DataFrame(rows).sort_values("lift_r2", ascending=False)

    print("\nRanked by the decision, not by the residual.")
    print("lift_rN = points per week gained starting the model's pick of N kickers, vs random.")
    print("share_rN = fraction of the oracle's available gain captured.\n")
    print(table.round(4).to_string(index=False))

    oracle2 = decision_lift(preds["ridge"], 2)["oracle_lift"]
    print(f"\noracle lift at R=2: {oracle2:.3f} pts/wk. That is the ceiling; "
          f"everything above is a slice of it.")

    best = table.model.iloc[0]
    print(f"\nBootstrap over {len(blocks)} week blocks, {BOOTSTRAP} resamples.")
    for other in ["constant_median", "kicker_career", "implied_total_only"]:
        if other == best:
            continue
        for label, stat in (("within-week rho", within_week_rho),
                            ("R=2 lift", lambda d: decision_lift(d, 2, n_draws=120)["lift"])):
            c = bootstrap_compare(preds, blocks, stat, best, other, n=600)
            print(f"  {best} vs {other:20} {label:16} "
                  f"{c['diff']:+.4f}  95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  p={c['p']:.4f}")

    pb = prob_best(preds, blocks, within_week_rho, higher_is_better=True, n=800)
    print("\nProbability each model is best on within-week rho, on a bootstrap redraw:")
    for k, v in sorted(pb.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22} {v:.3f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "decision_metrics.csv", index=False)
    pd.concat([d.assign(model=k) for k, d in preds.items()], ignore_index=True).to_parquet(
        REPORTS / "predictions.parquet", index=False)
    print(f"\nwrote reports/decision_metrics.csv and reports/predictions.parquet")
    return {"table": table, "preds": preds, "prob_best": pb}


def self_check() -> int:
    """The decision metric, which is now the headline and so must be right."""
    # Two weeks, four kickers each. The prediction is perfect, so choosing from
    # any R must equal the oracle exactly and the captured share must be 1.
    d = pd.DataFrame({
        "season": [2024] * 8, "week": [1] * 4 + [2] * 4,
        "y": [1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0],
    })
    d["p"] = d.y
    for R in (2, 3):
        r = decision_lift(d, R, n_draws=500)
        assert abs(r["share"] - 1.0) < 1e-9, (R, r["share"])
        assert r["lift"] > 0

    # A constant prediction must score exactly random, i.e. zero lift. The old
    # top-5 metric failed here: it fell back to id order and reported a hit rate.
    c = d.copy(); c["p"] = 7.0
    assert abs(decision_lift(c, 2, n_draws=2000)["lift"]) < 0.15, decision_lift(c, 2)["lift"]

    # A perfectly inverted prediction must be strictly WORSE than random.
    inv = d.copy(); inv["p"] = -d.y
    assert decision_lift(inv, 2, n_draws=500)["lift"] < -0.5

    # within_week_rho skips a constant week rather than returning NaN for the set.
    # The scorable weeks need 5+ rows each: the guard exists so a 3-kicker week
    # cannot contribute a rank correlation, and an earlier version of this test
    # used 4-row weeks, so EVERY week was skipped and the mean of nothing was NaN.
    # That was the test being wrong, not the function.
    good = pd.DataFrame({"season": [2024] * 10, "week": [1] * 5 + [2] * 5,
                         "y": [1.0, 2, 3, 4, 5] * 2})
    good["p"] = good.y
    flat = pd.DataFrame({"season": [2024] * 5, "week": [3] * 5,
                         "y": [1.0, 2, 3, 4, 5], "p": [7.0] * 5})
    mixed = pd.concat([good, flat], ignore_index=True)
    assert np.isfinite(within_week_rho(mixed)), within_week_rho(mixed)
    # The two real weeks are perfectly ordered; the constant week is dropped, not
    # averaged in as a zero, which would drag this to 0.67.
    assert abs(within_week_rho(mixed) - 1.0) < 1e-9, within_week_rho(mixed)
    assert np.isnan(within_week_rho(flat))
    # A week too small to score is skipped on size alone.
    small = pd.DataFrame({"season": [2024] * 4, "week": [1] * 4,
                          "y": [1.0, 2, 3, 4], "p": [1.0, 2, 3, 4]})
    assert np.isnan(within_week_rho(small))

    # Pooled RMSE is not the mean of per-group RMSEs. This is the bug the old
    # harness shipped, so it gets an explicit test.
    a = pd.DataFrame({"y": [0.0, 0.0], "p": [0.0, 4.0]})
    b = pd.DataFrame({"y": [0.0, 0.0], "p": [0.0, 0.0]})
    pooled = basic(pd.concat([a, b]))["rmse"]
    naive = np.mean([basic(a)["rmse"], basic(b)["rmse"]])
    assert abs(pooled - 2.0) < 1e-12, pooled
    assert abs(naive - 1.4142135) < 1e-6
    assert pooled > naive
    print("evaluate self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else (run(), 0)[1])
