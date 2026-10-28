"""The kicker models: a ridge that orders the board, and two count stages drawn
into a full predictive law.

WHY THIS SHAPE, which is the whole argument:

Fantasy points decompose exactly. points = 3*short + 4*mid + 5*long + 1*PAT, and
over the 2015-2025 regular-season rows the field-goal part's variance is 101.8%
of the total's, the PAT part's 10.5%, and twice their covariance -12.4%
(reports/variance_decomposition.csv, written by backtest.py). That negative
covariance is the red-zone substitution: a drive that ends in a touchdown pays 1
point, the same drive stalling pays 3.

Then the fact that decides the design. Season-forward, out of sample, the count
stage has close to no skill on weekly field-goal attempts: R2 against the
training-season mean is +0.003 (week-block 95% CI -0.003 to +0.009), and
within-week Spearman 0.068. On PAT attempts it has real skill, R2 +0.126 and
Spearman 0.348 (reports/component_skill.csv, written by backtest.py). The half
of the target carrying almost all the variance is close to unforecastable; the
half with real skill carries about a tenth of it. A single regression on total
points cannot show you that, and every attempt to improve the total is fighting
the unforecastable half.

So there are two models here, each used for the job it does better:

  RidgeRanker          orders the board. Pooled over the 8 folds it beats the
                       count model's expected points, the betting line alone
                       and a kicker's career average on the decision metric
                       (reports/backtest_summary.csv). Its edge over the line
                       is clear week to week and not established season to
                       season: ahead in 6 of 8 seasons, t-test on the season
                       means p = 0.095 (reports/season_level_tests.csv).
  CompoundKickerModel  predicts the two counts separately and combines them by
                       the scoring rule, which gives P(y >= x) for every
                       threshold with no threshold model trained. Pooled over
                       the 8 folds its P(y >= 15) averages 0.0839 against a
                       realised 0.0841, but it over-predicts 8+ and 10+ by
                       about two percentage points (+0.021 and +0.018;
                       backtest.py prints week-block intervals that exclude
                       zero). Kicker by kicker it is weak: at 15 points its
                       AUC is 0.55, its calibration slope 0.55, and its Brier
                       skill over a constant at the training base rate +0.001
                       (95% CI -0.004 to +0.006), no better than that constant
                       (reports/calibration_summary.csv). Its docstring lists
                       where the draw is knowingly wrong.

RETRACTED, because the backtest did not support it. An earlier version of this
docstring said a gradient-boosted mean ranking hit 15+ less often than random
(0.057 vs 0.085) and that ranking by P(y>=15) fixed it. Neither holds for this
model: in the ceiling_r2 column of reports/backtest_summary.csv both the compound
model's expected-points ranking and its P(y>=15) ranking pick a 15-point week
more often than a random pick does (the constant row, whose ties split at
random), and the second board adds nothing on top of the first.

Capacity is small on purpose: 100 trees of 7 leaves per count stage, on a few
thousand training rows, against the 500-tree, 31-leaf settings of the first
harness (legacy/evaluate_v1.py). That is a judgment, not a tuned result; no
committed code compares the two settings.

This module is a library: backtest.py runs the season-forward backtest and
predict_week.py writes this week's board.

Usage:
    python src/kicker_analysis/model.py --self-check
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.features import build_features, SERVE_TIME_UNAVAILABLE  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# An INCLUSION list, not an exclusion list. The previous version admitted every
# numeric column nobody had remembered to ban, so when 17 extra columns were
# merged into the table, three of them same-game outcomes, they went straight
# into the model and moved MAE from 3.49 to 2.65: a leak that looked like a gain.
# A feature set you cannot enumerate is a feature set you cannot check, so every
# column here is named and has a stated mechanism.
FEATURES = [
    # The market. One line already beats the 73-column model, because it is the
    # only input that prices this specific offence against this specific defence.
    # These are nflverse CLOSING lines, so the board is valid for a lineup set
    # close to kickoff, with the line as it stands then.
    "implied_total",          # the offence's own forecast points
    "opp_implied_total",      # the other side's, which drives game flow
    "total_line",
    "closeness",              # -|spread|: a close game keeps both offences trying
    # Team volume, lagged. A kicker's chances are his offence's failures.
    "t_fg_att_l8",
    "t_pat_att_l8",
    "t_team_score_l8",
    # Coach behaviour on fourth down in range, the choice that takes kicks away.
    "prev_go_rate",
    "t_fg_share_l8",
    # The leg. Accuracy is the kicker's own contribution; volume is not.
    "k_fg_pct_career",
    "k_fg_att_l8",
    "k_fantasy_points_l8",
    # Environment, in the only form knowable before kickoff: the building.
    "is_indoor_certain",
    "is_retractable",
    "climate_wind_exposed",
    "is_home",
]

# Scoring, matching build_dataset.SCORING. Kept here so the convolution and the
# target can be checked against each other.
PTS_SHORT, PTS_MID, PTS_LONG, PTS_PAT = 3.0, 4.0, 5.0, 1.0
N_SIMS = 4000
FIRST_SEASON = 2015
MIN_TEST_WEEKS = 8


def load(processed: pathlib.Path | None = None) -> pd.DataFrame:
    proc = processed or PROC
    games = pd.read_parquet(proc / "kicker_games.parquet")
    fourth = pd.read_parquet(PROC / "fourth_down.parquet")
    sched = pd.read_csv(ROOT / "data" / "external" / "games.csv", low_memory=False)
    df = games.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    df = df[(df.season >= FIRST_SEASON) & (df.season_type == "REG")].copy()
    return build_features(df, games=sched)


def _count_model():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="poisson", n_estimators=100, num_leaves=7,
                         learning_rate=0.05, min_child_samples=60,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         reg_lambda=1.0, verbose=-1, random_state=0, n_jobs=1)


class RidgeRanker:
    """The ORDERING model. Ridge on the 16 named features, heavily regularised.

    Scored on the decision metric (exact expected points per week gained by
    starting the better of two random kickers, 8 season-forward folds), it
    ranks ahead of the compound count model's expected points, the Vegas
    implied total alone and a kicker's career average
    (reports/backtest_summary.csv). Its margin over the line alone, +0.074
    points a week, is clear when weeks are resampled (95% CI +0.024 to
    +0.124) and not significant when each test season is one observation:
    ahead in 6 of 8, t-test p = 0.095 (reports/season_level_tests.csv). In
    the bake-off none of fourteen challengers could be told apart from it
    after Holm correction (reports/model_bakeoff_summary.csv,
    notes/model-selection-2026-09-29.md).
    So the count model is the probability machine and this is the ranker.

    alpha = 100 is a fixed choice, not a tuned optimum. The design matrix is
    collinear by construction (lagged windows of the same quantity), so heavy
    shrinkage is expected, but the data do not pin alpha down. In the nested
    bake-off an inner loop choosing alpha on training seasons only picked
    values from 100 to 10,000 across the 8 folds, 300 in five of them
    (reports/model_bakeoff_params.csv), and that nested ridge did not beat the
    fixed alpha = 100: -0.005 points a week, 95% CI -0.019 to +0.009
    (reports/model_bakeoff_summary.csv).

    Noise control, because adding columns can move a score on its own: at
    alpha = 100, columns of pure Gaussian noise make this ridge worse, not
    better, and more so the more are added. Averaged over 50 draws, 13 noise
    columns on top of the 16 features move pooled MAE from 3.472 to 3.477 and
    decision lift from 0.427 to 0.406; 61 move them to 3.503 and 0.338
    (experiments/feature_families/ridge_retest.py,
    reports/feature_family_ridge_retest.csv). So under this ridge a column
    cannot improve the score by its count alone. The case for the ridge rests
    on the decision metric above, not on MAE.
    """

    def __init__(self, features: list[str] | None = None, alpha: float = 100.0):
        self.features = features or FEATURES
        self.alpha = alpha

    def fit(self, train: pd.DataFrame):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.m_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                Ridge(alpha=self.alpha))
        self.m_.fit(train[self.features], train.fantasy_points)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.clip(self.m_.predict(test[self.features]), 0, None)


class CompoundKickerModel:
    """The PROBABILITY model. Two count stages drawn into a full law.

    Kept for what a point forecast cannot give: a whole predictive distribution
    (floor, median, ceiling, P(y >= x) at any threshold) without a model per
    threshold. Pooled over the 8 backtest folds its P(y >= 15) averages what
    happened, it over-predicts 8+ and 10+, and kicker by kicker it barely beats
    a constant at the base rate: the module docstring has the numbers, and
    reports/calibration_summary.csv and calibration_reliability.csv the tables.
    It is NOT used for ranking; see RidgeRanker for why.

    What _draw actually does, and where it is knowingly wrong:

      counts    each count is a binomial with n = max(ceil(3 * lambda), 1)
                trials and p = lambda / n, so its mean is lambda and its
                variance/mean is 1 - lambda / n: at least 2/3, under 3/4 once
                lambda is 1 or more, and set by the 3, not fitted. Weekly
                attempt counts are under-dispersed (variance below the mean),
                which is why a Poisson draw is not used: it would be too wide.
                But they are less under-dispersed than this, both raw and
                around the count stages' out-of-sample means, so the fixed
                ratio makes each count somewhat too narrow.
      FG, PAT   drawn independently. They are not independent: out of sample,
                a kicker's FG-attempt residual (attempts minus the count
                stage's mean) and his PAT-attempt residual are negatively
                correlated, pooled and in each of the 8 test seasons. A drive
                that ends in a touchdown is one that did not end in a field
                goal. Ignoring that makes the simulated total wider than it
                should be.
      distance  makes are split into the 3-, 4- and 5-point tiers with random
                per-draw weights (uniform times the league mix, renormalised)
                and rounding, with the long tier taking the remainder. That
                pulls the split away from the league mix toward the longer,
                rarer tiers, so each make is worth a little more than the mix
                says.

    No committed code prints these three checks; they were run against
    reports/backtest_predictions.parquet and the kicker-game table. So the
    pooled match at 15 points is not evidence that the draw is right: the
    counts are too narrow, the total too wide and the makes slightly too
    valuable, and errors in opposite directions can offset. Read P(15+) as a
    rough flag for upside, right on average and not sharp per kicker.
    """

    def __init__(self, features: list[str] | None = None, n_sims: int = N_SIMS):
        self.features = features or FEATURES
        self.n_sims = n_sims

    def fit(self, train: pd.DataFrame):
        from sklearn.impute import SimpleImputer
        self.imp_ = SimpleImputer(strategy="median").fit(train[self.features])
        X = self.imp_.transform(train[self.features])
        self.fg_ = _count_model().fit(X, train.fg_att)
        self.pat_ = _count_model().fit(X, train.pat_att)
        # Make rate and distance mix are league constants estimated on the
        # training years, not per-kicker: a kicker's own 50+ rate over a handful
        # of tries is mostly noise, and shrinking it all the way to the league
        # value is the conservative choice.
        att = train.fg_att.sum()
        self.make_rate_ = float(train.fg_made.sum() / att) if att else 0.83
        made = train[["fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
                      "fg_made_40_49", "fg_made_50_59", "fg_made_60_"]].sum()
        short = made[["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"]].sum()
        mid = made["fg_made_40_49"]
        long = made[["fg_made_50_59", "fg_made_60_"]].sum()
        tot = short + mid + long
        self.mix_ = (np.array([short, mid, long], dtype=float) / tot) if tot else \
            np.array([0.55, 0.28, 0.17])
        return self

    def _draw(self, X, seed: int = 0) -> np.ndarray:
        """Monte-Carlo the predictive law. Rows x sims of total fantasy points."""
        rng = np.random.default_rng(seed)
        lam_fg = np.clip(self.fg_.predict(X), 0.01, None)
        lam_pat = np.clip(self.pat_.predict(X), 0.01, None)
        n = len(X)
        # Counts are UNDER-dispersed, so a raw Poisson draw is too wide, and a
        # negative binomial could only widen it further. A binomial with mean
        # lambda over ceil(3 * lambda) trials narrows it: variance/mean is
        # 1 - lambda / trials, at least 2/3, a constant set by the 3 rather than
        # fitted (the class docstring says how that compares with the data).
        # FG and PAT counts are drawn independently of each other.
        def counts(lam):
            trials = np.maximum(np.ceil(lam * 3).astype(int), 1)
            p = np.clip(lam[:, None] / trials[:, None], 0, 1)
            return rng.binomial(trials[:, None], p, size=(n, self.n_sims))
        att = counts(lam_fg)
        pat = counts(lam_pat)
        made = rng.binomial(att, self.make_rate_)
        # Split makes across the three scoring tiers: uniform noise times the
        # league mix, renormalised, rounded, with the long tier taking the
        # remainder. The random weights shift makes toward the longer tiers
        # (see the class docstring), so this is not an exact draw at the mix.
        u = rng.random((n, self.n_sims, 3))
        w = u * self.mix_
        w = w / w.sum(axis=2, keepdims=True)
        short = np.round(made * w[:, :, 0])
        mid = np.round(made * w[:, :, 1])
        long = made - short - mid
        return (short * PTS_SHORT + mid * PTS_MID + np.maximum(long, 0) * PTS_LONG
                + pat * PTS_PAT)

    def predict(self, test: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        X = self.imp_.transform(test[self.features])
        sims = self._draw(X, seed=seed)
        out = pd.DataFrame(index=test.index)
        out["exp_points"] = sims.mean(axis=1)
        out["median_points"] = np.median(sims, axis=1)
        out["floor_p20"] = np.percentile(sims, 20, axis=1)
        out["ceiling_p90"] = np.percentile(sims, 90, axis=1)
        for thr in (8, 10, 12, 15):
            out[f"p_ge_{thr}"] = (sims >= thr).mean(axis=1)
        out["exp_fg_att"] = np.clip(self.fg_.predict(X), 0, None)
        out["exp_pat_att"] = np.clip(self.pat_.predict(X), 0, None)
        return out


def self_check() -> int:
    """The scoring identity and the direction of the two rankings."""
    # The convolution must reproduce the scoring rule in expectation. Build a
    # model with known constants and check the mean lands where arithmetic says.
    m = CompoundKickerModel(features=["a"], n_sims=8000)
    m.make_rate_ = 1.0
    m.mix_ = np.array([1.0, 0.0, 0.0])          # every make is a short one

    class _Const:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.full(len(X), self.v)

    m.fg_, m.pat_ = _Const(2.0), _Const(3.0)
    m.imp_ = type("I", (), {"transform": staticmethod(lambda d: np.asarray(d))})()
    got = m.predict(pd.DataFrame({"a": [0.0]}))
    # 2 makes at 3 points plus 3 extra points = 9.
    assert abs(float(got.exp_points.iloc[0]) - 9.0) < 0.3, got.exp_points.iloc[0]
    assert float(got.exp_fg_att.iloc[0]) == 2.0

    # A bigger mean must give a bigger ceiling probability, or the distribution
    # is not ordered and P(y>=x) is meaningless.
    m.fg_ = _Const(4.0)
    hi = m.predict(pd.DataFrame({"a": [0.0]}))
    assert float(hi.p_ge_15.iloc[0]) > float(got.p_ge_15.iloc[0])
    assert float(hi.ceiling_p90.iloc[0]) >= float(got.ceiling_p90.iloc[0])
    # Floor below median below ceiling, always.
    assert (hi.floor_p20 <= hi.median_points).all()
    assert (hi.median_points <= hi.ceiling_p90).all()
    # Probabilities are monotone in the threshold.
    assert float(hi.p_ge_8.iloc[0]) >= float(hi.p_ge_10.iloc[0]) >= float(hi.p_ge_15.iloc[0])

    # Every named feature must be absent from the serve-time ban list. This is
    # the inclusion list's whole point, so it is asserted rather than assumed.
    assert not (set(FEATURES) & set(SERVE_TIME_UNAVAILABLE)), \
        set(FEATURES) & set(SERVE_TIME_UNAVAILABLE)
    print("model self-check ok")
    return 0


if __name__ == "__main__":
    # A library with one CLI entry: its own check. The backtest and the weekly
    # board have their own scripts, so a flag here could only shadow them.
    ap = argparse.ArgumentParser(
        description="RidgeRanker and CompoundKickerModel. To run the backtest use "
                    "backtest.py; to write this week's board use predict_week.py.")
    ap.add_argument("--self-check", action="store_true",
                    help="offline checks of the scoring identity and the feature list")
    a = ap.parse_args()
    if a.self_check:
        sys.exit(self_check())
    ap.print_help()
    sys.exit(0)
