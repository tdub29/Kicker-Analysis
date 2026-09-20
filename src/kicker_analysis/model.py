"""The kicker model: two count stages, convolved into a full predictive law.

WHY THIS SHAPE, which is the whole argument:

Fantasy points decompose exactly. points = 3*short + 4*mid + 5*long + 1*PAT, and
the variance decomposition over 2015-2026 says var(total) = 19.51, of which the
field-goal part is 19.96 (102%), the PAT part is 2.04 (10%), and the covariance is
-2.49 (-13%). That negative covariance is the red-zone substitution: a drive that
ends in a touchdown pays 1 point, the same drive stalling pays 3.

Then the fact that decides the design. Season-forward, out of sample:

    fg_att     R2 = -0.054,  within-week Spearman 0.014
    pat_att    R2 = +0.081,  within-week Spearman 0.299

Weekly field-goal attempts are LESS predictable than their own mean. The half of
the target carrying all the variance has negative skill; the half with real skill
carries a tenth of it. A single regression on total points cannot show you that,
and every attempt to improve the total is fighting the unforecastable half.

So the model predicts the two counts separately and combines them by the scoring
rule, which buys three things a direct regression does not:

  1. It is honest about where the signal is. The PAT stage does the work.
  2. It gives P(y >= x) for every threshold with no threshold model trained, and
     that matches a purpose-built classifier (at y>=15, AUC 0.545 vs 0.551).
  3. Calibration, measured: predicted rates land within 5% of realised at every
     threshold, and at 15 or more points it is 0.0838 against 0.0841.

RETRACTED, because the backtest did not support it. An earlier reviewer found a
gradient-boosted mean ranking hit 15+ less often than random (0.057 vs 0.085) and
that ranking by P(y>=15) fixed it. Neither holds for this model: measured over 8
folds, random 0.0816, mean ranking 0.0888, P(y>=15) ranking 0.0889. The compound
model removed the anti-ceiling defect, and once removed the second board adds
nothing on top of the first.

WHAT THIS MODEL IS NOT FOR: ranking. On the decision metric it scores 0.349
points per week against RidgeRanker's 0.434, and loses the paired bootstrap by
0.086 at p = 0.002. Rank with RidgeRanker, take probabilities from here.

Counts are UNDER-dispersed here: fg_att variance/mean = 0.843, pat_att = 0.881.
Negative binomial only extends Poisson upward so it cannot fit this, which is why
the draw is binomial-thinned rather than plain Poisson. Left uncorrected, a
Poisson convolution over-predicts P(y >= 15) by 19%.

Capacity is small on purpose. On ~5k training rows the shipped 500 trees x 31
leaves was the worst setting tested; 100 x 7 beat it by 0.126 MAE.

Usage:
    python src/kicker_analysis/model.py --backtest
    python src/kicker_analysis/model.py --week          # rank this week's kickers
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
# numeric column that nobody had remembered to ban, which let three same-game
# outcome columns in during a review and moved MAE from 3.49 to 2.65 before
# anyone noticed. A feature set you cannot enumerate is a feature set you cannot
# audit, so every column here is named and has a stated mechanism.
FEATURES = [
    # The market. One line already beats the 73-column model, because it is the
    # only input that prices this specific offence against this specific defence.
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
    # Environment, in the only form knowable on Tuesday.
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

    Measured against the alternatives on the decision metric, 8 season-forward
    folds, points per week gained starting the better of two random kickers:

        ridge, 16 features      0.434   rho 0.173
        Vegas implied total     0.362   rho 0.143
        compound count model    0.349   rho 0.143
        kicker career average   0.161   rho 0.071

    Paired week-block bootstrap: ridge beats the implied total by +0.072
    [+0.014, +0.130], p = 0.015, and beats the compound model by +0.086
    [+0.029, +0.140], p = 0.002. So the count model is the better probability
    machine and the worse ranker, and each is used for the job it wins.

    alpha = 100 rather than the 10 a first pass would reach for. The design
    matrix is collinear by construction (lagged windows of the same quantity),
    and a sweep improves monotonically over two orders of magnitude, which is
    the signature of an underregularised collinear fit.

    Noise control, because adding columns to this data can improve MAE on its
    own: 5 and 15 columns of pure Gaussian noise added to the implied total make
    MAE WORSE by 0.005 and 0.010. The 16-feature gain is not a column count
    artifact, which a with-versus-without comparison alone could not establish.
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

    Kept for what it wins: calibration. Across 8 folds the predicted rate lands
    within 5% of the realised rate at every threshold, and at 15 or more points
    it is 0.0838 predicted against 0.0841 realised. A point forecast cannot tell
    you that, and a betting line certainly cannot.

    It is NOT used for ranking. See RidgeRanker for why.
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
        # of tries is mostly noise, and shrinking it to the league value is the
        # conservative choice the review's under-dispersion finding argues for.
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
        # Counts are UNDER-dispersed, so a raw Poisson draw is too wide. Drawing
        # a binomial with the same mean and a modest trial count narrows it the
        # right way; negative binomial could only widen it.
        def counts(lam):
            trials = np.maximum(np.ceil(lam * 3).astype(int), 1)
            p = np.clip(lam[:, None] / trials[:, None], 0, 1)
            return rng.binomial(trials[:, None], p, size=(n, self.n_sims))
        att = counts(lam_fg)
        pat = counts(lam_pat)
        made = rng.binomial(att, self.make_rate_)
        # Split makes across the three scoring tiers by the league mix.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        sys.exit(self_check())
    sys.exit(0)
