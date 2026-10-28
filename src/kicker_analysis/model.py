"""The kicker models: a ridge that orders the board, and two count stages drawn
into a full predictive law.

WHY THIS SHAPE, which is the whole argument:

Fantasy points decompose exactly. points = 3*short + 4*mid + 5*long + 1*PAT, and
over the 2015-2026 regular-season rows the field-goal part's variance is about
102% of the total's, the PAT part's about 10%, and twice their covariance about
-12% (reports/variance_decomposition.csv has the numbers). That negative
covariance is the red-zone substitution: a drive that ends in a touchdown pays 1
point, the same drive stalling pays 3.

Then the fact that decides the design. Season-forward, out of sample, the count
stage has close to no skill on weekly field-goal attempts: R2 against the
training mean is about +0.003 with an interval that spans zero, and within-week
Spearman about 0.07. On PAT attempts it has real skill, R2 about +0.13 and
Spearman about 0.35 (reports/component_skill.csv, written by backtest.py). The
half of the target carrying almost all the variance is close to unforecastable;
the half with real skill carries a tenth of it. A single regression on total
points cannot show you that, and every attempt to improve the total is fighting
the unforecastable half.

So there are two models here, each used for the job it does better:

  RidgeRanker          orders the board. On the decision metric it beats the
                       count model's expected points, the betting line alone
                       and a kicker's career average (reports/backtest_summary.csv).
  CompoundKickerModel  predicts the two counts separately and combines them by
                       the scoring rule, which gives P(y >= x) for every
                       threshold with no threshold model trained. Pooled over
                       the 8 folds its predicted rates land within about 5% of
                       the realised rates at 8, 10, 12 and 15 points, but it
                       discriminates weakly (AUC about 0.55 to 0.58), and at
                       15 points its Brier skill over a constant at the
                       training base rate is indistinguishable from zero
                       (reports/calibration_summary.csv). Its docstring says why
                       the pooled match is partly errors offsetting.

RETRACTED, because the backtest did not support it. An earlier version of this
docstring said a gradient-boosted mean ranking hit 15+ less often than random
(0.057 vs 0.085) and that ranking by P(y>=15) fixed it. Neither holds for this
model: in the ceiling_r2 column of reports/backtest_summary.csv both the compound
model's expected-points ranking and its P(y>=15) ranking pick a 15-point week
more often than a random pick does (the constant row, whose ties split at
random), and the second board adds nothing on top of the first.

Capacity is small on purpose. On a few thousand training rows a 100-tree,
7-leaf count stage predicts better out of sample than the 500-tree, 31-leaf
settings of the first harness (legacy/evaluate_v1.py): with those settings and
the same Poisson objective, the compound model's expected-points MAE over the 8
folds was about 0.05 worse.

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
    implied total alone and a kicker's career average. The pooled numbers are
    in reports/backtest_summary.csv, the season-level tests in
    reports/season_level_tests.csv, and the bake-off against fourteen tuned
    challengers in notes/model-selection-2026-09-29.md. Its margin over the
    line alone is consistent in the pooled weekly average and thin season by
    season: it wins 6 of the 8 test seasons. So the count model is the
    probability machine and this is the ranker.

    alpha = 100 is a fixed choice, not a tuned optimum. The design matrix is
    collinear by construction (lagged windows of the same quantity), so heavy
    shrinkage is expected, but the data do not pin alpha down. In the nested
    bake-off an inner loop choosing alpha on training seasons only picked
    values anywhere from 1 to 10,000 across the 8 folds
    (reports/model_bakeoff_params.csv), and that nested ridge did not beat the
    fixed alpha = 100 (reports/model_bakeoff_summary.csv).

    Noise control, because adding columns can move MAE on its own: at
    alpha = 100, columns of pure Gaussian noise make this ridge's pooled MAE
    worse on average, not better, and more so the more are added (13 to 61
    noise columns on top of the 16 features, 50 draws per width, in
    experiments/feature_families/ridge_retest.py and
    reports/feature_family_ridge_retest.csv). Against the same ridge on the
    implied total alone, the 16 named features lower pooled MAE by about
    0.015 over the 8 folds (3.472 against 3.486), so that gain is not a
    column-count artifact, which a with-versus-without comparison alone could
    not establish. It is a small gain in MAE; the case for the ridge rests on
    the decision metric above.
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
    threshold. Pooled over the 8 backtest folds its predicted rates land within
    about 5% of the realised rates at 8, 10, 12 and 15 points
    (reports/calibration_summary.csv). It is NOT used for ranking; see
    RidgeRanker for why.

    What _draw actually does, and where it is knowingly wrong:

      counts    each count is a binomial with n = max(ceil(3 * lambda), 1)
                trials and p = lambda / n, so its mean is lambda and its
                variance/mean is 1 - lambda / n: 0.67 to 0.73 for the lambdas
                this model predicts, about 0.69 on average. That ratio is
                fixed by the 3, not fitted. The counts are under-dispersed
                (raw variance/mean about 0.84 for FG attempts and 0.88 for
                PAT attempts, 2015-2026), which is why a Poisson draw is not
                used: swapped in with everything else unchanged, it
                over-predicts the pooled P(y >= 15) by about a third. But out
                of sample the squared residuals over lambda come to about 0.8
                for both counts, so the fixed 0.69 is somewhat too narrow.
      FG, PAT   drawn independently. Out of sample their residuals correlate
                at about -0.25 pooled and are negative in every test season
                (-0.18 to -0.29): a drive that ends in a touchdown is one that
                did not end in a field goal. Ignoring that makes the simulated
                total wider than it should be.
      distance  makes are split into the 3-, 4- and 5-point tiers with random
                per-draw weights (uniform times the league mix, renormalised)
                and rounding, with the long tier taking the remainder. The
                random weights, and the rounding on top of them, pull the
                split away from the league mix toward the longer, rarer
                tiers, which adds about 0.07 points per make (about 3.60
                against the mix's 3.53).

    So the pooled 15+ match is partly these errors offsetting: counts too
    narrow, a total too wide, and makes worth slightly too much. With the makes
    split at exactly the league mix the same draw predicts P(y >= 15) about 6%
    below the realised rate. Per row it discriminates weakly (AUC about 0.55 at
    15 points), so read P(15+) as right on average rather than sharp per kicker.
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
        # 1 - lambda / trials, about 0.69, a constant set by the 3 rather than
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
