"""Kicker research models with explicit selected-league outcome scoring.

Ridge remains a research ranker. GLMs model FG/PAT opportunities, attempted
distance and conversions separately. Shared event accounting produces the
distribution; conversion and count uncertainty are estimated on training data.
Historical reports from earlier simulator versions do not validate this law.
The product uses a transparent market reference until real policy evidence
supports a more complex recommendation. See the research charter and protocol.
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
from kicker_analysis.scoring import BASE_SCORING, MADE_BUCKETS  # noqa: E402

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
    "climate_wind_missing",
    "is_home",
]

# Scoring, matching build_dataset.SCORING. Kept here so the convolution and the
# target can be checked against each other.
PTS_SHORT, PTS_MID, PTS_LONG, PTS_PAT = 3.0, 4.0, 5.0, 1.0
N_SIMS = 4000
FIRST_SEASON = 2015
MIN_TEST_WEEKS = 8


def load(processed: pathlib.Path | None = None, scoring=BASE_SCORING) -> pd.DataFrame:
    proc = processed or PROC
    snapshot = None
    if proc != PROC or (proc / "quality.json").exists():
        import io
        from kicker_analysis.prepare import prepared_snapshot
        snapshot = prepared_snapshot(proc)
        games = pd.read_parquet(io.BytesIO(snapshot["kicker_games.parquet"]))
        fourth = pd.read_parquet(io.BytesIO(snapshot["fourth_down.parquet"]))
    else:
        games = pd.read_parquet(proc / "kicker_games.parquet")
        fourth = pd.read_parquet(proc / "fourth_down.parquet")
    sched = pd.read_csv(ROOT / "data" / "external" / "games.csv", low_memory=False)
    df = games.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    df = df[(df.season >= FIRST_SEASON) & (df.season_type == "REG")].copy()
    # Re-score raw labels BEFORE point lags are built, never adjust a fitted mean.
    df["fantasy_points"] = scoring.points(df)
    df = build_features(df, games=sched)
    coaches = proc / "coach_history.parquet"
    if snapshot or coaches.exists():
        coach_data = pd.read_parquet(io.BytesIO(snapshot["coach_history.parquet"])) if snapshot else pd.read_parquet(coaches)
        df = df.merge(coach_data.drop(columns=["coach"]), on=["season", "week", "team"],
                      how="left", validate="many_to_one")
    return df


def _count_model():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="poisson", n_estimators=100, num_leaves=7,
                         learning_rate=0.05, min_child_samples=60,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         reg_lambda=1.0, verbose=-1, random_state=0, n_jobs=1)


class RidgeRanker:
    """Regularized direct-points research comparator. No proven lineup edge.

    Alpha100 is fixed. Collinear features and adaptive historical model selection
    limit coefficient and superiority claims. Use corrected, versioned evidence.
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
        # Miss penalties permit negative fantasy outcomes and expected values.
        return self.m_.predict(test[self.features])


class CompoundKickerModel:
    """Experimental count distribution, separate from the ridge ranking mean.

    FG/PAT conversion probabilities and six made-distance proportions use only
    training rows. Exact categorical allocation conserves every made kick and
    failed PATs receive the selected scoring penalty. The arithmetic mean of
    this law is exported separately from its Monte Carlo estimate.

    Remaining assumptions are explicit research limitations: total-opportunity
    count family, conditional binomial FG/PAT allocation and pooled conversion.
    Earlier calibration reports concern the old simulator, whose PAT and distance
    allocation errors partly offset. They do not validate this corrected law.
    Re-evaluate season-forward before promoting probabilities or changing a lineup.
    """

    def __init__(self, features: list[str] | None = None, n_sims: int = N_SIMS,
                 scoring=BASE_SCORING):
        self.features = features or FEATURES
        self.n_sims = n_sims
        self.scoring = scoring

    def fit(self, train: pd.DataFrame):
        from sklearn.impute import SimpleImputer
        self.imp_, self.fg_, self.pat_ = self._fit_counts(train)
        # Make rate and distance mix are league constants estimated on the
        # training years, not per-kicker: a kicker's own 50+ rate over a handful
        # of tries is mostly noise, and shrinking it all the way to the league
        # value is the conservative choice.
        att = train.fg_att.sum()
        self.make_rate_ = float(train.fg_made.sum() / att) if att else 0.83
        pat_att = train.pat_att.sum()
        if not pat_att or not att:
            raise ValueError("training requires observed FG and PAT attempts")
        self.pat_make_rate_ = float(train.pat_made.sum() / pat_att)
        made = train[list(MADE_BUCKETS)].sum().to_numpy(dtype=float)
        if made.sum() <= 0 or not (0 <= self.make_rate_ <= 1
                                   and 0 <= self.pat_make_rate_ <= 1):
            raise ValueError("invalid training kick-conversion counts")
        self.mix_ = made / made.sum()
        from kicker_analysis.build_dataset import ATT_BUCKETS
        if set(ATT_BUCKETS) <= set(train):
            attempted = train[ATT_BUCKETS].sum().to_numpy(float)
            self.mix_ = attempted / attempted.sum()
            self.bucket_make_ = (made + .5) / (attempted + 1.)
        # Estimate dispersion on a final TRAINING season held out from count fits.
        # This cannot use the outer test or compensate for a desired tail rate.
        tail = int(train.season.max())
        early, calibration = train[train.season < tail], train[train.season == tail]
        if len(early) and len(calibration) >= 30:
            imp, fg, pat = self._fit_counts(early)
            Xc = imp.transform(calibration[self.features])
            mf, mp = fg.predict(Xc), pat.predict(Xc)
            residual = calibration.fg_att + calibration.pat_att - mf - mp
            self.raw_dispersion_ = float(np.mean(residual ** 2) / np.mean(mf + mp))
            self.total_dispersion_ = float(np.clip(self.raw_dispersion_, .05, 5.))
            self.residual_correlation_ = float(np.corrcoef(
                calibration.fg_att - mf, calibration.pat_att - mp)[0, 1])
        else:
            self.raw_dispersion_ = self.total_dispersion_ = 1.
            self.residual_correlation_ = np.nan
        return self

    def _fit_counts(self, train):
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median").fit(train[self.features])
        X = imp.transform(train[self.features])
        return (imp, _count_model().fit(X, train.fg_att),
                _count_model().fit(X, train.pat_att))

    def _probabilities(self, test, X):
        n = len(test)
        return {"mix": np.broadcast_to(self.mix_, (n, 6)),
                "fg": np.broadcast_to(getattr(self, "bucket_make_", self.make_rate_), (n, 6)),
                "pat": np.full(n, self.pat_make_rate_)}

    def _draw(self, X, seed: int = 0, probabilities=None) -> np.ndarray:
        """Monte-Carlo the predictive law. Rows x sims of total fantasy points."""
        rng = np.random.default_rng(seed)
        lam_fg = np.clip(self.fg_.predict(X), 0, None)
        lam_pat = np.clip(self.pat_.predict(X), 0, None)
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
        if hasattr(self, "total_dispersion_"):
            mean = lam_fg + lam_pat
            phi = self.total_dispersion_
            if phi < 1:
                trials = np.maximum(np.ceil(mean / (1 - phi)).astype(int), 1)
                total = rng.binomial(trials[:, None], mean[:, None] / trials[:, None],
                                     size=(n, self.n_sims))
            elif phi > 1:
                shape = np.maximum(mean / (phi - 1), 1e-8)
                total = rng.negative_binomial(shape[:, None], 1 / phi,
                                              size=(n, self.n_sims))
                total[mean == 0] = 0
            else:
                total = rng.poisson(mean[:, None], size=(n, self.n_sims))
            share = np.divide(lam_fg, mean, out=np.zeros(n), where=mean > 0)
            att = rng.binomial(total, share[:, None])
            pat = total - att
        else:
            att, pat = counts(lam_fg), counts(lam_pat)
        params = probabilities or self._probabilities(pd.DataFrame(index=range(n)), X)
        pp = (rng.beta(params["pat_a"][:, None], params["pat_b"][:, None],
                       size=(n, self.n_sims)) if "pat_a" in params else params["pat"][:, None])
        pat_made = rng.binomial(pat, pp)
        # Sequential conditional binomials are an exact multinomial allocation:
        # every make belongs to one bucket, with no rounding or extra long kicks.
        remaining = att.copy()
        buckets = []
        for i in range(6):
            tail = params["mix"][:, i:].sum(axis=1)
            weight = np.divide(params["mix"][:, i], tail,
                               out=np.zeros(n), where=tail > 0)
            count = (rng.binomial(remaining, np.clip(weight[:, None], 0, 1))
                     if i < 5 else remaining.copy())
            success = (rng.beta(params["fg_a"][:, i, None], params["fg_b"][:, i, None],
                               size=(n, self.n_sims)) if "fg_a" in params
                       else params["fg"][:, i, None])
            buckets.append(rng.binomial(count, success))
            remaining -= count
        return self.scoring.components(buckets, pat_made, att - sum(buckets), pat - pat_made)

    def predict(self, test: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        X = self.imp_.transform(test[self.features])
        params = self._probabilities(test, X)
        sims = self._draw(X, seed=seed, probabilities=params)
        out = pd.DataFrame(index=test.index)
        fg = np.clip(self.fg_.predict(X), 0, None)
        pat = np.clip(self.pat_.predict(X), 0, None)
        # Exact mean of this predictive law; tails remain Monte Carlo estimates.
        expected_made = fg[:, None] * params["mix"] * params["fg"]
        out["exp_fg_points"] = (expected_made @ np.asarray(self.scoring.fg_made)
            + (fg - expected_made.sum(axis=1)) * self.scoring.fg_missed)
        out["exp_pat_points"] = pat * (params["pat"] * self.scoring.pat_made
                                      + (1 - params["pat"]) * self.scoring.pat_missed)
        out["exp_points"] = out.exp_fg_points + out.exp_pat_points
        out["simulation_mean_points"] = sims.mean(axis=1)
        out["median_points"] = np.median(sims, axis=1)
        out["floor_p20"] = np.percentile(sims, 20, axis=1)
        out["ceiling_p90"] = np.percentile(sims, 90, axis=1)
        for thr in (8, 10, 12, 15):
            out[f"p_ge_{thr}"] = (sims >= thr).mean(axis=1)
        out["exp_fg_att"] = np.clip(self.fg_.predict(X), 0, None)
        out["exp_pat_att"] = np.clip(self.pat_.predict(X), 0, None)
        if "fantasy_points" in test:
            y = test.fantasy_points.to_numpy(float)
            ordered = np.sort(sims, axis=1)
            weights = 2 * np.arange(1, self.n_sims + 1) - self.n_sims - 1
            out["crps"] = np.abs(sims - y[:, None]).mean(axis=1) - (
                ordered @ weights) / self.n_sims ** 2
        return out


GLM_FEATURES = ["implied_total", "opp_implied_total", "is_home",
                "is_indoor_certain", "is_retractable"]
COACH_FEATURES = ["coach_fg_adjustment", "coach_go_adjustment", "coach_long_kick_rate",
                  "coach_long_menu_prior", "k_fg_att_55p_career", "k_fg_made_55p_career"]


class GlmKickerModel(CompoundKickerModel):
    """Small count GLMs with pooled or conjugate partially pooled conversion."""

    def __init__(self, strength=None, coach=False, **kwargs):
        super().__init__(features=GLM_FEATURES + (COACH_FEATURES if coach else []), **kwargs)
        self.strength, self.coach = strength, coach

    def _fit_counts(self, train):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import PoissonRegressor
        imp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        X = imp.fit_transform(train[self.features])
        return (imp, PoissonRegressor(alpha=1., max_iter=500).fit(X, train.fg_att),
                PoissonRegressor(alpha=1., max_iter=500).fit(X, train.pat_att))

    def fit(self, train):
        from kicker_analysis.build_dataset import ATT_BUCKETS
        super().fit(train)
        if not set(ATT_BUCKETS) <= set(train):
            raise ValueError("run prepare.py; distance-conditional conversion needs attempts")
        self.train_ = train.groupby("player_id")[list(MADE_BUCKETS) + ATT_BUCKETS
                            + ["pat_made", "pat_att"]].sum()
        self.population_ = train[list(MADE_BUCKETS) + ATT_BUCKETS + ["pat_made", "pat_att"]].sum()
        self.distance_ = None
        if self.coach:
            from sklearn.linear_model import LogisticRegression
            X = self.imp_.transform(train[self.features])
            counts = train[ATT_BUCKETS].to_numpy(float)
            rows, bucket = np.nonzero(counts)
            self.distance_ = LogisticRegression(C=1., max_iter=500).fit(
                X[rows], bucket, sample_weight=counts[rows, bucket])
        return self

    def _probabilities(self, test, X):
        from kicker_analysis.build_dataset import ATT_BUCKETS
        p = super()._probabilities(test, X)
        total = self.population_
        a = total[list(MADE_BUCKETS)].to_numpy(float) + .5
        b = total[ATT_BUCKETS].to_numpy(float) - a + 1.
        pa, pb = total.pat_made + .5, total.pat_att - total.pat_made + .5
        if self.strength is None:
            p.update(fg_a=np.broadcast_to(a, (len(test), 6)),
                     fg_b=np.broadcast_to(b, (len(test), 6)),
                     pat_a=np.full(len(test), pa), pat_b=np.full(len(test), pb))
        else:
            k = self.train_.reindex(test.player_id).fillna(0)
            made = k[list(MADE_BUCKETS)].to_numpy(float)
            attempts = k[ATT_BUCKETS].to_numpy(float)
            rate = a / (a + b)
            p.update(fg_a=made + self.strength * rate,
                     fg_b=attempts - made + self.strength * (1 - rate),
                     pat_a=k.pat_made.to_numpy() + self.strength * pa / (pa + pb),
                     pat_b=(k.pat_att - k.pat_made).to_numpy() + self.strength * pb / (pa + pb))
        p["fg"] = p["fg_a"] / (p["fg_a"] + p["fg_b"])
        p["pat"] = p["pat_a"] / (p["pat_a"] + p["pat_b"])
        if self.distance_ is not None:
            p["mix"] = np.zeros((len(test), 6))
            p["mix"][:, self.distance_.classes_] = self.distance_.predict_proba(X)
        return p


def self_check() -> int:
    """The scoring identity and the direction of the two rankings."""
    # The convolution must reproduce the scoring rule in expectation. Build a
    # model with known constants and check the mean lands where arithmetic says.
    m = CompoundKickerModel(features=["a"], n_sims=8000)
    m.make_rate_ = 1.0
    m.pat_make_rate_ = 1.0
    m.mix_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
