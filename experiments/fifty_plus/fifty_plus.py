"""Willingness to attempt a 50+ yard field goal, measured on its own.

WHY A SEPARATE MEASUREMENT. fourth_down.py counts every fourth down at
`yardline_100` <= 38 as one bucket, which mixes the routine 40-yard decision
with the long one. Whether a staff sends its kicker out from the opponent's 35
is a separate choice, and under this scoring a 5-point one, so it is measured
separately here, in two yardline bands:

  long band     the attempt would be 50 to 60 yards.
  short band    the attempt would be 37 to 49 yards. The control: if the long
                rate moves and this one does not, the movement is about
                distance and not about fourth downs in general.

WHICH YARDLINES THOSE ARE. METHOD.md converts with `distance = yardline_100 +
17` (ten yards of end zone, seven of snap and hold), giving yardline 33 to 43
and 20 to 32. The play-by-play disagrees by a yard: over 2015-2026 regular-season
attempts, kick_distance - yardline_100 is 18 for about nine in ten, 19 for most
of the rest and 17 for under 2% (the run prints the exact split). So the
documented long band is mostly 51 to 61 yard attempts and a typical 50-yarder
sits in the short control. Both conversions are run: +17 reproduces the
documented analysis, and the offset measured from the data (its mode) is the
band that actually matches the 5-point tier. Every output row says which one
(`fg_offset`).

Every other filter is fourth_down.py's, restated in band_plays with its
thresholds and franchise relocations imported from it: regular season only, win
probability between 5% and 95%, not in the last two minutes of either half, and
the play must be exactly one of field goal, punt or a run/pass attempt. Punts
stay in the denominator, because a punt from the 35 is a declined field goal.
The rate is FG attempts / decisions.

WHAT IS REPORTED, and why each piece is needed:

  rates      league and team-season rates per season, both bands
             (reports/fifty_plus_rates.csv).
  variance   a team gets few long-band chances a season, so much of the visible
             spread between teams is binomial sampling noise. Per season, the
             observed cross-team variance of team rates is split by the method of
             moments. For a team whose true rate theta has league mean mu and
             cross-team variance tau2, the observed rate r = x / n has
                 Var(r) = tau2 + (mu (1 - mu) - tau2) / n,
             so with m = mean(1 / n) over teams, the across-team sample variance
             (ddof 1, unbiased for the mean of the per-team variances) gives
                 tau2 = (observed - mu (1 - mu) m) / (1 - m).
             share_real = tau2 / observed is the part of the visible spread that
             is a property of the team. The empirical-Bayes prior strength, in
             pseudo-observations, is k = (mu (1 - mu) - tau2) / tau2 (the a + b
             of a beta prior with that mean and variance), and a team with n
             chances keeps n / (n + k) of its own rate. The interval on
             share_real resamples teams. Year-over-year persistence of the raw
             team rate is set beside that of fourth_down.py's go_rate
             (reports/fifty_plus_variance.csv).
  lift       does the team's prior-season willingness, shrunk by that k, help the
             16-feature ridge that orders the board? Same season-forward split as
             backtest.py (test seasons from 2018 with at least 8 weeks, fit on
             every earlier season), exact decision lift from backtest.decision,
             paired week-block bootstrap from backtest.boot_lift, and the same
             per-season differences treated as one observation per season,
             because a week-block interval treats weeks inside one season as
             independent (reports/fifty_plus_lift.csv).

NO LEAKAGE, by construction. The feature for season s uses only season s - 1:
the team's long-band decisions and attempts that season, shrunk toward that
season's league rate with k pooled over seasons up to and including s - 1. All
of it is fixed before week 1 of season s, the same lag as
features.prev_go_rate. The team x season grid is dense before shifting, so a
missing season becomes a missing feature, not the year before it. The raw
prior-season rate is run as a second arm to show what the shrinkage itself
buys. The arm under test, fixed before the run, is the shrunk feature on the
documented band (`primary` in the lift table).

WHAT THE RUN SAYS (fifty_plus.log beside this file; play-by-play 2015 to 2026
week 7; documented +17 band unless stated):

  * The league long rate sat between 0.243 and 0.296 from 2015 to 2020, then
    rose from 0.316 in 2021 to 0.440 in 2025. The short control fell from
    0.908 in 2015 to 0.781 in 2021 and has stayed between 0.78 and 0.83 since,
    so "long rises while short stays flat" holds from 2021 on, not over the
    whole record.
  * 2025: median 9 long-band chances per team, New England 0 of 7, Dallas 11 of
    12. Share of the cross-team spread that is real: 52.0%, teams-resampled
    interval [11.4%, 69.7%]. Across the 11 complete seasons the same estimate
    runs from -22.7% to 63.8% (median 24.5%), so one season's share is not a
    stable property. k is 8.0 from 2025 alone and 15.3 pooled over 2015-2025.
  * Year-over-year r of raw team long rates averages +0.20 over 10 season
    pairs (one negative); the short control +0.02; fourth_down.py's go_rate
    +0.27.
  * Decision lift, primary arm: +0.0023 points a week over the 16-feature ridge
    (0.4295 vs 0.4272), paired week-block 95% CI [-0.0074, +0.0121], p 0.64; it
    wins 4 of 8 test seasons (t-test on the 8 season means p 0.69). Its ridge
    coefficient is positive in 7 of 8 folds, but the folds are expanding
    windows that share most of their training seasons, so they are not eight
    independent confirmations of the sign. The raw rate does no better
    (+0.0031, p 0.31), and on the band that matches the 5-point tier (+18) both
    are within +0.001 of zero. No detectable gain; it stays out of the sixteen.

    python experiments/fifty_plus/fifty_plus.py               # writes the three CSVs
    python experiments/fifty_plus/fifty_plus.py --self-check  # offline, synthetic, writes nothing

Needs data/processed/ (build_dataset.py, fourth_down.py) and the play-by-play
cache in data/external/pbp_<season>.parquet that those two fill.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
# experiments/fifty_plus/<this file>, so the repo root is two levels up.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kicker_analysis.backtest import boot_lift, decision  # noqa: E402
from kicker_analysis.fourth_down import (  # noqa: E402
    MIN_HALF_SECONDS, RELOCATIONS, WP_HI, WP_LO, fourth_down_plays)
from kicker_analysis.model import FEATURES, RidgeRanker  # noqa: E402

CACHE = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FIRST_SEASON = 2015          # fourth_down.py's floor: clean wp and play typing
DOC_OFFSET = 17              # METHOD.md's conversion, distance = yardline_100 + 17
LONG_YARDS = (50, 60)        # the 5-point tier, up to a 60-yard try
SHORT_YARDS = (37, 49)       # the control
FULL_SEASON_WEEKS = 17       # 2015-2020 had 17 regular-season weeks, 2021 on 18
HEADLINE_SEASON = 2025
# The same split as backtest.py, restated so this file does not depend on two
# constants that module does not promise to keep.
FIRST_TEST = 2018
MIN_WEEKS = 8
N_BOOT = 10000
SEED = 0
BASE = "ridge16"

PBP_COLS = ["season", "week", "season_type", "game_id", "posteam", "down",
            "ydstogo", "yardline_100", "field_goal_attempt", "punt_attempt",
            "rush_attempt", "pass_attempt", "wp", "half_seconds_remaining",
            "kick_distance"]


def bands(offset: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(long, short) yardline_100 ranges for a distance = yardline + offset rule."""
    return ((LONG_YARDS[0] - offset, LONG_YARDS[1] - offset),
            (SHORT_YARDS[0] - offset, SHORT_YARDS[1] - offset))


def label(offset: int) -> str:
    (l0, l1), _ = bands(offset)
    return f"yardline {l0}-{l1}"


def feature_names(offset: int) -> tuple[str, str]:
    return f"prev_fifty_shrunk_o{offset}", f"prev_fifty_raw_o{offset}"


# ------------------------------------------------------------------- the plays

def measured_offset(pbp: pd.DataFrame) -> tuple[int, pd.Series]:
    """Modal kick_distance - yardline_100 on regular-season FG attempts, and the
    share of attempts at each offset."""
    f = pbp[(pbp.field_goal_attempt == 1) & (pbp.season_type == "REG")]
    off = (f.kick_distance - f.yardline_100).dropna().astype(int)
    share = off.value_counts(normalize=True)
    return int(share.idxmax()), share


def band_plays(pbp: pd.DataFrame, offset: int = DOC_OFFSET) -> pd.DataFrame:
    """Regular-season fourth-down decisions in either band, one row per play.

    The filters are fourth_down.py's, with its yardline cut replaced by the two
    bands, so the rates here and the fourth-down features count the same kind
    of play.
    """
    (l0, l1), (s0, _) = bands(offset)
    d = pbp[(pbp.down == 4) & pbp.posteam.notna() & (pbp.season_type == "REG")]
    d = d[d.yardline_100.between(s0, l1)]
    d = d[d.wp.between(WP_LO, WP_HI) & (d.half_seconds_remaining > MIN_HALF_SECONDS)]
    d = d.assign(is_fg=(d.field_goal_attempt == 1).astype(int),
                 is_punt=(d.punt_attempt == 1).astype(int),
                 is_go=((d.rush_attempt == 1) | (d.pass_attempt == 1)).astype(int))
    d = d[(d.is_fg + d.is_punt + d.is_go) == 1].copy()
    d["band"] = np.where(d.yardline_100 >= l0, "long", "short")
    d["team"] = d.posteam.replace(RELOCATIONS)
    return d


def rates(plays: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Decisions, FG attempts, go and punt counts and the FG rate, per band."""
    g = (plays.groupby(by + ["band"]).agg(n=("is_fg", "size"), fg=("is_fg", "sum"),
                                          go=("is_go", "sum"), punt=("is_punt", "sum"))
         .unstack("band", fill_value=0))
    out = pd.DataFrame(index=g.index)
    for band in ("long", "short"):
        for c in ("n", "fg", "go", "punt"):
            out[f"{band}_{c}"] = g[(c, band)] if (c, band) in g.columns else 0
        out[f"{band}_rate"] = out[f"{band}_fg"] / out[f"{band}_n"].replace(0, np.nan)
    return out.reset_index()


# -------------------------------------------------------------- the variance

def decompose(x: np.ndarray, n: np.ndarray) -> dict:
    """Split the cross-team variance of x / n into sampling noise and true spread.

    Method of moments, derived in the module docstring. Teams with n = 0 have no
    rate and are left out. tau2 can come out negative when the observed spread
    is no wider than binomial noise alone; it is reported as it falls, and the
    shrinkage constant is then infinite (pool every team to the league rate).
    """
    keep = np.asarray(n) > 0
    x, n = np.asarray(x, float)[keep], np.asarray(n, float)[keep]
    r = x / n
    mu = x.sum() / n.sum()
    obs = float(np.var(r, ddof=1))
    m = float(np.mean(1.0 / n))
    tau2 = (obs - mu * (1 - mu) * m) / (1 - m)
    k = (mu * (1 - mu) - tau2) / tau2 if tau2 > 0 else np.inf
    return {"n_teams": int(keep.sum()), "decisions": int(n.sum()),
            "median_n": float(np.median(n)), "league_rate": float(mu),
            "var_observed": obs, "var_noise": obs - tau2, "var_true": float(tau2),
            "share_real": float(tau2 / obs) if obs > 0 else np.nan,
            "k_prior": float(k), "pq": float(mu * (1 - mu))}


def share_interval(x: np.ndarray, n: np.ndarray, boot: int = N_BOOT,
                   seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval on share_real, resampling teams with replacement."""
    keep = np.asarray(n) > 0
    x, n = np.asarray(x, float)[keep], np.asarray(n, float)[keep]
    idx = np.random.default_rng(seed).integers(0, len(x), size=(boot, len(x)))
    xb, nb = x[idx], n[idx]
    mu = xb.sum(1) / nb.sum(1)
    obs = np.var(xb / nb, axis=1, ddof=1)
    m = np.mean(1.0 / nb, axis=1)
    tau2 = (obs - mu * (1 - mu) * m) / (1 - m)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(obs > 0, tau2 / obs, np.nan)
    lo, hi = np.nanpercentile(share, [2.5, 97.5])
    return float(lo), float(hi)


def pooled_k(var_table: pd.DataFrame, through: int) -> float:
    """Shrinkage constant from every season up to and including `through`.

    One season's tau2 rests on 32 noisy team rates and can fall at or below
    zero, so the moments are averaged over seasons first: mean tau2 and mean
    mu (1 - mu), then k = (pq - tau2) / tau2. Nothing after `through` is read.
    """
    v = var_table[var_table.season <= through]
    tau2, pq = float(v.var_true.mean()), float(v.pq.mean())
    return (pq - tau2) / tau2 if tau2 > 0 else np.inf


def persistence(team: pd.DataFrame, col: str) -> pd.Series:
    """Pearson r across teams between season s and season s - 1, indexed by s."""
    w = team.pivot(index="team", columns="season", values=col)
    out = {}
    for s in w.columns:
        if s - 1 in w.columns:
            pair = w[[s - 1, s]].dropna()
            out[s] = float(pair[s - 1].corr(pair[s])) if len(pair) > 2 else np.nan
    return pd.Series(out, dtype=float)


def variance_table(team: pd.DataFrame, go: pd.DataFrame, weeks: pd.Series,
                   boot: int = N_BOOT) -> pd.DataFrame:
    """One row per season: decomposition, pooled k, and year-over-year r."""
    rows = []
    for s, t in team.groupby("season"):
        d = decompose(t.long_fg.to_numpy(), t.long_n.to_numpy())
        lo, hi = share_interval(t.long_fg.to_numpy(), t.long_n.to_numpy(), boot)
        rows.append({"season": int(s), "weeks": int(weeks.get(s, 0)), **d,
                     "share_real_lo": lo, "share_real_hi": hi})
    v = pd.DataFrame(rows)
    v["complete"] = v.weeks >= FULL_SEASON_WEEKS
    for c in ("observed", "noise", "true"):
        v[f"sd_{c}"] = np.sqrt(v[f"var_{c}"].clip(lower=0))
    v["own_weight_at_median"] = v.median_n / (v.median_n + v.k_prior)
    v["k_pooled_through"] = [pooled_k(v, s) for s in v.season]
    v["yoy_r_long"] = v.season.map(persistence(team, "long_rate"))
    v["yoy_r_short"] = v.season.map(persistence(team, "short_rate"))
    v["yoy_r_go_rate"] = v.season.map(persistence(go, "go_rate"))
    return v[["season", "weeks", "complete", "n_teams", "decisions", "median_n",
              "league_rate", "var_observed", "var_noise", "var_true", "sd_observed",
              "sd_noise", "sd_true", "share_real", "share_real_lo", "share_real_hi",
              "k_prior", "own_weight_at_median", "k_pooled_through", "pq",
              "yoy_r_long", "yoy_r_short", "yoy_r_go_rate"]]


# --------------------------------------------------------------- the feature

def prior_season_feature(team: pd.DataFrame, var_table: pd.DataFrame,
                         names: tuple[str, str]) -> pd.DataFrame:
    """team x season -> last season's long-band rate, shrunk and raw.

    Built on a dense team x season grid and shifted one season, so season s sees
    season s - 1 and nothing later. The shrunk value pulls the team toward that
    season's league rate with k pooled through that season:
        (x + k mu) / (n + k),
    which is the league rate for a team with no chances and for k = inf. The
    raw value is x / n and is missing when n = 0, left to the ridge's imputer.
    """
    shrunk_col, raw_col = names
    seasons = range(int(team.season.min()), int(team.season.max()) + 1)
    grid = pd.MultiIndex.from_product([sorted(team.team.unique()), seasons],
                                      names=["team", "season"])
    t = team.set_index(["team", "season"])[["long_n", "long_fg"]].reindex(grid)
    t = t.reset_index()
    league = var_table.set_index("season")
    mu = t.season.map(league.league_rate)
    k = t.season.map(league.k_pooled_through)
    n, x = t.long_n.fillna(0), t.long_fg.fillna(0)
    # A season the league table has never seen (mu missing) stays missing.
    shrunk = np.where(np.isinf(k), mu, (x + k * mu) / (n + k))
    t[shrunk_col] = np.where(mu.notna(), shrunk, np.nan)
    t[raw_col] = t.long_fg / t.long_n.replace(0, np.nan)
    for c in (shrunk_col, raw_col):
        t[c] = t.groupby("team")[c].shift(1)
    return t[["team", "season", shrunk_col, raw_col]]


# ------------------------------------------------------------------ the test

def within_week_rho(df: pd.DataFrame, col: str) -> float:
    """Mean within-week Spearman of y on `col`, as backtest.rho defines it."""
    vals = []
    for _, g in df.groupby(["season", "week"]):
        if len(g) < 5 or g[col].nunique() < 2 or g.y.nunique() < 2:
            continue
        r = stats.spearmanr(g.y, g[col]).statistic
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def season_forward(df: pd.DataFrame, base: list[str], arms: dict,
                   first_test: int = FIRST_TEST) -> tuple[pd.DataFrame, dict]:
    """Fit every arm on seasons < s, predict s. Returns predictions and coefs.

    `coefs[arm][s]` is the standardised ridge coefficient on the added column;
    the mechanism predicts it positive (more willingness to kick long, more
    kicker points).
    """
    tests = [s for s in sorted(df.season.unique()) if s >= first_test
             and df[df.season == s].week.nunique() >= MIN_WEEKS]
    out, coefs = [], {a: {} for a in arms}
    for s in tests:
        tr, te = df[df.season < s], df[df.season == s]
        block = pd.DataFrame({"season": te.season.to_numpy(), "week": te.week.to_numpy(),
                              "y": te.fantasy_points.to_numpy()})
        for arm, extra in arms.items():
            cols = base + extra
            m = RidgeRanker(features=cols).fit(tr)
            block[arm] = m.predict(te)
            if extra:
                coef = m.m_[-1].coef_
                assert len(coef) == len(cols), "the imputer dropped an all-missing column"
                coefs[arm][int(s)] = float(coef[cols.index(extra[0])])
        out.append(block)
    return pd.concat(out, ignore_index=True), coefs


def lift_table(P: pd.DataFrame, arms: dict, coefs: dict, primary: str | None = None,
               boot: int = N_BOOT) -> pd.DataFrame:
    """Pooled and per-season lift for every arm, and the paired tests vs BASE."""
    seasons = sorted(int(s) for s in P.season.unique())
    base_rho = within_week_rho(P, BASE)
    rows = []
    for arm in arms:
        per = {}
        for s in seasons:
            sub = P[P.season == s]
            l2 = decision(sub, arm, 2)["lift"]
            per[s] = l2 - decision(sub, BASE, 2)["lift"]
            rows.append({"arm": arm, "season": str(s), "weeks": sub.week.nunique(),
                         "lift_r2": l2, "lift_r3": decision(sub, arm, 3)["lift"],
                         "rho": within_week_rho(sub, arm), "diff_r2_vs_base": per[s],
                         "coef_std": coefs[arm].get(s, np.nan)})
        r = {"arm": arm, "season": f"{seasons[0]}-{seasons[-1]}",
             "weeks": P.groupby(["season", "week"]).ngroups,
             "lift_r2": decision(P, arm, 2)["lift"], "lift_r3": decision(P, arm, 3)["lift"],
             "rho": within_week_rho(P, arm),
             "coef_std": float(np.mean(list(coefs[arm].values()))) if coefs[arm] else np.nan}
        if arm != BASE:
            b = boot_lift(P, arm, BASE, roster=2, n=boot)
            d = np.array([per[s] for s in seasons])
            r.update({"diff_r2_vs_base": b["diff"], "diff_lo": b["lo"], "diff_hi": b["hi"],
                      "p_week_boot": b["p"], "rho_diff_vs_base": r["rho"] - base_rho,
                      "season_mean_diff": float(d.mean()),
                      "season_t_p": float(stats.ttest_1samp(d, 0.0).pvalue),
                      "season_wilcoxon_p": float(stats.wilcoxon(d).pvalue),
                      "seasons_won": int((d > 0).sum()), "n_seasons": len(d),
                      "coef_positive_folds": int(sum(c > 0 for c in coefs[arm].values()))})
        rows.append(r)
    cols = ["arm", "primary", "season", "weeks", "lift_r2", "lift_r3", "rho",
            "diff_r2_vs_base", "diff_lo", "diff_hi", "p_week_boot", "rho_diff_vs_base",
            "season_mean_diff", "season_t_p", "season_wilcoxon_p", "seasons_won",
            "n_seasons", "coef_std", "coef_positive_folds"]
    out = pd.DataFrame(rows)
    out["primary"] = out.arm.eq(primary)
    out = out.reindex(columns=cols)
    # Counts stay integers in the CSV; they are blank on rows that have no test.
    for c in ("seasons_won", "n_seasons", "coef_positive_folds"):
        out[c] = out[c].astype("Int64")
    return out


# ---------------------------------------------------------------------- run

def missing_seasons(seasons) -> list[int]:
    """Seasons from FIRST_SEASON to the latest one cached that have no file.

    Counted from FIRST_SEASON, not from the earliest file present: a cache
    holding only the recent seasons would otherwise pass and every rate, k and
    lift would quietly rest on a shorter record.
    """
    have = {int(s) for s in seasons if s >= FIRST_SEASON}
    return sorted(set(range(FIRST_SEASON, max(have) + 1)) - have) if have else []


def load_pbp(cache: pathlib.Path) -> pd.DataFrame:
    paths = {int(p.stem.split("_")[1]): p for p in cache.glob("pbp_*.parquet")}
    seasons = sorted(s for s in paths if s >= FIRST_SEASON)
    if not seasons:
        raise SystemExit(f"no play-by-play in {cache}; run build_dataset.py and fourth_down.py")
    gap = missing_seasons(seasons)
    if gap:
        raise SystemExit(f"play-by-play missing for {gap} in {cache}; "
                         f"every season from {FIRST_SEASON} on is needed")
    return pd.concat([pd.read_parquet(paths[s], columns=PBP_COLS) for s in seasons],
                     ignore_index=True)


def report_band(offset: int, league: pd.DataFrame, team: pd.DataFrame,
                v: pd.DataFrame, weeks: pd.Series, season: int) -> None:
    (l0, l1), (s0, s1) = bands(offset)
    print(f"\n=== distance = yardline_100 + {offset}: long yardline {l0}-{l1}, "
          f"short yardline {s0}-{s1} ===")
    print("FG attempt rate on fourth down, by band (regular season):")
    print(f"  {'season':>6} {'weeks':>5}  {'long':>6} {'n':>4}  {'short':>6} {'n':>4}")
    for _, r in league.iterrows():
        print(f"  {int(r.season):>6} {int(weeks.get(r.season, 0)):>5}  {r.long_rate:>6.3f} "
              f"{int(r.long_n):>4}  {r.short_rate:>6.3f} {int(r.short_n):>4}")
    h = v.set_index("season").loc[season]
    ts = team[team.season == season]
    lo = ts.sort_values(["long_rate", "long_n"], ascending=[True, False]).iloc[0]
    hi = ts.sort_values(["long_rate", "long_n"], ascending=[False, False]).iloc[0]
    print(f"{season}: median {h.median_n:.0f} long-band chances per team, lowest "
          f"{lo.team} {int(lo.long_fg)}/{int(lo.long_n)}, highest "
          f"{hi.team} {int(hi.long_fg)}/{int(hi.long_n)}")
    print(f"  observed across teams   var {h.var_observed:.5f}  sd {h.sd_observed:.3f}")
    print(f"  binomial sampling noise var {h.var_noise:.5f}  sd {h.sd_noise:.3f}")
    print(f"  true team difference    var {h.var_true:.5f}  sd {h.sd_true:.3f}")
    print(f"  share real {h.share_real:.1%}  (teams resampled, 95% "
          f"[{h.share_real_lo:.1%}, {h.share_real_hi:.1%}])")
    print(f"  shrinkage constant k {h.k_prior:.1f} from {season} alone, "
          f"{h.k_pooled_through:.1f} pooled {int(v.season.min())}-{season}; a team with "
          f"{h.median_n:.0f} chances keeps {h.own_weight_at_median:.0%} of its own rate "
          f"({h.median_n / (h.median_n + h.k_pooled_through):.0%} with the pooled k)")
    full = v[v.complete]
    print(f"  {len(full)} complete seasons: share real median {full.share_real.median():.1%} "
          f"(range {full.share_real.min():.1%} to {full.share_real.max():.1%}), "
          f"{int((full.var_true <= 0).sum())} with true variance at or below zero")
    pairs = v[v.complete].dropna(subset=["yoy_r_long"])
    print(f"  year-over-year r of raw team rates, {len(pairs)} pairs of complete seasons: "
          f"long {pairs.yoy_r_long.mean():+.2f} ({int((pairs.yoy_r_long < 0).sum())} negative), "
          f"short {pairs.yoy_r_short.mean():+.2f}, fourth_down go_rate "
          f"{pairs.yoy_r_go_rate.mean():+.2f}")


def main(season: int = HEADLINE_SEASON, boot: int = N_BOOT) -> int:
    from kicker_analysis.model import load
    pbp = load_pbp(CACHE)
    weeks = pbp[pbp.season_type == "REG"].groupby("season").week.nunique()
    mode, share = measured_offset(pbp)
    print(f"regular-season FG attempts {int(weeks.index.min())}-{int(weeks.index.max())}, "
          f"kick_distance - yardline_100: "
          + ", ".join(f"+{o} {s:.1%}" for o, s in share.head(3).items()))
    offsets = [DOC_OFFSET] + ([mode] if mode != DOC_OFFSET else [])

    go = fourth_down_plays(pbp[pbp.season_type == "REG"])
    go = (go.groupby(["season", "posteam"]).agg(n=("is_go", "size"), g=("is_go", "sum"))
          .reset_index().rename(columns={"posteam": "team"}))
    go["go_rate"] = go.g / go.n

    df = load(processed=PROC)
    arms = {BASE: []}
    rate_rows, var_rows = [], []
    for off in offsets:
        plays = band_plays(pbp, off)
        league, team = rates(plays, ["season"]), rates(plays, ["season", "team"])
        v = variance_table(team, go, weeks, boot)
        report_band(off, league, team, v, weeks, season)
        (l0, l1), (s0, s1) = bands(off)
        tag = {"fg_offset": off, "long_band": f"{l0}-{l1}", "short_band": f"{s0}-{s1}"}
        rate_rows.append(pd.concat([league.assign(level="league", team="ALL"),
                                    team.assign(level="team")], ignore_index=True).assign(**tag))
        var_rows.append(v.assign(**tag))
        names = feature_names(off)
        df = df.merge(prior_season_feature(team, v, names), on=["team", "season"],
                      how="left", validate="many_to_one")
        arms[f"{BASE} + shrunk long rate, {label(off)}"] = [names[0]]
        arms[f"{BASE} + raw long rate, {label(off)}"] = [names[1]]
    primary = f"{BASE} + shrunk long rate, {label(DOC_OFFSET)}"

    P, coefs = season_forward(df, list(FEATURES), arms)
    L = lift_table(P, arms, coefs, primary=primary, boot=boot)
    pooled = L[L.season.str.contains("-")].set_index("arm")
    print(f"\nDecision lift, R=2, test seasons {pooled.season.iloc[0]} "
          f"({int(pooled.weeks.iloc[0])} weeks), ridge alpha 100 in every arm:")
    for arm, r in pooled.iterrows():
        print(f"  {arm:45} lift {r.lift_r2:.4f}  rho {r.rho:.4f}"
              + ("  (primary)" if r.primary else ""))
        if arm != BASE:
            print(f"  {'':45} diff {r.diff_r2_vs_base:+.4f} [{r.diff_lo:+.4f}, "
                  f"{r.diff_hi:+.4f}] p {r.p_week_boot:.2f}, rho {r.rho_diff_vs_base:+.4f}; "
                  f"seasons won {int(r.seasons_won)}/{int(r.n_seasons)}, t p "
                  f"{r.season_t_p:.2f}; coef > 0 in {int(r.coef_positive_folds)}/"
                  f"{int(r.n_seasons)} folds")
    tracked = REPORTS / "backtest_summary.csv"
    if tracked.exists():
        bs = pd.read_csv(tracked)
        ref = bs.loc[bs["rank by"].str.startswith("ridge"), "lift_r2"]
        if len(ref):
            ref, here = float(ref.iloc[0]), float(pooled.loc[BASE, "lift_r2"])
            print(f"  reports/backtest_summary.csv ridge lift_r2 {ref:.4f}, base arm here "
                  f"{here:.4f}: " + ("the same model on the same split" if abs(ref - here) < 5e-5
                                    else "they differ, so data or code changed since that "
                                         "file was written"))

    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = pd.concat(rate_rows, ignore_index=True)
    rows["weeks"] = rows.season.map(weeks).astype(int)
    lead = ["fg_offset", "long_band", "short_band", "level", "season", "team", "weeks"]
    rows = rows[lead + [c for c in rows.columns if c.startswith(("long_", "short_"))
                        and c not in lead]]
    rows.to_csv(REPORTS / "fifty_plus_rates.csv", index=False)
    var = pd.concat(var_rows, ignore_index=True)
    var = var[lead[:3] + [c for c in var.columns if c not in lead[:3]]]
    var.to_csv(REPORTS / "fifty_plus_variance.csv", index=False)
    L.to_csv(REPORTS / "fifty_plus_lift.csv", index=False)
    print("\nwrote reports/fifty_plus_rates.csv, fifty_plus_variance.csv, fifty_plus_lift.csv")
    return 0


# --------------------------------------------------------------- self-check

def self_check() -> int:
    """Bands, filters, the decomposition, the lag and the test, on synthetic data."""
    assert bands(17) == ((33, 43), (20, 32)), bands(17)
    assert bands(18) == ((32, 42), (19, 31)), bands(18)
    base = dict(season=2024, week=1, season_type="REG", game_id="g", posteam="KC",
                down=4, ydstogo=5, yardline_100=35, wp=0.5,
                half_seconds_remaining=600, field_goal_attempt=0, punt_attempt=0,
                rush_attempt=0, pass_attempt=0, kick_distance=np.nan)
    rows = [
        {**base, "field_goal_attempt": 1, "yardline_100": 33},   # long at +17 and +18
        {**base, "punt_attempt": 1, "yardline_100": 43},         # long at +17, out at +18
        {**base, "rush_attempt": 1, "yardline_100": 38},         # long, go
        {**base, "field_goal_attempt": 1, "yardline_100": 32},   # short at +17, long at +18
        {**base, "field_goal_attempt": 1, "yardline_100": 20},   # short
        {**base, "field_goal_attempt": 1, "yardline_100": 19},   # out at +17, short at +18
        {**base, "punt_attempt": 1, "yardline_100": 44},         # out, beyond a 60-yarder
        {**base, "field_goal_attempt": 1, "season_type": "POST"},  # dropped, playoffs
        {**base, "field_goal_attempt": 1, "wp": 0.97},           # dropped, decided game
        {**base, "field_goal_attempt": 1, "half_seconds_remaining": 30},  # dropped, clock
        {**base, "field_goal_attempt": 1, "down": 3},            # dropped, not 4th
        {**base},                                                # dropped, no option
        {**base, "field_goal_attempt": 1, "punt_attempt": 1},    # dropped, ambiguous
        {**base, "field_goal_attempt": 1, "posteam": "OAK"},     # long, relocated to LV
    ]
    pbp = pd.DataFrame(rows)
    p = band_plays(pbp, 17)
    assert len(p) == 6, len(p)
    assert (p.band == "long").sum() == 4 and (p.band == "short").sum() == 2
    assert set(p.team) == {"KC", "LV"}
    r = rates(p, ["season", "team"]).set_index("team")
    # The punt and the go stay in the long denominator: 1 FG of 3 decisions.
    assert r.loc["KC", "long_n"] == 3 and r.loc["KC", "long_fg"] == 1
    assert abs(r.loc["KC", "long_rate"] - 1 / 3) < 1e-12
    assert r.loc["KC", "short_rate"] == 1.0 and np.isnan(r.loc["LV", "short_rate"])
    p18 = band_plays(pbp, 18).set_index("yardline_100").band
    assert p18.loc[32] == "long" and p18.loc[19] == "short" and 43 not in p18.index
    # The offset is read from the data, not assumed.
    kd = pd.DataFrame({**{c: [base[c]] * 4 for c in base},
                       "field_goal_attempt": [1, 1, 1, 0],
                       "yardline_100": [30, 20, 25, 30],
                       "kick_distance": [48, 38, 42, np.nan]})
    assert measured_offset(kd)[0] == 18
    # The record must start at FIRST_SEASON: a cache of recent seasons only is
    # refused, a gap inside it is refused, seasons before the floor are ignored.
    assert missing_seasons(range(FIRST_SEASON, 2027)) == []
    assert missing_seasons([2021, 2022, 2025]) == [*range(FIRST_SEASON, 2021), 2023, 2024]
    assert missing_seasons([2010, FIRST_SEASON, FIRST_SEASON + 1]) == []

    # Decomposition. The algebra closes exactly, and it recovers a known truth.
    rng = np.random.default_rng(1)
    n = rng.integers(5, 16, size=4000)
    d = decompose(rng.binomial(n, 0.4), n)        # one true rate for everyone
    assert abs(d["var_noise"] + d["var_true"] - d["var_observed"]) < 1e-15
    assert abs(d["var_true"]) < 0.002 and abs(d["share_real"]) < 0.1, d
    a, b = 6.0, 9.0                               # theta ~ Beta(6, 9): k = a + b = 15
    x = rng.binomial(n, rng.beta(a, b, size=len(n)))
    d = decompose(x, n)
    true_var = a * b / ((a + b) ** 2 * (a + b + 1))
    assert abs(d["var_true"] - true_var) < 0.15 * true_var, (d["var_true"], true_var)
    assert 11 < d["k_prior"] < 20, d["k_prior"]
    lo, hi = share_interval(x[:200], n[:200], boot=400)
    assert lo < hi
    # Every team on the league rate: no true spread, so infinite shrinkage.
    flat = decompose(np.array([2, 2, 2, 2]), np.array([5, 5, 5, 5]))
    assert flat["var_true"] < 0 and np.isinf(flat["k_prior"])

    # The lag. Season s may read season s - 1 and nothing else.
    team = pd.DataFrame({"season": [2020, 2020, 2021, 2021, 2023, 2023],
                         "team": ["AA", "BB"] * 3,
                         "long_n": [10, 10, 10, 0, 10, 10],
                         "long_fg": [8, 2, 5, 0, 5, 5]})
    vt = pd.DataFrame({"season": [2020, 2021, 2022, 2023],
                       "league_rate": [0.5, 0.5, np.nan, 0.5],
                       "var_true": [0.02, 0.02, np.nan, 0.02],
                       "pq": [0.25, 0.25, np.nan, 0.25]})
    vt["k_pooled_through"] = [pooled_k(vt, s) for s in vt.season]
    assert abs(vt.k_pooled_through.iloc[0] - (0.25 - 0.02) / 0.02) < 1e-12
    S, R = "s", "r"
    f = prior_season_feature(team, vt, (S, R)).set_index(["team", "season"])
    k = vt.k_pooled_through.iloc[0]
    assert np.isnan(f.loc[("AA", 2020), S]), "the first season has no prior"
    assert abs(f.loc[("AA", 2021), S] - (8 + k * 0.5) / (10 + k)) < 1e-12
    assert abs(f.loc[("AA", 2021), R] - 0.8) < 1e-12
    # A team with no chances last season gets the league rate, and no raw rate.
    assert abs(f.loc[("BB", 2022), S] - 0.5) < 1e-12 and np.isnan(f.loc[("BB", 2022), R])
    # 2022 is absent: 2023 must see a gap, not 2021 handed forward.
    assert np.isnan(f.loc[("AA", 2023), S]) and np.isnan(f.loc[("AA", 2023), R])
    # Changing a season's own counts cannot move that season's feature.
    team2 = team.copy()
    team2.loc[team2.season == 2021, "long_fg"] = [0, 0]
    f2 = prior_season_feature(team2, vt, (S, R)).set_index(["team", "season"])
    assert f2.loc[("AA", 2021), S] == f.loc[("AA", 2021), S]
    # pooled_k reads nothing after `through`.
    vt2 = vt.copy()
    vt2.loc[vt2.season == 2023, "var_true"] = 0.2
    assert pooled_k(vt2, 2021) == pooled_k(vt, 2021)

    # Year-over-year r on a toy panel where the order holds.
    pt = pd.DataFrame({"team": ["AA", "BB", "CC"] * 2, "season": [1] * 3 + [2] * 3,
                       "x": [0.1, 0.5, 0.9, 0.2, 0.4, 0.8]})
    assert persistence(pt, "x").loc[2] > 0.95

    # The test end to end: an informative added column must raise the lift, and
    # the pooled difference is exactly plus-arm lift minus base-arm lift.
    rng = np.random.default_rng(2)
    m = pd.DataFrame([(s, w) for s in range(2015, 2021) for w in range(1, 10)
                      for _ in range(12)], columns=["season", "week"])
    m["a"] = rng.normal(size=len(m))
    m["sig"] = rng.normal(size=len(m))
    m["fantasy_points"] = 8 + 0.5 * m.a + 1.5 * m.sig + rng.normal(scale=2, size=len(m))
    toy_arms = {BASE: [], "plus": ["sig"]}
    P, coefs = season_forward(m, ["a"], toy_arms, first_test=2018)
    assert sorted(P.season.unique()) == [2018, 2019, 2020]
    L = lift_table(P, toy_arms, coefs, primary="plus", boot=200)
    pooled = L[L.season == "2018-2020"].set_index("arm")
    assert pooled.loc["plus", "primary"] and not pooled.loc[BASE, "primary"]
    assert pooled.loc["plus", "diff_r2_vs_base"] > 0 and pooled.loc["plus", "seasons_won"] == 3
    assert abs(pooled.loc["plus", "diff_r2_vs_base"]
               - (pooled.loc["plus", "lift_r2"] - pooled.loc[BASE, "lift_r2"])) < 1e-12
    assert all(c > 0 for c in coefs["plus"].values())
    print("fifty_plus self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--season", type=int, default=HEADLINE_SEASON,
                    help="season whose decomposition is printed (the CSV has all)")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main(a.season, a.boot))
