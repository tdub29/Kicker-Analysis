"""Do the three candidate feature families help the ridge that orders the board?

The red-zone, drive and leg/stadium families were each first evaluated by adding
them to a 73-column LightGBM and scoring MAE (the *_eval.py scripts in this
folder). The board is no longer ordered by that model. It is ordered by
model.RidgeRanker on the 16 named model.FEATURES, and it is scored on the exact
pairwise decision lift, so an answer about the old model is not an answer about
this one. This script asks the question again of the model that ships, on the
metric the README reports.

Protocol, identical to backtest.py so the baseline row reproduces the ridge row
of reports/backtest_summary.csv:

  folds     season-forward. Every season from backtest.FIRST_TEST with at least
            backtest.MIN_WEEKS weeks is a test season, and the model is fit on
            every regular-season row before it.
  model     RidgeRanker(features=FEATURES + family columns), unchanged: median
            imputation, standardisation, alpha = 100. Nothing is retuned for the
            wider matrix, because a family that only helps after retuning has
            not shown it carries information.
  metrics   pooled MAE of the ridge prediction over every test row; within-week
            Spearman (backtest.rho, the rho column of backtest_summary.csv);
            exact R=2 decision lift (backtest.decision), the points per week
            gained by starting the ridge's favourite of two kickers drawn at
            random from the week.

Arms: the 16 features alone; plus each family (red-zone 27 columns, drives 21,
leg/stadium 13); plus all three (61). The family columns come from the
builders in src/kicker_analysis/feat_*.py, rebuilt from the play-by-play cache
on every run and attached with a left join that must neither add nor drop rows.

WHY A NOISE CONTROL. Adding columns to a fit changes its predictions even when
the columns carry nothing, and on a metric this noisy a change of either sign
can look like a finding. So each family is compared with the same 16 features
plus the SAME NUMBER of independent standard-normal columns, drawn
--draws times (50 by default) with fixed seeds. The draws give the spread a
family of that width moves the metrics by chance. A family "beats its noise
control" on a metric only if it is better than the noise draws, reported two
ways: the share of draws it beats, and the Monte Carlo p-value
(1 + draws at least as good) / (draws + 1), which cannot go below 1/51 with
50 draws. A family worse than most noise draws is worse than adding nothing
informative at all.

WHY A SECOND, SHUFFLED CONTROL. Independent standard-normal columns are
uncorrelated with each other, and a family's columns are not: in the red-zone
and drive families every quantity appears over three windows (recent games and
career), and the leg/stadium family has its own l8/l16 pairs. Correlated
columns give the ridge fewer independent directions to fit noise in, so an
uninformative family of correlated columns costs less than the same number of
independent ones, and beating the standard-normal control can come from that
structure alone. The second control keeps the family's own values, with their
correlations, scales and missing cells, and permutes its rows jointly within
each season (--draws times, fixed seeds), which breaks only their link to the
kicker-week. Permuting inside a season means no value crosses a fold boundary,
and every fold of one draw sees the same permutation, as with the noise. A
family carries information the ridge can use only if it beats this control as
well; the same share and Monte Carlo p-value are reported against it.

The difference from the 16-feature baseline also gets backtest.boot_lift, the
paired week-block bootstrap the README uses, so a reader can see whether any
change is distinguishable from zero regardless of the control.

Writes reports/feature_family_ridge_retest.csv, one row per arm. Real arms
carry kind = "real"; the matched controls carry kind = "noise" (standard-normal
columns) or kind = "shuffle" (the family's own rows permuted) and report the
mean over draws with its spread (sd; min and max of the lift).

    python experiments/feature_families/ridge_retest.py
    python experiments/feature_families/ridge_retest.py --self-check   # offline, writes nothing

Needs data/processed/kicker_games.parquet and fourth_down.parquet
(build_dataset.py, fourth_down.py) and the data/external/pbp_*.parquet cache
fourth_down.py fills.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# experiments/feature_families/<this file>, so the repo root is two levels up.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kicker_analysis import feat_drives, feat_legstadium, feat_redzone  # noqa: E402
from kicker_analysis.backtest import (FIRST_TEST, MIN_WEEKS, _pick_probs,  # noqa: E402
                                      boot_lift, decision, rho)
from kicker_analysis.features import check_roof  # noqa: E402
from kicker_analysis.model import FEATURES, RidgeRanker, load  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OUT = REPORTS / "feature_family_ridge_retest.csv"

# Family name -> the builder module. Each exposes build(), FEATURE_COLS and
# JOIN_KEYS; red-zone and drives join on the team-week, leg/stadium on the
# kicker-week.
FAMILIES = {"red-zone": feat_redzone, "drives": feat_drives,
            "leg/stadium": feat_legstadium}
ALL = "all three"
BASELINE = "baseline"
ROSTER = 2
N_DRAWS = 50
NOISE_SEED = 20180          # fixed so a rerun draws the same controls
SHUFFLE_SEED = 20181
NOISE_PREFIX = "noise_"

COLUMNS = ["arm", "kind", "n_added", "draws", "n_rows", "n_weeks", "coverage",
           "mae", "mae_sd", "rho", "rho_sd",
           "lift_r2", "lift_r2_sd", "lift_r2_min", "lift_r2_max",
           "d_lift_vs_baseline", "d_lift_lo", "d_lift_hi", "d_lift_p",
           "lift_share_noise_below", "lift_noise_p",
           "mae_share_noise_worse", "mae_noise_p",
           "rho_share_noise_below", "rho_noise_p",
           "lift_share_shuffle_below", "lift_shuffle_p",
           "mae_share_shuffle_worse", "mae_shuffle_p",
           "rho_share_shuffle_below", "rho_shuffle_p"]


# ------------------------------------------------------------------ the frame

def attach(df: pd.DataFrame, table: pd.DataFrame, keys: list[str],
           cols: list[str]) -> pd.DataFrame:
    """Left-join `cols` from `table` onto `df` by `keys`, refusing to reweight.

    A duplicated key in the table would fan rows out and silently weight some
    kicker-weeks twice; a column already in `df` would come back suffixed and
    the arm would fit on the wrong one. Both are errors, not warnings.
    """
    clash = set(cols) & set(df.columns)
    assert not clash, f"family columns already in the frame: {sorted(clash)}"
    assert not table.duplicated(keys).any(), f"duplicate join keys {keys}"
    out = df.merge(table[keys + cols], on=keys, how="left", validate="many_to_one")
    assert len(out) == len(df), (len(out), len(df))
    out.index = df.index
    return out


def family_tables() -> dict[str, tuple[pd.DataFrame, list[str], list[str]]]:
    """Rebuild each family's table from the cache, quietly.

    Rebuilt rather than read from data/processed/, so the arms can never be fit
    on a table left over from an older kicker_games.parquet. Each build takes a
    few seconds; their sanity printouts are suppressed here.
    """
    out = {}
    for name, mod in FAMILIES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            table = mod.build()
        out[name] = (table, list(mod.JOIN_KEYS), list(mod.FEATURE_COLS))
    return out


def add_noise(df: pd.DataFrame, width: int, draw: int,
              seed: int = NOISE_SEED) -> tuple[pd.DataFrame, list[str]]:
    """`width` independent N(0, 1) columns over every row, fixed by (width, draw).

    Drawn over the whole frame once, so every fold of one draw sees the same
    columns, exactly as a real family's columns are the same across folds.
    """
    rng = np.random.default_rng([seed, width, draw])
    cols = [f"{NOISE_PREFIX}{i:02d}" for i in range(width)]
    noise = pd.DataFrame(rng.standard_normal((len(df), width)), columns=cols,
                         index=df.index)
    return pd.concat([df, noise], axis=1), cols


def shuffle_family(df: pd.DataFrame, cols: list[str], draw: int,
                   seed: int = SHUFFLE_SEED) -> pd.DataFrame:
    """`cols` with their rows permuted jointly within each season, fixed by draw.

    One permutation per season moves whole rows of the family, so every value,
    the correlation between the family's columns and the pattern of missing
    cells are kept, and only the link to the kicker-week is broken. Nothing
    moves between seasons, so no value crosses a fold boundary. Every other
    column is untouched.
    """
    rng = np.random.default_rng([seed, len(cols), draw])
    vals = df[cols].to_numpy(float)
    out_vals = vals.copy()
    season = df.season.to_numpy()
    for s in np.unique(season):
        at = np.flatnonzero(season == s)
        out_vals[at] = vals[rng.permutation(at)]
    out = df.copy()
    out[cols] = out_vals
    return out


# ------------------------------------------------------------------- scoring

def test_seasons(df: pd.DataFrame) -> list[int]:
    """backtest.py's fold rule, so the baseline lines up with its ridge row."""
    return [int(s) for s in sorted(df.season.unique())
            if s >= FIRST_TEST and df[df.season == s].week.nunique() >= MIN_WEEKS]


def season_forward(df: pd.DataFrame, features: list[str],
                   seasons: list[int]) -> np.ndarray:
    """Ridge predictions for every row of the test seasons, in frame order."""
    preds = []
    for s in seasons:
        tr, te = df[df.season < s], df[df.season == s]
        preds.append(RidgeRanker(features=list(features)).fit(tr).predict(te))
    return np.concatenate(preds)


def test_frame(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """season, week and y for the test rows, in the order season_forward uses."""
    te = pd.concat([df[df.season == s] for s in seasons])
    return pd.DataFrame({"season": te.season.to_numpy(), "week": te.week.to_numpy(),
                         "y": te.fantasy_points.to_numpy(float)})


def score(P: pd.DataFrame, col: str) -> dict:
    return {"mae": float(np.abs(P.y - P[col]).mean()),
            "rho": rho(P, col),
            "lift_r2": decision(P, col, ROSTER)["lift"]}


def noise_p(real: float, draws: np.ndarray, higher_is_better: bool = True) -> float:
    """Monte Carlo p-value that the real arm beats a control's draws.

    (1 + draws at least as good as the real arm) / (draws + 1): the standard
    finite-sample form, never zero, so 50 draws can say p = 1/51 and no less.
    """
    draws = np.asarray(draws, float)
    as_good = (draws >= real) if higher_is_better else (draws <= real)
    return float((1 + as_good.sum()) / (len(draws) + 1))


# ---------------------------------------------------------------------- main

def main(n_draws: int = N_DRAWS) -> int:
    df = load(processed=PROC)
    check_roof(df)
    seasons = test_seasons(df)
    tables = family_tables()
    fam_cols = {}
    for name, (table, keys, cols) in tables.items():
        df = attach(df, table, keys, cols)
        fam_cols[name] = cols
    arms = {BASELINE: []} | fam_cols | {ALL: [c for cols in fam_cols.values() for c in cols]}
    assert len(set(arms[ALL])) == len(arms[ALL]), "families share a column name"

    P = test_frame(df, seasons)
    te_rows = pd.concat([df[df.season == s] for s in seasons])
    n_weeks = int(P.groupby(["season", "week"]).ngroups)
    print(f"{len(df):,} modelling rows, {len(P):,} test rows in {n_weeks} weeks, "
          f"test seasons {seasons[0]}-{seasons[-1]} ({len(seasons)} folds)")
    print(f"{len(FEATURES)} baseline features; families: "
          + ", ".join(f"{k} +{len(v)}" for k, v in arms.items() if k != BASELINE)
          + f"; {n_draws} noise draws per width\n")

    real = {}
    for name, extra in arms.items():
        P[name] = season_forward(df, FEATURES + extra, seasons)
        real[name] = score(P, name)
        cov = float(te_rows[extra].notna().mean().mean()) if extra else 1.0
        real[name]["coverage"] = cov

    # The matched-width controls. Widths are the real arms' widths, so the
    # all-three arm gets a 61-column control of its own.
    widths = sorted({len(v) for k, v in arms.items() if k != BASELINE})
    ctl: dict[int, pd.DataFrame] = {}
    for w in widths:
        rows = []
        for d in range(n_draws):
            dn, cols = add_noise(df, w, d)
            Q = P[["season", "week", "y"]].copy()
            Q["s"] = season_forward(dn, FEATURES + cols, seasons)
            rows.append(score(Q, "s"))
        ctl[w] = pd.DataFrame(rows)
        c = ctl[w]
        print(f"  noise x{w:<2}  lift {c.lift_r2.mean():.4f} (sd {c.lift_r2.std(ddof=1):.4f}, "
              f"{c.lift_r2.min():.4f} to {c.lift_r2.max():.4f})  "
              f"MAE {c.mae.mean():.4f}  rho {c.rho.mean():.4f}", flush=True)

    # The structure-matched controls: each real arm's own columns, rows
    # permuted within season.
    shf: dict[str, pd.DataFrame] = {}
    for name, extra in arms.items():
        if not extra:
            continue
        rows = []
        for d in range(n_draws):
            Q = P[["season", "week", "y"]].copy()
            Q["s"] = season_forward(shuffle_family(df, extra, d), FEATURES + extra, seasons)
            rows.append(score(Q, "s"))
        shf[name] = c = pd.DataFrame(rows)
        print(f"  shuffled {name:<12} lift {c.lift_r2.mean():.4f} (sd {c.lift_r2.std(ddof=1):.4f}, "
              f"{c.lift_r2.min():.4f} to {c.lift_r2.max():.4f})  "
              f"MAE {c.mae.mean():.4f}  rho {c.rho.mean():.4f}", flush=True)

    base = real[BASELINE]
    out = []
    for name, extra in arms.items():
        r = real[name]
        row = {"arm": name, "kind": "real", "n_added": len(extra), "draws": 1,
               "n_rows": len(P), "n_weeks": n_weeks, "coverage": r["coverage"],
               "mae": r["mae"], "rho": r["rho"], "lift_r2": r["lift_r2"],
               "d_lift_vs_baseline": r["lift_r2"] - base["lift_r2"]}
        if extra:
            b = boot_lift(P, name, BASELINE, ROSTER)
            row.update(d_lift_lo=b["lo"], d_lift_hi=b["hi"], d_lift_p=b["p"])
            c = ctl[len(extra)]
            row.update(
                lift_share_noise_below=float((c.lift_r2 < r["lift_r2"]).mean()),
                lift_noise_p=noise_p(r["lift_r2"], c.lift_r2),
                mae_share_noise_worse=float((c.mae > r["mae"]).mean()),
                mae_noise_p=noise_p(r["mae"], c.mae, higher_is_better=False),
                rho_share_noise_below=float((c.rho < r["rho"]).mean()),
                rho_noise_p=noise_p(r["rho"], c.rho))
            c = shf[name]
            row.update(
                lift_share_shuffle_below=float((c.lift_r2 < r["lift_r2"]).mean()),
                lift_shuffle_p=noise_p(r["lift_r2"], c.lift_r2),
                mae_share_shuffle_worse=float((c.mae > r["mae"]).mean()),
                mae_shuffle_p=noise_p(r["mae"], c.mae, higher_is_better=False),
                rho_share_shuffle_below=float((c.rho < r["rho"]).mean()),
                rho_shuffle_p=noise_p(r["rho"], c.rho))
        out.append(row)
    controls = ([(f"noise x{w}", "noise", w, ctl[w]) for w in widths]
                + [(f"shuffled {k}", "shuffle", len(arms[k]), c) for k, c in shf.items()])
    for arm, kind, w, c in controls:
        out.append({"arm": arm, "kind": kind, "n_added": w,
                    "draws": n_draws, "n_rows": len(P), "n_weeks": n_weeks,
                    "mae": c.mae.mean(), "mae_sd": c.mae.std(ddof=1),
                    "rho": c.rho.mean(), "rho_sd": c.rho.std(ddof=1),
                    "lift_r2": c.lift_r2.mean(), "lift_r2_sd": c.lift_r2.std(ddof=1),
                    "lift_r2_min": c.lift_r2.min(), "lift_r2_max": c.lift_r2.max(),
                    "d_lift_vs_baseline": c.lift_r2.mean() - base["lift_r2"]})
    res = pd.DataFrame(out).reindex(columns=COLUMNS)

    print("\nPer season R=2 lift, real arms:")
    per = {name: [decision(P[P.season == s], name, ROSTER)["lift"] for s in seasons]
           for name in arms}
    print(pd.DataFrame(per, index=seasons).round(4).to_string())

    show = ["arm", "n_added", "mae", "rho", "lift_r2", "d_lift_vs_baseline",
            "d_lift_lo", "d_lift_hi", "d_lift_p", "lift_share_noise_below", "lift_noise_p"]
    print("\nSummary (noise and shuffled rows are means over draws):")
    print(res[show].round(4).to_string(index=False))
    print("\nAgainst the matched noise control (share of draws the family beats):")
    for _, r in res[res.kind == "real"].iloc[1:].iterrows():
        print(f"  {r.arm:12s} lift {r.lift_share_noise_below:.2f} (p {r.lift_noise_p:.3f})  "
              f"MAE {r.mae_share_noise_worse:.2f} (p {r.mae_noise_p:.3f})  "
              f"rho {r.rho_share_noise_below:.2f} (p {r.rho_noise_p:.3f})")
    print("\nAgainst its own rows shuffled within season (share of draws the family beats):")
    for _, r in res[res.kind == "real"].iloc[1:].iterrows():
        print(f"  {r.arm:12s} lift {r.lift_share_shuffle_below:.2f} (p {r.lift_shuffle_p:.3f})  "
              f"MAE {r.mae_share_shuffle_worse:.2f} (p {r.mae_shuffle_p:.3f})  "
              f"rho {r.rho_share_shuffle_below:.2f} (p {r.rho_shuffle_p:.3f})")

    REPORTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


# ----------------------------------------------------------------- self-check

def self_check() -> int:
    """Join guard, both controls, the lift metric and the p-value, on toys."""
    # The families are the widths the documents quote, share no column with
    # each other or with the 16 features, and never use the noise prefix.
    widths = {k: len(m.FEATURE_COLS) for k, m in FAMILIES.items()}
    assert widths == {"red-zone": 27, "drives": 21, "leg/stadium": 13}, widths
    every = [c for m in FAMILIES.values() for c in m.FEATURE_COLS]
    assert len(set(every)) == 61 and not set(every) & set(FEATURES)
    assert not any(c.startswith(NOISE_PREFIX) for c in every + FEATURES)

    # attach: unmatched rows survive with NaN, a duplicated key is refused, and
    # a column clash is refused, so an arm can never fit on reweighted rows.
    df = pd.DataFrame({"season": [2020, 2020, 2021], "team": ["KC", "LV", "KC"],
                       "fantasy_points": [9.0, 4.0, 7.0]}, index=[10, 11, 12])
    t = pd.DataFrame({"season": [2020, 2021], "team": ["KC", "KC"], "f": [1.0, 2.0]})
    a = attach(df, t, ["season", "team"], ["f"])
    assert list(a.index) == [10, 11, 12] and a.f.isna().tolist() == [False, True, False]
    for bad, cols in ((pd.concat([t, t.iloc[:1]]), ["f"]),
                      (t.rename(columns={"f": "fantasy_points"}), ["fantasy_points"])):
        try:
            attach(df, bad, ["season", "team"], cols)
        except AssertionError:
            pass
        else:
            raise AssertionError("attach accepted a fan-out or a column clash")

    # Noise: matched width, identical for the same (width, draw), different
    # across draws, and the frame itself untouched.
    n1, c1 = add_noise(df, 3, 0)
    n2, _ = add_noise(df, 3, 0)
    n3, _ = add_noise(df, 3, 1)
    assert len(c1) == 3 and list(df.columns) == ["season", "team", "fantasy_points"]
    assert np.array_equal(n1[c1].to_numpy(), n2[c1].to_numpy())
    assert not np.allclose(n1[c1].to_numpy(), n3[c1].to_numpy())

    # Shuffle: whole rows of the family move together (b = 10a survives, and the
    # missing cell travels with its row), only within a season, every other
    # column and the input frame are untouched, and a draw is reproducible.
    sf = pd.DataFrame({"season": np.repeat([2020, 2021], 6), "fantasy_points": 1.0,
                       "a": np.arange(12.0), "b": np.arange(12.0) * 10})
    sf.loc[3, "b"] = np.nan
    before = sf.copy()
    s1, s2, s3 = (shuffle_family(sf, ["a", "b"], d) for d in (0, 0, 1))
    assert sf.equals(before) and s1.equals(s2) and not s1.equals(s3)
    assert s1[["season", "fantasy_points"]].equals(sf[["season", "fantasy_points"]])
    assert not np.array_equal(s1.a, sf.a)
    ok = s1.b.notna()
    assert np.array_equal(s1.b[ok], 10 * s1.a[ok]) and s1.a[~ok].tolist() == [3.0]
    for season, g in s1.groupby("season"):
        assert sorted(g.a) == sorted(sf.a[sf.season == season])

    # The imported lift is the exact pairwise one: average over every pair in
    # the week of y(the pair's higher score, ties split) minus the week mean.
    y = np.array([3.0, 12.0, 7.0, 7.0, 0.0])
    s = np.array([1.0, 4.0, 2.0, 2.0, 0.5])
    brute = []
    for i, j in itertools.combinations(range(len(y)), 2):
        pick = y[i] if s[i] > s[j] else y[j] if s[j] > s[i] else (y[i] + y[j]) / 2
        brute.append(pick - y.mean())
    wk = pd.DataFrame({"season": 2020, "week": 1, "y": y, "s": s})
    assert abs(decision(wk, "s", 2)["lift"] - np.mean(brute)) < 1e-12
    assert abs(_pick_probs(s, 2).sum() - 1.0) < 1e-12

    # The Monte Carlo p-value, by hand: two of three draws are at least as good
    # as 0.5 when lower is better, one when higher is better.
    assert noise_p(0.5, [0.1, 0.2, 0.6]) == 0.5
    assert noise_p(0.5, [0.1, 0.2, 0.6], higher_is_better=False) == 0.75
    assert noise_p(9.0, np.zeros(50)) == 1 / 51

    # End to end on a toy league: one informative column against pure noise and
    # against itself shuffled. The fold rule must pick only full test seasons,
    # and the informative arm must order the week better than either control.
    rng = np.random.default_rng(0)
    toy = []
    for season in (FIRST_TEST - 2, FIRST_TEST - 1, FIRST_TEST, FIRST_TEST + 1):
        weeks = MIN_WEEKS if season != FIRST_TEST + 1 else MIN_WEEKS - 1
        for week in range(1, weeks + 1):
            x = rng.standard_normal(12)
            toy.append(pd.DataFrame({"season": season, "week": week, "x": x,
                                     "fantasy_points": 8 + 2 * x + rng.standard_normal(12)}))
    toy = pd.concat(toy, ignore_index=True)
    seasons = test_seasons(toy)
    assert seasons == [FIRST_TEST], seasons      # the short season is not a fold
    T = test_frame(toy, seasons)
    T["x"] = season_forward(toy, ["x"], seasons)
    tn, cols = add_noise(toy, 1, 0)
    T["n"] = season_forward(tn, cols, seasons)
    T["xs"] = season_forward(shuffle_family(toy, ["x"], 0), ["x"], seasons)
    assert len(T) == 12 * MIN_WEEKS
    sx, sn, ss = score(T, "x"), score(T, "n"), score(T, "xs")
    assert sx["lift_r2"] > sn["lift_r2"] and sx["rho"] > sn["rho"], (sx, sn)
    assert sx["mae"] < sn["mae"], (sx, sn)
    assert sx["lift_r2"] > ss["lift_r2"] and sx["mae"] < ss["mae"], (sx, ss)
    print("ridge_retest self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--draws", type=int, default=N_DRAWS,
                    help="noise-control draws per width (default %(default)s)")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main(a.draws))
