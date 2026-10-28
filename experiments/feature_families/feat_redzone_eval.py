"""Does red-zone efficiency add anything to the kicker model? A one-off
measurement harness, not part of the pipeline.

Season-forward: for test season S in 2019..2025, train on every season < S.
One LightGBM spec, fixed, never tuned, run over four feature sets:

    base       whatever features.feature_columns() already gives
    +composite base plus rz_exp_fg_* only (trips x (1 - td rate))
    +raw pair  base plus rz_trips_pg_* and rz_td_rate_* (the two it is built from)
    +all rz    base plus every column in feat_redzone.FEATURE_COLS

Paired t-test, Wilcoxon and a sign count over the per-season MAE deltas. The
fold count is small and is printed rather than glossed.

This predates the 16-feature ridge that now orders the board: `base` is every
numeric column features.feature_columns() admitted at the time (73 of them),
and the score is MAE, not the decision lift. The ridge re-test of the same
family is ridge_retest.py in this folder.

When the logged run was made (feat_redzone_eval.log, results in
reports/feat_redzone_results.csv), 2021 was MISSING from kicker_games.parquet:
the nflverse kicking release has no 2021 rows. That gave 6 usable folds, not 7,
and hit both arms of every comparison identically, so the pairing was
unaffected. build_dataset.py now fills 2021 from play-by-play, so a rerun has 7
folds and will not reproduce the logged numbers exactly.

    python experiments/feature_families/feat_redzone_eval.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error

# experiments/feature_families/<this file>, so the repo root is two levels up.
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kicker_analysis import feat_redzone                       # noqa: E402
from kicker_analysis.features import build_features, feature_columns  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TEST_SEASONS = range(2019, 2026)
FRAME = REPORTS / "_feat_redzone_frame.parquet"
BASECOLS = REPORTS / "_feat_redzone_base.json"
PARTIAL = REPORTS / "_feat_redzone_partial.csv"


def fit(Xtr, ytr, Xte):
    # n_jobs is a runtime setting, not a model choice, and it is the same in
    # every arm. One thread keeps LightGBM's OpenMP pool from oversubscribing a
    # busy machine and fixes the thread count from one run to the next.
    m = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                      min_child_samples=40, verbose=-1, random_state=0, n_jobs=1)
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xte), 0, None)


def frame() -> tuple[pd.DataFrame, list[str]]:
    """Base features plus the red-zone merge, cached so a restart is cheap."""
    if FRAME.exists() and BASECOLS.exists():
        return pd.read_parquet(FRAME), json.loads(BASECOLS.read_text())

    games = pd.read_parquet(PROC / "kicker_games.parquet")
    fourth = pd.read_parquet(PROC / "fourth_down.parquet")
    df = games.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    df = df[(df.season >= 2015) & (df.season_type == "REG")].copy()
    df = build_features(df)
    base = feature_columns(df)          # computed BEFORE the merge, so the
                                        # baseline cannot pick up red-zone columns

    rz = pd.read_parquet(PROC / "redzone.parquet")
    n = len(df)
    df = df.merge(rz, on=feat_redzone.JOIN_KEYS, how="left")
    assert len(df) == n, (len(df), n)   # a fan-out here would silently reweight
    keep = base + feat_redzone.FEATURE_COLS + ["season", "week", "fantasy_points"]
    df = df[keep]
    df.to_parquet(FRAME, index=False)
    BASECOLS.write_text(json.dumps(base))
    return df, base


def main() -> int:
    df, base = frame()
    cov = df[feat_redzone.FEATURE_COLS[0]].notna().mean()
    print(f"{len(df):,} kicker-games, {len(base)} base features, "
          f"{len(feat_redzone.FEATURE_COLS)} red-zone features, "
          f"coverage {cov:.3f}\n", flush=True)

    sets = {
        "base": base,
        "+composite": base + feat_redzone.COMPOSITE_COLS,
        "+raw pair": base + feat_redzone.RAW_PAIR_COLS,
        "+all rz": base + feat_redzone.FEATURE_COLS,
    }

    # Folds are checkpointed, so an interrupted sweep resumes where it stopped
    # instead of refitting the folds that already finished.
    done = pd.read_csv(PARTIAL) if PARTIAL.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for s in TEST_SEASONS:
        if len(done) and (done.season == s).any():
            print(f"  fold {s} from checkpoint", flush=True)
            continue
        tr, te = df[df.season < s], df[df.season == s]
        if len(te) < 50:                      # a season absent from the data
                                              # (2021, when the log was made)
            print(f"  fold {s} skipped: {len(te)} games in the data", flush=True)
            continue
        print(f"  fold {s}: train {len(tr)}, test {len(te)}", flush=True)
        for name, cols in sets.items():
            p = fit(tr[cols], tr.fantasy_points, te[cols])
            rows.append({"season": s, "set": name, "n": len(te),
                         "mae": mean_absolute_error(te.fantasy_points, p),
                         "spearman": spearmanr(te.fantasy_points, p).statistic})
        pd.DataFrame(rows).to_csv(PARTIAL, index=False)
    res = pd.DataFrame(rows)

    print("\nPer season MAE:")
    print(res.pivot(index="season", columns="set", values="mae")[list(sets)]
          .round(4).to_string())
    print("\nPer season Spearman:")
    print(res.pivot(index="season", columns="set", values="spearman")[list(sets)]
          .round(4).to_string())

    print("\nPooled (weighted by games):")
    agg = (res.groupby("set").apply(
        lambda g: pd.Series({"mae": np.average(g.mae, weights=g.n),
                             "spearman": np.average(g.spearman, weights=g.n),
                             "games": int(g.n.sum())}), include_groups=False)
           .loc[list(sets)])
    print(agg.round(4).to_string())

    b = res[res.set == "base"].set_index("season").sort_index()
    k = len(b)
    print(f"\nPaired over the {k} folds, against base "
          "(negative dMAE = the red-zone features help):")
    for name in list(sets)[1:]:
        v = res[res.set == name].set_index("season").sort_index()
        d = (v.mae - b.mae).to_numpy()
        t, pt = ttest_rel(v.mae, b.mae)
        try:
            _, pw = wilcoxon(v.mae, b.mae)
        except ValueError:
            pw = float("nan")
        ds = (v.spearman - b.spearman).to_numpy()
        print(f"  {name:11s} dMAE {d.mean():+.4f}  better {int((d < 0).sum())}/{k}"
              f"  t={t:+.3f} p={pt:.3f}  wilcoxon p={pw:.3f}")
        print(f"  {'':11s} dSpearman {ds.mean():+.4f}  "
              f"better {int((ds > 0).sum())}/{k}")

    out = REPORTS / "feat_redzone_results.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
