"""Throwaway: does feat_drives buy anything over the existing feature set?

Season-forward CV, test seasons 2019-2025, train on everything strictly before.
One LightGBM spec, fixed, used for both arms. Nothing is tuned; the only thing
that changes between the two arms is whether the 21 dv_* columns are in X.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kicker_analysis.features import build_features, feature_columns  # noqa: E402
from kicker_analysis import feat_drives  # noqa: E402

PROC = ROOT / "data" / "processed"
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=40, verbose=-1, random_state=0,
              # n_jobs is a runtime knob, not a hyperparameter, and it is the
              # same in both arms. Other agents are running in this repo and
              # LightGBM's default all-cores setting thrashed against them.
              n_jobs=2)


def main() -> int:
    games = pd.read_parquet(PROC / "kicker_games.parquet")
    fourth = pd.read_parquet(PROC / "fourth_down.parquet")
    df = games.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    df = df[(df.season >= 2015) & (df.season_type == "REG")].copy()
    df = build_features(df)
    base = feature_columns(df)

    dv = pd.read_parquet(PROC / "drives.parquet")
    n_before = len(df)
    df = df.merge(dv, on=feat_drives.JOIN_KEYS, how="left")
    assert len(df) == n_before, (len(df), n_before)
    new = [c for c in feat_drives.FEATURE_COLS if c in df.columns]
    base = [c for c in base if c not in new]
    cov = df[new].notna().mean().mean()
    print(f"{len(df):,} kicker-games, {len(base)} base features, "
          f"{len(new)} new; new-feature non-null coverage {cov:.3f}")

    rows = []
    for s in range(2019, 2026):
        tr, te = df[df.season < s], df[df.season == s]
        if len(te) < 50:
            continue
        r = {"season": s, "n": len(te)}
        for arm, cols in (("without", base), ("with", base + new)):
            m = LGBMRegressor(**PARAMS).fit(tr[cols], tr.fantasy_points)
            p = np.clip(m.predict(te[cols]), 0, None)
            r[f"mae_{arm}"] = mean_absolute_error(te.fantasy_points, p)
            r[f"rho_{arm}"] = spearmanr(te.fantasy_points, p).statistic
        rows.append(r)
    res = pd.DataFrame(rows)
    res["mae_delta"] = res.mae_with - res.mae_without      # negative = better
    res["rho_delta"] = res.rho_with - res.rho_without      # positive = better
    print("\nPer season (mae_delta < 0 and rho_delta > 0 mean the new features help)")
    print(res.round(4).to_string(index=False))

    w = res.n
    pooled = {k: np.average(res[k], weights=w) for k in
              ("mae_without", "mae_with", "rho_without", "rho_with")}
    print(f"\nPooled, weighted by games ({int(w.sum()):,} kicker-games)")
    print(f"  MAE      without {pooled['mae_without']:.4f}   with {pooled['mae_with']:.4f}"
          f"   delta {pooled['mae_with'] - pooled['mae_without']:+.4f}")
    print(f"  Spearman without {pooled['rho_without']:.4f}   with {pooled['rho_with']:.4f}"
          f"   delta {pooled['rho_with'] - pooled['rho_without']:+.4f}")

    print(f"\nPaired over the {len(res)} folds")
    for name, a, b in (("MAE", res.mae_with, res.mae_without),
                       ("Spearman", res.rho_with, res.rho_without)):
        t = ttest_rel(a, b)
        wx = wilcoxon(a, b)
        d = (a - b)
        print(f"  {name:9s} mean delta {d.mean():+.4f} (sd {d.std(ddof=1):.4f}), "
              f"wins {int((d < 0).sum() if name == 'MAE' else (d > 0).sum())}/{len(d)}, "
              f"t={t.statistic:+.3f} p={t.pvalue:.3f}, wilcoxon W={wx.statistic:.1f} "
              f"p={wx.pvalue:.3f}")

    # Where the new columns rank when they are allowed in, on the last fold.
    tr, te = df[df.season < 2025], df[df.season == 2025]
    m = LGBMRegressor(**PARAMS).fit(tr[base + new], tr.fantasy_points)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=base + new)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    rank = {c: int(np.where(gain.index == c)[0][0]) + 1 for c in new}
    print(f"\nGain share of all {len(new)} dv_* columns combined: "
          f"{gain[new].sum():.3f} of 1.000")
    print("Best-ranked new columns (rank of " + str(len(gain)) + "):")
    for c in sorted(new, key=lambda c: rank[c])[:6]:
        print(f"  #{rank[c]:3d}  {c:34s} gain {gain[c]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
