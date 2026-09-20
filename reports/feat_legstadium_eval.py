"""Does feat_legstadium earn its place? Season-forward, with and without.

Throwaway measurement script, not part of the pipeline. Same protocol both
ways: identical rows, identical folds, identical LightGBM settings, the only
difference being whether FEATURE_COLS is appended to the design matrix. Nothing
is tuned; the hyperparameters are the ones the assignment fixed.
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
from kicker_analysis import feat_legstadium as LS                     # noqa: E402

PROC = ROOT / "data" / "processed"
# n_jobs is pinned only because this box is running many agents at once and
# oversubscribed OpenMP threads make the fits take minutes instead of seconds.
# It does not change the fitted model.
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=40, verbose=-1, random_state=0, n_jobs=2)
TEST_SEASONS = range(2019, 2026)


def main() -> int:
    kg = pd.read_parquet(PROC / "kicker_games.parquet")
    fourth = pd.read_parquet(PROC / "fourth_down.parquet")
    df = kg.merge(fourth, on=["season", "week", "season_type", "team"], how="left")
    df = df[(df.season >= 2015) & (df.season_type == "REG")].copy()
    df = build_features(df)

    ls = pd.read_parquet(PROC / "legstadium.parquet")
    n_before = len(df)
    df = df.merge(ls, on=LS.JOIN_KEYS, how="left")
    assert len(df) == n_before, "the legstadium join changed the row count"
    print(f"{len(df):,} rows; legstadium matched "
          f"{df[LS.FEATURE_COLS].notna().any(axis=1).mean():.1%} of them")

    base = feature_columns(df)
    base = [c for c in base if c not in LS.FEATURE_COLS]
    full = base + LS.FEATURE_COLS
    print(f"{len(base)} baseline features, {len(full)} with legstadium\n")

    rows = []
    for s in TEST_SEASONS:
        tr, te = df[df.season < s], df[df.season == s]
        # kicker_games.parquet as built has NO 2021 rows at all: the nflverse
        # kicking release is missing that season and build_dataset.py only
        # backfills from pbp for seasons after RELEASE_LAST_SEASON. Same
        # skip-if-tiny rule evaluate.py already uses, so the fold count drops
        # from 7 to 6 for both arms identically.
        if len(te) < 50:
            print(f"  fold {s} skipped, {len(te)} rows in the dataset", flush=True)
            continue
        r = {"season": s, "n": len(te)}
        for tag, cols in (("without", base), ("with", full)):
            m = LGBMRegressor(**PARAMS).fit(tr[cols], tr.fantasy_points)
            p = np.clip(m.predict(te[cols]), 0, None)
            r[f"mae_{tag}"] = mean_absolute_error(te.fantasy_points, p)
            r[f"sp_{tag}"] = spearmanr(te.fantasy_points, p).statistic
        print(f"  fold {s} done", flush=True)
        rows.append(r)
    res = pd.DataFrame(rows)
    res["mae_delta"] = res.mae_with - res.mae_without      # negative = better
    res["sp_delta"] = res.sp_with - res.sp_without         # positive = better

    print("per season (mae_delta < 0 and sp_delta > 0 mean the features helped):")
    print(res.round(4).to_string(index=False))

    w = res.n
    pool = {
        "mae_without": np.average(res.mae_without, weights=w),
        "mae_with": np.average(res.mae_with, weights=w),
        "sp_without": np.average(res.sp_without, weights=w),
        "sp_with": np.average(res.sp_with, weights=w),
    }
    print(f"\npooled (game-weighted over {int(w.sum()):,} kicker-games)")
    print(f"  MAE      without {pool['mae_without']:.4f}   with {pool['mae_with']:.4f}"
          f"   delta {pool['mae_with'] - pool['mae_without']:+.4f}")
    print(f"  Spearman without {pool['sp_without']:.4f}   with {pool['sp_with']:.4f}"
          f"   delta {pool['sp_with'] - pool['sp_without']:+.4f}")

    t = ttest_rel(res.mae_with, res.mae_without)
    wx = wilcoxon(res.mae_with, res.mae_without)
    print(f"\npaired over the {len(res)} folds, MAE with vs without:")
    print(f"  t = {t.statistic:+.3f}, p = {t.pvalue:.3f}")
    print(f"  wilcoxon W = {wx.statistic:.1f}, p = {wx.pvalue:.3f}")
    print(f"  folds improved: {(res.mae_delta < 0).sum()} of {len(res)}")
    ts = ttest_rel(res.sp_with, res.sp_without)
    print(f"  spearman t = {ts.statistic:+.3f}, p = {ts.pvalue:.3f}, "
          f"folds improved: {(res.sp_delta > 0).sum()} of {len(res)}")

    # Where the model spends its attention, if it takes the features at all.
    tr, te = df[df.season < 2025], df[df.season == 2025]
    m = LGBMRegressor(**PARAMS).fit(tr[full], tr.fantasy_points)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=full)
    gain = gain / gain.sum()
    print(f"\nshare of total split gain taken by the 13 new features, 2025 fit: "
          f"{gain[LS.FEATURE_COLS].sum():.1%}")
    print(gain[LS.FEATURE_COLS].sort_values(ascending=False).round(4).to_string())
    print(f"\nrank of each new feature among all {len(full)}:")
    rank = gain.rank(ascending=False)
    print(rank[LS.FEATURE_COLS].sort_values().astype(int).to_string())

    # The stadium question specifically: is the venue index anything the model
    # does not already get from is_dome, and does the SIGN hold out of sample?
    print("\nstadium index, out-of-sample check:")
    print(f"  corr(stad_fg_oe, is_dome) = {df.stad_fg_oe.corr(df.is_dome):+.3f}")
    print(f"  corr(stad_fg_oe, wind_outdoor) = "
          f"{df.stad_fg_oe.corr(df.wind_outdoor):+.3f}")
    # Per test season, split venues at the median prior index and compare the
    # ACTUAL make rate of the kicks played there that season.
    att = LS.load_attempts()
    games = pd.read_csv(ROOT / "data" / "external" / "games.csv", low_memory=False)
    att = att.merge(games[["game_id", "stadium_id"]], on="game_id", how="left")
    st = LS.stadium_index(LS.load_attempts(), games)
    a = att.merge(st, on=["stadium_id", "season"], how="inner")
    out = []
    for s in TEST_SEASONS:
        x = a[a.season == s]
        if x.empty:
            continue
        hi = x[x.stad_fg_oe > x.stad_fg_oe.median()]
        lo = x[x.stad_fg_oe <= x.stad_fg_oe.median()]
        out.append({"season": s, "n": len(x),
                    "make_at_easy_venues": hi.made.mean(),
                    "make_at_hard_venues": lo.made.mean(),
                    "gap": hi.made.mean() - lo.made.mean()})
    sp = pd.DataFrame(out)
    print("  prior-season index split at its median vs the make rate that "
          "actually happened:")
    print(sp.round(4).to_string(index=False))
    print(f"  mean gap {sp.gap.mean():+.4f}, seasons with the right sign "
          f"{(sp.gap > 0).sum()} of {len(sp)}")
    tg = ttest_rel(sp.make_at_easy_venues, sp.make_at_hard_venues)
    print(f"  paired t over seasons: t = {tg.statistic:+.3f}, p = {tg.pvalue:.3f}")
    # Where Denver actually lands, since it is the claim everybody makes.
    den = st[(st.season == st.season.max())].merge(
        games.drop_duplicates("stadium_id")[["stadium_id", "stadium"]],
        on="stadium_id", how="left").sort_values("stad_fg_oe", ascending=False)
    den = den.reset_index(drop=True)
    i = den.index[den.stadium.str.contains("Mile High|Empower|INVESCO", na=False)]
    for j in i:
        print(f"  Denver ({den.stadium[j]}): index {den.stad_fg_oe[j]:+.4f}, "
              f"rank {j + 1} of {len(den)}, {int(den.stad_fg_att_prior[j])} prior attempts")

    res.to_csv(ROOT / "reports" / "feat_legstadium_results.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
