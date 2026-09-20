"""Adversarial control for the +all rz result, plus which columns carry it.

The headline from feat_redzone_eval.py is that 27 red-zone columns cut pooled
MAE by 0.047 at p=0.054 over 6 folds. Twenty-seven columns is a lot to hand a
gradient booster, and "more columns changed the answer" is not the same claim as
"red-zone behaviour predicts kicker points". So:

  control    the SAME 27 columns with their values shuffled, breaking the link
             to the team-week while preserving every marginal distribution and
             the missingness pattern. Shuffled inside the test/train split it
             belongs to, so no information crosses the fold boundary. If the
             control also gains 0.047, the real arm proved nothing.
  importance LightGBM split-gain on the last fold, to name the columns that
             actually carry the effect rather than assert a mechanism.

The shuffle uses a fixed seed and is run three times with different seeds,
because one draw of a 27-column control is itself noisy.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kicker_analysis import feat_redzone                       # noqa: E402

REPORTS = ROOT / "reports"
RZ = feat_redzone.FEATURE_COLS
TEST_SEASONS = [2019, 2020, 2022, 2023, 2024, 2025]     # 2021 is not in the data


def fit(Xtr, ytr, Xte):
    m = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                      min_child_samples=40, verbose=-1, random_state=0, n_jobs=1)
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xte), 0, None), m


def main() -> int:
    df = pd.read_parquet(REPORTS / "_feat_redzone_frame.parquet")
    base = json.loads((REPORTS / "_feat_redzone_base.json").read_text())
    cols = base + RZ
    real = pd.read_csv(REPORTS / "feat_redzone_results.csv")

    rows = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        for s in TEST_SEASONS:
            tr, te = df[df.season < s].copy(), df[df.season == s].copy()
            # Shuffle within each split: the control must not become a channel
            # for test-set information, only a destroyer of the real signal.
            for part in (tr, te):
                idx = rng.permutation(len(part))
                part[RZ] = part[RZ].to_numpy()[idx]
            p, _ = fit(tr[cols], tr.fantasy_points, te[cols])
            rows.append({"seed": seed, "season": s, "n": len(te),
                         "mae": mean_absolute_error(te.fantasy_points, p),
                         "spearman": spearmanr(te.fantasy_points, p).statistic})
            print(f"  seed {seed} fold {s} done", flush=True)
    ctl = pd.DataFrame(rows)
    ctl.to_csv(REPORTS / "feat_redzone_control.csv", index=False)

    b = real[real.set == "base"].set_index("season").sort_index()
    a = real[real.set == "+all rz"].set_index("season").sort_index()
    print("\nPooled MAE, weighted by games:")
    print(f"  base            {np.average(b.mae, weights=b.n):.4f}")
    print(f"  +all rz (real)  {np.average(a.mae, weights=a.n):.4f}  "
          f"delta {np.average(a.mae, weights=a.n) - np.average(b.mae, weights=b.n):+.4f}")
    for seed, g in ctl.groupby("seed"):
        g = g.set_index("season").sort_index()
        print(f"  +27 shuffled s{seed}  {np.average(g.mae, weights=g.n):.4f}  "
              f"delta {np.average(g.mae, weights=g.n) - np.average(b.mae, weights=b.n):+.4f}"
              f"   Spearman {np.average(g.spearman, weights=g.n):.4f}")
    pooled = ctl.groupby("season").mae.mean()
    print(f"  +27 shuffled avg {np.average(pooled, weights=b.n):.4f}  "
          f"delta {np.average(pooled, weights=b.n) - np.average(b.mae, weights=b.n):+.4f}")

    # Which red-zone columns the model actually leans on, last fold.
    tr, te = df[df.season < 2025], df[df.season == 2025]
    _, m = fit(tr[cols], tr.fantasy_points, te[cols])
    gain = pd.Series(m.booster_.feature_importance("gain"), index=cols)
    gain = gain / gain.sum()
    rank = gain.rank(ascending=False).astype(int)
    print(f"\nCombined gain share of all 27 red-zone columns: "
          f"{gain[RZ].sum():.3f} of 1.000 across {len(cols)} features "
          f"(27/{len(cols)} = {27 / len(cols):.3f} would be neutral)")
    print("Red-zone columns by gain:")
    for c in gain[RZ].sort_values(ascending=False).index[:10]:
        print(f"  #{rank[c]:>3} of {len(cols)}   {c:<24} gain {gain[c]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
