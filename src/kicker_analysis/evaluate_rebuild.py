"""Finite, exploratory 2018-2025 comparison specified before the rebuild fits.

Random observed pairs are diagnostic only. No historical free-agent availability
is inferred, and no candidate can pass the real-policy promotion bar from this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from kicker_analysis.model import load, RidgeRanker, CompoundKickerModel, GlmKickerModel
from kicker_analysis.scoring import load_scoring
from kicker_analysis.prepare import DEFAULT_OUT
from kicker_analysis.backtest import weekly_lifts, season_tests


def empirical_crps(train_y, y):
    """Exact CRPS of the unconditioned empirical TRAINING outcome distribution."""
    x = np.sort(np.asarray(train_y, float))
    y = np.asarray(y, float)
    index = np.searchsorted(x, y)
    partial = np.r_[0., np.cumsum(x)]
    absolute = (y * index - partial[index] + partial[-1] - partial[index]
                - y * (len(x) - index)) / len(x)
    pair = np.dot(2 * np.arange(1, len(x) + 1) - len(x) - 1, x) / len(x) ** 2
    return absolute - pair


def market_fit(train):
    return make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(
        train[["implied_total"]], train.fantasy_points)


def select_strength(train, coach):
    years = sorted(train.season.unique())[1:][-2:]
    losses = {}
    for strength in [20., 100.]:
        scores = []
        for year in years:
            a, b = train[train.season < year], train[train.season == year]
            pred = GlmKickerModel(strength=strength, coach=coach, n_sims=1,
                                 scoring=train.attrs["scoring"]).fit(a).predict(b)
            scores.append(float(np.abs(pred.exp_points.to_numpy() - b.fantasy_points).mean()))
        losses[strength] = float(np.mean(scores))
    return min(losses, key=losses.get), losses


def run(processed, scoring_path, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run-inputs.json").exists():
        raise ValueError("evaluation identity already exists; use a new run directory")
    identity = {"status": "frozen-before-fit", "scoring_snapshot": hashlib.sha256(scoring_path.read_bytes()).hexdigest(),
        "protocol": hashlib.sha256((Path(__file__).parents[2] / "notes/kicker-rebuild-protocol-20261115.md").read_bytes()).hexdigest(),
        "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [processed / "kicker_games.parquet", processed / "fourth_down.parquet",
             processed / "coach_history.parquet", Path(__file__).parents[2] / "data/external/games.csv"]},
        "source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    (out / "run-inputs.json").write_text(json.dumps(identity, indent=2) + "\n")
    scoring = load_scoring(scoring_path)
    data = load(processed, scoring=scoring)
    data = data[data.model_eligible & data.season.le(2025)].copy()
    rows, diagnostics = [], []
    for year in range(2018, 2026):
        train, test = data[data.season < year].copy(), data[data.season == year].copy()
        train.attrs["scoring"] = scoring
        ridge = RidgeRanker().fit(train).predict(test)
        market = market_fit(train).predict(test[["implied_total"]])
        frame = test[["season", "week", "game_id", "team", "player_id",
                      "player_display_name", "fantasy_points", "participation_status"]].reset_index(drop=True)
        frame = frame.rename(columns={"fantasy_points": "y"})
        frame["market"] = market
        frame["implied"] = test.implied_total.fillna(train.implied_total.median()).to_numpy()
        frame["ridge"] = ridge
        frame["constant"] = float(train.fantasy_points.mean())
        frame["empirical_crps"] = empirical_crps(train.fantasy_points, test.fantasy_points)
        models = {"compound": CompoundKickerModel(scoring=scoring),
                  "glm_pool": GlmKickerModel(scoring=scoring)}
        chosen = {}
        for name, coach in [("glm_shrink", False), ("glm_coach", True)]:
            strength, losses = select_strength(train, coach)
            chosen[name] = {"strength": strength, "inner_mae": losses}
            models[name] = GlmKickerModel(strength=strength, coach=coach, scoring=scoring)
        for name, model in models.items():
            model.fit(train)
            p = model.predict(test)
            frame[name] = p.exp_points.to_numpy()
            for c in ["crps", "floor_p20", "ceiling_p90", "p_ge_8", "p_ge_10", "p_ge_12", "p_ge_15"]:
                frame[name + "|" + c] = p[c].to_numpy()
            diagnostics.append({"season": year, "model": name,
                "dispersion": model.raw_dispersion_, "bounded_dispersion": model.total_dispersion_,
                "residual_fg_pat_correlation": model.residual_correlation_,
                "inner_selection": chosen.get(name), "train_n": len(train), "test_n": len(test)})
        for t in [8, 10, 12, 15]:
            frame[f"base_{t}"] = float(train.fantasy_points.ge(t).mean())
        rows.append(frame)
        print(f"season {year}: {len(test)} test rows, inner choices {chosen}", flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    methods = ["constant", "implied", "market", "ridge", "compound", "glm_pool", "glm_shrink", "glm_coach"]
    lifts = weekly_lifts(predictions, methods)
    summary, comparisons = [], []
    for name in methods:
        y, estimate = predictions.y.to_numpy(), predictions[name].to_numpy()
        row = {"model": name, "mae": float(np.abs(y - estimate).mean()) if name != "implied" else None,
               "rmse": float(np.sqrt(np.mean((y - estimate) ** 2))) if name != "implied" else None,
               "equal_season_pair_lift": float(lifts.groupby("season")[name].mean().mean())}
        if name in ["compound", "glm_pool", "glm_shrink", "glm_coach"]:
            row["crps"] = float(predictions[name + "|crps"].mean())
            row["crps_train_empirical"] = float(predictions.empirical_crps.mean())
            row["crps_skill_vs_empirical"] = 1 - row["crps"] / row["crps_train_empirical"]
            row["coverage_p20_p90"] = float(((y >= predictions[name + "|floor_p20"])
                & (y <= predictions[name + "|ceiling_p90"])).mean())
            row["interval_width"] = float((predictions[name + "|ceiling_p90"]
                                             - predictions[name + "|floor_p20"]).mean())
            for t in [8, 10, 12, 15]:
                observed = (y >= t).astype(float)
                prob = predictions[f"{name}|p_ge_{t}"]
                loss = float(np.mean((prob - observed) ** 2))
                base = float(np.mean((predictions[f"base_{t}"] - observed) ** 2))
                row[f"brier_{t}"] = loss
                row[f"brier_skill_{t}"] = 1 - loss / base
                row[f"predicted_rate_{t}"] = float(prob.mean())
                row[f"observed_rate_{t}"] = float(observed.mean())
        summary.append(row)
        if name not in ["market", "constant", "implied"]:
            comparisons.append({"model": name, "reference": "market", **season_tests(lifts, name, "market")})
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    from kicker_analysis.backtest import reliability
    calibration_rows = []
    for name in ["compound", "glm_pool", "glm_shrink", "glm_coach"]:
        for season, fold in predictions.groupby("season"):
            for t in [8, 10, 12, 15]:
                table = reliability(fold[f"{name}|p_ge_{t}"].to_numpy(),
                                    fold.y.ge(t).astype(float).to_numpy(), bins=5)
                table["model"], table["season"], table["threshold"] = name, season, t
                calibration_rows.append(table)
    pd.concat(calibration_rows).to_csv(out / "conditional_reliability.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "season_comparisons.csv", index=False)
    lifts.to_csv(out / "weekly_pair_diagnostics.csv", index=False)
    inputs_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [processed / "kicker_games.parquet", processed / "fourth_down.parquet",
         processed / "coach_history.parquet", Path(__file__).parents[2] / "data/external/games.csv"]}
    source_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    protocol_after = hashlib.sha256((Path(__file__).parents[2] / "notes/kicker-rebuild-protocol-20261115.md").read_bytes()).hexdigest()
    if (identity["inputs"] != inputs_after or identity["source"] != source_after
            or identity["protocol"] != protocol_after):
        raise ValueError("run inputs/source changed during execution; results are not accepted")
    doc = {"status": "exploratory", "historical_policy_value": "unavailable: no archived feasible action sets",
        "decision": "simplify conservatively; the real-feasible-policy promotion bar is untested, not numerically failed",
        "scoring_id": scoring.identity, "scoring": scoring.__dict__, "folds": diagnostics,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")},
        "protocol_sha256": hashlib.sha256((Path(__file__).parents[2] / "notes/kicker-rebuild-protocol-20261115.md").read_bytes()).hexdigest()}
    (out / "evaluation.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scoring-settings", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.processed_dir, args.scoring_settings, args.out_dir)
