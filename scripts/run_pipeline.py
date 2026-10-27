#!/usr/bin/env python3
"""Run the current pipeline end to end, one module at a time.

This replaced an entry point for the retired pipeline that imported names the
package no longer exports, so it raised ImportError before doing anything. It is
deliberately thin: every step is the plain command a reader would type,
`python src/kicker_analysis/<step>.py`, the form the README documents, run as a
subprocess with this same interpreter. Importing the modules instead would
create a second way to invoke them, one the README does not document, so the two
could quietly diverge.

Steps, in order, each reading what an earlier one wrote:

  build_dataset   nflverse + schedule  -> data/processed/kicker_games.parquet,
                                          reports/data_manifest.json
  fourth_down     play-by-play cache   -> data/processed/fourth_down.parquet
  backtest        season-forward folds -> reports/backtest_summary.csv and the
                                          component, variance, season-level and
                                          calibration CSVs beside it
  model_bakeoff   only with --bakeoff  -> reports/model_bakeoff_*.csv
  figures         the CSVs above       -> docs/images/*.png
  score_boards    published boards     -> reports/live_<season>_scorecard*.csv
  predict_week    this week's board    -> reports/week_<season>_<week>_kickers.csv

figures reads the bake-off's CSVs as well as the backtest's, so the bake-off,
when it runs, runs before it. It is opt-in for two reasons. It is the slow step:
it tunes and refits 14 challengers over 8 season-forward folds, and in one full
run on a 4-CPU Linux machine with OMP_NUM_THREADS=1 it took 135 s where the
other six steps took 33 s together. And it is a model-selection study, not part
of producing a board. Without --bakeoff, figures draws from the bake-off CSVs
already tracked in reports/, and its season chart's footnote says so when they
came from a different data build than the backtest's. score_boards runs before
predict_week so the scorecard describes the published boards as they stood
before this run wrote or replaced any.

It stops at the first step that exits non-zero. Carrying on would be worse than
failing: a board trained after a failed build is a board trained on whatever the
cache last held, and it looks exactly like a current one.

The weekly board is a record, so predict_week will not replace an existing
reports/week_<season>_<week>_kickers.csv; it says so and exits 0, which lets a
second run in the same week finish without touching it. `--force` is passed
through to replace it. The site payload is written only where KICKER_SITE_JS
points, and that variable reaches predict_week unchanged from this environment.

`--self-check` runs every module's own `--self-check` instead: offline, no data,
and it reports every failure rather than stopping at the first. A module counts
as having one if it defines `self_check()` or a `--self-check` flag, the same
rule tests/test_self_checks.py collects by. A module without one is listed as
not run rather than run, because it would ignore the flag and start real work.

    python scripts/run_pipeline.py                     # every step but the bake-off
    python scripts/run_pipeline.py --bakeoff           # the bake-off too, before figures
    python scripts/run_pipeline.py --skip-build        # reuse data/processed/
    python scripts/run_pipeline.py --steps backtest figures   # any subset, run in order
    python scripts/run_pipeline.py --week 8            # forwarded to predict_week
    python scripts/run_pipeline.py --week 8 --force    # rebuild an existing board over it
    python scripts/run_pipeline.py --dry-run           # print the commands only
    python scripts/run_pipeline.py --self-check        # every module, offline
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "kicker_analysis"
STEPS = ["build_dataset", "fourth_down", "backtest", "model_bakeoff",
         "figures", "score_boards", "predict_week"]
BUILD_STEPS = {"build_dataset", "fourth_down"}
# Runs only when --bakeoff asks for it; naming it in --steps is not enough.
OPT_IN = {"model_bakeoff"}
DEFAULT_STEPS = [s for s in STEPS if s not in OPT_IN]


def plan(steps: list[str] | None = None, skip_build: bool = False,
         bakeoff: bool = False) -> list[str]:
    """The steps to run, always in pipeline order whatever order they were named
    in, because each one reads what an earlier one wrote."""
    wanted = set(steps or DEFAULT_STEPS) - OPT_IN
    if bakeoff:
        wanted |= OPT_IN
    if skip_build:
        wanted -= BUILD_STEPS
    return [s for s in STEPS if s in wanted]


def command(step: str, week: int | None = None, force: bool = False) -> list[str]:
    cmd = [sys.executable, str(PKG / f"{step}.py")]
    if step == "predict_week":
        if week is not None:
            cmd += ["--week", str(week)]
        if force:
            cmd.append("--force")
    return cmd


def has_self_check(path: pathlib.Path) -> bool:
    """A top-level `def self_check` or a literal "--self-check" flag, the same
    rule tests/test_self_checks.py collects by, so the two lists cannot differ."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return (any(isinstance(n, ast.FunctionDef) and n.name == "self_check" for n in tree.body)
            or any(isinstance(n, ast.Constant) and n.value == "--self-check"
                   for n in ast.walk(tree)))


def env() -> dict[str, str]:
    # The README's advice, applied rather than hoped for: LightGBM thread
    # contention turns a sub-second fit into minutes on some machines. An
    # explicit setting in the caller's environment still wins. Everything else,
    # KICKER_SITE_JS included, is passed through as it is.
    e = dict(os.environ)
    e.setdefault("OMP_NUM_THREADS", "1")
    return e


def show(cmd: list[str]) -> str:
    return " ".join(["python"] + [str(pathlib.Path(c).relative_to(ROOT))
                                  if c.startswith(str(ROOT)) else c for c in cmd[1:]])


def run_steps(steps: list[str], week: int | None, dry_run: bool,
              force: bool = False) -> int:
    if not steps:
        print("run_pipeline: nothing to run (every requested step was skipped)")
        return 2
    for i, step in enumerate(steps, 1):
        cmd = command(step, week, force)
        print(f"\n==> [{i}/{len(steps)}] {step}: {show(cmd)}", flush=True)
        if dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, env=env()).returncode
        if rc != 0:
            print(f"\nrun_pipeline: {step} failed with exit code {rc} after "
                  f"{time.time() - t0:.1f}s; not run: {', '.join(steps[i:]) or 'nothing'}",
                  file=sys.stderr)
            return rc
        print(f"==> {step} ok, {time.time() - t0:.1f}s", flush=True)
    print(f"\nrun_pipeline: {'would run' if dry_run else 'ran'} {len(steps)} step(s): "
          f"{', '.join(steps)}")
    return 0


def self_checks(dry_run: bool) -> int:
    _check_plan()
    modules = sorted(p for p in PKG.glob("*.py") if p.name != "__init__.py")
    runnable = [p for p in modules if has_self_check(p)]
    skipped = [p.stem for p in modules if p not in runnable]
    failed = []
    for p in runnable:
        cmd = [sys.executable, str(p), "--self-check"]
        print(f"==> {p.stem}: {show(cmd)}", flush=True)
        if not dry_run and subprocess.run(cmd, cwd=ROOT, env=env()).returncode != 0:
            failed.append(p.stem)
    if dry_run:
        print(f"\nrun_pipeline self-check: would run {len(runnable)} module self-checks")
    else:
        print(f"\nrun_pipeline self-check: {len(runnable) - len(failed)} of "
              f"{len(runnable)} module self-checks passed")
    if skipped:
        print(f"  no self-check, not run: {', '.join(skipped)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _check_plan() -> None:
    """This script's own logic: order, opt-in, skipping, passthrough, and the
    files it points at."""
    assert plan() == DEFAULT_STEPS and "model_bakeoff" not in plan()
    assert plan(bakeoff=True) == STEPS
    # The bake-off is never run by naming it alone, and lands before figures.
    assert plan(["model_bakeoff"]) == []
    assert plan(["figures", "backtest"], bakeoff=True) == ["backtest", "model_bakeoff",
                                                           "figures"]
    assert plan(["predict_week", "build_dataset"]) == ["build_dataset", "predict_week"]
    assert plan(skip_build=True) == ["backtest", "figures", "score_boards", "predict_week"]
    assert plan(["fourth_down"], skip_build=True) == []
    assert STEPS.index("model_bakeoff") < STEPS.index("figures")
    assert STEPS.index("backtest") < STEPS.index("figures")
    assert STEPS.index("score_boards") < STEPS.index("predict_week")
    assert command("predict_week", 8)[-2:] == ["--week", "8"]
    assert command("predict_week", 8, force=True)[-3:] == ["--week", "8", "--force"]
    assert command("predict_week", force=True)[-1] == "--force"
    assert command("predict_week")[-1].endswith("predict_week.py")
    # --week and --force belong to predict_week alone.
    assert command("backtest", 8, force=True)[-1].endswith("backtest.py")
    assert command("score_boards", 8, force=True)[-1].endswith("score_boards.py")
    assert command("build_dataset")[0] == sys.executable
    for s in STEPS:
        assert (PKG / f"{s}.py").is_file(), f"pipeline step missing: {s}.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the kicker pipeline (" + " -> ".join(
            f"[{s}]" if s in OPT_IN else s for s in STEPS) + "), "
                    "stopping at the first failure. [model_bakeoff] runs only with --bakeoff.")
    ap.add_argument("--steps", nargs="+", choices=STEPS, metavar="STEP",
                    help="run only these steps, in pipeline order "
                         f"(choices: {', '.join(STEPS)}; model_bakeoff also "
                         "needs --bakeoff)")
    ap.add_argument("--bakeoff", action="store_true",
                    help="also run model_bakeoff, before figures; it is the slowest "
                         "step by far, so it is never part of a default run")
    ap.add_argument("--skip-build", action="store_true",
                    help="skip build_dataset and fourth_down and reuse the tables "
                         "already in data/processed/")
    ap.add_argument("--week", type=int, default=None,
                    help="forecast this week of the current season (passed to predict_week)")
    ap.add_argument("--force", action="store_true",
                    help="let predict_week replace an existing board CSV (passed to "
                         "predict_week; by default it keeps the published one)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run, run nothing")
    ap.add_argument("--self-check", action="store_true",
                    help="run every module's offline --self-check instead of the pipeline")
    a = ap.parse_args()
    if a.self_check:
        return self_checks(a.dry_run)
    if a.week is not None and a.week < 1:
        ap.error("--week must be 1 or more")
    if a.steps and set(a.steps) & OPT_IN and not a.bakeoff:
        ap.error("model_bakeoff runs only with --bakeoff; add it to run the bake-off")
    steps = plan(a.steps, a.skip_build, a.bakeoff)
    # Checked before anything runs: finding out after a build and a backtest that
    # the week was never going to be used is minutes wasted and easy to miss.
    if (a.week is not None or a.force) and "predict_week" not in steps:
        ap.error("--week and --force are passed only to predict_week, which this "
                 "run does not include")
    return run_steps(steps, a.week, a.dry_run, a.force)


if __name__ == "__main__":
    sys.exit(main())
