"""Run every module's `--self-check`, offline, as one pytest suite.

WHY: most modules in src/kicker_analysis/ carry an assert-based self-check that
pins an invariant whose silent failure would corrupt every number downstream:
the scoring and implied-total sign convention, the lag that keeps rolling
windows pre-kickoff, the closed-form lift against brute-force enumeration. They
were only ever run by hand from a README loop that had already drifted: it named
six modules while ten carried a check. So the list is not written down here.
Modules are found by reading the source: a top-level `def self_check` or a
literal `--self-check` flag. A new module is tested the day it lands, and one
that loses its check visibly drops out of the collected test ids.

Each check runs in a fresh interpreter exactly as a reader would run it,
`python src/kicker_analysis/<module>.py --self-check` from the repo root, so a
failure at import time (a module-level path that only resolves on one OS) is
caught too. Three things are enforced beyond the exit code:

  * offline. tests/offline/sitecustomize.py makes any TCP connect raise, so a
    check that starts downloading fails here even on a machine with network.
    test_offline_guard_blocks_tcp proves the guard is actually loaded.
  * wired. A module with self_check() whose `__main__` block never calls it would
    exit 0 without checking anything, so that fails as its own test.
  * bounded. OMP_NUM_THREADS=1 (LightGBM thread contention turns a sub-second fit
    into minutes) and a per-module timeout, so a hang fails instead of stalling.

scripts/run_pipeline.py is checked here too: its own asserts (step order, the
bake-off opt-in, the --week and --force passthrough, every step's file present)
run, and the modules it would self-check must be exactly the ones collected
here, since both claim to discover by the same rule.

    pytest -q
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "kicker_analysis"
RUNNER = ROOT / "scripts" / "run_pipeline.py"
OFFLINE = pathlib.Path(__file__).resolve().parent / "offline"
FLAG = "--self-check"
# Each check measured 0.5-4.2 s wall, interpreter start included, at the last
# revision of this file; figures, which renders its charts, is the slowest. The
# bound is for a hang, not for speed, so a cold CI runner has room.
TIMEOUT_S = 120


def _is_main_guard(node: ast.stmt) -> bool:
    """`if __name__ == "__main__":`"""
    t = node.test if isinstance(node, ast.If) else None
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == "__name__" and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


def _mentions_self_check(nodes: list[ast.stmt]) -> bool:
    """True if the statements call self_check() or read the parsed flag."""
    for n in (sub for node in nodes for sub in ast.walk(node)):
        if isinstance(n, ast.Name) and n.id == "self_check":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "self_check":
            return True
    return False


def discover() -> dict[str, tuple[pathlib.Path, bool]]:
    """{module: (path, wired)} for every module that has a self-check.

    A module qualifies with a top-level `def self_check` OR a literal
    "--self-check" constant (an argparse flag; a docstring that merely mentions
    the flag is one longer string and does not match). `wired` says whether its
    `__main__` block actually reaches the check.
    """
    found = {}
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        has_def = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and n.name == "self_check" for n in tree.body)
        has_flag = any(isinstance(n, ast.Constant) and n.value == FLAG
                       for n in ast.walk(tree))
        if not (has_def or has_flag):
            continue
        mains = [n for n in tree.body if _is_main_guard(n)]
        found[path.stem] = (path, any(_mentions_self_check(m.body) for m in mains))
    return found


MODULES = discover()


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"  # a figure module must not reach for a display
    env["PYTHONPATH"] = os.pathsep.join(
        [str(OFFLINE)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=_env(),
                          capture_output=True, text=True, timeout=TIMEOUT_S)


def _report(what: str, proc) -> str:
    """Message plus both streams. A TimeoutExpired carries raw bytes, or None."""
    def text(s) -> str:
        return s.decode(errors="replace") if isinstance(s, bytes) else (s or "")
    return (f"{what}\n--- stdout ---\n{text(proc.stdout)}"
            f"\n--- stderr ---\n{text(proc.stderr)}")


def test_discovery_found_modules():
    # An empty parametrize list is reported as a skip, which would turn a broken
    # discovery into a green run. Fail loudly instead.
    assert MODULES, f"no module under {PKG} has a self_check() or {FLAG} flag"


def test_offline_guard_blocks_tcp():
    """The guard is loaded in the subprocess and refuses a TCP connect.

    Without this, a typo in the PYTHONPATH wiring would leave every self-check
    running with full network and the offline claim untested.
    """
    proc = _run(["-c", "import socket; socket.create_connection(('127.0.0.1', 9), 1)"])
    assert proc.returncode != 0, _report("TCP connect was not blocked", proc)
    assert "--self-check must run offline" in proc.stderr, _report(
        "connect failed, but not because of the offline guard", proc)


def test_pipeline_runner_self_check():
    """run_pipeline.py's own checks pass and it finds the same modules as this file.

    Without --dry-run its --self-check would run every module check a second
    time, so the dry run is used: it still executes the script's asserts and
    prints each module it would run.
    """
    proc = _run([str(RUNNER), FLAG, "--dry-run"])
    assert proc.returncode == 0, _report(
        f"python scripts/run_pipeline.py {FLAG} --dry-run exited {proc.returncode}", proc)
    listed = set(re.findall(r"^==> (\w+): ", proc.stdout, re.M))
    assert listed == set(MODULES), _report(
        f"run_pipeline would self-check {sorted(listed)}, pytest collects "
        f"{sorted(MODULES)}", proc)


@pytest.mark.parametrize("module", sorted(MODULES))
def test_self_check(module: str):
    path, wired = MODULES[module]
    rel = path.relative_to(ROOT).as_posix()
    if not wired:
        pytest.fail(f"{rel} has a self-check, but its `if __name__ == \"__main__\":` "
                    f"block never calls it, so `python {rel} {FLAG}` checks nothing",
                    pytrace=False)
    timed_out = None
    try:
        proc = _run([str(path), FLAG])
    except subprocess.TimeoutExpired as e:
        timed_out = e
    # Failed outside the handler, so the report is not chained to the timeout.
    if timed_out is not None:
        pytest.fail(_report(f"python {rel} {FLAG} did not finish in {TIMEOUT_S}s",
                            timed_out), pytrace=False)
    if proc.returncode != 0:
        pytest.fail(_report(f"python {rel} {FLAG} exited {proc.returncode}", proc),
                    pytrace=False)
