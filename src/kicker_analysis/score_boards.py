"""Score the published weekly boards against what happened, and say which were forecasts.

The backtest measures the method on seasons it was not fit on. The weekly boards
in reports/ are the method doing the job live, and until this module nothing
checked them. A board is evidence about forecasting only if it was fixed before
the games it covers. A board written, or rewritten, after its games can look
good for reasons that have nothing to do with the model: an injured kicker
dropped from the list, a line that moved, a pick swapped. So every row is
scored, and every row also says whether it was a forecast when it was written.

Provenance is decided per GAME, against two clocks for the board's bytes:

  committed  the later of the git author and committer dates of the earliest
             commit whose copy of the board has exactly the bytes on disk now
             (matched by git blob hash, so an edited board loses the date of the
             version it replaced). The machine that makes a commit writes these
             dates and git lets anyone set them, so this clock is a claim.
  public     the time of the earliest push to the GitHub repository whose head
             commit carries those bytes, from GitHub's repository activity log.
             GitHub's server writes that time, not the author. Only the head of
             each push is inspected, so bytes that were public only in the
             middle of a multi-commit push are missed; that errs toward
             "retrospective", never toward "prospective".

A row is prospective on a clock when that clock is strictly before its game's
kickoff (games.csv gameday + gametime, America/New_York). A clock that cannot be
established counts as after kickoff. Every statistic is reported three ways:
all rows, rows prospective on the commit clock, rows prospective on the public
clock. Only the last is checkable by someone who does not trust the author's
machine. Both clocks are written to the provenance table, so a later run with
no git history or no network reuses them for any board whose bytes still match.

Scoring reuses backtest.py's exact code rather than restating it: the R=2
decision lift (decision, built on the closed form in _pick_probs) and the
paired week-block bootstrap (boot_lift). The within-week Spearman follows the
backtest's rule (weeks with fewer than five scored kickers are skipped). The
board's order is its rank_points column, the order a reader sees; the
comparison is the same week's kickers ordered by implied_total alone, the
market baseline the backtest also uses. Per week and pooled over weeks (each
week weighted equally, as in the backtest):

  kickers_scored           board rows with a final game and a kick row
  spearman_board           within-week Spearman of board order vs points
  lift_board, lift_implied exact R=2 lift, points per week over a random start
  oracle_lift, share_*     the lift of perfect hindsight, and each lift's share
  board_minus_implied      lift_board - lift_implied
  boot_*, t_*              pooled only: week-block bootstrap and a t interval
                           on the per-week differences (df = weeks - 1)
  pred_p15, realised_p15   mean predicted P(15+) against the realised 15+ rate

Excluded, and counted: rows whose game is not final (games.csv has no score, or
the kick table has no row for the game yet), and board kickers with no row in
the kick table for their week. kicker_games.parquet has one row per kicker with
a FG or PAT attempt, so a kicker who was inactive, and one who was active with
no attempt, both land in did_not_kick rather than at 0 points. Kickers who did
kick in a scored game but were not on the board are counted as unlisted_kicked.

A handful of weeks cannot separate a board from the implied total.
reports/season_level_tests.csv puts the backtest's average edge of the ridge
over the implied total at under a tenth of a point a week, while the live
weekly differences move by several tenths from one week to the next. With this
few weeks the percentile bootstrap is also too narrow (it can only resample the
weeks there are), so the t interval is printed beside it and is the one to
read. Neither is evidence either way until many more weeks exist.

    python src/kicker_analysis/score_boards.py               # writes reports/live_<season>_scorecard*.csv
    python src/kicker_analysis/score_boards.py --offline     # no GitHub calls; reuse recorded clocks
    python src/kicker_analysis/score_boards.py --self-check  # offline, synthetic, writes nothing to the repo

Inputs: data/processed/kicker_games.parquet (build_dataset.py) and
data/external/games.csv (cached by build_dataset.py). Run build_dataset.py first
so the latest week's kicks are in the table.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kicker_analysis.backtest import _pick_probs, boot_lift, decision  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
KICKER_GAMES = ROOT / "data" / "processed" / "kicker_games.parquet"
GAMES_CSV = ROOT / "data" / "external" / "games.csv"
SEASON = 2026
ROSTER = 2
N_BOOT = 10000
# A pooled interval over fewer weeks than this is not printed at all.
MIN_CI_WEEKS = 3
GAME_TZ = "America/New_York"
GITHUB_REPO = "tdub29/Kicker-Analysis"
GITHUB_API = "https://api.github.com"
BOARD_RE = re.compile(r"^week_(\d{4})_(\d+)_kickers\.csv$")
SUBSETS = ("all", "prospective_commit", "prospective_public")
SCORE_COLUMNS = ["subset", "week", "n_weeks", "board_rows", "not_final", "did_not_kick",
                 "kickers_scored", "unlisted_kicked", "spearman_board", "spearman_implied",
                 "lift_board", "lift_implied", "oracle_lift", "share_board", "share_implied",
                 "board_minus_implied", "boot_lo", "boot_hi", "boot_p", "t_lo", "t_hi", "t_p",
                 "pred_p15", "realised_p15"]
PROV_COLUMNS = ["season", "week", "board", "board_blob", "first_commit", "committed_utc",
                "public_commit", "public_utc", "game_id", "kickoff_utc", "final", "board_rows",
                "scored_rows", "prospective_commit", "prospective_public"]


# ------------------------------------------------------------------ provenance

def git_blob_sha(data: bytes) -> str:
    """The id git gives these bytes as a blob, computed without git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(args: list[str], cwd: pathlib.Path) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def board_blob(path: pathlib.Path) -> str:
    """Blob id of the board as git would store it.

    `git hash-object` applies the repository's checkout filters (a Windows clone
    with autocrlf holds CRLF on disk and LF in the blob), so it is preferred;
    without git the raw bytes are hashed.
    """
    got = _git(["hash-object", path.name], path.parent)
    return got if got else git_blob_sha(path.read_bytes())


def git_first_commit(path: pathlib.Path, blob: str) -> tuple[str | None, pd.Timestamp]:
    """Earliest commit on HEAD's history whose copy of `path` is `blob`.

    A shallow clone (CI checks out one commit) would report its own cut-off
    commit as the first, which is later than the truth, so it counts as no
    history and the recorded clock is used instead.
    """
    top = _git(["rev-parse", "--show-toplevel"], path.parent)
    if not top or _git(["rev-parse", "--is-shallow-repository"], path.parent) != "false":
        return None, pd.NaT
    rel = path.resolve().relative_to(pathlib.Path(top).resolve()).as_posix()
    log = _git(["log", "--reverse", "--format=%H %aI %cI", "--", rel], pathlib.Path(top))
    for line in (log or "").splitlines():
        sha, authored, committed = line.split()
        if _git(["rev-parse", f"{sha}:{rel}"], pathlib.Path(top)) == blob:
            return sha, max(_utc(authored), _utc(committed))
    return None, pd.NaT


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _github_json(url: str) -> tuple[object, str]:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "kicker-analysis-score-boards"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r), r.headers.get("Link", "") or ""


def github_pushes(repo: str) -> list[dict]:
    """Every ref update GitHub logged for `repo`, oldest first: {timestamp, after}."""
    url, out = f"{GITHUB_API}/repos/{repo}/activity?per_page=100", []
    while url:
        data, link = _github_json(url)
        for a in data:
            after = a.get("after") or ""
            if after and set(after) != {"0"}:            # a deletion has no head
                out.append({"timestamp": _utc(a["timestamp"]), "after": after})
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = nxt.group(1) if nxt else None
    return sorted(out, key=lambda a: a["timestamp"])


def github_tree_lookup(repo: str):
    """sha -> {path: blob} for a commit's full tree, cached; {} if GitHub lacks it."""
    cache: dict[str, dict] = {}

    def lookup(sha: str) -> dict:
        if sha not in cache:
            try:
                data, _ = _github_json(f"{GITHUB_API}/repos/{repo}/git/trees/{sha}?recursive=1")
            except urllib.error.HTTPError as e:
                if e.code not in (404, 409, 422):
                    raise
                data = {"tree": []}
            if data.get("truncated"):
                print(f"  warning: GitHub truncated the tree of {sha[:7]}; a board "
                      f"missing from it reads as not yet public")
            cache[sha] = {e["path"]: e["sha"] for e in data["tree"] if e["type"] == "blob"}
        return cache[sha]
    return lookup


def first_public(pushes: list[dict], path: str, blob: str, tree) -> tuple[str | None, pd.Timestamp]:
    """Earliest push whose head commit has `blob` at `path`. `tree(sha)` -> {path: blob}."""
    for p in sorted(pushes, key=lambda a: a["timestamp"]):
        if tree(p["after"]).get(path) == blob:
            return p["after"], p["timestamp"]
    return None, pd.NaT


def kickoff_utc(games: pd.DataFrame) -> pd.Series:
    """games.csv lists gameday and gametime in US Eastern; return UTC kickoffs."""
    local = pd.to_datetime(games.gameday.astype(str) + " " + games.gametime.astype(str),
                           format="%Y-%m-%d %H:%M", errors="coerce")
    return local.dt.tz_localize(GAME_TZ, ambiguous="NaT",
                                nonexistent="NaT").dt.tz_convert("UTC")


def is_prospective(clock: pd.Series, kickoff: pd.Series) -> pd.Series:
    """True where the clock is known and strictly before kickoff."""
    c = pd.to_datetime(clock, utc=True)
    k = pd.to_datetime(kickoff, utc=True)
    return (c.notna() & k.notna() & (c < k)).astype(bool)


# ------------------------------------------------------------------ boards

def load_boards(boards_dir: pathlib.Path, season: int) -> list[dict]:
    out = []
    paths = [p for p in boards_dir.glob(f"week_{season}_*_kickers.csv") if BOARD_RE.match(p.name)]
    for path in sorted(paths, key=lambda p: int(BOARD_RE.match(p.name).group(2))):
        week = int(BOARD_RE.match(path.name).group(2))
        b = pd.read_csv(path)
        assert (b.season == season).all() and (b.week == week).all(), path.name
        assert b.player_id.is_unique, f"{path.name} lists a kicker twice"
        for c in ("rank_points", "implied_total", "p_ge_15"):
            assert b[c].notna().all(), f"{path.name}: {c} has a missing value"
        out.append({"path": path, "week": week, "frame": b})
    return out


def resolve_clocks(boards: list[dict], prior: pd.DataFrame | None,
                   repo: str | None, use_git: bool = True) -> list[dict]:
    """Attach both clocks to each board, reusing a recorded clock when the bytes match.

    `repo` None skips GitHub, `use_git` False skips git. A clock that cannot be
    computed now is taken from `prior` (the provenance table a previous run
    wrote) only for the same blob.
    """
    pushes, tree = None, None
    if repo:
        try:
            pushes, tree = github_pushes(repo), github_tree_lookup(repo)
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"  warning: GitHub activity unavailable ({e}); public clock "
                  f"reused from the last provenance table where the bytes match")
    rec = {}
    if prior is not None and len(prior):
        for _, r in prior.drop_duplicates("board").iterrows():
            rec[r.board] = r
    for bd in boards:
        path = bd["path"]
        bd["blob"] = board_blob(path) if use_git else git_blob_sha(path.read_bytes())
        sha, when = git_first_commit(path, bd["blob"]) if use_git else (None, pd.NaT)
        top = _git(["rev-parse", "--show-toplevel"], path.parent) if use_git else None
        rel = (path.resolve().relative_to(pathlib.Path(top).resolve()).as_posix() if top
               else f"reports/{path.name}")
        psha, pwhen = None, pd.NaT
        if pushes is not None:
            try:
                psha, pwhen = first_public(pushes, rel, bd["blob"], tree)
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"  warning: GitHub tree lookup failed for {path.name} ({e})")
                pushes = None
        old = rec.get(path.name)
        same = old is not None and old.board_blob == bd["blob"]
        if sha is None and same and pd.notna(old.committed_utc):
            sha, when = old.first_commit, _utc(old.committed_utc)
        if pushes is None and same and pd.notna(old.public_utc):
            psha, pwhen = old.public_commit, _utc(old.public_utc)
        bd.update(first_commit=sha, committed_utc=when, public_commit=psha, public_utc=pwhen)
    return boards


# ------------------------------------------------------------------ scoring

def board_rows(boards: list[dict], games: pd.DataFrame, kicks: pd.DataFrame) -> pd.DataFrame:
    """One row per board kicker with kickoff, clocks, status and actual points.

    `kicks` is the season's regular-season kick table (kicker_games.parquet).
    status: scored | did_not_kick | not_final.
    """
    g = games.assign(kickoff_utc=kickoff_utc(games))[["game_id", "kickoff_utc", "home_score"]]
    kicked_games = set(kicks.game_id.dropna())
    rows = []
    for bd in boards:
        b = bd["frame"].merge(g, on="game_id", how="left")
        b["board"] = bd["path"].name
        for c in ("blob", "first_commit", "committed_utc", "public_commit", "public_utc"):
            b["board_blob" if c == "blob" else c] = bd.get(c)
        rows.append(b)
    R = pd.concat(rows, ignore_index=True)
    R["final"] = R.home_score.notna() & R.game_id.isin(kicked_games)
    act = kicks[["week", "player_id", "fantasy_points"]].rename(columns={"fantasy_points": "y"})
    assert not act.duplicated(["week", "player_id"]).any(), "a kicker kicked twice in a week"
    R = R.merge(act, on=["week", "player_id"], how="left")
    R["status"] = np.where(~R.final, "not_final", np.where(R.y.isna(), "did_not_kick", "scored"))
    R["prospective_commit"] = is_prospective(R.committed_utc, R.kickoff_utc)
    R["prospective_public"] = is_prospective(R.public_utc, R.kickoff_utc)
    R["all"] = True
    return R


def week_spearman(P: pd.DataFrame, col: str) -> float:
    """Mean over weeks of the within-week Spearman of points y on `col`.

    Weeks with fewer than five rows, or a constant y or `col`, are skipped, the
    same rule backtest.py applies to its rho column.
    """
    vals = []
    for _, g in P.groupby("week"):
        if len(g) < 5 or g[col].nunique() < 2 or g.y.nunique() < 2:
            continue
        r = stats.spearmanr(g.y, g[col]).statistic
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def _t_interval(d: np.ndarray) -> tuple[float, float, float]:
    k = len(d)
    if k < 2 or not np.isfinite(d).all():
        return np.nan, np.nan, np.nan
    se = d.std(ddof=1) / np.sqrt(k)
    if se == 0:
        return float(d.mean()), float(d.mean()), (1.0 if d.mean() == 0 else 0.0)
    h = stats.t.ppf(0.975, k - 1) * se
    return float(d.mean() - h), float(d.mean() + h), float(stats.ttest_1samp(d, 0.0).pvalue)


def score(R: pd.DataFrame, kicks: pd.DataFrame, pooled: bool, n_boot: int = N_BOOT) -> dict:
    """Every scorecard statistic for the board rows in R (one week, or several)."""
    S = R[R.status == "scored"]
    out = {"n_weeks": int(S.week.nunique()), "board_rows": len(R),
           "not_final": int((R.status == "not_final").sum()),
           "did_not_kick": int((R.status == "did_not_kick").sum()),
           "kickers_scored": len(S)}
    scored_games = kicks[kicks.game_id.isin(S.game_id)]
    listed = set(zip(R.week, R.player_id))
    out["unlisted_kicked"] = int(sum((w, p) not in listed
                                     for w, p in zip(scored_games.week, scored_games.player_id)))
    # rank_points is 1 for the favourite; negate so a higher score is a better start.
    P = pd.DataFrame({"season": S.season.to_numpy(), "week": S.week.to_numpy(),
                      "y": S.y.to_numpy(float), "board": -S.rank_points.to_numpy(float),
                      "implied": S.implied_total.to_numpy(float)})
    out["spearman_board"] = week_spearman(P, "board")
    out["spearman_implied"] = week_spearman(P, "implied")
    db, di = decision(P, "board", ROSTER), decision(P, "implied", ROSTER)
    out.update(lift_board=db["lift"], lift_implied=di["lift"], oracle_lift=db["oracle_lift"],
               share_board=db["share"], share_implied=di["share"],
               board_minus_implied=db["lift"] - di["lift"])
    nan = dict.fromkeys(("boot_lo", "boot_hi", "boot_p", "t_lo", "t_hi", "t_p"), np.nan)
    out.update(nan)
    if pooled and out["n_weeks"] >= MIN_CI_WEEKS:
        bl = boot_lift(P, "board", "implied", ROSTER, n=n_boot)
        assert abs(bl["diff"] - out["board_minus_implied"]) < 1e-12
        d = np.array([decision(g, "board", ROSTER)["lift"] - decision(g, "implied", ROSTER)["lift"]
                      for _, g in P.groupby("week") if len(g) >= ROSTER])
        assert abs(d.mean() - out["board_minus_implied"]) < 1e-12
        t_lo, t_hi, t_p = _t_interval(d)
        out.update(boot_lo=bl["lo"], boot_hi=bl["hi"], boot_p=bl["p"],
                   t_lo=t_lo, t_hi=t_hi, t_p=t_p)
    out["pred_p15"] = float(S.p_ge_15.mean()) if len(S) else np.nan
    out["realised_p15"] = float((S.y >= 15).mean()) if len(S) else np.nan
    return out


def scorecard(R: pd.DataFrame, kicks: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for subset in SUBSETS:
        Rs = R[R[subset]]
        for week, Rw in Rs.groupby("week", sort=True):
            rows.append({"subset": subset, "week": int(week),
                         **score(Rw, kicks, pooled=False, n_boot=n_boot)})
        if len(Rs):
            rows.append({"subset": subset, "week": "pooled",
                         **score(Rs, kicks, pooled=True, n_boot=n_boot)})
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def provenance_table(R: pd.DataFrame) -> pd.DataFrame:
    def iso(t):
        return t.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(t) else ""
    agg = (R.groupby(["season", "week", "board", "game_id"], sort=True)
           .agg(board_blob=("board_blob", "first"), first_commit=("first_commit", "first"),
                committed_utc=("committed_utc", "first"), public_commit=("public_commit", "first"),
                public_utc=("public_utc", "first"), kickoff_utc=("kickoff_utc", "first"),
                final=("final", "first"), board_rows=("player_id", "size"),
                scored_rows=("status", lambda s: int((s == "scored").sum())),
                prospective_commit=("prospective_commit", "first"),
                prospective_public=("prospective_public", "first"))
           .reset_index())
    for c in ("committed_utc", "public_utc", "kickoff_utc"):
        agg[c] = agg[c].map(lambda t: iso(_utc(t)) if pd.notna(t) else "")
    return agg[PROV_COLUMNS]


def main(args: argparse.Namespace) -> int:
    boards_dir, out_dir = pathlib.Path(args.boards), pathlib.Path(args.out)
    boards = load_boards(boards_dir, args.season)
    if not boards:
        raise SystemExit(f"no week_{args.season}_*_kickers.csv in {boards_dir}")
    weeks_csv = out_dir / f"live_{args.season}_scorecard_weeks.csv"
    prior = pd.read_csv(weeks_csv, dtype=str, keep_default_na=False).replace("", np.nan) \
        if weeks_csv.exists() else None
    boards = resolve_clocks(boards, prior, None if args.offline else args.github_repo)

    games = pd.read_csv(args.games, low_memory=False)
    games = games[games.season == args.season]
    kicks = pd.read_parquet(args.kicker_games)
    kicks = kicks[(kicks.season == args.season) & (kicks.season_type == "REG")]
    R = board_rows(boards, games, kicks)

    print(f"{args.season} boards: {len(boards)}, kicker-rows: {len(R)}")
    for bd in boards:
        c, p = bd["committed_utc"], bd["public_utc"]
        sub = R[R.board == bd["path"].name]
        print(f"  week {bd['week']}: committed {c if pd.notna(c) else 'unknown'} "
              f"({(bd['first_commit'] or '-')[:7]}), public {p if pd.notna(p) else 'unknown'}; "
              f"prospective rows {int(sub.prospective_commit.sum())}/{len(sub)} by commit, "
              f"{int(sub.prospective_public.sum())}/{len(sub)} by public push")

    card = scorecard(R, kicks)
    prov = provenance_table(R)
    out_dir.mkdir(parents=True, exist_ok=True)
    card.to_csv(out_dir / f"live_{args.season}_scorecard.csv", index=False)
    prov.to_csv(weeks_csv, index=False)

    show = ["subset", "week", "kickers_scored", "did_not_kick", "not_final", "spearman_board",
            "lift_board", "lift_implied", "oracle_lift", "share_board", "board_minus_implied",
            "pred_p15", "realised_p15"]
    print("\nR=2 decision lift, points per week over a random start:\n")
    print(card[show].round(3).to_string(index=False))
    pooled = card[card.week == "pooled"].set_index("subset")
    print("\nboard minus implied, pooled over weeks (underpowered at this many weeks):")
    for s, r in pooled.iterrows():
        print(f"  {s:20} {r.board_minus_implied:+.3f} over {int(r.n_weeks)} weeks; "
              f"t 95% [{r.t_lo:+.3f}, {r.t_hi:+.3f}], "
              f"week-block bootstrap [{r.boot_lo:+.3f}, {r.boot_hi:+.3f}]")
    print(f"\nwrote {out_dir / f'live_{args.season}_scorecard.csv'} and {weeks_csv}")
    return 0


# ------------------------------------------------------------------ self-check

def self_check() -> int:
    """Synthetic boards and fake clocks: no git, no network, nothing written to the repo."""
    rng = np.random.default_rng(7)

    # Blob ids: git's own answer for "hello\n", and hashing is byte-exact.
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert git_blob_sha(b"hello\r\n") != git_blob_sha(b"hello\n")

    # Kickoffs are Eastern wall-clock: 13:00 is 17:00Z under daylight time and
    # 18:00Z after the clocks change; a 09:30 London game is 13:30Z.
    G = pd.DataFrame({"gameday": ["2026-09-27", "2026-12-27", "2026-10-04"],
                      "gametime": ["13:00", "13:00", "09:30"]})
    k = kickoff_utc(G)
    assert list(k) == [pd.Timestamp("2026-09-27 17:00", tz="UTC"),
                       pd.Timestamp("2026-12-27 18:00", tz="UTC"),
                       pd.Timestamp("2026-10-04 13:30", tz="UTC")], list(k)

    # Prospective means known and strictly before kickoff.
    ko = pd.Series([pd.Timestamp("2026-09-27 17:00", tz="UTC")] * 4)
    clk = pd.Series([pd.Timestamp("2026-09-22 09:09", tz="UTC"),
                     pd.Timestamp("2026-09-27 17:00", tz="UTC"),
                     pd.Timestamp("2026-10-16 10:15", tz="UTC"), pd.NaT])
    assert list(is_prospective(clk, ko)) == [True, False, False, False]

    # First public push: the earliest push whose HEAD has the bytes, whatever
    # order the log arrives in; bytes never pushed have no public time.
    t = [pd.Timestamp(f"2026-09-{d} 12:00", tz="UTC") for d in (20, 22, 30)]
    trees = {"a": {"r/b.csv": "x"}, "b": {"r/b.csv": "y"}, "c": {"r/b.csv": "y"}}
    pushes = [{"timestamp": t[2], "after": "c"}, {"timestamp": t[0], "after": "a"},
              {"timestamp": t[1], "after": "b"}]
    assert first_public(pushes, "r/b.csv", "y", trees.get) == ("b", t[1])
    assert first_public(pushes, "r/b.csv", "x", trees.get) == ("a", t[0])
    sha, when = first_public(pushes, "r/b.csv", "z", trees.get)
    assert sha is None and pd.isna(when)

    # Synthetic season: 5 weeks x 6 games x 2 kickers, integer points so ties
    # occur. Board order is perfect; the implied total is noise.
    weeks, gpw = 5, 6
    rows, kick = [], []
    for w in range(1, weeks + 1):
        for gi in range(gpw):
            gid = f"2026_{w:02d}_G{gi}"
            for side in range(2):
                pid = f"P{w}_{gi}_{side}"
                y = float(rng.integers(0, 20))
                if (w, gi, side) == (1, 0, 0):
                    y = 15.0                       # the 15+ boundary, counted as a hit
                rows.append({"player_id": pid, "game_id": gid, "season": 2026, "week": w,
                             "implied_total": float(rng.integers(17, 30)),
                             "p_ge_15": 0.1, "y_true": y})
                kick.append({"season": 2026, "week": w, "player_id": pid, "game_id": gid,
                             "season_type": "REG", "fantasy_points": y})
    B = pd.DataFrame(rows)
    B["rank_points"] = B.groupby("week").y_true.rank(ascending=False, method="first").astype(int)
    kicks = pd.DataFrame(kick)
    # Week 5's last game is not final. The game before it has a score in
    # games.csv but no kick rows yet (the play-by-play lags the score), so it
    # must read as not final too, not as two kickers who did not kick. Week 4
    # loses one kicker who did not kick, and a kicker nobody listed kicks in
    # his place.
    last, lagging = "2026_05_G5", "2026_05_G4"
    kicks = kicks[~kicks.game_id.isin([last, lagging])]
    gone = B[(B.week == 4)].player_id.iloc[0]
    sub = kicks[kicks.player_id == gone].assign(player_id="UNLISTED")
    kicks = pd.concat([kicks[kicks.player_id != gone], sub], ignore_index=True)
    games = pd.DataFrame({"game_id": B.game_id.unique()})
    games["home_score"] = np.where(games.game_id == last, np.nan, 20.0)
    games["gameday"], games["gametime"] = "2026-10-04", "13:00"

    kick_t = pd.Timestamp("2026-10-04 17:00", tz="UTC")
    before, after = kick_t - pd.Timedelta(days=2), kick_t + pd.Timedelta(days=2)
    boards = []
    for w in range(1, weeks + 1):
        boards.append({"path": pathlib.Path(f"week_2026_{w}_kickers.csv"), "week": w,
                       "frame": B[B.week == w].drop(columns="y_true"), "blob": f"blob{w}",
                       "first_commit": f"c{w}",
                       # Week 3 was committed in time but pushed late; week 5
                       # was committed after its games.
                       "committed_utc": after if w == 5 else before,
                       "public_commit": f"c{w}",
                       "public_utc": after if w in (3, 5) else before})
    R = board_rows(boards, games, kicks)
    assert (R.status == "not_final").sum() == 4 and (R.status == "did_not_kick").sum() == 1
    assert set(R.loc[R.status == "not_final", "game_id"]) == {last, lagging}
    assert R.loc[R.status == "scored", "y"].notna().all()
    assert R.groupby("week").prospective_commit.all().to_dict() == {
        1: True, 2: True, 3: True, 4: True, 5: False}
    assert R.groupby("week").prospective_public.all().to_dict() == {
        1: True, 2: True, 3: False, 4: True, 5: False}

    card = scorecard(R, kicks, n_boot=500)
    assert list(card.columns) == SCORE_COLUMNS
    c = card.set_index(["subset", "week"])
    allp = c.loc[("all", "pooled")]
    assert allp.n_weeks == 5 and allp.board_rows == 60 and allp.kickers_scored == 55
    assert allp.not_final == 4 and allp.did_not_kick == 1 and allp.unlisted_kicked == 1
    assert c.loc[("prospective_commit", "pooled")].n_weeks == 4
    assert c.loc[("prospective_public", "pooled")].n_weeks == 3
    assert ("prospective_public", 3) not in c.index and ("prospective_commit", 3) in c.index

    # A perfect board earns the oracle's lift exactly and a share of one. Its
    # Spearman is just under 1: the board breaks ties in points, Spearman
    # averages them.
    for w in range(1, 6):
        r = c.loc[("all", w)]
        assert abs(r.lift_board - r.oracle_lift) < 1e-12 and abs(r.share_board - 1) < 1e-12
        assert 0.95 < r.spearman_board <= 1.0, r.spearman_board
        assert abs(r.board_minus_implied - (r.lift_board - r.lift_implied)) < 1e-12
    # Pooled lift is the mean over weeks, as in the backtest.
    wk = card[(card.subset == "all") & (card.week != "pooled")]
    assert abs(allp.lift_board - wk.lift_board.mean()) < 1e-12
    assert abs(allp.board_minus_implied - wk.board_minus_implied.mean()) < 1e-12
    assert allp.t_lo <= allp.board_minus_implied <= allp.t_hi
    assert allp.boot_lo <= allp.board_minus_implied <= allp.boot_hi
    # Per-week rows carry no interval.
    assert wk[["boot_lo", "t_lo"]].isna().all().all()

    # Rank 1 is the favourite: the lift fed through score() matches brute force
    # over every pair in the week, starting the better-ranked of the two.
    S = R[(R.week == 2) & (R.status == "scored")]
    y, rk = S.y.to_numpy(float), S.rank_points.to_numpy()
    pairs = list(itertools.combinations(range(len(y)), 2))
    brute = np.mean([y[i] if rk[i] < rk[j] else y[j] for i, j in pairs]) - y.mean()
    assert abs(c.loc[("all", 2)].lift_board - brute) < 1e-12
    assert np.allclose(_pick_probs(-rk.astype(float), 2).sum(), 1.0)

    # A reversed board is the oracle's mirror at R=2.
    Rr = R.copy()
    Rr["rank_points"] = Rr.groupby("week").rank_points.rank(ascending=False).astype(int)
    rr = score(Rr[Rr.week == 2], kicks, pooled=False)
    assert abs(rr["lift_board"] + rr["oracle_lift"]) < 1e-12
    assert -1.0 <= rr["spearman_board"] < -0.95, rr["spearman_board"]

    # The calibration pair is a plain mean and a plain rate, and exactly 15
    # points is a 15+ week.
    assert abs(allp.pred_p15 - 0.1) < 1e-12
    Sa = R[R.status == "scored"]
    assert (Sa.y == 15).any()
    assert abs(allp.realised_p15 - (Sa.y >= 15).mean()) < 1e-12

    # Provenance table: one row per game, the contract's columns, ISO clocks.
    prov = provenance_table(R)
    assert list(prov.columns) == PROV_COLUMNS and len(prov) == weeks * gpw
    assert prov.committed_utc.str.endswith("Z").all()
    assert int(prov.final.sum()) == weeks * gpw - 2

    # A recorded clock is reused only for the same bytes, and only when the
    # live lookup has nothing. Git is switched off, and GitHub too for the
    # first call, so the bytes are hashed directly; the two files live in a
    # temporary directory outside the repository.
    with tempfile.TemporaryDirectory() as d:
        a, b = pathlib.Path(d) / "week_2026_1_kickers.csv", pathlib.Path(d) / "week_2026_2_kickers.csv"
        a.write_bytes(b"one\n")
        b.write_bytes(b"two\n")
        prior = pd.DataFrame({"board": [a.name, b.name],
                              "board_blob": [git_blob_sha(b"one\n"),       # unchanged
                                             git_blob_sha(b"edited\n")],   # since rewritten
                              "first_commit": ["c1", "c2"],
                              "committed_utc": ["2026-09-01T00:00:00Z"] * 2,
                              "public_commit": ["c1", "c2"],
                              "public_utc": ["2026-09-02T00:00:00Z"] * 2})
        got = resolve_clocks([{"path": a, "week": 1}, {"path": b, "week": 2}], prior,
                             repo=None, use_git=False)
        # With GitHub reachable its answer stands, even where it finds nothing
        # and a matching recorded clock exists: the table is only a fallback.
        # The push log and trees are stand-ins, so nothing leaves the machine.
        g, t5 = globals(), pd.Timestamp("2026-09-05", tz="UTC")
        saved = g["github_pushes"], g["github_tree_lookup"]
        g["github_pushes"] = lambda repo: [{"timestamp": t5, "after": "p1"}]
        g["github_tree_lookup"] = lambda repo: {
            "p1": {f"reports/{b.name}": git_blob_sha(b"two\n")}}.get
        try:
            live = resolve_clocks([{"path": a, "week": 1}, {"path": b, "week": 2}], prior,
                                  repo="owner/name", use_git=False)
        finally:
            g["github_pushes"], g["github_tree_lookup"] = saved
    assert live[0]["committed_utc"] == pd.Timestamp("2026-09-01", tz="UTC")
    assert live[0]["public_commit"] is None and pd.isna(live[0]["public_utc"])
    assert live[1]["public_commit"] == "p1" and live[1]["public_utc"] == t5
    assert got[0]["first_commit"] == "c1"
    assert got[0]["committed_utc"] == pd.Timestamp("2026-09-01", tz="UTC")
    assert got[0]["public_utc"] == pd.Timestamp("2026-09-02", tz="UTC")
    assert got[1]["first_commit"] is None and pd.isna(got[1]["committed_utc"])
    assert pd.isna(got[1]["public_utc"])
    print("score_boards self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score the published weekly kicker boards.")
    ap.add_argument("--self-check", action="store_true",
                    help="offline checks on synthetic boards; no git, no network, "
                         "nothing written to the repo")
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--boards", default=str(REPORTS),
                    help="directory holding week_<season>_<week>_kickers.csv")
    ap.add_argument("--kicker-games", default=str(KICKER_GAMES))
    ap.add_argument("--games", default=str(GAMES_CSV))
    ap.add_argument("--out", default=str(REPORTS), help="where the two scorecard CSVs go")
    ap.add_argument("--github-repo", default=GITHUB_REPO,
                    help="owner/name whose push log is the public clock")
    ap.add_argument("--offline", action="store_true",
                    help="skip GitHub; reuse public clocks recorded for unchanged boards")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main(a))
