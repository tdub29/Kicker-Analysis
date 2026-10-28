"""The README's three charts, drawn only from the tracked evidence CSVs in reports/.

WHY: the README argues in tables, and a reader skimming it sees numbers without
shape. Three of its claims are about shape, and each gets one chart:

  decision-lift   how much of the oracle's weekly gain each ranking method
                  captures. From reports/backtest_summary.csv.
  season-edge     the ridge's lift minus the Vegas implied total's, one column
                  per test season, with the season-level test beside it. The
                  columns come from model_bakeoff_folds.csv and the era averages
                  from model_bakeoff_slices.csv; the test, from
                  season_level_tests.csv, takes each season as one observation
                  (a one-sample t-test on the season means). A week-block
                  bootstrap would treat every week as an independent draw, and
                  with this few seasons that understates how much the edge moves
                  from one season to the next, so it is not the interval shown.
                  Seasons the ridge loses or ties are drawn below the line and
                  labelled like the rest, and the p-value is printed whether or
                  not it clears 0.05.
  reliability     a reliability diagram for the count model's P(15+), the
                  probability on the weekly board, and its P(10+): ten
                  equal-count bins of the predicted chance against how often it
                  happened, with each bin's interval, the diagonal and the
                  overall rate, plus the Brier skill against a constant at the
                  training base rate (with its interval), the calibration slope
                  and the AUC. From calibration_reliability.csv and
                  calibration_summary.csv. One pooled rate per threshold would
                  only show that the averages match, and a constant at the
                  base rate matches them too; the bins show whether the model
                  tells one kicker-week from another.

Nothing here recomputes a result. backtest.py and model_bakeoff.py write those
CSVs; this file reads them, checks them, and draws. No expected value is written
into this file, so a rerun on a newer data build redraws without edits.

Two scripts write the CSVs, and they need not have run on the same data build:
backtest.py runs in the pipeline on every rebuild, model_bakeoff.py is a separate
study run by hand. So the checks come in two kinds.

  Within one script's output every overlap must agree, or the run raises before
  anything is drawn. Every row of backtest_summary.csv implies the same oracle
  (lift / share). The bake-off's "seasons won" equals its count of positive
  per-season differences. The reliability bins hold equal counts (to within one
  row, as many bins at every threshold, since the chart names one count), add
  up to their summary row's n, mean prediction and realised rate, and each
  bin's realised rate is a whole number of hits. ratio and Brier skill follow from their own columns. The
  t-test's p-value and interval follow from its t statistic and df.

  Across the two scripts, the files are first matched on the rows both write:
  the ridge's and the implied total's pooled lift. Equal lifts mean the two runs
  saw the same data, and then the season chart's columns must agree exactly with
  the test drawn beside them (number of seasons, seasons won, mean, t). Unequal
  lifts mean different builds: nothing raises, a warning is printed, and the
  season chart's footnote says so with both lifts and both means, so the chart
  never mixes two builds without saying it.

Each chart is written twice, light and dark, from the same data, because one PNG
cannot be legible on both of GitHub's themes (a README picks between them with a
<picture> element). Colours are light and dark steps of the same hues, checked
with a palette validator in both modes (OKLab distance under simulated
protanopia and deuteranopia, contrast against the surface): the season chart's
blue and red poles pass every check in both modes, and its sign is also carried
by each column's direction and printed value, never by colour alone. The grey
for bars that are not the point is a de-emphasis grey, not a series colour, and
each of those bars carries its label and value in text. The reliability chart
names its marks in a key above the panels rather than with labels inside them,
because the bins' whiskers can cover every spot along both grey reference lines.

`--markdown` prints the README's <picture> blocks, with alt text built from the
same CSVs, so the alt text can be refreshed whenever the charts are.

Deterministic: Agg, matplotlib's bundled DejaVu Sans, and no software or date
stamp in the PNG metadata, so a rerun on the same CSVs with the same matplotlib
writes byte-identical files and git shows no diff.

    python src/kicker_analysis/figures.py               # writes docs/images/*.png
    python src/kicker_analysis/figures.py --markdown    # prints README blocks, writes nothing
    python src/kicker_analysis/figures.py --self-check  # synthetic CSVs, temp dir
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT = ROOT / "docs" / "images"

# Layout is set in inches at 100 px per inch, so the pixel specs below (bar
# thickness, corner radius, marker size) are CSS pixels; files are written at
# 2x for high-density screens.
LAYOUT_DPI = 100
DPI = 200
WIDTH_IN = 8.0
FONT = "DejaVu Sans"
MINUS = "−"

# Light and dark steps of the same hues, plus chart chrome. s1 is the series
# colour; pos/neg are the diverging blue and red poles; context is the
# de-emphasis grey for bars that are not the point, each of which also carries
# its value in text.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "s1": "#2a78d6", "pos": "#2a78d6", "neg": "#e34948",
              "context": "#a3a29b"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "s1": "#3987e5", "pos": "#3987e5", "neg": "#e66767",
             "context": "#6b6a65"},
}

# backtest_summary.csv "rank by" value -> label, in the README table's order.
# The compound model's P(15+) ranking is in the CSV but not in the README table.
METHODS = (
    ("ridge (the board)", "Ridge on 16 features, which orders the board"),
    ("compound E[y]", "Count model, expected points"),
    ("implied total only", "Vegas implied team total alone"),
    ("kicker career mean", "Kicker's own career average"),
    ("constant 8.0", "A constant 8.0"),
)
EMPHASIS = "ridge (the board)"

# The rows backtest_summary.csv ("rank by") and model_bakeoff_summary.csv ("key")
# both write. Their pooled lifts say whether the two scripts saw the same data.
SHARED_ROWS = (("ridge (the board)", "ridge"), ("implied total only", "implied"))

# Reliability panels, left to right: P(15+) is the probability on the weekly
# board (p_ge_15); P(10+) is the threshold nearest the median week.
PANELS = (15, 10)

TINY = 1e-9  # float noise: a constant's lift is stored as a few times 1e-16
COUNT = {n: w for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


# ------------------------------------------------------------------- formatting

def _zero(x: float) -> float:
    """Exact zero for float noise, so a label never reads -0.000."""
    return 0.0 if abs(x) < TINY else float(x)


def _signed(x: float, nd: int = 3) -> str:
    """+0.050 / -0.110, with a real minus sign (U+2212) for the negative."""
    s = f"{_zero(x):+.{nd}f}"
    return s.replace("-", MINUS)


def _pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def _p(p: float) -> str:
    """A p-value to two significant figures (0.095, 0.012, 0.50), three
    decimals below 0.01, and "< 0.001" below that."""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.0995 else f"{p:.2f}"


def _p_eq(p: float) -> str:
    """'p = 0.095', or 'p < 0.001' rather than 'p = < 0.001'."""
    s = _p(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


def _bins_share(k: int) -> str:
    """'a tenth' for ten equal-count bins, '1/k' otherwise."""
    return {4: "a quarter", 5: "a fifth", 10: "a tenth"}.get(k, f"1/{k}")


def _num(x: float, nd: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def _close(a: float, b: float, tol: float = TINY) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def _words(n: int) -> str:
    return COUNT.get(n, str(n))


def _and(items: list[str]) -> str:
    """'2019', '2019 and 2021', '2018, 2019 and 2021'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------- data and checks

def match_builds(reports: pathlib.Path) -> dict:
    """Do backtest.py's and model_bakeoff.py's CSVs come from the same data?

    Compared on the pooled lifts both scripts write. Not an error either way:
    the answer decides which cross-file checks apply and what the season chart
    has to disclose.
    """
    b = pd.read_csv(reports / "backtest_summary.csv").set_index("rank by")
    k = pd.read_csv(reports / "model_bakeoff_summary.csv").set_index("key")
    lifts = {}
    for bk, kk in SHARED_ROWS:
        if bk not in b.index or kk not in k.index:
            raise ValueError(f"backtest_summary.csv needs '{bk}' and "
                             f"model_bakeoff_summary.csv needs '{kk}'")
        lifts[kk] = (float(b.at[bk, "lift_r2"]), float(k.at[kk, "lift_r2"]))
    return {"same": all(_close(x, y) for x, y in lifts.values()), "lifts": lifts}


def load_lift(reports: pathlib.Path) -> tuple[pd.DataFrame, float]:
    """Lift and share of oracle per method, plus the oracle every row implies."""
    t = pd.read_csv(reports / "backtest_summary.csv").set_index("rank by")
    missing = [k for k, _ in METHODS if k not in t.index]
    if missing:
        raise ValueError(f"backtest_summary.csv has no row for {missing}")
    live = t[t.share_r2.abs() > TINY]
    implied = live.lift_r2 / live.share_r2
    oracle = float(implied.mean())
    if (implied.max() - implied.min()) > TINY * oracle:
        raise ValueError(f"rows imply different oracles: {implied.round(6).to_dict()}")
    rows = pd.DataFrame({
        "key": [k for k, _ in METHODS],
        "label": [lbl for _, lbl in METHODS],
        "lift": [_zero(t.at[k, "lift_r2"]) for k, _ in METHODS],
        "share": [_zero(t.at[k, "share_r2"]) for k, _ in METHODS],
    })
    return rows, oracle


def load_season_test(reports: pathlib.Path) -> dict:
    """The ridge-vs-implied-total row of season_level_tests.csv, checked in itself."""
    from scipy import stats

    t = pd.read_csv(reports / "season_level_tests.csv")
    rows = t[t.comparison.str.contains("ridge") & t.comparison.str.contains("implied")]
    if len(rows) != 1:
        raise ValueError("season_level_tests.csv needs exactly one ridge vs implied "
                         f"total row, has {list(rows.comparison)}")
    r = rows.iloc[0]
    n, df, won = int(r.n_seasons), int(r.df), int(r.seasons_won)
    if df != n - 1:
        raise ValueError(f"season test: df {df} is not n_seasons - 1 ({n} seasons)")
    if not 0 <= won <= n:
        raise ValueError(f"season test: {won} seasons won of {n}")
    if not _close(2 * stats.t.sf(abs(r.t_stat), df), r.t_p, 1e-8):
        raise ValueError(f"season test: t_p {r.t_p} does not follow from "
                         f"t_stat {r.t_stat} on {df} df")
    if r.t_stat != 0:
        half = stats.t.ppf(0.975, df) * r.mean_diff / r.t_stat
        if not (_close(r.t_ci_lo, r.mean_diff - half, 1e-8)
                and _close(r.t_ci_hi, r.mean_diff + half, 1e-8)):
            raise ValueError("season test: interval is not mean_diff +/- t(0.975, df) * se")
    return {"name": str(r.comparison), "mean": float(r.mean_diff), "t": float(r.t_stat),
            "df": df, "p": float(r.t_p), "lo": float(r.t_ci_lo), "hi": float(r.t_ci_hi),
            "won": won, "n": n}


def load_seasons(reports: pathlib.Path, build: dict, test: dict) -> dict:
    """Per-season ridge minus implied-total lift, era means, and the season test.

    The bake-off's own files must agree with each other. The columns must agree
    with the test only when both scripts ran on the same build (match_builds);
    otherwise `note` says what differs, in numbers, for the chart's footnote.
    """
    f = pd.read_csv(reports / "model_bakeoff_folds.csv", index_col=0)
    f.columns = [int(c) for c in f.columns]
    edge = (f.loc["ridge"] - f.loc["implied"]).rename("edge")

    s = pd.read_csv(reports / "model_bakeoff_summary.csv").set_index("key").loc["implied"]
    if int(s.seasons_won) != int((edge > 0).sum()):
        raise ValueError(f"summary says ridge won {int(s.seasons_won)} seasons, "
                         f"folds say {int((edge > 0).sum())}")

    sl = pd.read_csv(reports / "model_bakeoff_slices.csv", index_col=0)
    eras = []
    for col in sl.columns:
        m = re.fullmatch(r"(\d{4})-(\d{4})", col)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            eras.append((lo, hi, float(sl.at["ridge", col] - sl.at["implied", col])))

    note = None
    k = len(edge)
    if build["same"]:
        sd = float(edge.std(ddof=1))
        t_cols = edge.mean() / (sd / math.sqrt(k)) if sd > 0 else math.nan
        bad = []
        if test["n"] != k:
            bad.append(f"{test['n']} seasons in the test, {k} columns")
        if test["won"] != int((edge > 0).sum()):
            bad.append(f"test says ridge won {test['won']}, columns say {int((edge > 0).sum())}")
        if not _close(test["mean"], edge.mean(), 1e-8):
            bad.append(f"test mean {test['mean']:.6f}, column mean {edge.mean():.6f}")
        if np.isfinite(t_cols) and not _close(test["t"], t_cols, 1e-6):
            bad.append(f"test t {test['t']:.6f}, t over the columns {t_cols:.6f}")
        if bad:
            raise ValueError("same data build, but the season columns and "
                             "season_level_tests.csv disagree: " + "; ".join(bad))
    else:
        rb, rk = build["lifts"]["ridge"]
        note = ("The columns and era averages come from a different data build than "
                f"the test: pooled ridge lift {rk:.3f} in the bake-off, {rb:.3f}\n"
                f"in the backtest. So the {k} columns average {_signed(edge.mean())} a "
                f"week, while the test's {test['n']} season means average "
                f"{_signed(test['mean'])}.")
    return {"edge": edge, "eras": sorted(eras), "test": test, "note": note}


def season_span(build: dict, test: dict, seasons: pd.DataFrame | None = None) -> str:
    """'eight season-forward test seasons, 2018 to 2025'.

    The count is the backtest's own (season_level_tests.csv). The years come
    from the bake-off's fold table, so they are named only when both scripts saw
    the same build.
    """
    s = f"{_words(test['n'])} season-forward test seasons"
    if build["same"] and seasons is not None and len(seasons) == test["n"]:
        s += f", {min(seasons)} to {max(seasons)}"
    return s


def load_reliability(reports: pathlib.Path) -> dict:
    """Reliability bins and summary per panel threshold, checked against each other."""
    rel = pd.read_csv(reports / "calibration_reliability.csv")
    summ = pd.read_csv(reports / "calibration_summary.csv").set_index("threshold")
    if set(rel.threshold) != set(summ.index):
        raise ValueError(f"thresholds differ: reliability {sorted(set(rel.threshold))}, "
                         f"summary {sorted(summ.index)}")
    missing = [t for t in PANELS if t not in summ.index]
    if missing:
        raise ValueError(f"calibration CSVs have no threshold {missing}")
    # The subtitle, key and alt text name one bin count for every panel.
    counts = rel.groupby("threshold").size()
    if counts.nunique() != 1:
        raise ValueError(f"thresholds have different bin counts: {counts.to_dict()}")
    for thr, s in summ.iterrows():
        if not _close(s.ratio, s.pred_mean / s.realised_rate):
            raise ValueError(f"{thr}+: ratio is not pred_mean / realised_rate")
        if not _close(s.brier_skill, 1 - s.brier / s.brier_base):
            raise ValueError(f"{thr}+: brier_skill is not 1 - brier / brier_base")
        if not s.brier_skill_lo <= s.brier_skill_hi:
            raise ValueError(f"{thr}+: Brier skill interval runs backwards")
        g = rel[rel.threshold == thr].sort_values("bin")
        if list(g["bin"]) != list(range(1, len(g) + 1)):
            raise ValueError(f"{thr}+: bins are {list(g['bin'])}, not 1..{len(g)}")
        # Each bin on its own first, then the bins against the summary row.
        n = g.n.to_numpy(float)
        if n.max() - n.min() > 1:
            raise ValueError(f"{thr}+: bins hold {int(n.min())} to {int(n.max())} rows, "
                             "not equal counts")
        hits = (n * g.realised_rate).to_numpy()
        if (np.abs(hits - np.round(hits)) > 1e-6).any():
            raise ValueError(f"{thr}+: a bin's realised_rate x n is not a whole count")
        if not ((g.realised_lo <= g.realised_rate + TINY)
                & (g.realised_rate <= g.realised_hi + TINY)
                & (g.realised_lo >= 0) & (g.realised_hi <= 1)).all():
            raise ValueError(f"{thr}+: a bin's interval does not contain its rate")
        if (np.diff(g.pred_mean.to_numpy()) < -TINY).any():
            raise ValueError(f"{thr}+: bins are not in order of predicted chance")
        if int(n.sum()) != int(s.n):
            raise ValueError(f"{thr}+: bins hold {int(n.sum())} rows, summary n is {int(s.n)}")
        for col in ("pred_mean", "realised_rate"):
            if not _close((n * g[col]).sum() / n.sum(), s[col]):
                raise ValueError(f"{thr}+: bins' weighted {col} is not the summary's")
    return {thr: (rel[rel.threshold == thr].sort_values("bin").reset_index(drop=True),
                  summ.loc[thr].to_dict()) for thr in PANELS}


def load_all(reports: pathlib.Path) -> dict:
    """Every figure's data, loaded and checked, before anything is drawn.

    So a bad CSV cannot leave docs/images/ half old, half new.
    """
    build = match_builds(reports)
    test = load_season_test(reports)
    seasons = load_seasons(reports, build, test)
    rows, oracle = load_lift(reports)
    span = season_span(build, test, list(seasons["edge"].index))
    return {
        "build": build,
        "decision-lift-by-method": (rows, oracle, span),
        "ridge-vs-implied-total-by-season": (seasons,),
        "calibration-reliability": (load_reliability(reports),),
    }


# ---------------------------------------------------------------------- drawing

def _figure(theme: dict, height_in: float):
    from matplotlib.figure import Figure
    return Figure(figsize=(WIDTH_IN, height_in), dpi=LAYOUT_DPI,
                  facecolor=theme["surface"])


def _axes(fig, theme: dict, left: float, bottom: float, width: float, height: float):
    """Axes placed in inches from the bottom-left, every spine and tick off."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes((left / w, bottom / h, width / w, height / h))
    ax.set_facecolor(theme["surface"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, colors=theme["muted"], labelsize=9)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(FONT)
    return ax


def _text(fig_or_ax, x, y, s, theme, role="ink", size=9.5, **kw):
    kw.setdefault("fontfamily", FONT)
    return fig_or_ax.text(x, y, s, color=theme[role], fontsize=size, **kw)


FOOT_LINE_IN = 0.165  # one 8 pt footer line


def _header(fig, theme: dict, title: str, subtitle: str, source: str) -> None:
    """Title, subtitle and a source footer naming the CSVs, at fixed inch offsets."""
    w, h = fig.get_size_inches()
    x = 0.35 / w
    _text(fig, x, 1 - 0.42 / h, title, theme, size=13.5, weight="bold", va="baseline")
    _text(fig, x, 1 - 0.58 / h, subtitle, theme, "ink2", size=9.5, va="top",
          linespacing=1.5)
    _text(fig, x, 0.12 / h, source, theme, "muted", size=8, va="bottom",
          linespacing=1.45)


def _px(ax, n: float) -> tuple[float, float]:
    """n layout pixels expressed in data units along x and along y."""
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0.0, 0.0), (n, n)])
    return abs(x1 - x0), abs(y1 - y0)


def _bar(ax, cat: float, thick_px: float, base: float, end: float, color: str,
         horizontal: bool, radius_px: float = 4.0):
    """A bar with a rounded data end and a square baseline end.

    Drawn as a path in data units, so it has to be called after the axes'
    position and limits are final. A zero-length bar draws nothing.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if end == base:
        return None
    px_x, px_y = _px(ax, 1.0)
    pv, pc = (px_x, px_y) if horizontal else (px_y, px_x)
    thick = thick_px * pc
    s = 1.0 if end > base else -1.0
    rv = min(radius_px * pv, abs(end - base))
    rc = min(radius_px * pc, thick / 2)
    c0, c1, k = cat - thick / 2, cat + thick / 2, 0.5523
    vc = [(base, c0), (end - s * rv, c0),
          (end - s * rv + s * k * rv, c0), (end, c0 + rc - k * rc), (end, c0 + rc),
          (end, c1 - rc),
          (end, c1 - rc + k * rc), (end - s * rv + s * k * rv, c1), (end - s * rv, c1),
          (base, c1), (base, c0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CLOSEPOLY]
    xy = vc if horizontal else [(c, v) for v, c in vc]
    patch = PathPatch(Path(xy, codes), facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(patch)
    return patch


def _check_fits(fig) -> None:
    """Every piece of text lies inside the canvas and clear of every other.

    The layout is fixed in inches and the strings come partly from the CSVs, so
    a longer label would otherwise be clipped at the edge, or printed over its
    neighbour, without any error.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # axis labels are only placed during a draw
    renderer = canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    boxes = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > w + 0.5 or b.y1 > h + 0.5:
            raise ValueError(f"text runs off the figure: {t.get_text()[:60]!r} "
                             f"at x {b.x0:.0f}-{b.x1:.0f} of {w:.0f}, "
                             f"y {b.y0:.0f}-{b.y1:.0f} of {h:.0f}")
        boxes.append((t.get_text(), b.padded(-1)))  # touching is not overlapping
    for i, (s1, b1) in enumerate(boxes):
        for s2, b2 in boxes[i + 1:]:
            if b1.overlaps(b2):
                raise ValueError(f"text overlaps text: {s1[:40]!r} and {s2[:40]!r}")


def _save(fig, path: pathlib.Path, dpi: int) -> pathlib.Path:
    _check_fits(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(),
                metadata={"Software": None})
    return path


def draw_lift(rows: pd.DataFrame, oracle: float, seasons: str, theme: dict):
    """Horizontal bars, label above each, value and share of oracle at the tip."""
    n, row_in, top_in, bottom_in = len(rows), 0.58, 1.12, 0.42
    fig = _figure(theme, top_in + n * row_in + bottom_in)
    _header(fig, theme,
            "Decision lift by ranking method",
            "Points a week gained by starting whichever of two random starters "
            "the method prefers, against a coin flip.\n"
            f"{seasons[0].upper() + seasons[1:]}. The oracle, who always starts "
            f"the better kicker, gains {oracle:.2f}.",
            "Source: reports/backtest_summary.csv (lift_r2, share_r2). "
            f"Share of oracle = lift / {oracle:.3f}.")
    ax = _axes(fig, theme, 0.35, bottom_in, WIDTH_IN - 0.7, n * row_in)
    xmax = rows.lift.max() / 0.6  # the longest bar takes 60%, its label the rest
    ax.set_xlim(0, xmax)
    ax.set_ylim(n, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for i, r in rows.iterrows():
        # The zero baseline, one segment per bar slot so it never cuts a label.
        ax.vlines(0, i + 0.40, i + 0.84, color=theme["axis"], lw=1, zorder=4)
        strong = r.key == EMPHASIS
        _text(ax, 0, i + 0.33, r.label, theme, "ink" if strong else "ink2", size=10,
              weight="bold" if strong else "normal", va="bottom")
        _bar(ax, i + 0.62, 20, 0.0, r.lift, theme["s1" if strong else "context"],
             horizontal=True)
        v = ax.annotate(f"{r.lift:.3f}", (r.lift, i + 0.62), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=10, color=theme["ink"], fontfamily=FONT,
                        weight="bold" if strong else "normal")
        ax.annotate(f"  {100 * r.share:.1f}% of oracle", (1, 0.5), xycoords=v,
                    va="center", ha="left", fontsize=9, color=theme["ink2"],
                    fontfamily=FONT)
    return fig


def _season_lines(d: dict) -> tuple[str, str]:
    """Subtitle and footer of the season chart, every number from the CSVs."""
    t = d["test"]
    verdict = ("not significant at the 5% level" if t["p"] >= 0.05
               else "significant at the 5% level")
    subtitle = ("Ridge lift minus implied-total lift, points a week. Above zero, "
                "the ridge's pick gained more.\n"
                f"Season average {_signed(t['mean'])}: t-test on the {t['n']} season "
                f"means, {t['df']} df, {_p_eq(t['p'])} (95% CI {_signed(t['lo'])} "
                f"to {_signed(t['hi'])}).\n"
                f"The ridge is ahead in {t['won']} of {t['n']} seasons. Counting "
                f"each season once, the edge is {verdict}.")
    source = ("Source: columns, reports/model_bakeoff_folds.csv; era averages, "
              "model_bakeoff_slices.csv, every week weighted equally;\n"
              "season-level t-test, reports/season_level_tests.csv.")
    if d["note"]:
        source += "\n" + d["note"]
    return subtitle, source


def draw_seasons(d: dict, theme: dict):
    """Diverging columns per season, labelled, with an era bracket overhead."""
    edge, eras = d["edge"], d["eras"]
    subtitle, source = _season_lines(d)
    foot = source.count("\n") + 1
    fig = _figure(theme, 4.65 + 0.24 + foot * FOOT_LINE_IN)
    _header(fig, theme, "The ridge's edge over the Vegas implied total, by season",
            subtitle, source)
    ax = _axes(fig, theme, 0.35, 0.42 + foot * FOOT_LINE_IN, WIDTH_IN - 0.7, 2.75)
    seasons = list(edge.index)
    lo, hi = min(edge.min(), 0.0), max(edge.max(), 0.0)
    span = hi - lo
    ax.set_xlim(seasons[0] - 0.6, seasons[-1] + 0.6)
    ax.set_ylim(lo - 0.16 * span, hi + 0.34 * span)
    ax.set_yticks([])
    ax.set_xticks(seasons)
    ax.set_xticklabels([str(s) for s in seasons], fontsize=9.5, color=theme["ink2"],
                       fontfamily=FONT)
    ax.axhline(0, color=theme["axis"], lw=1, zorder=4)
    for season, v in edge.items():
        v = _zero(v)
        _bar(ax, season, 24, 0.0, v, theme["pos" if v >= 0 else "neg"],
             horizontal=False)
        ax.annotate(_signed(v), (season, v), xytext=(0, 5 if v >= 0 else -5),
                    textcoords="offset points", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9.5,
                    color=theme["ink"], fontfamily=FONT)
    # Era brackets, drawn in the band above the tallest label.
    y = hi + 0.24 * span
    tick = _px(ax, 5)[1]
    for a, b, mean in eras:
        if a not in seasons or b not in seasons:
            continue
        x0, x1 = a - 0.35, b + 0.35
        ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=theme["muted"],
                lw=1, solid_joinstyle="miter", zorder=4)
        ax.annotate(f"{a}–{b} average: {_signed(mean)} a week", ((x0 + x1) / 2, y),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9.5, color=theme["ink2"], fontfamily=FONT)
    return fig


def _ticks(lo: float, hi: float, most: int = 7) -> tuple[float, float, float]:
    """The finest round step that spans lo..hi in at most `most` ticks."""
    for step in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25):
        a = math.floor(lo / step + 1e-9) * step
        b = math.ceil(hi / step - 1e-9) * step
        if round((b - a) / step) + 1 <= most:
            return a, b, step
    return 0.0, 1.0, 0.2


def _text_width_in(fig, renderer, t) -> float:
    """Width of a drawn text in inches (layout pixels / LAYOUT_DPI)."""
    return t.get_window_extent(renderer).width / fig.dpi


def _key(fig, renderer, theme: dict, left: float, y: float, bins: int) -> None:
    """The reliability chart's key, one row at height y (inches): a bin, the
    diagonal and the flat line, each with its glyph.

    A key rather than labels inside the panels, because the bins' whiskers can
    cover every spot along both reference lines (they do in the P(15+) panel),
    and a label drawn over a whisker hides the data it is meant to explain.
    """
    w, h = fig.get_size_inches()
    ax = fig.add_axes((0, (y - 0.15) / h, 1, 0.3 / h))
    ax.set_xlim(0, w)
    ax.set_ylim(-0.15, 0.15)
    ax.set_axis_off()
    items = (("bin", f"a bin, {_bins_share(bins)} of the rows, and its 95% interval"),
             ("diagonal", "stated = happened"),
             ("flat", "overall rate (no skill)"))
    x = left
    for glyph, label in items:
        if glyph == "bin":
            ax.plot([x + 0.08, x + 0.08], [-0.1, 0.1], color=theme["s1"], lw=1.5)
            ax.plot([x + 0.08], [0], ls="none", marker="o", ms=7.5, mfc=theme["s1"],
                    mec=theme["surface"], mew=1.5)
            gw = 0.16
        elif glyph == "diagonal":
            ax.plot([x, x + 0.2], [-0.1, 0.1], color=theme["muted"], lw=1)
            gw = 0.2
        else:
            ax.plot([x, x + 0.3], [0, 0], color=theme["muted"], lw=1)
            gw = 0.3
        t = _text(ax, x + gw + 0.08, 0, label, theme, "ink2", size=9, va="center")
        x += gw + 0.08 + _text_width_in(fig, renderer, t) + 0.3


def draw_reliability(panels: dict, theme: dict):
    """One reliability diagram per threshold, side by side, stats underneath.

    Laid out from the bottom up in inches: footer, statistics table, panels,
    panel headings, key, then title and subtitle, so no block can run into the
    next whatever the CSV's numbers make the strings.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    first = panels[PANELS[0]][1]
    bins = len(panels[PANELS[0]][0])
    subtitle = ("Season-forward predicted chance of at least 15 and at least 10 "
                f"fantasy points, in {_words(bins)} equal-count bins,\n"
                "against how often it happened. Bins on the diagonal come true at "
                "the stated rate. Bins along the flat line\n"
                "would come from a forecast that could not tell one kicker from another.")
    source = (f"Source: reports/calibration_reliability.csv and calibration_summary.csv, "
              f"{int(first['n']):,} season-forward test rows.\n"
              "Whiskers: 95% Wilson intervals, which treat rows as independent and so "
              "run somewhat narrow. Brier skill: 1 − Brier / Brier\n"
              "of a constant at the training base rate, interval by week-block "
              "bootstrap; 0 is no better than that constant.\n"
              "Calibration slope: logistic fit of the outcome on logit(p); 1 is ideal, "
              "below 1 the stated chances are too spread out.\n"
              "AUC: chance a game that reached the mark was given a higher "
              "probability than one that did not; 0.5 is a coin flip.")
    side, row, n_rows = 2.85, 0.215, 5
    lefts = (0.95, 4.75)
    foot_top = 0.12 + (source.count("\n") + 1) * FOOT_LINE_IN
    table_top = foot_top + 0.22 + (n_rows - 1) * row + 0.16
    bottom = table_top + 0.62             # room for tick labels and the x label
    heading = bottom + side + 0.12
    key_y = heading + 0.2 + 0.3
    sub_lines = subtitle.count("\n") + 1
    height = key_y + 0.15 + 0.25 + 0.58 + sub_lines * 0.198

    fig = _figure(theme, height)
    _header(fig, theme, "How often the count model's P(15+) and P(10+) come true",
            subtitle, source)
    renderer = FigureCanvasAgg(fig).get_renderer()
    w, h = fig.get_size_inches()
    _key(fig, renderer, theme, 0.35, key_y, bins)
    for left, thr in zip(lefts, PANELS):
        g, s = panels[thr]
        ax = _axes(fig, theme, left, bottom, side, side)
        vals = np.r_[g.pred_mean, g.realised_lo, g.realised_hi]
        a, b, step = _ticks(float(vals.min()), float(vals.max()))
        pad = 0.04 * (b - a)
        lim = (a - pad, b + pad)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ticks = [a + i * step for i in range(int(round((b - a) / step)) + 1)]
        nd = 0 if all(abs(100 * t - round(100 * t)) < 1e-9 for t in ticks) else 1
        for setter in (ax.set_xticks, ax.set_yticks):
            setter(ticks)
        ax.set_xticklabels([_pct(t, nd) for t in ticks], fontfamily=FONT)
        ax.set_yticklabels([_pct(t, nd) for t in ticks], fontfamily=FONT)
        ax.tick_params(pad=5)
        ax.grid(True, color=theme["grid"], lw=1)
        ax.set_axisbelow(True)
        ax.set_xlabel("predicted chance", color=theme["ink2"], fontsize=9.5,
                      fontfamily=FONT, labelpad=6)
        if left == lefts[0]:
            ax.set_ylabel("how often it happened", color=theme["ink2"], fontsize=9.5,
                          fontfamily=FONT, labelpad=6)
        ax.plot(lim, lim, color=theme["muted"], lw=1, zorder=2)
        base = float(s["realised_rate"])
        ax.plot(lim, (base, base), color=theme["muted"], lw=1, zorder=2)
        ax.vlines(g.pred_mean, g.realised_lo, g.realised_hi, color=theme["s1"],
                  lw=1.5, zorder=4)
        ax.plot(g.pred_mean, g.realised_rate, ls="none", marker="o", ms=7.5,
                mfc=theme["s1"], mec=theme["surface"], mew=1.5, zorder=5)

        # Panel heading, with the P(15+) tag set after the heading's measured width.
        t = _text(fig, left / w, heading / h, f"P({thr}+ points)", theme, size=10.5,
                  weight="bold", va="baseline")
        if thr == 15:
            _text(fig, (left + _text_width_in(fig, renderer, t) + 0.08) / w, heading / h,
                  "the probability on the weekly board", theme, "ink2", size=9,
                  va="baseline")
        # Statistics as a two-column table under the panel.
        includes = s["brier_skill_lo"] <= 0 <= s["brier_skill_hi"]
        table = (("predicted, happened (mean)",
                  f"{_pct(s['pred_mean'])}, {_pct(s['realised_rate'])}"),
                 ("Brier skill vs base rate", _signed(s["brier_skill"])),
                 ("   95% interval", f"{_signed(s['brier_skill_lo'])} to "
                                    f"{_signed(s['brier_skill_hi'])}"
                                    + (", spans 0" if includes else "")),
                 ("calibration slope", _num(s["cal_slope"])),
                 ("AUC", _num(s["auc"])))
        assert len(table) == n_rows
        for i, (k, v) in enumerate(table):
            yy = (table_top - row * i) / h
            _text(fig, left / w, yy, k, theme, "ink2", size=9, va="top")
            _text(fig, (left + side) / w, yy, v, theme, "ink", size=9, va="top", ha="right")
    return fig

# name -> draw: (*data, theme) -> Figure
FIGURES = {
    "decision-lift-by-method": draw_lift,
    "ridge-vs-implied-total-by-season": draw_seasons,
    "calibration-reliability": draw_reliability,
}


def run(reports: pathlib.Path = REPORTS, out: pathlib.Path = OUT,
        dpi: int = DPI, verbose: bool = True) -> list[pathlib.Path]:
    data = load_all(reports)
    note = data["ridge-vs-implied-total-by-season"][0]["note"]
    if note and verbose:
        print("figures: WARNING, backtest.py and model_bakeoff.py CSVs are from "
              "different data builds; the season chart says so in its footnote. "
              "Rerun model_bakeoff.py on the current build to make them agree.",
              file=sys.stderr)
    figs = {f"{name}-{mode}.png": draw(*data[name], theme)
            for name, draw in FIGURES.items() for mode, theme in THEMES.items()}
    written = [_save(fig, out / fname, dpi) for fname, fig in figs.items()]
    if verbose:
        for p in written:
            print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return written


# ----------------------------------------------------------------- alt text

def alt_text(data: dict) -> dict[str, str]:
    """A screen-reader description of each chart, from the same loaded data."""
    rows, oracle, span = data["decision-lift-by-method"]
    lift = "; ".join(f"{r.label}: {r.lift:.3f} ({100 * r.share:.1f}% of the oracle)"
                     for r in rows.itertuples())
    out = {"decision-lift-by-method":
           "Horizontal bar chart of decision lift: points a week gained over a coin "
           "flip by starting whichever of two random starters each method prefers, "
           f"over {span}. {lift}. The oracle gains "
           f"{oracle:.2f}."}

    d = data["ridge-vs-implied-total-by-season"][0]
    e, t = d["edge"], d["test"]
    cols = ", ".join(f"{s} {_signed(v)}" for s, v in e.items())
    behind = [str(s) for s, v in e.items() if v <= 0]
    eras = "; ".join(f"{a} to {b} average {_signed(m)}" for a, b, m in d["eras"])
    out["ridge-vs-implied-total-by-season"] = (
        "Column chart of the ridge's lift minus the Vegas implied total's lift, "
        f"points a week, by test season: {cols}. "
        + (f"The ridge is behind or level in {_and(behind)}. " if behind else "")
        + (f"Era averages: {eras}. " if eras else "")
        + f"Season-level t-test on the {t['n']} season means: average "
        f"{_signed(t['mean'])}, {t['df']} df, {_p_eq(t['p'])}, 95% CI "
        f"{_signed(t['lo'])} to {_signed(t['hi'])}; ahead in {t['won']} of {t['n']} "
        "seasons." + (" " + d["note"].replace("\n", " ") if d["note"] else ""))

    parts, bins = [], 0
    for thr, (g, s) in data["calibration-reliability"][0].items():
        bins = len(g)
        parts.append(
            f"P({thr}+): bins run from {_pct(g.pred_mean.min())} to "
            f"{_pct(g.pred_mean.max())} predicted and from {_pct(g.realised_rate.min())} "
            f"to {_pct(g.realised_rate.max())} realised; mean predicted "
            f"{_pct(s['pred_mean'])}, realised {_pct(s['realised_rate'])}; Brier skill "
            f"over the base rate {_signed(s['brier_skill'])} (95% CI "
            f"{_signed(s['brier_skill_lo'])} to {_signed(s['brier_skill_hi'])}), "
            f"calibration slope {_num(s['cal_slope'])}, AUC {_num(s['auc'])}.")
    out["calibration-reliability"] = (
        "Two reliability diagrams for the count model, predicted chance against how "
        f"often it happened in {_words(bins)} equal-count bins with 95% intervals, the diagonal "
        "and the overall rate. " + " ".join(parts) + " Brier skill 0 is no better "
        "than a constant at the base rate, a calibration slope below 1 means the "
        "stated chances are too spread out, and an AUC of 0.5 is a coin flip.")
    return out


def markdown(data: dict, prefix: str = "docs/images") -> str:
    """The README's <picture> block for each chart, light by default, dark on request."""
    blocks = []
    for name, alt in alt_text(data).items():
        alt = alt.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        blocks.append(
            "<picture>\n"
            f'  <source media="(prefers-color-scheme: dark)" srcset="{prefix}/{name}-dark.png">\n'
            f'  <img src="{prefix}/{name}-light.png" width="800" alt="{alt}">\n'
            "</picture>")
    return "\n\n".join(blocks)


# -------------------------------------------------------------------- self-check

def _synthetic(d: pathlib.Path, scale: float = 1.0) -> None:
    """Eight small CSVs with the real files' shapes and internally consistent values.

    `scale` stretches every lift and moves every probability, standing in for a
    rerun on a newer data build: nothing here may depend on one set of values.
    """
    from scipy import stats

    oracle = 2.5 * scale
    lifts = {"ridge (the board)": 0.45, "compound E[y]": 0.40,
             "compound P(y>=15)": 0.39, "kicker career mean": 0.20,
             "implied total only": 0.35, "constant 8.0": -2.8e-16}
    lifts = {k: v * scale for k, v in lifts.items()}
    pd.DataFrame({"rank by": list(lifts), "lift_r2": list(lifts.values()),
                  "share_r2": [v / oracle for v in lifts.values()]}
                 ).to_csv(d / "backtest_summary.csv", index=False)
    seasons = list(range(2018, 2026))
    ridge = np.array([0.40, 0.40, 0.50, 0.40, 0.20, 0.40, 0.60, 0.50]) * scale
    line = np.array([0.35, 0.50, 0.40, 0.40, 0.15, 0.20, 0.45, 0.30]) * scale  # loses one, ties one
    pd.DataFrame([ridge, line], index=["ridge", "implied"],
                 columns=[str(s) for s in seasons]).to_csv(d / "model_bakeoff_folds.csv")
    pd.DataFrame([[ridge[:4].mean(), ridge[4:].mean()], [line[:4].mean(), line[4:].mean()]],
                 index=["ridge", "implied"], columns=["2018-2021", "2022-2025"]
                 ).to_csv(d / "model_bakeoff_slices.csv")
    pd.DataFrame({"key": ["ridge", "implied"],
                  "lift_r2": [lifts["ridge (the board)"], lifts["implied total only"]],
                  "seasons_won": [None, int(((ridge - line) > 0).sum())]}
                 ).to_csv(d / "model_bakeoff_summary.csv", index=False)

    e = ridge - line
    k = len(e)
    tt = stats.ttest_1samp(e, 0.0)
    half = stats.t.ppf(0.975, k - 1) * e.std(ddof=1) / np.sqrt(k)
    pd.DataFrame([{"comparison": "ridge vs implied total", "mean_diff": e.mean(),
                   "t_stat": tt.statistic, "df": k - 1, "t_p": tt.pvalue,
                   "t_ci_lo": e.mean() - half, "t_ci_hi": e.mean() + half,
                   "wilcoxon_p": 0.1, "seasons_won": int((e > 0).sum()), "n_seasons": k,
                   "sign_p": 0.3, "cluster_boot_lo": 0.0, "cluster_boot_hi": 0.1,
                   "cluster_boot_p": 0.05},
                  {"comparison": "compound E[y] vs implied total", "mean_diff": 0.0,
                   "t_stat": 0.0, "df": k - 1, "t_p": 1.0, "t_ci_lo": -0.1, "t_ci_hi": 0.1,
                   "wilcoxon_p": 1.0, "seasons_won": 4, "n_seasons": k, "sign_p": 1.0,
                   "cluster_boot_lo": -0.1, "cluster_boot_hi": 0.1, "cluster_boot_p": 1.0}]
                 ).to_csv(d / "season_level_tests.csv", index=False)

    rel, summ = [], []
    for thr, centre in ((8, 0.52), (10, 0.36), (12, 0.22), (15, 0.08)):
        p = centre * scale ** 0.5 * np.linspace(0.6, 1.5, 10)
        n = np.array([101] + [100] * 9)
        hits = np.round(n * (centre * scale ** 0.5 + 0.5 * (p - centre * scale ** 0.5)))
        rate = hits / n
        rel.append(pd.DataFrame({"threshold": thr, "bin": range(1, 11), "n": n,
                                 "pred_mean": p, "realised_rate": rate,
                                 "realised_lo": np.clip(rate - 0.05, 0, 1),
                                 "realised_hi": np.clip(rate + 0.05, 0, 1)}))
        pm, rr = (n * p).sum() / n.sum(), hits.sum() / n.sum()
        brier, base = 0.20, 0.2004
        summ.append({"threshold": thr, "pred_mean": pm, "realised_rate": rr,
                     "ratio": pm / rr, "brier": brier, "brier_base": base,
                     "brier_skill": 1 - brier / base, "brier_skill_lo": -0.004,
                     "brier_skill_hi": 0.006, "cal_slope": 0.55, "auc": 0.55,
                     "n": int(n.sum())})
    pd.concat(rel).to_csv(d / "calibration_reliability.csv", index=False)
    pd.DataFrame(summ).to_csv(d / "calibration_summary.csv", index=False)


def self_check() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        _synthetic(d)

        # Data: what the charts plot is what the CSVs say, signs and all.
        data = load_all(d)
        assert data["build"]["same"], data["build"]
        rows, oracle, span = data["decision-lift-by-method"]
        assert abs(oracle - 2.5) < 1e-12, oracle
        assert span == "eight season-forward test seasons, 2018 to 2025", span
        assert list(rows.key) == [k for k, _ in METHODS]  # README order, P(15+) out
        assert rows.set_index("key").at["constant 8.0", "lift"] == 0.0
        assert _signed(-2.8e-16) == "+0.000" and _signed(-0.11) == MINUS + "0.110"
        assert [_p(x) for x in (0.0946, 0.0119, 0.0042, 0.0004, 0.0997, 0.5)] == \
            ["0.095", "0.012", "0.004", "< 0.001", "0.10", "0.50"]
        assert _p_eq(0.0004) == "p < 0.001" and _p_eq(0.0946) == "p = 0.095"
        sd = data["ridge-vs-implied-total-by-season"][0]
        edge, test = sd["edge"], sd["test"]
        assert list(edge.index) == list(range(2018, 2026))
        assert abs(edge[2019] + 0.10) < 1e-12 and abs(edge[2021]) < 1e-12
        assert test["won"] == 6 and test["n"] == 8 and test["df"] == 7 and sd["note"] is None
        assert abs(test["mean"] - edge.mean()) < 1e-12
        assert [(a, b) for a, b, _ in sd["eras"]] == [(2018, 2021), (2022, 2025)]
        sub, _ = _season_lines(sd)
        assert _p_eq(test["p"]) in sub and "6 of 8" in sub, sub
        panels = data["calibration-reliability"][0]
        assert list(panels) == list(PANELS) and len(panels[15][0]) == 10
        assert _ticks(0.029, 0.132) == (0.02, 0.14, 0.02)
        assert _ticks(0.2166, 0.4825) == (0.2, 0.5, 0.05)
        alt = alt_text(data)
        assert set(alt) == set(FIGURES)
        assert "behind or level in 2019 and 2021." in alt["ridge-vs-implied-total-by-season"]
        assert _and(["a"]) == "a" and _and(["a", "b", "c"]) == "a, b and c"
        assert "Vegas implied team total alone: " in alt["decision-lift-by-method"]
        assert "calibration slope 0.55" in alt["calibration-reliability"]
        assert "in ten equal-count bins" in alt["calibration-reliability"]
        md = markdown(data)
        assert md.count("<picture>") == len(FIGURES) and '"' not in alt["calibration-reliability"]

        # A rounded bar spans exactly base..end, either direction, and a
        # zero-length bar draws nothing.
        fig = _figure(THEMES["light"], 3.0)
        ax = _axes(fig, THEMES["light"], 0.5, 0.5, 5.0, 2.0)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 1)
        for base, end in ((0.0, 0.5), (0.0, -0.4), (0.0, 0.004)):
            p = _bar(ax, 5.0, 24, base, end, "#000000", horizontal=False)
            ext = p.get_path().get_extents()
            assert abs(ext.y0 - min(base, end)) < 1e-12, (ext, base, end)
            assert abs(ext.y1 - max(base, end)) < 1e-12, (ext, base, end)
            assert abs(ext.width - 24 * _px(ax, 1)[0]) < 1e-9, ext
        assert _bar(ax, 5.0, 24, 0.0, 0.0, "#000000", horizontal=False) is None
        # Text that runs past the edge, or over other text, is an error, not a
        # silently clipped or unreadable label.
        _check_fits(fig)
        for x, s, why in ((0.9, "a label far too long to fit " * 3, "runs off"),
                          (0.3, "two labels", "overlaps")):
            t = _text(fig, x, 0.5, s, THEMES["light"])
            u = _text(fig, 0.31, 0.5, s, THEMES["light"]) if why == "overlaps" else None
            try:
                _check_fits(fig)
            except ValueError as e:
                assert why in str(e), (why, str(e))
            else:
                raise AssertionError(f"a label that {why} was not caught")
            t.remove()
            if u is not None:
                u.remove()

        # The reliability chart names its marks in a key, P(15+) panel first.
        fig = draw_reliability(panels, THEMES["light"])
        texts = [t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text"))]
        for want in ("stated = happened", "overall rate (no skill)", "P(15+ points)",
                     "a bin, a tenth of the rows", "in ten equal-count bins",
                     "the probability on the weekly board", ", spans 0"):
            assert any(want in t for t in texts), want

        # Every chart, both themes, written as a PNG.
        out = d / "img"
        files = run(d, out, dpi=30, verbose=False)
        assert len(files) == 2 * len(FIGURES)
        for f in files:
            assert f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f
            assert f.stat().st_size > 2000, (f, f.stat().st_size)
        # Deterministic: a rerun is byte-identical, so git sees no churn.
        again = _save(draw_seasons(*load_all(d)["ridge-vs-implied-total-by-season"],
                                   THEMES["dark"]), d / "again.png", 30)
        assert again.read_bytes() == (out / "ridge-vs-implied-total-by-season-dark.png"
                                      ).read_bytes()

        # Disagreements inside one script's output raise instead of drawing.
        def must_fail(csv, row, col, value, why):
            src = (d / csv).read_text()
            t = pd.read_csv(d / csv)
            t.loc[row, col] = value
            t.to_csv(d / csv, index=False)
            try:
                load_all(d)
            except ValueError as e:
                assert why in str(e), (why, str(e))  # failed for the right reason
            else:
                raise AssertionError(f"{csv} with {col}[{row}] = {value} was accepted")
            finally:
                (d / csv).write_text(src)

        must_fail("backtest_summary.csv", 1, "share_r2", 0.17, "different oracles")
        must_fail("model_bakeoff_summary.csv", 1, "seasons_won", 7, "won 7 seasons")
        must_fail("season_level_tests.csv", 0, "df", 6, "not n_seasons - 1")
        must_fail("season_level_tests.csv", 0, "t_p", 0.5, "does not follow")
        must_fail("season_level_tests.csv", 0, "t_ci_hi", 9.0, "interval is not")
        must_fail("calibration_summary.csv", 3, "n", 999, "summary n is 999")
        must_fail("calibration_reliability.csv", 35, "pred_mean", 0.09, "weighted pred_mean")
        must_fail("calibration_summary.csv", 3, "brier_skill", 0.5, "brier_skill is not")
        must_fail("calibration_reliability.csv", 35, "realised_rate", 0.123, "whole count")
        must_fail("calibration_reliability.csv", 35, "realised_lo", 0.9, "does not contain")
        must_fail("calibration_reliability.csv", 35, "n", 150, "not equal counts")
        # Same build, so the columns and the test must agree: one season's sign.
        must_fail("season_level_tests.csv", 0, "seasons_won", 5, "test says ridge won 5")

        # A different build behind the backtest's files is disclosed, not raised.
        b = pd.read_csv(d / "backtest_summary.csv")
        b.loc[0, ["lift_r2", "share_r2"]] = [0.44, 0.44 / 2.5]
        b.to_csv(d / "backtest_summary.csv", index=False)
        mixed = load_all(d)
        note = mixed["ridge-vs-implied-total-by-season"][0]["note"]
        assert not mixed["build"]["same"] and "0.450 in the bake-off, 0.440" in note, note
        assert mixed["decision-lift-by-method"][2] == "eight season-forward test seasons"
        fig = draw_seasons(*mixed["ridge-vs-implied-total-by-season"], THEMES["light"])
        _check_fits(fig)
        assert any("different data build" in t.get_text() for t in fig.texts)

        # A rerun of both scripts on other data: every check still holds.
        _synthetic(d, scale=1.3)
        newer = load_all(d)
        assert newer["build"]["same"]
        assert abs(newer["decision-lift-by-method"][1] - 2.5 * 1.3) < 1e-12
        _check_fits(draw_reliability(*newer["calibration-reliability"], THEMES["dark"]))
    print("figures self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--markdown", action="store_true",
                    help="print the README's <picture> blocks with alt text; writes nothing")
    a = ap.parse_args()
    if a.self_check:
        sys.exit(self_check())
    if a.markdown:
        print(markdown(load_all(REPORTS)))
        sys.exit(0)
    run()
