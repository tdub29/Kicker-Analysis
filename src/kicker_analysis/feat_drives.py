"""Drive-level offensive shape, per team-game, lagged. 2015 onward.

A kicker does not choose to attempt field goals. His offence hands him one by
FAILING, in a specific place: it crosses midfield, stalls somewhere inside the
opponent's 40, and the kicking team sends him out. Everything in the existing
feature set that gestures at this is a box-score total (`team_score`, `fg_att`)
or a fourth-down decision rate, and neither one sees the drive.

Why "reached field-goal range and did not score a touchdown" is closer to the
mechanism than "points scored":

  Points scored is the SUM of two opposing forces. A drive that ends in a
  touchdown is a good offence and a dead week for the kicker; a drive that ends
  on the opponent's 25 is a bad offence and three points for him. Team points
  therefore moves the kicker's expected volume in both directions at once and
  the net is close to flat, which is why `t_team_score_*` is a weak feature.
  Splitting the two apart recovers the signal: the numerator counts only the
  drives that actually generate a kick opportunity, and the denominator is the
  offence's total traffic. 2019 Tampa Bay and 2019 Baltimore both averaged about
  the same drives per game; Tampa stalled in range on 30% of drives and Baltimore
  on 21%, because Baltimore finished. Same "good offence", very different kicker.

The seven per-team-game quantities, and what each is for:

  drives            raw opportunity count. Pace and turnovers move it by two or
                    three a game, and every rate below is conditioned on it.
  reach rate        share of drives that get to the opponent 40 or closer. This
                    is the offence's ability to travel, independent of finishing.
  stall rate        of the drives that reached range, the share that ended
                    without a touchdown. The direct field-goal-chance rate, and
                    the column this module exists for.
  points per drive  efficiency, kept as the control: if stall rate is only
                    proxying "bad offence", this absorbs it and stall rate loses
                    its importance. That is the test, not a decoration.
  yards per drive   measured as PENETRATION, start yardline minus the closest
                    the drive ever got to the end zone. Net yardage would credit
                    a drive that gained 40 and then lost 12 on a sack the same
                    as one that never got there, and the kicker only cares about
                    the closest point, which is where he kicks from.
  three-and-out     drives that punted after three or fewer scrimmage plays. A
                    separate failure mode from stalling in range: it costs the
                    kicker an attempt rather than creating one.
  start field pos   mean starting yards from the offence's own goal line. Short
                    fields turn into field goals at a much higher rate than long
                    ones, and it is a defence/special-teams property the offence
                    inherits.

Leakage: nothing here is knowable before kickoff, so NOTHING is exported raw.
Each quantity is shift(1)-ed within the team across its own game sequence and
only then rolled, exactly as `features._lagged` does. The two headline rates are
rolled as numerator and denominator SEPARATELY and divided afterwards, so a
window is a properly weighted rate rather than a mean of per-game rates, which
would let one 4-drive game count as much as one 14-drive game.

Verified in `self_check()`: a team's first game has every history column null
rather than zero, a 4-game window is the four games strictly before, the stall
rate divides by drives-that-reached rather than by all drives, and OAK collapses
to LV before any grouping.

Usage:
    python src/kicker_analysis/feat_drives.py               # writes the table
    python src/kicker_analysis/feat_drives.py --self-check
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed" / "drives.parquet"

# Same floor as fourth_down.py: 2015 is where the pbp drive columns and the
# post-2015 extra point both start.
FIRST_SEASON = 2015

# "Field-goal range" for a DRIVE is looser than the fourth-down module's 38-yard
# line, deliberately. There the question was whether a specific kick was on the
# menu; here it is whether the drive got close enough that stalling produces an
# attempt at all. The opponent 40 is a 57-yard try, which is the outer edge of
# where a coach will send the unit out, and it is the conventional line.
FG_RANGE_YARDLINE = 40

RELOCATIONS = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

JOIN_KEYS = ["season", "week", "season_type", "team"]

# Windows mirror features.WINDOWS so the two families are comparable.
WINDOWS = (4, 8)

PBP_COLS = ["season", "week", "season_type", "game_id", "posteam", "fixed_drive",
            "fixed_drive_result", "yardline_100", "play_type", "play_id"]

# Counts that get lagged and rolled. Rates are formed from these AFTER rolling.
_COUNTS = ["dr_n", "dr_reach", "dr_stall", "dr_points", "dr_pen", "dr_3out",
           "dr_start_sum"]

# What the ratios are built from: (output stem, numerator, denominator).
_RATIOS = [("drives_per_game", "dr_n", "_games"),
           ("reach_rate", "dr_reach", "dr_n"),
           ("stall_rate", "dr_stall", "dr_reach"),
           ("pts_per_drive", "dr_points", "dr_n"),
           ("pen_per_drive", "dr_pen", "dr_n"),
           ("three_out_rate", "dr_3out", "dr_n"),
           ("start_own_yl", "dr_start_sum", "dr_n")]

FEATURE_COLS = [f"dv_{stem}_{suf}"
                for stem, _, _ in _RATIOS
                for suf in [f"l{w}" for w in WINDOWS] + ["career"]]


def drive_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per offensive drive, with the properties a kicker's week depends on.

    `fixed_drive` is nflverse's repaired drive counter and is the only one that is
    100% populated back to 2015 (`drive` is 98.9%); it numbers drives across the
    whole game, so (game_id, fixed_drive) is the unit. A handful of drives per
    season carry two different `posteam` values after a muffed punt or a fumble
    return, so the drive is attributed to its FIRST offence and plays run by the
    other team are dropped before any yardline is measured -- otherwise a pick-six
    return would register as that offence reaching the 5-yard line.

    The kickoff is the trap here. nflverse files the kickoff INSIDE the receiving
    team's drive with `posteam` already flipped to the receiver and
    `yardline_100` = 35, the spot of the kick. Left in, every drive "reaches the
    opponent 35" before a snap is taken: the first cut of this module reported a
    74% reach rate and a mean start at the offence's own 46, against true values
    near 47% and the own 28. Only plays from scrimmage are allowed to set a
    yardline, which fixes both.
    """
    d = pbp[pbp.fixed_drive.notna()].copy()
    # Punts and field goals stay: they are snapped from a real spot and are
    # usually the closest the drive ever gets. no_play (penalty) rows and extra
    # points do not move the ball and are dropped.
    d = d[d.play_type.isin(["run", "pass", "punt", "field_goal",
                            "qb_kneel", "qb_spike"])]
    d = d.sort_values(["game_id", "fixed_drive", "play_id"])
    key = ["game_id", "fixed_drive"]

    owner = d[d.posteam.notna()].groupby(key, sort=False).posteam.first().rename("team")
    d = d.merge(owner, left_on=key, right_index=True, how="inner")
    d = d[d.posteam == d.team]

    # A scrimmage play is what makes a drive a drive. Kneels and spikes are
    # excluded from the play count so a two-kneel end-of-half possession is not a
    # "three-and-out", and drives with none are dropped whole.
    d["is_scrim"] = d.play_type.isin(["run", "pass"]).astype(int)

    g = d.groupby(key, sort=False)
    out = pd.DataFrame({
        "season": g.season.first(), "week": g.week.first(),
        "season_type": g.season_type.first(), "team": g.team.first(),
        "result": g.fixed_drive_result.first(),
        "n_scrim": g.is_scrim.sum(),
        "start_yl": g.yardline_100.first(),
        "best_yl": g.yardline_100.min(),
    }).reset_index(drop=True)
    out = out[out.n_scrim > 0]
    out["team"] = out.team.replace(RELOCATIONS)

    out["reached"] = (out.best_yl <= FG_RANGE_YARDLINE).astype(int)
    out["td"] = (out.result == "Touchdown").astype(int)
    # The whole point of the module: got there, did not finish.
    out["stalled_in_range"] = (out.reached & (1 - out.td)).astype(int)
    out["three_out"] = ((out.result == "Punt") & (out.n_scrim <= 3)).astype(int)
    # Touchdown is scored at 7 rather than 6 because the kicker's extra point is
    # part of the drive's value to him, and a missed PAT is rare enough that the
    # bias is under 0.02 points per drive.
    out["points"] = 7 * out.td + 3 * (out.result == "Field goal")
    # Penetration, not net yardage: how far toward the end zone the drive ever
    # got. Negative is impossible by construction since best_yl <= start_yl.
    out["pen"] = out.start_yl - out.best_yl
    return out


def team_game_drives(drives: pd.DataFrame) -> pd.DataFrame:
    """Collapse drives to one row per team-game, as raw counts only."""
    g = drives.groupby(JOIN_KEYS, dropna=False)
    out = g.agg(dr_n=("reached", "size"),
                dr_reach=("reached", "sum"),
                dr_stall=("stalled_in_range", "sum"),
                dr_points=("points", "sum"),
                dr_pen=("pen", "sum"),
                dr_3out=("three_out", "sum"),
                # Yards from the team's OWN goal line, so bigger is a better
                # start. yardline_100 is distance to the opponent's end zone.
                dr_start_sum=("start_yl", lambda s: float((100 - s).sum()))
                ).reset_index()
    return out


def lag(tg: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw team-game counts into pre-kickoff history features.

    shift(1) is applied to the raw count FIRST and the window runs on the shifted
    series, which is the only ordering that is also correct at the start of a
    group. Rates are then formed from rolled sums, so `stall_rate_l4` is
    (stalls over the last four games) / (drives that reached range over the last
    four games) and not the average of four per-game fractions.
    """
    # POST weeks continue the REG numbering in nflverse, so (season, week) is the
    # true chronological order of a team's games.
    d = tg.sort_values(["team", "season", "week"]).reset_index(drop=True)
    d["_games"] = 1.0

    cols = _COUNTS + ["_games"]
    g = d.groupby("team", sort=False)
    rolled: dict[str, pd.Series] = {}
    for c in cols:
        prior = g[c].shift(1)
        by = prior.groupby(d["team"], sort=False)
        for w in WINDOWS:
            rolled[f"{c}_l{w}"] = by.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).sum())
        rolled[f"{c}_career"] = by.transform(lambda s: s.expanding().sum())

    out = {}
    for suf in [f"l{w}" for w in WINDOWS] + ["career"]:
        for stem, num, den in _RATIOS:
            n = rolled[f"{num}_{suf}"]
            q = rolled[f"{den}_{suf}"].replace(0, np.nan)
            out[f"dv_{stem}_{suf}"] = n / q
    res = pd.concat([d[JOIN_KEYS], pd.DataFrame(out, index=d.index)], axis=1)
    return res


def build(first_season: int = FIRST_SEASON) -> pd.DataFrame:
    frames = []
    for path in sorted(CACHE.glob("pbp_*.parquet")):
        season = int(path.stem.split("_")[1])
        if season < first_season:
            continue
        frames.append(team_game_drives(
            drive_table(pd.read_parquet(path, columns=PBP_COLS))))
    if not frames:
        raise SystemExit(f"no pbp parquet files in {CACHE}")
    tg = pd.concat(frames, ignore_index=True)
    df = lag(tg).sort_values(["season", "week", "team"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    return df


def main() -> int:
    tg_preview = None
    df = build()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} team-games, "
          f"{df.season.min()}-{df.season.max()}, {len(FEATURE_COLS)} features")
    # Sanity: the raw, UNLAGGED league rates, printed so the definitions can be
    # checked against common knowledge rather than taken on trust.
    frames = [team_game_drives(drive_table(pd.read_parquet(p, columns=PBP_COLS)))
              for p in sorted(CACHE.glob("pbp_*.parquet"))
              if int(p.stem.split("_")[1]) >= FIRST_SEASON]
    tg_preview = pd.concat(frames, ignore_index=True)
    t = tg_preview
    print(f"  drives per team-game: {t.dr_n.mean():.2f}")
    print(f"  reach rate (opp 40 or closer): {t.dr_reach.sum() / t.dr_n.sum():.3f}")
    print(f"  stall rate given reached: {t.dr_stall.sum() / t.dr_reach.sum():.3f}")
    print(f"  points per drive: {t.dr_points.sum() / t.dr_n.sum():.2f}")
    print(f"  three-and-out rate: {t.dr_3out.sum() / t.dr_n.sum():.3f}")
    print(f"  mean start (own yardline): {t.dr_start_sum.sum() / t.dr_n.sum():.1f}")
    yr = t.groupby("season").apply(
        lambda g: pd.Series({"drives": g.dr_n.sum() / len(g),
                             "reach": g.dr_reach.sum() / g.dr_n.sum(),
                             "stall": g.dr_stall.sum() / g.dr_reach.sum(),
                             "ppd": g.dr_points.sum() / g.dr_n.sum()}),
        include_groups=False)
    print("\n  season  drives  reach  stall   ppd")
    for s, r in yr.iterrows():
        print(f"    {s}   {r.drives:5.2f}  {r.reach:.3f}  {r.stall:.3f}  {r.ppd:.2f}")
    print(f"\n  null rate in first-game rows (must be 1.0): "
          f"{df.groupby('team').head(1)[FEATURE_COLS].isna().mean().mean():.3f}")
    return 0


def self_check() -> int:
    """Drive classification and the lag, on frames small enough to verify by eye."""
    # --- drive classification -------------------------------------------------
    def play(drive, team, yl, pt="pass", res="Punt", pid=0):
        return dict(season=2024, week=1, season_type="REG", game_id="g",
                    posteam=team, fixed_drive=drive, fixed_drive_result=res,
                    yardline_100=yl, play_type=pt, play_id=pid)

    rows = [
        # Drive 1: starts own 25, gets to the opponent 32, punts. Reached range,
        # no touchdown -> a stall in range, and 4 scrimmage plays so not a 3-out.
        # The leading kickoff row is nflverse's, posteam already flipped to the
        # receiving team and spotted at the 35. It must not set the start or the
        # penetration; if it does, drive 2 below turns into a false "reached".
        play(1, "KC", 35, pt="kickoff", pid=0),
        play(1, "KC", 75, pid=1), play(1, "KC", 60, pid=2),
        play(1, "KC", 45, pid=3), play(1, "KC", 32, pid=4),
        # Drive 2: three plays then a punt from the own 30. Reached nothing.
        play(2, "KC", 80, pid=5), play(2, "KC", 78, pid=6), play(2, "KC", 70, pid=7),
        # Drive 3: reached the 5 and scored. Reached range but NOT a stall.
        play(3, "KC", 70, res="Touchdown", pid=8),
        play(3, "KC", 5, res="Touchdown", pid=9),
        # Drive 4: two kneels. No scrimmage plays, so the whole drive is dropped
        # and cannot masquerade as a three-and-out.
        play(4, "KC", 60, pt="qb_kneel", res="End of half", pid=10),
        play(4, "KC", 61, pt="qb_kneel", res="End of half", pid=11),
        # Drive 5: KC pass intercepted and returned to KC's 3. The return play
        # belongs to BAL and must not count as KC reaching the opponent 3.
        play(5, "KC", 80, res="Turnover", pid=12),
        {**play(5, "BAL", 3, res="Turnover", pid=13)},
    ]
    dv = drive_table(pd.DataFrame(rows))
    assert len(dv) == 4, dv[["result", "n_scrim"]].to_dict("list")
    dv = dv.set_index(dv.index if "drive" not in dv else "drive").reset_index(drop=True)
    d1, d2, d3, d5 = (dv.iloc[0], dv.iloc[1], dv.iloc[2], dv.iloc[3])
    assert d1.reached == 1 and d1.td == 0 and d1.stalled_in_range == 1
    assert d1.three_out == 0 and d1.pen == 75 - 32, d1.pen
    assert d2.reached == 0 and d2.three_out == 1 and d2.stalled_in_range == 0
    # Reached range and finished: counts in the denominator, not the numerator.
    assert d3.reached == 1 and d3.td == 1 and d3.stalled_in_range == 0
    assert d3.points == 7
    # The interception return is BAL's play; KC's drive never left its own 20.
    assert d5.best_yl == 80, d5.best_yl
    assert d5.reached == 0

    tg = team_game_drives(dv).iloc[0]
    assert tg.dr_n == 4 and tg.dr_reach == 2 and tg.dr_stall == 1, tg.to_dict()
    # Start field position is yards from the team's OWN goal: 25, 20, 30, 20.
    assert tg.dr_start_sum == (25 + 20 + 30 + 20), tg.dr_start_sum

    # --- lagging --------------------------------------------------------------
    # Five KC games with 10 drives each and a stall count that climbs 1..5, plus
    # a separate team whose history must not bleed across.
    t = pd.DataFrame({
        "season": [2024] * 5 + [2024] * 2, "week": [1, 2, 3, 4, 5, 1, 2],
        "season_type": ["REG"] * 7, "team": ["KC"] * 5 + ["BAL"] * 2,
        "dr_n": [10.0] * 7, "dr_reach": [10.0] * 7,
        "dr_stall": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 9.0],
        "dr_points": [20.0] * 7, "dr_pen": [300.0] * 7, "dr_3out": [2.0] * 7,
        "dr_start_sum": [250.0] * 7,
    })
    f = lag(t)
    kc = f[f.team == "KC"].sort_values("week").reset_index(drop=True)
    # Game 1 has no prior game: undefined, not zero. A zero here would tell the
    # model that every kicker's debut offence never stalls.
    assert kc.dv_stall_rate_career.isna().iloc[0], kc.dv_stall_rate_career.iloc[0]
    assert kc.dv_drives_per_game_l4.isna().iloc[0]
    # Game 3 sees games 1-2 only: (1+2)/(10+10) = 0.15.
    assert abs(kc.dv_stall_rate_career.iloc[2] - 0.15) < 1e-12, kc.dv_stall_rate_career.iloc[2]
    # Game 5's 4-game window is games 1-4: (1+2+3+4)/40 = 0.25. Windowing before
    # shifting would give (2+3+4+5)/40 = 0.35 and the row would see itself.
    assert abs(kc.dv_stall_rate_l4.iloc[4] - 0.25) < 1e-12, kc.dv_stall_rate_l4.iloc[4]
    assert abs(kc.dv_drives_per_game_l4.iloc[4] - 10.0) < 1e-12
    # BAL's first row is still null despite KC's five games sitting above it.
    bal = f[f.team == "BAL"].sort_values("week").reset_index(drop=True)
    assert bal.dv_stall_rate_career.isna().iloc[0]
    assert abs(bal.dv_stall_rate_career.iloc[1] - 0.9) < 1e-12

    # stall_rate divides by drives that REACHED, not by all drives: halve the
    # reach count and the rate must double.
    t2 = t.assign(dr_reach=5.0)
    k2 = lag(t2)
    k2 = k2[k2.team == "KC"].sort_values("week").reset_index(drop=True)
    assert abs(k2.dv_stall_rate_l4.iloc[4] - 0.5) < 1e-12, k2.dv_stall_rate_l4.iloc[4]
    # ... while reach_rate, which divides by all drives, halves.
    assert abs(k2.dv_reach_rate_l4.iloc[4] - 0.5) < 1e-12

    # Relocated franchises collapse before grouping, or a team's history splits.
    d3f = drive_table(pd.DataFrame([play(1, "OAK", 75, pid=1), play(1, "OAK", 30, pid=2)]))
    assert d3f.iloc[0].team == "LV", d3f.iloc[0].team

    assert set(FEATURE_COLS) <= set(f.columns)
    assert len(FEATURE_COLS) == 21, len(FEATURE_COLS)
    print("feat_drives self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main())
