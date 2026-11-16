"""Evidence-classified outcomes and pregame coach/range history, from 2015 on.

Snap counts establish participation after a game, not pregame availability.
Unknown identity/participation remains in the ledger; it is never filled as zero.
Original forecast archives and processed tables are not overwritten by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import hashlib
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from kicker_analysis.build_dataset import (
    ROOT, CACHE, RELEASE, KEEP, ATT_BUCKETS, LONG_COLS, RELOCATIONS, kicking_from_pbp,
    fetch, pbp_file, team_game_context, normalise_games, write_manifest, load_games,
    LIVE_MAX_AGE_HOURS, season_settled, _write_atomic)
from kicker_analysis.fourth_down import COLS, fourth_down_plays, team_game_rates
from kicker_analysis.scoring import BASE_SCORING

DEFAULT_OUT = ROOT / "data/processed/rebuilt"


def prepared_snapshot(directory):
    """Read one completed generation. Partial or changed generations fail closed."""
    directory = Path(directory)
    raw = (directory / "quality.json").read_bytes()
    meta = json.loads(raw)
    relative = meta.get("generation")
    expected = meta.get("prepared_sha256", {})
    if not relative or not {"kicker_games.parquet", "fourth_down.parquet", "coach_history.parquet"} <= set(expected):
        raise ValueError("run prepare.py; no completed bound data generation")
    generation = (directory / relative).resolve()
    if not generation.is_relative_to(directory.resolve()):
        raise ValueError("prepared generation escapes data directory")
    result = {"quality.json": raw}
    for name, digest in expected.items():
        path = (generation / name).resolve()
        if not path.is_relative_to(generation):
            raise ValueError("prepared file escapes generation")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"incomplete or modified data generation: {name}")
        result[name] = blob
    return result


def participation(kicks, snaps, players, context):
    """Full team-game ledger, plus verified zero-outcome K participants.

    Static IDs/position identify a person; actual game snaps establish activity.
    Multiple regular kickers and non-K emergency kickers stay distinct.
    """
    ids = players[["pfr_id", "gsis_id", "position"]].dropna().drop_duplicates()
    if ids.pfr_id.duplicated().any():
        raise ValueError("ambiguous PFR to GSIS identity mapping")
    k = snaps[snaps.position.eq("K") & snaps.st_snaps.gt(0)].copy()
    k["team"] = k.team.replace({**RELOCATIONS, "LAR": "LA"})
    k = k.merge(ids.rename(columns={"position": "identity_position"}),
                left_on="pfr_player_id", right_on="pfr_id", how="left",
                validate="many_to_one")
    key = ["game_id", "team", "player_id"]
    active = k.rename(columns={"gsis_id": "player_id", "player": "snap_name"})
    active = active.drop_duplicates(key)
    observed = kicks.merge(context, on=["season", "week", "team"], how="inner",
                            validate="many_to_one")
    observed = observed.merge(players[["gsis_id", "position"]].dropna(
        subset=["gsis_id"]).drop_duplicates("gsis_id"),
        left_on="player_id", right_on="gsis_id", how="left", validate="many_to_one")
    observed["participation_status"] = np.where(observed.position.eq("K"),
        "attempting_regular_kicker", "emergency_or_unknown_role")
    candidates = active.merge(observed[key], on=key, how="left", indicator=True,
                              validate="one_to_one")
    zero = candidates[candidates._merge.eq("left_only") & candidates.player_id.notna()
                      & candidates.identity_position.eq("K")].copy()
    # A K special-teams snap can be kickoff-only. Require prior placekicking role
    # and no observed regular placekicker taking this game's attempts.
    regular = observed[observed.position.eq("K")].copy()
    zero["known_placekicker"] = False
    for index, row in zero.iterrows():
        clock = int(row.season) * 100 + int(row.week)
        prior_career = regular[(regular.player_id == row.player_id)
                               & ((regular.season * 100 + regular.week) < clock)]
        current = regular[regular.game_id.eq(row.game_id) & regular.team.eq(row.team)]
        zero.loc[index, "known_placekicker"] = current.empty and not prior_career.empty
    zero = zero[["season", "week", "game_type", "team", "player_id", "snap_name", "known_placekicker"]].rename(
        columns={"game_type": "season_type", "snap_name": "player_display_name"})
    for c in set(KEEP + ATT_BUCKETS + LONG_COLS) - set(zero):
        zero[c] = 0
    zero = zero.merge(context, on=["season", "week", "team"], how="inner",
                       validate="many_to_one")
    zero["position"] = "K"
    zero["participation_status"] = np.where(zero.known_placekicker,
        "verified_active_zero", "special_teams_only_role_unresolved")
    out = pd.concat([observed, zero], ignore_index=True)
    role = out.participation_status.isin(["attempting_regular_kicker", "verified_active_zero"])
    n = out[role].groupby(["game_id", "team"]).size()
    out["regular_kickers_in_game"] = pd.MultiIndex.from_frame(out[["game_id", "team"]]).map(n)
    out["model_eligible"] = role & out.regular_kickers_in_game.eq(1)
    emergency_keys = pd.MultiIndex.from_frame(out.loc[
        out.participation_status.eq("emergency_or_unknown_role"), ["game_id", "team"]])
    out["emergency_kicking_context"] = pd.MultiIndex.from_frame(out[["game_id", "team"]]).isin(emergency_keys)
    out.loc[role & out.regular_kickers_in_game.gt(1),
            "participation_status"] = "multiple_regular_kickers"
    ledger = context[["season", "week", "game_id", "team"]].merge(
        out.groupby(["game_id", "team"]).agg(
            kicker_rows=("player_id", "size"),
            eligible_rows=("model_eligible", "sum"),
            statuses=("participation_status", lambda x: ";".join(sorted(set(x))))).reset_index(),
        on=["game_id", "team"], how="left", validate="one_to_one")
    ledger["statuses"] = ledger.statuses.fillna("unresolved_no_verified_kicker")
    unresolved = active[active.player_id.isna()][["game_id", "team", "pfr_player_id", "snap_name"]]
    return out, ledger, unresolved


def coach_history(pbp, context):
    """Context-adjusted coach rates. Every forecast row excludes its entire week.

    Counts, denominators and uncertainty support survive; no-kick choices count.
    Reference strata: distance, yards to gain and score state. Sparse effects
    shrink with 20 prior equivalent decisions, fixed before model evaluation.
    """
    plays = fourth_down_plays(pbp, max_yardline=47).copy()  # <=64-yard menu
    plays["home_team"] = plays.home_team.replace(RELOCATIONS)
    plays["coach"] = np.where(plays.posteam.eq(plays.home_team),
                               plays.home_coach, plays.away_coach)
    plays["distance_band"] = pd.cut(plays.yardline_100 + 17,
        [0, 39, 49, 54, 59, 64], labels=["short", "40s", "50_54", "55_59", "60_64"])
    plays["togo_band"] = pd.cut(plays.ydstogo, [-1, 1, 3, 6, 100], labels=["1", "2_3", "4_6", "7p"])
    plays["score_state"] = np.sign(plays.score_differential).fillna(0).astype(int)
    plays["clock"] = plays.season * 100 + plays.week
    strata = ["distance_band", "togo_band", "score_state"]
    cells = plays.groupby(strata + ["clock"], observed=True).agg(
        n=("is_fg", "size"), fg=("is_fg", "sum"), go=("is_go", "sum")).reset_index()
    for target in ["n", "fg", "go"]:
        cells["prior_" + target] = cells.groupby(strata, observed=True)[target].transform(
            lambda x: x.cumsum().shift(1).fillna(0))
    plays = plays.merge(cells[strata + ["clock", "prior_n", "prior_fg", "prior_go"]],
                         on=strata + ["clock"], how="left", validate="many_to_one")
    # No reference history means no measured adjustment, not an invented rate.
    for action in ["fg", "go"]:
        plays[action + "_residual"] = (plays["is_" + action] -
            plays["prior_" + action] / plays.prior_n.replace(0, np.nan))
    plays["long_menu"] = (plays.yardline_100 >= 38).astype(int)  # 55+ yards
    plays["long_kick"] = plays.long_menu * plays.is_fg
    plays["actual_fg55"] = (plays.is_fg.eq(1) & plays.kick_distance.ge(55)).astype(int)
    plays["menu_distance_proxy"] = plays.yardline_100 + 17
    group = ["coach", "season", "week"]
    agg = plays.groupby(group).agg(
        coach_decisions=("is_fg", "size"), coach_fg=("is_fg", "sum"),
        coach_go=("is_go", "sum"), coach_punt=("is_punt", "sum"),
        coach_long_menu=("long_menu", "sum"), coach_long_kick=("long_kick", "sum"),
        coach_fg_residual=("fg_residual", "sum"), coach_go_residual=("go_residual", "sum"),
        coach_reference_n=("fg_residual", "count")).reset_index()
    # Dense coach/week grid includes zero-decision games and new coaching staffs.
    grid = context[["season", "week", "team", "coach"]].drop_duplicates()
    counts = list(agg.columns.difference(group))
    g = grid.merge(agg, on=group, how="left", validate="many_to_one").fillna({c: 0 for c in counts})
    # A coach appearing on two teams in a week is ambiguous, never double-count.
    if g.duplicated(group).any():
        raise ValueError("ambiguous coach/team/week")
    g = g.sort_values(["season", "week", "coach"])
    for c in counts:
        g[c + "_prior"] = g.groupby("coach")[c].transform(lambda x: x.cumsum().shift(1).fillna(0))
        g[c + "_last16"] = g.groupby("coach")[c].transform(
            lambda x: x.shift(1).rolling(16, min_periods=1).sum().fillna(0))
    for action in ["fg", "go"]:
        g["coach_" + action + "_adjustment"] = g["coach_" + action + "_residual_prior"] / (
            g.coach_reference_n_prior + 20)
    g["coach_long_kick_rate"] = g.coach_long_kick_prior / g.coach_long_menu_prior.replace(0, np.nan)
    g["coach_long_kick_rate_last16"] = g.coach_long_kick_last16 / g.coach_long_menu_last16.replace(0, np.nan)
    return g[["season", "week", "team", "coach"] + [c for c in g if c.endswith("_prior")]
             + [c for c in g if c.endswith("_last16")]
             + ["coach_fg_adjustment", "coach_go_adjustment", "coach_long_kick_rate"]], plays


def build(destination: Path = DEFAULT_OUT, first=2015, last=None, refresh=False):
    published = destination
    destination = published / "generations" / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=True)
    # The existing snapshot is preserved; live acquisition is an explicit builder task.
    games = (load_games() if refresh else normalise_games(
        pd.read_csv(CACHE / "games.csv", low_memory=False)))
    last = int(games.loc[games.home_score.notna(), "season"].max()) if last is None else last
    games = games[games.season.between(first, last) & games.game_type.eq("REG")]
    context = team_game_context(games)
    context = context[context.team_score.notna()].copy()
    coach_map = pd.concat([games[["game_id", side + "_team", side + "_coach"]].rename(
        columns={side + "_team": "team", side + "_coach": "coach"}) for side in ["home", "away"]])
    context = context.merge(coach_map, on=["game_id", "team"], validate="one_to_one")
    players = pd.read_csv(fetch(f"{RELEASE}/players/players.csv", "players.csv",
                                max_age_hours=24.), low_memory=False)
    kicks, snaps, plays, missing, snap_coverage = [], [], [], {}, {}
    for season in range(first, last + 1):
        kicks.append(kicking_from_pbp(season, include_distances=True))
        columns = list(dict.fromkeys(COLS + ["home_team", "home_coach", "away_coach", "play_id", "desc"]))
        plays.append(pd.read_parquet(pbp_file(season), columns=columns))
        name = f"snap_counts_{season}.parquet"
        cached = CACHE / name
        provisional = not (cached.exists() and season_settled(season, cached.stat().st_mtime))
        try:
            frame = pd.read_parquet(fetch(f"{RELEASE}/snap_counts/{name}", name,
                max_age_hours=LIVE_MAX_AGE_HOURS if provisional else None))
            snaps.append(frame)
            expected = set(context.loc[context.season.eq(season), "game_id"])
            available = set(frame.loc[frame.game_type.eq("REG"), "game_id"])
            snap_coverage[str(season)] = {"played_games": len(expected),
                "covered_games": len(expected & available),
                "missing_games": sorted(expected - available)}
        except Exception as exc:
            missing[str(season)] = f"{type(exc).__name__}: {str(exc)[:160]}"
    kicks = pd.concat(kicks, ignore_index=True)
    kicks["team"] = kicks.team.replace(RELOCATIONS)
    out, ledger, unmatched = participation(kicks, pd.concat(snaps), players, context)
    out["fantasy_points"] = BASE_SCORING.points(out)
    if out.duplicated(["game_id", "team", "player_id"]).any():
        raise ValueError("duplicate player/team/game outcome")
    if not out[ATT_BUCKETS].sum(axis=1).eq(out.fg_att).all():
        raise ValueError("unclassified FG attempt distances")
    pbp = pd.concat(plays, ignore_index=True)
    pbp = pbp[pbp.season_type.eq("REG")].copy()
    coaches, choices = coach_history(pbp, context)
    fourth = team_game_rates(pbp)
    out.to_parquet(destination / "kicker_games.parquet", index=False)
    fourth.to_parquet(destination / "fourth_down.parquet", index=False)
    coaches.to_parquet(destination / "coach_history.parquet", index=False)
    choices.to_parquet(destination / "coach_decisions.parquet", index=False)
    trends = choices.groupby(["season", "coach", "distance_band"], observed=True).agg(
        choices=("is_fg", "size"), fg=("is_fg", "sum"), go=("is_go", "sum"),
        punt=("is_punt", "sum"), actual_55p_attempts=("actual_fg55", "sum")).reset_index()
    trends["fg_share"] = trends.fg / trends.choices
    trends["go_share"] = trends.go / trends.choices
    trends.to_csv(destination / "coach_distance_trends.csv", index=False)
    sensitivity = []
    for boundary in [37, 38, 39]:
        menu = choices[choices.yardline_100.ge(boundary)]
        for season, group in menu.groupby("season"):
            sensitivity.append({"season": int(season), "minimum_yardline": boundary,
                "proxy_minimum_distance": boundary + 17, "choices": len(group),
                "fg_choices": int(group.is_fg.sum()), "go_choices": int(group.is_go.sum()),
                "punt_choices": int(group.is_punt.sum()), "actual_55p_attempts": int(group.actual_fg55.sum())})
    pd.DataFrame(sensitivity).to_csv(destination / "long_menu_sensitivity.csv", index=False)
    ledger.to_csv(destination / "participation_ledger.csv", index=False)
    unmatched.to_csv(destination / "unmatched_snap_identities.csv", index=False)
    report = {"first": first, "last": last, "team_games": len(context),
        "rows": len(out), "eligible": int(out.model_eligible.sum()),
        "participation": out.participation_status.value_counts().to_dict(),
        "unresolved_team_games": int(ledger.statuses.eq("unresolved_no_verified_kicker").sum()),
        "unmatched_snap_identities": len(unmatched), "missing_snap_sources": missing,
        "snap_coverage": snap_coverage,
        "historical_availability": "unavailable: participation is not pregame eligibility",
        "wind": "unknown retained; no historical forecast vintage supplied",
        "coach": "context-adjusted and prior-week only; reconstructed assignment, no publication vintage",
        "long_menu": "yardline+17 is a proxy, not actual kick distance; +/-1-yard boundary sensitivity retained"}
    write_manifest(destination / "data_manifest.json")
    manifest = json.loads((destination / "data_manifest.json").read_text())
    report["participation_source_freshness"] = {r["name"]: r["fetched_utc"]
        for r in manifest["inputs"] if r["name"].startswith("snap_counts_") or r["name"] == "players.csv"}
    report["generation"] = destination.relative_to(published).as_posix()
    report["prepared_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.iterdir()) if p.is_file()}
    metadata = (json.dumps(report, indent=2) + "\n").encode()
    _write_atomic(destination / "quality.json", metadata)
    # The pointer moves only after all generation files and their digests exist.
    # Previous complete directories survive an interruption or failed refresh.
    _write_atomic(published / "quality.json", metadata)
    print(json.dumps(report, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--first-season", type=int, default=2015)
    p.add_argument("--last-season", type=int)
    p.add_argument("--refresh", action="store_true", help="refresh schedule before production preparation")
    a = p.parse_args()
    build(a.out_dir, a.first_season, a.last_season, a.refresh)
