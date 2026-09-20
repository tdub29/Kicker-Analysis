"""Leg strength and stadium difficulty, per kicker-game.

Two ideas the existing features cannot express, both built from attempt-level
play-by-play rather than the game-level counts in `kicker_games.parquet`.

LEG STRENGTH. `features.py` knows a kicker's make rate and his longest MAKE.
Neither is leg strength. A kicker whose coach lets him try from 54 has a
different point distribution from one who is hooked at 48, and the tell is the
distance distribution of his ATTEMPTS, which is a joint statement about his leg
and his staff's belief in it. A 50-yarder is worth 5 points against 3 for a
39-yarder, so the same number of makes is not the same fantasy week. Three
things come out of the attempt distribution, all over the kicker's prior games
only:

  ls_att_mean_l8/l16, ls_att_p90_l8/l16
      the centre and the tail of the pool of attempts in the prior 8 and 16
      games. The p90 is the point of the pair: the mean moves with field
      position, the 90th percentile moves with what he is trusted to try.
  ls_long_rate_l8, ls_long_rate_trend8
      share of attempts from 50+, and that share over the last 8 games minus
      the 8 before them. A rising long rate is a staff that has decided the leg
      is live, usually before the box score says so.
  ls_make_short/mid/long_shrunk
      career make rate SPLIT at 40 and 50, shrunk toward the league rate for
      that bucket. Raw splits are unusable at this grain: the median kicker-game
      in 2015-2026 arrives with fewer than 20 career 50+ attempts, so a raw rate
      reads 1.000 or 0.000 for exactly the kickers the model has least reason to
      trust. Add-k shrinkage with k = 10 attempts, i.e.

          rate = (made + 10 * p_league_bucket) / (attempts + 10)

      k = 10 is chosen so that a kicker's own 50+ record outweighs the league
      prior only once he has more than 10 career attempts from there. Ten
      attempts at the league 50+ rate of about 0.65 has a standard error of
      0.15, which is still most of the spread between the best and worst legs,
      so weighting his own record equally at that point is already generous.
      p_league_bucket is itself the league rate over PRIOR SEASONS ONLY; for
      2015, which has no prior season in the downloaded pbp, it falls back to
      the stated constants in LEAGUE_PRIOR_2015.
  ls_att_long_n_career
      how many 50+ attempts the shrunk rate rests on, so the model can discount
      the rate itself rather than being asked to trust it blindly.

STADIUM. Denver is a real effect and everybody knows it; the question is whether
anything survives once the distance mix is controlled for and the venue is
judged only on seasons that already finished. `stad_fg_oe` is makes minus
expected makes at that venue over prior seasons, per attempt, where expected
comes from a league distance curve also fit on prior seasons only:

    stad_fg_oe = (made - expected) / (attempts + 200)

The +200 in the denominator is the shrinkage and it is deliberately harsh. It
means a venue needs roughly 200 prior attempts, about eight home seasons, before
its index reads at half strength. That is what handles Mexico City, London,
Sao Paulo, Munich and the rest without an exclusion list: MEX00 carries about 20
prior attempts, so a +3 make surplus there reports as +0.013 rather than +0.150.
Nothing is dropped by name, and the same rule shrinks a new stadium's first
season too.

VERIFIED (self_check, and the numbers in reports/feat_legstadium_eval.py):
  * every feature is a function of games strictly before the row's own kickoff.
    The game sequence per kicker is built from finished games only and the
    prior-window slices end at the current game's first attempt, never past it.
  * the stadium index for season S uses only seasons < S, on both sides of the
    subtraction, so the distance curve cannot be fit on the games it grades.
  * pooled-window means and percentiles are computed over the POOLED ATTEMPTS
    of the prior N games, not as a mean of per-game means, which would weight a
    one-attempt game the same as a five-attempt one.
  * a blocked field goal counts as an attempt and not a make. It is a
    protection failure, but it is also a kick that did not go through, and the
    alternative (dropping it) inflates every make rate by about half a point.

Usage:
    python src/kicker_analysis/feat_legstadium.py            # writes the table
    python src/kicker_analysis/feat_legstadium.py --self-check
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
OUT = PROC / "legstadium.parquet"

JOIN_KEYS = ["season", "week", "season_type", "player_id"]

FEATURE_COLS = [
    "ls_att_mean_l8", "ls_att_p90_l8", "ls_att_mean_l16", "ls_att_p90_l16",
    "ls_long_rate_l8", "ls_long_rate_trend8",
    "ls_make_short_shrunk", "ls_make_mid_shrunk", "ls_make_long_shrunk",
    "ls_att_long_n_career",
    "stad_fg_oe", "stad_fg_att_prior", "stad_mean_dist_prior",
]

RELOCATIONS = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

# pbp with clean kick_distance and the post-2015 extra point. Same floor the
# rest of the pipeline uses.
FIRST_SEASON = 2015

# Distance splits. 40 and 50 are the fantasy scoring boundaries, so the buckets
# the model cares about and the buckets the points are paid in are the same.
BUCKETS = [("short", 0, 39), ("mid", 40, 49), ("long", 50, 99)]

# Shrinkage weight, in pseudo-attempts, for the per-kicker bucket make rates.
SHRINK_K = 10.0
# Shrinkage weight, in attempts, for the stadium index. An order of magnitude
# harder because a venue effect is an order of magnitude smaller than a leg.
STADIUM_K = 200.0

# Used only for 2015 rows, which have no prior season in the downloaded pbp.
# League make rates by bucket, pre-2015 era. Stated rather than computed so it
# cannot quietly become a peek at the seasons being predicted.
LEAGUE_PRIOR_2015 = {"short": 0.93, "mid": 0.82, "long": 0.60}

# 5-yard bins for the league distance curve. Attempts outside 18-70 do not exist
# in practice; np.digitize puts the strays in the end bins.
DIST_EDGES = np.arange(20, 71, 5)

PBP_COLS = ["season", "week", "season_type", "game_id", "posteam",
            "kicker_player_id", "field_goal_attempt", "field_goal_result",
            "kick_distance"]


def load_attempts(first_season: int = FIRST_SEASON) -> pd.DataFrame:
    """One row per field-goal ATTEMPT, 2015 on. Makes and misses alike."""
    frames = []
    for path in sorted(CACHE.glob("pbp_*.parquet")):
        season = int(path.stem.split("_")[1])
        if season < first_season:
            continue
        p = pd.read_parquet(path, columns=PBP_COLS)
        p = p[(p.field_goal_attempt == 1) & p.kicker_player_id.notna()
              & p.kick_distance.notna()].copy()
        frames.append(p)
    if not frames:
        raise SystemExit(f"no pbp parquet files in {CACHE}")
    a = pd.concat(frames, ignore_index=True)
    a = a.rename(columns={"kicker_player_id": "player_id", "posteam": "team"})
    a["team"] = a.team.replace(RELOCATIONS)
    a["made"] = (a.field_goal_result == "made").astype(int)
    a["dist"] = a.kick_distance.astype(float)
    return a[["season", "week", "season_type", "game_id", "team", "player_id",
              "dist", "made"]]


def _game_order(games: pd.DataFrame) -> pd.DataFrame:
    """game_id -> calendar order, so a kicker's history is in real time order.

    Sorting on (season, week) is nearly right and quietly wrong for the
    playoffs, where week numbering has changed twice. The kickoff date has not.
    """
    g = games[["game_id", "gameday", "stadium_id"]].copy()
    g["gameday"] = pd.to_datetime(g.gameday)
    return g


def kicker_leg_features(skeleton: pd.DataFrame, att: pd.DataFrame) -> pd.DataFrame:
    """Prior-window attempt-distance and bucket-accuracy features per kicker-game.

    `skeleton` is every kicker-game including the ones with no field goal
    attempted; those still consume a slot in the 8- and 16-game windows, because
    "his last 8 games" means games, not games in which he happened to kick.

    The windows are computed over POOLED attempts. Per kicker the attempts are
    laid out in one flat array with a game boundary index, so the prior-8-game
    pool for game i is exactly the slice [start[i-8], start[i]) and can never
    reach the current game's own kicks.
    """
    a = att.sort_values(["player_id", "order", "dist"])
    bucket_of = np.select(
        [a.dist.values <= 39, a.dist.values <= 49], [0, 1], default=2)
    a = a.assign(bucket=bucket_of)

    # Attempts grouped onto the skeleton row they belong to.
    s = skeleton.sort_values(["player_id", "order"]).reset_index(drop=True)
    s["_row"] = np.arange(len(s))
    a = a.merge(s[["player_id", "order", "_row"]], on=["player_id", "order"],
                how="inner")
    a = a.sort_values(["_row", "dist"])

    n_rows = len(s)
    counts = np.bincount(a._row.values, minlength=n_rows)
    start = np.concatenate([[0], np.cumsum(counts)])      # start[i]..start[i+1]
    dist = a.dist.values
    made = a.made.values.astype(float)
    bucket = a.bucket.values

    # Per-attempt cumulative counts, so career bucket totals prior to game i are
    # a difference of two cumulative rows at the game boundary.
    cum_att = np.zeros((len(dist) + 1, 3))
    cum_made = np.zeros((len(dist) + 1, 3))
    for j in range(len(dist)):
        cum_att[j + 1] = cum_att[j]
        cum_made[j + 1] = cum_made[j]
        cum_att[j + 1, bucket[j]] += 1
        cum_made[j + 1, bucket[j]] += made[j]

    out = {c: np.full(n_rows, np.nan) for c in
           ["ls_att_mean_l8", "ls_att_p90_l8", "ls_att_mean_l16", "ls_att_p90_l16",
            "ls_long_rate_l8", "ls_long_rate_trend8"]}
    car_att = np.zeros((n_rows, 3))
    car_made = np.zeros((n_rows, 3))

    # Row index of the first game of each kicker, so a window never crosses from
    # one man's career into the previous man's.
    pid = s.player_id.values
    first = np.zeros(n_rows, dtype=int)
    f = 0
    for i in range(1, n_rows):
        if pid[i] != pid[i - 1]:
            f = i
        first[i] = f

    for i in range(n_rows):
        lo8 = max(first[i], i - 8)
        lo16 = max(first[i], i - 16)
        p8 = dist[start[lo8]:start[i]]
        p16 = dist[start[lo16]:start[i]]
        prev8 = dist[start[lo16]:start[lo8]]
        if p8.size:
            out["ls_att_mean_l8"][i] = p8.mean()
            out["ls_att_p90_l8"][i] = np.quantile(p8, 0.90)
            out["ls_long_rate_l8"][i] = (p8 >= 50).mean()
        if p16.size:
            out["ls_att_mean_l16"][i] = p16.mean()
            out["ls_att_p90_l16"][i] = np.quantile(p16, 0.90)
        if p8.size and prev8.size:
            out["ls_long_rate_trend8"][i] = (p8 >= 50).mean() - (prev8 >= 50).mean()
        # Career to date = everything from this kicker's first attempt up to,
        # and not including, the current game.
        car_att[i] = cum_att[start[i]] - cum_att[start[first[i]]]
        car_made[i] = cum_made[start[i]] - cum_made[start[first[i]]]

    res = pd.DataFrame(out, index=s.index)
    for bi, (name, _, _) in enumerate(BUCKETS):
        res[f"_att_{name}"] = car_att[:, bi]
        res[f"_made_{name}"] = car_made[:, bi]
    res["ls_att_long_n_career"] = car_att[:, 2]
    return pd.concat([s.drop(columns=["_row"]), res], axis=1)


def league_bucket_rates(att: pd.DataFrame) -> pd.DataFrame:
    """League make rate per bucket using PRIOR seasons only, one row per season.

    This is the shrinkage target. It has to move: the league 50+ rate ran about
    0.55 in 2015 and about 0.70 by 2024, and anchoring a 2024 kicker to a 2015
    prior would pull every long leg down by real points.
    """
    a = att.assign(bucket=np.select(
        [att.dist <= 39, att.dist <= 49], ["short", "mid"], default="long"))
    per = (a.groupby(["season", "bucket"])
           .agg(att=("made", "size"), made=("made", "sum")).reset_index())
    wide = per.pivot(index="season", columns="bucket",
                     values=["att", "made"]).sort_index()
    # A bucket with no attempts in a season is a zero, not a missing column;
    # reindex so a thin hand-built frame takes the same path as the real data.
    wide = wide.reindex(columns=pd.MultiIndex.from_product(
        [["att", "made"], [b[0] for b in BUCKETS]])).fillna(0)
    prior = wide.cumsum().shift(1)          # strictly previous seasons
    out = pd.DataFrame(index=wide.index)
    for name, _, _ in BUCKETS:
        r = prior[("made", name)] / prior[("att", name)].replace(0, np.nan)
        out[f"p_{name}"] = r.fillna(LEAGUE_PRIOR_2015[name])
    return out.reset_index()


def stadium_index(att: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (stadium_id, season): makes over expected, from prior seasons only.

    Expected uses a league make-rate curve in 5-yard distance bins, refit for
    each test season on the seasons before it. Grading a venue against a curve
    that includes that venue's own future games is the leak this function is
    shaped to avoid, and it is the reason both the curve and the venue totals
    are cumulative-then-shifted.
    """
    a = att.merge(games[["game_id", "stadium_id"]], on="game_id", how="left")
    a = a[a.stadium_id.notna()].copy()
    a["bin"] = np.digitize(a.dist.values, DIST_EDGES)

    # League curve per season, cumulative over prior seasons.
    curve = (a.groupby(["season", "bin"])
             .agg(n=("made", "size"), m=("made", "sum")).reset_index())
    seasons = sorted(a.season.unique())
    rows = []
    for s in seasons:
        prior = curve[curve.season < s]
        if prior.empty:
            continue
        g = prior.groupby("bin").agg(n=("n", "sum"), m=("m", "sum"))
        # A bin with almost nothing in it (a 20-yard attempt, a 68-yarder) gets
        # the pooled rate rather than a rate built on four kicks.
        pooled = g.m.sum() / g.n.sum()
        rate = np.where(g.n >= 50, g.m / g.n.replace(0, np.nan), pooled)
        rows.append(pd.DataFrame({"season": s, "bin": g.index, "bin_rate": rate}))
    if not rows:
        return pd.DataFrame(columns=["stadium_id", "season", "stad_fg_oe",
                                     "stad_fg_att_prior", "stad_mean_dist_prior"])
    curves = pd.concat(rows, ignore_index=True)

    # Venue totals, cumulative over prior seasons, scored against that season's
    # curve. Expected has to be recomputed per test season rather than carried
    # forward, because the curve itself changes as the league gets better.
    out = []
    for s in seasons:
        prior = a[a.season < s]
        if prior.empty:
            continue
        c = curves[curves.season == s][["bin", "bin_rate"]]
        p = prior.merge(c, on="bin", how="left")
        p["bin_rate"] = p.bin_rate.fillna(p.made.mean())
        agg = p.groupby("stadium_id").agg(
            n=("made", "size"), m=("made", "sum"), e=("bin_rate", "sum"),
            md=("dist", "mean")).reset_index()
        agg["season"] = s
        agg["stad_fg_oe"] = (agg.m - agg.e) / (agg.n + STADIUM_K)
        agg["stad_fg_att_prior"] = agg.n
        agg["stad_mean_dist_prior"] = agg.md
        out.append(agg[["stadium_id", "season", "stad_fg_oe",
                        "stad_fg_att_prior", "stad_mean_dist_prior"]])
    return pd.concat(out, ignore_index=True)


def build(first_season: int = FIRST_SEASON) -> pd.DataFrame:
    kg = pd.read_parquet(PROC / "kicker_games.parquet",
                         columns=JOIN_KEYS + ["team", "game_id"])
    kg = kg[kg.season >= first_season].copy()
    kg["team"] = kg.team.replace(RELOCATIONS)

    games = pd.read_csv(CACHE / "games.csv", low_memory=False)
    order = _game_order(games)
    kg = kg.merge(order[["game_id", "gameday", "stadium_id"]], on="game_id", how="left")
    kg = kg.sort_values(["player_id", "gameday", "week"]).reset_index(drop=True)
    kg["order"] = kg.groupby("player_id").cumcount()

    att = load_attempts(first_season)
    att = att.merge(kg[["player_id", "game_id", "order"]],
                    on=["player_id", "game_id"], how="inner")

    feat = kicker_leg_features(kg, att)

    league = league_bucket_rates(att)
    feat = feat.merge(league, on="season", how="left")
    for name, _, _ in BUCKETS:
        n = feat[f"_att_{name}"]
        made = feat[f"_made_{name}"]
        feat[f"ls_make_{name}_shrunk"] = ((made + SHRINK_K * feat[f"p_{name}"])
                                          / (n + SHRINK_K))

    stad = stadium_index(att, games)
    feat = feat.merge(stad, on=["stadium_id", "season"], how="left")
    return feat[JOIN_KEYS + FEATURE_COLS].sort_values(JOIN_KEYS).reset_index(drop=True)


def main() -> int:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df):,} kicker-games, "
          f"{df.season.min()}-{df.season.max()}, {len(FEATURE_COLS)} features")
    print("\ncoverage and spread:")
    print(df[FEATURE_COLS].describe().T[["count", "mean", "std", "min", "max"]]
          .round(3).to_string())

    # The two eyeball tests. If the p90 of attempt distance does not separate
    # the known big legs, the pooling is wrong; if Denver is not near the top of
    # the venue index, the expected-makes side is wrong.
    print("\nhighest ls_att_p90_l16 in 2025 (min 8 games):")
    k = pd.read_parquet(PROC / "kicker_games.parquet",
                        columns=JOIN_KEYS + ["player_display_name"])
    m = df.merge(k, on=JOIN_KEYS, how="left")
    t = (m[m.season == 2025].groupby("player_display_name")
         .agg(g=("ls_att_p90_l16", "size"), p90=("ls_att_p90_l16", "mean"),
              lr=("ls_long_rate_l8", "mean")))
    print(t[t.g >= 8].sort_values("p90", ascending=False).head(8).round(3).to_string())

    games = pd.read_csv(CACHE / "games.csv", low_memory=False)
    st = stadium_index(load_attempts(), games)
    last = st[st.season == st.season.max()].merge(
        games.drop_duplicates("stadium_id")[["stadium_id", "stadium"]],
        on="stadium_id", how="left")
    last = last.sort_values("stad_fg_oe", ascending=False)
    print(f"\nstadium index as of season {st.season.max()} "
          f"(prior seasons only), best and worst 6:")
    cols = ["stadium", "stad_fg_oe", "stad_fg_att_prior", "stad_mean_dist_prior"]
    print(pd.concat([last.head(6), last.tail(6)])[cols].round(4).to_string(index=False))
    return 0


def self_check() -> int:
    """Lagging, pooled windows, shrinkage arithmetic and the venue index.

    Hand-built frames throughout. Every assertion below fails if the shift that
    keeps a row out of its own window is removed, or if a rate is computed over
    games instead of over attempts.
    """
    # ---- pooled windows and career buckets -------------------------------
    # One kicker, 4 games. Distances chosen so a mean-of-game-means would give a
    # different answer from a pooled mean, which is the bug being ruled out.
    sk = pd.DataFrame({
        "player_id": ["a"] * 4 + ["b"] * 2,
        "season": [2020] * 6, "week": [1, 2, 3, 4, 1, 2],
        "season_type": ["REG"] * 6,
        "order": [0, 1, 2, 3, 0, 1],
    })
    rows = [
        # game 0: three attempts, one from 50+, two made
        ("a", 0, 30.0, 1), ("a", 0, 36.0, 1), ("a", 0, 54.0, 0),
        # game 1: one attempt from 60
        ("a", 1, 60.0, 1),
        # game 2: two attempts
        ("a", 2, 20.0, 1), ("a", 2, 44.0, 0),
        # game 3: irrelevant to games 0-2's features; present to prove it is
        ("a", 3, 99.0, 1),
        ("b", 0, 40.0, 1), ("b", 1, 41.0, 0),
    ]
    att = pd.DataFrame(rows, columns=["player_id", "order", "dist", "made"])
    f = kicker_leg_features(sk, att).set_index(["player_id", "order"])

    # Game 0 has no prior game: undefined, not zero. A zero here would read to
    # the model as "he has never been trusted past the line of scrimmage".
    assert pd.isna(f.loc[("a", 0), "ls_att_mean_l8"])
    assert f.loc[("a", 0), "ls_att_long_n_career"] == 0
    # Game 1 sees game 0 only: pooled mean of 30, 36, 54 = 40.0.
    assert abs(f.loc[("a", 1), "ls_att_mean_l8"] - 40.0) < 1e-9, f.loc[("a", 1)]
    assert abs(f.loc[("a", 1), "ls_long_rate_l8"] - 1 / 3) < 1e-9
    # Game 2 sees games 0 and 1: pooled over FOUR attempts = (30+36+54+60)/4 = 45.
    # A mean of per-game means would be (40 + 60)/2 = 50. That is the difference
    # this assertion exists to catch.
    assert abs(f.loc[("a", 2), "ls_att_mean_l8"] - 45.0) < 1e-9, f.loc[("a", 2)]
    assert abs(f.loc[("a", 2), "ls_long_rate_l8"] - 0.5) < 1e-9
    # Nothing anywhere may see the 99-yard attempt from game 3.
    assert f.ls_att_p90_l16.max() < 99.0
    assert f.ls_att_mean_l16.max() < 61.0
    # Careers do not bleed across kickers: b's first game knows nothing of a.
    assert pd.isna(f.loc[("b", 0), "ls_att_mean_l8"])
    assert f.loc[("b", 1), "_att_mid"] == 1 and f.loc[("b", 1), "_att_long"] == 0
    # a's career buckets before game 2: short = 30, 36 (2 att, 2 made),
    # long = 54, 60 (2 att, 1 made), mid = none.
    assert (f.loc[("a", 2), "_att_short"], f.loc[("a", 2), "_made_short"]) == (2.0, 2.0)
    assert (f.loc[("a", 2), "_att_long"], f.loc[("a", 2), "_made_long"]) == (2.0, 1.0)
    assert f.loc[("a", 2), "_att_mid"] == 0.0

    # ---- shrinkage -------------------------------------------------------
    # 2 of 2 from 50+ against a league prior of 0.60 must not read as 1.000.
    r = (2 + SHRINK_K * 0.60) / (2 + SHRINK_K)
    assert abs(r - 0.6667) < 1e-3, r
    # 0 of 3 must not read as 0.000 either, and the two must straddle the prior.
    r0 = (0 + SHRINK_K * 0.60) / (3 + SHRINK_K)
    assert r0 < 0.60 < r, (r0, r)
    # With no attempts at all the rate IS the prior, which is the right default
    # for a rookie: league average until he shows otherwise.
    assert abs((0 + SHRINK_K * 0.60) / (0 + SHRINK_K) - 0.60) < 1e-12

    # ---- league prior moves, and only backwards --------------------------
    la = pd.DataFrame({"season": [2015] * 4 + [2016] * 4,
                       "dist": [55.0, 55.0, 55.0, 55.0] * 2,
                       "made": [1, 0, 0, 0, 1, 1, 1, 1]})
    lb = league_bucket_rates(la).set_index("season")
    # 2015 has no prior season, so it falls back to the stated constant.
    assert lb.loc[2015, "p_long"] == LEAGUE_PRIOR_2015["long"]
    # 2016's prior is 2015's 1-of-4 and specifically NOT 2016's own 4-of-4.
    assert abs(lb.loc[2016, "p_long"] - 0.25) < 1e-12, lb.loc[2016, "p_long"]

    # ---- stadium index ---------------------------------------------------
    # Two venues, same distance mix, different results, over one prior season.
    g = pd.DataFrame({"game_id": ["g1", "g2", "g3"],
                      "stadium_id": ["DEN00", "CHI00", "DEN00"],
                      "gameday": ["2015-09-13"] * 3})
    sa = pd.DataFrame({
        "game_id": ["g1"] * 4 + ["g2"] * 4 + ["g3"] * 2,
        "season": [2015] * 8 + [2016] * 2,
        "dist": [45.0] * 10,
        "made": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]})
    si = stadium_index(sa, g).set_index(["stadium_id", "season"])
    # Only 2016 is gradeable, and it may only see 2015.
    assert set(si.index.get_level_values("season")) == {2016}
    assert si.loc[("DEN00", 2016), "stad_fg_att_prior"] == 4
    # Denver went 4/4 against an expected 2/4 (the 2015 league rate at 45 yards
    # is 4 of 8), Chicago 0/4 against the same. The sign must split.
    assert si.loc[("DEN00", 2016), "stad_fg_oe"] > 0
    assert si.loc[("CHI00", 2016), "stad_fg_oe"] < 0
    # And the harsh shrinkage must keep a 4-attempt sample near zero: the raw
    # over-expected is +0.50 per attempt, the reported index is +2/204.
    assert abs(si.loc[("DEN00", 2016), "stad_fg_oe"] - 2 / 204) < 1e-9, \
        si.loc[("DEN00", 2016), "stad_fg_oe"]
    assert abs(si.loc[("DEN00", 2016), "stad_fg_oe"]) < 0.02

    print("feat_legstadium self-check ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    sys.exit(self_check() if a.self_check else main())
