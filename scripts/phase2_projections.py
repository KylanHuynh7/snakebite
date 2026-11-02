"""Phase 2: project 2026-season full-PPR points per game with gradient boosting.

Approach: build one row per player-season from 2016-2025 weekly data, pair each
season's features with the player's NEXT season PPR points per game, and train
LightGBM quantile models (20th / 50th / 80th percentile) on those transitions.
Backtest holds out the 2024->2025 transition; the final model trains on all
transitions and projects the upcoming 2026 season from 2025 stats.

Rookies (2026 draft class) have no NFL stats and are NOT projected here -- they
enter the board in Phase 3 via ADP.
"""

from pathlib import Path

import lightgbm as lgb
import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

SEASONS = list(range(2016, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]
REPLACEMENT_RANK = {"QB": 13, "RB": 28, "WR": 28, "TE": 13}
MIN_GAMES = 4

# --- TD dependency, made position-relative (item 8, Aug 10 2026) -------------
# The old flag was `td_pt_share >= 0.40`, an absolute cut. It flagged almost
# nothing but quarterbacks, because a passing TD is 4 points against a passing
# yard's 0.04 and the MEDIAN QB season sits at 0.43 -- above the line. So the
# rule fired on the median player at one position and never on a WR at 0.35,
# who is near the top of his own distribution. It described positions, not
# players.
#
# `scripts/td_dependency.py` re-derives the flag from 2,194 player-season
# pairs and controls for LEVEL, which is the test the old rule never faced:
# TD-heavy seasons are good seasons and good seasons regress, so an
# uncontrolled number looks convincing whether or not anything is there. The
# uncontrolled gap for a top-decile RB is -2.45 PPG; after regressing
# next-year PPG on year-1 PPG (quadratic, within position) it is -2.08. Most
# of it is real, but a third of the raw effect was regression to the mean.
#
# Measured gap vs same-level peers, top decile within position, and the count
# of the nine transition seasons where the TD-heavy half did worse:
#
#     RB  -2.08 PPG  7/9      <- where this flag earns its place
#     QB  -1.56 PPG  1/1      <- n=13, only one season had the sample. Thin.
#     TE  -0.93 PPG  6/9
#     WR  -0.53 PPG  7/9      <- consistent in sign, small in size
#
# Fires at the 85th percentile within position: the effect is monotone from
# the quintile to the decile, so a hard decile cut would drop real cases, and
# the percentile is printed so a human can weigh a p86 differently from a p99.
TD_DEP_PCT = 0.85
# Rates need a real sample. A 5-game TD rate is noise, and flagging it would
# put a warning on a player whose "dependency" is one goal-line week.
TD_DEP_MIN_GAMES = 10
TD_DEP_EFFECT = {"RB": -2.08, "QB": -1.56, "TE": -0.93, "WR": -0.53}

# --- current status, blind spot #3 (Aug 10 2026) -----------------------------
# Median games actually played the season AFTER a lost one, by position, from
# `injury_return_trends.py` section A (308 cases, 1999-2025: a >=14-game,
# >=10 PPG season followed by a <=10-game one). These are not guesses about
# any individual, they are what the cohort did -- and they are sobering: only
# 40-56% of returning players make it through a full season, so the honest
# ceiling for a player still rehabbing in August is well under 17.
RETURN_GAMES = {"QB": 9.0, "RB": 13.0, "TE": 14.0, "WR": 14.0}
# Age moves the return sharply -- a 31+ player coming back from a lost season
# plays a median of 10 games against 14 for one aged 27 or under. Applied as a
# multiplier on the POSITION median rather than reading the position-by-age
# cells directly, because those cells thin out fast (TE at 31+ is n=6, which is
# an anecdote). The pooled age buckets are n=145/88/75 and the position bases
# are n=78-115, so this keeps both signals on samples that can carry them.
RETURN_AGE_FACTOR = ((27, 1.077), (30, 1.000), (99, 0.769))
# A season this short is the injury year itself rather than evidence about how
# durable a player is, which is what `recovered` gets to discount.
INJURY_SEASON_GAMES = 10

FEATURES = [
    "games", "ppr_pg", "ppr_total", "late_ppg", "trend",
    "targets_pg", "rec_pg", "rec_yd_pg", "rec_td_pg",
    "carries_pg", "rush_yd_pg", "rush_td_pg",
    "attempts_pg", "pass_yd_pg", "pass_td_pg", "int_pg",
    "tgt_share", "air_share", "td_pt_share",
    "prior2_ppg", "has_prior2", "age_next", "years_exp", "team_change",
    "pos_QB", "pos_RB", "pos_TE", "pos_WR",
    # --- team context (blind spot #4, added Aug 9 2026) ---------------------
    # Every feature above describes the player alone. The model therefore could
    # not see that Montgomery's 9.3 car/g and 8 rushing TDs left Detroit, nor
    # that Detroit scores more than most teams -- so "who else is on the field"
    # was invisible, which is the single most common thing fantasy analysis
    # actually argues about.
    "team_off_td", "vac_opp_pos", "inc_opp_pos", "vac_td_pos", "net_opp_pos",
    # --- multi-year record (blind spot #5, added Aug 9 2026) ----------------
    # One lag is not a record. A player with a sustained level regresses
    # toward his OWN mean, not the population's -- see add_history_features.
    "mean3_ppg", "prior3_ppg", "ppg_sd3", "ppg_max3", "games_3yr", "n_seasons3",
]

# --- expected points and loss weighting: BUILT, TESTED, REJECTED (Aug 9) -----
# Both were the top-ranked candidates on paper and both failed walk-forward.
#
# EXPECTED POINTS (ffverse opportunity model). Measured +0.0045 incremental
# R^2 over actual PPG -- 6x the rejected O-line -- and expected points really
# are more stable year to year than actual points (r 0.806 vs 0.770), with
# finishing luck barely repeating (r 0.194). But that R^2 was measured in a
# LINEAR model whose only other input was PPG. This model already has
# targets_pg, carries_pg, rec_pg, tgt_share, air_share and the rest -- the
# very usage that expected points is built from -- so a gradient booster can
# already assemble it. Adding the finished quantity was redundant: every
# variant (luck term alone, level alone, all three) came out neutral or worse
# on all four metrics.
#
# LOSS WEIGHTING toward draft-relevant players failed more interestingly. It
# was tested as a strength dial (weight ** exp) rather than on/off, and
# MAE-among-relevant-players degrades MONOTONICALLY as the weighting gets
# stronger: 3.250 (uniform) -> 3.358 -> 3.388 -> 3.440. It makes worse exactly
# the players it was meant to help. Two reasons, both worth remembering:
# down-weighting ordinary players removes the model's evidence about how
# production regresses, and the high-VOR players being up-weighted are
# precisely the ones most subject to regression -- so the model is pushed to
# fit season highs that will not repeat. The uniform loss is doing real work.
#
# Columns are still computed (cheap, and useful for display and re-testing);
# they are simply not in FEATURES. `eval_variants()` reruns the comparison.
XFP_FEATURES = ["xfp_pg", "fp_over_exp", "xfp_mean3"]

# --- run blocking: BUILT, TESTED, AND REJECTED (Aug 9 2026) ------------------
# `line_history.py` computes real run-block metrics for every team-season from
# play-by-play, and they are genuinely measuring blocking -- they correlate
# -0.49 with an independent hand-graded ranking of the same season.
#
# They still do not belong in this model. On a SINGLE held-out season they
# looked good (+1.7% for RBs), but that was one seed on one split. Averaged
# over three seeds and five walk-forward seasons the gain is NEGATIVE for
# every position: RB -0.1%, QB -0.2%, WR -0.3%, TE -1.0%. The 2024-only
# result was noise, and it is exactly the kind of noise that looks like a
# breakthrough if you stop testing when the answer pleases you.
#
# The likely reason is that last year's line barely predicts next year's --
# personnel turns over, and the player-level features already carry most of
# what the line contributed to his own production. Kept OUT of FEATURES,
# kept ON the board as an annotation, and left here wired up so the decision
# is re-testable when there is more data.
OL_FEATURES = ["ol_line_yards", "ol_stuff", "ol_power"]

HISTORY_FEATURES = ["mean3_ppg", "prior3_ppg", "ppg_sd3", "ppg_max3",
                    "games_3yr", "n_seasons3"]


def load_weekly_full() -> pl.DataFrame:
    cache = DATA / "weekly_stats_2016_2025.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    df = nfl.load_player_stats(SEASONS).filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(POSITIONS)
    )
    df = df.select([
        "player_id", "player_display_name", "position", "team", "season", "week",
        "fantasy_points_ppr", "targets", "receptions", "receiving_yards",
        "receiving_tds", "carries", "rushing_yards", "rushing_tds",
        "attempts", "passing_yards", "passing_tds", "passing_interceptions",
        "target_share", "air_yards_share",
    ])
    df.write_parquet(cache)
    return df


def season_features(weekly: pl.DataFrame) -> pl.DataFrame:
    weekly = weekly.sort(["player_id", "season", "week"])
    agg = (
        weekly.group_by(["player_id", "player_display_name", "position", "season"])
        .agg(
            games=pl.col("fantasy_points_ppr").count(),
            ppr_total=pl.col("fantasy_points_ppr").sum(),
            ppr_pg=pl.col("fantasy_points_ppr").mean(),
            late_ppg=pl.col("fantasy_points_ppr").tail(6).mean(),
            team=pl.col("team").last(),
            targets_pg=pl.col("targets").mean(),
            rec_pg=pl.col("receptions").mean(),
            rec_yd_pg=pl.col("receiving_yards").mean(),
            rec_td_pg=pl.col("receiving_tds").mean(),
            carries_pg=pl.col("carries").mean(),
            rush_yd_pg=pl.col("rushing_yards").mean(),
            rush_td_pg=pl.col("rushing_tds").mean(),
            attempts_pg=pl.col("attempts").mean(),
            pass_yd_pg=pl.col("passing_yards").mean(),
            pass_td_pg=pl.col("passing_tds").mean(),
            int_pg=pl.col("passing_interceptions").mean(),
            tgt_share=pl.col("target_share").fill_null(0).mean(),
            air_share=pl.col("air_yards_share").fill_null(0).mean(),
            rec_tds=pl.col("receiving_tds").sum(),
            rush_tds=pl.col("rushing_tds").sum(),
            pass_tds=pl.col("passing_tds").sum(),
            # season totals, needed to add up what a team gained and lost
            carries_tot=pl.col("carries").sum(),
            targets_tot=pl.col("targets").sum(),
        )
        .with_columns(
            # polars' count() is UNSIGNED. `games - next_games` then underflows
            # to 4294967295 the moment a player plays more games than last
            # year, which is silent and looks like a plausible large number
            # rather than an error. Cast once, here, so no downstream
            # arithmetic can step on it.
            games=pl.col("games").cast(pl.Int32),
            trend=pl.col("late_ppg") - pl.col("ppr_pg"),
            td_pt_share=(
                6 * (pl.col("rec_tds") + pl.col("rush_tds"))
                + 4 * pl.col("pass_tds")
            )
            / pl.col("ppr_total").clip(lower_bound=1.0),
        )
    )
    # --- multi-year memory (blind spot #5, added Aug 9 2026) ----------------
    # The model's only memory beyond last season was `prior2_ppg`, and a
    # single lag is not the same as a RECORD. The same defect showed up twice
    # in one day: the team-TD model over-regressed Detroit until a three-year
    # mean was added, and this model projected Josh Allen at 18.4 PPG after
    # four straight seasons of 22.8-24.7. A player with a sustained level
    # regresses toward HIS OWN mean, not the population's.
    #
    # Joined by SEASON NUMBER, never by row order: a player who misses a whole
    # year must not have the season before it silently slide into the lag-1
    # slot as though it were consecutive.
    for k in (1, 2):
        lag = agg.select(
            "player_id",
            (pl.col("season") + k).alias("season"),
            pl.col("ppr_pg").alias(f"lag{k}_ppg"),
            pl.col("games").alias(f"lag{k}_games"),
        )
        agg = agg.join(lag, on=["player_id", "season"], how="left")
    # keep the original encoding of prior2 exactly as it was, so this change
    # cannot quietly move the previously-validated features underneath us
    agg = agg.with_columns(
        has_prior2=pl.col("lag1_ppg").is_not_null().cast(pl.Int8),
        prior2_ppg=pl.col("lag1_ppg").fill_null(-1.0),
    )
    return agg


def load_expected_points() -> pd.DataFrame:
    """Per-season expected fantasy points from ffverse's opportunity model.

    `total_fantasy_points_exp` is what a player's USAGE was worth -- his
    targets, carries and where on the field they happened -- before any credit
    for how well he finished. The gap between that and what he actually scored
    is finishing luck, and finishing luck barely repeats (r=0.194 year over
    year, against r=0.806 for the expected number itself).

    This is the property the rejected O-line feature lacked. A team-level
    feature duplicates what a player's own PPG already encodes; this SPLITS
    that PPG into the half that persists and the half that does not.

    Actual and expected are both taken from THIS source so they share a scoring
    system -- mixing our PPR mean with their expected number would put a
    scoring mismatch straight into the feature.
    """
    cache = DATA / "expected_points_2016_2025.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = nfl.load_ff_opportunity(SEASONS, stat_type="weekly")
    keep = [c for c in ("season", "player_id", "total_fantasy_points_exp",
                        "total_fantasy_points") if c in d.columns]
    p = d.select(keep).to_pandas()
    p["season"] = p["season"].astype(int)
    g = (p.groupby(["player_id", "season"])
         .agg(xfp_games=("total_fantasy_points_exp", "size"),
              xfp_pg=("total_fantasy_points_exp", "mean"),
              fp_src_pg=("total_fantasy_points", "mean"))
         .reset_index())
    g["fp_over_exp"] = g["fp_src_pg"] - g["xfp_pg"]
    g = g.drop(columns=["fp_src_pg"])
    g.to_parquet(cache, index=False)
    return g


def add_expected_points(df: pd.DataFrame) -> pd.DataFrame:
    """Join expected points, plus a three-year expected-points baseline.

    The three-year version exists for the same reason `mean3_ppg` does: one
    season of usage is a noisy read on a player's role.
    """
    xp = load_expected_points()
    out = df.merge(xp[["player_id", "season", "xfp_pg", "fp_over_exp"]],
                   on=["player_id", "season"], how="left")
    lag = xp[["player_id", "season", "xfp_pg"]].copy()
    for k in (1, 2):
        l = lag.rename(columns={"xfp_pg": f"xfp_lag{k}"}).copy()
        l["season"] = l["season"] + k
        out = out.merge(l, on=["player_id", "season"], how="left")
    stack = pd.concat([out["xfp_pg"], out["xfp_lag1"], out["xfp_lag2"]], axis=1)
    out["xfp_mean3"] = stack.mean(axis=1)
    out = out.drop(columns=["xfp_lag1", "xfp_lag2"])
    hit = out["xfp_pg"].notna().mean()
    print(f"expected points joined for {hit:.0%} of player-seasons")
    return out


def add_draft_weight(df: pd.DataFrame) -> pd.DataFrame:
    """How much a training row should count, given the model is used to RANK.

    The loss was uniform MAE over every player, but the board is a ranking and
    the errors are not equally costly. QB is the worst position by MAE (3.50)
    and the least consequential -- QB VOR is about 1.3 in a 1QB league, so a
    large QB miss barely moves a pick, while a small miss on an early-round RB
    moves it a lot. Uniform loss spends the model's capacity on the players who
    matter least.

    Weight is value over replacement in the CURRENT season -- known at training
    time, no peeking at the outcome being predicted. Floored at 1.0 so nobody
    is dropped, and capped so a single outlier season cannot dominate.
    """
    w = pd.Series(1.0, index=df.index)
    for season, g in df.groupby("season"):
        for pos, gp in g.groupby("position"):
            n = REPLACEMENT_RANK.get(pos)
            if not n or len(gp) < n:
                continue
            repl = gp["ppr_pg"].nlargest(n).iloc[-1]
            w.loc[gp.index] = 1.0 + (gp["ppr_pg"] - repl).clip(lower=0)
    df["draft_weight"] = w.clip(upper=w.quantile(0.99))
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Three-year record: level, consistency, and availability.

    Weighted by GAMES, because a five-game season is weaker evidence of a
    player's level than a seventeen-game one and averaging them equally would
    let one injury-shortened year drag a career baseline around.

    Missing seasons stay NaN rather than taking a sentinel value -- LightGBM
    splits on missingness natively, and "no third year on record" is real
    information (it is what separates a two-year wonder from Josh Allen).
    """
    ppg = [df["ppr_pg"], df["lag1_ppg"], df["lag2_ppg"]]
    gms = [df["games"], df["lag1_games"], df["lag2_games"]]

    num = sum(p.fillna(0) * g.fillna(0) for p, g in zip(ppg, gms))
    den = sum(g.fillna(0) for g in gms)
    df["mean3_ppg"] = num / den.replace(0, np.nan)
    df["games_3yr"] = den                       # also a durability signal
    df["n_seasons3"] = sum(p.notna().astype(int) for p in ppg)
    stacked = pd.concat(ppg, axis=1)
    df["ppg_sd3"] = stacked.std(axis=1)         # NaN with fewer than 2 seasons
    df["ppg_max3"] = stacked.max(axis=1)
    df["prior3_ppg"] = df["lag2_ppg"]
    return df


def to_frame(agg: pl.DataFrame, players: pl.DataFrame) -> pd.DataFrame:
    """Pandas frame with age/experience joined; one row per player-season."""
    meta = players.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("birth_date").cast(pl.Date, strict=False),
        "draft_year",
    )
    df = agg.join(meta, on="player_id", how="left").to_pandas()
    df["birth_year"] = pd.to_datetime(df["birth_date"]).dt.year
    # age entering the NEXT season (the one being predicted)
    df["age_next"] = (df["season"] + 1) - df["birth_year"]
    df["age_next"] = df["age_next"].fillna(df["age_next"].median())
    df["years_exp"] = (df["season"] + 1) - df["draft_year"].fillna(
        df.groupby("player_id")["season"].transform("min")
    )
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    df = add_history_features(df)
    df = add_expected_points(df)
    return add_draft_weight(df)


def add_team_context(df: pd.DataFrame, dest: pd.Series) -> pd.DataFrame:
    """Team-context features for each player-season, given where he plays NEXT.

    `dest` is the destination team, aligned to `df.index`: `next_team` when
    building training transitions, `team_2026` when projecting. Using the
    destination for both is what keeps training and serving symmetric -- the
    question is always "what does the roster he is about to play on look like?"

    Three quantities, all measured in season s and all excluding the player:

      vac_opp_pos  opportunities (carries + targets) at his own position that
                   were on that roster in s and are NOT on it in s+1
      inc_opp_pos  opportunities at his own position arriving from elsewhere
      team_off_td  the destination team's offensive TDs in s

    Detroit 2025 -> 2026 for Gibbs: Montgomery leaves (vacated), Pacheco
    arrives (incoming). The net is what actually changed, and `vac_td_pos`
    carries the goal-line half of it separately, because vacated touchdowns and
    vacated carries are not the same commodity.

    Caveat: for training, "on the roster in s+1" means "played for them in s+1",
    since that is all the weekly data can show. For 2026 it is the actual roster
    file. A player signed but never used therefore counts as vacated in training
    and as present in 2026 -- a small train/serve skew, in the conservative
    direction.
    """
    d = df[["player_id", "season", "team", "position",
            "carries_tot", "targets_tot", "rush_tds", "rec_tds"]].copy()
    d["dest"] = dest.values
    d["opp"] = d["carries_tot"].fillna(0) + d["targets_tot"].fillna(0)
    d["td"] = d["rush_tds"].fillna(0) + d["rec_tds"].fillna(0)

    # who played where, per season -- the source of truth for "was he here?"
    where = d[["player_id", "season", "team"]].rename(columns={"team": "was"})

    # a team's offensive TDs in season s
    team_td = d.groupby(["team", "season"])["td"].sum().rename("team_off_td")

    # roster membership in s+1, expressed as of season s
    nxt_member = d[["player_id", "season", "dest"]].copy()
    nxt_member["season"] -= 1          # so it lines up with season s rows

    # For every (destination team T, season s) we need the position-level
    # opportunity totals of: players on T in s, and players on T in s+1.
    on_now = d.groupby(["team", "season", "position"])[["opp", "td"]].sum()
    on_now.index.names = ["dest", "season", "position"]

    # players who were on T in s AND on T in s+1 -> they did not vacate
    d["stayed"] = d["team"] == d["dest"]
    stayed = (d[d["stayed"]].groupby(["team", "season", "position"])[["opp", "td"]]
              .sum().rename(columns={"opp": "stay_opp", "td": "stay_td"}))
    stayed.index.names = ["dest", "season", "position"]

    ctx = on_now.join(stayed, how="left").fillna(0.0)
    ctx["vac_opp"] = ctx["opp"] - ctx["stay_opp"]
    ctx["vac_td"] = ctx["td"] - ctx["stay_td"]

    # incoming: on T in s+1 but produced for someone else in s
    inc = (d[~d["stayed"] & d["dest"].notna()]
           .groupby(["dest", "season", "position"])["opp"].sum().rename("inc_opp"))

    out = df.copy()
    key = pd.MultiIndex.from_arrays(
        [d["dest"], d["season"], d["position"]], names=["dest", "season", "position"]
    )
    out["vac_opp_pos"] = ctx["vac_opp"].reindex(key).to_numpy()
    out["vac_td_pos"] = ctx["vac_td"].reindex(key).to_numpy()
    out["inc_opp_pos"] = inc.reindex(key).to_numpy()
    out["team_off_td"] = team_td.reindex(
        pd.MultiIndex.from_arrays([d["dest"], d["season"]], names=["team", "season"])
    ).to_numpy()

    for c in ("vac_opp_pos", "vac_td_pos", "inc_opp_pos"):
        out[c] = out[c].fillna(0.0)
    # a player with no destination has no team context to speak of
    out["team_off_td"] = out["team_off_td"].fillna(out["team_off_td"].median())
    # the player's own arrival is not competition for himself
    own_inc = (~d["stayed"] & d["dest"].notna()).to_numpy() * d["opp"].to_numpy()
    out["inc_opp_pos"] = (out["inc_opp_pos"] - own_inc).clip(lower=0)
    out["net_opp_pos"] = out["vac_opp_pos"] - out["inc_opp_pos"]

    # --- run blocking on the destination team (blind spot #1) ---------------
    # Computed by line_history.py from play-by-play, so it exists for every
    # season and can therefore be trained and ablated -- unlike a third
    # party's forecast for an unplayed season. Measured in season s, which is
    # what a drafter actually knows when he picks.
    ol_p = DATA / "ol_history.parquet"
    if ol_p.exists():
        ol = pl.read_parquet(ol_p).to_pandas()
        okey = pd.MultiIndex.from_arrays([d["dest"], d["season"]],
                                         names=["team", "season"])
        for col, name in (("adj_line_yards_dev", "ol_line_yards"),
                          ("stuff_rate_dev", "ol_stuff"),
                          ("power_success_dev", "ol_power")):
            s = ol.set_index(["team", "season"])[col]
            out[name] = s.reindex(okey).to_numpy()
            # leave missing as NaN: LightGBM splits on it, and "no destination"
            # is real information rather than a league-average line
    else:
        for name in ("ol_line_yards", "ol_stuff", "ol_power"):
            out[name] = np.nan
    return out


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Pair season s features with season s+1 outcomes; add team-change flag."""
    nxt = df[["player_id", "season", "ppr_pg", "games", "team"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(
        columns={"ppr_pg": "next_ppg", "games": "next_games", "team": "next_team"}
    )
    tr = df.merge(nxt, on=["player_id", "season"], how="inner")
    tr = tr[(tr["games"] >= MIN_GAMES) & (tr["next_games"] >= MIN_GAMES)]
    tr["team_change"] = (tr["team"] != tr["next_team"]).astype(int)
    tr = add_team_context(tr.reset_index(drop=True), tr["next_team"].reset_index(drop=True))
    return tr


# --- quantile calibration (Aug 10 2026) --------------------------------------
# `floor_ppg` was fitted at alpha 0.2 and labelled a floor, which invites the
# reading "one year in five is worse than this". Walk-forward over 2020-2024,
# 29.0% of players finished below it -- the nominal 60% interval covered 50%,
# and the whole miss was on the downside. Stable to within a point in every one
# of the five seasons, so systematic, not noise.
#
# In football terms the left tail is career collapse -- injury, benching, a
# rookie taking the job -- and it is fat, sudden and bounded at zero. A
# regularised tree (num_leaves 15, min_child_samples 30) cannot reach into it,
# because predicting that low needs a leaf built almost entirely from collapse
# seasons and the regularisation exists to prevent exactly that leaf. The upper
# tail is smooth, which is why the ceiling was fine and is left alone.
#
# `scripts/calibrate_quantiles.py` re-derives these. Asking for 0.10 yields a
# true 20th percentile: leave-one-season-out chose 0.10 on all five folds and
# landed 17.5-20.0% out of sample (mean error 1.1%).
#
# TE is the ONE position that needs its own alpha, and it was accepted only
# because per-position fitting was tested and mostly FAILED. Out-of-sample
# error, per-position alpha vs pooled 0.10: RB 2.8% vs 1.7%, WR 2.3% vs 1.3%,
# QB 2.8% vs 2.9% -- pooled wins or ties for three of four, i.e. fitting four
# alphas on ~78 rows per position-season overfits. TE is the exception that
# earns it: 2.4% vs 5.3% pooled, choosing 0.14 on all five folds.
FLOOR_ALPHA = 0.10
FLOOR_ALPHA_BY_POS = {"TE": 0.14}
CEIL_ALPHA = 0.80


def fit_models(train: pd.DataFrame) -> dict[float, lgb.LGBMRegressor]:
    """Median, ceiling, and every distinct floor alpha the board needs."""
    alphas = {0.5, CEIL_ALPHA, FLOOR_ALPHA, *FLOOR_ALPHA_BY_POS.values()}
    models = {}
    for q in sorted(alphas):
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=400, learning_rate=0.03,
            num_leaves=15, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, random_state=7, verbose=-1,
        )
        m.fit(train[FEATURES], train["next_ppg"])
        models[q] = m
    return models


def predict_floor(models: dict[float, lgb.LGBMRegressor],
                  frame: pd.DataFrame) -> np.ndarray:
    """Floor at the calibrated alpha for each player's position."""
    out = models[FLOOR_ALPHA].predict(frame[FEATURES])
    for pos, alpha in FLOOR_ALPHA_BY_POS.items():
        m = (frame["position"] == pos).values
        if m.any():
            out[m] = models[alpha].predict(frame.loc[m, FEATURES])
    return out


TEAM_FEATURES = ["team_off_td", "vac_opp_pos", "inc_opp_pos",
                 "vac_td_pos", "net_opp_pos"]


def _fit_eval(train, test, feats):
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=400,
        learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8, random_state=7, verbose=-1)
    m.fit(train[feats], train["next_ppg"])
    return m.predict(test[feats])


def ablation(transitions: pd.DataFrame) -> None:
    """Does each feature family actually earn its place? Same split, removed.

    Reported honestly either way -- a feature set that does not improve the
    held-out error is not worth the extra machinery, however good the story is.
    And reported by COHORT, because a feature aimed at one kind of player is
    diluted to nothing in an aggregate that is mostly other kinds of player:
    team context gains 7% on high-churn rosters and nil overall, which would
    look like failure if you only read the average.
    """
    train = transitions[transitions["season"] <= 2023]
    test = transitions[transitions["season"] == 2024].copy()

    print("\n=== Ablation: team-context features ===")
    t_wo = _fit_eval(train, test, [f for f in FEATURES if f not in TEAM_FEATURES])
    t_w = _fit_eval(train, test, FEATURES)
    for label, pred in (("without team context", t_wo), ("with team context", t_w)):
        print(f"  {label:<22} MAE {np.abs(pred - test['next_ppg']).mean():.3f}"
              f"   Spearman {spearmanr(pred, test['next_ppg']).statistic:.4f}")
    # aggregate is the wrong lens here and always was: most players have no
    # roster churn, so the cohort this feature exists for gets averaged away
    ch = test["net_opp_pos"].abs()
    tw = test.assign(e_wo=np.abs(t_wo - test["next_ppg"]),
                     e_w=np.abs(t_w - test["next_ppg"]))
    print(f"  {'cohort':<26}{'n':>5}{'without':>9}{'with':>8}{'change':>9}")
    for label, mask in (("high churn (top 25%)", ch >= ch.quantile(0.75)),
                        ("very high churn (10%)", ch >= ch.quantile(0.90)),
                        ("low churn (bottom 50%)", ch <= ch.median())):
        s = tw[mask]
        a, b = s["e_wo"].mean(), s["e_w"].mean()
        print(f"  {label:<26}{len(s):>5}{a:>9.3f}{b:>8.3f}{(a-b)/a:>+8.1%}")

    print("\n=== Ablation: multi-year record ===")
    without = [f for f in FEATURES if f not in HISTORY_FEATURES]
    p_wo = _fit_eval(train, test, without)
    p_w = _fit_eval(train, test, FEATURES)
    for label, pred in (("without 3-yr record", p_wo), ("with 3-yr record", p_w)):
        print(f"  {label:<22} MAE {np.abs(pred - test['next_ppg']).mean():.3f}"
              f"   Spearman {spearmanr(pred, test['next_ppg']).statistic:.4f}")

    # the cohort the feature is FOR: players with a real multi-year record,
    # and among them the ones whose level has actually been stable
    test = test.assign(e_wo=np.abs(p_wo - test["next_ppg"]),
                       e_w=np.abs(p_w - test["next_ppg"]))
    three = test["n_seasons3"] >= 3
    steady = three & (test["ppg_sd3"] <= test.loc[three, "ppg_sd3"].median())
    elite = three & (test["mean3_ppg"] >= test.loc[three, "mean3_ppg"].quantile(0.75))
    print(f"  {'cohort':<26}{'n':>5}{'without':>9}{'with':>8}{'change':>9}")
    for label, mask in (("all held-out players", pd.Series(True, index=test.index)),
                        ("3 seasons on record", three),
                        ("  ...and consistent", steady),
                        ("  ...and top-quartile", elite),
                        ("fewer than 3 seasons", ~three)):
        s = test[mask]
        if not len(s):
            continue
        a, b = s["e_wo"].mean(), s["e_w"].mean()
        print(f"  {label:<26}{len(s):>5}{a:>9.3f}{b:>8.3f}{(a-b)/a:>+8.1%}")

    print("\n=== O-line features: computed, tested, REJECTED ===")
    print("  They are not in FEATURES. See the comment on OL_FEATURES: a")
    print("  single-split gain of +1.7% for RBs did not survive seed averaging")
    print("  or walk-forward testing. `line_history.py` still builds them and")
    print("  the board still shows them; the model does not use them.")


def _draft_relevant(te: pd.DataFrame) -> pd.Series:
    """The players a draft actually turns on: above replacement at their spot.

    Evaluating on everyone is how a model gets rewarded for nailing QB28. This
    mask is built from the OUTCOME season, which is fine for scoring (it is not
    given to the model) and is the honest definition of "did the ranking that
    mattered come out right".
    """
    keep = pd.Series(False, index=te.index)
    for pos, g in te.groupby("position"):
        n = REPLACEMENT_RANK.get(pos)
        if n and len(g) >= n:
            keep.loc[g.nlargest(n, "next_ppg").index] = True
    return keep


def eval_variants(transitions: pd.DataFrame, seeds=(7, 17, 23)) -> None:
    """Do expected points help, and does weighting the loss help? Walk-forward.

    Reported on FOUR numbers, because the two changes target different things:
    overall MAE and Spearman, then the same restricted to draft-relevant
    players. Loss re-weighting is expected to make overall MAE slightly WORSE
    while improving the numbers that decide picks -- if it improved everything
    it would not be doing what it was asked to do.
    """
    test_years = [y for y in (2020, 2021, 2022, 2023, 2024)
                  if (transitions["season"] == y).sum() >= 100]
    no_x = [f for f in FEATURES if f not in XFP_FEATURES]
    variants = [
        ("baseline (no xFP, no wt)", no_x, False),
        ("+ expected points", FEATURES, False),
        ("+ draft weighting", no_x, True),
        ("+ both", FEATURES, True),
    ]
    print("\n=== Expected points and loss weighting, walk-forward ===")
    print(f"    ({len(seeds)} seeds x {len(test_years)} held-out seasons)")
    print(f"  {'variant':<26}{'MAE':>8}{'rho':>8}"
          f"{'MAE-rel':>9}{'rho-rel':>9}")
    base = None
    for label, feats, weighted in variants:
        mae, rho, mae_r, rho_r = [], [], [], []
        for ty in test_years:
            tr = transitions[transitions["season"] < ty]
            te = transitions[transitions["season"] == ty]
            rel = _draft_relevant(te)
            for sd in seeds:
                m = lgb.LGBMRegressor(
                    objective="quantile", alpha=0.5, n_estimators=400,
                    learning_rate=0.03, num_leaves=15, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8, random_state=sd,
                    verbose=-1)
                m.fit(tr[feats], tr["next_ppg"],
                      sample_weight=tr["draft_weight"] if weighted else None)
                pr = m.predict(te[feats])
                mae.append(np.abs(pr - te["next_ppg"]).mean())
                rho.append(spearmanr(pr, te["next_ppg"]).statistic)
                mae_r.append(np.abs(pr[rel.values] - te["next_ppg"][rel.values]).mean())
                rho_r.append(spearmanr(pr[rel.values],
                                       te["next_ppg"][rel.values]).statistic)
        row = (np.mean(mae), np.mean(rho), np.mean(mae_r), np.mean(rho_r))
        if base is None:
            base = row
            print(f"  {label:<26}{row[0]:>8.3f}{row[1]:>8.4f}"
                  f"{row[2]:>9.3f}{row[3]:>9.4f}")
        else:
            d = [row[i] - base[i] for i in range(4)]
            print(f"  {label:<26}{row[0]:>8.3f}{row[1]:>8.4f}"
                  f"{row[2]:>9.3f}{row[3]:>9.4f}")
            print(f"  {'':<26}{d[0]:>+8.3f}{d[1]:>+8.4f}"
                  f"{d[2]:>+9.3f}{d[3]:>+9.4f}   <- vs baseline")
    print("  MAE lower is better; rho higher is better.")
    print("  '-rel' = restricted to players who finished above replacement,")
    print("  i.e. the ones a draft is actually decided by.")


def rolling_backtest(transitions: pd.DataFrame, seeds=(7, 17, 23)) -> None:
    """Walk-forward evaluation over several held-out seasons.

    A single held-out season is 369 players and one draw of the world. Feature
    families have been accepted on gaps of ~0.04 MAE, which is inside the
    seed-to-seed noise of a single split -- so a one-season test cannot tell a
    real improvement from a lucky one. This retrains for every test season
    using only earlier data (no peeking) and averages over seeds, which is the
    honest way to ask whether a feature helps.

    Reported per family so a family that only looks good on 2024 is exposed.
    """
    families = {
        "full model": FEATURES,
        "  - 3yr record": [f for f in FEATURES if f not in HISTORY_FEATURES],
        "  - team context": [f for f in FEATURES if f not in TEAM_FEATURES],
        "  - both": [f for f in FEATURES if f not in
                    set(HISTORY_FEATURES) | set(TEAM_FEATURES)],
    }
    test_years = [y for y in (2020, 2021, 2022, 2023, 2024)
                  if (transitions["season"] == y).sum() >= 100]
    print("\n=== Rolling backtest: train on everything BEFORE each test season")
    print(f"    ({len(seeds)} seeds averaged, {len(test_years)} held-out seasons) ===")
    header = "".join(f"{y:>8}" for y in test_years)
    print(f"  {'features':<16}{header}{'mean':>9}{'vs full':>9}")
    results = {}
    for label, feats in families.items():
        per_year = []
        for ty in test_years:
            tr = transitions[transitions["season"] < ty]
            te = transitions[transitions["season"] == ty]
            errs = []
            for sd in seeds:
                m = lgb.LGBMRegressor(
                    objective="quantile", alpha=0.5, n_estimators=400,
                    learning_rate=0.03, num_leaves=15, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8, random_state=sd,
                    verbose=-1)
                m.fit(tr[feats], tr["next_ppg"])
                errs.append(np.abs(m.predict(te[feats]) - te["next_ppg"]).mean())
            per_year.append(float(np.mean(errs)))
        results[label] = per_year
        mean = float(np.mean(per_year))
        delta = mean - float(np.mean(results["full model"]))
        cells = "".join(f"{v:>8.3f}" for v in per_year)
        print(f"  {label:<16}{cells}{mean:>9.3f}"
              f"{'' if label == 'full model' else f'{delta:>+9.3f}'}")
    print("  positive 'vs full' = removing that family made the model WORSE,")
    print("  i.e. the family earns its place. Read the per-season columns too:")
    print("  a family that only helps in one year is fitting that year.")


def backtest(transitions: pd.DataFrame) -> None:
    train = transitions[transitions["season"] <= 2023]
    test = transitions[transitions["season"] == 2024].copy()
    models = fit_models(train)
    test["pred"] = models[0.5].predict(test[FEATURES])
    mae_model = (test["pred"] - test["next_ppg"]).abs().mean()
    mae_naive = (test["ppr_pg"] - test["next_ppg"]).abs().mean()
    rho_model = spearmanr(test["pred"], test["next_ppg"]).statistic
    rho_naive = spearmanr(test["ppr_pg"], test["next_ppg"]).statistic
    print("=== Backtest: trained 2016-2023 transitions, held out 2024->2025 ===")
    print(f"n test players: {len(test)}")
    print(f"MAE  (PPG): model {mae_model:.2f}  vs naive last-year {mae_naive:.2f}")
    print(f"Spearman :  model {rho_model:.3f}  vs naive last-year {rho_naive:.3f}")
    for pos in POSITIONS:
        t = test[test["position"] == pos]
        print(
            f"  {pos}: model MAE {(t['pred']-t['next_ppg']).abs().mean():.2f} "
            f"naive {(t['ppr_pg']-t['next_ppg']).abs().mean():.2f}  (n={len(t)})"
        )


AVAIL_FEATURES = [
    "games", "lag1_games", "lag2_games", "games_3yr", "n_seasons3",
    "age_next", "years_exp", "ppr_pg", "mean3_ppg",
    "carries_pg", "targets_pg", "attempts_pg", "rec_pg",
    "pos_QB", "pos_RB", "pos_TE", "pos_WR",
]


def availability_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Player-seasons paired with next-season GAMES -- crucially, UNCENSORED.

    `build_transitions()` drops pairs with `next_games < 4`, which is right for
    a per-game POINTS target (two games cannot give a meaningful PPG) and
    exactly backwards for an AVAILABILITY target: it throws away the outcomes
    being predicted. 22.9% of rostered player-seasons fall below that line,
    13.1% of them at zero games. The old heuristic was tuned on the censored
    sample, scored 2.993 there, and was **+1.45 games optimistic** against what
    actually happened.

    Conditioned on being ROSTERED the following season. A retirement is not a
    projection error, and the board never shows `proj_games` for a retiree --
    `on_2026_roster` already handles that case separately.
    """
    nxt = df[["player_id", "season", "games"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"games": "next_games"})
    d = df[df["games"] >= MIN_GAMES].merge(nxt, on=["player_id", "season"],
                                           how="left")
    d = d[d["season"] <= 2024].copy()
    d["next_games"] = d["next_games"].fillna(0)
    seasons = list(range(2017, 2027))
    r = (nfl.load_rosters(seasons).select(["season", "gsis_id"]).unique()
         .to_pandas().rename(columns={"gsis_id": "player_id"}))
    r["season"] -= 1
    r["rostered"] = 1
    d = d.merge(r, on=["player_id", "season"], how="left")
    return d[d["rostered"].fillna(0) == 1].copy()


def fit_availability(train: pd.DataFrame) -> list[lgb.LGBMRegressor]:
    """Three seeds, averaged. Objective l1 -- it won on MAE *and* RMSE.

    An l2 (mean-targeting) variant was tested because `proj_games` is
    multiplied into a season total and a total wants an expectation. It came
    out nearly unbiased (-0.433 vs +0.707) but worse on both MAE and RMSE, so
    l1 ships and the residual optimism is documented rather than traded for
    accuracy. See `scripts/availability_model.py`.
    """
    models = []
    for seed in (7, 13, 42):
        m = lgb.LGBMRegressor(
            objective="l1", n_estimators=300, learning_rate=0.03,
            num_leaves=15, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(train[AVAIL_FEATURES], train["next_games"])
        models.append(m)
    return models


def predict_games(models: list[lgb.LGBMRegressor],
                  frame: pd.DataFrame) -> np.ndarray:
    p = np.mean([m.predict(frame[AVAIL_FEATURES]) for m in models], axis=0)
    return np.clip(p, 1.0, 17.0).round(1)


def apply_status(cur: pd.DataFrame, avail_models) -> pd.DataFrame:
    """Fold the hand-maintained status table into `proj_games`.

    This is the one input on the board that cannot come from the season logs,
    because "is he hurt in August 2026" is not in them. `data/
    player_status_2026.csv` documents the vocabulary; the short version is that
    `rehab` caps availability at the measured return-cohort median for the
    position, `recovered` stops an injury season counting as evidence about
    durability, `out` is an explicit override, and `legal`/`healthy` annotate
    without touching a number.

    Only `proj_games` moves. It is tempting to haircut `proj_ppg` too -- the
    return cohort retains only 60-79% of prior PPG -- but the model has already
    seen the short season and regressed him for it, so a second multiplier
    would double-count an effect nothing here can measure the size of. The
    retention rate goes on the board as a note instead, where a human weighs
    it, exactly like the OL and playcaller tables.

    Unmatched names are reported loudly. A status row that matches nobody fails
    silently and invisibly, which is how a Godwin AVOID went missing for weeks.
    """
    cur["status"] = ""
    cur["status_note"] = ""
    path = DATA / "player_status_2026.csv"
    if not path.exists():
        print("\nno player_status_2026.csv -- model stays blind to current "
              "injuries (blind spot #3)")
        return cur

    st = pd.read_csv(path, comment="#")
    st["player"] = st["player"].str.strip()
    known = set(cur["player_display_name"])
    unmatched = [p for p in st["player"] if p not in known]
    if unmatched:
        print(f"\n!! {len(unmatched)} STATUS ROW(S) MATCHED NO PLAYER — "
              f"silently dropped:")
        for name in unmatched:
            near = [p for p in known
                    if name.split()[-1].rstrip(".") in p or p.split()[-1] in name]
            hint = f"   did you mean: {', '.join(sorted(near)[:3])}" if near else ""
            print(f"     {name!r}{hint}")
    st = st[st["player"].isin(known)]

    idx = cur.set_index("player_display_name").index
    before = cur["proj_games"].copy()
    for _, r in st.iterrows():
        m = idx == r["player"]
        pos = cur.loc[m, "position"].iloc[0]
        status = str(r["status"]).strip().lower()
        cur.loc[m, "status"] = status
        cur.loc[m, "status_note"] = str(r.get("note", "") or "")

        if status == "rehab":
            age = float(cur.loc[m, "age_next"].iloc[0])
            factor = next(f for cut, f in RETURN_AGE_FACTOR if age <= cut)
            cap = round(RETURN_GAMES.get(pos, 13.0) * factor, 1)
            cur.loc[m, "proj_games"] = np.minimum(cur.loc[m, "proj_games"], cap)
        elif status == "recovered":
            # Ask the availability model the counterfactual question: what
            # would it say if the injury season had looked like his healthy
            # ones? Replacing the short season and re-predicting keeps ONE code
            # path -- the alternative, a separate formula for recovered
            # players, would silently diverge from the model the moment either
            # changed. If every season on record is short there is nothing to
            # average, so fall back to the league norm rather than inventing a
            # healthy history he does not have.
            row = cur.loc[m].copy()
            cols = ["games", "lag1_games", "lag2_games"]
            gs = [row[c].iloc[0] for c in cols]
            full = [g for g in gs if pd.notna(g) and g > INJURY_SEASON_GAMES]
            healthy = float(np.mean(full)) if full else 15.0
            for c, g in zip(cols, gs):
                if pd.notna(g) and g <= INJURY_SEASON_GAMES:
                    row[c] = healthy
            row["games_3yr"] = sum(
                row[c].iloc[0] for c in cols if pd.notna(row[c].iloc[0]))
            cur.loc[m, "proj_games"] = predict_games(avail_models, row)
        elif status == "out":
            n = r.get("games_out")
            if pd.notna(n):
                cur.loc[m, "proj_games"] = np.minimum(
                    cur.loc[m, "proj_games"], max(17.0 - float(n), 1.0))
            else:
                print(f"  !! {r['player']}: status=out with no games_out "
                      f"-- no adjustment applied")

    moved = st[st["status"].isin(["rehab", "recovered", "out"])]
    print(f"\n=== Current status (blind spot #3): {len(st)} row(s), "
          f"{len(moved)} affecting proj_games ===")
    for _, r in st.iterrows():
        m = idx == r["player"]
        b, a = before[m].iloc[0], cur.loc[m, "proj_games"].iloc[0]
        delta = (f"proj_games {b:.1f} -> {a:.1f}" if abs(a - b) > 0.05
                 else "no numeric change")
        print(f"  {r['player']:<22} {str(r['status']):<10} {delta}")
    return cur


def add_td_dependency(cur: pd.DataFrame) -> pd.DataFrame:
    """Rank each player's TD reliance against his OWN position, not a constant.

    The reference distribution is the 2025 season at his position among players
    with a real sample, so the percentile answers "how TD-reliant was he for a
    starting running back last year", which is the question a human is actually
    asking when he looks at the flag.

    Players under `TD_DEP_MIN_GAMES` get no percentile rather than a shaky one.
    That is deliberate: this measures a RATE, and the players whose rates are
    least trustworthy are exactly the injured ones the board is already unsure
    about, so a spurious flag would land where it does the most damage.
    """
    cur["td_dep_pct"] = np.nan
    ref = cur["games"] >= TD_DEP_MIN_GAMES
    for pos in POSITIONS:
        m = (cur["position"] == pos) & ref
        if m.sum() < 10:            # too few to define a distribution
            continue
        cur.loc[m, "td_dep_pct"] = cur.loc[m, "td_pt_share"].rank(pct=True)

    hit = cur["td_dep_pct"] >= TD_DEP_PCT
    cur["td_dep"] = ""
    cur.loc[hit, "td_dep"] = [
        f"p{pct * 100:.0f} ({TD_DEP_EFFECT.get(pos, 0):+.1f} PPG hist)"
        for pct, pos in zip(cur.loc[hit, "td_dep_pct"], cur.loc[hit, "position"])
    ]
    cur["td_dep_pct"] = (cur["td_dep_pct"] * 100).round(0)
    return cur


def project_2026(df: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    models = fit_models(transitions)  # all transitions through 2024->2025

    cur = df[(df["season"] == 2025) & (df["games"] >= MIN_GAMES)].copy()
    rosters = nfl.load_rosters([2026]).select(
        pl.col("gsis_id").alias("player_id"), pl.col("team").alias("team_2026")
    ).unique(subset=["player_id"]).to_pandas()
    # 2026 roster files use "AZ" for Arizona; player stats use "ARI"
    rosters["team_2026"] = rosters["team_2026"].replace({"AZ": "ARI"})
    cur = cur.merge(rosters, on="player_id", how="left")
    cur["on_2026_roster"] = cur["team_2026"].notna()
    cur["team_change"] = (
        cur["team_2026"].notna() & (cur["team_2026"] != cur["team"])
    ).astype(int)
    cur = add_team_context(cur.reset_index(drop=True),
                           cur["team_2026"].reset_index(drop=True))

    cur["proj_ppg"] = models[0.5].predict(cur[FEATURES])
    cur["floor_ppg"] = predict_floor(models, cur)
    cur["ceil_ppg"] = models[CEIL_ALPHA].predict(cur[FEATURES])
    # The three quantiles come from three INDEPENDENT fits, so nothing makes
    # them monotone -- "quantile crossing" is a known behaviour, not a bug we
    # introduced, and two players on the Aug 9 board were already crossed. The
    # median is the fit the board actually ranks on, so it wins: the interval
    # is clamped around it rather than the median being moved to fit the
    # interval. Caught by validate_board.py, which is how it surfaced at all.
    crossed = ((cur["floor_ppg"] > cur["proj_ppg"])
               | (cur["ceil_ppg"] < cur["proj_ppg"])).sum()
    if crossed:
        print(f"  note: clamped {crossed} crossed quantile interval(s) to the "
              f"median (independent fits are not guaranteed monotone)")
    cur["floor_ppg"] = np.minimum(cur["floor_ppg"], cur["proj_ppg"])
    cur["ceil_ppg"] = np.maximum(cur["ceil_ppg"], cur["proj_ppg"])
    # Durability: a learned model, replacing `0.6 * g3 + 0.4 * 15` (Aug 10).
    # The old formula was not badly tuned -- re-fitting its two constants on
    # uncensored data made it WORSE (MAE 3.043 vs 3.018). It was tuned on a
    # sample that excluded the outcomes it needed to predict. Trained here on
    # 4,001 rostered player-seasons with the zeros left in; walk-forward on the
    # draftable cohort it wins MAE 3.018 -> 2.961, RMSE 4.275 -> 4.096, and
    # cuts the optimism from +1.301 games to +0.707.
    #
    # ⚠️ It is still optimistic, and the reason is structural rather than
    # fixable by tuning: next-season games are heavily LEFT-skewed (median 15,
    # mean 13.11 for draftable starters -- 15% of them play 8 games or fewer),
    # so any median-targeting fit sits above the mean. `proj_total` multiplies
    # this by a PPG that is itself a median, so read season totals as a
    # typical-case number, not an expectation.
    avail = fit_availability(availability_dataset(df))
    cur["proj_games"] = predict_games(avail, cur)
    # Status BEFORE proj_total, so the season-long total reflects the games we
    # actually expect him to play rather than the backward-looking guess.
    cur = apply_status(cur, avail)
    cur["proj_total"] = (cur["proj_ppg"] * cur["proj_games"]).round(0)
    cur = add_td_dependency(cur)

    out = []
    for pos, rank in REPLACEMENT_RANK.items():
        p = cur[cur["position"] == pos].sort_values("proj_ppg", ascending=False)
        repl = p["proj_ppg"].iloc[min(rank, len(p)) - 1]
        p["vor_pg"] = (p["proj_ppg"] - repl).round(2)
        p["pos_rank"] = range(1, len(p) + 1)
        out.append(p)
    res = pd.concat(out).sort_values("vor_pg", ascending=False)
    res["overall_rank"] = range(1, len(res) + 1)
    res["ppg_2025"] = res["ppr_pg"].round(2)
    res["delta_vs_2025"] = (res["proj_ppg"] - res["ppr_pg"]).round(2)
    for c in ("proj_ppg", "floor_ppg", "ceil_ppg", "td_pt_share"):
        res[c] = res[c].round(2)

    cols = [
        "overall_rank", "player_display_name", "position", "team_2026", "team",
        "age_next", "pos_rank", "proj_ppg", "floor_ppg", "ceil_ppg",
        "proj_games", "proj_total", "vor_pg", "ppg_2025", "delta_vs_2025",
        "games", "td_pt_share", "td_dep_pct", "td_dep", "status",
        "status_note", "team_change", "on_2026_roster",
    ]
    res = res[cols].rename(columns={"team": "team_2025", "games": "games_2025"})
    res.to_csv(OUT / "projections_2026.csv", index=False)
    return res


def print_flags(res: pd.DataFrame) -> None:
    top = res[res["overall_rank"] <= 75]
    print("\n=== RED FLAGS (top-75 projected players) ===")
    print("\n-- Model projects a big DROP from 2025 (regression candidates) --")
    print(top.nsmallest(12, "delta_vs_2025")[
        ["overall_rank", "player_display_name", "position", "team_2026",
         "ppg_2025", "proj_ppg", "delta_vs_2025"]].to_string(index=False))
    print("\n-- Age cliff: RB age 28+, WR/TE age 30+ entering 2026 --")
    aged = top[((top["position"] == "RB") & (top["age_next"] >= 28))
               | (top["position"].isin(["WR", "TE"]) & (top["age_next"] >= 30))]
    print(aged[["overall_rank", "player_display_name", "position", "team_2026",
                "age_next", "proj_ppg"]].to_string(index=False))
    print(f"\n-- TD-dependent (top {(1 - TD_DEP_PCT) * 100:.0f}% of his OWN "
          f"position for share of points from TDs) --")
    dep = top[top["td_dep"] != ""].sort_values("td_dep_pct", ascending=False)
    if dep.empty:
        print("   none in the top 75")
    else:
        print(dep[["overall_rank", "player_display_name", "position",
                   "team_2026", "td_pt_share", "td_dep", "proj_ppg"]]
              .to_string(index=False))
    print("   'hist' is the measured gap vs same-level peers at his position,")
    print("   after controlling for how good the season was. RB is where this")
    print("   flag has teeth (-2.08 PPG, 7 of 9 seasons); WR barely (-0.53).")
    print("   See scripts/td_dependency.py.")
    print("\n-- Changed teams for 2026 --")
    print(top[top["team_change"] == 1][
        ["overall_rank", "player_display_name", "position", "team_2025",
         "team_2026", "proj_ppg"]].to_string(index=False))
    print("\n-- Missed major time in 2025 (10 or fewer games) --")
    print(top[top["games_2025"] <= 10][
        ["overall_rank", "player_display_name", "position", "team_2026",
         "games_2025", "ppg_2025", "proj_ppg"]].to_string(index=False))
    print("\n-- Not on any 2026 roster yet (unsigned / retired?) --")
    print(top[~top["on_2026_roster"]][
        ["overall_rank", "player_display_name", "position", "team_2025",
         "proj_ppg"]].to_string(index=False))


def main() -> None:
    weekly = load_weekly_full()
    agg = season_features(weekly)
    df = to_frame(agg, nfl.load_players())
    transitions = build_transitions(df)
    print(f"training transitions: {len(transitions)} player-season pairs "
          f"({transitions['season'].min()}->{transitions['season'].min()+1} "
          f"through {transitions['season'].max()}->{transitions['season'].max()+1})")
    backtest(transitions)
    ablation(transitions)
    rolling_backtest(transitions)
    eval_variants(transitions)
    res = project_2026(df, transitions)

    show = ["overall_rank", "player_display_name", "position", "team_2026",
            "age_next", "proj_ppg", "floor_ppg", "ceil_ppg", "proj_total",
            "vor_pg", "ppg_2025", "delta_vs_2025"]
    print("\n=== Top 40 projected for the 2026 season (full PPR, VOR/game) ===")
    print(res[show].head(40).to_string(index=False))
    print_flags(res)


if __name__ == "__main__":
    main()
