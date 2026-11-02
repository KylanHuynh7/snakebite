"""Would this board actually have beaten the market? Rebuild it for 2024 and
2025 as it would have been built at the time, and score both against what
happened.

Every component of this project has been validated on its own -- projections,
availability, TD regression, quantile coverage. None of that answers the only
question that decides a season: **at the draft table, does ranking by this
board beat ranking by ADP?** A model can be more accurate than the market in
aggregate and still add nothing, because the market's errors have to be in
players you can actually draft, at prices you can actually get.

WHAT "AS IT WOULD HAVE BEEN BUILT" MEANS. For target year Y, everything is cut
off at Y-1:

  projections   trained only on transitions ending in Y-1 (season <= Y-2), then
                predicting Y from Y-1 statistics
  availability  the same cutoff, using the uncensored games model
  rookies       trained only on draft classes before Y
  rosters       year-Y rosters ARE used, and legitimately so -- they are known
                in August when the draft happens
  ADP           the real FFC 12-team PPR market for year Y, pulled the week
                before that season started

DELIBERATELY EXCLUDED, because including them would be cheating:

  stances       hand-written for 2026 with full knowledge of what happened in
                2024 and 2025. The single largest leak available, and the
                easiest to overlook because they feel like "the board".
  status file   2026 only, and written in hindsight for those years.

Display-only columns (offence tags, TD-dependency flags, floors) are not part
of the ranking and so cannot affect the result either way.

THREE MEASURES, in increasing order of how much they matter and how noisy they
are:

  A  rank correlation with actual outcomes -- board vs ADP, over the draftable
     range. The basic "who is more right" question.
  B  the disagreement test. The board can ONLY add value where it disagrees
     with the market, so this looks exclusively at those players and asks
     whether the disagreements pointed the right way. This is the sharpest
     measure and the one to trust.
  C  a simulated draft at all 12 slots -- board-ranked team versus ADP-ranked
     team, scored on actual season points. The most decision-relevant and by
     far the noisiest, since one injury swings a whole season.

Usage:  .venv/bin/python scripts/board_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import nflreadpy as nfl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_projections as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HIST = DATA / "league_history"

TEAMS = 12
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX_SLOTS = 2
FLEX_POS = ("RB", "WR", "TE")
POSITIONS = ("QB", "RB", "WR", "TE")
TEST_YEARS = (2024, 2025)
ROSTER_ALIAS = {"AZ": "ARI", "LAR": "LA", "WSH": "WAS"}


def norm(s: pd.Series) -> pd.Series:
    s = s.str.lower().str.replace(r"[^a-z\s]", "", regex=True)
    for suf in (" jr", " sr", " ii", " iii", " iv", " v"):
        s = s.str.removesuffix(suf)
    return s.str.strip()


# ------------------------------------------------------------------ outcomes
def actuals(weekly: pl.DataFrame, year: int) -> pd.DataFrame:
    """What each player ACTUALLY scored in year Y. Absent = zero."""
    return (weekly.filter(pl.col("season") == year)
            .group_by(["player_id", "player_display_name", "position"])
            .agg(act_total=pl.col("fantasy_points_ppr").sum(),
                 act_games=pl.len(),
                 act_ppg=pl.col("fantasy_points_ppr").mean())
            .to_pandas())


def load_adp(year: int) -> pd.DataFrame:
    path = HIST / f"adp_ffc_ppr_{year}.json"
    d = json.loads(path.read_text())
    a = pd.DataFrame(d["players"])[["name", "position", "adp", "times_drafted"]]
    a = a[a["position"].isin(POSITIONS)].copy()
    a["key"] = norm(a["name"])
    return a.sort_values("adp").drop_duplicates("key"), d["meta"]


# ------------------------------------------------------------------ the board
def rookie_board(year: int) -> pd.DataFrame:
    """Rookie projections for year Y, trained only on earlier classes."""
    dp = (nfl.load_draft_picks()
          .filter(pl.col("position").is_in(list(POSITIONS))
                  & (pl.col("season") >= 2010))
          .select("season", "round", "pick", "gsis_id", "pfr_player_name",
                  "position", "age").to_pandas())
    weekly = pl.read_parquet(DATA / "weekly_stats_2016_2025.parquet")
    rook = (weekly.group_by(["player_id", "season"])
            .agg(games=pl.len(), ppr_pg=pl.col("fantasy_points_ppr").mean())
            .to_pandas().rename(columns={"player_id": "gsis_id"}))
    dp = dp.merge(rook, on=["gsis_id", "season"], how="left")
    dp["games"] = dp["games"].fillna(0)
    dp["log_pick"] = np.log(dp["pick"])
    dp["draft_age"] = dp["age"].fillna(dp["age"].median())
    for pos in POSITIONS:
        dp[f"pos_{pos}"] = (dp["position"] == pos).astype(int)
    feats = ["log_pick", "round", "draft_age"] + [f"pos_{p}" for p in POSITIONS]

    train = dp[(dp["season"] < year) & (dp["games"] >= 4)]
    cls = dp[dp["season"] == year].copy()
    if train.empty or cls.empty:
        return pd.DataFrame()
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=300,
                          learning_rate=0.03, num_leaves=7,
                          min_child_samples=20, random_state=7, verbose=-1)
    m.fit(train[feats], train["ppr_pg"])
    cls["proj_ppg"] = m.predict(cls[feats])
    games_by_round = train.groupby("round")["games"].mean()
    cls["proj_games"] = cls["round"].map(games_by_round).fillna(8.0)
    cls = cls.rename(columns={"pfr_player_name": "player"})
    cls["is_rookie"] = 1
    return cls[["player", "position", "proj_ppg", "proj_games", "is_rookie"]]


def replacement(board: pd.DataFrame) -> dict[str, float]:
    """Fill every team's starting lineup; the first player left over at each
    position is replacement level. Identical logic to phase3."""
    pools = {p: board[board["position"] == p]
             .sort_values("proj_ppg", ascending=False)["proj_ppg"].tolist()
             for p in POSITIONS}
    used = {p: STARTERS[p] * TEAMS for p in POSITIONS}
    for _ in range(FLEX_SLOTS * TEAMS):
        best, bp = -9e9, None
        for p in FLEX_POS:
            if used[p] < len(pools[p]) and pools[p][used[p]] > best:
                best, bp = pools[p][used[p]], p
        if bp is None:
            break
        used[bp] += 1
    return {p: (pools[p][min(used[p], len(pools[p])) - 1] if pools[p] else 0.0)
            for p in POSITIONS}


def build_board(df: pd.DataFrame, transitions: pd.DataFrame,
                year: int) -> pd.DataFrame:
    """The board for year Y, using nothing from year Y or later."""
    train = transitions[transitions["season"] <= year - 2]
    models = P.fit_models(train)

    cur = df[(df["season"] == year - 1) & (df["games"] >= P.MIN_GAMES)].copy()
    rost = (nfl.load_rosters([year])
            .select(pl.col("gsis_id").alias("player_id"),
                    pl.col("team").alias("dest"))
            .unique(subset=["player_id"]).to_pandas())
    rost["dest"] = rost["dest"].replace(ROSTER_ALIAS)
    cur = cur.merge(rost, on="player_id", how="left")
    cur["team_change"] = (cur["dest"].notna()
                          & (cur["dest"] != cur["team"])).astype(int)
    cur = P.add_team_context(cur.reset_index(drop=True),
                             cur["dest"].reset_index(drop=True))
    cur["proj_ppg"] = models[0.5].predict(cur[P.FEATURES])

    avail_src = P.availability_dataset(df)
    avail = P.fit_availability(avail_src[avail_src["season"] <= year - 2])
    cur["proj_games"] = P.predict_games(avail, cur)
    cur = cur.rename(columns={"player_display_name": "player"})
    cur["is_rookie"] = 0

    cols = ["player", "position", "proj_ppg", "proj_games", "is_rookie"]
    board = pd.concat([cur[cols], rookie_board(year)], ignore_index=True)
    board = board[board["position"].isin(POSITIONS)]

    repl = replacement(board)
    board["vor_pg"] = board.apply(
        lambda r: r["proj_ppg"] - repl[r["position"]], axis=1)
    # TWO rankings, because the board ships both and they are different claims.
    # `my_rank` sorts on a PER-GAME number, so a player projected for 10 games
    # ranks identically to one projected for 16 at the same PPG. `rank_dur`
    # spreads that value over the games he is expected to play. Scoring against
    # season TOTALS, the durability-aware ordering is the one that matches the
    # objective -- so testing only `my_rank` would be grading the board on a
    # question it does not claim to answer.
    board["vor_season"] = board["vor_pg"] * board["proj_games"]
    board = board.sort_values("vor_pg", ascending=False).reset_index(drop=True)
    board["my_rank"] = np.arange(1, len(board) + 1)
    board["rank_dur"] = board["vor_season"].rank(ascending=False,
                                                 method="first")
    board["key"] = norm(board["player"])
    return board.drop_duplicates("key")


# ------------------------------------------------------------------ measures
def assemble(year: int, df, transitions, weekly) -> pd.DataFrame:
    board = build_board(df, transitions, year)
    adp, meta = load_adp(year)
    act = actuals(weekly, year)
    act["key"] = norm(act["player_display_name"])
    act = act.sort_values("act_total", ascending=False).drop_duplicates("key")

    m = adp.merge(board[["key", "my_rank", "rank_dur", "proj_ppg", "vor_pg",
                         "is_rookie"]], on="key", how="left")
    m = m.merge(act[["key", "act_total", "act_ppg", "act_games"]], on="key",
                how="left")
    for c in ("act_total", "act_ppg", "act_games"):
        m[c] = m[c].fillna(0.0)
    m["adp_rank"] = m["adp"].rank(method="first")
    # A player the board never ranked cannot be drafted off it; park him at the
    # back rather than dropping him, which would quietly grade the board on a
    # friendlier population than ADP.
    m["my_rank"] = m["my_rank"].fillna(m["my_rank"].max() + 1)
    m["rank_dur"] = m["rank_dur"].fillna(m["rank_dur"].max() + 1)
    m["board_rank"] = m["my_rank"].rank(method="first")
    m["dur_rank"] = m["rank_dur"].rank(method="first")
    # ⚠️ Rank on ACTUAL VALUE OVER REPLACEMENT, not raw points. The first
    # version of this backtest ranked outcomes by raw season total pooled
    # across positions, where quarterbacks always win -- a starting QB scores
    # ~350 points and a top WR ~250. That systematically punished the board
    # for the one thing VOR exists to do: fade a position whose replacement
    # level is high. Joe Burrow "finishing 4th overall" in 2024 is what a
    # healthy starting QB does in a 1-QB league; it does not make him the 4th
    # most valuable draft asset. Replacement is recomputed from what actually
    # happened, using the same starter-filling logic as the board.
    act_repl = actual_replacement(m)
    m["act_value"] = m["act_total"] - m["position"].map(act_repl)
    m["act_rank"] = (-m["act_value"]).rank(method="first")
    print(f"  {year}: {len(m)} priced players, "
          f"{m['proj_ppg'].notna().sum()} matched to the board "
          f"({meta['total_drafts']} drafts, {meta['start_date']})")
    return m


def actual_replacement(m: pd.DataFrame) -> dict[str, float]:
    """Replacement level from ACTUAL season totals, same lineup logic."""
    pools = {p: sorted(m.loc[m["position"] == p, "act_total"], reverse=True)
             for p in POSITIONS}
    used = {p: STARTERS[p] * TEAMS for p in POSITIONS}
    for _ in range(FLEX_SLOTS * TEAMS):
        best, bp = -9e9, None
        for p in FLEX_POS:
            if used[p] < len(pools[p]) and pools[p][used[p]] > best:
                best, bp = pools[p][used[p]], p
        if bp is None:
            break
        used[bp] += 1
    return {p: (pools[p][min(used[p], len(pools[p])) - 1] if pools[p] else 0.0)
            for p in POSITIONS}


def measure_a(m: pd.DataFrame, year: int) -> dict:
    out = {}
    for label, sub in (("all priced", m), ("top 120 by ADP",
                                           m[m["adp_rank"] <= 120])):
        rb = spearmanr(sub["board_rank"], sub["act_rank"]).statistic
        rd = spearmanr(sub["dur_rank"], sub["act_rank"]).statistic
        ra = spearmanr(sub["adp_rank"], sub["act_rank"]).statistic
        out[label] = (rb, rd, ra, len(sub))
    return out


def measure_b(m: pd.DataFrame) -> pd.DataFrame:
    """Only where board and market disagree -- the board's entire claim.

    'Board likes him' means the board ranks him meaningfully ahead of his ADP.
    If those players systematically outperform their ADP-implied finish, the
    disagreements are informative. If they do not, the board is noise dressed
    as an edge, however good its aggregate correlation looks.
    """
    d = m[m["adp_rank"] <= 150].copy()
    d["gap"] = d["adp_rank"] - d["board_rank"]      # +ve = board likes him
    d["beat_adp"] = d["act_rank"] < d["adp_rank"]   # finished better than price
    rows = []
    for label, sub in (
            ("board LIKES (>=24 ranks)", d[d["gap"] >= 24]),
            ("mild like (8-23)", d[d["gap"].between(8, 23)]),
            ("agree (within 8)", d[d["gap"].abs() < 8]),
            ("mild fade (8-23)", d[d["gap"].between(-23, -8)]),
            ("board FADES (<=-24 ranks)", d[d["gap"] <= -24])):
        if len(sub) < 5:
            continue
        rows.append({
            "cohort": label, "n": len(sub),
            "beat ADP %": 100 * sub["beat_adp"].mean(),
            "mean act pts": sub["act_total"].mean(),
            "mean ADP rank": sub["adp_rank"].mean(),
            "mean act rank": sub["act_rank"].mean(),
            "rank gained": (sub["adp_rank"] - sub["act_rank"]).mean(),
        })
    return pd.DataFrame(rows)


def draft_sim(m: pd.DataFrame, year: int) -> pd.DataFrame:
    """A 12-team snake where one seat drafts by board and the rest by ADP.

    Roster needs are respected so neither strategy can hoard one position.
    Scored on actual season points of the starting lineup.
    """
    # Realistic caps for a 15-round, 1QB/2RB/2WR/1TE/2FLEX roster. The first
    # version allowed FOUR tight ends, and the board strategy duly drafted
    # them -- VOR loves an elite TE, and nothing stopped it stacking the
    # position. That was measuring my cap logic, not the board.
    CAP = {"QB": 2, "RB": 6, "WR": 7, "TE": 2}
    rounds = 15

    def run(my_slot: int, my_key: str) -> float:
        pool = m.sort_values("adp_rank").to_dict("records")
        taken, rosters = set(), {t: [] for t in range(1, TEAMS + 1)}
        order = []
        for r in range(1, rounds + 1):
            seats = range(1, TEAMS + 1) if r % 2 else range(TEAMS, 0, -1)
            order.extend(seats)
        for pick_no, seat in enumerate(order, start=1):
            key = my_key if seat == my_slot else "adp_rank"
            cand = sorted((p for p in pool if p["key"] not in taken),
                          key=lambda p: p[key])
            counts = {}
            for p in rosters[seat]:
                counts[p["position"]] = counts.get(p["position"], 0) + 1
            pickd = None
            for p in cand:
                if counts.get(p["position"], 0) < CAP[p["position"]]:
                    pickd = p
                    break
            if pickd is None:
                pickd = cand[0] if cand else None
            if pickd is None:
                break
            taken.add(pickd["key"])
            rosters[seat].append(pickd)
        return lineup_points(rosters[my_slot])

    def lineup_points(roster: list) -> float:
        by = {p: sorted((r for r in roster if r["position"] == p),
                        key=lambda r: -r["act_total"]) for p in POSITIONS}
        pts, used = 0.0, {p: 0 for p in POSITIONS}
        for p, n in STARTERS.items():
            for i in range(n):
                if i < len(by[p]):
                    pts += by[p][i]["act_total"]
                    used[p] += 1
        for _ in range(FLEX_SLOTS):
            best, bp = -1, None
            for p in FLEX_POS:
                if used[p] < len(by[p]) and by[p][used[p]]["act_total"] > best:
                    best, bp = by[p][used[p]]["act_total"], p
            if bp:
                pts += best
                used[bp] += 1
        return pts

    rows = []
    for slot in range(1, TEAMS + 1):
        b = run(slot, "board_rank")
        d = run(slot, "dur_rank")
        a = run(slot, "adp_rank")
        rows.append({"slot": slot, "vor_pg": b, "rank_dur": d, "adp": a,
                     "d_pg": b - a, "d_dur": d - a})
    return pd.DataFrame(rows)


def main() -> None:
    weekly = P.load_weekly_full()
    df = P.to_frame(P.season_features(weekly), nfl.load_players())
    transitions = P.build_transitions(df)
    wk = pl.read_parquet(DATA / "weekly_stats_2016_2025.parquet")

    print("=" * 78)
    print("REBUILDING THE BOARD FOR EACH TEST YEAR (no data from that year on)")
    print("=" * 78)
    frames = {y: assemble(y, df, transitions, wk) for y in TEST_YEARS}

    print("\n" + "=" * 78)
    print("A. RANK CORRELATION WITH ACTUAL SEASON POINTS")
    print("=" * 78)
    print(f"  {'year':<6}{'population':<18}{'n':>5}{'vor_pg':>9}{'rank_dur':>10}"
          f"{'ADP':>9}{'best edge':>11}")
    for y, m in frames.items():
        for label, (rb, rd, ra, n) in measure_a(m, y).items():
            print(f"  {y:<6}{label:<18}{n:>5}{rb:>9.3f}{rd:>10.3f}{ra:>9.3f}"
                  f"{max(rb, rd) - ra:>+11.3f}")
    print("\n  Positive 'edge' means the board ordered players better than the")
    print("  market did. This is necessary but NOT sufficient -- see B.")

    print("\n" + "=" * 78)
    print("B. THE DISAGREEMENT TEST -- where the board's value has to come from")
    print("=" * 78)
    for y, m in frames.items():
        print(f"\n  --- {y} (top 150 by ADP) ---")
        t = measure_b(m)
        print(t.to_string(index=False, float_format=lambda v: f"{v:7.1f}"))
    print("\n  'rank gained' is ADP rank minus actual finish rank: positive")
    print("  means the player outperformed his price. The board earns its")
    print("  place only if LIKES gains more than FADES.")

    print("\n" + "=" * 78)
    print("C. SIMULATED DRAFT -- board seat vs the same seat drafting by ADP")
    print("=" * 78)
    for y, m in frames.items():
        sim = draft_sim(m, y)
        print(f"\n  --- {y} ---")
        print(f"    {'slot':<6}{'vor_pg':>9}{'rank_dur':>10}{'ADP':>9}"
              f"{'d(pg)':>8}{'d(dur)':>8}")
        for _, r in sim.iterrows():
            print(f"    {int(r['slot']):<6}{r['vor_pg']:>9.0f}"
                  f"{r['rank_dur']:>10.0f}{r['adp']:>9.0f}"
                  f"{r['d_pg']:>+8.0f}{r['d_dur']:>+8.0f}")
        print(f"    {'MEAN':<6}{sim['vor_pg'].mean():>9.0f}"
              f"{sim['rank_dur'].mean():>10.0f}{sim['adp'].mean():>9.0f}"
              f"{sim['d_pg'].mean():>+8.0f}{sim['d_dur'].mean():>+8.0f}")
        print(f"    slots won:  vor_pg {int((sim['d_pg'] > 0).sum())}/12, "
              f"rank_dur {int((sim['d_dur'] > 0).sum())}/12")
    print("\n  One injury swings a season, so read C as a direction, not a")
    print("  measurement. B is the number to trust.")


if __name__ == "__main__":
    main()
