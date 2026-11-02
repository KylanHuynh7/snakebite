"""Phase 3: unified 2026 draft board -- veterans + rookies + ADP availability.

Three pieces:
1. Rookie projections. 2026 rookies have no NFL stats, so train on the
   2016-2025 draft classes: NFL draft capital (log pick) + position + age ->
   rookie-season PPR points per game, with the same LightGBM quantile setup as
   Phase 2. Draft capital is the strongest public predictor of rookie fantasy
   output; college-stat translation can be layered on later via the CFBD API.
2. Market ADP. Pulled from the FantasyFootballCalculator API, which publishes a
   real per-player standard deviation of draft slot -- so availability uses
   measured spread instead of the old sigma = 1 + 0.14 * ADP approximation
   (that formula ran ~35-40% hot at every tier, which made players look likelier
   to fall to you than they are). FantasyPros is no longer usable: it moved its
   ADP table behind client-side rendering, so the HTML has no table to scrape.
3. Availability. For a snake-draft slot, P(player available at your pick) =
   1 - NormalCDF(pick; ADP, sigma).

usage:  phase3_draft_board.py [slot]      slot omitted -> compare all 12 slots

Output: output/draft_board_2026.csv + a pick-by-pick plan printed for the slot.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import requests
from scipy.stats import norm

import nflreadpy as nfl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

# 15, not 16. The extension carried the same off-by-one (bug #3, Aug 9): a
# 16th round in a 15-round league makes `my_picks()` emit a pick past the end
# of the draft -- slot 4 would price the board against pick 189 of 180. It is
# currently masked here because every caller slices to [:8], which is exactly
# how it survived the last sweep. FFC's own metadata independently says 15.
import league

_L = league.settings()
TEAMS, ROUNDS = _L["teams"], _L["rounds"]
POSITIONS = ["QB", "RB", "WR", "TE"]
# 2026 league: 1QB / 2RB / 2WR / 1TE / 2FLEX (+ DST, K, 3 IR).
# Two flex spots put 12 * 7 = 84 skill players in starting lineups league-wide,
# which is far deeper than the 10-team QB11/RB24/WR24/TE11 baseline this used to
# hard-code. Rather than guess how the flex splits across RB/WR/TE, fill the
# lineups from the projections and read the replacement level back off.
STARTERS = _L["lineup"]
FLEX_SLOTS = _L["flex"]
FLEX_POS = ["RB", "WR", "TE"]
ADP_URL = (f"https://fantasyfootballcalculator.com/api/v1/adp/{_L['scoring']}"
           f"?teams={_L['teams']}&year={_L['season']}&position=all")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Optional per-player annotations carried onto the board ("FADE -- ...",
# "TARGET AT 46 -- ..."): a dict of exact board name -> short label, read from
# data/stances_private.py if it exists. Nothing here feeds a probability; the
# availability numbers are the same with or without it.
def _load_stances() -> dict:
    path = DATA / "stances_private.py"
    if not path.exists():
        return {}
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns.get("STANCES", {})


STANCES = _load_stances()

ROOKIE_FEATURES = ["log_pick", "round", "draft_age",
                   "pos_QB", "pos_RB", "pos_TE", "pos_WR"]

# The rookie floor is miscalibrated the same way the veteran one was, but by a
# DIFFERENT amount -- alpha 0.20 puts 24.3% of rookies below the floor against
# 29.0% for veterans, measured walk-forward over the 2020-2025 classes. So the
# veteran fix cannot be copied across: alpha 0.10 would overcorrect rookies to
# 15.7%. 0.14 lands at 19.6%. Two models, two data sets, two calibrations.
# Evidence here is thinner than for the veterans (6 classes at ~60 players
# each), so this is the honest correction rather than a precise one.
# See `scripts/calibrate_quantiles.py` for the veteran equivalent.
ROOKIE_FLOOR_ALPHA = 0.14

# A stance that argues against taking a player argues against his PRICE, not
# against his existence. Kept as a tuple so the plan can gate on it by pick.
NEGATIVE = ("AVOID", "FADE", "RISK", "IR STASH")


def suppressed(board: pd.DataFrame, pick: int) -> pd.Series:
    """Price-aware replacement for the old blanket AVOID filter.

    The plan used to drop every note starting with "AVOID" at EVERY pick, which
    silently deleted five players from the board no matter how far they fell --
    a stale June label cost real optionality rather than just being a label.
    Hide a negatively-flagged player only while you would still be paying above
    model value for him (your pick is ahead of his model rank); past that he
    resurfaces, marked with a `!`.
    """
    return board["note"].str.startswith(NEGATIVE) & (pick < board["my_rank"])


# Given-name differences the suffix/punctuation rules below cannot reach: the
# ADP source and nflverse disagree on what the man is called, not on spelling.
# Keyed on the ALREADY-NORMALISED name. Add to this only after seeing the
# "priced player(s) did not join the board" report name the pair -- a wrong
# alias is a silent mis-join, which is the failure this exists to prevent.
NAME_ALIASES = {
    "kenny gainwell": "kenneth gainwell",     # FFC "Kenny", nflverse "Kenneth"
}


def norm_name(s: pd.Series) -> pd.Series:
    s = s.str.lower().str.replace(r"[^a-z\s]", "", regex=True)
    s = s.str.replace(r"\b(jr|sr|ii|iii|iv|v)$", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.replace(NAME_ALIASES)


def fetch_adp() -> pd.DataFrame:
    """FantasyFootballCalculator consensus PPR ADP, with measured spread.

    Note the API's `teams` parameter is cosmetic -- 8/10/12/14 all return the
    same rows and the same total_drafts -- so this is a pooled consensus, not
    12-team-specific ADP. That is fine here: ADP as an absolute pick number is a
    market ranking, and the 12-team effects that matter (where your picks land,
    where replacement level sits) are handled elsewhere.
    """
    cache = league.adp_cache()
    if not cache.exists():
        cache.write_text(requests.get(ADP_URL, headers=UA, timeout=30).text)
    d = json.loads(cache.read_text())
    m = d["meta"]
    print(f"ADP: FFC {m['type']}, {m['total_drafts']} drafts, "
          f"{m['start_date']} to {m['end_date']}")

    t = pd.DataFrame(d["players"])
    # Market rank is computed BEFORE dropping K/DEF, because that is the rank
    # space league_tendencies trains in. Filtering first shifts every player
    # past the first kicker (market rank ~98) by up to 8 ranks and would ask
    # the calibration curve a different question than it was fitted on --
    # the same train/serve symmetry rule as phase2's destination-team keying.
    t["mrank"] = t["adp"].rank(method="first")
    t = t[t["position"].isin(POSITIONS)].copy()
    t["key"] = norm_name(t["name"])
    # thinly-drafted players get an implausibly tight stdev, so floor it
    t["adp_sigma"] = np.maximum(t["stdev"], 0.5 + 0.04 * t["adp"])

    # league sigma calibration (league_tendencies.py, validated by
    # replay_brier.py on 2023-25 replays): this league tracks the market
    # TIGHTER than FFC's own drafters in round 1 (~0.7x) and looser from
    # there, ~2.2x by the middle rounds. mult(rank) = a + b*ln(rank),
    # clipped; 16% Brier improvement out-of-sample, and rank-dependence is
    # what keeps the conservation identity intact (a flat 2.07x broke it at
    # 1.32x). Every availability number downstream (board columns, panel
    # availAt, draft_advise) inherits adp_sigma, so this is the single
    # integration point. No tendencies file -> exactly the old behaviour.
    tend = league.tendencies_path()
    t["adp_shift"] = 0.0
    if tend.exists():
        c = json.loads(tend.read_text())["sigma_curve"]
        mult = np.clip(
            np.interp(np.log(np.maximum(t["mrank"], 1.0)),
                      np.log([p[0] for p in c["pts"]]),
                      [p[1] for p in c["pts"]]),
            c["floor"], c["cap"])
        # drift: this league takes players about a pick LATER than their
        # market rank (+0.285 sigma, pooled over ranks 1-84). One pooled
        # number, not a per-bucket curve -- the per-bucket estimates are not
        # individually significant. Converted to picks with the RAW sigma,
        # before the spread multiplier is applied.
        taper = np.clip(np.interp(t["mrank"], [c["drift_full"], c["drift_zero"]],
                                  [1.0, 0.0]), 0.0, 1.0)
        t["adp_shift"] = (c["drift"] * taper * t["adp_sigma"]).round(2)
        t["adp_sigma"] *= mult
        pts = ", ".join(f"r{p[0]:g}:x{p[1]:.2f}" for p in c["pts"])
        print(f"ADP sigma x spread(rank) [{pts}]; mean +{c['drift']:.3f}"
              f" sigma to rank {c['drift_full']} (league history, {tend.name})")
    else:
        print("no league_tendencies.json -- using raw FFC sigma "
              "(run league_tendencies.py to calibrate)")
    return t[["key", "position", "adp", "mrank", "adp_sigma", "adp_shift",
              "bye", "times_drafted"]]


# the rookie source spells the Rams LAR, nflverse says LA; same for a few
# others that have moved or been rebranded. A silent join miss here would
# just blank the label, which is exactly the failure mode the STANCES bug
# taught us to make loud.
TEAM_ALIAS = {"LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA",
              "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
              "HST": "HOU"}


def team_offense_labels(teams: pd.Series) -> pd.Series:
    """Per-player tag for the offence he plays on in 2026.

    Comes from `offense_regression.py --write`. This is a TEAM-level
    scoring-environment forecast, not a player projection -- it says whether
    the touchdown pie is expected to shrink, not who gets a slice. Absent
    file, or an unmatched team, yields an empty label rather than a guess.
    """
    path = DATA / "team_offense_2026.json"
    if not path.exists():
        print("no team_offense_2026.json -- no offence labels "
              "(run offense_regression.py --write)")
        return pd.Series([""] * len(teams), index=teams.index)
    tbl = json.loads(path.read_text())
    meta = tbl.pop("_meta", {})
    norm = teams.fillna("").map(lambda t: TEAM_ALIAS.get(t, t))
    lab = norm.map(lambda t: tbl.get(t, {}).get("label", ""))
    missing = sorted({t for t in norm[lab == ""] if t})
    print(f"offence labels: {(lab != '').sum()}/{len(lab)} players "
          f"({meta.get('model', '?')})")
    if missing:
        print(f"  !! no offence forecast for: {', '.join(missing)} "
              f"-- add to TEAM_ALIAS if this is a spelling difference")
    return lab


def team_ctx_labels(teams: pd.Series, positions: pd.Series) -> pd.Series:
    """Offensive line and playcaller context, from the hand-entered tables.

    Both are ANNOTATIONS, not model features, and the distinction is the whole
    point. `validate_ol_playcaller.py` shows the OL table measures real
    football -- his 2025 run-block rank correlates -0.54 with yards per carry
    and -0.55 with team RB fantasy PPG, and his top-8 lines outrushed his
    bottom-8 by 0.64 ypc. But the column that matters for 2026 is his
    FORECAST, and a forecast for an unplayed season cannot be trained or
    ablated against anything. Encoding it as a model feature would import one
    person's opinion with a decimal point on it and no way to ever check it.

    So it goes on the board where a human weighs it, exactly like the offence
    tags. Shown for RBs (where run blocking is the argument) and for everyone
    when the playcaller is new -- a new playcaller is the single largest
    unmodelled uncertainty on any offence, and our own model is blind to it.
    """
    ol_p = DATA / "ol_ratings_2026.csv"
    pc_p = DATA / "playcallers_2026.csv"
    if not (ol_p.exists() and pc_p.exists()):
        print("no OL/playcaller tables -- skipping team-context labels")
        return pd.Series([""] * len(teams), index=teams.index)
    ol = pd.read_csv(ol_p, comment="#").set_index("team")
    pc = pd.read_csv(pc_p, comment="#").set_index("team")

    out = []
    for t, pos in zip(teams.fillna("").map(lambda x: TEAM_ALIAS.get(x, x)),
                      positions):
        bits = []
        if t in ol.index and pos in ("RB", "QB"):
            r = ol.loc[t]
            arrow = {2: "++", 1: "+", 0: "", -1: "-"}.get(int(r["direction"]), "")
            bits.append(f"OL {r['rank_2026']:.1f}/5{arrow}")
        if t in pc.index:
            r = pc.loc[t]
            if int(r["first_time"]) == 1:
                bits.append(f"NEW PC (1st time, {r['lineage']} tree)")
            elif int(r["new_team"]) == 1:
                rb = "" if pd.isna(r["rb_rank"]) else f", RB{int(r['rb_rank'])}"
                bits.append(f"new PC {r['playcaller'].split('/')[0]}{rb}")
        out.append(" · ".join(bits))
    lab = pd.Series(out, index=teams.index)
    print(f"team context: {(lab != '').sum()}/{len(lab)} players labelled "
          f"(OL + playcaller, annotations only)")
    return lab


def replacement_levels(board: pd.DataFrame) -> dict[str, float]:
    """Fill every team's starting lineup from the projections; the first player
    left over at each position is replacement level."""
    pools = {p: board[board["position"] == p]
             .sort_values("proj_ppg", ascending=False)["proj_ppg"].tolist()
             for p in POSITIONS}
    used = {p: STARTERS[p] * TEAMS for p in POSITIONS}
    for _ in range(FLEX_SLOTS * TEAMS):  # best remaining flex-eligible player
        best = max(FLEX_POS, key=lambda p: pools[p][used[p]])
        used[best] += 1
    for p in POSITIONS:
        print(f"  {p} replacement = {p}{used[p] + 1} "
              f"({pools[p][used[p]]:.2f} PPG)")
    return {p: pools[p][used[p]] for p in POSITIONS}


def rookie_training() -> tuple[pd.DataFrame, pd.DataFrame]:
    """historical drafted skill players + their rookie-season PPG, and the
    2026 class."""
    dp = nfl.load_draft_picks().filter(
        pl.col("position").is_in(POSITIONS) & (pl.col("season") >= 2016)
    ).select("season", "round", "pick", "team", "gsis_id",
             "pfr_player_name", "position", "age").to_pandas()
    weekly = pl.read_parquet(DATA / "weekly_stats_2016_2025.parquet")
    rook = (
        weekly.group_by(["player_id", "season"])
        .agg(games=pl.col("fantasy_points_ppr").count(),
             ppr_pg=pl.col("fantasy_points_ppr").mean())
        .to_pandas()
        .rename(columns={"player_id": "gsis_id"})
    )
    dp = dp.merge(rook, on=["gsis_id", "season"], how="left")
    dp["games"] = dp["games"].fillna(0)
    dp["log_pick"] = np.log(dp["pick"])
    dp["draft_age"] = dp["age"].fillna(dp["age"].median())
    for pos in POSITIONS:
        dp[f"pos_{pos}"] = (dp["position"] == pos).astype(int)
    hist = dp[dp["season"] <= 2025]
    cls26 = dp[dp["season"] == 2026].copy()
    return hist, cls26


def project_rookies() -> pd.DataFrame:
    hist, cls26 = rookie_training()
    played = hist[hist["games"] >= 4]

    # backtest: hold out the 2025 class
    tr, te = played[played["season"] <= 2024], played[played["season"] == 2025]
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=300,
                          learning_rate=0.03, num_leaves=7,
                          min_child_samples=20, random_state=7, verbose=-1)
    m.fit(tr[ROOKIE_FEATURES], tr["ppr_pg"])
    mae = (m.predict(te[ROOKIE_FEATURES]) - te["ppr_pg"]).abs().mean()
    naive = (te.groupby("position")["ppr_pg"].transform(
        lambda s: tr[tr["position"] == s.name]["ppr_pg"].mean())
        - te["ppr_pg"]).abs().mean()
    print(f"rookie model backtest (2025 class, n={len(te)}): "
          f"MAE {mae:.2f} PPG vs position-mean naive {naive:.2f}")

    models = {}
    for q in (ROOKIE_FLOOR_ALPHA, 0.5, 0.8):
        mq = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=300,
                               learning_rate=0.03, num_leaves=7,
                               min_child_samples=20, random_state=7, verbose=-1)
        mq.fit(played[ROOKIE_FEATURES], played["ppr_pg"])
        models[q] = mq

    cls26["proj_ppg"] = models[0.5].predict(cls26[ROOKIE_FEATURES])
    cls26["floor_ppg"] = models[ROOKIE_FLOOR_ALPHA].predict(cls26[ROOKIE_FEATURES])
    cls26["ceil_ppg"] = models[0.8].predict(cls26[ROOKIE_FEATURES])
    # Independent quantile fits are not guaranteed monotone; the median is what
    # the board ranks on, so clamp the interval around it. Mirrors phase2.
    cls26["floor_ppg"] = np.minimum(cls26["floor_ppg"], cls26["proj_ppg"])
    cls26["ceil_ppg"] = np.maximum(cls26["ceil_ppg"], cls26["proj_ppg"])
    games_by_round = hist.groupby("round")["games"].mean()
    cls26["proj_games"] = cls26["round"].map(games_by_round).round(1)
    cls26["proj_total"] = (cls26["proj_ppg"] * cls26["proj_games"]).round(0)
    cls26["rookie_info"] = ("R" + cls26["round"].astype(str)
                            + " pick " + cls26["pick"].astype(str))
    cls26 = cls26.rename(columns={"pfr_player_name": "player",
                                  "team": "team_2026"})
    # draft data uses PFR team codes; the rest of the board uses nflverse codes
    cls26["team_2026"] = cls26["team_2026"].replace({
        "NOR": "NO", "LVR": "LV", "KAN": "KC", "SFO": "SF",
        "TAM": "TB", "GNB": "GB", "NWE": "NE"})
    keep = ["player", "position", "team_2026", "proj_ppg", "floor_ppg",
            "ceil_ppg", "proj_games", "proj_total", "rookie_info"]
    return cls26[keep]


def my_picks(slot: int) -> list[int]:
    return [(r - 1) * TEAMS + slot if r % 2 else r * TEAMS - slot + 1
            for r in range(1, ROUNDS + 1)]


def build_board(slot: int) -> pd.DataFrame:
    vets = pd.read_csv(OUT / "projections_2026.csv")
    vets = vets.rename(columns={"player_display_name": "player"})
    vets["rookie_info"] = ""
    cols = ["player", "position", "team_2026", "proj_ppg", "floor_ppg",
            "ceil_ppg", "proj_games", "proj_total", "rookie_info", "td_dep",
            "status", "status_note"]
    # Rookies have no prior TD rate to be dependent on, so they carry an empty
    # flag rather than being scored against a distribution they were not in.
    # Same for status: nothing known is the honest default, not "healthy".
    board = pd.concat([vets[cols], project_rookies()], ignore_index=True)
    for c in ("td_dep", "status", "status_note"):
        board[c] = board[c].fillna("")

    print(f"\nreplacement level, {TEAMS} teams x "
          f"({'/'.join(f'{n}{p}' for p, n in STARTERS.items())}+{FLEX_SLOTS}FLX):")
    repl = replacement_levels(board)
    out = []
    for pos in POSITIONS:
        p = board[board["position"] == pos].copy()
        p["vor_pg"] = (p["proj_ppg"] - repl[pos]).round(2)
        out.append(p)
    board = pd.concat(out).sort_values("vor_pg", ascending=False)
    board["my_rank"] = range(1, len(board) + 1)

    # vor_pg is a PER-GAME number and my_rank sorts on it, so games played
    # currently affect the ranking not at all: two WRs at the same PPG rank
    # identically whether one projects 16 games or 10. That is a real gap for
    # players coming off a lost season (Nabers, Kittle, Rice). Season-long VOR
    # is the same value spread over the games he is expected to play, assuming
    # the missed weeks get filled at replacement level -- which a 12-team
    # waiver wire plus 3 IR spots makes a fair assumption.
    board["vor_season"] = (board["vor_pg"] * board["proj_games"]).round(1)
    board["rank_dur"] = (board["vor_season"]
                         .rank(ascending=False, method="first").astype(int))

    board["key"] = norm_name(board["player"])
    adp = fetch_adp()
    board = board.merge(adp, on=["key", "position"], how="left")

    # A priced player whose ADP does not land is WORSE than an unpriced one:
    # the panel is ADP-ORDERED (v2.3), so he is not merely missing a number, he
    # is missing from the ordering entirely, and gets no availability either.
    # Found Aug 13 2026: FFC spells him "Kenny Gainwell", nflverse "Kenneth",
    # so an ADP-89.6 back sat on the board unpriced at my_rank 143. Same failure
    # as "Chris Godwin" vs "Chris Godwin Jr." -- silence, not an error. So say
    # it out loud, and separate the two causes, because they need different
    # fixes: a NEAR-MISS is a name to alias here; "no board row" means the
    # player never reached the projections at all (see MIN_GAMES in phase2).
    landed = set(board.loc[board["adp"].notna(), "key"])
    lost = adp[~adp["key"].isin(landed)].sort_values("adp")
    if len(lost):
        import difflib
        keys = list(board["key"])
        print(f"\n!! {len(lost)} priced player(s) did not join the board:")
        for r in lost.itertuples():
            near = difflib.get_close_matches(r.key, keys, n=1, cutoff=0.8)
            hint = (f"NEAR-MISS -> board has "
                    f"'{board.loc[board['key'] == near[0], 'player'].iloc[0]}'"
                    if near else "no board row at all (missing from phase2)")
            print(f"   ADP {r.adp:6.1f}  {r.key:22s} {r.position}  {hint}")
    board["off_2026"] = team_offense_labels(board["team_2026"])
    board["team_ctx"] = team_ctx_labels(board["team_2026"], board["position"])
    # STANCES is keyed on the exact board name, so a stale or mistyped key just
    # disappears -- "Chris Godwin" vs "Chris Godwin Jr." cost a silently missing
    # AVOID. A dropped stance is worse than a wrong one: you never see it.
    unmatched = sorted(set(STANCES) - set(board["player"]))
    if unmatched:
        print(f"\n!! {len(unmatched)} STANCE(S) MATCHED NO PLAYER — silently "
              f"dropped from the board:")
        for name in unmatched:
            near = [p for p in board["player"]
                    if name.split()[-1].rstrip(".") in p or p.split()[-1] in name]
            hint = f"   did you mean: {', '.join(near[:3])}" if near else ""
            print(f"     {name!r}{hint}")
        print()
    board["note"] = board["player"].map(STANCES).fillna("")

    picks = my_picks(slot)
    # Price EVERY pick, not the first 8. The old picks[:8] cap silently hid
    # rounds 9-15 (99/118/123/142/147/166/171) -- which is where the bench is
    # drafted, and the reason the agenda kept omitting pick 99 entirely.
    for i, p in enumerate(picks, start=1):
        board[f"avail_pk{p}"] = (
            1 - norm.cdf(p, loc=board["adp"] + board["adp_shift"],
                         scale=board["adp_sigma"])
        ).round(2)
    # latest of your picks where this player is still >=50% to be there
    av = board[[f"avail_pk{p}" for p in picks]].values
    safe_idx = (av >= 0.5).sum(axis=1)  # availability only decreases
    board["safe_until"] = [picks[i - 1] if i else 0 for i in safe_idx]
    board.loc[board["adp"].isna(), "safe_until"] = picks[-1]

    front = ["my_rank", "rank_dur", "player", "position", "team_2026",
             "proj_ppg", "floor_ppg", "ceil_ppg", "proj_games", "proj_total",
             "vor_pg", "vor_season", "adp", "mrank", "adp_sigma", "adp_shift",
             "times_drafted", "safe_until", "rookie_info", "off_2026",
             "team_ctx", "td_dep", "status", "status_note", "note", "bye"]
    board = board[front + [f"avail_pk{p}" for p in picks]]
    for c in ("proj_ppg", "floor_ppg", "ceil_ppg"):
        board[c] = board[c].round(2)
    board.to_csv(OUT / "draft_board_2026.csv", index=False)
    return board


def print_plan(board: pd.DataFrame, slot: int) -> None:
    picks = my_picks(slot)[:8]
    drafted = board["adp"].notna()
    print(f"\n=== Pick-by-pick plan, {TEAMS}-team snake, slot {slot} ===")
    for p in picks:
        live = board[drafted & (board[f"avail_pk{p}"] >= 0.5)
                     & ~suppressed(board, p)]
        live = live.nsmallest(5, "my_rank")
        names = ", ".join(
            f"{'!' if r['note'].startswith(NEGATIVE) else ''}{r['player']} "
            f"({r['position']}, {int(r[f'avail_pk{p}']*100)}%)"
            for _, r in live.iterrows()
        )
        print(f"  pick {p:>3}: {names}")
    print("  ! = flagged stance, surfacing only now that the price has come to "
          "you; read the note before taking him")

    print("\n=== Market mismatches: model rank far better than ADP ===")
    m = board[drafted & (board["my_rank"] <= 60)].copy()
    m["adp_vs_rank"] = m["adp"] - m["my_rank"]
    print(m.nlargest(10, "adp_vs_rank")[
        ["my_rank", "player", "position", "proj_ppg", "adp", "note"]
    ].to_string(index=False))

    print("\n=== Durability check: per-game rank vs season-long rank ===")
    print("(my_rank ignores games played entirely; rank_dur spreads the same "
          "VOR over\n projected games. Big gaps are the players the per-game "
          "board flatters or buries.)")
    d = board[drafted & ((board["my_rank"] <= 60) | (board["rank_dur"] <= 60))]
    d = d.assign(shift=d["my_rank"] - d["rank_dur"])
    cols = ["my_rank", "rank_dur", "shift", "player", "position", "proj_ppg",
            "proj_games", "vor_season", "adp"]
    print("  flattered by the per-game view (fall on games played):")
    print(d.nsmallest(6, "shift")[cols].to_string(index=False))
    print("  buried by it (durable, and the board gives no credit):")
    print(d.nlargest(6, "shift")[cols].to_string(index=False))

    print("\n=== Rookies on the board (top 12 by ADP) ===")
    r = board[(board["rookie_info"] != "") & drafted]
    print(r.nsmallest(12, "adp")[
        ["my_rank", "player", "position", "team_2026", "rookie_info",
         "proj_ppg", "ceil_ppg", "adp", "safe_until"]].to_string(index=False))


def print_tiers(board: pd.DataFrame, pos: str, n: int = 14) -> None:
    """Positional tiers -- the slot-agnostic part of draft prep. Tier breaks
    are drops of >=0.8 VOR PPG between consecutive players."""
    p = board[board["position"] == pos].nlargest(n, "vor_pg")
    tier, prev = 1, None
    print(f"\n--- {pos} tiers ---")
    for _, r in p.iterrows():
        if prev is not None and prev - r["vor_pg"] >= 0.8:
            tier += 1
            print()
        adp = "--" if pd.isna(r["adp"]) else f"{r['adp']:.1f}"
        print(f"  T{tier}  {r['player']:<24} {r['proj_ppg']:>5.1f} PPG  "
              f"VOR {r['vor_pg']:>5.1f}  ADP {adp:>5}  {r['note']}")
        prev = r["vor_pg"]


def print_slot_survey(board: pd.DataFrame) -> None:
    """Draft slot is not known yet, so show what rounds 1-2 look like from
    every seat: the best player who is >=50% to reach each of your first
    two picks."""
    print(f"\n=== Rounds 1-2 by draft slot ({TEAMS}-team snake) ===")
    for slot in range(1, TEAMS + 1):
        p1, p2 = my_picks(slot)[:2]
        best = []
        for pk in (p1, p2):
            avail = 1 - norm.cdf(pk, loc=board["adp"] + board["adp_shift"],
                                 scale=board["adp_sigma"])
            ok = board["adp"].notna() & ~suppressed(board, pk)
            c = board[ok & (avail >= 0.5)].nsmallest(1, "my_rank")
            best.append(f"{c.iloc[0]['player']} ({c.iloc[0]['position']})"
                        if len(c) else "--")
        print(f"  slot {slot:>2}  pk {p1:>2}: {best[0]:<26} "
              f"pk {p2:>2}: {best[1]}")


def main() -> None:
    slot = int(sys.argv[1]) if len(sys.argv) > 1 else None
    board = build_board(slot or 6)
    show = ["my_rank", "player", "position", "team_2026", "proj_ppg",
            "ceil_ppg", "vor_pg", "adp", "safe_until", "note"]
    print("\n=== Top 40, unified 2026 board (vets + rookies) ===")
    print(board[show].head(40).to_string(index=False))
    for pos in POSITIONS:
        print_tiers(board, pos)
    if slot is None:
        print("\n(no slot given -- surveying all seats)")
        print_slot_survey(board)
    else:
        print_plan(board, slot)


if __name__ == "__main__":
    main()
