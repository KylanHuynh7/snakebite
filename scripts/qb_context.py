"""QB dependency, turned into a board annotation.

WHAT THIS SHIPS, AND WHAT IT DELIBERATELY DOES NOT.

`qb_dependency.py` measured the effect within-player over 2016-2025:

  * **41% of team-seasons lose 3+ starts from their week-1 QB** (66% lose >=1).
    One team in three. This is the number that should change a decision.
  * Cost when it happens: **-0.53 PPG** for pass catchers generally,
    **-1.0 to -1.3 for true #1 targets**.
  * The "gunslinger backup is different" split is NOT shipped: +0.61 PPG,
    t=1.37, and it vanishes entirely (-0.20, t=-0.22) for the #1 targets the
    claim was actually about. Underpowered, and its stated mechanism runs
    backwards (a gunslinger REDUCES the WR1's air-yards share).

So this is a **downside annotation, not a projection adjustment**. The expected
cost is ~0.5 PPG once multiplied by the base rate -- too small to move a
ranking honestly, and the real information is variance. It answers "how much of
this receiver's value rides on one man staying upright?", which is a question
the projection model structurally cannot see: the model reads a receiver's own
past PPG, and that number is an AVERAGE over whoever happened to be throwing.

NOTHING HERE IS HAND-MAINTAINED. Both inputs are derived:
  * the presumed 2026 starter is the best-ADP QB on each roster, which is what
    the market itself believes;
  * durability comes from measured starts missed 2023-25.
That matters because `player_status_2026.csv` is already the one input that
decays, and adding a second hand-kept table before a draft is a maintenance
burden that would go stale exactly when it is needed.

⚠️ Rookie and newly-installed starters have no NFL start history, so they get
the league base rate rather than a flattering zero. Absence of evidence is not
durability.

usage: qb_context.py [--write]      # --write updates data/qb_context_2026.json
"""

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "qb_context_2026.json"
CACHE = DATA / "qb_dep_weekly.parquet"

HIST = [2023, 2024, 2025]
MIN_ATT_START = 10

# measured in qb_dependency.py -- quoted, never re-derived by hand
BASE_MISS3 = 0.41          # P(team loses 3+ starts from its week-1 QB)
COST_WR1 = -1.3            # PPG, true #1 target (share >= 22%)
COST_OTHER = -0.53         # PPG, pass catchers generally


def qb_durability() -> dict:
    """Starts made vs starts available, per QB, over HIST."""
    if not CACHE.exists():
        raise SystemExit("run qb_dependency.py first to build "
                         f"{CACHE.relative_to(ROOT)}")
    # ⚠️ weeks 19-22 are the POSTSEASON. Leaving them in gives playoff teams up
    # to 4 extra games of denominator, so their QBs read as more durable purely
    # for having gone further -- a bias pointing straight at the good offences
    # where the good receivers are. Jayden Daniels showed 37 available starts
    # across two week-1-starter seasons when the true figure is 34.
    df = (pl.read_parquet(CACHE)
            .filter(pl.col("season").is_in(HIST) & (pl.col("week") <= 18)))
    qb = df.filter((pl.col("position") == "QB")
                   & (pl.col("attempts") >= MIN_ATT_START))
    starts = (qb.sort(["season", "week", "team", "attempts"], descending=True)
                .group_by(["season", "week", "team"]).first())
    # games each team actually played, to get the denominator right
    team_games = (df.group_by(["season", "team"])
                    .agg(g=pl.col("week").n_unique()))
    # ⚠️ ONLY count seasons where he was the WEEK-1 STARTER.
    # The first cut counted any season with 4+ starts, which conflated ROLE
    # with DURABILITY: a rookie who took over in week 9 read as "missed 8 of
    # 17 starts" and landed in the high-risk tier for having been a backup.
    # Shedeur Sanders, Tyler Shough and Jaxson Dart all came out "high risk"
    # that way -- confidently wrong, and about exactly the young QBs whose
    # receivers this annotation is meant to inform.
    # If he opened the season as the starter, later non-starts are injury or
    # benching, which is what we want to measure. QBs who were never a week-1
    # starter fall through to "unknown", which is the honest answer.
    wk1 = (starts.sort("week").group_by(["season", "team"]).first()
                 .select(["season", "team",
                          pl.col("player_display_name").alias("w1")]))
    per = (starts.group_by(["player_display_name", "season", "team"])
                 .agg(st=pl.len())
                 .join(wk1, on=["season", "team"], how="left")
                 .filter(pl.col("player_display_name") == pl.col("w1"))
                 .join(team_games, on=["season", "team"], how="left"))
    out = {}
    for r in per.group_by("player_display_name").agg(
            st=pl.col("st").sum(), g=pl.col("g").sum(),
            yrs=pl.len()).iter_rows(named=True):
        missed = int(r["g"] - r["st"])
        out[r["player_display_name"]] = {
            "starts": int(r["st"]), "available": int(r["g"]),
            "missed": missed, "seasons": int(r["yrs"]),
            "miss_rate": round(missed / max(r["g"], 1), 3)}
    return out


def risk_tier(d: dict | None) -> tuple:
    """(tier, why). No history -> base rate, never a flattering zero."""
    if not d or d["available"] < 12:
        return "unknown", "no established starting history — league base rate"
    m = d["miss_rate"]
    why = (f"missed {d['missed']} of {d['available']} starts "
           f"{HIST[0]}-{HIST[-1]}")
    if m >= 0.25:
        return "high", why
    if m >= 0.10:
        return "elevated", why
    return "low", why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    board = pd.read_csv(ROOT / "output" / "draft_board_2026.csv")
    dur = qb_durability()

    # ⚠️ The board carries the Rams under BOTH "LA" and "LAR", which produced a
    # 33rd team and split their quarterbacks across two entries. Normalise
    # before anything is keyed on the abbreviation.
    ALIAS = {"LA": "LAR", "SD": "LAC", "OAK": "LV", "STL": "LAR", "WSH": "WAS"}
    board["team_2026"] = board["team_2026"].replace(ALIAS)

    # presumed 2026 starter = the best-ADP QB on the roster. That is the
    # market's own belief, which is the right anchor for a market-priced tool.
    qbs = board[(board["position"] == "QB") & board["team_2026"].notna()]
    qbs = qbs.sort_values("adp", na_position="last")
    starter = {}
    for _, r in qbs.iterrows():
        starter.setdefault(r["team_2026"], r["player"])

    out = {"_meta": {"base_miss3": BASE_MISS3, "cost_wr1": COST_WR1,
                     "cost_other": COST_OTHER, "hist": HIST,
                     "source": "scripts/qb_dependency.py"}}
    for team, qb in starter.items():
        d = dur.get(qb)
        tier, why = risk_tier(d)
        out[team] = {"qb": qb, "tier": tier, "why": why,
                     "miss_rate": None if not d else d["miss_rate"]}

    print(f"presumed 2026 starters: {len(starter)} teams "
          f"({sum(1 for v in out.values() if isinstance(v, dict) and v.get('tier') == 'unknown')} "
          f"with no start history)")
    print(f"\nbase rate: {BASE_MISS3:.0%} of teams lose 3+ starts from their "
          f"week-1 QB; cost {COST_WR1} PPG to a #1 target, {COST_OTHER} to "
          f"other pass catchers\n")
    print(f"{'team':<6}{'presumed 2026 starter':<24}{'tier':<11}why")
    order = {"high": 0, "unknown": 1, "elevated": 2, "low": 3}
    for team, v in sorted(out.items(),
                          key=lambda kv: (order.get(kv[1].get("tier"), 9)
                                          if isinstance(kv[1], dict) else 9,
                                          kv[0])):
        if team == "_meta":
            continue
        print(f"{team:<6}{v['qb']:<24}{v['tier']:<11}{v['why']}")

    if a.write:
        OUT.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    else:
        print("\n(--write to update data/qb_context_2026.json)")


if __name__ == "__main__":
    main()
