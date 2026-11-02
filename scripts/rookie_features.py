"""Can anything beat draft capital at predicting a rookie's fantasy season?

The board's rookie model knows seven things: `log_pick`, `round`, `draft_age`
and four position dummies. That is draft capital and nothing else -- and it is
currently putting **Jeremiyah Love at my_rank 14 against ADP 25.6**, one of the
largest edges on the board, on the thinnest evidence behind any number on it.

Rookies are also the one place where the failure mode that killed the O-line,
expected-points and loss-weighting features **cannot** apply. Those all failed
because the model already knew it -- the information was inside the player's own
NFL production. A rookie has no NFL production. Anything true about him before
his first snap is, by construction, information the model does not have.

So this is worth testing properly. It is still not worth ASSUMING: "athleticism
matters" is exactly the kind of real-football intuition that produced +0.599
correlations and a useless feature last time.

WHAT IS TESTED. NFL combine measurables (`nflreadpy.load_combine()`, free, no
API key, coverage through the 2026 class), joined on draft slot rather than
name. Raw drills plus the two derived metrics that the public work actually
leans on:

  speed_score  (weight x 200) / forty^4 -- a big back running fast is a
               different prospect from a small one running the same time, and
               the raw forty cannot express that
  bmi          mass relative to height, a crude frame proxy

PROTOCOL. Walk-forward over held-out draft classes with three seeds, per the
rule that no feature family is accepted or rejected on one split. Reported
whichever way it comes out.

=============================================================================
VERDICT (Aug 10, 2026): ** REJECTED **. Do not re-propose without new data.
=============================================================================

Walk-forward over six held-out classes: mean MAE 2.882 -> 2.877, a gain of
+0.005 that is indistinguishable from zero, and it **improved only 2 of 6
classes**. Rank correlation got *worse* on average (rho 0.604 -> 0.592, better
in 3 of 6). This is the signature of a family that fits whichever class you
happened to look at -- exactly what `rolling_backtest()` was built to catch.

The RB-specific claim fails on its own terms too. Against rookie PPG:

    speed_score   rho +0.129  (p=0.149, not significant)
    forty         rho -0.032  (p=0.719, nothing)
    weight        rho +0.174  (p=0.052, marginal)
    log_pick      rho -0.515  (p<0.001)   <- the reference

**This is the O-line failure mode again, in a new costume.** The intuition is
sound -- athleticism obviously matters -- but the information is already priced
in by the people who set draft position. Scouts ran the same stopwatch, and
`log_pick` is their conclusion. A feature that duplicates what draft capital
already encodes adds nothing however real the underlying football is.

SEPARATE, PRACTICAL PROBLEM. The 2026 combine file has 319 skill players and
189 fortys, but **zero draft slots** -- nflverse has not back-filled draft
results for that class. Joining it to the 2026 board would require matching on
name and school, the fragile path that has already cost this project a stance
(`"Chris Godwin"` vs `"Chris Godwin Jr."`). Irrelevant given the rejection
above, but worth knowing before anyone tries again.

WHAT IS STILL UNTESTED, and is the only rookie candidate left: **college
production** (receptions, yards, target share, dominator rating) via the CFBD
API, which needs a free key from collegefootballdata.com. That is production
rather than testing, so it is a genuinely different claim from this one -- it
is not obvious that draft position fully encodes it. Nothing here speaks to
whether it works.

Usage:  .venv/bin/python scripts/rookie_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import nflreadpy as nfl
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
POSITIONS = ["QB", "RB", "WR", "TE"]
SEEDS = (7, 13, 42)
TEST_CLASSES = (2020, 2021, 2022, 2023, 2024, 2025)

BASE = ["log_pick", "round", "draft_age", "pos_QB", "pos_RB", "pos_TE", "pos_WR"]
COMBINE = ["forty", "wt", "vertical", "broad_jump", "cone", "shuttle", "bench",
           "speed_score", "bmi"]


def height_in(ht: pd.Series) -> pd.Series:
    """PFR stores height as feet-inches ('6-2'). Anything else becomes NaN."""
    def one(v):
        try:
            f, i = str(v).split("-")
            return int(f) * 12 + int(i)
        except Exception:
            return np.nan
    return ht.map(one)


def build() -> pd.DataFrame:
    dp = nfl.load_draft_picks().filter(
        pl.col("position").is_in(POSITIONS) & (pl.col("season") >= 2010)
    ).select("season", "round", "pick", "gsis_id", "pfr_player_name",
             "position", "age").to_pandas()

    weekly = pl.read_parquet(DATA / "weekly_stats_2016_2025.parquet")
    rook = (weekly.group_by(["player_id", "season"])
            .agg(games=pl.col("fantasy_points_ppr").count(),
                 ppr_pg=pl.col("fantasy_points_ppr").mean())
            .to_pandas().rename(columns={"player_id": "gsis_id"}))
    dp = dp.merge(rook, on=["gsis_id", "season"], how="left")
    dp["games"] = dp["games"].fillna(0)

    # Join the combine on DRAFT SLOT, not on name. Names disagree across
    # sources in exactly the way that silently drops rows ("Chris Godwin" vs
    # "Chris Godwin Jr."); (draft year, overall pick) is unique and exact.
    cb = nfl.load_combine().to_pandas()
    cb = cb[cb["pos"].isin(POSITIONS + ["FB", "HB"])]
    cb["season"] = cb["draft_year"].fillna(cb["season"]).astype("Int64")
    cb = cb.dropna(subset=["draft_ovr"])
    cb["pick"] = cb["draft_ovr"].astype(int)
    cb["ht_in"] = height_in(cb["ht"])
    cb = cb.drop_duplicates(subset=["season", "pick"])
    keep = ["season", "pick", "forty", "wt", "ht_in", "vertical", "broad_jump",
            "cone", "shuttle", "bench"]
    dp = dp.merge(cb[keep], on=["season", "pick"], how="left")

    dp["log_pick"] = np.log(dp["pick"])
    dp["draft_age"] = dp["age"].fillna(dp["age"].median())
    for pos in POSITIONS:
        dp[f"pos_{pos}"] = (dp["position"] == pos).astype(int)
    # Derived. Speed score rewards mass at a given forty; BMI is a frame proxy.
    dp["speed_score"] = (dp["wt"] * 200.0) / (dp["forty"] ** 4)
    dp["bmi"] = 703.0 * dp["wt"] / (dp["ht_in"] ** 2)
    return dp


def coverage(dp: pd.DataFrame) -> None:
    print("=" * 74)
    print("COMBINE COVERAGE -- how often do we even have the measurement?")
    print("=" * 74)
    print("  A feature present for half the rows can still help (LightGBM")
    print("  splits on missingness), but it caps how much it CAN help.\n")
    d = dp[dp["season"] >= 2016]
    print(f"  {'feature':<14}{'coverage':>10}{'  by position (QB/RB/WR/TE)'}")
    for c in ["forty", "wt", "vertical", "broad_jump", "cone", "shuttle",
              "bench"]:
        by = "/".join(f"{d[d['position'] == p][c].notna().mean():.0%}"
                      for p in POSITIONS)
        print(f"  {c:<14}{d[c].notna().mean():>9.0%}   {by}")
    print(f"\n  2026 class specifically: "
          f"{dp[dp['season'] == 2026]['forty'].notna().mean():.0%} have a forty "
          f"(n={len(dp[dp['season'] == 2026])})")


def evaluate(dp: pd.DataFrame) -> None:
    played = dp[(dp["games"] >= 4) & (dp["season"] >= 2016)
                & (dp["season"] <= 2025)].copy()
    print("\n" + "=" * 74)
    print("WALK-FORWARD: does adding the combine beat draft capital alone?")
    print("=" * 74)
    print(f"  {len(played)} rookie seasons, 2016-2025 (>=4 games played).")
    print("  Each class predicted by a model trained ONLY on earlier classes,")
    print("  averaged over 3 seeds.\n")

    def run(feats, seed, yr):
        tr = played[played["season"] < yr]
        te = played[played["season"] == yr]
        if len(te) < 10 or len(tr) < 100:
            return None
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=300,
                              learning_rate=0.03, num_leaves=7,
                              min_child_samples=20, random_state=seed,
                              verbose=-1)
        m.fit(tr[feats], tr["ppr_pg"])
        p = m.predict(te[feats])
        return p, te["ppr_pg"].values

    print(f"  {'class':<8}{'n':>5}{'base MAE':>11}{'+combine':>11}{'change':>10}"
          f"{'base rho':>11}{'+combine':>11}")
    agg = {"base": [], "comb": [], "rb": [], "rc": []}
    for yr in TEST_CLASSES:
        maes, rhos = {}, {}
        for label, feats in (("base", BASE), ("comb", BASE + COMBINE)):
            ms, rs = [], []
            for seed in SEEDS:
                out = run(feats, seed, yr)
                if out is None:
                    break
                p, y = out
                ms.append(np.abs(p - y).mean())
                rs.append(spearmanr(p, y).statistic)
            if not ms:
                break
            maes[label], rhos[label] = np.mean(ms), np.mean(rs)
        if len(maes) < 2:
            continue
        n = len(played[played["season"] == yr])
        agg["base"].append(maes["base"]); agg["comb"].append(maes["comb"])
        agg["rb"].append(rhos["base"]); agg["rc"].append(rhos["comb"])
        print(f"  {yr:<8}{n:>5}{maes['base']:>11.3f}{maes['comb']:>11.3f}"
              f"{maes['base'] - maes['comb']:>+10.3f}"
              f"{rhos['base']:>11.3f}{rhos['comb']:>11.3f}")

    mb, mc = np.mean(agg["base"]), np.mean(agg["comb"])
    rb, rc = np.mean(agg["rb"]), np.mean(agg["rc"])
    print(f"\n  {'MEAN':<8}{'':>5}{mb:>11.3f}{mc:>11.3f}{mb - mc:>+10.3f}"
          f"{rb:>11.3f}{rc:>11.3f}")
    wins = sum(1 for b, c in zip(agg["base"], agg["comb"]) if c < b)
    print(f"\n  combine improved MAE in {wins}/{len(agg['base'])} held-out "
          f"classes (rho in "
          f"{sum(1 for b, c in zip(agg['rb'], agg['rc']) if c > b)}"
          f"/{len(agg['rb'])})")
    print("\n  A family that helps in one class is fitting that class. The")
    print("  decision rule is the mean AND the per-class count, same as")
    print("  rolling_backtest() in phase2_projections.py.")


def rb_only(dp: pd.DataFrame) -> None:
    """Speed score is an RB claim specifically -- test it where it is claimed."""
    print("\n" + "=" * 74)
    print("THE RB-SPECIFIC CLAIM -- does speed score do anything for backs?")
    print("=" * 74)
    d = dp[(dp["position"] == "RB") & (dp["games"] >= 4)
           & (dp["season"].between(2016, 2025)) & dp["speed_score"].notna()]
    print(f"  {len(d)} RBs with a combine forty and a real rookie season.")
    for col in ("speed_score", "forty", "wt", "log_pick"):
        r = spearmanr(d[col], d["ppr_pg"])
        print(f"    rho({col:<12} , rookie PPG) = {r.statistic:>+.3f} "
              f"(p={r.pvalue:.3f})")
    print("\n  log_pick is the reference. A combine number that cannot beat")
    print("  'where he was drafted' is telling you what the scouts already")
    print("  priced in -- the same reason the O-line feature failed.")


def main() -> None:
    dp = build()
    coverage(dp)
    evaluate(dp)
    rb_only(dp)


if __name__ == "__main__":
    main()
