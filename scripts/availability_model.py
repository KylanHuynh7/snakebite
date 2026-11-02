"""Should `proj_games` be a learned model instead of two hand-set constants?

`proj_games = 0.6 * (3-year games rate) + 0.4 * 15` is the only number on the
board that is still a hand-tuned formula while everything beside it is a
gradient booster with forty features. It also matters more than it used to: it
drives `proj_total`, `vor_season` and `rank_dur`, and it is what the Aug 10
status column overrides.

THE REAL PROBLEM TURNED OUT NOT TO BE THE FORMULA. It is what the formula was
tuned ON. `build_transitions()` keeps only pairs with `next_games >= 4`, which
is right for a POINTS-PER-GAME target -- two games cannot give you a meaningful
PPG -- and exactly backwards for an AVAILABILITY target, because it discards
the outcomes you are trying to predict. Across 2016-2024, 22.9% of rostered
player-seasons fall below that line, including 13.1% who play zero games.

Tuned on the censored sample the heuristic scores MAE 2.993, which is the
number in the docs. Scored against what actually happened it is **biased +1.45
games high for draftable starters** (MAE 3.022) and +3.30 across all rostered
players. The bias points the same way as the quantile miscalibration fixed the
same day: the board is systematically too optimistic about downside.

So this script tests three arms, in increasing order of machinery, because the
cheapest fix that works is the one to ship:

  1. the current heuristic, constants as shipped
  2. the SAME formula with its two constants re-tuned on uncensored data
  3. LightGBM on age, position, games history and workload

If arm 2 closes the gap, arm 3 is unnecessary complexity. Walk-forward with
three seeds, per the standing rule.

POPULATION. Trained on players who were rostered the following season -- a
retirement is not a projection error, and the board only ever shows
`proj_games` for players on a 2026 roster. Evaluated on the DRAFTABLE cohort
(>=8 games and >=12 PPG in the base year), because that is who the number is
actually read for.

Usage:  .venv/bin/python scripts/availability_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import nflreadpy as nfl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_projections as P  # noqa: E402

SEEDS = (7, 13, 42)
TEST_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)

FEATURES = [
    "games", "lag1_games", "lag2_games", "games_3yr", "n_seasons3",
    "age_next", "years_exp", "ppr_pg", "mean3_ppg",
    "carries_pg", "targets_pg", "attempts_pg", "rec_pg",
    "pos_QB", "pos_RB", "pos_TE", "pos_WR",
]


def dataset() -> pd.DataFrame:
    """Player-seasons paired with next-season GAMES, uncensored."""
    df = P.to_frame(P.season_features(P.load_weekly_full()), nfl.load_players())
    nxt = df[["player_id", "season", "games"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"games": "next_games"})
    d = df[df["games"] >= P.MIN_GAMES].merge(nxt, on=["player_id", "season"],
                                             how="left")
    d = d[d["season"].between(2016, 2024)].copy()
    # A missing next season is zero games PLAYED -- but only counts as an
    # availability failure if he was still on a roster. Retirement is not a
    # projection error and the board never shows proj_games for a retiree.
    d["next_games"] = d["next_games"].fillna(0)
    r = (nfl.load_rosters(list(range(2017, 2026)))
         .select(["season", "gsis_id"]).unique().to_pandas()
         .rename(columns={"gsis_id": "player_id"}))
    r["season"] -= 1
    r["rostered"] = 1
    d = d.merge(r, on=["player_id", "season"], how="left")
    d = d[d["rostered"].fillna(0) == 1].copy()
    d["g3"] = d["games_3yr"] / d["n_seasons3"]
    return d


def heuristic(g3: pd.Series, w: float, anchor: float) -> pd.Series:
    return (w * g3 + (1 - w) * anchor).clip(1, 17)


def tune(train: pd.DataFrame) -> tuple[float, float]:
    """Re-fit the two constants on uncensored training data, minimising MAE."""
    best, best_mae = (0.6, 15.0), 9e9
    for w in np.arange(0.30, 1.01, 0.05):
        for anchor in np.arange(6.0, 15.5, 0.5):
            mae = (heuristic(train["g3"], w, anchor)
                   - train["next_games"]).abs().mean()
            if mae < best_mae:
                best, best_mae = (float(w), float(anchor)), mae
    return best


def main() -> None:
    d = dataset()
    draftable = (d["games"] >= 8) & (d["ppr_pg"] >= 12)
    print(f"{len(d)} rostered player-seasons 2016-2024; "
          f"{draftable.sum()} draftable-cohort rows\n")

    print("=" * 78)
    print("WALK-FORWARD: three arms, evaluated on the draftable cohort")
    print("=" * 78)
    print("  Each season predicted from earlier seasons only. 'shipped' is the")
    print("  current 0.6/15 formula; 'retuned' is the same formula with its two")
    print("  constants re-fitted on UNCENSORED training data; 'lgbm' is the")
    print("  learned model.\n")
    print(f"  {'test yr':<9}{'n':>5}{'shipped':>10}{'retuned':>10}{'lgbm(l1)':>10}"
          f"{'lgbm(l2)':>10}{'  (w, anchor)':<16}")

    ARMS = ("shipped", "retuned", "lgbm", "lgbm_l2")
    agg = {k: [] for k in ARMS}
    rmse = {k: [] for k in ARMS}
    bias = {k: [] for k in ARMS}
    for yr in TEST_SEASONS:
        tr, te = d[d["season"] < yr], d[d["season"] == yr]
        te = te[(te["games"] >= 8) & (te["ppr_pg"] >= 12)]
        if len(tr) < 300 or len(te) < 30:
            continue
        y = te["next_games"].values

        p_ship = heuristic(te["g3"], 0.6, 15.0).values
        # Tune on the same cohort we score on. Fitting the constants to the
        # whole rostered population and then grading them on draftable
        # starters would be testing a mismatch, not the formula.
        w, anchor = tune(tr[(tr["games"] >= 8) & (tr["ppr_pg"] >= 12)])
        p_tune = heuristic(te["g3"], w, anchor).values

        def lgbm(objective):
            preds = []
            for seed in SEEDS:
                m = lgb.LGBMRegressor(
                    objective=objective, n_estimators=300, learning_rate=0.03,
                    num_leaves=15, min_child_samples=30, subsample=0.8,
                    colsample_bytree=0.8, random_state=seed, verbose=-1)
                m.fit(tr[FEATURES], tr["next_games"])
                preds.append(m.predict(te[FEATURES]))
            return np.clip(np.mean(preds, axis=0), 1, 17)

        p_lgbm = lgbm("l1")     # minimises absolute error -> targets the MEDIAN
        p_mean = lgbm("l2")     # minimises squared error  -> targets the MEAN

        for k, p in (("shipped", p_ship), ("retuned", p_tune),
                     ("lgbm", p_lgbm), ("lgbm_l2", p_mean)):
            agg[k].append(np.abs(p - y).mean())
            rmse[k].append(float(np.sqrt(np.mean((p - y) ** 2))))
            bias[k].append(np.mean(p - y))
        print(f"  {yr:<9}{len(te):>5}" + "".join(
            f"{agg[k][-1]:>10.3f}" for k in ARMS) + f"   ({w:.2f}, {anchor:.1f})")

    print(f"\n  {'MEAN MAE':<9}{'':>5}" + "".join(
        f"{np.mean(agg[k]):>10.3f}" for k in ARMS))
    print(f"  {'RMSE':<9}{'':>5}" + "".join(
        f"{np.mean(rmse[k]):>10.3f}" for k in ARMS))
    print(f"  {'MEAN BIAS':<9}{'':>5}" + "".join(
        f"{np.mean(bias[k]):>+10.3f}" for k in ARMS))
    n = len(agg["shipped"])
    print(f"\n  beat 'shipped' on MAE: "
          f"retuned {sum(1 for a, b in zip(agg['retuned'], agg['shipped']) if a < b)}/{n}, "
          f"lgbm {sum(1 for a, b in zip(agg['lgbm'], agg['shipped']) if a < b)}/{n}")
    print(f"  lgbm(l2) bias closer to zero than shipped: "
          f"{sum(1 for a, b in zip(bias['lgbm_l2'], bias['shipped']) if abs(a) < abs(b))}/{n}")
    print("\n  Bias is the column to read second. MAE can improve while the")
    print("  number stays systematically optimistic, and it is the optimism")
    print("  that misleads a drafter -- it makes every injury-risk player look")
    print("  cheaper than he is.")


if __name__ == "__main__":
    main()
