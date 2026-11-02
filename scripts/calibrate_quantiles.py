"""Are `floor_ppg` and `ceil_ppg` the percentiles they claim to be?

They are not. The board fits LightGBM quantile models at alpha 0.2 and 0.8 and
labels the output a floor and a ceiling, which invites the reading "one year in
five is worse than this". Measured walk-forward over 2020-2024, **29.0% of
players finish below the floor** while 20.7% finish above the ceiling -- so the
nominal 60% interval covers 50%, and the entire miss is on the downside.

The error is stable to within a point across all five seasons, so it is
systematic rather than noise, and it is worse for RB (30.2%) and WR (30.7%)
than for QB (25.5%) or TE (26.3%).

WHY IT HAPPENS, in football terms. The left tail is career collapse -- an
injury, a benching, a rookie taking the job -- and it is fat, sudden, and
bounded below by zero. A regularised tree model (num_leaves 15,
min_child_samples 30) cannot reach into it: producing a prediction that low
requires a leaf built almost entirely from collapse seasons, and the
regularisation exists precisely to prevent leaves like that. The upside tail is
smoother, which is why the ceiling is close to right.

WHY IT MATTERS. `floor_ppg` is a number the user reads directly off the board when
deciding whether a player is safe. Believing a floor breaks 1 year in 5 when it
really breaks closer to 3 in 10 is a 45% understatement of downside, and it
compounds with the availability model already running ~8% pessimistic in the
same direction.

THE FIX is not a new feature; it is asking for the quantile you actually want.
Fitting at a lower alpha pushes the prediction down until observed coverage
matches the label. This script finds that alpha and -- the part that matters --
checks it generalises to a season it was not chosen on.

Usage:  .venv/bin/python scripts/calibrate_quantiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_projections as P  # noqa: E402
import nflreadpy as nfl  # noqa: E402

TEST_SEASONS = (2020, 2021, 2022, 2023, 2024)
FLOOR_GRID = (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20)
CEIL_GRID = (0.76, 0.78, 0.80, 0.82, 0.84)
POSITIONS = ("QB", "RB", "WR", "TE")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> np.ndarray:
    """Same hyper-parameters as the shipped model -- only alpha varies.

    Anything else would be tuning two things at once and would make the
    coverage number unattributable.
    """
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=400,
        learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8, random_state=7, verbose=-1)
    m.fit(train[P.FEATURES], train["next_ppg"])
    return m.predict(test[P.FEATURES])


def coverage_curve(tr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Observed 'fraction below' for each candidate alpha, by season."""
    rows, per_season = [], {}
    for alpha in FLOOR_GRID + CEIL_GRID:
        cov, by_pos = [], {p: [] for p in POSITIONS}
        for yr in TEST_SEASONS:
            train, test = tr[tr["season"] < yr], tr[tr["season"] == yr]
            if len(test) < 50:
                continue
            pred = fit_predict(train, test, alpha)
            below = test["next_ppg"].values < pred
            cov.append(below.mean())
            for p in POSITIONS:
                m = (test["position"] == p).values
                if m.sum() >= 20:
                    by_pos[p].append(below[m].mean())
        per_season[alpha] = cov
        rows.append({"alpha": alpha, "observed": np.mean(cov),
                     "sd_across_seasons": np.std(cov),
                     **{p: (np.mean(v) if v else np.nan)
                        for p, v in by_pos.items()}})
    return pd.DataFrame(rows), per_season


def pick(curve: pd.DataFrame, grid, target: float) -> float:
    sub = curve[curve["alpha"].isin(grid)]
    return float(sub.iloc[(sub["observed"] - target).abs().argmin()]["alpha"])


def validate_out_of_sample(tr: pd.DataFrame, grid, target: float,
                           label: str) -> None:
    """Choose the alpha on four seasons, then score it on the fifth.

    This is the step that separates a calibration from a curve fit. An alpha
    tuned and reported on the same seasons will always look perfect.
    """
    print(f"\n  leave-one-season-out check -- {label} (target {target:.0%})")
    print(f"    {'held-out':<10}{'alpha chosen':>14}{'coverage on it':>17}")
    errs = []
    for held in TEST_SEASONS:
        others = [s for s in TEST_SEASONS if s != held]
        best, best_gap = None, 9e9
        for alpha in grid:
            cov = []
            for yr in others:
                train, test = tr[tr["season"] < yr], tr[tr["season"] == yr]
                if len(test) < 50:
                    continue
                cov.append((test["next_ppg"].values
                            < fit_predict(train, test, alpha)).mean())
            gap = abs(np.mean(cov) - target)
            if gap < best_gap:
                best, best_gap = alpha, gap
        train, test = tr[tr["season"] < held], tr[tr["season"] == held]
        got = (test["next_ppg"].values < fit_predict(train, test, best)).mean()
        errs.append(abs(got - target))
        print(f"    {held:<10}{best:>14.2f}{got:>16.1%}")
    print(f"    mean |coverage - target| out of sample: {np.mean(errs):.1%}")


def main() -> None:
    weekly = P.load_weekly_full()
    df = P.to_frame(P.season_features(weekly), nfl.load_players())
    tr = P.build_transitions(df)
    print(f"{len(tr)} transitions; testing on {TEST_SEASONS}\n")

    curve, per_season = coverage_curve(tr)
    print("=" * 74)
    print("OBSERVED COVERAGE BY REQUESTED ALPHA  ('fraction finishing below')")
    print("=" * 74)
    print("  alpha is what we ASK LightGBM for; observed is what we GET. A")
    print("  well-calibrated model would put these two columns on top of each")
    print("  other. The gap is the bias, and it is one-sided.\n")
    print(f"  {'alpha':>7}{'observed':>11}{'sd/season':>11}"
          + "".join(f"{p:>8}" for p in POSITIONS))
    for _, r in curve.iterrows():
        print(f"  {r['alpha']:>7.2f}{r['observed']:>10.1%}{r['sd_across_seasons']:>11.3f}"
              + "".join(f"{r[p]:>8.1%}" for p in POSITIONS))

    floor_a = pick(curve, FLOOR_GRID, 0.20)
    ceil_a = pick(curve, CEIL_GRID, 0.80)
    cur_floor = curve.loc[curve["alpha"] == 0.20, "observed"].iloc[0]
    cur_ceil = curve.loc[curve["alpha"] == 0.80, "observed"].iloc[0]

    print("\n" + "=" * 74)
    print("RECOMMENDATION")
    print("=" * 74)
    print(f"  floor: ask for {floor_a:.2f} to get a true 20th percentile "
          f"(current 0.20 delivers {cur_floor:.1%})")
    print(f"  ceil : ask for {ceil_a:.2f} to get a true 80th percentile "
          f"(current 0.80 delivers {cur_ceil:.1%})")

    validate_out_of_sample(tr, FLOOR_GRID, 0.20, "floor")
    validate_out_of_sample(tr, CEIL_GRID, 0.80, "ceiling")

    print("\n" + "=" * 74)
    print("PER-POSITION alphas were considered and are NOT recommended:")
    print("  the spread between positions (QB 25.5% vs WR 30.7% at alpha 0.20)")
    print("  is real, but fitting four separate alphas on five seasons is four")
    print("  parameters on ~350 rows each, and the pooled correction already")
    print("  moves every position most of the way. Revisit if a position is")
    print("  still off by >5 points after the pooled fix.")
    print("=" * 74)


if __name__ == "__main__":
    main()
