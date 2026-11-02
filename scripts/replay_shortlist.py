"""Shortlist survival: will AT LEAST ONE of my targets still be there?

A drafter rarely wants one player; they want one of three. The per-player
availability numbers do not answer that, and combining them is not obviously
legitimate: survival events inside one draft are not independent, because a
fixed number of picks happens between now and your next turn.

  independent   Q = 1 - prod(1 - p_i)      treats each target as its own coin
  bounded       Q is capped by what the picks in between can possibly remove:
                if fewer than N picks happen before you, at least one of N
                targets MUST survive (Q = 1), whatever the p_i say

This script replays the league's real drafts and scores both against what
happened, leave-one-season-out like everything else. The shortlist at each human
turn is chosen three ways, never by the model's own probabilities, so the test
cannot flatter it by picking easy cases:

  best   the N best available players by market rank -- what a drafter who has
         not yet picked actually wants, and the hardest case: these are the
         players most likely to be taken before you are back
  mix    N drawn at random from the best 3N available
  near   the N whose market rank is closest to the pick that is coming -- the
         easy case (some of them always survive), kept to show why it is not
         the test

usage:  replay_shortlist.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from league_tendencies import (SEASONS, SKILL, build_picks, curve_mult,  # noqa: E402
                               curve_shift, fetch_adp, fit_sigma_curve,
                               load_season, norm_name)
from replay_brier import avail_at  # noqa: E402

NS = (2, 3, 5)
KINDS = ("best", "mix", "near")


GAMMAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def q_gamma(g_prod, gamma):
    """1 - (prod(1-p))^gamma. gamma > 1 discounts the chance that EVERY target
    is gone, which is what a fixed number of picks between turns enforces and
    independent coin flips do not."""
    return 1.0 - np.power(g_prod, gamma)


def shortlist_q(ps: list, gap: int, bounded: bool) -> float:
    """P(at least one survives). `gap` = picks made before the user's turn."""
    q = 1.0 - float(np.prod([1.0 - p for p in ps]))
    if bounded and gap < len(ps):
        return 1.0                       # cannot remove them all
    return q


def replay(year: int, df) -> dict:
    s = load_season(year)
    adp = fetch_adp(year)
    tr = df[(df["year"] != year) & ~df["auto"]]
    curve = fit_sigma_curve(tr)
    curve.pop("buckets", None)

    uni = {k: (float(r["mrank"]), float(r["adp_sigma"]))
           for k, r in adp.iterrows() if r["position"] in SKILL}
    taken_at = {}
    for p in s["picks"]:
        key = norm_name(s["players"][str(p["pid"])]["n"])
        if key in uni:
            taken_at[key] = p["o"]
    team_turns = {}
    for p in s["picks"]:
        team_turns.setdefault(p["team"], []).append(p["o"])

    rng = np.random.default_rng(5)
    rows = []   # (N, indep, bounded, y, gap, best single, kind, prod(1-p), year)
    for p in s["picks"]:
        if p["auto"]:
            continue
        k = p["o"]
        later = [o for o in team_turns[p["team"]] if o > k]
        if not later:
            continue
        k2 = later[0]
        gap = k2 - k - 1
        gone = {key for key, o in taken_at.items() if o < k}
        pool = [(key, mu, sg) for key, (mu, sg) in uni.items()
                if key not in gone and mu >= k]
        if len(pool) < max(NS):
            continue
        by_rank = sorted(pool, key=lambda t: t[1])
        by_near = sorted(pool, key=lambda t: abs(t[1] - k2))
        for ki, kind, n in ((i, kd, nn) for i, kd in enumerate(KINDS) for nn in NS):
            if kind == "best":
                picks = by_rank[:n]
            elif kind == "near":
                picks = by_near[:n]
            else:
                top = by_rank[:3 * n]
                picks = [top[j] for j in rng.choice(len(top), n, replace=False)]
            ps = []
            for key, mu, sg in picks:
                mu3 = mu + float(curve_shift(curve, mu, sg))
                ps.append(avail_at(mu3, sg * float(curve_mult(curve, mu)), k2, k))
            y = 1.0 if any(taken_at.get(key, 10 ** 9) >= k2 for key, _, _ in picks) else 0.0
            rows.append((n, shortlist_q(ps, gap, False),
                         shortlist_q(ps, gap, True), y, gap, max(ps), ki,
                         float(np.prod([1.0 - p_ for p_ in ps])), year))
    return {"year": year, "rows": rows}


def main() -> None:
    df = build_picks()
    allrows = []
    for year in SEASONS:
        allrows += replay(year, df)["rows"]
    arr = np.array(allrows)

    for ki, kind in enumerate(KINDS):
        sub = arr[arr[:, 6] == ki]
        print(f"\nshortlist = {kind}")
        print(f"{'N':>2} {'turns':>6}  {'indep':>8} {'bounded':>8} {'best single':>12}"
              f"   {'mean pred':>9} {'actual':>7}")
        for n in NS:
            a = sub[sub[:, 0] == n]
            y = a[:, 3]
            b = lambda p: float(np.mean((p - y) ** 2))      # noqa: E731
            print(f"{n:>2} {len(a):>6}  {b(a[:, 1]):>8.4f} {b(a[:, 2]):>8.4f} "
                  f"{b(a[:, 5]):>12.4f}   {a[:, 1].mean():>9.3f} {y.mean():>7.3f}")

    for kind in ("best", "mix"):
        ki = KINDS.index(kind)
        a = arr[(arr[:, 6] == ki) & (arr[:, 0] == 3)]
        print(f"\nreliability, shortlist={kind}, N=3 (predicted bin -> actual survived):")
        for lo in np.arange(0, 1.0, 0.2):
            m = (a[:, 1] >= lo) & (a[:, 1] < lo + 0.2 + (1e-9 if lo > 0.7 else 0))
            if m.sum() >= 10:
                print(f"  {lo:.1f}-{lo + .2:.1f}  n={int(m.sum()):>4}  "
                      f"predicted {a[m, 1].mean():.3f}  actual {a[m, 3].mean():.3f}")
        print(f"by picks between your turns (shortlist={kind}, N=3):")
        for lo, hi in ((0, 6), (7, 12), (13, 18), (19, 30)):
            m = (a[:, 4] >= lo) & (a[:, 4] <= hi)
            if m.sum() >= 10:
                print(f"  gap {lo:>2}-{hi:<2} n={int(m.sum()):>4}  "
                      f"predicted {a[m, 1].mean():.3f}  actual {a[m, 3].mean():.3f}")
    print("\ndependence discount: Q = 1 - prod(1-p)^gamma, gamma picked "
          "leave-one-season-out on the other two seasons")
    print(f"{'shortlist':<9}{'N':>3}" + "".join(f"{'g='+str(g):>9}" for g in GAMMAS)
          + f"{'LOSO':>9}{'gamma/yr':>14}")
    for ki, kind in enumerate(KINDS):
        for n in NS:
            a = arr[(arr[:, 6] == ki) & (arr[:, 0] == n)]
            y, gp, yr = a[:, 3], a[:, 7], a[:, 8]
            br = [float(np.mean((q_gamma(gp, g) - y) ** 2)) for g in GAMMAS]
            loso, picked = [], []
            for hold in SEASONS:
                tr, te = yr != hold, yr == hold
                best = min(GAMMAS, key=lambda g: float(np.mean(
                    (q_gamma(gp[tr], g) - y[tr]) ** 2)))
                picked.append(best)
                loso.append(((q_gamma(gp[te], best) - y[te]) ** 2).sum())
            print(f"{kind:<9}{n:>3}" + "".join(f"{b:>9.4f}" for b in br)
                  + f"{sum(loso) / len(y):>9.4f}" + "      "
                  + "/".join(f"{g:g}" for g in picked))
    print("\nlower Brier is better; 'best single' is the naive answer of "
          "reporting only the most likely target.")


if __name__ == "__main__":
    main()
