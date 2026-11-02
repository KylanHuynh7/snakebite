"""How much opponent information belongs in the availability numbers?

A reviewer's challenge, and it is a fair one: the first test rejected roster-need
and manager-lean weighting, but it only ever ran the terms at FULL strength
with hand-set constants. "The idea is wrong" and "the constants are too
aggressive" are different failures and that test could not tell them apart.

So scan the strength instead of assuming it. Both terms are applied as
`multiplier ** lambda`, so lambda = 0 switches a term off (multiplier -> 1)
and lambda = 1 is the original full-strength version. If the best lambda is
greater than zero, the opponent information helps and belongs in the maths.

Scored exactly like `replay_brier.py`: leave-one-season-out, event = "is this
player still there at this manager's next turn?", Brier lower is better, and
significance is bootstrapped over DRAFT TURNS because rows within a turn are
not independent.

usage: scan_opponent_weight.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from league_tendencies import (SEASONS, build_picks, curve_mult, curve_shift,
                               fetch_adp, load_season, norm_name)
from replay_brier import CONSIDER, avail_at, loso_params, simulate

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def events(year, df, combos):
    """One pass per season; every lambda combo scored on the same events."""
    s = load_season(year)
    adp = fetch_adp(year)
    mult, curve, leans = loso_params(df, year)
    size, total = s["size"], s["size"] * 15
    slot_team = {}
    for o in range(1, total + 1):
        r, i = (o - 1) // size, (o - 1) % size
        slot_team[o] = s["pickOrder"][i if r % 2 == 0 else size - 1 - i]
    mgr_of = {t["id"]: (t["owners"][0] if t["owners"] else None)
              for t in s["teams"]}
    uni = {k: (float(r["mrank"]), float(r["adp_sigma"]), r["position"])
           for k, r in adp.iterrows()
           if r["position"] in ("QB", "RB", "WR", "TE", "PK", "DEF")}
    taken = {}
    for p in s["picks"]:
        k = norm_name(s["players"][str(p["pid"])]["n"])
        if k in uni:
            taken[k] = p["o"]
    turns = {}
    for p in s["picks"]:
        turns.setdefault(p["team"], []).append(p["o"])

    recs = []
    for p in s["picks"]:
        if p["auto"]:
            continue
        k = p["o"]
        later = [o for o in turns[p["team"]] if o > k]
        if not later:
            continue
        k2 = later[0]
        gone = {x for x, o in taken.items() if o < k}
        av = {x: v for x, v in uni.items()
              if x not in gone and v[0] <= k + CONSIDER}
        rosters = {t["id"]: {} for t in s["teams"]}
        for q in s["picks"]:
            if q["o"] < k:
                pos = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "PK"}.get(
                    s["players"][str(q["pid"])]["pos"])
                if pos:
                    rosters[q["team"]][pos] = rosters[q["team"]].get(pos, 0) + 1
        order = [(o, slot_team[o]) for o in range(k + 1, k2)]
        sims = {c: simulate(av, order, mgr_of, leans, rosters, k2, k + 1, mult,
                            lam_lean=c[0], lam_need=c[1]) for c in combos}
        for x, (mu, sig, pos) in av.items():
            if pos in ("PK", "DEF"):
                continue
            y = 1.0 if taken.get(x, 10 ** 9) >= k2 else 0.0
            cm = float(curve_mult(curve, mu))
            b3 = avail_at(mu + float(curve_shift(curve, mu, sig)),
                          sig * cm, k2, k)
            recs.append((year, k, y, b3, {c: sims[c][x] for c in combos}))
    return recs


def main():
    df = build_picks()
    combos = [(l, n) for l in GRID for n in GRID]
    recs = []
    for y in SEASONS:
        print(f"  replaying {y} ...", flush=True)
        recs += events(y, df, combos)
    y = np.array([r[2] for r in recs])
    b3 = np.array([r[3] for r in recs])
    turn = np.array([hash((r[0], r[1])) for r in recs])
    _, ti = np.unique(turn, return_inverse=True)
    groups = [np.where(ti == i)[0] for i in range(ti.max() + 1)]
    print(f"\n{len(y)} events across {len(groups)} draft turns")
    print(f"shipped model (B3, no opponent info): Brier "
          f"{np.mean((b3 - y) ** 2):.5f}\n")

    print("Brier by strength dial (lower is better).")
    print("rows = manager positional LEAN, cols = live ROSTER NEED\n")
    print("        " + "".join(f"need={n:<7}" for n in GRID))
    grid = {}
    for l in GRID:
        cells = []
        for n in GRID:
            v = np.array([r[4][(l, n)] for r in recs])
            grid[(l, n)] = v
            cells.append(f"{np.mean((v - y) ** 2):.5f} ")
        print(f"lean={l:<4}" + "".join(cells))

    best = min(grid, key=lambda c: np.mean((grid[c] - y) ** 2))
    bb = np.mean((grid[best] - y) ** 2)
    print(f"\nbest cell: lean={best[0]}, need={best[1]}  Brier {bb:.5f}")
    print(f"full strength (1,1):              Brier "
          f"{np.mean((grid[(1.0, 1.0)] - y) ** 2):.5f}")
    print(f"both off      (0,0):              Brier "
          f"{np.mean((grid[(0.0, 0.0)] - y) ** 2):.5f}")

    # is the best cell actually better than the shipped model?
    rng = np.random.default_rng(0)
    for label, cand in (("best sim cell vs B3", grid[best]),
                        ("full strength vs B3", grid[(1.0, 1.0)])):
        da = (cand - y) ** 2 - (b3 - y) ** 2
        pt = np.array([da[g].sum() for g in groups])
        cnt = np.array([len(g) for g in groups])
        reps = np.empty(2000)
        for i in range(2000):
            sel = rng.integers(0, len(groups), len(groups))
            reps[i] = pt[sel].sum() / cnt[sel].sum()
        lo, hi = np.percentile(reps, [2.5, 97.5])
        verdict = ("BETTER than B3" if hi < 0 else
                   "WORSE than B3" if lo > 0 else "indistinguishable from B3")
        print(f"  {label}: {pt.sum()/cnt.sum():+.5f} [{lo:+.5f},{hi:+.5f}] "
              f"-> {verdict}")
    print("\nnegative = the simulation beats the shipped model")


if __name__ == "__main__":
    main()
