"""Second attempt at putting opponent behaviour into the availability maths.

THE FIRST ATTEMPT FAILED AND THAT RESULT STANDS. `scan_opponent_weight.py`
turned roster-need and per-manager leans into strength dials and scanned a 5x5
grid: the Brier surface rose monotonically in both, the optimum was both off,
and need was ~50x more harmful than lean. The user's question -- "if the manager
before me needs a WR, can I wait on the RB?" -- got the answer "no, he takes
the RB often enough that assuming otherwise makes the forecast worse."

SO WHY RE-TEST. Because that design had three specific weaknesses that are
about the ESTIMAND, not the idea, and this project's own record says the
estimand is the better place to look (validation-protocol: five feature
families rejected, and everything that ever worked came from fixing what was
being asked, not from adding inputs).

  1. DILUTION. Personalisation was applied to every player within 50 ranks of
     the pick -- ~50 rows per turn. If need moves only the 3-4 players actually
     in contention, the effect is averaged against 46 rows where it is noise.
     Fixed here by `CONTEND`: personalise only inside a narrow window of the
     current pick, leave everyone else on the market prior.

  2. WRONG SIGNAL. "Need" was unfilled STARTER SLOTS -- a static property of a
     roster. But what actually moves a draft room is the RUN: three receivers
     in five picks makes the fourth more likely, whoever is picking and
     whatever his roster says. A run is observable live, needs no history, and
     is a property of the room rather than of one manager's intentions.

  3. NO URGENCY TERM. A manager missing a starting TE in round 4 behaves
     nothing like one missing it in round 13. Need was never scaled by how
     many picks remain to fill the hole.

Each is a dial, all default off, and the grid is scanned LOSO exactly as
before. If the surface again points at zero, that is a real answer and the
opponent layer stays displayed rather than modelled -- which is fine, because
the display is what the user actually asked to see.

⚠️ VERDICT (Aug 11, 2026): THE REDESIGN DOES NOT BEAT THE SHIPPED MODEL.
Third independent rejection of opponent behaviour in the availability maths.

                 need 0.00   need 0.50   need 1.00
    run 0.00     0.10017     0.09966     0.10116
    run 0.50     0.10323     0.10123     0.10199
    run 1.00     0.10977     0.10716     0.10667     (shipped B3: 0.09976)

  * THE RUN TERM IS ACTIVELY HARMFUL and monotone in its dial. Fixing the
    dilution and urgency problems did not rescue it.
  * The nominal best cell (run 0, need 0.5) beats B3 by +0.00009 with CI
    [-0.00124, +0.00151]. That is a tie, and it is the MINIMUM OF NINE CELLS
    -- selecting the best of a grid and reporting it as an improvement is how
    fake findings are manufactured. It is not a result.
  * Inside the contention window the best walk is 0.17590 against B3's
    0.17547 -- i.e. no better exactly where the redesign was supposed to help.

WHY THE RUN TERM FAILS, measured directly rather than inferred from Brier:
runs simply do not continue in this league. P(next pick is that position)
against base rate, 391 picks:

    2 of last 3 were RB   n=90   0.322 vs 0.350   lift 0.92
    3 of last 3 were WR   n=24   0.458 vs 0.424   lift 1.08
    4 of last 5 were WR   n=34   0.382 vs 0.424   lift 0.90

Every lift is inside noise of 1.0, and the WR ones lean slightly NEGATIVE.
"Get ahead of the run" has no support here: a WR run is not evidence that
more receivers are going, because **WR is 42% of every skill pick in this
league anyway**. Same class of error as the bust-rate claim -- a scary
pattern that is really just the base rate.

CONCLUSION: opponent behaviour is DISPLAYED, not modelled. Three designs have
now failed (full-strength, strength-scanned, and estimand-redesigned), which is
enough. Re-run after the 2026 draft adds ~165 picks at 12 teams -- and note the
league gains two managers with no history, one of them a complete beginner,
which is new variance rather than new signal.

usage: need_redesign.py [--grid] [--boot 2000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from league_tendencies import (SEASONS, SKILL, build_picks, curve_mult,
                               curve_shift, fetch_adp, load_season, norm_name)
from replay_brier import CONSIDER, STARTERS, avail_at, loso_params

RNG = np.random.default_rng(11)

# Only players within this many ranks of the current pick are treated as
# genuinely in contention. Outside it, no personalisation at all.
CONTEND = 15
# Window of recent picks used to detect a positional run.
RUN_WINDOW = 10


def run_mult(pos: str, recent: list, base: dict, lam: float) -> float:
    """Positional run in the last RUN_WINDOW picks, as a multiplier.

    Ratio of observed share to this league's base rate for that position,
    damped toward 1 and raised to the strength dial. A run is evidence about
    the ROOM, not about any one manager -- which is the point, because the
    room is what we can actually observe live.
    """
    if lam <= 0 or not recent:
        return 1.0
    share = sum(1 for p in recent if p == pos) / len(recent)
    b = max(base.get(pos, 0.25), 0.05)
    # damped: a 2x run reads as ~1.4x appetite, not 2x
    ratio = 1.0 + 0.5 * (share / b - 1.0)
    return float(max(0.4, min(2.0, ratio)) ** lam)


def need_mult2(pos: str, count: int, picks_left: int, lam: float) -> float:
    """Roster need, scaled by URGENCY rather than applied flat.

    A hole is only pressure if there is little draft left to fill it. With
    picks_left large the multiplier goes to 1 (no opinion); as it shrinks the
    unfilled starter slot bites. This is the term the first test found most
    harmful, so it is the one most likely to be switched off again -- but it
    was previously applied at full strength in round 2, which is exactly where
    a manager has every reason to ignore need and take the best player.
    """
    if lam <= 0:
        return 1.0
    want = STARTERS.get(pos, 1)
    if count >= want:
        surplus = count - want
        m = max(0.35, 1.0 - 0.3 * (surplus + 1))
    else:
        urgency = float(np.clip(1.0 - picks_left / 8.0, 0.0, 1.0))
        m = 1.0 + 1.2 * urgency
    return float(m ** lam)


def walk(avail: dict, order: list, rosters: dict, mgr_of: dict, leans: dict,
         target: int, now: int, curve: dict, recent: list, base: dict,
         lam_run: float, lam_need: float, lam_lean: float,
         size: int) -> dict:
    """Pick-by-pick survival walk with the redesigned terms.

    Subtractive survival update, as established: 'taken at pick j' is mutually
    exclusive across players and picks, and the product form leaks ~35% of the
    departure mass.
    """
    surv = {k: 1.0 for k in avail}
    counts = {t: dict(c) for t, c in rosters.items()}
    rec = list(recent)
    for j, team in order:
        if not (now <= j < target):
            continue
        lean = leans.get(mgr_of.get(team), {}).get("pos", {})
        picks_left = max(0, (size * 15 - j) // max(size, 1))
        w = {}
        for key, (mu, sig, pos) in avail.items():
            sm = sig * float(curve_mult(curve, mu))
            mu_s = mu + float(curve_shift(curve, mu, sig))
            base_w = max(0.0, avail_at(mu_s, sm, j, now - 1)
                         - avail_at(mu_s, sm, j + 1, now - 1))
            # personalise ONLY inside the contention window (weakness 1)
            if base_w > 0 and mu <= j + CONTEND:
                base_w *= run_mult(pos, rec, base, lam_run)
                base_w *= need_mult2(pos, counts[team].get(pos, 0),
                                     picks_left, lam_need)
                if lam_lean > 0:
                    base_w *= lean.get(pos, 1.0) ** lam_lean
            w[key] = base_w
        tot = sum(w.values())
        if tot <= 0:
            continue
        exp_pos = {}
        for key, wi in w.items():
            pt = wi / tot
            surv[key] = max(0.0, surv[key] - pt)
            p = avail[key][2]
            exp_pos[p] = exp_pos.get(p, 0.0) + pt
        for p, f in exp_pos.items():
            counts[team][p] = counts[team].get(p, 0) + f
        # the modal position advances the run window
        if exp_pos:
            rec.append(max(exp_pos, key=exp_pos.get))
            rec[:] = rec[-RUN_WINDOW:]
    return surv


def replay(year: int, df: pd.DataFrame, lams: tuple) -> pd.DataFrame:
    lam_run, lam_need, lam_lean = lams
    s = load_season(year)
    adp = fetch_adp(year)
    mult, curve, leans = loso_params(df, year)
    size = s["size"]
    total = size * 15

    slot_team = {}
    for o in range(1, total + 1):
        r, i = (o - 1) // size, (o - 1) % size
        slot_team[o] = s["pickOrder"][i if r % 2 == 0 else size - 1 - i]
    mgr_of = {t["id"]: (t["owners"][0] if t["owners"] else None)
              for t in s["teams"]}

    uni = {}
    for key, row in adp.iterrows():
        if row["position"] in ("QB", "RB", "WR", "TE", "PK", "DEF"):
            uni[key] = (float(row["mrank"]), float(row["adp_sigma"]),
                        row["position"])

    taken_at, pos_of = {}, {}
    POSMAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "PK"}
    for p in s["picks"]:
        key = norm_name(s["players"][str(p["pid"])]["n"])
        pos_of[p["o"]] = POSMAP.get(s["players"][str(p["pid"])]["pos"])
        if key in uni:
            taken_at[key] = p["o"]

    skill = df[(df["year"] == year) & df["pos"].isin(SKILL)]
    base = {p: float((skill["pos"] == p).mean()) for p in SKILL}

    tt = {}
    for p in s["picks"]:
        tt.setdefault(p["team"], []).append(p["o"])

    rows = []
    for p in s["picks"]:
        if p["auto"]:
            continue
        k, team = p["o"], p["team"]
        later = [o for o in tt[team] if o > k]
        if not later:
            continue
        k2 = later[0]
        gone = {q for q, o in taken_at.items() if o < k}
        avail = {q: v for q, v in uni.items()
                 if q not in gone and v[0] <= k + CONSIDER}
        rosters = {t["id"]: {} for t in s["teams"]}
        for q in s["picks"]:
            if q["o"] < k:
                ps = POSMAP.get(s["players"][str(q["pid"])]["pos"])
                if ps:
                    rosters[q["team"]][ps] = rosters[q["team"]].get(ps, 0) + 1
        recent = [pos_of[o] for o in range(max(1, k - RUN_WINDOW), k)
                  if pos_of.get(o)]
        order = [(o, slot_team[o]) for o in range(k + 1, k2)]
        surv = walk(avail, order, rosters, mgr_of, leans, k2, k + 1, curve,
                    recent, base, lam_run, lam_need, lam_lean, size)
        for key, (mu, sig, pos) in avail.items():
            if pos in ("PK", "DEF"):
                continue
            y = 1.0 if taken_at.get(key, 10 ** 9) >= k2 else 0.0
            mu3 = mu + float(curve_shift(curve, mu, sig))
            b3 = avail_at(mu3, sig * float(curve_mult(curve, mu)), k2, k)
            rows.append({"turn": f"{year}-{k}", "year": year, "pos": pos,
                         "mrank": mu, "y": y, "b3": b3, "sim": surv[key],
                         "contend": mu <= k + CONTEND})
    return pd.DataFrame(rows)


def brier(d: pd.DataFrame, col: str) -> float:
    return float(((d[col] - d["y"]) ** 2).mean())


def boot(d: pd.DataFrame, a: str, b: str, n: int = 2000) -> tuple:
    turns = d["turn"].unique()
    idx = {t: g.index.to_numpy() for t, g in d.groupby("turn")}
    diff = ((d[a] - d["y"]) ** 2 - (d[b] - d["y"]) ** 2).to_numpy()
    st = np.empty(n)
    for i in range(n):
        sel = np.concatenate([idx[t] for t in RNG.choice(turns, len(turns), True)])
        st[i] = diff[sel].mean()
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    df = build_picks()

    def run_all(lams):
        return pd.concat([replay(y, df, lams) for y in SEASONS],
                         ignore_index=True)

    # reference: the shipped conditional normal, and the walk with everything
    # switched off (which should reproduce it up to normalisation)
    d0 = run_all((0.0, 0.0, 0.0))
    print(f"events {len(d0)} over {d0['turn'].nunique()} turns "
          f"({d0['contend'].sum()} inside the contention window)")
    print(f"\nB3 (shipped, no walk)        {brier(d0, 'b3'):.5f}")
    print(f"walk, all dials off          {brier(d0, 'sim'):.5f}")

    if not a.grid:
        print("\n(--grid to scan the redesigned terms)")
        return

    print(f"\n=== grid scan: run x need, lean off (contention window "
          f"{CONTEND} ranks) ===")
    print(f"{'':<10}" + "".join(f"need {v:<7.2f}" for v in (0.0, 0.5, 1.0)))
    best = (10.0, None)
    for lr in (0.0, 0.5, 1.0):
        line = f"run {lr:<6.2f}"
        for ln in (0.0, 0.5, 1.0):
            d = run_all((lr, ln, 0.0))
            b = brier(d, "sim")
            line += f"{b:<12.5f}"
            if b < best[0]:
                best = (b, (lr, ln, 0.0), d)
        print(line)

    b3 = brier(d0, "b3")
    print(f"\nshipped B3 = {b3:.5f}")
    bb, lams, dbest = best
    lo, hi = boot(dbest, "b3", "sim", a.boot)
    sig = "SIGNIFICANT" if (lo > 0) or (hi < 0) else "not significant"
    print(f"best cell {lams} -> {bb:.5f}   B3-minus-best "
          f"{b3 - bb:+.5f} [{lo:+.5f}, {hi:+.5f}]  {sig}")
    print("  (positive = the redesigned walk beats the shipped model)")

    # does it help where it is supposed to -- on players actually in contention?
    print(f"\nrestricted to the contention window (the rows the redesign "
          f"actually touches):")
    c0 = d0[d0["contend"]]
    cb = dbest[dbest["contend"]]
    print(f"  B3 {brier(c0, 'b3'):.5f}   best walk {brier(cb, 'sim'):.5f}   "
          f"n={len(cb)}")
    for pos in SKILL:
        s0, sb = c0[c0["pos"] == pos], cb[cb["pos"] == pos]
        if len(sb) < 50:
            continue
        print(f"    {pos:<4} B3 {brier(s0, 'b3'):.5f}  walk "
              f"{brier(sb, 'sim'):.5f}  ({len(sb)})")


if __name__ == "__main__":
    main()
