"""Replay the league's real drafts and score P(available) predictions.

The only labelled test set for "will he make it back to me?" that will ever
exist is what actually happened in this league's drafts. Each historical
season is replayed pick by pick; at every human turn we ask, for every
plausibly-relevant player still on the board: will he still be there at this
manager's NEXT turn? Brier score, five models:

  A   baseline         the pre-Aug-9 availAt(): independent normals, FFC sigma
  B   flat mult        sigma x the pooled league multiplier (~2.07)
  B2  rank curve       sigma x spread(rank), interpolated bucket points
  B3  curve + drift    B2 plus the pooled drift (+0.285 sigma) on the mean
  Cm/C simulation      pick-by-pick walk over the intervening picks,
                       market-only / + roster-need and positional leans

Leave-one-season-out everywhere, so numbers are honest out-of-sample.

VERDICT (Aug 9, 2026 -- 361 human picks, 2023-25): pooled Brier
  A 0.1200   B 0.1003   B2 0.1008   B3 0.0999   Cm 0.1011   C 0.1033
B3 won every fold and also fixed the board's conservation identity (flat B
broke it at 1.32x by over-widening round 1), so B3 is what phase3 ships.
The simulation never beat the plain conditional normal at next-turn
horizons, and personalisation (leans + need) made it slightly WORSE -- three
seasons is not enough to out-predict the market prior, so manager
tendencies are displayed in draft_advise, not folded into probabilities.
LIVE DRIFT (Nov 1, 2026): model L re-estimates the chalk drift from the draft in
progress, shrunk toward the league's fitted drift with K picks of prior. It
LOSES to the static B3 in every season at every K tried (pooled 0.1001-0.1007
vs 0.0998) and improves monotonically as K grows, i.e. as it is allowed to
move less. Not shipped. Likely reason: the room's drift barely varies between
drafts (+0.29 / +0.28 / +0.28 by season in league_tendencies.py), so a few
dozen noisy picks add more variance than information.

Re-run this after the 2026 draft lands (~165 more picks) before promoting
anything new into the availability math.

All modelling happens in market-RANK space (see league_tendencies.py for why
fractions are a trap across league sizes).

usage: replay_brier.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from league_tendencies import (SEASONS, SKILL, K_PRIOR, build_picks,
                               curve_mult, curve_shift, fetch_adp,
                               fit_sigma_curve, load_season, norm_name, shrink)

CONSIDER = 50          # score players within this many ranks of the current pick
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
POS_ALL = ["QB", "RB", "WR", "TE", "PK", "DEF"]
# model L: B3, but the chalk drift is re-estimated from THIS draft's picks so
# far, shrunk toward the league's fitted drift with K picks of prior strength
LIVE_K = (5, 10, 20, 40)


# ---------- model A/B: the panel's availAt(), in rank space ----------------
def avail_at(mu: float, sig: float, at: float, now: float) -> float:
    """P(still there at `at` | on the board at `now`). Mirrors app.js."""
    if now > mu + sig:                       # fallen: prior is dead, re-price
        mu2, sig2 = now + (now - mu), max(3.0, sig * 1.5)
        return float(np.clip(1 - norm.cdf(at, mu2, sig2), 0, 1))
    s_now = 1 - norm.cdf(now, mu, sig)
    if s_now <= 0.02:
        return float(np.clip(1 - norm.cdf(at, now, max(3.0, sig * 1.5)), 0, 1))
    s_at = 1 - norm.cdf(at, mu, sig)
    return float(min(1.0, s_at / s_now))


# ---------- model C: one pick at a time -----------------------------------
def hazard(mu: float, sig: float, j: float) -> float:
    """P(this player is taken around pick j | still available), unnormalised."""
    if j > mu + sig:                         # fallen -> re-priced curve
        mu, sig = j + (j - mu), max(3.0, sig * 1.5)
    s = 1 - norm.cdf(j, mu, sig)
    return float(norm.pdf(j, mu, sig) / max(s, 0.02))


def need_mult(pos: str, count: int, round_: int, rounds: int = 15) -> float:
    """Appetite for another player at `pos` given `count` already rostered.
    Hand-shaped, judged only by whether it helps the Brier score."""
    if pos in ("PK", "DEF"):
        if round_ >= rounds - 2:
            return 3.0 if count == 0 else 0.05
        return 0.02
    extra = count - STARTERS[pos]
    if pos in ("QB", "TE"):
        return 1.0 if count == 0 else (0.25 if count == 1 else 0.05)
    return max(0.15, 1.0 - 0.35 * max(0, extra + 1) if extra >= 0 else 1.0)


def simulate(avail: dict, order: list, mgr_of: dict, leans: dict,
             rosters: dict, target: int, now: int, sig_mult: float,
             personalize: bool = True,
             lam_lean: float = 1.0, lam_need: float = 1.0) -> dict:
    """Walk picks `now`..`target-1`; return P(survive) per available player.

    avail: pid -> (mu, sigma, pos). order: [(overall, teamId), ...] for the
    intervening picks. Each step distributes exactly one departure across the
    board (normalised hazard), which is what restores conservation.

    The survival update is SUBTRACTIVE, not multiplicative: "taken at pick j"
    is mutually exclusive across players within a pick and across picks for a
    player, so P(survives) = 1 - sum_j P(taken at j). A product update leaks
    ~35% of the departure mass (found by the conservation check)."""
    surv = {pid: 1.0 for pid in avail}
    # expected roster growth as the walk progresses, per team
    counts = {t: dict(c) for t, c in rosters.items()}
    for j, team in order:
        if not (now <= j < target):
            continue
        m = mgr_of.get(team)
        lean = leans.get(m, {})
        w = {}
        for pid, (mu, sig, pos) in avail.items():
            # base weight = model B's own departure mass across pick j, so
            # with no personalisation the walk reproduces B's calibrated
            # marginals (up to normalisation) instead of re-deriving them
            # from a hazard that scored worse
            sm = sig * sig_mult
            base = max(0.0, avail_at(mu, sm, j, now - 1)
                       - avail_at(mu, sm, j + 1, now - 1))
            if personalize:
                # lam_* are STRENGTH dials in [0,1], applied as an exponent:
                # 0 switches the term off entirely (multiplier -> 1), 1 is
                # full strength. The original test only ever ran at 1.0 with
                # hand-set constants, which conflates "the idea is wrong"
                # with "the constants are too aggressive". Scanning lambda
                # separates them.
                pos_lean = lean.get("pos", {}).get(pos, 1.0)
                nd = need_mult(pos, counts[team].get(pos, 0),
                               (j - 1) // len(counts) + 1)
                base *= (pos_lean ** lam_lean) * (nd ** lam_need)
            w[pid] = base
        tot = sum(w.values())
        if tot <= 0:
            continue
        exp_pos = {}
        for pid, wi in w.items():
            p_take = wi / tot
            surv[pid] = max(0.0, surv[pid] - p_take)
            pos = avail[pid][2]
            exp_pos[pos] = exp_pos.get(pos, 0.0) + p_take
        # grow the picking team's expected roster fractionally
        for pos, f in exp_pos.items():
            counts[team][pos] = counts[team].get(pos, 0) + f
    return surv


# ---------- LOSO training helpers ------------------------------------------
def loso_params(df, held_out: int):
    tr = df[(df["year"] != held_out) & ~df["auto"]]
    z = tr["z"].dropna()
    mult = float(1.4826 * np.median(np.abs(z - np.median(z))))
    curve = fit_sigma_curve(tr)
    curve.pop("buckets", None)
    skill = tr[tr["pos"].isin(SKILL)]
    lg_share = {p: float((skill["pos"] == p).mean()) for p in SKILL}
    leans = {}
    for owner, g in skill.groupby("owner"):
        n = len(g)
        leans[owner] = {"pos": {
            p: shrink(float((g["pos"] == p).mean()), n, lg_share[p]) / lg_share[p]
            for p in SKILL}}
    return mult, curve, leans


# ---------- the replay ------------------------------------------------------
def replay(year: int, df) -> dict:
    s = load_season(year)
    adp = fetch_adp(year)
    mult, curve, leans = loso_params(df, year)
    size, rounds = s["size"], 15
    total = size * rounds

    # full slot -> team map from the snake + pickOrder (covers D/ST gaps too)
    slot_team = {}
    for o in range(1, total + 1):
        r = (o - 1) // size
        i = (o - 1) % size
        slot_team[o + 0] = s["pickOrder"][i if r % 2 == 0 else size - 1 - i]
    mgr_of = {t["id"]: (t["owners"][0] if t["owners"] else None)
              for t in s["teams"]}

    # universe: that year's market, keyed by rank; map the actual picks onto it
    uni = {}
    for key, row in adp.iterrows():
        pos = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
               "PK": "PK", "DEF": "DEF"}.get(row["position"])
        if pos:
            uni[key] = (float(row["mrank"]), float(row["adp_sigma"]), pos)

    taken_at = {}                      # key -> overall pick it actually left
    for p in s["picks"]:
        key = norm_name(s["players"][str(p["pid"])]["n"])
        if key in uni:
            taken_at[key] = p["o"]

    turns = sorted(p["o"] for p in s["picks"] if not p["auto"])
    team_turns = {}
    for p in s["picks"]:
        team_turns.setdefault(p["team"], []).append(p["o"])

    briers = {"A": [], "B": [], "B2": [], "B3": [], "Cm": [], "C": []}
    briers.update({f"L{k}": [] for k in LIVE_K})
    # this draft's own deviation from the market, pick by pick (ranks inside the
    # drift window only -- beyond it the estimate is selection-biased)
    z_seen = []                        # (overall, z), in pick order
    for q in sorted(s["picks"], key=lambda x: x["o"]):
        kq = norm_name(s["players"][str(q["pid"])]["n"])
        if kq in uni and uni[kq][0] <= curve["drift_full"]:
            z_seen.append((q["o"], (q["o"] - uni[kq][0]) / uni[kq][1]))
    events = 0
    for p in s["picks"]:
        if p["auto"]:
            continue
        k, team = p["o"], p["team"]
        later = [o for o in team_turns[team] if o > k]
        if not later:
            continue
        k2 = later[0]
        gone = {key for key, o in taken_at.items() if o < k}
        avail = {key: v for key, v in uni.items()
                 if key not in gone and v[0] <= k + CONSIDER}
        # rosters as they stand before pick k (skill + PK/DEF actually seen)
        rosters = {t["id"]: {} for t in s["teams"]}
        for q in s["picks"]:
            if q["o"] < k:
                pos = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "PK"}.get(
                    s["players"][str(q["pid"])]["pos"])
                if pos:
                    rosters[q["team"]][pos] = rosters[q["team"]].get(pos, 0) + 1
        order = [(o, slot_team[o]) for o in range(k + 1, k2)]
        surv = simulate(avail, order, mgr_of, leans, rosters, k2, k + 1, mult)
        surv_m = simulate(avail, order, mgr_of, leans, rosters, k2, k + 1,
                          mult, personalize=False)
        for key, (mu, sig, pos) in avail.items():
            if pos in ("PK", "DEF"):
                continue               # scored on skill players only
            y = 1.0 if taken_at.get(key, 10 ** 9) >= k2 else 0.0
            pa = avail_at(mu, sig, k2, k)
            pb = avail_at(mu, sig * mult, k2, k)
            pb2 = avail_at(mu, sig * float(curve_mult(curve, mu)), k2, k)
            mu3 = mu + float(curve_shift(curve, mu, sig))
            pb3 = avail_at(mu3, sig * float(curve_mult(curve, mu)), k2, k)
            # live drift: only picks already made (o < k) are visible
            zs = [z for o, z in z_seen if o < k]
            for kk in LIVE_K:
                d_live = (kk * curve["drift"] + sum(zs)) / (kk + len(zs))
                taper = float(np.clip(np.interp(
                    mu, [curve["drift_full"], curve["drift_zero"]], [1.0, 0.0]),
                    0.0, 1.0))
                pl = avail_at(mu + d_live * taper * sig,
                              sig * float(curve_mult(curve, mu)), k2, k)
                briers[f"L{kk}"].append((pl - y) ** 2)
            briers["A"].append((pa - y) ** 2)
            briers["B"].append((pb - y) ** 2)
            briers["B2"].append((pb2 - y) ** 2)
            briers["B3"].append((pb3 - y) ** 2)
            briers["Cm"].append((surv_m[key] - y) ** 2)
            briers["C"].append((surv[key] - y) ** 2)
            events += 1

    # conservation: sum of P(gone by k) vs k-1, from the draft's start
    cons = {}
    start_avail = {k: v for k, v in uni.items() if v[0] <= 1 + CONSIDER}
    order_all = [(o, slot_team[o]) for o in range(1, total + 1)]
    surv0 = simulate(start_avail, order_all[:40], mgr_of, leans,
                     {t["id"]: {} for t in s["teams"]}, 41, 1, mult)
    kchk = 41
    exp_gone_c = sum(1 - v for v in surv0.values())
    exp_gone_a = sum(1 - avail_at(mu, sig, kchk, 1)
                     for mu, sig, pos in start_avail.values())
    cons = {"pick": kchk, "true_gone_universe":
            sum(1 for key, o in taken_at.items()
                if o < kchk and key in start_avail),
            "indep": round(exp_gone_a, 1), "sim": round(exp_gone_c, 1)}

    return {"year": year, "events": events,
            "sigma_mult_loso": round(mult, 3),
            "brier": {m: round(float(np.mean(v)), 4) for m, v in briers.items()},
            "conservation": cons}


def main() -> None:
    df = build_picks()
    print(f"{'year':<6}{'events':>7}  {'mult':>5}  "
          f"{'A base':>8}{'B mult':>8}{'B2 crv':>8}{'B3 c+s':>8}{'Cm mkt':>8}{'C sim':>8}   conservation@41")
    allb = {"A": [], "B": [], "B2": [], "B3": [], "Cm": [], "C": []}
    allb.update({f"L{k}": [] for k in LIVE_K})
    for year in SEASONS:
        r = replay(year, df)
        c = r["conservation"]
        print(f"{r['year']:<6}{r['events']:>7}  {r['sigma_mult_loso']:>5}  "
              f"{r['brier']['A']:>8}{r['brier']['B']:>8}{r['brier']['B2']:>8}{r['brier']['B3']:>8}{r['brier']['Cm']:>8}{r['brier']['C']:>8}"
              f"   true {c['true_gone_universe']}, indep {c['indep']}, "
              f"sim {c['sim']}")
        for m in allb:
            allb[m].append((r["brier"][m], r["events"]))
    pooled = {m: sum(b * n for b, n in v) / sum(n for _, n in v)
              for m, v in allb.items()}
    print(f"\npooled  A {pooled['A']:.4f}   B {pooled['B']:.4f}   "
          f"B2 {pooled['B2']:.4f}   B3 {pooled['B3']:.4f}   Cm {pooled['Cm']:.4f}   C {pooled['C']:.4f}")
    print("lower is better; B<A means the sigma multiplier is real signal, "
          "C<B means the simulation earns its complexity")
    print("\nlive drift (B3 with the drift re-estimated from this draft's own picks;"
          " K = prior strength in picks)")
    print(f"{'year':<6}" + "".join(f"{'L'+str(k):>9}" for k in LIVE_K)
          + f"{'B3':>9}")
    for i, year in enumerate(SEASONS):
        print(f"{year:<6}" + "".join(f"{allb['L'+str(k)][i][0]:>9}" for k in LIVE_K)
              + f"{allb['B3'][i][0]:>9}")
    print(f"{'pooled':<6}" + "".join(f"{pooled['L'+str(k)]:>9.4f}" for k in LIVE_K)
          + f"{pooled['B3']:>9.4f}")


if __name__ == "__main__":
    main()
