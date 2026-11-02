"""Does an ESPN-shaped market prior predict THIS league's drafts better than
the FFC one it has been using?

This is the decisive test for the Aug 11 2026 reframe onto ESPN ADP. The board
has always priced availability off FantasyFootballCalculator. The league drafts
on ESPN. Those are different markets -- mid-round RB/WR go ~31 picks LATER on
ESPN and TEs ~12 picks EARLIER (espn_adp.py) -- so the question is whether
reshaping the prior along ESPN's positional timing predicts real picks better.

⚠️ VERDICT (Aug 11, 2026): THE ESPN-SHAPED PRIOR IS SIGNIFICANTLY WORSE.
Pooled Brier 0.1127 vs 0.0998 for the shipped FFC prior, gap +0.0129
[+0.0102, +0.0156] cluster-bootstrapped over 343 turns. It loses in all three
seasons and in every market-rank band inside the top 120.

  strength scan   lam 0.00  0.0998   <- optimum is NO translation
                      0.25  0.1013
                      0.50  0.1027
                      0.75  0.1083
                      1.00  0.1127
  monotone in lambda, so this is not a tuning failure.

  decomposition   FFC mean, widened sigma      +0.0035   (noise cost only)
                  ESPN mean, no widening       +0.0244   <- the damage
                  full swap                    +0.0129
  The harm is the MEAN, not the translation's noise. The map is not too
  fuzzy; it points the wrong way.

WHY, IN FOOTBALL TERMS. ESPN's ADP is pooled over thousands of casual leagues
that autodraft, take K/DST inside the top 120, and take TEs early. The user's
league is 8-12 engaged managers, and they demonstrably draft like the *serious*
consensus FFC measures -- even though they physically click buttons on ESPN.
The TE column is the smoking gun: ESPN says TEs go 12-19 ranks earlier, this
league does not, and forcing that shift nearly doubles TE error (0.1185 ->
0.1985). Drafting ON ESPN is not the same as drafting LIKE ESPN.

Honest limits: FFC's history is the real market those drafts happened against
with measured per-player stdev, while pseudo-ESPN is a reconstruction -- so the
test is not perfectly even. The decomposition above is the defence: a
reconstruction penalty would show up as noise cost, and the noise cost is small
while the mean cost is large. Re-run after the 2026 draft.

WHY THIS IS INDIRECT, AND WHY IT IS STILL A REAL TEST. ESPN publishes no usable
historical ADP (2025 returns the 170.0 sentinel for every player), so an ESPN
prior cannot be built for 2023-25 by observation. Instead the 2026 FFC->ESPN
rank map (espn_translate.py) is applied to each of those years' FFC markets to
get a pseudo-ESPN market, which IS replayable. The map is a fixed, deterministic
transform fitted on a different season from the one being scored, so this does
not leak: it asks whether ESPN's positional-timing SHAPE helps, not whether
ESPN's 2026 numbers predict 2023-25.

⚠️ What it cannot tell us is whether the 2026 map held in 2023-25. If ESPN's
positional flavour moved between seasons, this understates the real gain.

TWO METHODOLOGY POINTS THAT DECIDE WHETHER THE COMPARISON MEANS ANYTHING

1. THE SCORED UNIVERSE IS FIXED ON FFC RANK for both arms. replay_brier picks
   which players to score with `mrank <= pick + CONSIDER`; if each arm used its
   own rank, the two models would be scored on DIFFERENT EVENT SETS and the
   Brier numbers would not be comparable at all. Same players, same labels,
   only the prior changes.
2. SIGMA IS PROPAGATED, NOT COPIED. Moving a spread through a monotone rank map
   means rescaling by the local slope and adding the map's own residual in
   quadrature (see espn_translate.translate). Copying the FFC sigma across
   would quietly understate the width of the ESPN prior.

Everything is leave-one-season-out and significance is cluster-bootstrapped by
draft TURN -- one turn contributes ~50 correlated player rows, so treating rows
as independent would call noise significant (the lesson from replay_brier).

usage: replay_market.py [--boot 2000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espn_translate as XL
from league_tendencies import (SEASONS, SKILL, build_picks, curve_mult,
                               curve_shift, fetch_adp, fit_sigma_curve,
                               load_season, norm_name)
from replay_brier import CONSIDER, avail_at

RNG = np.random.default_rng(7)


def market_frames(model: dict) -> dict:
    """Per season: the FFC market plus its pseudo-ESPN translation."""
    out = {}
    for year in SEASONS:
        adp = fetch_adp(year).reset_index()
        t = XL.translate(adp, model, rank_col="mrank", pos_col="position")
        # propagate the spread through the map (see module note 2)
        t["sigma_pe"] = np.sqrt((t["slope_pe"] * t["adp_sigma"]) ** 2
                                + t["xsigma_pe"] ** 2)
        out[year] = t.set_index("key")
    return out


def build_z(df: pd.DataFrame, mk: dict) -> pd.DataFrame:
    """Add pseudo-ESPN rank/sigma/z to the historical pick table."""
    d = df.copy()
    pe_rank, pe_sig = [], []
    for _, r in d.iterrows():
        m = mk[r["year"]]
        if r["key"] in m.index:
            row = m.loc[r["key"]]
            pe_rank.append(float(row["mrank_pe"]))
            pe_sig.append(float(row["sigma_pe"]))
        else:
            pe_rank.append(np.nan)
            pe_sig.append(np.nan)
    d["mrank_pe"] = pe_rank
    d["sigma_pe"] = pe_sig
    # z in the pseudo-ESPN axis -- the league's drift and spread have to be
    # re-estimated here, because a market that already places players later
    # implies a different residual drift than FFC did.
    d["z_pe"] = (d["overall"] - d["mrank_pe"]) / d["sigma_pe"]
    return d


def loso_curves(df: pd.DataFrame, held_out: int) -> tuple:
    """Fit the spread curve + drift in BOTH rank spaces, excluding `held_out`."""
    tr = df[(df["year"] != held_out) & ~df["auto"]]
    c_ffc = fit_sigma_curve(tr)
    c_ffc.pop("buckets", None)
    # same estimator, pseudo-ESPN axis: swap z and mrank in, fit, swap back
    tr_pe = tr.assign(z=tr["z_pe"], mrank=tr["mrank_pe"])
    c_pe = fit_sigma_curve(tr_pe)
    c_pe.pop("buckets", None)
    return c_ffc, c_pe


def blend(mu_f, sig_f, mu_p, sig_p, slope, xsig, lam: float):
    """Partial translation at strength `lam`: 0 = pure FFC, 1 = pure ESPN.

    WHY A DIAL AND NOT A SWITCH. The last idea this project rejected on a
    single full-strength run (opponent roster-need) was correctly challenged on
    exactly that basis: testing only at lambda=1 with hand-set constants cannot
    separate "the idea is wrong" from "the constants are too aggressive". The
    scan settled it there and it settles it here.

    It also decomposes the two things the translation does at once -- MOVING the
    mean into ESPN's positional timing, and WIDENING sigma by the map's own
    uncertainty. The widening alone would hurt Brier alone, so without this the
    two effects are inseparable.
    """
    mu = mu_f + lam * (mu_p - mu_f)
    scale = (1 - lam) * sig_f + lam * slope * sig_f
    return mu, float(np.sqrt(scale ** 2 + (lam * xsig) ** 2))


def replay(year: int, df: pd.DataFrame, mk: dict) -> dict:
    s = load_season(year)
    adp = fetch_adp(year)
    m_pe = mk[year]
    c_ffc, c_pe = loso_curves(df, year)
    size, rounds = s["size"], 15

    # universe keyed on FFC rank -- FIXED for both arms (module note 1)
    uni = {}
    for key, row in adp.iterrows():
        pos = row["position"]
        pos = {"PK": "PK", "DEF": "DEF"}.get(pos, pos)
        if pos in ("QB", "RB", "WR", "TE", "PK", "DEF"):
            pe = m_pe.loc[key] if key in m_pe.index else None
            uni[key] = (float(row["mrank"]), float(row["adp_sigma"]), pos,
                        float(pe["mrank_pe"]) if pe is not None else np.nan,
                        float(pe["sigma_pe"]) if pe is not None else np.nan)

    taken_at = {}
    for p in s["picks"]:
        key = norm_name(s["players"][str(p["pid"])]["n"])
        if key in uni:
            taken_at[key] = p["o"]

    team_turns = {}
    for p in s["picks"]:
        team_turns.setdefault(p["team"], []).append(p["o"])

    rows = []
    for p in s["picks"]:
        if p["auto"]:
            continue
        k, team = p["o"], p["team"]
        later = [o for o in team_turns[team] if o > k]
        if not later:
            continue
        k2 = later[0]
        gone = {key for key, o in taken_at.items() if o < k}
        for key, (mu, sig, pos, mu_pe, sig_pe) in uni.items():
            if key in gone or mu > k + CONSIDER or pos in ("PK", "DEF"):
                continue
            y = 1.0 if taken_at.get(key, 10 ** 9) >= k2 else 0.0

            # arm 1: shipped B3, FFC rank space
            mu3 = mu + float(curve_shift(c_ffc, mu, sig))
            p_ffc = avail_at(mu3, sig * float(curve_mult(c_ffc, mu)), k2, k)

            # arm 2: same estimator, pseudo-ESPN rank space
            if mu_pe == mu_pe:
                mu3p = mu_pe + float(curve_shift(c_pe, mu_pe, sig_pe))
                p_pe = avail_at(mu3p, sig_pe * float(curve_mult(c_pe, mu_pe)),
                                k2, k)
            else:
                p_pe = p_ffc        # unmapped -> no opinion, inherit FFC

            rows.append({"turn": f"{year}-{k}", "key": key, "pos": pos,
                         "mrank": mu, "y": y, "ffc": p_ffc, "espn": p_pe})
    return {"year": year, "rows": pd.DataFrame(rows)}


def cluster_boot(d: pd.DataFrame, n: int = 2000) -> tuple:
    """Bootstrap the FFC-minus-ESPN Brier gap, resampling whole draft turns."""
    turns = d["turn"].unique()
    idx = {t: g.index.to_numpy() for t, g in d.groupby("turn")}
    diff = ((d["ffc"] - d["y"]) ** 2 - (d["espn"] - d["y"]) ** 2).to_numpy()
    stats = np.empty(n)
    for b in range(n):
        pick = RNG.choice(turns, size=len(turns), replace=True)
        sel = np.concatenate([idx[t] for t in pick])
        stats[b] = diff[sel].mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def scan(df: pd.DataFrame, mk: dict, lams) -> None:
    """Brier as a function of translation strength, LOSO, re-fitting the league
    curve and drift at every lambda so each arm is given its best shot."""
    print(f"\n=== translation-strength scan ===")
    print(f"{'lambda':<9}{'events':>8}{'Brier':>10}{'vs lam=0':>11}   per-season")
    base = None
    for lam in lams:
        rows = []
        for year in SEASONS:
            s = load_season(year)
            adp = fetch_adp(year)
            m_pe = mk[year]
            # re-fit the curve at THIS lambda, in the blended axis
            tr = df[(df["year"] != year) & ~df["auto"]].copy()
            mb, sb = [], []
            for _, r in tr.iterrows():
                if r["mrank_pe"] != r["mrank_pe"]:
                    mb.append(r["mrank"]); sb.append(r["adp_sigma"]); continue
                m_ = mk[r["year"]].loc[r["key"]]
                u, g = blend(r["mrank"], r["adp_sigma"], r["mrank_pe"],
                             r["sigma_pe"], float(m_["slope_pe"]),
                             float(m_["xsigma_pe"]), lam)
                mb.append(u); sb.append(g)
            tr["mrank"] = mb
            tr["z"] = (tr["overall"] - tr["mrank"]) / np.array(sb)
            c = fit_sigma_curve(tr)
            c.pop("buckets", None)

            uni = {}
            for key, row in adp.iterrows():
                if row["position"] not in ("QB", "RB", "WR", "TE"):
                    continue
                if key in m_pe.index:
                    pe = m_pe.loc[key]
                    u, g = blend(float(row["mrank"]), float(row["adp_sigma"]),
                                 float(pe["mrank_pe"]), float(pe["sigma_pe"]),
                                 float(pe["slope_pe"]), float(pe["xsigma_pe"]),
                                 lam)
                else:
                    u, g = float(row["mrank"]), float(row["adp_sigma"])
                uni[key] = (float(row["mrank"]), u, g)

            taken_at = {}
            for p in s["picks"]:
                k_ = norm_name(s["players"][str(p["pid"])]["n"])
                if k_ in uni:
                    taken_at[k_] = p["o"]
            tt = {}
            for p in s["picks"]:
                tt.setdefault(p["team"], []).append(p["o"])

            for p in s["picks"]:
                if p["auto"]:
                    continue
                k = p["o"]
                later = [o for o in tt[p["team"]] if o > k]
                if not later:
                    continue
                k2 = later[0]
                gone = {q for q, o in taken_at.items() if o < k}
                for key, (mrank, u, g) in uni.items():
                    if key in gone or mrank > k + CONSIDER:
                        continue
                    y = 1.0 if taken_at.get(key, 10 ** 9) >= k2 else 0.0
                    mu = u + float(curve_shift(c, u, g))
                    pr = avail_at(mu, g * float(curve_mult(c, u)), k2, k)
                    rows.append({"turn": f"{year}-{k}", "year": year,
                                 "y": y, "p": pr})
        d = pd.DataFrame(rows)
        b = float(((d["p"] - d["y"]) ** 2).mean())
        if base is None:
            base = b
        per = "  ".join(
            f"{y}:{float((( g['p']-g['y'])**2).mean()):.4f}"
            for y, g in d.groupby("year"))
        print(f"{lam:<9.2f}{len(d):>8}{b:>10.4f}{b - base:>+11.4f}   {per}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--scan", action="store_true",
                    help="scan translation strength lambda 0..1")
    a = ap.parse_args()

    model = XL.fit(verbose=False)
    mk = market_frames(model)
    df = build_z(build_picks(), mk)

    matched = df["mrank_pe"].notna().sum()
    print(f"picks {len(df)}, translated into pseudo-ESPN space: {matched}")
    print(f"\n{'year':<6}{'events':>8}{'FFC (B3)':>11}{'ESPN (B3)':>11}"
          f"{'gap':>9}   (negative gap = ESPN better)")

    allr = []
    for year in SEASONS:
        r = replay(year, df, mk)
        d = r["rows"]
        b_f = float(((d["ffc"] - d["y"]) ** 2).mean())
        b_e = float(((d["espn"] - d["y"]) ** 2).mean())
        print(f"{year:<6}{len(d):>8}{b_f:>11.4f}{b_e:>11.4f}{b_e - b_f:>+9.4f}")
        allr.append(d)

    d = pd.concat(allr, ignore_index=True)
    b_f = float(((d["ffc"] - d["y"]) ** 2).mean())
    b_e = float(((d["espn"] - d["y"]) ** 2).mean())
    lo, hi = cluster_boot(d, a.boot)
    sig = "SIGNIFICANT" if (lo > 0) or (hi < 0) else "not significant"
    print(f"\npooled  FFC {b_f:.4f}   ESPN {b_e:.4f}   gap {b_e - b_f:+.4f}")
    print(f"  FFC-minus-ESPN advantage: {b_f - b_e:+.5f} "
          f"[{lo:+.5f}, {hi:+.5f}] over {d['turn'].nunique()} turns -- {sig}")
    print("  (positive = the ESPN-shaped prior is better)")

    # where the difference lives -- a pooled number hides whether the gain is
    # in the rounds that matter or in deep noise
    print(f"\n{'FFC rank band':<16}{'n':>7}{'FFC':>9}{'ESPN':>9}{'gain':>9}")
    for lo_r, hi_r in [(1, 24), (25, 60), (61, 120), (121, 999)]:
        s = d[(d["mrank"] >= lo_r) & (d["mrank"] <= hi_r)]
        if s.empty:
            continue
        f = float(((s["ffc"] - s["y"]) ** 2).mean())
        e = float(((s["espn"] - s["y"]) ** 2).mean())
        print(f"{f'{lo_r}-{hi_r}':<16}{len(s):>7}{f:>9.4f}{e:>9.4f}{f - e:>+9.4f}")

    print(f"\n{'pos':<16}{'n':>7}{'FFC':>9}{'ESPN':>9}{'gain':>9}")
    for pos in SKILL:
        s = d[d["pos"] == pos]
        if s.empty:
            continue
        f = float(((s["ffc"] - s["y"]) ** 2).mean())
        e = float(((s["espn"] - s["y"]) ** 2).mean())
        print(f"{pos:<16}{len(s):>7}{f:>9.4f}{e:>9.4f}{f - e:>+9.4f}")

    if a.scan:
        scan(df, mk, [0.0, 0.25, 0.5, 0.75, 1.0])


if __name__ == "__main__":
    main()
