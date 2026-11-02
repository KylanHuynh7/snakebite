"""League draft tendencies from the pulled ESPN history (2023-2025).

Trains the *history* half of opponent-aware availability (AGENDA item 9):
what this league's managers actually do, expressed as small, shrunk
adjustments on top of the market prior. The output is consumed by the
replay validator and, if it survives the Brier test, by the live tooling.

The power situation is worse than the original design assumed: the ESPN
league only goes back to 2023 (8 teams, then 10, 10), so there are ~390
picks, ~360 of them human. The design table's verdicts shift accordingly:

    sigma multiplier (league-wide)      ~360 picks   fine
    positional rate by phase            ~120/phase   fine pooled, thin split
    per-manager positional lean         ~40/manager  needs heavy shrinkage
    per-manager x position x round      ~3/cell      noise, not estimated

Everything per-manager is empirical-Bayes shrunk toward the league mean:
    shrunk = (n * observed + K * league) / (n + K)
with K the prior strength in picks. A manager with no history (the two 2026
rookies) has n = 0 and lands exactly on the league mean -- that is the
no-imputation fallback, not synthetic data.

Cross-league normalisation: league sizes differ (8/10/12) and FFC's consensus
is a pooled 12-team market. Positional *phases* use draft fraction
(pick / total slots). Market deviation (z) uses market RANK, not fraction:
a draft consumes players roughly in market-rank order whatever the league
size, so `overall pick - market rank` is comparable across seasons, while
fractions stretch a 180-slot market over a 120-slot draft and manufacture a
fake +2.7-sigma "everyone drafts late" bias (found the hard way on the first
run). FFC's list includes K and DEF, and so do the league drafts, so rank
and pick live in the same space.

Autopicked selections (auto != 0) are excluded from manager behaviour --
they are ESPN's autodraft logic, not the friend's -- but they still consume
roster slots, so the replay validator keeps them as ground truth.

usage: league_tendencies.py            writes data/league_tendencies.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HIST = DATA / "league_history"
OUT = DATA / "league_tendencies.json"

SEASONS = [2023, 2024, 2025]
FFC_SLOTS = 180                      # FFC consensus is a 12-team, 15-round market
POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K"}
SKILL = ["QB", "RB", "WR", "TE"]     # leans are estimated over these only
PHASES = [(0.0, 1 / 3, "early"), (1 / 3, 2 / 3, "mid"), (2 / 3, 1.01, "late")]
K_PRIOR = 20                         # shrinkage strength, in picks
UA = {"User-Agent": "Mozilla/5.0"}

# FFC and ESPN disagree on a handful of spellings beyond suffix noise
ALIASES = {
    "hollywood brown": "marquise brown",
    "dj chark": "d j chark",
}


def norm_name(s: str) -> str:
    import re
    s = re.sub(r"[^a-z\s]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def fetch_adp(year: int) -> pd.DataFrame:
    """FFC final pre-season PPR market for `year` (the last ~days before
    Labor Day weekend -- exactly the market the league drafted against)."""
    cache = HIST / f"adp_ffc_ppr_{year}.json"
    if not cache.exists():
        url = f"https://fantasyfootballcalculator.com/api/v1/adp/ppr?year={year}"
        cache.write_text(requests.get(url, headers=UA, timeout=30).text)
    d = json.loads(cache.read_text())
    t = pd.DataFrame(d["players"])
    t["key"] = t["name"].map(norm_name)
    # same floor as phase3: thinly-drafted players report implausible spread
    t["adp_sigma"] = np.maximum(t["stdev"], 0.5 + 0.04 * t["adp"])
    t = t.drop_duplicates("key", keep="first")
    t["mrank"] = t["adp"].rank(method="first")
    return t.set_index("key")[["adp", "adp_sigma", "mrank", "position"]]


def load_season(year: int) -> dict:
    return json.loads((HIST / f"season_{year}.json").read_text())


def phase_of(frac: float) -> str:
    for lo, hi, name in PHASES:
        if lo <= frac < hi:
            return name
    return "late"


def build_picks() -> pd.DataFrame:
    """One row per historical pick, joined to that year's market."""
    rows = []
    for year in SEASONS:
        s = load_season(year)
        team_owner = {t["id"]: (t["owners"][0] if t["owners"] else None)
                      for t in s["teams"]}
        adp = fetch_adp(year)
        total = s["size"] * 15          # slots incl. the D/ST picks we dropped
        for p in s["picks"]:
            pl = s["players"][str(p["pid"])]
            key = norm_name(pl["n"])
            hit = adp.loc[key] if key in adp.index else None
            rows.append({
                "year": year, "size": s["size"], "total": total,
                "overall": p["o"], "frac": (p["o"] - 0.5) / total,
                "round": p["r"], "auto": p["auto"] != 0,
                "owner": team_owner[p["team"]],
                "player": pl["n"], "key": key, "pos": POS.get(pl["pos"], "?"),
                "adp": None if hit is None else float(hit["adp"]),
                "mrank": None if hit is None else float(hit["mrank"]),
                "adp_sigma": None if hit is None else float(hit["adp_sigma"]),
            })
    df = pd.DataFrame(rows)
    df["phase"] = df["frac"].map(phase_of)
    df["z"] = (df["overall"] - df["mrank"]) / df["adp_sigma"]
    return df


def shrink(observed: float, n: float, prior: float, k: float = K_PRIOR) -> float:
    return (n * observed + k * prior) / (n + k)


SIGMA_BUCKETS = [(1, 12), (13, 24), (25, 48), (49, 84), (85, 132)]
DRIFT_FULL = 84       # drift is measured over ranks 1..DRIFT_FULL
DRIFT_ZERO = 106      # and tapers to zero by here (last supported bucket)


def fit_sigma_curve(human: pd.DataFrame) -> dict:
    """Two separate things, estimated at the resolution each can support.

    SPREAD (`pts`): robust z-spread per market-rank bucket, used as
    interpolation control points. Strongly rank-dependent and worth
    resolving per bucket -- round 1 is chalk (0.71x, TIGHTER than the pooled
    market) while the middle rounds run ~2.1x. A parametric log fit was
    tried first and overshot ranks ~8-30, which broke the board's
    conservation identity, so the buckets themselves are the model.

    DRIFT (`drift`): ONE pooled number, not per bucket. Mean z is the
    league's systematic offset vs the market -- players go about a pick
    later than their market rank. Per-bucket drift estimates (+0.41, +0.44,
    +0.33, +0.16) are individually NOT significant (every 95% CI spans
    zero), so resolving drift by bucket is fitting noise; pooled over ranks
    1-84 it is +0.285 sigma [+0.046, +0.524], which IS significant and is
    also stable across league size (+0.29 at 8 teams, +0.28 at 10, +0.28 at
    10) -- that stability is the licence to apply it to a 12-team 2026
    draft. Beyond rank ~84 the estimate is selection-biased (deep fliers
    only enter the data when someone reached for them), so it tapers to
    zero and stays there."""
    pts, buckets = [], []
    for lo, hi in SIGMA_BUCKETS:
        g = human[(human["mrank"] >= lo) & (human["mrank"] <= hi)]["z"].dropna()
        if len(g) < 8:
            continue
        mad = float(1.4826 * np.median(np.abs(g - np.median(g))))
        se = float(g.std(ddof=1) / np.sqrt(len(g)))
        pts.append([round(float(np.sqrt(lo * hi)), 1), round(mad, 3)])
        buckets.append((lo, hi, mad, float(g.mean()), se, len(g)))

    sh = human[human["mrank"] <= DRIFT_FULL]["z"].dropna()
    drift = float(sh.mean())
    drift_se = float(sh.std(ddof=1) / np.sqrt(len(sh)))
    return {"pts": pts, "floor": 0.6, "cap": 2.3,
            "drift": round(drift, 3), "drift_se": round(drift_se, 3),
            "drift_n": int(len(sh)),
            "drift_full": DRIFT_FULL, "drift_zero": DRIFT_ZERO,
            "buckets": buckets}


def curve_mult(curve: dict, rank) -> np.ndarray:
    """Multiplier at a market rank: piecewise-linear in ln(rank) through the
    bucket control points, constant beyond the ends, clipped."""
    xs = np.log([p[0] for p in curve["pts"]])
    ys = [p[1] for p in curve["pts"]]
    v = np.interp(np.log(np.maximum(rank, 1.0)), xs, ys)
    return np.clip(v, curve["floor"], curve["cap"])


def curve_shift(curve: dict, rank, sigma_raw) -> np.ndarray:
    """Systematic drift at a rank, in PICKS: the pooled drift (sigma units)
    times the player's raw market sigma, held flat to `drift_full` and
    tapered linearly to zero at `drift_zero`."""
    taper = np.clip(np.interp(rank, [curve["drift_full"], curve["drift_zero"]],
                              [1.0, 0.0]), 0.0, 1.0)
    return curve["drift"] * taper * sigma_raw


def main() -> None:
    df = build_picks()
    human = df[~df["auto"]]

    # ---- join diagnostics: every silently dropped name is a lie waiting ----
    missing = df[df["adp"].isna() & df["pos"].isin(SKILL)]
    print(f"picks {len(df)} ({len(human)} human), "
          f"ADP matched {df['adp'].notna().sum()}/{len(df)}")
    if len(missing):
        print("  no FFC ADP (deep picks are expected to miss):")
        for _, r in missing.iterrows():
            print(f"    {r['year']} pick {r['overall']:>3} {r['player']} ({r['pos']})")

    # ---- 1. sigma multiplier: does this league track the market's spread? --
    z = human["z"].dropna()
    # robust scale: one keeper-league lunatic should not set the multiplier
    mad_sigma = float(1.4826 * np.median(np.abs(z - np.median(z))))
    sig_by_year = {
        int(y): round(float(1.4826 * np.median(np.abs(g - np.median(g)))), 3)
        for y, g in human.dropna(subset=["z"]).groupby("year")["z"]}
    print(f"\nsigma multiplier (robust, pooled): {mad_sigma:.3f}  "
          f"by year: {sig_by_year}")
    print(f"  raw std {z.std():.3f}, mean z {z.mean():+.3f} "
          f"(mean != 0 is fine; late fliers skew it)")

    # the multiplier is strongly rank-dependent -- round 1 is chalk (0.7x,
    # TIGHTER than the pooled market) and the middle rounds are ~2.2x. A
    # flat multiplier over-widens the top and broke the board's conservation
    # identity (1.32x at pick ~25). The curve is the measured buckets,
    # interpolated in ln(rank); ranks beyond the last bucket hold its value
    # (the deepest picks are dominated by fliers, mean z -2.6, whose spread
    # estimate is not trustworthy).
    curve = fit_sigma_curve(human)
    print("  rank-dependent SPREAD (interpolated control points, clipped to "
          f"[{curve['floor']}, {curve['cap']}]);")
    print("  per-bucket drift shown only to record that it is NOT resolvable:")
    for lo, hi, mad, mz, se, n in curve.pop("buckets"):
        flag = "sig" if abs(mz) > 1.96 * se else "not sig"
        print(f"    rank {lo:>3}-{hi:<3} n={n:>3}  spread x{mad:.2f}   "
              f"drift {mz:+.2f} [{mz-1.96*se:+.2f},{mz+1.96*se:+.2f}] {flag}")
    d, dse, dn = curve["drift"], curve["drift_se"], curve["drift_n"]
    print(f"  POOLED drift, ranks 1-{curve['drift_full']}: {d:+.3f} sigma "
          f"[{d-1.96*dse:+.3f},{d+1.96*dse:+.3f}]  n={dn}  "
          f"{'SIGNIFICANT' if abs(d) > 1.96 * dse else 'not significant'}"
          f"  -> tapers to 0 by rank {curve['drift_zero']}")
    by_size = human[human["mrank"] <= curve["drift_full"]].groupby(
        ["year", "size"])["z"].mean()
    print("  drift by season/size (stability check, licences the 12-team "
          "extrapolation): " + ", ".join(
              f"{y} {s}tm {v:+.2f}" for (y, s), v in by_size.items()))

    # ---- 2. league positional rate by phase --------------------------------
    lg = {}
    for ph in ["early", "mid", "late"]:
        sub = human[(human["phase"] == ph) & human["pos"].isin(SKILL)]
        lg[ph] = {p: round(float((sub["pos"] == p).mean()), 3) for p in SKILL}
        lg[ph]["n"] = int(len(sub))
    print("\nleague positional share by phase (human picks, skill only):")
    for ph, d in lg.items():
        print(f"  {ph:<6} n={d['n']:>3}  " +
              "  ".join(f"{p} {d[p]:.0%}" for p in SKILL))

    # ---- 3. per-manager: leans, reach bias, QB timing ----------------------
    league_share = {p: float((human[human["pos"].isin(SKILL)]["pos"] == p).mean())
                    for p in SKILL}
    league_qb_frac = float(
        human[human["pos"] == "QB"].groupby(["year", "owner"])["frac"].min().median())

    teams_2026 = json.loads((HIST / "teams_2026.json").read_text())
    label = {}
    for t in teams_2026["teams"]:
        for o in t["owners"]:
            label[o] = t["name"]

    managers = {}
    for owner, g in human[human["pos"].isin(SKILL)].groupby("owner"):
        n = len(g)
        gz = g["z"].dropna()
        first_qb = g[g["pos"] == "QB"].groupby("year")["frac"].min()
        managers[owner] = {
            "team_2026": label.get(owner),   # None = not in the 2026 league
            "seasons": sorted(int(y) for y in g["year"].unique()),
            "n_picks": int(n),
            "pos_share_raw": {p: round(float((g["pos"] == p).mean()), 3)
                              for p in SKILL},
            "pos_share_shrunk": {
                p: round(shrink(float((g["pos"] == p).mean()), n,
                                league_share[p]), 3) for p in SKILL},
            "mean_z_raw": round(float(gz.mean()), 3) if len(gz) else None,
            # reach bias, shrunk toward 0: negative = drafts guys early
            "reach_shrunk": round(shrink(float(gz.mean()), len(gz), 0.0), 3)
                            if len(gz) else 0.0,
            "first_qb_frac": round(float(first_qb.median()), 3)
                             if len(first_qb) else None,
        }

    current = {o for t in teams_2026["teams"] for o in t["owners"]}
    veterans = {o: m for o, m in managers.items() if o in current}
    print(f"\nmanagers with history in the 2026 league: {len(veterans)} "
          f"(of {len(current)} seats filled; rookies fall back to league mean)")
    print(f"{'2026 team':<26}{'n':>4}  {'QB/RB/WR/TE shrunk':<26}"
          f"{'reach':>7}  {'1st QB':>7}")
    for o, m in sorted(veterans.items(), key=lambda kv: kv[1]["team_2026"] or ""):
        sh = m["pos_share_shrunk"]
        print(f"{m['team_2026']:<26}{m['n_picks']:>4}  "
              f"{sh['QB']:.2f}/{sh['RB']:.2f}/{sh['WR']:.2f}/{sh['TE']:.2f}     "
              f"{m['reach_shrunk']:>+7.2f}  "
              f"{m['first_qb_frac'] if m['first_qb_frac'] is not None else '--':>7}")

    out = {
        "generated_from": SEASONS,
        "n_picks": int(len(df)), "n_human": int(len(human)),
        "k_prior": K_PRIOR,
        "sigma_multiplier": round(mad_sigma, 3),
        "sigma_multiplier_by_year": sig_by_year,
        "sigma_curve": curve,          # a + b*ln(rank), clipped [floor, cap]
        "league_pos_share": {p: round(v, 3) for p, v in league_share.items()},
        "league_pos_share_by_phase": lg,
        "league_first_qb_frac": round(league_qb_frac, 3),
        "managers": managers,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
