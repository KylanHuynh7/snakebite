"""ESPN ADP -- the market the target league actually drafts in.

WHY THIS EXISTS. The board has used FantasyFootballCalculator ADP since Aug
2026. FFC is a good consensus, but it is a consensus of *FFC's* drafters, and
The target league drafts on ESPN. Those are not the same market, and the gap is
biggest in exactly the rounds where "will he come back to me?" is a live
question (see the mapping report below: ~+23 picks at FFC rank 49-84).

WHAT ESPN GIVES, AND WHAT IT DOES NOT

  ownership.averageDraftPosition   real behaviour across all ESPN drafts
  draftRanksByRankType.PPR.rank    ESPN's PUBLISHED rank -- the pre-draft queue
                                   order autodrafters and casual managers
                                   follow. Diverges from realised ADP by up to
                                   ~50 places (Lamar Jackson: rank 88, ADP 39),
                                   and that divergence is signal about who is
                                   actually going early.
  ownership.percentOwned           adoption, useful as a thin-data filter

  NO per-player stdev.  FFC publishes one; ESPN does not. The availability
  model is entirely a function of sigma, so ESPN cannot replace FFC on its own
  -- sigma has to be mapped across from FFC in rank space. This is the reason
  the swap is a project rather than a one-line change.

THREE TRAPS, ALL VERIFIED Aug 11 2026

1. ⚠️ 170.0 IS A "NO DATA" SENTINEL, NOT AN ADP -- and it fails silently.
   ESPN writes exactly 170.0 for any player it has no draft data on. In the
   CURRENT season that is 167 of the top 1000 (Tua, Darren Waller, plus deep
   camp bodies), all at 0.01-2.04% owned; they are dropped here. For PAST
   seasons it is far worse: `seasons/2025` returns 170.0 for **every** player,
   and 2023/2024 return values that have drifted post-hoc (McCaffrey reads 2.07
   for 2023, when he actually went ~7). Nothing ever errors. This is the
   FantasyPros failure mode again -- plausible-looking garbage instead of a
   failure -- so `--season` refuses anything but the current year unless forced.
   ⚠️ THERE IS NO USABLE HISTORICAL ESPN ADP. Any plan that needs to calibrate
   an ESPN-based prior against past drafts has to solve this first.

2. ESPN ADP IS POLLUTED FOR A 12-TEAM ENGAGED LEAGUE. It averages over
   thousands of casual leagues that draft kickers and defences absurdly early:
   8 K/DST land inside ESPN's top 120 (Brandon Aubrey at 84.5) against 4 in
   FFC's. Every skill player is pushed later as a result. the target league will
   not take a kicker at 85, so the raw ESPN number overstates how long skill
   players last. Handle in the mapping, not by trusting the number.

3. RANK SPACE INCLUDES K/DST, in training and serving alike -- the same
   train/serve symmetry rule as phase3's `mrank` and phase2's destination-team
   keying. Filtering to skill positions before ranking shifts everyone past the
   first kicker and asks the calibration a question it was not fitted on.

usage:  espn_adp.py [--refresh] [--season YYYY --force] [--map]
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEASON = 2026
# leaguedefaults/3 is the public PPR default league -- no auth, no cookies, no
# league id. The private league endpoint needs the user's session and returns the
# same ownership block, so there is no reason to use it.
URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
       "{season}/segments/0/leaguedefaults/3?view=kona_player_info")
UA = {"accept": "application/json",
      "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# ESPN's defaultPositionId. 5=K, 16=D/ST are kept deliberately: they occupy
# real draft slots and therefore belong in rank space (trap 3).
POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}


def norm_name(s: str) -> str:
    """Same normalisation as phase3/draft_advise -- must stay identical or the
    ESPN and FFC tables will not join."""
    s = re.sub(r"[^a-z\s]", "", str(s).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch(season: int = SEASON, limit: int = 1000) -> dict:
    """Pull the raw ownership block for the top `limit` players by adoption."""
    filt = {"players": {"limit": limit,
                        "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}}
    r = requests.get(URL.format(season=season),
                     headers={**UA, "X-Fantasy-Filter": json.dumps(filt)},
                     timeout=45)
    r.raise_for_status()
    return r.json()


def parse(raw: dict, season: int) -> dict:
    players, sentinel = [], 0
    for entry in raw.get("players", []):
        p = entry.get("player") or {}
        own = p.get("ownership") or {}
        adp = own.get("averageDraftPosition")
        if not adp or adp <= 0:
            continue
        # Exactly 170.0 means "no draft data" (trap 1). Drop, but COUNT -- a
        # table that is mostly sentinel has to be loud rather than merely
        # short. In 2026 these land at ranks 621+ so they never touch a
        # draftable player, but they would silently corrupt rank space in any
        # season where ESPN has thinner data.
        if abs(adp - 170.0) < 1e-9:
            sentinel += 1
            continue
        ranks = (p.get("draftRanksByRankType") or {}).get("PPR") or {}
        players.append({
            "espn_id": p.get("id"),
            "name": p.get("fullName"),
            "key": norm_name(p.get("fullName", "")),
            "position": POS.get(p.get("defaultPositionId"), "?"),
            "team_id": p.get("proTeamId"),
            "adp": round(float(adp), 2),
            "adp_change": own.get("averageDraftPositionPercentChange"),
            "espn_rank": ranks.get("rank"),
            "auction": own.get("auctionValueAverage"),
            "pct_owned": own.get("percentOwned"),
            "injury_status": p.get("injuryStatus"),
        })
    players.sort(key=lambda x: x["adp"])
    # Rank over the FULL list including K/DST (trap 3).
    for i, p in enumerate(players, 1):
        p["mrank_espn"] = i
    return {
        "meta": {
            "source": "ESPN kona_player_info leaguedefaults/3",
            "season": season,
            "pulled": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_players": len(players),
            "n_sentinel_170": sentinel,
            "has_stdev": False,
        },
        "players": players,
    }


def report(d: dict) -> None:
    m = d["meta"]
    print(f"ESPN ADP: {m['n_players']} players, season {m['season']}, "
          f"pulled {m['pulled']}")
    if m["n_sentinel_170"]:
        share = m["n_sentinel_170"] / (m["n_players"] + m["n_sentinel_170"])
        note = ("  <- normal: undrafted deep players"
                if share < 0.5 else
                "  <- ⚠️ MOSTLY SENTINEL. This table is NOT ADP. Do not use it.")
        print(f"  dropped {m['n_sentinel_170']} players reading exactly 170.0 "
              f"({share:.0%} of the pull){note}")
    from collections import Counter
    for cut in (60, 120, 180):
        c = Counter(p["position"] for p in d["players"] if p["mrank_espn"] <= cut)
        print(f"  top {cut:>3}: " + "  ".join(f"{k}{v}" for k, v in sorted(c.items())))
    kd = [p for p in d["players"] if p["position"] in ("K", "DST")
          and p["mrank_espn"] <= 120]
    if kd:
        print(f"  K/DST inside the top 120 ({len(kd)}) -- these displace skill "
              f"players and are why raw ESPN ADP runs late:")
        for p in kd:
            print(f"    {p['adp']:>6.1f}  {p['position']:<4} {p['name']}")

    # Where ESPN's published queue order disagrees with ESPN's own drafters.
    # Autodrafters follow the rank; humans do not. Large positive = goes much
    # earlier than the queue implies.
    gaps = [(p["espn_rank"] - p["mrank_espn"], p) for p in d["players"]
            if p["espn_rank"] and p["mrank_espn"] <= 150]
    gaps.sort(key=lambda t: -t[0])
    print("\n  published rank vs realised ADP (goes EARLIER than the queue):")
    for g, p in gaps[:10]:
        print(f"    {p['name']:<24} {p['position']:<4} rank {p['espn_rank']:>4}"
              f"  adp {p['adp']:>6.1f}  ({g:+d})")


def mapping_report() -> None:
    """The load-bearing assumption of the ESPN switch: that ESPN and FFC ADP
    are related by a smooth, stable function of market rank. If that holds, the
    2023-25 FFC history can be translated into pseudo-ESPN ADP and the league
    calibration re-fitted in ESPN space. If it does not, the switch cannot be
    validated and ESPN stays a display-only price.
    """
    import statistics as st

    ep = DATA / f"adp_espn_ppr_{SEASON}.json"
    fp = DATA / f"adp_ffc_ppr_{SEASON}.json"
    if not (ep.exists() and fp.exists()):
        print("need both ESPN and FFC caches for the mapping report")
        return
    E = {p["key"]: p for p in json.loads(ep.read_text())["players"]}
    ffc = json.loads(fp.read_text())["players"]
    for i, r in enumerate(sorted(ffc, key=lambda r: r["adp"]), 1):
        r["mrank_ffc"] = i
    both = [(r, E[norm_name(r["name"])]) for r in ffc
            if norm_name(r["name"]) in E]

    print(f"\n=== ESPN <- FFC mapping ({len(both)} matched of {len(ffc)} FFC) ===")
    print(f"{'FFC rank':<12}{'n':>4}{'med ESPN-FFC':>14}{'med FFCsd':>11}"
          f"{'offset/sd':>11}{'IQR of diff':>14}")
    for lo, hi in [(1, 12), (13, 24), (25, 48), (49, 84), (85, 132), (133, 200)]:
        sub = [(e["adp"] - f["adp"], f.get("stdev") or 0)
               for f, e in both if lo <= f["mrank_ffc"] <= hi and (f.get("stdev") or 0) > 0]
        if not sub:
            continue
        diffs = sorted(d for d, _ in sub)
        sd = st.median([s for _, s in sub])
        dm = st.median(diffs)
        q1, q3 = diffs[len(diffs) // 4], diffs[3 * len(diffs) // 4]
        print(f"{f'{lo}-{hi}':<12}{len(sub):>4}{dm:>14.1f}{sd:>11.1f}"
              f"{dm / sd:>11.2f}{f'{q1:+.0f}..{q3:+.0f}':>14}")
    print("\n  offset/sd is the number that matters: the league drift term the "
          "board already applies is +0.285 sigma. Any band above ~1.0 means "
          "swapping the mean to ESPN without re-fitting moves availability by "
          "more than every calibration in the project combined -- and in the "
          "SAME direction as the drift, so the errors compound toward "
          "'you can wait'.")

    # The pooled table above shows a wide IQR, which reads like "the mapping is
    # noise". It is not -- it is CONFOUNDED BY POSITION, and the split below is
    # the useful view. ESPN drafters and FFC drafters disagree about positional
    # timing, not about players, so a rank-only mapping averages two opposite
    # effects together and looks unstable.
    print(f"\n=== the same mapping, split by position ===")
    print(f"{'pos':<6}{'FFC rank':<12}{'n':>4}{'med diff':>10}{'IQR':>14}{'width':>8}")
    for pos in ("RB", "WR", "TE", "QB"):
        for lo, hi in [(1, 48), (49, 120), (121, 200)]:
            sub = sorted(e["adp"] - f["adp"] for f, e in both
                         if f["position"] == pos and lo <= f["mrank_ffc"] <= hi)
            if len(sub) < 5:
                continue
            q1, q3 = sub[len(sub) // 4], sub[3 * len(sub) // 4]
            print(f"{pos:<6}{f'{lo}-{hi}':<12}{len(sub):>4}{st.median(sub):>10.1f}"
                  f"{f'{q1:+.0f}..{q3:+.0f}':>14}{q3 - q1:>8.0f}")
    print("\n  Read this as football, because it is: ESPN leagues take TEs and "
          "QBs earlier and let mid-round RB/WR slide. In the FFC 49-120 band "
          "that is RB +32 and WR +31 picks LATER on ESPN, against TE ~12 picks "
          "EARLIER. Those cancel in the pooled table.")
    print("  ⚠️ Residual IQR width is still 10-20 picks WITHIN position, so the "
          "mapping is a band, not a point. Translating 2023-25 FFC ADP through "
          "it injects that spread into the calibration -- it must be carried "
          "into sigma, not discarded.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--refresh", action="store_true", help="re-pull even if cached")
    ap.add_argument("--force", action="store_true",
                    help="allow a non-current season despite the sentinel trap")
    ap.add_argument("--map", action="store_true",
                    help="print the ESPN<-FFC rank mapping report")
    a = ap.parse_args()

    if a.season != SEASON and not a.force:
        raise SystemExit(
            f"refusing season {a.season}: ESPN's historical ADP is unusable "
            f"(2025 returns 170.0 for every player; 2023/24 have drifted "
            f"post-hoc). Pass --force only to inspect it. See trap 1 in the "
            f"module docstring.")

    cache = DATA / f"adp_espn_ppr_{a.season}.json"
    if a.refresh or not cache.exists():
        d = parse(fetch(a.season), a.season)
        cache.write_text(json.dumps(d, indent=1))
        print(f"wrote {cache.relative_to(ROOT)}")
    else:
        d = json.loads(cache.read_text())
        print(f"cached {cache.relative_to(ROOT)} "
              f"(--refresh to re-pull; delete before the draft)")
    report(d)
    if a.map:
        mapping_report()


if __name__ == "__main__":
    main()
