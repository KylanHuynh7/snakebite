"""Translate FFC market rank into ESPN market rank, per position.

WHY. The league drafts on ESPN, so ESPN's market is the right prior. But ESPN
publishes no per-player stdev and NO USABLE HISTORY (see espn_adp.py trap 1:
2025 returns 170.0 for every player), so an ESPN prior cannot be fitted or
validated directly against the 2023-25 league drafts. The way through is to fit
the FFC->ESPN relationship on 2026, where both markets exist, and push the
2023-25 FFC ADPs through it to get a pseudo-ESPN market that CAN be replayed.

That makes this module the load-bearing assumption of the whole ESPN switch,
so it is deliberately conservative and reports its own error.

WHY ISOTONIC, NOT AN OFFSET CURVE. The first cut modelled `espn_adp - ffc_adp`
as a function of rank and it looked like noise (IQR +2..+33 at FFC rank 49-84).
It was not noise, it was two opposite effects cancelling: ESPN drafters take
TEs and QBs EARLY and let mid-round RB/WR slide (RB +32, WR +31, TE -11 picks
in the FFC 49-120 band). Split by position the structure is clear, but the
offset is still non-monotonic in rank, which an offset curve handles badly.

Both markets are ORDERINGS of the same players, so the honest object is a
monotone map rank->rank. Isotonic regression is exactly that estimator: it
cannot invert two players the market agrees about, it absorbs the
non-monotonicity as flat segments, and its residual is a real spread estimate
rather than a fit artefact.

⚠️ THE RESIDUAL IS NOT OPTIONAL, AND IT IS WILDLY RANK-DEPENDENT. Measured
per position, in rank space:

    FFC rank    RB     WR     TE     QB       verdict
    1-120      3.2    8.1    2.2   14.4       tight -- trust it
    <=180      6.2  119.3   14.4   13.4       WR falls apart
    201-300     --  ~296     --     --        meaningless

The map is excellent exactly where drafts are decided and collapses past FFC
rank ~120. The collapse is not a fitting failure, it is the two markets
genuinely disagreeing about deep players: Ricky Pearsall is FFC 121 and ESPN
770, Rashod Bateman FFC 160 and ESPN 821. FFC's list is 256 deep and made of
players its drafters actually take; ESPN's ADP past ~120 is thin-sample noise.

So `RELIABLE_TO = 120` is a real horizon, not a tuning knob. Inside it the
translation is worth trusting; outside it `translate()` widens sigma by the
band-local residual, which is large enough to make those players read as
"unknown" rather than as confident predictions. QB carries ~14 ranks of
residual even inside the horizon -- consistent with QB showing no clean
ESPN-vs-FFC offset at all -- so QB translations stay deliberately vague.

Ignoring any of this errs toward "you can wait", which is the direction
[[board-optimism-bias]] says this project already fails in.

usage:  espn_translate.py            # fit + diagnostics
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEASON = 2026

# Positions fitted separately. K/DST get their own map because ESPN drafters
# take them absurdly early (8 inside the top 120 vs FFC's 4) -- that is a real
# part of why skill players slide, so it must be represented, not dropped.
FIT_POS = ["RB", "WR", "TE", "QB", "K", "DST"]
MIN_N = 12          # below this a position falls back to the pooled map
FFC_POS = {"PK": "K", "DEF": "DST"}     # FFC's spellings -> ours

# Past here the two markets stop agreeing about who is even draftable, so the
# translation is reported but not trusted. See the docstring -- this is
# measured (residual sd jumps from 8 to 119 ranks for WR), not chosen.
RELIABLE_TO = 120
# Residual is estimated inside these bands rather than pooled, because a single
# number would import the deep tail's noise into round-2 decisions.
RESID_BANDS = [(1, 24), (25, 60), (61, 120), (121, 200), (201, 10 ** 6)]

# ESPN ranks are CENSORED before fitting. A 12-team 15-round draft is 180
# picks, so market ranks of 220 and 770 encode the identical event -- "nobody
# drafts him" -- and the difference between them is thin-sample noise about
# players who are never selected. Isotonic minimises squared error, so leaving
# those raw lets a handful of undrafted deep receivers drag flat segments
# upward and corrupt the map INSIDE the reliable range: uncensored, WR's
# 61-120 residual is 38 ranks; censored it is 16. This is bounding an outcome
# space, not trimming inconvenient data -- no player changes side of the
# "drafted?" line.
CENSOR = 220


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z\s]", "", str(s).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_2026() -> pd.DataFrame:
    """Matched FFC/ESPN table for the fit year, in rank space."""
    ep = DATA / f"adp_espn_ppr_{SEASON}.json"
    fp = DATA / f"adp_ffc_ppr_{SEASON}.json"
    if not ep.exists():
        raise SystemExit("no ESPN cache -- run: espn_adp.py")
    espn = {p["key"]: p for p in json.loads(ep.read_text())["players"]}

    ffc = pd.DataFrame(json.loads(fp.read_text())["players"])
    ffc["key"] = ffc["name"].map(norm_name)
    ffc = ffc.drop_duplicates("key", keep="first")
    ffc["position"] = ffc["position"].replace(FFC_POS)
    # rank over the FULL list including K/DST -- the same rank space phase3 and
    # league_tendencies train in. Filtering first shifts everyone past the
    # first kicker and asks the map a different question than it was fitted on.
    ffc["mrank_ffc"] = ffc["adp"].rank(method="first")

    ffc["mrank_espn"] = ffc["key"].map(
        lambda k: espn[k]["mrank_espn"] if k in espn else np.nan)
    ffc["adp_espn"] = ffc["key"].map(
        lambda k: espn[k]["adp"] if k in espn else np.nan)
    return ffc


def _band(rank: float) -> tuple:
    for lo, hi in RESID_BANDS:
        if lo <= rank <= hi:
            return (lo, hi)
    return RESID_BANDS[-1]


def fit(verbose: bool = True) -> dict:
    """Per-position isotonic maps FFC rank -> ESPN rank, with BAND-LOCAL
    residual spread.

    Nothing is excluded from the fit. The deep-rank blow-ups are real market
    disagreement, not outliers to be trimmed -- an ESPN rank of 800 on an FFC
    rank-160 player is the market saying "nobody on ESPN drafts him", which is
    exactly the kind of thing this map exists to carry. They are handled by
    letting the residual be large there, not by deleting them.
    """
    df = _load_2026()
    m = df.dropna(subset=["mrank_espn"])
    if verbose:
        print(f"=== FFC -> ESPN rank map, fitted on {SEASON} ===")
        print(f"matched {len(m)} of {len(df)} FFC players "
              f"(reliable to FFC rank {RELIABLE_TO})\n")
        print(f"{'pos':<6}{'n':>4}  {'map (ffc->espn)':<34}"
              f"{'residual sd by FFC rank band':<30}")

    models, out = {}, {}
    for pos in ["_pooled"] + FIT_POS:
        sub = m if pos == "_pooled" else m[m["position"] == pos]
        if len(sub) < MIN_N and pos != "_pooled":
            if verbose:
                print(f"{pos:<6}{len(sub):>4}  n<{MIN_N} -> pooled map")
            continue
        y = np.minimum(sub["mrank_espn"].to_numpy(), CENSOR)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(sub["mrank_ffc"].to_numpy(), y)
        resid = y - iso.predict(sub["mrank_ffc"].to_numpy())

        bands = {}
        for lo, hi in RESID_BANDS:
            sel = (sub["mrank_ffc"] >= lo) & (sub["mrank_ffc"] <= hi)
            r = resid[sel.to_numpy()]
            # a band with too little data inherits the widest seen so far --
            # never the narrowest, because an unknown band must not read as
            # confident
            bands[(lo, hi)] = (float(np.std(r, ddof=1)) if len(r) >= 5
                               else max([v for v in bands.values()] or [np.nan]))
        models[pos] = iso
        out[pos] = {"n": int(len(sub)), "bands": bands,
                    "resid_sd": float(np.std(resid, ddof=1))}
        if verbose:
            probe = [r for r in (12, 36, 72, 120) if r <= sub["mrank_ffc"].max()]
            s = "  ".join(f"{r}->{iso.predict([r])[0]:.0f}" for r in probe)
            bs = " ".join(f"{lo}-{hi if hi < 10**6 else '+'}:"
                          f"{bands[(lo,hi)]:.0f}" if bands[(lo, hi)] == bands[(lo, hi)]
                          else f"{lo}:--" for lo, hi in RESID_BANDS)
            print(f"{pos:<6}{len(sub):>4}  {s:<34}{bs:<30}")

    if verbose:
        print(f"\n  Residual sd = how far a player typically lands from where "
              f"his FFC rank predicts. It is the extra spread translate() adds "
              f"to sigma -- not an error to minimise away, but the honest width "
              f"of a market with no observable history.")
        print(f"  Note the shape: single digits inside FFC rank {RELIABLE_TO}, "
              f"then an order of magnitude worse. That cliff is why "
              f"RELIABLE_TO exists and why deep translations must not be "
              f"read as predictions.")
    return {"models": models, "stats": out}


def translate(adp: pd.DataFrame, model: dict, rank_col: str = "mrank",
              pos_col: str = "position") -> pd.DataFrame:
    """Map an FFC market table into pseudo-ESPN rank space.

    Returns the frame with `mrank_pe` (pseudo-ESPN market rank, re-ranked so it
    is a clean ordering), `xsigma_pe` (the translation residual in ranks) and
    `slope_pe` (the map's local derivative).

    ⚠️ BOTH extra columns are needed to move a sigma across, and for different
    reasons. A market spread lives in the units of its own rank axis, so under
    a monotone map it must be RESCALED by the local slope -- where ESPN
    compresses a stretch of the FFC board, an FFC sigma of 8 is not 8 ESPN
    ranks. Separately, the map itself is uncertain, which ADDS spread. The
    correct combination is in quadrature:

        sigma_pe = sqrt((slope * sigma_ffc)^2 + xsigma_pe^2)

    Using either term alone understates the width, and understating the width
    means overstating how long a player lasts.
    """
    t = adp.copy()
    pos = t[pos_col].replace(FFC_POS)
    pred = np.empty(len(t), dtype=float)
    xsig = np.empty(len(t), dtype=float)
    slope = np.empty(len(t), dtype=float)
    H = 5.0                       # half-width of the numeric derivative
    for i, (r, p) in enumerate(zip(t[rank_col].to_numpy(), pos.to_numpy())):
        key = p if p in model["models"] else "_pooled"
        iso = model["models"][key]
        pred[i] = iso.predict([r])[0]
        lo, hi = iso.predict([max(1.0, r - H)])[0], iso.predict([r + H])[0]
        # Isotonic is piecewise constant, so a player sitting inside a flat
        # segment reads slope 0 -- which would collapse his sigma to the
        # translation residual alone and make him look far more predictable
        # than he is. Floor it.
        slope[i] = max(0.25, (hi - lo) / (min(r + H, r + H) - max(1.0, r - H)))
        b = model["stats"][key]["bands"].get(_band(r), np.nan)
        if b != b:      # NaN -> fall back to that position's pooled residual
            b = model["stats"][key]["resid_sd"]
        xsig[i] = b
    t["mrank_pe_raw"] = pred
    t["slope_pe"] = slope
    # isotonic produces ties (flat segments); re-rank to recover a strict
    # ordering, breaking ties on the original FFC rank so the map stays stable
    t["mrank_pe"] = (pd.DataFrame({"a": pred, "b": t[rank_col]})
                     .apply(tuple, axis=1).rank(method="first").to_numpy())
    t["xsigma_pe"] = xsig
    return t


def main() -> None:
    model = fit(verbose=True)

    # What the map does to the current board -- the sanity check that matters,
    # because these are players the user knows the market on.
    df = _load_2026()
    tr = translate(df.rename(columns={"mrank_ffc": "mrank"}), model)
    tr = tr.sort_values("mrank").head(200)
    tr["move"] = tr["mrank_pe"] - tr["mrank"]
    print(f"\n=== biggest rank moves in the top 200 (FFC rank -> pseudo-ESPN) ===")
    print(f"{'player':<24}{'pos':<5}{'FFC':>6}{'pseudo':>8}{'move':>7}"
          f"{'  actual ESPN':>14}")
    for _, r in tr.reindex(tr["move"].abs().sort_values(ascending=False).index).head(16).iterrows():
        act = "—" if pd.isna(r["mrank_espn"]) else f"{r['mrank_espn']:.0f}"
        print(f"{r['name']:<24}{r['position']:<5}{r['mrank']:>6.0f}"
              f"{r['mrank_pe']:>8.0f}{r['move']:>+7.0f}{act:>14}")
    print("\n  'actual ESPN' is the real observed rank -- the map is fitted to "
          "it, so these should track closely. Divergence here is the residual, "
          "and it is why xsigma_pe exists.")


if __name__ == "__main__":
    main()
