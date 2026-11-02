"""Sanity checks on output/draft_board_2026.csv. Run after every rebuild.

Nothing here needs a live draft, a browser, or the internet. It exists because
the availability model has no internal consistency constraint -- every player
gets an independent normal, and nothing stops the set of them from implying
that 32 players come off the board in 30 picks. The old
`sigma = 1 + 0.14 * ADP` approximation ran 35-40% hot for months without
anything catching it. These checks would have.

    .venv/bin/python scripts/validate_board.py

Exits nonzero if a check fails, so it can gate a rebuild.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import league  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BOARD = ROOT / "output" / "draft_board_2026.csv"

# How far the implied number of drafted players may drift from the true count
# before it is a problem. 1.15 is deliberately loose: independent marginals
# cannot satisfy the identity exactly, and the point is to catch a broken sigma,
# not to demand coherence the model cannot deliver.
CONSERVATION_TOL = 1.15
CHECKPOINTS = [7, 13, 19, 25, 31, 43, 55, 67, 79, 91, 121, 151]

failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def main() -> int:
    if not BOARD.exists():
        print(f"no board at {BOARD} -- run phase3_draft_board.py first")
        return 2
    b = pd.read_csv(BOARD)
    print(f"board: {len(b)} players, {b['adp'].notna().sum()} with an ADP line\n")

    priced = b.dropna(subset=["adp", "adp_sigma"]).copy()

    # --- 1. conservation ---------------------------------------------------
    # Between pick 1 and pick k, exactly k-1 players are gone. Summing
    # P(drafted by k) over the board should land near that. The ratio's SIGN is
    # the diagnostic: >1 means the model expects more players gone than can be,
    # so it understates availability and the panel will tell you to reach.
    print("conservation -- implied players gone vs actually gone")
    # the model's mean is adp + adp_shift (league chalk drift); scoring the
    # raw adp here would test a curve nothing downstream actually uses
    shift = priced["adp_shift"] if "adp_shift" in priced else 0.0
    ratios = []
    for k in CHECKPOINTS:
        implied = norm.cdf(k, loc=priced["adp"] + shift,
                           scale=priced["adp_sigma"]).sum()
        truth = k - 1
        r = implied / truth
        ratios.append(r)
        print(f"    pick {k:>4}: {truth:>4} gone, model implies {implied:>6.1f}  ({r:.2f}x)")
    worst = max(ratios, key=lambda r: abs(np.log(r)))
    check(
        "conservation",
        abs(np.log(worst)) <= np.log(CONSERVATION_TOL),
        f"mean {np.mean(ratios):.2f}x, worst {worst:.2f}x (tolerance {CONSERVATION_TOL}x)",
    )
    if np.mean(ratios) > 1.0:
        notes.append(
            f"model runs {(np.mean(ratios)-1)*100:.0f}% pessimistic on availability "
            "-- it expects more players gone than can be, which biases the panel "
            "toward taking players early."
        )

    # --- 2. sigma floor ----------------------------------------------------
    # Thinly-drafted players get an implausibly tight stdev from FFC; phase3
    # floors it. A floor violation means the floor stopped being applied, which
    # shows up as a late-round player looking far more predictable than he is.
    # When league calibration is active (league_tendencies.json), the board's
    # sigmas are the FFC floor TIMES the rank-dependent league multiplier --
    # round 1 deliberately sits below the raw floor because this league drafts
    # the top tighter than the pooled market -- so the check must apply the
    # same curve or it just re-detects the calibration. It reads `mrank` off
    # the board rather than re-ranking: re-ranking the 205 priced rows would
    # be a THIRD rank space, agreeing with neither training nor phase3.
    floor = 0.5 + 0.04 * priced["adp"]
    tend = league.tendencies_path()
    floor_label = "0.5 + 0.04*ADP"
    if tend.exists() and "mrank" in priced:
        c = json.loads(tend.read_text())["sigma_curve"]
        mult = np.clip(
            np.interp(np.log(np.maximum(priced["mrank"], 1.0)),
                      np.log([p[0] for p in c["pts"]]),
                      [p[1] for p in c["pts"]]),
            c["floor"], c["cap"])
        floor = floor * mult
        floor_label += " x league spread(rank)"
    bad = priced[priced["adp_sigma"] < floor - 1e-6]
    check("sigma floor", bad.empty, f"{len(bad)} players below {floor_label}")

    # --- 3. availability is monotone --------------------------------------
    # You cannot become MORE likely to be available at a later pick.
    av_cols = [c for c in b.columns if c.startswith("avail_pk")]
    av_cols.sort(key=lambda c: int(c.removeprefix("avail_pk")))
    if len(av_cols) >= 2:
        av = b[av_cols].to_numpy()
        rising = np.nansum(np.diff(av, axis=1) > 1e-9)
        check("availability monotone", rising == 0,
              f"{rising} increases across {' -> '.join(av_cols)}")
        rng = np.nanmin(av), np.nanmax(av)
        check("availability in [0,1]", rng[0] >= -1e-9 and rng[1] <= 1 + 1e-9,
              f"range {rng[0]:.3f} to {rng[1]:.3f}")
    else:
        notes.append("board has no avail_pk columns -- run phase3 with a slot")

    # --- 4. positional coverage -------------------------------------------
    # A silent ADP-join failure shows up as a whole position losing its prices.
    print("\npriced share by position")
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = b[b["position"] == pos]
        if sub.empty:
            continue
        share = sub["adp"].notna().mean()
        print(f"    {pos}: {sub['adp'].notna().sum():>3}/{len(sub):<4} priced ({share:.0%})")

    top60 = b.nsmallest(60, "my_rank")
    check("top-60 priced", top60["adp"].notna().mean() >= 0.90,
          f"{top60['adp'].notna().mean():.0%} of the top 60 have an ADP")

    # --- 5. no duplicate players ------------------------------------------
    dupes = b["player"].duplicated().sum()
    check("no duplicate players", dupes == 0, f"{dupes} duplicated names")

    # --- 6. the TD-dependency flag is position-RELATIVE --------------------
    # The flag this replaced was an absolute `td_pt_share >= 0.40`, which fired
    # on 11 quarterbacks and 1 running back -- not because those QBs were
    # unusual but because a passing TD is worth 4 points and the median QB
    # season clears 0.40. The defect was invisible for months: the column was
    # populated, the flag "worked", it just described the position.
    #
    # A percentile cut cannot drift back into that by accident, but a broken
    # join or a bad groupby could collapse it to one position again, so assert
    # the SHAPE rather than the values. Roughly (1-cut) of each position should
    # fire; no single position should own most of the flags.
    if "td_dep" in b.columns:
        dep = b[b["td_dep"].fillna("") != ""]
        if dep.empty:
            notes.append("no TD-dependency flags on the board at all "
                         "-- rerun phase2_projections.py")
        else:
            share = dep["position"].value_counts(normalize=True)
            top_pos, top_share = share.index[0], share.iloc[0]
            check("TD-dep flag spans positions", top_share <= 0.55,
                  f"{len(dep)} flagged, most-flagged position {top_pos} "
                  f"holds {top_share:.0%} (old absolute rule: QB 85%)")

    # --- 6b. the interval is ordered ---------------------------------------
    # floor/median/ceiling come from three SEPARATE quantile fits, which are
    # not guaranteed to be monotone -- quantile crossing is a known LightGBM
    # behaviour and it got more likely on Aug 10, when the floor alpha moved
    # from 0.20 to 0.10 and pulled the floor further from the median. A
    # crossed interval on the board would read as a player whose floor beats
    # his projection, which is nonsense a human would trust anyway.
    if {"floor_ppg", "proj_ppg", "ceil_ppg"} <= set(b.columns):
        crossed = ((b["floor_ppg"] > b["proj_ppg"] + 1e-9)
                   | (b["proj_ppg"] > b["ceil_ppg"] + 1e-9)).sum()
        check("quantiles ordered (floor <= proj <= ceil)", crossed == 0,
              f"{crossed} players with a crossed interval")

    # --- 7. every hand-maintained status row reached the board ------------
    # This project has now been bitten twice by a hand-keyed table whose names
    # did not match: the Godwin AVOID that matched "Chris Godwin" against
    # "Chris Godwin Jr." and vanished, and the phantom-pick state that survived
    # a schema change. phase2 prints unmatched status rows, but a print scrolls
    # past. Assert it, because the failure is invisible by construction -- the
    # board looks completely normal with the row missing.
    status_csv = ROOT / "data" / "player_status_2026.csv"
    if status_csv.exists() and "status" in b.columns:
        st = pd.read_csv(status_csv, comment="#")
        want = {p.strip() for p in st["player"]}
        got = set(b.loc[b["status"].fillna("") != "", "player"])
        lost = sorted(want - got)
        check("status rows reached the board", not lost,
              f"{len(got)}/{len(want)} applied"
              + (f"; MISSING: {', '.join(lost)}" if lost else ""))
    elif not status_csv.exists():
        notes.append("no player_status_2026.csv -- the model is blind to "
                     "current injuries (blind spot #3)")

    # --- 8. watchlist names all resolve ------------------------------------
    # Same failure as the status file and the stances: a name that matches
    # nothing vanishes without a trace, and this is the list the user actually
    # drafts off, so a silent drop is the worst one yet.
    wl_csv = ROOT / "data" / "watchlist_private.csv"   # yours, gitignored
    if not wl_csv.exists():
        wl_csv = ROOT / "data" / "watchlist_2026.csv"  # neutral demo
    wl = pd.DataFrame()
    if wl_csv.exists():
        wl = pd.read_csv(wl_csv, comment="#")
        if not wl.empty:
            import re as _re
            def _n(x):
                x = _re.sub(r"[^a-z\s]", "", str(x).lower())
                x = _re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", x)
                return _re.sub(r"\s+", " ", x).strip()
            known = {_n(p) for p in b["player"]}
            lost = sorted(p for p in wl["player"] if _n(p) not in known)
            check("watchlist names resolve", not lost,
                  f"{len(wl) - len(lost)}/{len(wl)} matched"
                  + (f"; MISSING: {', '.join(lost)}" if lost else ""))
        else:
            notes.append("watchlist is empty -- add players to "
                         "data/watchlist_2026.csv")

    # --- 9. labels stay short, and the watchlist note does not restate them --
    # The live panel prints, per row: status, watchlist note, team context, QB
    # flag and stance -- five lines. On Aug 13 the user reported the panel saying
    # the same thing twice and needing constant scrolling, because the watchlist
    # note was written as a second copy of the stance. Division of labour now:
    # the STANCE is the label, the watchlist note is a TRIGGER the stance does
    # not already say, and the full reasoning lives in the STANCES comment
    # blocks, which are never rendered. This check keeps it that way.
    LABEL_MAX, NOTE_MAX = 70, 60
    if "note" in b.columns:
        long_lbl = sorted(
            f"{r.player} ({len(r.note)})"
            for r in b.itertuples() if isinstance(r.note, str)
            and len(r.note) > LABEL_MAX)
        check(f"stance labels <= {LABEL_MAX} chars", not long_lbl,
              "; ".join(long_lbl) if long_lbl else "all short")

    if wl_csv.exists() and not wl.empty and "note" in b.columns:
        stance = dict(zip(b["player"], b["note"].fillna("")))
        bad = []
        for r in wl.itertuples():
            note = "" if pd.isna(r.note) else str(r.note)
            if not note:
                continue
            if len(note) > NOTE_MAX:
                bad.append(f"{r.player}: {len(note)} chars")
            # a note that opens with the stance's own label is a duplicate
            lbl = stance.get(r.player, "").split("—")[0].strip().lstrip("! ")
            if lbl and note.lstrip("⚠️ ").startswith(lbl[:12]):
                bad.append(f"{r.player}: restates the stance label")
        check("watchlist notes are triggers, not duplicate labels", not bad,
              "; ".join(bad) if bad else f"{len(wl)} rows clean")

    print()
    for n in notes:
        print(f"  note: {n}")
    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
