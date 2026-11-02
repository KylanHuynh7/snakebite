"""Compact draft briefing from live state -- the deterministic half of the
latency fix.

The reason an agent feels slow on the clock is that it is being asked to do
reasoning at decision time. But nearly all of the reasoning here was done in
August: the stances, the comp cohorts, the tiers, the blind-spot flags. What
draft night needs is a lookup, not a think.

So this does the lookup, in milliseconds, with no model involved:

  * who is actually left, ranked
  * conditional availability to your next pick, given who is gone
  * how many players remain in each positional tier -- the cliff, which is the
    thing that actually forces a pick
  * roster needs against 1QB/2RB/2WR/1TE/2FLEX
  * the positional run in the last 12 picks

Print it between picks and you rarely need to ask anything. When you DO have a
genuinely novel question, an agent reads this output -- a few hundred tokens --
instead of a browser snapshot, so it costs one inference rather than a
snapshot/parse loop.

usage:  draft_advise.py [--state PATH] [--slot N] [--pos RB]
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# 15, not 16 -- same off-by-one as the extension (bug #3) and phase3. A 16th
# round in a 15-round league puts a pick past the end of the draft, which
# corrupts every horizon calculation that reads from it.
import league                                   # noqa: E402

_L = league.settings()
TEAMS, ROUNDS = _L["teams"], _L["rounds"]
LINEUP = _L["lineup"]
FLEX = _L["flex"]
TIER_BREAK = 0.8
NEGATIVE = ("AVOID", "FADE", "RISK", "IR STASH")


def norm(s: str) -> str:
    import re
    s = re.sub(r"[^a-z\s]", "", str(s).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def my_picks(slot: int) -> list[int]:
    return [(r - 1) * TEAMS + slot if r % 2 else r * TEAMS - slot + 1
            for r in range(1, ROUNDS + 1)]


def seat_of_pick(pick: int, teams: int = TEAMS) -> int:
    """Which draft slot owns an overall pick number, in a snake."""
    r, i = (pick - 1) // teams, (pick - 1) % teams
    return i + 1 if r % 2 == 0 else teams - i


def cdf(x: float, m: float, sd: float) -> float:
    return 0.5 * (1 + math.erf((x - m) / (sd * math.sqrt(2))))


def avail(row, at: int, now: int):
    """P(there at `at` | there at `now`). Returns None once the market has
    already contradicted the prior -- see the same guard in the browser tool.
    The mean is league-calibrated: adp + adp_shift (chalk drift), with
    adp_sigma already carrying the rank-dependent multiplier."""
    if pd.isna(row["adp"]) or pd.isna(row["adp_sigma"]):
        return None
    mu = row["adp"] + (0 if pd.isna(row.get("adp_shift")) else row["adp_shift"])
    if now > mu + row["adp_sigma"]:
        return None
    s_now = 1 - cdf(now, mu, row["adp_sigma"])
    s_at = 1 - cdf(at, mu, row["adp_sigma"])
    if s_now <= 0.02:
        return None
    return min(1.0, s_at / s_now)


# A "can I wait" call needs thresholds, and vague ones are useless on the
# clock. These are deliberately asymmetric: losing a player you specifically
# wanted costs more than reaching a round early for him, and the board's
# availability runs ~8% PESSIMISTIC (validate_board.py), so a WAIT call is
# already the conservative side of the model's own error.
WAIT_SAFE = 0.75      # very likely still there next time round
WAIT_RISKY = 0.45     # a coin flip, and the coin is not in your favour


def print_watchlist(board, live, gone, nxt, later, now) -> None:
    """The user's own players, and the only question that matters: can I wait?

    This is the inverse of reading a ranking. The board's ORDER lost to ADP in
    the end-to-end backtest, but its conditional availability is the one output
    validated against real league drafts -- so the tool is pointed at "when
    does he go", not "who is better".
    """
    path = OUT.parent / "data" / "watchlist_private.csv"   # yours, gitignored
    if not path.exists():
        path = OUT.parent / "data" / "watchlist_2026.csv"  # neutral demo
    if not path.exists():
        return
    wl = pd.read_csv(path, comment="#")
    if wl.empty:
        print("\n=== WATCHLIST === (empty -- add players to "
              "data/watchlist_2026.csv)")
        return
    wl["player"] = wl["player"].str.strip()
    wl["key"] = wl["player"].map(norm)

    known = set(board["key"])
    missing = wl[~wl["key"].isin(known)]
    for _, r in missing.iterrows():
        near = [p for p in board["player"]
                if r["player"].split()[-1].rstrip(".") in p]
        hint = f"  did you mean: {', '.join(near[:3])}" if near else ""
        print(f"  !! watchlist entry {r['player']!r} matches no player{hint}")
    wl = wl[wl["key"].isin(known)]

    print(f"\n=== WATCHLIST — can I wait? "
          f"(next pick {nxt}{', then ' + str(later[0]) if later else ''}) ===")
    rows = []
    for _, w in wl.sort_values(["priority", "player"]).iterrows():
        if w["key"] in gone:
            rows.append((w, None, None, "GONE"))
            continue
        r = live[live["key"] == w["key"]]
        if r.empty:
            continue
        r = r.iloc[0]
        p_next = avail(r, nxt, now) if nxt else None
        p_later = avail(r, later[0], now) if later else None
        # Three different questions, and conflating them is what makes a
        # draft tool useless on the clock:
        #   1. will he even reach my next pick?      -> if not, he is not a
        #      decision, he is someone else's player
        #   2. if he reaches it, must I take him there?
        #   3. or can I spend that pick elsewhere and still get him after?
        if p_next is None:
            verdict = "FELL"          # market already contradicted his ADP
        elif p_next < 0.15:
            verdict = f"GONE B4 {nxt}"
        elif p_later is None:
            verdict = f"TAKE AT {nxt}"
        elif p_later >= WAIT_SAFE:
            verdict = "CAN WAIT"
        elif p_later >= WAIT_RISKY:
            verdict = "RISKY"
        else:
            verdict = f"TAKE AT {nxt}"
        rows.append((w, p_next, p_later, verdict))

    print(f"  {'':<2}{'player':<22}{'pos':<5}{'ADP':>6}{'rank':>6}"
          f"{'@' + str(nxt or '-'):>7}{'@' + str(later[0]) if later else '':>7}"
          f"   verdict")
    for w, pn, pl, v in rows:
        if v == "GONE":
            print(f"  {'x':<2}{w['player']:<22}{'':<5}{'':>6}{'':>6}{'':>7}{'':>7}"
                  f"   — drafted")
            continue
        r = live[live["key"] == w["key"]].iloc[0]
        mark = ("!" if v.startswith("TAKE") else
                "~" if v == "RISKY" else
                "?" if v == "FELL" else
                "x" if v.startswith("GONE") else " ")
        f_n = "—" if pn is None else f"{pn:.0%}"
        f_l = "—" if pl is None else f"{pl:.0%}"
        adp = "—" if pd.isna(r["adp"]) else f"{r['adp']:.1f}"
        print(f"  {mark:<2}{w['player']:<22}{r['position']:<5}{adp:>6}"
              f"{int(r['my_rank']):>6}{f_n:>7}{f_l:>7}   {v}")
        if isinstance(w.get("note"), str) and w["note"].strip():
            print(f"      {w['note']}")
    print(f"  GONE B4 = under 15% to reach your next pick, so not your "
          f"decision · TAKE AT n = he reaches you but will not last · "
          f"RISKY >={WAIT_RISKY:.0%} · CAN WAIT >={WAIT_SAFE:.0%}")
    print("  'FELL' = he is already past his ADP, so the prior is dead "
          "information — judge him on football, not on the number.")


def load_opponents() -> dict:
    """Profiles from data/opponents_2026.csv, keyed by team name."""
    path = league.opponents_csv()
    if not path.exists():
        return {}
    t = pd.read_csv(path, comment="#")
    out = {}
    for _, r in t.iterrows():
        # keyed by the display label; the panel keys by espn_team_id instead
        # because ESPN names change on rename (an unowned seat still reads
        # "Team 11"), but here the label IS what gets printed
        out[str(r["team"]).strip()] = {
            "slot": None if pd.isna(r.get("slot")) else int(r["slot"]),
            "experience": str(r.get("experience") or "vet").strip(),
            "note": "" if pd.isna(r.get("note")) else str(r["note"]).strip(),
        }
    return out


def espn_queue(live) -> list:
    """Top available players in ESPN's PUBLISHED rank order -- the literal
    list a manager sees in the ESPN draft room.

    ⚠️ ESPN's market was tested as a league-wide prior and REJECTED: it is
    significantly worse than FFC at predicting this league's picks
    (replay_market.py, Brier 0.1127 vs 0.0998), because drafting on ESPN is not
    drafting like ESPN. This is the one exception, and the reasoning is
    behavioural rather than statistical: a first-timer has no rankings of his
    own, so he takes what is at the top of his screen. ESPN's published rank
    is that screen -- note it is the QUEUE order, not ESPN's ADP, and the two
    disagree wildly (Lamar Jackson: rank 88, ADP 39).

    Untestable at n=0 history, so it is labelled an assumption everywhere it
    appears and never enters a probability.
    """
    path = OUT.parent / "data" / "adp_espn_ppr_2026.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())["players"]
    rank = {r["key"]: r["espn_rank"] for r in rows if r.get("espn_rank")}
    pool = [(rank[k], p, pos) for k, p, pos in
            zip(live["key"], live["player"], live["position"]) if k in rank]
    return sorted(pool)[:4]


def print_opponents(board, picks, nxt, now, slot, live=None) -> None:
    """Who picks between now and your next turn, and what is actually known
    about each of them.

    ⚠️ EVERYTHING HERE IS DISPLAY. Opponent behaviour has failed three separate
    attempts to earn a place in the availability numbers (need_redesign.py):
    full-strength, strength-scanned, and estimand-redesigned. So this states
    FACTS WITH PROVENANCE and never a synthesised probability. The distinction
    matters on the clock: "took his first QB in R3, both seasons (24 picks)" is
    checkable and lets you weigh it yourself; "70% to take a QB" would be a
    number invented from ~5 picks per cell.
    """
    tend_path = league.tendencies_path()
    teams_path = league.teams_path()
    if not (tend_path.exists() and teams_path.exists() and nxt):
        return
    td = json.loads(tend_path.read_text())
    teams = json.loads(teams_path.read_text())
    prof = load_opponents()

    by_slot = {v["slot"]: k for k, v in prof.items() if v["slot"]}
    mgr_by_team = {}
    for t in teams["teams"]:
        owner = t["owners"][0] if t["owners"] else None
        mgr_by_team[t["name"]] = td["managers"].get(owner) if owner else None

    between = list(range(now, nxt))
    if not between:
        return
    print(f"\n=== WHO PICKS BEFORE YOU (picks {now}-{nxt - 1}, "
          f"{len(between)} of them) ===")
    if not by_slot:
        print("  draft order not set yet -- fill the `slot` column in "
              "data/opponents_2026.csv and this becomes seat-by-seat.")
        return

    # live rosters from the pick stream (teamId rides on every ESPN pick)
    rosters = {}
    for p in picks:
        # the panel's payload carries `at` (overall pick), not a slot
        seat = p.get("slot") or (seat_of_pick(p["at"]) if p.get("at") else None)
        tname = p.get("team_name") or by_slot.get(seat)
        if tname:
            pos = p.get("pos")
            if not pos:
                m = board[board["key"] == norm(p["name"])]
                pos = m.iloc[0]["position"] if len(m) else None
            if pos:
                rosters.setdefault(tname, {})
                rosters[tname][pos] = rosters[tname].get(pos, 0) + 1

    lg = td["league_pos_share"]
    seen = []
    for pk in between:
        seat = seat_of_pick(pk)
        team = by_slot.get(seat)
        if team is None:
            print(f"  {pk:>3}. seat {seat} — unknown (set slots in "
                  f"opponents_2026.csv)")
            continue
        if seat == slot:
            continue
        c = rosters.get(team, {})
        owed = [f"{p}" for p, n in LINEUP.items() if c.get(p, 0) < n]
        flex_have = sum(max(0, c.get(k, 0) - LINEUP[k]) for k in ("RB", "WR", "TE"))
        if flex_have < FLEX:
            owed.append(f"FLEX x{FLEX - flex_have}")
        have = "  ".join(f"{p}{c[p]}" for p in ("QB", "RB", "WR", "TE")
                         if c.get(p)) or "—"

        m = mgr_by_team.get(team)
        pr = prof.get(team, {})
        exp = pr.get("experience", "vet")
        facts = []
        if m and m.get("n_picks"):
            n, yrs = m["n_picks"], len(m.get("seasons", []))
            for pos in ("QB", "RB", "WR", "TE"):
                ratio = m["pos_share_shrunk"][pos] / lg[pos]
                if ratio >= 1.15:
                    facts.append(f"{pos}-heavy")
                elif ratio <= 0.85:
                    facts.append(f"{pos}-light")
            if m.get("first_qb_frac") is not None:
                facts.append(f"1st QB ~R{max(1, round(m['first_qb_frac'] * 15))}")
            r = m.get("reach_shrunk", 0)
            if r <= -0.3:
                facts.append("reaches")
            elif r >= 0.3:
                facts.append("lets value fall")
            prov = f"[{n} picks/{yrs}yr]"
        elif exp == "beginner":
            facts = ["FIRST-TIMER — no rankings of his own, so expect him to "
                     "take what is top of his ESPN screen"]
            prov = "[no history — assumption, not measured]"
        else:
            facts = ["no history — assume league-typical"]
            prov = "[no history]"

        print(f"  {pk:>3}. {team:<26} has: {have:<22} owes: "
              f"{', '.join(owed) if owed else 'starters full'}")
        print(f"       {', '.join(facts) or 'league-typical'} {prov}")
        if exp == "beginner" and live is not None:
            q = espn_queue(live)
            if q:
                print("       ESPN's board would show him: " + ", ".join(
                    f"{p} ({pos}, ESPN #{r})" for r, p, pos in q))
        if pr.get("note"):
            print(f"       note: {pr['note']}")
        seen.append(team)

    print("  ⚠️ context only — none of this moves the availability numbers. "
          "Three designs tried and failed (see need_redesign.py); managers "
          "here deviate from roster need often enough that assuming they "
          "follow it makes forecasts worse.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(OUT / "draft_state.json"))
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--pos", default=None, help="limit the board to one position")
    ap.add_argument("-n", type=int, default=12)
    a = ap.parse_args()

    st = {}
    p = Path(a.state)
    if p.exists():
        st = json.loads(p.read_text())
    picks = st.get("picks", [])
    slot = a.slot or st.get("slot", 6)
    now = st.get("pick", len(picks) + 1)

    b = pd.read_csv(OUT / "draft_board_2026.csv")
    b["note"] = b["note"].fillna("")
    b["key"] = b["player"].map(norm)

    # QB dependency label. Read from the JSON rather than the board CSV because
    # phase3 does not write it -- it is an annotation, not a model output, and
    # keeping it out of the projection pipeline is deliberate (the measured
    # expected cost is ~0.5 PPG, too small to move a ranking honestly).
    b["qbd"] = ""
    qbc = OUT.parent / "data" / "qb_context_2026.json"
    if qbc.exists():
        q = json.loads(qbc.read_text())
        alias = {"LA": "LAR", "SD": "LAC", "OAK": "LV", "STL": "LAR"}
        tm = b["team_2026"].replace(alias)
        pc = b["position"].isin(["WR", "TE"])
        b.loc[pc, "qbd"] = tm[pc].map(lambda t: (
            f"QB {q[t]['qb']} · {q[t]['tier']} risk"
            if isinstance(q.get(t), dict)
            and q[t]["tier"] in ("high", "elevated", "unknown") else "")
        ).fillna("")

    gone = {norm(x["name"]) for x in picks}
    mine = [x["name"] for x in picks if x.get("mine")]
    mine_keys = {norm(m) for m in mine}
    live = b[~b["key"].isin(gone)].copy()

    nxt = next((q for q in my_picks(slot) if q >= now), None)
    later = [q for q in my_picks(slot) if q > (nxt or now)][:1]

    print(f"\n=== pick {now} · slot {slot} · {len(picks)} off the board ===")
    if nxt:
        print(f"your next pick: {nxt}" + (f"  (then {later[0]})" if later else ""))

    print_watchlist(b, live, gone, nxt, later, now)

    # roster + needs
    have = b[b["key"].isin(mine_keys)]
    c = have["position"].value_counts().to_dict()
    need = []
    for pos, n in LINEUP.items():
        if c.get(pos, 0) < n:
            need.append(f"{pos} {c.get(pos,0)}/{n}")
    flex_have = sum(max(0, c.get(k, 0) - LINEUP[k]) for k in ("RB", "WR", "TE"))
    if flex_have < FLEX:
        need.append(f"FLEX {flex_have}/{FLEX}")
    print(f"roster ({len(mine)}): " + (", ".join(
        f"{r['player']} ({r['position']})" for _, r in have.iterrows()) or "—"))
    print("still need: " + (", ".join(need) if need else "starters full"))

    # Positional run -- reported as a COUNT, never as momentum. Measured over
    # this league's 391 historical picks, runs do not continue: 2-of-last-3 RB
    # is followed by an RB at lift 0.92, 4-of-last-5 WR at 0.90 -- both inside
    # noise of 1.0, and if anything negative. A WR run only looks like a
    # signal because WR is 42% of every skill pick here. Saying "run on WRs"
    # without that context is how a tool talks someone into reaching.
    recent = [norm(x["name"]) for x in picks[-12:]]
    rb = b[b["key"].isin(recent)]["position"].value_counts().to_dict()
    if rb:
        print("last 12 picks: " + "  ".join(f"{k} {v}" for k, v in rb.items())
              + "   (counts only — runs have not predicted more of that "
                "position in this league)")

    print_opponents(b, picks, nxt, now, slot, live)

    # Tier cliffs -- the thing that actually forces a pick. A small number here
    # means "take one now or lose the tier"; no cliff at all means the position
    # is smooth and you can wait, which is just as useful to know.
    WINDOW = 24
    print("\ntier cliff (how many left before the next real drop):")
    for pos in ("RB", "WR", "TE", "QB"):
        pl = live[live["position"] == pos].nlargest(WINDOW, "vor_pg")
        if pl.empty:
            continue
        vor = pl["vor_pg"].tolist()
        brk = next((i for i in range(1, len(vor))
                    if vor[i - 1] - vor[i] >= TIER_BREAK), None)
        top = pl.iloc[0]["player"]
        if brk is None:
            # no gap >= TIER_BREAK anywhere in the window: the position is flat,
            # so there is nothing to be forced by
            print(f"  {pos}: no cliff in the top {min(WINDOW, len(vor))} "
                  f"({vor[0]:.1f}→{vor[-1]:.1f} VOR) — safe to wait   "
                  f"(top: {top})")
        else:
            drop = vor[brk - 1] - vor[brk]
            print(f"  {pos}: {brk} left at {vor[0]:.1f}–{vor[brk-1]:.1f} VOR, "
                  f"then −{drop:.1f}   (top: {top})")

    # the board
    pool = live if not a.pos else live[live["position"] == a.pos.upper()]
    pool = pool.nsmallest(a.n, "my_rank")
    print(f"\nbest available{' — ' + a.pos.upper() if a.pos else ''}:")
    for _, r in pool.iterrows():
        av = avail(r, nxt, now) if nxt else None
        av_s = "fell" if av is None and not pd.isna(r["adp"]) else (
            f"{av*100:.0f}%" if av is not None else "—")
        flag = "!" if r["note"].startswith(NEGATIVE) else " "
        print(f" {flag}{int(r['my_rank']):>3}. {r['player']:<22} "
              f"{r['position']} {r['proj_ppg']:>5.1f} PPG  "
              f"VOR {r['vor_pg']:>4.1f}  ADP {r['adp']:>5.1f}  "
              f"→{nxt or '?'} {av_s:>5}")
        off = r.get("off_2026", "")
        if isinstance(off, str) and off and not off.startswith("OFF="):
            print(f"       [{off} team TDs vs 2025]")
        ctx = r.get("team_ctx", "")
        if isinstance(ctx, str) and ctx:
            print(f"       [{ctx}]")
        # QB dependency -- 41% of teams lose 3+ starts from their week-1 QB,
        # costing a #1 target ~1.3 PPG. A variance label, not a projection.
        qbd = r.get("qbd", "")
        if isinstance(qbd, str) and qbd:
            print(f"       [{qbd} — 41% of teams lose 3+ QB starts]")
        if r["note"]:
            print(f"       {r['note']}")
    print()


if __name__ == "__main__":
    main()
