"""League import, offline: a synthetic Sleeper league through the whole path.

No network. A fake two-season league is fed through `fetch_chain` via an
injected `get`, written to a temp league directory, fitted by the real
league_tendencies.py (with an FFC market cached from the repo's own data), and
read back through league.py. The point is the contract: whatever the importer
writes must be exactly what the existing fitting code and the panel build read.

usage:  python scripts/test_league_import.py
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import league_import as li  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# real names + positions, so the fitted market can match them
ffc = json.loads((ROOT / "data" / "adp_ffc_ppr_2026.json").read_text())["players"]
POOL = [p for p in ffc if p["position"] in ("QB", "RB", "WR", "TE")]
TEAMS, ROUNDS = 10, 12
random.seed(3)


def fake_draft(season):
    """snake draft taken roughly in ADP order, with noise"""
    order = sorted(POOL, key=lambda p: p["adp"] + random.gauss(0, 4))
    picks = []
    for o in range(1, TEAMS * ROUNDS + 1):
        r, i = (o - 1) // TEAMS, (o - 1) % TEAMS
        slot = i + 1 if r % 2 == 0 else TEAMS - i
        pl = order[o - 1]
        first, _, last = pl["name"].partition(" ")
        picks.append({"round": r + 1, "pick_no": o, "draft_slot": slot,
                      "roster_id": slot, "player_id": str(pl["player_id"]),
                      "is_keeper": None,
                      "metadata": {"first_name": first, "last_name": last,
                                   "position": pl["position"]}})
    return picks


def make_api():
    api = {}
    for lid, prev, season, status in (("L25", "L24", "2025", "complete"),
                                      ("L24", None, "2024", "complete")):
        api[f"{li.API}/league/{lid}"] = {
            "league_id": lid, "name": "Test League", "season": season,
            "status": status, "previous_league_id": prev, "total_rosters": TEAMS,
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                                 "K", "DEF"] + ["BN"] * 6,
            "scoring_settings": {"rec": 0.5}}
        api[f"{li.API}/league/{lid}/drafts"] = [
            {"draft_id": f"D{lid}", "status": "complete", "type": "snake",
             "settings": {"teams": TEAMS, "rounds": ROUNDS},
             "slot_to_roster_id": {str(i): i for i in range(1, TEAMS + 1)}}]
        api[f"{li.API}/draft/D{lid}/picks"] = fake_draft(season)
        api[f"{li.API}/league/{lid}/users"] = [
            {"user_id": f"u{i}", "display_name": f"manager{i}"} for i in range(1, TEAMS + 1)]
        api[f"{li.API}/league/{lid}/rosters"] = [
            {"roster_id": i, "owner_id": f"u{i}"} for i in range(1, TEAMS + 1)]
    return api


api = make_api()
chain = li.fetch_chain("L25", 4, get=lambda url: api[url])
check("walks the previous-season chain", [c["league"]["season"] for c in chain] == ["2025", "2024"])

import contextlib
KEEP = os.environ.get("SNAKEBITE_TEST_KEEP")     # keep the league dir for inspection
with (contextlib.nullcontext(KEEP) if KEEP else tempfile.TemporaryDirectory()) as tmp:
    out = Path(tmp) / "L25"
    s = li.write_league(chain, out)
    check("settings read from the league, not defaults",
          s["teams"] == TEAMS and s["rounds"] == ROUNDS and s["scoring"] == "half-ppr"
          and s["lineup"] == {"QB": 1, "RB": 2, "WR": 2, "TE": 1} and s["flex"] == 1,
          str(s))
    check("a season file per completed draft",
          sorted(p.name for p in (out / "history").glob("season_*.json"))
          == ["season_2024.json", "season_2025.json"])
    check("pick count", s["_picks"] == 2 * TEAMS * ROUNDS, str(s["_picks"]))

    s25 = json.loads((out / "history" / "season_2025.json").read_text())
    check("rounds carried into the season file", s25["rounds"] == ROUNDS)
    check("picks carry overall/round/team/pid like the ESPN shape",
          set(s25["picks"][0]) >= {"o", "r", "rp", "team", "pid", "auto"})

    # market the fitter will read: cache the repo's FFC pull as each year's market
    for y in (2024, 2025):
        (out / "history" / f"adp_ffc_half_ppr_{y}.json").write_text(
            (ROOT / "data" / "adp_ffc_ppr_2026.json").read_text())

    env = {**os.environ, "SNAKEBITE_LEAGUE": str(out)}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "league_tendencies.py")],
                       env=env, capture_output=True, text=True)
    check("league_tendencies fits on the imported history", r.returncode == 0,
          (r.stderr or r.stdout)[-300:])
    t = json.loads((out / "league_tendencies.json").read_text())
    check("fit used both seasons", t["generated_from"] == [2024, 2025])
    check("every manager has a record", len(t["managers"]) == TEAMS, str(len(t["managers"])))
    check("a sigma curve was fitted", len(t["sigma_curve"]["pts"]) >= 3)

    li.write_opponents(out, None)
    rows = (out / "opponents.csv").read_text().splitlines()
    check("opponents.csv has one row per team, all vets", len(rows) == TEAMS + 1
          and all(",vet," in r for r in rows[1:]))

    # the active-league plumbing reads the same directory
    os.environ["SNAKEBITE_LEAGUE"] = str(out)
    import league
    check("league.py resolves the directory",
          league.settings()["teams"] == TEAMS
          and league.tendencies_path() == out / "league_tendencies.json"
          and league.history_dir() == out / "history")
    check("non-PPR scoring is flagged, not hidden",
          any("half-ppr" in w for w in league.warnings()))

# a pending (unplayed) draft supplies seats; an auction draft is refused
auc = li.convert_draft(chain[0]["league"], {"type": "auction"}, [], [], [])
check("auction drafts are skipped, not mis-parsed", auc is None)

print(f"\n{'all passed' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
