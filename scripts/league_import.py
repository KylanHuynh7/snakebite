"""Import a league's draft history and fit its calibration.

The spread and chalk-drift curves behind every availability number were fitted
on one league's 2023-25 drafts (league_tendencies.py). That makes them true for
that room and an assumption everywhere else. This closes the gap: point it at a
league and it fits the curves on THAT league's own history.

Sleeper only, because it is the one major platform whose draft data is public
and needs no login: league ids are not secret, and these endpoints are the
documented read-only API (https://docs.sleeper.com). ESPN and Yahoo drafts sit
behind a login, so those leagues stay manual.

    league_import.py sleeper <league_id> [--seasons 4] [--no-fit]

What it does:
  1. walks the league's `previous_league_id` chain, newest season first
  2. converts each completed SNAKE draft into the season_YYYY.json shape that
     league_tendencies.py already reads (so the fitting code is unchanged)
  3. writes league.json (teams, rounds, lineup, scoring) from the league's
     own settings
  4. fits league_tendencies.json and writes opponents.csv, with managers who
     have history marked `vet` and the rest `new`

Everything lands in data/leagues/<league_id>/, which is gitignored: it holds
real people's draft behaviour, and that is theirs, not the repo's.

LIMITS, stated up front:
  * Sleeper does not flag autopicks, so every pick counts as a human pick. In a
    league where people leave the draft to the clock this overstates how
    human the history is.
  * Keeper picks are dropped (they are not market decisions). Auction and
    linear drafts are skipped.
  * Fewer than ~150 picks of history is thin. The fit says so rather than
    pretending; `league_tendencies.py` shrinks per-manager numbers toward the
    league mean, and a manager with no history sits exactly on it.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.sleeper.app/v1"
POS_CODE = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DEF": 16}
MIN_PICKS = 150


def http_get(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- conversion

def lineup_from(league: dict) -> tuple[dict, int, bool]:
    """roster_positions -> (QB/RB/WR/TE starters, flex spots, superflex)."""
    pos = league.get("roster_positions") or []
    lineup = {p: pos.count(p) for p in ("QB", "RB", "WR", "TE")}
    flex = pos.count("FLEX") + pos.count("REC_FLEX") + pos.count("WRRB_FLEX")
    return lineup, flex, "SUPER_FLEX" in pos


def scoring_from(league: dict) -> str:
    """FFC's format names, from the per-reception value."""
    rec = (league.get("scoring_settings") or {}).get("rec", 0)
    return "ppr" if rec >= 1 else "half-ppr" if rec >= 0.5 else "standard"


def settings_from(league: dict, draft: dict | None) -> dict:
    lineup, flex, sflex = lineup_from(league)
    dset = (draft or {}).get("settings") or {}
    return {
        "name": league.get("name", league.get("league_id")),
        "source": "sleeper", "league_id": league["league_id"],
        "teams": int(dset.get("teams") or league.get("total_rosters") or 12),
        "rounds": int(dset.get("rounds") or 15),
        "lineup": lineup, "flex": flex, "superflex": sflex,
        "scoring": scoring_from(league),
        "season": int(league["season"]) + (1 if league.get("status") == "complete" else 0),
    }


def convert_draft(league: dict, draft: dict, picks: list, users: list,
                  rosters: list) -> dict | None:
    """One Sleeper draft -> the ESPN-shaped season file. None if unusable."""
    if draft.get("type") != "snake":
        print(f"  {league['season']}: skipped, {draft.get('type')} draft")
        return None
    teams_n = int((draft.get("settings") or {}).get("teams") or len(rosters))
    owner_of = {r["roster_id"]: r.get("owner_id") for r in rosters}
    name_of = {u["user_id"]: u.get("display_name") or u["user_id"] for u in users}

    out_picks, players, kept = [], {}, 0
    for p in sorted(picks, key=lambda x: x["pick_no"]):
        if p.get("is_keeper"):
            kept += 1
            continue
        md = p.get("metadata") or {}
        pid = str(p["player_id"])
        players[pid] = {"n": f"{md.get('first_name', '')} {md.get('last_name', '')}".strip(),
                        "pos": POS_CODE.get(md.get("position"), 0), "pro": 0}
        out_picks.append({"o": p["pick_no"], "r": p["round"],
                          "rp": (p["pick_no"] - 1) % teams_n + 1,
                          "team": p["roster_id"], "pid": pid, "auto": 0,
                          "keeper": False})
    if not out_picks:
        return None
    print(f"  {league['season']}: {len(out_picks)} picks"
          + (f" ({kept} keepers dropped)" if kept else ""))
    return {
        "year": int(league["season"]), "size": teams_n,
        "rounds": int((draft.get("settings") or {}).get("rounds") or 15),
        "draftType": "SNAKE",
        "pickOrder": [draft["slot_to_roster_id"][str(i)]
                      for i in range(1, teams_n + 1)]
        if draft.get("slot_to_roster_id") else [],
        "members": [{"id": u["user_id"], "name": name_of[u["user_id"]]} for u in users],
        "teams": [{"id": r["roster_id"],
                   "name": name_of.get(r.get("owner_id"), f"Team {r['roster_id']}"),
                   "owners": [r["owner_id"]] if r.get("owner_id") else []}
                  for r in rosters],
        "picks": out_picks, "players": players,
    }


# --------------------------------------------------------------------- fetch

def fetch_chain(league_id: str, seasons: int, get=http_get) -> list[dict]:
    """[{league, draft, picks, users, rosters}] newest first."""
    chain, lid = [], league_id
    while lid and lid != "0" and len(chain) < seasons:
        lg = get(f"{API}/league/{lid}")
        alld = get(f"{API}/league/{lid}/drafts")
        done = [d for d in alld if d.get("status") == "complete"]
        todo = [d for d in alld if d.get("status") != "complete"]
        d = done[0] if done else None
        # `pending`: a draft not run yet -- it is what tells us who sits where
        chain.append({"league": lg, "draft": d,
                      "pending": todo[0] if todo else None,
                      "picks": get(f"{API}/draft/{d['draft_id']}/picks") if d else [],
                      "users": get(f"{API}/league/{lid}/users"),
                      "rosters": get(f"{API}/league/{lid}/rosters")})
        lid = lg.get("previous_league_id")
    return chain


# --------------------------------------------------------------------- write

def write_league(chain: list[dict], out: Path) -> dict:
    """Write the league directory. Returns the settings it wrote."""
    hist = out / "history"
    hist.mkdir(parents=True, exist_ok=True)
    newest = chain[0]
    s = settings_from(newest["league"], newest["draft"]
                      or next((c["draft"] for c in chain if c["draft"]), None))
    (out / "league.json").write_text(json.dumps(s, indent=1))

    n = 0
    for c in chain:
        if c["draft"] is None:
            continue
        season = convert_draft(c["league"], c["draft"], c["picks"],
                               c["users"], c["rosters"])
        if season:
            (hist / f"season_{season['year']}.json").write_text(
                json.dumps(season, separators=(",", ":")))
            n += len(season["picks"])

    users = {u["user_id"]: u.get("display_name") or u["user_id"]
             for u in newest["users"]}
    cur = {"year": s["season"],
           "members": [{"id": k, "name": v} for k, v in users.items()],
           "teams": [{"id": r["roster_id"],
                      "name": users.get(r.get("owner_id"), f"Team {r['roster_id']}"),
                      "owners": [r["owner_id"]] if r.get("owner_id") else []}
                     for r in newest["rosters"]]}
    (hist / "teams_current.json").write_text(json.dumps(cur, separators=(",", ":")))
    s["_picks"] = n
    return s


def write_opponents(out: Path, newest_draft: dict | None) -> None:
    """opponents.csv: seat + experience. `vet` = has pick history here."""
    tend = json.loads((out / "league_tendencies.json").read_text())["managers"]
    cur = json.loads((out / "history" / "teams_current.json").read_text())
    slot_of = {}
    if newest_draft and newest_draft.get("slot_to_roster_id"):
        slot_of = {int(v): int(k) for k, v in newest_draft["slot_to_roster_id"].items()}
    with (out / "opponents.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["espn_team_id", "team", "manager", "slot", "experience", "note"])
        for t in cur["teams"]:
            owner = t["owners"][0] if t["owners"] else None
            w.writerow([t["id"], t["name"], t["name"], slot_of.get(t["id"], ""),
                        "vet" if owner in tend else "new", ""])


def fit(out: Path) -> None:
    env = {**__import__("os").environ, "SNAKEBITE_LEAGUE": str(out)}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "league_tendencies.py")],
                   check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sl = sub.add_parser("sleeper")
    sl.add_argument("league_id")
    sl.add_argument("--seasons", type=int, default=4)
    sl.add_argument("--dir", default=None)
    sl.add_argument("--no-fit", action="store_true")
    a = ap.parse_args()

    out = Path(a.dir) if a.dir else ROOT / "data" / "leagues" / a.league_id
    print(f"importing Sleeper league {a.league_id} -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    chain = fetch_chain(a.league_id, a.seasons)
    s = write_league(chain, out)
    picks = s.pop("_picks")
    print(f"\n{s['name']}: {s['teams']} teams, {s['rounds']} rounds, {s['scoring']}, "
          f"lineup {s['lineup']} + {s['flex']} flex")
    if picks < MIN_PICKS:
        print(f"  !! only {picks} picks of history (< {MIN_PICKS}): the fit will be thin")
    if picks == 0:
        sys.exit("no completed snake drafts found -- nothing to fit")
    if a.no_fit:
        return
    fit(out)
    write_opponents(out, chain[0].get("pending") or chain[0]["draft"])
    print(f"\ndone. build the board and panel for this league with:\n"
          f"  export SNAKEBITE_LEAGUE={out}\n"
          f"  python scripts/phase3_draft_board.py <your slot>\n"
          f"  python scripts/build_draft_app.py")


if __name__ == "__main__":
    main()
