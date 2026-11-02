"""The active league: its settings and where its files live.

Until now teams, rounds, lineup and the file locations were constants copied
into four scripts (the 15-vs-16-round off-by-one bit twice because of it). They
live here once.

With nothing set, the active league is the bundled demo league (12 teams, 15
rounds, PPR, files in data/). To work on another league, point
SNAKEBITE_LEAGUE at its directory, which `league_import.py` creates:

    SNAKEBITE_LEAGUE=data/leagues/<id> python scripts/phase3_draft_board.py 3

A league directory holds league.json (settings), history/ (past drafts in the
season_YYYY.json shape), league_tendencies.json (fitted by league_tendencies.py)
and optionally opponents.csv. It is private to the user and gitignored.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DEFAULTS = {
    "name": "demo",
    "teams": 12,
    "rounds": 15,
    "lineup": {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    "flex": 2,                 # RB/WR/TE flex spots
    "superflex": False,        # not modelled -- see warnings()
    "scoring": "ppr",          # ppr | half-ppr | standard (FFC's spellings)
    "season": 2026,
}


def active_dir() -> Path | None:
    d = os.environ.get("SNAKEBITE_LEAGUE")
    return (Path(d) if Path(d).is_absolute() else ROOT / d) if d else None


def settings() -> dict:
    s = dict(DEFAULTS)
    d = active_dir()
    if d and (d / "league.json").exists():
        s.update(json.loads((d / "league.json").read_text()))
    return s


def history_dir() -> Path:
    d = active_dir()
    return d / "history" if d else DATA / "league_history"


def tendencies_path() -> Path:
    d = active_dir()
    return d / "league_tendencies.json" if d else DATA / "league_tendencies.json"


def teams_path() -> Path:
    """The current season's teamId -> owner map."""
    d = active_dir()
    return (d / "history" / "teams_current.json" if d
            else DATA / "league_history" / "teams_2026.json")


def opponents_csv() -> Path:
    d = active_dir()
    return d / "opponents.csv" if d else DATA / "opponents_2026.csv"


def adp_cache(season: int | None = None) -> Path:
    """FFC pull for the league's scoring format. The default league keeps the
    original file name so nothing already committed moves."""
    s = settings()
    fmt = s["scoring"]
    return DATA / f"adp_ffc_{fmt.replace('-', '_')}_{season or s['season']}.json"


def warnings() -> list[str]:
    """Things the settings ask for that this tool does not model. Said once, up
    front, rather than silently producing a board for a different league."""
    s, out = settings(), []
    if s["superflex"]:
        out.append("superflex/2QB: the board values QBs as a 1QB league. "
                   "Availability is unaffected (it comes from ADP).")
    if s["scoring"] != "ppr":
        out.append(f"scoring is {s['scoring']}: projections and VOR are "
                   "full-PPR. Availability is unaffected (it comes from ADP).")
    return out
