"""Fantasy platforms and their ADP, normalised to one shape.

Each platform publishes its own market, and the markets disagree enough to
matter (ESPN drafters take TEs and QBs early and let mid-round RB/WR slide, see
espn_translate.py). So the panel carries one ADP column per platform and the
user picks the one their league drafts on.

Where the numbers come from -- and what is NOT supported, said plainly:

  espn      cached pull of ESPN's public ADP            (data/adp_espn_ppr_2026.json)
  sleeper   Sleeper's public projections API, `adp_ppr`  (data/adp_sleeper_ppr_2026.json)
              refresh:  platforms.py fetch sleeper
  ffc       FantasyFootballCalculator, a cross-site consensus pooled over many
            leagues -- the default, and the only one with a measured per-player
            stdev                                        (data/adp_ffc_ppr_2026.json)
  yahoo, nfl, cbs, underdog
            NO public ADP endpoint we can use. Drop an export or a copy-paste of
            the site's ADP page into data/adp_import/<id>.csv (columns: name,adp)
            and the tab lights up. Any other site: add a row to PLATFORMS and a
            CSV of the same name.

⚠️ CALIBRATION. The availability spread (`sig`) and the chalk drift (`shift`)
were fitted against a league's 2023-25 drafts in FFC rank space
(league_tendencies.json, validated by replay_brier.py). For FFC that is exactly
the validated model. For every other platform the SAME curve is applied in that
platform's own rank space -- an approximation, not a measurement, and the panel
says so on the tab. Re-fit by running league_tendencies.py on a league that
drafted on that platform.

usage:  platforms.py fetch sleeper [--scoring ppr|half_ppr|std]
        platforms.py status
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMPORT = DATA / "adp_import"
POSITIONS = ("QB", "RB", "WR", "TE")

# id -> label, kind, how to get it. Order is tab order.
PLATFORMS = {
    "ffc":      {"label": "Consensus", "kind": "cache",  "calibrated": True,
                 "src": "FantasyFootballCalculator, pooled across sites"},
    "espn":     {"label": "ESPN",      "kind": "cache",  "calibrated": False,
                 "src": "ESPN public ADP"},
    "sleeper":  {"label": "Sleeper",   "kind": "cache",  "calibrated": False,
                 "src": "Sleeper projections API (adp_ppr)"},
    "yahoo":    {"label": "Yahoo",     "kind": "import", "calibrated": False,
                 "src": "no public ADP API -- import a CSV"},
    "nfl":      {"label": "NFL.com",   "kind": "import", "calibrated": False,
                 "src": "no public ADP API -- import a CSV"},
    "cbs":      {"label": "CBS",       "kind": "import", "calibrated": False,
                 "src": "no public ADP API -- import a CSV"},
    "underdog": {"label": "Underdog",  "kind": "import", "calibrated": False,
                 "src": "no public ADP API -- import a CSV"},
}

NAME_ALIASES = {"kenny gainwell": "kenneth gainwell"}


def norm(s: str) -> str:
    s = re.sub(r"[^a-z\s]", "", str(s).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NAME_ALIASES.get(s, s)


# ------------------------------------------------------------------- loading

def _cache_path(pid: str) -> Path:
    return DATA / f"adp_{pid}_ppr_2026.json"


def load(pid: str) -> pd.DataFrame | None:
    """key / adp for every player the platform prices, or None if unavailable."""
    p = PLATFORMS[pid]
    if p["kind"] == "cache":
        f = _cache_path(pid)
        if not f.exists():
            return None
        rows = json.loads(f.read_text())["players"]
        t = pd.DataFrame({"name": [r["name"] for r in rows],
                          "adp": [r.get("adp") for r in rows]})
    else:
        f = IMPORT / f"{pid}.csv"
        if not f.exists():
            return None
        t = pd.read_csv(f, comment="#")
        cols = {c.lower().strip(): c for c in t.columns}
        name = next((cols[c] for c in ("name", "player") if c in cols), None)
        adp = next((cols[c] for c in ("adp", "avg pick", "average pick", "avg")
                    if c in cols), None)
        if name is None or adp is None:
            raise SystemExit(f"{f}: need columns name,adp (found {list(t.columns)})")
        t = t.rename(columns={name: "name", adp: "adp"})[["name", "adp"]]
    t["adp"] = pd.to_numeric(t["adp"], errors="coerce")
    t = t.dropna(subset=["adp"])
    t = t[t["adp"] < 900]               # Sleeper marks "undrafted" as 999
    t["key"] = t["name"].map(norm)
    return t.sort_values("adp").drop_duplicates("key")[["key", "adp"]]


def ffc_stdev() -> dict:
    """key -> FFC's measured stdev of draft slot. Only FFC publishes one."""
    f = _cache_path("ffc")
    if not f.exists():
        return {}
    return {norm(r["name"]): r["stdev"]
            for r in json.loads(f.read_text())["players"] if r.get("stdev")}


def espn_ids() -> dict:
    """key -> ESPN player id, used only to look up a headshot."""
    f = _cache_path("espn")
    if not f.exists():
        return {}
    return {norm(r["name"]): r["espn_id"]
            for r in json.loads(f.read_text())["players"] if r.get("espn_id")}


def spread(adp: pd.Series, keys: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(sig, shift) for a platform's ADP, using the league calibration.

    Same construction as phase3_draft_board.fetch_adp: rank-dependent spread
    multiplier and a pooled chalk drift, both from league_tendencies.json.
    Without that file the raw spread is used and drift is zero.

    The raw spread is `max(0.5 + 0.04*adp, FFC's measured stdev)`. The measured
    stdev matters: QBs carry a stdev ~13 at ADP 30 because drafters disagree on
    when to take one, and the 0.04*adp floor alone says 1.7 -- which would make
    every "you can wait" call far too confident. Other platforms publish no
    spread, so FFC's per-player value is borrowed (an assumption).
    """
    sd = keys.map(ffc_stdev()).fillna(0.0)
    raw = np.maximum(0.5 + 0.04 * adp, sd)
    tend = DATA / "league_tendencies.json"
    if not tend.exists():
        return raw, pd.Series(0.0, index=adp.index)
    c = json.loads(tend.read_text())["sigma_curve"]
    mrank = adp.rank(method="first")
    mult = np.clip(np.interp(np.log(np.maximum(mrank, 1.0)),
                             np.log([p[0] for p in c["pts"]]),
                             [p[1] for p in c["pts"]]), c["floor"], c["cap"])
    taper = np.clip(np.interp(mrank, [c["drift_full"], c["drift_zero"]],
                              [1.0, 0.0]), 0.0, 1.0)
    return raw * mult, (c["drift"] * taper * raw).round(2)


def for_board(board: pd.DataFrame) -> tuple[dict, list]:
    """Per-player [adp, sig, shift] for every ready platform, plus tab metadata.

    `board` needs player / adp / adp_sigma / adp_shift. The default platform
    (ffc) passes the board's own validated values through untouched.
    """
    keys = board["player"].map(norm)
    per_player = {name: {} for name in board["player"]}
    meta = []
    for pid, p in PLATFORMS.items():
        if pid == "ffc":
            ok = board["adp"].notna()
            for name, a, s, sh in zip(board.loc[ok, "player"], board.loc[ok, "adp"],
                                      board.loc[ok, "adp_sigma"],
                                      board.loc[ok, "adp_shift"]):
                per_player[name][pid] = [round(float(a), 1), round(float(s), 2),
                                         round(float(sh), 2)]
            meta.append({"id": pid, "label": p["label"], "ready": True,
                         "n": int(ok.sum()), "calibrated": True, "src": p["src"]})
            continue
        t = load(pid)
        if t is None or t.empty:
            meta.append({"id": pid, "label": p["label"], "ready": False,
                         "n": 0, "calibrated": False, "src": p["src"]})
            continue
        t["sig"], t["shift"] = spread(t["adp"], t["key"])
        hit = t.set_index("key")
        n = 0
        for name, k in zip(board["player"], keys):
            if k in hit.index:
                r = hit.loc[k]
                per_player[name][pid] = [round(float(r["adp"]), 1),
                                         round(float(r["sig"]), 2),
                                         round(float(r["shift"]), 2)]
                n += 1
        meta.append({"id": pid, "label": p["label"], "ready": n > 0, "n": n,
                     "calibrated": False, "src": p["src"]})
    return per_player, meta


# --------------------------------------------------------------------- fetch

def fetch_sleeper(scoring: str = "ppr") -> Path:
    import requests
    url = ("https://api.sleeper.com/projections/nfl/2026?season_type=regular"
           + "".join(f"&position[]={p}" for p in POSITIONS)
           + f"&order_by=adp_{scoring}")
    rows = requests.get(url, timeout=60).json()
    out = []
    for r in rows:
        adp = (r.get("stats") or {}).get(f"adp_{scoring}")
        pl = r.get("player") or {}
        if adp is None or adp >= 900 or pl.get("position") not in POSITIONS:
            continue
        out.append({"name": f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip(),
                    "position": pl["position"], "team": r.get("team"), "adp": adp})
    out.sort(key=lambda x: x["adp"])
    dest = DATA / f"adp_sleeper_{'ppr' if scoring == 'ppr' else scoring}_2026.json"
    dest.write_text(json.dumps({"meta": {"source": "api.sleeper.com projections",
                                         "field": f"adp_{scoring}", "n": len(out)},
                                "players": out}, indent=1))
    print(f"sleeper: {len(out)} priced players -> {dest.relative_to(ROOT)}")
    return dest


def status() -> None:
    for pid, p in PLATFORMS.items():
        t = load(pid)
        state = f"{len(t)} players" if t is not None else "not loaded"
        print(f"  {pid:<9}{p['label']:<10}{state:<16}{p['src']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("platform", choices=("sleeper",))
    f.add_argument("--scoring", default="ppr", choices=("ppr", "half_ppr", "std"))
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch_sleeper(a.scoring)
    else:
        status()


if __name__ == "__main__":
    main()
