"""Zero-config demo: replay a synthetic draft through the real state path.

A stranger cloning this repo has no ESPN league and no draft room. This script
fakes the draft room and nothing else: it generates picks, POSTs them to the
real `draft_state_server.py` handler in the exact payload the panel sends
(`{pick, slot, picks:[{name, mine, auto, team, at}], ts}`), and then runs the
real `draft_advise.py` over the file that handler wrote.

⚠️ THE PICKS ARE SIMULATED, NOT OBSERVED. Every seat -- yours included --
takes the player with the lowest `platform ADP + noise`. ADP is the platform's
published market (--platform espn|ffc), NOT this project's model ranking: the
backtest (board_backtest.py) found platform ADP beats the model's ordering in
the top 120, so the simulator drafts the way the market does. Noise is a
rank-dependent spread fitted to the board's own sigma. That makes the briefing
a demonstration of the mechanics, not evidence the model is right. The evidence
is `replay_brier.py`, which scores the same probabilities against real drafts.

State goes to output/demo_state.json, never output/draft_state.json, so this
cannot clobber a live draft.

usage:  demo.py [--through 27] [--slot 3] [--seed 7] [--platform espn|ffc]
                [--quiet] [--panel]

--panel also opens the browser panel and replays the same picks into it, one
at a time, through the same `DRAFTAPP.take` call the ESPN feed uses.
"""

import argparse
import contextlib
import csv
import io
import json
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_state_server as server          # noqa: E402
from draft_advise import TEAMS, ROUNDS, norm, seat_of_pick  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DEMO_STATE = OUT / "demo_state.json"
PORT = 8799        # not 8788: a real draft's listener may be running


def seat_to_team() -> dict:
    """draft slot -> ESPN team id, from the opponent profiles."""
    txt = "\n".join(l for l in (ROOT / "data" / "opponents_2026.csv")
                    .read_text().splitlines() if not l.lstrip().startswith("#"))
    return {int(r["slot"]): int(r["espn_team_id"])
            for r in csv.DictReader(txt.splitlines())
            if r["slot"].strip() and r["espn_team_id"].strip()}


def platform_adp(platform: str, board: pd.DataFrame) -> pd.DataFrame:
    """name / adp of every player on the board that the platform prices."""
    f = ROOT / "data" / f"adp_{platform}_ppr_2026.json"
    rows = json.loads(f.read_text())["players"]
    t = pd.DataFrame({"key": [norm(r["name"]) for r in rows],
                      "padp": [r.get("adp") for r in rows]})
    t = t.dropna().drop_duplicates("key")
    b = board.assign(key=board["player"].map(norm))
    return b.merge(t, on="key")[["player", "key", "padp"]]


def simulate(board: pd.DataFrame, through: int, slot: int, seed: int,
             platform: str) -> list:
    rng = np.random.default_rng(seed)
    pool = platform_adp(platform, board)
    # spread: fitted to the board's own (league-calibrated) adp_sigma vs adp
    fit = board.dropna(subset=["adp", "adp_sigma"])
    co = np.polyfit(np.sqrt(fit["adp"]), fit["adp_sigma"], 1)
    sd = np.clip(np.polyval(co, np.sqrt(pool["padp"])), 0.5, None)
    mu = pool["padp"].to_numpy()
    names = pool["player"].to_numpy()
    seat_team = seat_to_team()
    taken = np.zeros(len(pool), dtype=bool)
    picks = []
    for at in range(1, through + 1):
        seat = seat_of_pick(at)
        score = mu + sd * rng.standard_normal(len(pool))
        score[taken] = np.inf
        i = int(np.argmin(score))
        taken[i] = True
        picks.append({"name": names[i], "mine": seat == slot, "auto": True,
                      "team": seat_team.get(seat), "at": at})
    return picks


PANEL_JS = """<script>
/* demo replay: feeds the simulated picks through the same call the ESPN feed
   uses, one at a time, so the panel reacts as it would live */
(function(){
  const PICKS = __PICKS__, SLOT = __SLOT__;
  const go = () => {
    const A = window.DRAFTAPP;
    if (!A) return setTimeout(go, 100);
    const st = A.state(); st.picks.length = 0; st.seats = {}; st.apiPick = null;
    A.setSlot(SLOT);
    let i = 0;
    const tick = () => {
      if (i >= PICKS.length) return;
      const p = PICKS[i++];
      A.take(p.name, p.mine, true, p.team, p.at);
      setTimeout(tick, 500);
    };
    setTimeout(tick, 800);
  };
  window.addEventListener("load", go);
})();
</script>
"""


def open_panel(picks: list, slot: int) -> Path:
    html = OUT / "draft_day.html"
    if not html.exists():
        import build_draft_app
        build_draft_app.main()
    page = html.read_text() + PANEL_JS.replace(
        "__PICKS__", json.dumps(picks)).replace("__SLOT__", str(slot))
    dest = OUT / "demo_panel.html"
    dest.write_text(page)
    webbrowser.open(dest.as_uri())
    return dest


def post(port: int, payload: dict) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/state", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=5).close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", type=int, default=27,
                    help="stop after this overall pick (default 27: the "
                         "27->46 gap surfaces nine opponents)")
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--platform", choices=("espn", "ffc"), default="espn",
                    help="whose ADP drives the simulated picks (default espn)")
    ap.add_argument("--panel", action="store_true",
                    help="also open the browser panel and replay into it")
    ap.add_argument("--quiet", action="store_true", help="skip the briefing")
    a = ap.parse_args()
    if not 1 <= a.through < TEAMS * ROUNDS:
        sys.exit(f"--through must be 1..{TEAMS * ROUNDS - 1}")

    board_csv = OUT / "draft_board_2026.csv"
    if not board_csv.exists():
        sys.exit(f"{board_csv} missing -- run the rebuild in README first")
    board = pd.read_csv(board_csv)

    # the real handler, pointed at a demo file, on a demo port
    server.STATE = DEMO_STATE
    httpd = HTTPServer(("127.0.0.1", PORT), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    picks = simulate(board, a.through, a.slot, a.seed, a.platform)
    with contextlib.redirect_stdout(io.StringIO()):   # handler logs each POST
        for n in range(1, a.through + 1):
            post(PORT, {"pick": n + 1, "slot": a.slot, "picks": picks[:n],
                        "ts": n})
    httpd.shutdown()

    mine = [p["name"] for p in picks if p["mine"]]
    print(f"replayed {a.through} simulated picks through the state server "
          f"-> {DEMO_STATE.relative_to(ROOT)}")
    print(f"picks drafted off {a.platform.upper()} ADP, seed {a.seed}")
    print(f"your simulated picks (slot {a.slot}): {', '.join(mine) or '—'}")
    if a.panel:
        print(f"panel: {open_panel(picks, a.slot).relative_to(ROOT)} "
              f"(replaying {a.through} picks)")
    if a.quiet:
        return
    subprocess.run([sys.executable, str(Path(__file__).with_name(
        "draft_advise.py")), "--state", str(DEMO_STATE), "--slot", str(a.slot)],
        check=True)


if __name__ == "__main__":
    main()
