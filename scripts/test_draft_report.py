"""draft_report.py: a hand-checkable case, then agreement with the panel.

The panel and the script score the same log with the same rules; if they ever
disagree one of them is wrong. So: (1) a tiny state where every number can be
worked out by hand, and (2) a 120-pick draft run through the panel's real
take() path, with the script's answer compared to the panel's own.

usage:  python scripts/test_draft_report.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_report as dr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---- 1. by hand --------------------------------------------------------------
# five players; the user (slot 1) holds picks 1 and 4. After pick 1 the panel
# said: A 90% to last to pick 4, B 50%, C 10%. Then picks 2,3 were B and C.
names = ["A", "B", "C", "D", "E"]
state = {
    "names": names,
    "picks": [{"name": "A", "mine": True}, {"name": "B"}, {"name": "C"},
              {"name": "D", "mine": True}],
    "log": [[2, 4, [[0, 0], [1, 50], [2, 10], [3, 80]]]],  # at pick 2: A already gone
}
# fix the story: at pick 2 (A gone) the panel said B 50%, C 10%, D 80%, to last to 4.
# Outcomes: B gone (picked at 2 < 4) -> 0; C gone (3 < 4) -> 0; D picked AT 4 -> 1.
# A (index 0) is in the log with 0% but A was taken at pick 1 < 4: y = 0.
r = dr.score(state)
ev = {e["name"]: e for e in r["events"]}
check("a player taken before the target pick did not last", ev["B"]["y"] == 0 and ev["C"]["y"] == 0)
check("a player taken AT the target pick is still there for it", ev["D"]["y"] == 1)
check("n counts every logged row", r["n"] == 4, str(r["n"]))
brier = ((0 - 0) ** 2 + (.5 - 0) ** 2 + (.1 - 0) ** 2 + (.8 - 1) ** 2) / 4
check("Brier is the mean squared error", abs(r["brier"] - brier) < 1e-12, f"{r['brier']} vs {brier}")
check("each prediction lands in exactly one bin", sum(b["n"] for b in r["bins"]) == r["n"])
check("0.8 goes in the top bin, 1.0 would too (no float seam)",
      next(b for b in r["bins"] if b["lo"] == 0.8)["n"] == 1)

future = {"names": names, "picks": state["picks"][:2], "log": [[2, 4, [[2, 10]]]]}
check("a prediction about a pick that has not happened is not scored",
      dr.score(future)["n"] == 0)

calls = dr.score({"names": names, "picks": state["picks"],
                  "log": [[2, 4, [[3, 80]]]]})["calls"]
check("your pick is matched to what was said at the turn before",
      [c["waited"] for c in calls] == [None, 0.8], str(calls))

same = {"names": names, "picks": state["picks"],
        "log": [[2, 4, [[3, 80]]]]}        # one event, and it lasted
r1 = dr.score(same)
check("all-same outcomes: skill is undefined, not a crash", r1["skill"] is None and r1["n"] == 1)
dr.print_report(r1)

# ---- 2. agreement with the panel ----------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    env = {**os.environ, "SB_DUMP": tmp}
    run = subprocess.run(["node", str(ROOT / "scripts" / "test_ingestion.mjs")],
                         env=env, capture_output=True, text=True)
    if run.returncode != 0 or not (Path(tmp) / "state.json").exists():
        check("the panel test ran and dumped its state", False, run.stdout[-300:])
    else:
        st = json.loads((Path(tmp) / "state.json").read_text())
        pj = json.loads((Path(tmp) / "report.json").read_text())
        py = dr.score(st)
        check("same number of predictions as the panel", py["n"] == pj["n"], f"{py['n']} vs {pj['n']}")
        check("same Brier as the panel", abs(py["brier"] - pj["brier"]) < 1e-9)
        check("same bins as the panel", all(
            a["n"] == b["n"] and (a["pred"] is None or abs(a["pred"] - b["pred"]) < 1e-9)
            and (a["act"] is None or abs(a["act"] - b["act"]) < 1e-9)
            for a, b in zip(py["bins"], pj["bins"])))
        check("same list of your picks, with the same odds",
              [(c["pick"], c["waited"]) for c in py["calls"]]
              == [(c["pick"], c["waited"]) for c in pj["calls"]])
        check("same biggest misses",
              [m["name"] for m in py["misses"]] == [m["name"] for m in pj["misses"]])

print(f"\n{'all passed' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
