"""Post-draft report: score the panel's predictions against what happened.

During a draft the panel writes down what it claimed -- after every pick, for the
40 best players left, "will he last to YOUR next pick?" -- and mirrors that log
to output/draft_state.json along with the picks. Afterwards this answers the
questions the numbers on screen cannot:

  * was it calibrated this draft? (said 70% -> did ~70% last?)
  * did it beat simply guessing the average? (a Brier skill score)
  * what did it say about the players you took, one turn earlier?
  * where was it most wrong?

It is a RECEIPT, not evidence. One draft is ~4,000 correlated predictions from
~12 independent picking decisions; the evidence that the model is calibrated is
replay_brier.py over 343 real turns. This is for a user to see how it did for
them, and the events it writes out (--csv) are the raw material for fitting a
league's calibration on its own picks.

The scoring mirrors the panel's Report button (buildReport in
build_draft_app.py) exactly -- same events, same bins -- so the two never
disagree.

usage:  draft_report.py [--state output/draft_state.json] [--csv events.csv]
                        [--plot reliability.png]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score(state: dict) -> dict:
    """Same maths as the panel. `state` is the mirrored draft_state.json."""
    names = state.get("names") or []
    picks = state.get("picks") or []
    log = state.get("log") or []
    total = len(picks)
    pick_of = {p["name"]: i + 1 for i, p in enumerate(picks)}

    ev = []
    for now, nxt, rows in log:
        if nxt - 1 > total:                      # his fate is not known yet
            continue
        for i, pct in rows:
            at = pick_of.get(names[i])
            ev.append({"name": names[i], "p": pct / 100,
                       "y": 1 if (at is None or at >= nxt) else 0,
                       "now": now, "nxt": nxt})
    if not ev:
        return {"n": 0}

    n = len(ev)
    brier = sum((e["p"] - e["y"]) ** 2 for e in ev) / n
    base = sum(e["y"] for e in ev) / n
    baseline = base * (1 - base)

    def bin_of(e):                               # integer maths, as in the panel
        return min(4, round(e["p"] * 100) // 20)

    bins = []
    for k in range(5):
        g = [e for e in ev if bin_of(e) == k]
        bins.append({"lo": k * 0.2, "n": len(g),
                     "pred": sum(e["p"] for e in g) / len(g) if g else None,
                     "act": sum(e["y"] for e in g) / len(g) if g else None})

    calls = []
    for i, pk in enumerate(picks):
        if not pk.get("mine"):
            continue
        k = i + 1
        waited = None
        for now, nxt, rows in log:
            if nxt == k:
                hit = next((pct for j, pct in rows if names[j] == pk["name"]), None)
                if hit is not None:
                    waited = hit / 100
                    break
        calls.append({"pick": k, "name": pk["name"], "waited": waited})

    worst, lucky = {}, {}
    for e in ev:                       # per player: most-wrong version of each
        if e["y"] == 0 and e["p"] >= .7 and e["p"] > worst.get(e["name"], {"p": -1})["p"]:
            worst[e["name"]] = e
        if e["y"] == 1 and e["p"] <= .3 and e["p"] < lucky.get(e["name"], {"p": 2})["p"]:
            lucky[e["name"]] = e

    return {"n": n, "brier": brier, "baseline": baseline,
            # undefined when every outcome was the same: nothing to beat
            "skill": 1 - brier / baseline if baseline > 0 else None,
            "mean_pred": sum(e["p"] for e in ev) / n, "mean_act": base,
            "bins": bins, "calls": calls, "events": ev,
            "misses": sorted(worst.values(), key=lambda e: -e["p"])[:5],
            "lucky": sorted(lucky.values(), key=lambda e: e["p"])[:3]}


def print_report(r: dict) -> None:
    if not r["n"]:
        print("nothing to score yet: no logged prediction has resolved")
        return
    pct = lambda v: f"{min(99, round(v * 100))}%"   # noqa: E731  (never claim 100%)
    print(f"DRAFT REPORT  ({r['n']} predictions scored)\n")
    skill = "n/a" if r["skill"] is None else f"{r['skill'] * 100:+.0f}%"
    print(f"  Brier {r['brier']:.4f}   guessing the average {r['baseline']:.4f}   "
          f"skill {skill}")
    gap = r["mean_pred"] - r["mean_act"]
    verdict = ("ran cautious, as it does in replay" if gap < -.03
               else "ran optimistic this draft" if gap > .03 else "well calibrated")
    print(f"  said {pct(r['mean_pred'])} on average, {pct(r['mean_act'])} lasted: {verdict}\n")
    print("  said        n   predicted -> actually lasted")
    for b in r["bins"]:
        if b["n"]:
            print(f"  {b['lo'] * 100:>3.0f}-{(b['lo'] + .2) * 100:<3.0f}% {b['n']:>5}   "
                  f"{pct(b['pred']):>5} -> {pct(b['act'])}")
    if r["calls"]:
        print("\n  your picks (what it said at the turn before)")
        for c in r["calls"]:
            w = "—" if c["waited"] is None else pct(c["waited"])
            print(f"   pick {c['pick']:>3}  {c['name']:<26} had {w:>4} to last")
    if r["misses"]:
        print("\n  biggest misses (said 70%+, gone)")
        for e in r["misses"]:
            print(f"   {e['name']:<26} {pct(e['p'])} at pick {e['now']}, gone before {e['nxt']}")
    if r["lucky"]:
        print("\n  lucky survivors (said 30% or less, lasted)")
        for e in r["lucky"]:
            print(f"   {e['name']:<26} {pct(e['p'])} at pick {e['now']}, there at {e['nxt']}")
    print("\n  a receipt, not evidence: one draft is ~12 independent decisions. "
          "See RESULTS.md.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(ROOT / "output" / "draft_state.json"))
    ap.add_argument("--csv", help="write every scored prediction here")
    ap.add_argument("--plot", help="write a reliability diagram here (needs matplotlib)")
    a = ap.parse_args()

    path = Path(a.state)
    if not path.exists():
        sys.exit(f"{path} not found -- run the draft with draft_state_server.py listening")
    state = json.loads(path.read_text())
    if "log" not in state:
        sys.exit("this state file has no prediction log (it predates the report); "
                 "rebuild the panel with build_draft_app.py and draft again")
    r = score(state)
    print_report(r)

    if a.csv and r["n"]:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "predicted", "lasted", "at_pick", "next_pick"])
            for e in r["events"]:
                w.writerow([e["name"], e["p"], e["y"], e["now"], e["nxt"]])
        print(f"\nwrote {a.csv}")
    if a.plot and r["n"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = [(b["pred"], b["act"], b["n"]) for b in r["bins"] if b["n"]]
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        ax.plot([0, 1], [0, 1], "--", color="#172019", lw=.8)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   s=[max(30, p[2] / 4) for p in pts], color="#1d7a4b")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="what the panel said",
               ylabel="share that actually lasted",
               title=f"This draft: Brier {r['brier']:.3f}")
        fig.savefig(a.plot, dpi=160, bbox_inches="tight")
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
