"""Draft-day tool: bake the board into a standalone page and a Chrome extension.

Why a generated file and not a web app: on draft night the only thing that
matters is that it opens and works. A single self-contained file with the data
baked in has no server to start, no CORS problem reading a local CSV, no
network dependency, and nothing that can be down.

Why generated and not hand-written: the board changes. Re-pull ADP, rerun
phase3, rerun this, and the tool is current. Every number in it comes from
`output/draft_board_2026.csv`.

The panel leads with a RECOMMENDATION rather than a ranked list, because a
ranked list is the thing you already have on paper. The recommendation is
value-over-next-available: for each position, compare the best man on the board
now against the value you can expect to still be there at your next pick, given
who has actually gone. The position with the most to lose by waiting is the
position to draft. That is the question a draft actually asks -- not "who is
best?" but "who will not be here when I come back?"

  * Conditional availability. The board's `avail_pk` columns answer "what were
    the odds, preseason, that he lasts to pick 30?" Once picks are happening
    the useful question is "he is STILL here at 18 -- will he last to 30?",
    which is P(avail at 30) / P(avail at 18).
  * Run detection. Shows the position split of recent picks. Deliberately does
    NOT adjust ADP for it -- that would be inventing precision.

Two build targets, and they differ in one way that matters:

  * `output/draft_day.html` -- everything inline. Opens off the filesystem.
  * `output/espn_extension/` -- JS in SEPARATE FILES. MV3 extension pages block
    inline <script> under the default CSP, so an inline build renders the
    markup and silently executes nothing.

usage:  build_draft_app.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd

import league
import platforms

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
EXT = OUT / "espn_extension"
SEASON = 2026
TIER_BREAK = 0.8
STATE_PORT = 8788
PANEL_W = 380          # narrow enough to sit beside the ESPN draft board

# a stance that argues against a player argues against his price -- mirrors
# NEGATIVE in phase3_draft_board.py so the two views agree
NEGATIVE = ("AVOID", "FADE", "RISK", "IR STASH")
POSITIVE = ("TARGET", "BOUNCE-BACK")


def load_board() -> tuple[list[dict], list[dict]]:
    b = pd.read_csv(OUT / "draft_board_2026.csv")
    for c in ("note", "rookie_info", "team_2026", "td_dep", "status",
              "status_note"):
        b[c] = b[c].fillna("")

    # Keep anyone the market drafts, plus the top of the model board even if
    # FFC has no line on him -- those are exactly the late fliers worth having
    # in front of you when the room has thinned out.
    b = b[b["adp"].notna() | (b["my_rank"] <= 200)].copy()

    b = b.sort_values("vor_pg", ascending=False)
    b["tier"] = 0
    for pos in ("QB", "RB", "WR", "TE"):
        m = b["position"] == pos
        vor = b.loc[m, "vor_pg"]
        b.loc[m, "tier"] = (vor.diff().fillna(0) <= -TIER_BREAK).cumsum() + 1

    def flag(note: str) -> str:
        if note.startswith(NEGATIVE):
            return "neg"
        if note.startswith(POSITIVE):
            return "pos"
        return "warn" if note else ""

    b["flag"] = b["note"].map(flag)

    # The board stores the TD-dependency flag in full ("p96 (-2.1 PPG hist)").
    # The panel has one line per player during a live draft, so compact it to
    # "TD p96 -2.1" and keep the magnitude -- the percentile alone would make a
    # WR at -0.5 look like the RB at -2.1, which is the exact failure the old
    # absolute threshold had.
    b["td_dep"] = b["td_dep"].str.replace(
        r"^p(\d+) \(([-+.\d]+) PPG hist\)$", r"TD p\1 \2", regex=True)

    # Current status is the one thing on this board the model could not have
    # worked out for itself, so it gets its own line in the panel rather than
    # being folded into the stance note. In a live draft it is also the fact
    # most likely to have changed since the board was built.
    b["stat"] = b["status"].where(
        ~b["status"].isin(("rehab", "out", "legal")), b["status"].str.upper())

    # The user's own watchlist. The board's ORDER lost to ADP in the end-to-end
    # backtest, but its conditional availability is the one output validated
    # against real league drafts -- so the panel's most useful view is not
    # "who is best" but "are MY guys going to last". A WATCH filter puts that
    # one question on screen during the draft.
    wl_path = OUT.parent / "data" / "watchlist_private.csv"   # yours, gitignored
    if not wl_path.exists():
        wl_path = OUT.parent / "data" / "watchlist_2026.csv"  # neutral demo
    b["watch"] = 0
    b["wnote"] = ""
    if wl_path.exists():
        wl = pd.read_csv(wl_path, comment="#")
        if not wl.empty:
            wl["k"] = wl["player"].str.strip()
            hit = b["player"].isin(set(wl["k"]))
            b.loc[hit, "watch"] = 1
            nmap = dict(zip(wl["k"], wl["note"].fillna("")))
            pmap = dict(zip(wl["k"], wl["priority"]))
            b.loc[hit, "wnote"] = b.loc[hit, "player"].map(nmap).fillna("")
            b.loc[hit, "watch"] = b.loc[hit, "player"].map(pmap).fillna(1)
            missed = sorted(set(wl["k"]) - set(b["player"]))
            if missed:
                print(f"  !! watchlist entries matching no board player: "
                      f"{', '.join(missed)}")
            print(f"watchlist: {int(hit.sum())}/{len(wl)} players flagged")
    # ---- QB dependency, as a downside annotation on pass catchers ---------
    # 41% of teams lose 3+ starts from their week-1 QB, and it costs a #1
    # target ~1.3 PPG (scripts/qb_dependency.py, 352 within-player comparisons).
    # The projection model cannot see this at all: it reads a receiver's own
    # past PPG, which is an AVERAGE over whoever was throwing. Deliberately a
    # label, not a projection adjustment -- expected cost is only ~0.5 PPG once
    # multiplied by the base rate, and the real information is variance.
    qbc = ROOT / "data" / "qb_context_2026.json"
    b["qbd"] = ""
    if qbc.exists():
        q = json.loads(qbc.read_text())
        alias = {"LA": "LAR", "SD": "LAC", "OAK": "LV", "STL": "LAR"}
        tm = b["team_2026"].replace(alias)
        pc = b["position"].isin(["WR", "TE"])
        lab = tm.map(lambda t: (
            f"QB {q[t]['qb']} · {q[t]['tier']} risk"
            if isinstance(q.get(t), dict) and q[t]["tier"] in
            ("high", "elevated", "unknown") else ""))
        b.loc[pc, "qbd"] = lab[pc].fillna("")
        n = int((b["qbd"] != "").sum())
        print(f"QB dependency: {n} pass catchers flagged "
              f"(41% base rate, −1.3 PPG to a #1 target)")

    # ---- ADP is the spine; the model is a divergence flag ------------------
    # The end-to-end backtest (board_backtest.py) settled this: ordering by
    # `my_rank` LOST to ADP in the top 120 in both 2024 and 2025, and a
    # simulated draft off it lost at 12/12 slots in 2024. What survived was the
    # board's LARGE disagreements -- gaps of 24+ ranks beat their price in both
    # years, while mild 8-23 leans were the worst cohort on the board. So the
    # panel now sorts by market rank and shows the model only where it diverges
    # far enough to have earned an opinion.
    b["adp_rank"] = b["adp"].rank(method="first")
    # gap > 0 means the model likes him MORE than the market does
    b["gap"] = (b["adp_rank"] - b["my_rank"]).round(0)
    b.loc[b["adp"].isna(), "gap"] = float("nan")
    b.loc[b["gap"].abs() < 24, "gap"] = float("nan")     # mild leans are noise
    # ...and the 24+ rule was only ever measured on the TOP 150 BY ADP. Past
    # that the model and the market diverge as a matter of course -- 41 of the
    # 55 players beyond rank 150 clear 24, i.e. 75%, against 18% inside the top
    # 60. A badge that fires on three quarters of a range is decoration, and
    # worse, it would carry the authority of a backtest that never covered it.
    b.loc[b["adp_rank"] > 150, "gap"] = float("nan")
    # unpriced players have no market opinion, so they sort last on the model
    b = b.sort_values(["adp_rank", "my_rank"], na_position="last")

    cols = {"my_rank": "rank", "player": "name", "position": "pos",
            "team_2026": "team", "proj_ppg": "ppg", "vor_pg": "vor",
            "adp": "adp", "adp_sigma": "sig", "adp_shift": "shift",
            "note": "note", "bye": "bye", "off_2026": "off",
            "team_ctx": "ctx", "td_dep": "tdd",
            "stat": "stat", "status_note": "stn",
            "watch": "watch", "wnote": "wnote", "gap": "gap", "qbd": "qbd",
            "rookie_info": "rook", "tier": "tier", "flag": "flag"}
    # one [adp, sig, shift] triple per ready platform; the panel swaps these in
    # when a platform tab is picked, so every availability number follows it
    per_platform, platform_meta = platforms.for_board(b)
    names = b["player"].tolist()
    b = b[list(cols)].rename(columns=cols)
    rows = json.loads(b.to_json(orient="records"))
    eid = platforms.espn_ids()
    for name, r in zip(names, rows):
        r["pl"] = per_platform.get(name, {})
        if platforms.norm(name) in eid:
            r["eid"] = eid[platforms.norm(name)]   # headshot lookup
    return rows, platform_meta


# --------------------------------------------------------------------- markup

MARKUP = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Snakebite</title>
<style>
:root{
  color-scheme:light dark;
  --bg:#edf0ea; --card:#ffffff; --sunk:#e6eae3; --line:#dde2d9;
  --ink:#172019; --dim:#5f6b62; --faint:#929d94;
  --acc:#1d7a4b; --acc-ink:#ffffff; --gold:#f2c14e; --gold-ink:#3d2e05;
  --good:#23a05f; --mid:#e0a21b; --low:#e0564a;
  --good-soft:#dff1e7; --mid-soft:#faf0d6; --low-soft:#f9e2df;
  --hero-a:#0e3b27; --hero-b:#1b6b45; --hero-ink:#ffffff;
  --QB:#d2662d; --RB:#2f9460; --WR:#3f66c8; --TE:#8b55bf;
  --shadow:0 1px 2px rgba(23,32,25,.06),0 6px 18px rgba(23,32,25,.06);
  --round:"SF Pro Rounded",ui-rounded,"Nunito","Quicksand",
          ui-sans-serif,-apple-system,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d130f; --card:#151c17; --sunk:#1b231d; --line:#26302a;
    --ink:#e8efe9; --dim:#9aa89e; --faint:#6b7a70;
    --acc:#4dbb83; --acc-ink:#06130b;
    --good:#4fc386; --mid:#e8b043; --low:#ec7a6e;
    --good-soft:#15291d; --mid-soft:#2b2514; --low-soft:#2e1b19;
    --hero-a:#0a2a1b; --hero-b:#14583a;
    --QB:#e88e5c; --RB:#5cc590; --WR:#82a1ec; --TE:#b88fe3;
    --shadow:0 1px 2px rgba(0,0,0,.35),0 6px 18px rgba(0,0,0,.25);
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.45 var(--round);-webkit-font-smoothing:antialiased}
#wrap{padding:16px 14px 48px;max-width:560px;margin:0 auto}
button{font:inherit;color:inherit;background:var(--card);
  border:1px solid var(--line);border-radius:999px;padding:5px 12px;
  font-size:12px;cursor:pointer;transition:border-color .12s,background .12s}
button:hover{border-color:var(--acc)}
button:focus-visible,input:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.num{font-variant-numeric:tabular-nums}
.dim{color:var(--dim)} .small{font-size:11px;color:var(--dim)}
.neg{color:var(--low)} .pos-n{color:var(--good)} .warn{color:var(--mid)}

/* header + pick tape */
#top{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:12px}
.brand{display:flex;align-items:center;gap:8px;font-weight:800;
  font-size:18px;letter-spacing:-.03em}
.brand .logo{width:26px;height:26px;border-radius:9px;display:grid;
  place-items:center;background:var(--acc);color:var(--acc-ink);font-size:14px}
#turn{text-align:right;font-size:12px;color:var(--dim);line-height:1.25}
#turn b{color:var(--ink)}
#turn .up{background:var(--gold);color:var(--gold-ink);padding:3px 11px;
  border-radius:999px;font-weight:800;font-size:11px;letter-spacing:.04em}
#tape{display:flex;align-items:center;gap:4px;margin:0 2px 14px;
  overflow:hidden}
#tape i{flex:none;width:7px;height:7px;border-radius:50%;background:var(--line)}
#tape i.me{width:11px;height:11px;background:var(--gold);
  box-shadow:0 0 0 3px color-mix(in srgb,var(--gold) 30%,transparent)}
#tape i.now{background:var(--acc)}
#tape span{margin-left:auto;padding-left:8px;font-size:11px;color:var(--faint);
  white-space:nowrap}

#search{width:100%;padding:12px 16px;font:inherit;font-size:14px;
  color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:999px;outline:none;box-shadow:var(--shadow)}
#search:focus{border-color:var(--acc)}
#search::placeholder{color:var(--faint)}
#hints{display:none;font-size:11px;color:var(--dim);margin:8px 8px 0}
kbd{background:var(--sunk);border:1px solid var(--line);border-radius:4px;
  padding:0 5px;font-size:10px;font-family:ui-monospace,monospace}
#meta{display:flex;gap:8px;align-items:center;margin:12px 4px 4px;
  font-size:12px;color:var(--dim)}
#meta .sp{flex:1}
#meta input{width:44px;background:var(--card);color:var(--ink);
  border:1px solid var(--line);border-radius:8px;padding:2px 6px;font-size:12px}
#meta button{padding:3px 11px}
#sync{font-size:11px;color:var(--faint);margin:0 4px 10px}
#log{display:none;max-height:240px;overflow:auto;font-size:10.5px;
  margin:0 0 10px;color:var(--dim);white-space:pre-wrap;word-break:break-all;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--sunk);
  border-radius:12px;padding:10px;user-select:all}

/* chip rows: platform + position filter */
.seg{display:flex;gap:6px;flex-wrap:wrap}
#plat{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;padding-bottom:2px}
#plat::-webkit-scrollbar{display:none}
#plat button{flex:none}
.seg button{background:var(--card);color:var(--dim);font-weight:600}
.seg button:hover{color:var(--ink)}
.seg button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.seg button:disabled{opacity:.35;cursor:not-allowed}
#platnote{font-size:11px;color:var(--faint);margin:7px 4px 14px}
#bar{display:flex;align-items:center;gap:8px;margin:18px 0 10px}
#bar .seg{flex:1}
#bar .reset{border:0;background:none;color:var(--faint);padding:4px 6px}
#bar .reset:hover{color:var(--low)}

/* hero: the one decision that matters right now */
#rec{position:relative;overflow:hidden;color:var(--hero-ink);
  background:radial-gradient(120% 140% at 100% 0%,var(--hero-b),var(--hero-a) 62%);
  border-radius:22px;padding:18px 18px 16px;box-shadow:var(--shadow)}
#rec.thin{background:var(--card);color:var(--ink);border:1px solid var(--line)}
#rec .rlab{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
  font-weight:800;color:var(--gold);margin-bottom:10px}
#rec.thin .rlab{color:var(--faint)}
.rmain{display:flex;align-items:center;gap:14px}
.rmain .hs{width:78px;height:78px;border-width:3px}
.rbody{flex:1;min-width:0}
.rname{font-size:22px;font-weight:800;letter-spacing:-.03em;line-height:1.1}
.rsub{margin-top:6px;display:flex;gap:8px;align-items:center;font-size:12px;
  opacity:.85}
#rec .pos{background:rgba(255,255,255,.16);color:#fff}
#rec.thin .pos.QB{color:var(--QB)} #rec.thin .pos.RB{color:var(--RB)}
#rec.thin .pos.WR{color:var(--WR)} #rec.thin .pos.TE{color:var(--TE)}
#rec.thin .pos{background:var(--sunk)}
.rav{flex:none;text-align:center}
.rav .g{width:72px;height:72px}
.rav .g text{font-size:11.5px}
.rav small{display:block;font-size:10.5px;opacity:.75;margin-top:2px}
.rwhy{margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.16);
  font-size:12.5px;opacity:.92}
#rec.thin .rwhy{border-color:var(--line);color:var(--dim)}
.rnote{margin-top:8px;font-size:12px;color:var(--gold)}
#rec.thin .rnote{color:var(--mid)}
.alt{margin-top:12px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;
  font-size:11px;opacity:.9}
.alt .c{background:rgba(255,255,255,.12);border-radius:999px;padding:3px 10px}
#rec.thin .alt .c{background:var(--sunk)}

/* gauge ring */
.g{display:block}
.g circle{fill:none;stroke-width:5;stroke-linecap:round}
.g .tr{stroke:currentColor;opacity:.14}
.g text{fill:currentColor;font-weight:800;text-anchor:middle;
  dominant-baseline:central;font-family:var(--round)}

/* roster chips */
#ctx{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 12px}
.pill{display:inline-flex;align-items:center;gap:5px;background:var(--card);
  border:1px solid var(--line);border-radius:999px;padding:3px 11px;
  font-size:11.5px;color:var(--dim)}
.pill b{color:var(--ink);font-weight:700}
.need{border-color:var(--mid);background:var(--mid-soft)}
.done{border-color:var(--good);background:var(--good-soft)}
.pill.run{border-style:dashed;background:none}

/* opponents */
#opp{background:var(--card);border:1px solid var(--line);border-radius:18px;
  margin-bottom:12px;overflow:hidden}
#opp .hd{display:flex;justify-content:space-between;padding:11px 16px;
  font-size:12px;font-weight:700;cursor:pointer}
#opp .hd span{color:var(--faint);font-weight:500}
.orow{display:grid;grid-template-columns:34px 1fr;gap:6px 8px;
  padding:10px 16px;border-top:1px solid var(--line);align-items:start}
.orow .pk{font-size:11px;color:var(--faint);padding-top:2px}
.orow .tm{font-weight:700;font-size:12.5px}
.orow .ln{font-size:11px;color:var(--dim);margin-top:2px}
.chip{display:inline-block;font-size:10px;font-weight:700;border-radius:999px;
  padding:1px 7px;margin-left:5px;background:var(--sunk);color:var(--dim);
  vertical-align:1px}
.chip.beginner{background:var(--mid-soft);color:var(--mid)}
.nothing{padding:4px 16px 14px;font-size:12px;color:var(--dim)}

/* headshots: initials underneath, photo on top, ring in position colour */
.hs{position:relative;flex:none;width:46px;height:46px;border-radius:50%;
  overflow:hidden;background:var(--sunk);border:2px solid var(--line);
  display:grid;place-items:end center}
.hs b{position:absolute;inset:0;display:grid;place-items:center;
  font-size:13px;font-weight:800;color:var(--dim)}
.hs img{position:relative;width:100%;height:100%;object-fit:cover;
  object-position:50% 12%;display:block}
.hs.QB{border-color:var(--QB)} .hs.RB{border-color:var(--RB)}
.hs.WR{border-color:var(--WR)} .hs.TE{border-color:var(--TE)}
#rec .hs{background:rgba(255,255,255,.12)} #rec .hs b{color:#fff}

/* board: one card per player */
#rows{display:flex;flex-direction:column;gap:8px}
.row{background:var(--card);border:1px solid var(--line);border-radius:18px;
  cursor:pointer;transition:transform .1s,border-color .12s;overflow:hidden}
.row:hover{border-color:var(--acc)}
.row .main{display:grid;grid-template-columns:46px 1fr 48px;gap:12px;
  align-items:center;padding:10px 14px 10px 10px}
.nm{font-weight:750;font-size:14.5px;letter-spacing:-.015em}
.star{color:var(--gold);margin-right:4px;font-size:12px}
.sub{display:flex;gap:6px;align-items:center;margin-top:3px;font-size:11.5px;
  color:var(--dim);min-width:0}
.sub .sig{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.pos{display:inline-block;border-radius:999px;padding:0 7px;font-size:10px;
  font-weight:800;letter-spacing:.04em;line-height:17px}
.QB{color:var(--QB);background:color-mix(in srgb,var(--QB) 15%,transparent)}
.RB{color:var(--RB);background:color-mix(in srgb,var(--RB) 15%,transparent)}
.WR{color:var(--WR);background:color-mix(in srgb,var(--WR) 15%,transparent)}
.TE{color:var(--TE);background:color-mix(in srgb,var(--TE) 15%,transparent)}
.adp{font-size:11px;color:var(--faint);white-space:nowrap}
.gap{font-weight:800;font-size:10.5px;padding:0 6px;border-radius:999px}
.gap.up{color:var(--good);background:var(--good-soft)}
.gap.dn{color:var(--low);background:var(--low-soft)}
.av{width:48px;height:48px}
.av .g text{font-size:13px}
.tierbreak{margin-top:6px}

/* expanded detail */
.detail{display:none;padding:2px 16px 14px 70px;font-size:12px;color:var(--dim)}
.row.open .detail{display:block}
.row.open{border-color:var(--acc)}
.detail p{margin:0 0 6px}
.detail .tags{display:flex;gap:5px;flex-wrap:wrap;margin:8px 0}
.detail .tags span{background:var(--sunk);border-radius:999px;
  padding:2px 10px;font-size:11px}
.detail .acts{display:flex;gap:6px;margin-top:10px}
.detail .acts .go{background:var(--acc);color:var(--acc-ink);
  border-color:var(--acc);font-weight:700}
.empty{padding:26px;text-align:center;color:var(--faint)}
@media (max-width:420px){
  #wrap{padding:12px 10px 48px}
  .rname{font-size:19px} .rmain .hs{width:62px;height:62px}
  .rav .g{width:60px;height:60px}
  .row .main{gap:10px;padding:9px 10px 9px 8px}
  .detail{padding-left:62px}
}
</style>

<div id="wrap">
  <div id="top">
    <div class="brand"><span class="logo">S</span>snakebite</div>
    <div id="turn"></div>
  </div>
  <div id="tape"></div>
  <input id="search" placeholder="Search a player — Enter marks gone, Shift+Enter marks yours"
         autocomplete="off" spellcheck="false">
  <div id="hints">
    <kbd>Enter</kbd> mark top match gone · <kbd>Shift</kbd>+<kbd>Enter</kbd>
    your pick · <kbd>Esc</kbd> clear · <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo ·
    click a row for detail
  </div>
  <div id="meta">
    <span>Pick <b id="pick" class="num">1</b></span>
    <span>· Slot <input type="number" id="slot" min="1" max="12" value="3"></span>
    <span id="nextpk"></span>
    <span class="sp"></span>
    <button id="undo">Undo</button>
    <button id="more" title="shortcuts and pick log">···</button>
    <button id="diag" title="run ESPN sync diagnostics">Diag</button>
  </div>
  <div id="sync">manual mode</div>
  <div id="log"></div>

  <div id="plat" class="seg" title="whose ADP drives every availability number"></div>
  <div id="platnote"></div>
  <div id="rec"></div>
  <div id="ctx"></div>
  <div id="opp"></div>

  <div id="bar">
    <div class="seg">
      <button data-f="ALL" class="on">All</button>
      <button data-f="QB">QB</button><button data-f="RB">RB</button>
      <button data-f="WR">WR</button><button data-f="TE">TE</button>
      <button data-f="WATCH" title="only my watchlist">★ Mine</button>
    </div>
    <button id="reset" class="reset">Reset</button>
  </div>
  <div id="rows"></div>
</div>
"""

# ------------------------------------------------------------------------ app

APP_JS = r"""
const BOARD = __DATA__;
/* teamId -> {name, lean}. Baked in at build time from the league's own draft
   history, so the panel needs no extra request and works if the roster
   endpoint ever changes shape. `lean` is DESCRIPTION, not prediction --
   see renderOpponents. */
const OPPONENTS = __OPPONENTS__;
const STATE_PORT = __PORT__;
/* ADP platforms: [{id,label,ready,n,calibrated,src}], tab order. Each BOARD row
   carries `pl[id] = [adp, sig, shift]` for the platforms that price him. */
const PLATFORMS = __PLATFORMS__;
/* ROUNDS must match the actual roster size -- ESPN's own header reads
   "RND 11 OF 15" and the roster panel reads "10/15 Players". It was 16, which
   made myPicks() emit a 16th pick beyond the end of the draft (slot 4 -> pick
   189 in a 180-pick draft), so late on, horizon() compared the board against a
   pick that does not exist. */
const TEAMS = __TEAMS__, ROUNDS = __ROUNDS__;
const LINEUP = __LINEUP__;   // + FLEX (RB/WR/TE)
const FLEX = __FLEX__;
const KEY = "draftday2026";
/* Bump whenever a fix changes what counts as a valid pick. Saved state from an
   older schema is DISCARDED rather than migrated, because the reason to bump is
   that the old ingestion was wrong -- so its output is wrong too.

   This exists because of a real incident: v1.4 logged 154 phantom picks off the
   mock-draft lobby, v1.5 gated ingestion to fix it, and the bad picks survived
   the upgrade anyway. Reloading the extension and the page both leave
   localStorage intact, so the panel opened on a poisoned board with no clue
   why, and the only cure was a `reset` button nobody knew to press. */
/* 3: picks now carry `team` and `at`. Old records have neither, so the
   opponent view would silently show empty rosters for a draft in progress --
   worse than showing nothing, because it looks like information. */
const SCHEMA = 3;

let state = {picks:[], slot:3, filter:"ALL", apiPick:null, schema:SCHEMA,
             seats:{}};
const byName = Object.fromEntries(BOARD.map(p => [p.name, p]));

/* ---------- who sits where ----------------------------------------------
   A snake draft's seat for pick `o` is fixed by arithmetic, so one observed
   pick teaches us that team's seat for the whole draft. Round 1 fills the
   table immediately; nothing needs fetching. */
function seatOfPick(o){
  const r = Math.floor((o - 1) / TEAMS), i = (o - 1) % TEAMS;
  return (r % 2 === 0) ? i + 1 : TEAMS - i;
}
function learnSeat(team, at){
  const s = seatOfPick(at);
  if (state.seats[s] !== team){ state.seats[s] = team; }
}
const teamAtPick = (o) => state.seats[seatOfPick(o)];

/* opponent roster, built purely from the pick stream we already ingest */
function rosterOf(team){
  const c = {QB:0, RB:0, WR:0, TE:0, n:0};
  state.picks.forEach(p => {
    if (p.team !== team) return;
    c.n++;
    const pl = byName[p.name];
    if (pl && c[pl.pos] !== undefined) c[pl.pos]++;
  });
  return c;
}
/* what they still owe a starting lineup -- the part of "will he last?" that
   history cannot tell you, because it is about tonight, not last year */
function needsOf(c){
  const out = [];
  for (const [pos, n] of Object.entries(LINEUP)){
    if (c[pos] < n) out.push(pos + " " + c[pos] + "/" + n);
  }
  const flex = Math.max(0, c.RB - LINEUP.RB) + Math.max(0, c.WR - LINEUP.WR)
             + Math.max(0, c.TE - LINEUP.TE);
  if (flex < FLEX) out.push("FLX " + flex + "/" + FLEX);
  return out;
}

/* ---------- persistence -------------------------------------------------- */
/* Mirrors to a local listener if one is running. That file is the latency fix
   for agent use: reading one small JSON blob costs a single round trip, where
   driving the browser costs a snapshot + parse on every step. Optional -- a
   dead port is swallowed. */
let mirrorOff = false;   // stop after the first refusal: no listener, no noise
function mirror(){
  if (mirrorOff) return;
  try{
    fetch("http://127.0.0.1:" + STATE_PORT + "/state", {
      method:"POST", headers:{"Content-Type":"application/json"},
      /* curPick(), not picks.length + 1 -- otherwise draft_state.json (and so
         draft_advise.py) reports a different pick number than the panel. */
      body: JSON.stringify({pick: curPick(), slot: state.slot,
                            picks: state.picks, ts: Date.now()})
    }).catch(() => { mirrorOff = true; });
  }catch(e){ mirrorOff = true; }
}
function save(){
  try{ localStorage.setItem(KEY, JSON.stringify(state)); }catch(e){}
  mirror();
}
function load(){
  try{
    const s = JSON.parse(localStorage.getItem(KEY));
    if (!s || !Array.isArray(s.picks)) return;
    if (s.schema !== SCHEMA){
      localStorage.removeItem(KEY);
      STALE = s.picks.length;      // report it; silent data loss is its own bug
      return;
    }
    state = Object.assign(state, s);
  }catch(e){}
}
let STALE = 0;

/* ---------- draft maths -------------------------------------------------- */
function myPicks(slot){
  const out = [];
  for (let r = 1; r <= ROUNDS; r++)
    out.push(r % 2 ? (r-1)*TEAMS + slot : r*TEAMS - slot + 1);
  return out;
}
function erf(x){
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1/(1 + 0.3275911*x);
  const y = 1 - ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t
                  - 0.284496736)*t + 0.254829592)*t*Math.exp(-x*x);
  return s*y;
}
const cdf = (x, m, sd) => 0.5*(1 + erf((x - m)/(sd*Math.SQRT2)));

/* P(still there at `at`), GIVEN he is on the board at `now`.

   Two regimes, because one model does not cover both:

   NORMAL -- he is still inside his ADP band, so condition the preseason prior
   on having survived: P(at) / P(now).

   FALLEN -- he has outlived his ADP by more than a standard deviation. The
   prior is now actively wrong (it says ~0% while he sits right there), so
   conditioning on it divides by almost nothing and returns noise. Re-price
   him instead: if the room has let him slide this far, the honest projection
   is that the slide continues at the same rate, so centre a fresh curve at
   `now + (now - adp)` and widen sigma, because we genuinely know less about
   him than we did. Flagged `est` so the UI can show it is a re-estimate. */
function availAt(p, at, now){
  if (p.adp == null || p.sig == null) return null;
  /* mu is the league-calibrated expectation, not raw market ADP: sig
     already carries the rank-dependent multiplier and shift carries the
     measured chalk drift (league_tendencies.json, validated by
     replay_brier.py). shift is 0 when no league history exists. */
  const mu = p.adp + (p.shift || 0);
  const fallen = now > mu + p.sig;
  if (fallen){
    const adp2 = now + (now - mu);               // project the slide forward
    const sig2 = Math.max(3, p.sig * 1.5);       // and admit the added doubt
    return {v: Math.max(0, Math.min(1, 1 - cdf(at, adp2, sig2))), est:true};
  }
  const sNow = 1 - cdf(now, mu, p.sig);
  const sAt  = 1 - cdf(at,  mu, p.sig);
  if (sNow <= 0.02){
    const sig2 = Math.max(3, p.sig * 1.5);
    return {v: Math.max(0, Math.min(1, 1 - cdf(at, now, sig2))), est:true};
  }
  return {v: Math.min(1, sAt / sNow)};
}
/* single number for the recommendation maths */
function pAvail(p, at, now){
  const a = availAt(p, at, now);
  if (a == null) return 0.55;          // no ADP line at all -- market ignores him
  return a.v;
}

const goneSet = () => new Set(state.picks.map(p => p.name));
/* ESPN's number when we have it, our own count when we do not. Counting our
   own matched picks undercounts by one for every drafted player who is not on
   the board (kickers, defences, deep bench), and that error only ever grows. */
const curPick = () => state.apiPick || (state.picks.length + 1);
const nextOfMine = () => myPicks(state.slot).find(p => p >= curPick());

/* The pick every "will he still be there?" question is really about.
   When you are ON THE CLOCK, your next pick equals the current pick, and
   comparing the board against itself makes every cost-of-waiting zero -- which
   is exactly backwards, because being on the clock is when the question
   matters most. On the clock, the horizon is your NEXT pick (the far side of
   the turn); otherwise it is the pick you are waiting for. */
function horizon(){
  const mine = myPicks(state.slot), now = curPick();
  const nxt = mine.find(p => p >= now);
  return nxt === now ? mine.find(p => p > now) : nxt;
}

function rosterCounts(){
  const c = {QB:0, RB:0, WR:0, TE:0};
  state.picks.filter(p => p.mine).forEach(p => {
    const pl = byName[p.name]; if (pl) c[pl.pos]++;
  });
  return c;
}

/* ---------- the recommendation ------------------------------------------
   Value over next available. For each position: the best man on the board now
   versus the value you can EXPECT to still be there at your next pick, given
   who has actually gone. Walk that position in rank order accumulating
   VOR * P(he lasts) * P(nobody better lasts) -- that expectation is what
   waiting actually buys you. The gap is the cost of waiting.

   Then weight by need. A third RB is worth less than a first WR no matter what
   the raw numbers say, because you cannot start him.                        */
function expectedNext(pos, live, nxt, now){
  let acc = 0, noneBetter = 1;
  for (const p of live.filter(x => x.pos === pos)){
    const pa = pAvail(p, nxt, now);
    acc += p.vor * pa * noneBetter;
    noneBetter *= (1 - pa);
    if (noneBetter < 0.01) break;
  }
  return acc;
}
function needWeight(pos, c){
  if (c[pos] < LINEUP[pos]) return 1.0;              // starter still empty
  const flexHave = ["RB","WR","TE"].reduce(
    (a,k) => a + Math.max(0, c[k] - LINEUP[k]), 0);
  if (pos !== "QB" && flexHave < FLEX) return 0.75;  // flex-eligible
  return 0.3;                                        // depth only
}
function recommend(live, nxt, now){
  const c = rosterCounts();
  const rows = [];
  for (const pos of ["RB","WR","TE","QB"]){
    const pool = live.filter(p => p.pos === pos);
    if (!pool.length) continue;
    const best = pool[0];
    const exp = nxt ? expectedNext(pos, live, nxt, now) : best.vor;
    const w = needWeight(pos, c);
    rows.push({pos, best, cost: best.vor - exp, exp, w,
               score: (best.vor - exp) * w});
  }
  rows.sort((a,b) => b.score - a.score);
  return rows;
}

/* ---------- name matching ------------------------------------------------ */
/* Mirrors norm_name() in phase3_draft_board.py. "D.J. Moore"/"DJ Moore" and
   "Kenneth Walker III"/"Kenneth Walker" must land on the same key. */
function normName(s){
  return (s || "").toLowerCase()
    .replace(/[^a-z\s]/g, "")
    .replace(/\b(jr|sr|ii|iii|iv|v)$/g, "")
    .replace(/\s+/g, " ").trim();
}
const normIndex = {};
BOARD.forEach(p => { normIndex[normName(p.name)] = p.name; });

/* Exact normalised hit, then last-name + first-initial. Ambiguity returns null
   rather than guessing: "A. Brown" is both A.J. Brown and Amon-Ra St. Brown,
   and picking one on draft night would be worse than asking. */
function resolveName(raw){
  const n = normName(raw);
  if (!n) return null;
  if (normIndex[n]) return normIndex[n];
  const parts = n.split(" ");
  if (parts.length < 2) return null;
  const last = parts[parts.length - 1], first = parts[0][0];
  const hits = Object.keys(normIndex).filter(k => {
    const kp = k.split(" ");
    return kp[kp.length - 1] === last && kp[0][0] === first;
  });
  return hits.length === 1 ? normIndex[hits[0]] : null;
}

/* ---------- render ------------------------------------------------------- */
/* ---------- the opponent view -------------------------------------------
   The question this answers is the one the availability percentage cannot:
   *who* actually picks between now and your next turn, and what does each of
   them still need. The percentage treats the eleven managers in between as an
   anonymous count; this names them.

   Every input is free -- teamId rides on each ESPN pick, seats are arithmetic
   from the snake, and the lean tags were baked in at build time from the
   league's own draft history.

   Deliberately NOT wired into the availability maths: on a replay of this
   league's 2023-25 drafts, weighting the probabilities by roster need and
   manager lean scored WORSE than ignoring them. Three seasons is enough to
   describe a drafter, not to out-predict the market on a specific player.
   So this informs you; it does not move a number. */
let OPP_OPEN = false;   // collapsed: the board is what you are scanning
const esc = s => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");
/* one colour per band, used for the bar and the number alike */
const availCol = v => v >= .7 ? "var(--good)" : v >= .35 ? "var(--mid)"
                                                          : "var(--low)";
/* ring gauge: the availability number as a share of a circle */
function gauge(v, label, onDark){
  const R = 20, C = 2 * Math.PI * R;
  /* on the dark hero the ring and number switch to brighter tones of the same
     three bands so the colour still means good / borderline / gone */
  const col = onDark ? (v >= .7 ? "#7fe3b0" : v >= .35 ? "#ffd166" : "#ff8f80")
                     : availCol(v);
  const ink = onDark ? "#ffffff" : col;
  return `<svg class="g" viewBox="0 0 50 50" style="color:${ink}">`
    + `<circle class="tr" cx="25" cy="25" r="${R}"/>`
    + `<circle cx="25" cy="25" r="${R}" stroke="${col}" `
    + `stroke-dasharray="${(C * v).toFixed(1)} ${C.toFixed(1)}" `
    + `transform="rotate(-90 25 25)"/>`
    + `<text x="25" y="26">${label}</text></svg>`;
}
const initials = n => n.replace(/\b(jr|sr|ii|iii|iv)\.?$/i, "").trim()
  .split(/\s+/).map(w => w[0]).slice(0, 2).join("").toUpperCase();
/* Photo from ESPN's image CDN, resized server-side (~7 KB). Initials sit
   underneath and stay if there is no id or the request fails -- offline, or a
   rookie ESPN has no photo for, still reads as a person, not a broken image.
   Built in JS rather than with an onerror attribute: MV3 extension pages block
   inline handlers. */
function headshot(p, size){
  const d = document.createElement("div");
  d.className = "hs " + p.pos;
  d.innerHTML = `<b>${esc(initials(p.name))}</b>`;
  if (p.eid && typeof Image !== "undefined"){
    const img = new Image();
    img.alt = "";
    img.onerror = () => img.remove();
    img.src = "https://a.espncdn.com/combiner/i?img=/i/headshots/nfl/players/full/"
      + p.eid + ".png&w=" + size * 2 + "&h=" + Math.round(size * 1.45 );
    d.appendChild(img);
  }
  return d;
}
let OPEN_ROW = null;               // the player whose detail is expanded

function renderOpponents(now, nxt){
  const el = document.getElementById("opp");
  if (!el) return;
  const known = state.picks.filter(p => p.team != null).length;
  if (!known){
    el.innerHTML = `<div class="hd">Opponents<span>waiting for the feed</span></div>`
      + `<div class="nothing">Manual picks carry no team, so rosters cannot be `
      + `attributed. Connect the ESPN feed to see who picks before you.</div>`;
    return;
  }
  const upto = nxt || (now + TEAMS);
  const between = [];
  for (let o = now; o < upto; o++){
    const t = teamAtPick(o);
    if (t == null || t === MY_TEAM) continue;
    const hit = between.find(b => b.team === t);
    if (hit) hit.picks.push(o); else between.push({team:t, picks:[o]});
  }
  const rows = between.map(b => {
    const c = rosterOf(b.team);
    const info = OPPONENTS[b.team] || {};
    const need = needsOf(c);
    /* An "assumed" lean is reasoning from a profile, not from picks -- it must
       never look like the measured ones sitting next to it. */
    const lean = info.lean
      ? esc(info.lean) + (info.assumed ? " (assumed)" : "") : "";
    const exp = (info.exp && info.exp !== "vet")
      ? `<span class="chip ${esc(info.exp)}">${esc(info.exp)}</span>` : "";
    const needTxt = need.length ? "needs " + need.join(", ") : "starters full";
    return `<div class="orow" title="${esc(info.note || "")}">`
      + `<div class="pk num">${b.picks.join(",")}</div><div>`
      + `<div class="tm">${esc(info.name || ("Team " + b.team))}${exp}</div>`
      + `<div class="ln">${c.n} picked · ${esc(needTxt)}</div>`
      + (lean ? `<div class="ln">${lean}</div>` : "")
      + `</div></div>`;
  }).join("");
  el.innerHTML = `<div class="hd" id="opphd">Before your pick${nxt ? " " + nxt : ""}`
    + `<span>${between.length} manager${between.length === 1 ? "" : "s"} `
    + `${OPP_OPEN ? "▾" : "▸"}</span></div>`
    + (OPP_OPEN
        ? (rows || `<div class="nothing">Nobody picks before you.</div>`) : "");
  const hd = document.getElementById("opphd");
  if (hd) hd.onclick = () => { OPP_OPEN = !OPP_OPEN; render(); };
}

/* The one line a row earns on the board: the single most decision-relevant
   fact, picked by priority. Everything else waits in the expanded detail so a
   row never shows more than a glance can hold. */
function headline(p){
  if (p.note) return {t: (p.flag === "neg" ? "! " : "") + p.note,
    c: p.flag === "neg" ? "neg" : p.flag === "pos" ? "pos-n" : "warn"};
  if (p.stat) return {t: p.stat.toLowerCase()
    + (p.stn ? " — " + p.stn.split(/(?<=[.;])\s/)[0] : ""),
    c: ["REHAB","OUT","LEGAL"].includes(p.stat) ? "warn" : "dim"};
  if (p.watch && p.wnote) return {t: p.wnote, c: "dim"};
  if (p.qbd && /high|unknown/.test(p.qbd)) return {t: p.qbd, c: "warn"};
  return null;
}

function detailOf(p){
  const parts = [];
  if (p.note) parts.push(`<p class="${p.flag==="neg"?"neg":p.flag==="pos"?"pos-n":"warn"}">${esc(p.note)}</p>`);
  if (p.stat) parts.push(`<p><b>${esc(p.stat)}</b>${p.stn ? " — " + esc(p.stn) : ""}</p>`);
  if (p.watch && p.wnote) parts.push(`<p>${esc(p.wnote)}</p>`);
  if (p.ctx) parts.push(`<p>${esc(p.ctx)}</p>`);
  if (p.qbd) parts.push(`<p title="41% of team-seasons lose 3+ starts from their week-1 QB; that costs a true #1 target about 1.3 PPG. A variance label, not a projection adjustment.">${esc(p.qbd)}</p>`);
  const tags = [];
  if (p.vor != null) tags.push(`VOR ${p.vor.toFixed(1)}`);
  if (p.ppg != null) tags.push(`${p.ppg.toFixed(1)} PPG proj`);
  if (p.bye) tags.push(`bye ${p.bye}`);
  if (p.off) tags.push(esc(p.off));
  if (p.tdd) tags.push(esc(p.tdd));
  if (p.gap) tags.push(`model ${p.gap > 0 ? "likes" : "fades"} him ${Math.abs(p.gap)} ranks vs market`);
  if (tags.length) parts.push(`<div class="tags">${tags.map(t => `<span>${t}</span>`).join("")}</div>`);
  parts.push(`<div class="acts"><button class="go" data-a="gone">Mark gone</button>`
    + `<button data-a="mine">Mine</button></div>`);
  return parts.join("");
}

function render(){
  const gone = goneSet(), now = curPick();
  const mine = nextOfMine(), nxt = horizon();
  const onClock = mine === now;
  const q = document.getElementById("search").value.trim().toLowerCase();
  const live = BOARD.filter(p => !gone.has(p.name));

  document.getElementById("pick").textContent = now;
  document.getElementById("nextpk").textContent = "";
  document.getElementById("turn").innerHTML = !mine ? `<b>Draft complete</b>`
    : onClock
      ? `<span class="up">YOU'RE UP</span>`
        + (nxt ? `<div class="small" style="margin-top:4px">back at ${nxt}</div>` : "")
      : `Your pick <b class="num">${mine}</b>`
        + `<div class="small" style="margin-top:2px">${mine - now} away</div>`;
  /* pick tape: one dot per pick until you are back on the clock */
  const tape = document.getElementById("tape");
  const endAt = onClock ? (nxt || now) : (mine || now);
  let dots = "";
  for (let o = now; o <= endAt && o < now + 40; o++)
    dots += `<i class="${o === now ? "now" : ""} ${o === endAt && mine ? "me" : ""}"></i>`;
  tape.innerHTML = (mine ? dots : "")
    + (mine ? `<span>${onClock ? "your pick" : (mine - now) + " until you"}</span>` : "");

  /* recommendation */
  const rec = document.getElementById("rec");
  const rows = recommend(live, nxt, now);
  if (!rows.length){ rec.innerHTML = ""; rec.style.display = "none"; }
  else {
    rec.style.display = "";
    const t = rows[0];
    const a = nxt ? availAt(t.best, nxt, now) : null;
    const thin = t.cost < 0.35;
    rec.className = thin ? "thin" : "";
    const rnote = t.best.note
      ? `<div class="rnote ${t.best.flag==="neg"?"neg":t.best.flag==="pos"?"pos-n":"warn"}">${
          t.best.flag==="neg"?"! ":""}${esc(t.best.note)}</div>` : "";
    const av = a == null ? "" : `<div class="rav">${gauge(a.v, (a.est ? "~" : "")
      + Math.round(a.v * 100) + "%", !thin)}<small>to last to ${nxt}${a.est ? " · sliding" : ""}</small></div>`;
    rec.innerHTML = `
      <div class="rlab">${thin ? "No urgency · best available" : "Take now"}</div>
      <div class="rmain">
        <div class="hsw"></div>
        <div class="rbody">
          <div class="rname">${esc(t.best.name)}</div>
          <div class="rsub"><span class="pos ${t.best.pos}">${t.best.pos}</span>${esc(t.best.team)}</div>
        </div>
        ${av}
      </div>
      <div class="rwhy">${
        thin
        ? `${t.pos} holds up — expect about <b>${t.exp.toFixed(1)} VOR</b> still there at ${nxt||"your next pick"}.`
        : `Waiting costs <b>${t.cost.toFixed(1)} VOR</b> at ${t.pos}: ${
            t.best.vor.toFixed(1)} now vs about ${t.exp.toFixed(1)} at ${nxt}.`}</div>
      ${rnote}
      ${rows.length > 1 ? `<div class="alt">Then ${rows.slice(1,4).map(r =>
        `<span class="c"><b>${r.pos}</b> ${esc(r.best.name.split(" ").slice(-1)[0])} −${
          r.cost.toFixed(1)}</span>`).join("")}</div>` : ""}`;
    rec.querySelector(".hsw").replaceWith(headshot(t.best, 78));
  }

  /* context: needs + what the room has been doing */
  const c = rosterCounts();
  let pills = "";
  for (const [pos, n] of Object.entries(LINEUP))
    pills += `<span class="pill ${c[pos]>=n?"done":"need"}">${pos} <b class="num">${c[pos]}/${n}</b></span>`;
  const flexHave = ["RB","WR","TE"].reduce(
    (a,k) => a + Math.max(0, c[k] - LINEUP[k]), 0);
  pills += `<span class="pill ${flexHave>=FLEX?"done":"need"}">Flex <b class="num">${
    Math.min(flexHave,FLEX)}/${FLEX}</b></span>`;
  const recent = state.picks.slice(-12);
  const tally = {};
  recent.forEach(p => { const pl = byName[p.name];
                        if (pl) tally[pl.pos] = (tally[pl.pos]||0)+1; });
  const runTxt = Object.entries(tally).sort((a,b)=>b[1]-a[1])
    .map(([k,v]) => `${k} ${v}`).join(" · ");
  /* Counts, NOT momentum. Over this league's 391 historical picks a run does
     not predict more of that position: 2-of-last-3 RB is followed by an RB at
     lift 0.92, 4-of-last-5 WR at 0.90 -- both inside noise of 1.0 and if
     anything negative. A WR run looks like a signal only because WR is 42% of
     every skill pick here. A pill that implies "get ahead of the run" is the
     tool talking you into a reach. */
  if (runTxt) pills += `<span class="pill run" title="counts only -- runs have not
predicted more of that position in this league (lift 0.90-1.08 over 391 picks)"
>last 12 · ${runTxt}</span>`;
  document.getElementById("ctx").innerHTML = pills;
  renderOpponents(now, nxt);

  /* board */
  let list = live;
  if (state.filter === "WATCH") list = list.filter(p => p.watch);
  else if (state.filter !== "ALL") list = list.filter(p => p.pos === state.filter);
  if (q) list = list.filter(p => p.name.toLowerCase().includes(q));
  list = list.slice(0, 40);

  const body = document.getElementById("rows");
  body.innerHTML = "";
  if (!list.length)
    body.innerHTML = `<div class="empty">No players match.</div>`;
  let lastTier = null, lastPos = null;
  for (const p of list){
    const row = document.createElement("div");
    row.className = "row" + (OPEN_ROW === p.name ? " open" : "");
    if (state.filter !== "ALL" && lastTier !== null
        && p.tier !== lastTier && p.pos === lastPos) row.className += " tierbreak";
    lastTier = p.tier; lastPos = p.pos;
    const a = nxt ? availAt(p, nxt, now) : null;
    const h = headline(p);
    row.innerHTML = `
      <div class="main">
        <div class="hsw"></div>
        <div style="min-width:0">
          <div><span class="nm">${p.watch ? '<span class="star">★</span>' : ""}${esc(p.name)}</span></div>
          <div class="sub"><span class="pos ${p.pos}">${p.pos}</span>
            <span>${esc(p.team)}</span>
            ${p.gap ? `<span class="gap ${p.gap>0?"up":"dn"}"
              title="model disagrees with the market by ${Math.abs(p.gap)} ranks -- only gaps of 24+ beat their price in the backtest">${
              p.gap>0?"+":""}${p.gap}</span>`
              : (p.adp ? `<span class="adp">ADP ${Math.round(p.adp)}</span>` : "")}
          </div>
          ${h ? `<div class="sub"><span class="sig ${h.c}">${esc(h.t)}</span></div>` : ""}
        </div>
        <div class="av" title="${a && a.est ? "re-estimated: he has already fallen past his ADP" : "chance he is still there at your next pick"}">
          ${a == null ? `<span class="dim">–</span>` : gauge(a.v, (a.est ? "~" : "")
            + Math.round(a.v * 100))}
        </div>
      </div>
      <div class="detail">${detailOf(p)}</div>`;
    row.querySelector(".hsw").replaceWith(headshot(p, 46));
    row.querySelector(".main").onclick = () => {
      OPEN_ROW = OPEN_ROW === p.name ? null : p.name;
      render();
    };
    row.querySelectorAll(".acts button").forEach(btn => {
      btn.onclick = e => {
        e.stopPropagation();
        OPEN_ROW = null;
        take(p.name, btn.dataset.a === "mine");
      };
    });
    body.appendChild(row);
  }

  document.getElementById("log").innerHTML = state.picks.slice().reverse()
    .map((p,i) => `${state.picks.length-i}. ${esc(p.name)}${p.mine?" ← me":""}`)
    .join("<br>");
}

/* ---------- actions ------------------------------------------------------ */
/* A manual take has to advance the pick number too. Once the API has set
   `apiPick`, `curPick()` stops looking at `state.picks.length` -- so if the
   feed then dies and you fall back to typing names, the counter would freeze
   at the last number ESPN gave us while the draft kept going. The next
   successful poll overwrites this with the truth; until then it tracks. */
function bumpPick(d){
  if (typeof state.apiPick === "number")
    state.apiPick = Math.max(1, state.apiPick + d);
}
function take(name, mine, auto, team, at){
  if (!name || goneSet().has(name)) return false;
  /* `team` is ESPN's teamId and `at` the overall pick number. Both are free
     in the payload and both are what the opponent view is built from; a
     manual take has neither, which is why that view degrades to "known
     picks only" rather than pretending. */
  state.picks.push({name, mine, auto: !!auto,
                    team: (team == null ? null : team),
                    at: (at == null ? null : at)});
  if (team != null && at != null) learnSeat(team, at);
  if (!auto) bumpPick(+1);            // API picks carry their own numbering
  document.getElementById("search").value = "";
  save(); render();
  return true;
}
function undo(){
  const p = state.picks.pop();
  if (p && !p.auto) bumpPick(-1);
  save(); render();
}
function takeByName(raw, mine, auto, team, at){
  const name = resolveName(raw);
  if (!name) return {ok:false, reason:"no match", raw};
  if (goneSet().has(name)) return {ok:false, reason:"already gone", name};
  take(name, mine, auto, team, at);
  return {ok:true, name};
}
function setSync(msg, cls){
  const el = document.getElementById("sync");
  if (!el) return;
  el.textContent = msg;
  el.className = "small " + (cls || "dim");
}
function showDiag(txt){
  const el = document.getElementById("log");
  el.style.display = "block";
  el.textContent = txt;
  window.scrollTo(0, 0);
}
/* ESPN's authoritative pick number, refreshed every successful poll. A manual
   take between polls is corrected within one poll, which is what we want --
   the feed is the truth and our count is the guess. */
function setPickNo(n){
  n = Number(n);
  if (!Number.isFinite(n) || n < 1) return;
  if (n === state.apiPick) return;
  state.apiPick = n;
  save(); render();
}
/* your own teamId, so the opponent view can leave you out of it */
let MY_TEAM = null;
function setMyTeam(t){
  if (t == null || MY_TEAM === t) return;
  MY_TEAM = t;
  render();
}
/* the draft payload knows your seat, so it can set it rather than you */
function setSlot(n){
  n = Number(n);
  if (!n || n < 1 || n > TEAMS || state.slot === n) return;
  state.slot = n;
  const el = document.getElementById("slot");
  if (el) el.value = n;
  save(); render();
}
window.DRAFTAPP = {take, takeByName, resolveName, undo, setSync, showDiag,
                   setSlot, setPickNo, setMyTeam, setPlatform,
                   rosterOf, needsOf, teamAtPick,
                   names: () => BOARD.map(p => p.name),
                   state: () => state};

/* ---------- wiring ------------------------------------------------------- */
const search = document.getElementById("search");
search.addEventListener("input", render);
search.addEventListener("keydown", e => {
  if (e.key === "Escape"){ search.value = ""; render(); return; }
  if (e.key === "Enter"){
    const first = document.querySelector("#rows .row .nm");
    if (first) take(first.textContent.replace(/^★/, "").trim(), e.shiftKey);
  }
});
document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z"){
    e.preventDefault(); undo();
  } else if (document.activeElement !== search && e.key.length === 1){
    search.focus();
  }
});
document.querySelectorAll("#bar button[data-f]").forEach(b => {
  b.onclick = () => {
    state.filter = b.dataset.f;
    document.querySelectorAll("#bar button[data-f]")
      .forEach(x => x.classList.toggle("on", x === b));
    save(); render(); search.focus();
  };
});
document.getElementById("undo").onclick = () => { undo(); search.focus(); };
document.getElementById("more").onclick = () => {
  for (const id of ["hints","log"]){
    const el = document.getElementById(id);
    el.style.display = el.style.display === "block" ? "none" : "block";
  }
};
document.getElementById("diag").onclick = () => {
  const el = document.getElementById("log");
  el.style.display = "block";
  el.textContent = "running diagnostics…";
  window.scrollTo(0, 0);
  try{ parent.postMessage({__draft:1, type:"probe"}, "*"); }catch(e){}
  setTimeout(() => {
    if (el.textContent === "running diagnostics…")
      el.textContent = "No response. This only works inside the ESPN draft "
        + "page with the extension loaded — not in the standalone file.";
  }, 6000);
};
document.getElementById("reset").onclick = () => {
  if (confirm("Clear every pick and start over?")){
    state.picks = []; state.apiPick = null; save(); render();
  }
};
document.getElementById("slot").onchange = e => {
  state.slot = Math.min(TEAMS, Math.max(1, +e.target.value || 3));
  save(); render();
};

/* ---------- platform tabs -------------------------------------------------
   Each platform publishes its own market, and they disagree (ESPN drafters take
   TE/QB early and let mid-round RB/WR slide). The tab picks whose ADP prices
   every player: swap adp/sig/shift from the baked triple, re-sort, redraw.
   Unpriced players get adp = null, which the panel already renders as "-". */
function setPlatform(id){
  const meta = PLATFORMS.find(x => x.id === id);
  if (!meta || !meta.ready) return;
  state.platform = id;
  for (const p of BOARD){
    const v = p.pl && p.pl[id];
    p.adp = v ? v[0] : null;
    p.sig = v ? v[1] : null;
    p.shift = v ? v[2] : 0;
  }
  BOARD.sort((a, b) => (a.adp == null) - (b.adp == null)
                       || (a.adp - b.adp) || (a.rank - b.rank));
  renderPlatforms();
  save(); render();
}
function renderPlatforms(){
  const el = document.getElementById("plat");
  el.innerHTML = "";
  for (const m of PLATFORMS){
    const b = document.createElement("button");
    b.textContent = m.label;
    b.className = m.id === state.platform ? "on" : "";
    b.disabled = !m.ready;
    b.title = m.ready ? `${m.src} · ${m.n} players on the board`
                      : `${m.src} (data/adp_import/${m.id}.csv)`;
    b.onclick = () => setPlatform(m.id);
    el.appendChild(b);
  }
  const cur = PLATFORMS.find(x => x.id === state.platform);
  document.getElementById("platnote").textContent = !cur ? "" :
    cur.calibrated
      ? `${cur.src} · spread calibrated on league history`
      : `${cur.src} · spread borrowed from consensus history — approximate`;
}

load();
if (!PLATFORMS.some(m => m.id === state.platform && m.ready))
  /* the extension panel is an iframe, so the page it sits on is the ancestor */
  state.platform = (typeof location !== "undefined"
    && ((location.ancestorOrigins && location.ancestorOrigins[0])
        || location.hostname || "").includes("espn.com")) ? "espn" : "ffc";
if (STALE)
  setSync(`discarded ${STALE} picks saved by an older version — starting clean`,
          "warn");
document.getElementById("slot").value = state.slot;
document.getElementById("slot").max = TEAMS;
document.querySelectorAll("#bar button[data-f]").forEach(b =>
  b.classList.toggle("on", b.dataset.f === state.filter));
setPlatform(state.platform);   // also renders
search.focus();
"""

BRIDGE_JS = r"""
/* The panel runs in an iframe; the content script owns the ESPN page. All
   traffic goes over postMessage, and every applied pick is announced. */
window.addEventListener("message", ev => {
  const m = ev.data;
  if (!m || m.__draft !== 1) return;
  if (m.type === "picks"){
    let applied = 0; const missed = [];
    for (const p of m.picks){
      const r = window.DRAFTAPP.takeByName(p.name, !!p.mine, true,
                                           p.team, p.pick);
      if (r.ok) applied++;
      else if (r.reason === "no match") missed.push(p.name);
    }
    window.DRAFTAPP.setSync(
      `synced ${m.source} · ${m.picks.length} seen` +
      (missed.length ? ` · UNMATCHED: ${missed.join(", ")}` : ""),
      missed.length ? "warn" : "pos-n");
  } else if (m.type === "status"){
    window.DRAFTAPP.setSync(m.text, m.cls);
  } else if (m.type === "probeResult"){
    window.DRAFTAPP.showDiag(m.text);
  } else if (m.type === "slot"){
    window.DRAFTAPP.setSlot(m.slot);
  } else if (m.type === "myteam"){
    window.DRAFTAPP.setMyTeam(m.team);
  } else if (m.type === "pickno"){
    window.DRAFTAPP.setPickNo(m.pick);
  }
});
parent.postMessage({__draft:1, type:"ready",
                    names: window.DRAFTAPP.names()}, "*");
"""

MANIFEST = {
    "manifest_version": 3,
    "name": "Draft Day 2026 — ESPN sync",
    "version": "2.4",
    "description": "Overlays the model board on the ESPN draft room and picks "
                   "up drafted players automatically.",
    "permissions": ["storage"],
    "host_permissions": [
        "https://fantasy.espn.com/*",
        "https://lm-api-reads.fantasy.espn.com/*",
    ],
    "content_scripts": [{
        "matches": ["https://fantasy.espn.com/football/*"],
        "js": ["content.js"],
        "run_at": "document_idle",
    }],
    "web_accessible_resources": [{
        "resources": ["panel.html", "app.js", "bridge.js"],
        "matches": ["https://fantasy.espn.com/*"],
    }],
}

CONTENT_JS = r"""
/* Runs on the ESPN draft page. Two jobs: put the panel on screen, and get
   picks out of ESPN into it.

   Ingestion order:
     1. ESPN's JSON API (mDraftDetail). Structured, survives a UI redesign.
     2. DOM scan, as a fallback. Matches on KNOWN PLAYER NAMES rather than CSS
        selectors -- ESPN's class names are obfuscated and change between
        seasons, player names do not.
     3. Neither working -> say so and leave the panel in manual mode. Manual is
        always sufficient; that is the whole design.

   Not verified against a live draft room. Run __draftProbe() in the console
   (switch the console context dropdown to this extension first) to see exactly
   what each path can see. */

const SEASON = __SEASON__;
const POLL_MS = 2500;
const PANEL_W = __PANEL_W__;

/* The panel is welcome on any ESPN fantasy page, but INGESTION must not be.
   The mock draft LOBBY lists rooms with times and counts ("3:55 PM", "0/10"),
   and the old DOM scan happily read those as picks -- it logged 154 phantom
   picks off a page where no draft was happening, and localStorage carried the
   damage forward. Ingest only where a draft actually exists.

   The trailing (\/|$) is not decoration: a bare prefix test also matches
   /football/draftrecap, a real ESPN page that lists a completed draft in full.
   Scanning it would ingest a whole finished draft as though it were happening
   now -- the lobby bug again, with more convincing data. */
function inDraftRoom(){
  return /^\/football\/draft(\/|$)/.test(location.pathname);
}

let PANEL = null, NAMES = [], seen = new Set(), mode = "starting";

function mountPanel(){
  if (document.getElementById("draftday-panel")) return;
  const host = document.createElement("div");
  host.id = "draftday-panel";
  host.style.cssText = `position:fixed;top:0;right:0;width:${PANEL_W}px;
    height:100vh;z-index:2147483647;box-shadow:-2px 0 18px rgba(0,0,0,.55);
    border-left:1px solid #252a34;background:#0f1115;`;
  const f = document.createElement("iframe");
  f.src = chrome.runtime.getURL("panel.html");
  f.style.cssText = "width:100%;height:100%;border:0;";
  host.appendChild(f);

  const grip = document.createElement("div");
  grip.textContent = "⇔";
  grip.title = "drag to resize · double-click to collapse";
  grip.style.cssText = `position:absolute;left:-15px;top:50%;width:15px;
    height:44px;background:#252a34;color:#8b93a3;cursor:col-resize;
    display:flex;align-items:center;justify-content:center;
    border-radius:5px 0 0 5px;font-size:11px;`;
  let drag = false;
  grip.onmousedown = e => { drag = true; e.preventDefault(); };
  window.addEventListener("mousemove", e => {
    if (drag) host.style.width =
      Math.min(760, Math.max(280, window.innerWidth - e.clientX)) + "px";
  });
  window.addEventListener("mouseup", () => { drag = false; });
  grip.ondblclick = () => {
    host.style.width = host.style.width === "20px" ? PANEL_W + "px" : "20px";
  };
  host.appendChild(grip);
  document.body.appendChild(host);
  PANEL = f;
}

function send(msg){
  if (PANEL && PANEL.contentWindow)
    PANEL.contentWindow.postMessage(Object.assign({__draft:1}, msg), "*");
}
function status(text, cls){ send({type:"status", text, cls}); }

/* ---------- path 1: ESPN JSON API --------------------------------------- */
function leagueId(){
  const u = new URL(location.href);
  return u.searchParams.get("leagueId")
      || (location.pathname.match(/leagueId\/(\d+)/) || [])[1]
      || null;
}

let SLOT_SENT = false, TEAM_SENT = false;
function myTeamId(){
  const v = new URL(location.href).searchParams.get("teamId");
  return v == null ? null : Number(v);
}

let PLAYERS = null;
async function loadPlayerMap(){
  if (PLAYERS) return PLAYERS;
  const cached = await chrome.storage.local.get("playerMap");
  if (cached.playerMap && cached.playerMap.season === SEASON){
    PLAYERS = cached.playerMap.map; return PLAYERS;
  }
  const url = `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/`
            + `seasons/${SEASON}/players?scoringPeriodId=0&view=players_wl`;
  const r = await fetch(url, {credentials:"include",
    headers:{"x-fantasy-filter": JSON.stringify({players:{limit:2000}})}});
  if (!r.ok) throw new Error("player map HTTP " + r.status);
  const list = await r.json();
  PLAYERS = {};
  for (const p of list) PLAYERS[p.id] = p.fullName || p.name;
  await chrome.storage.local.set({playerMap:{season:SEASON, map:PLAYERS}});
  return PLAYERS;
}

async function pollApi(){
  const lid = leagueId();
  if (!lid) throw new Error("no leagueId in URL");
  const url = `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/`
            + `seasons/${SEASON}/segments/0/leagues/${lid}?view=mDraftDetail`;
  const r = await fetch(url, {credentials:"include"});
  if (!r.ok) throw new Error("draft HTTP " + r.status);
  const j = await r.json();
  const all = ((j.draftDetail || {}).picks) || [];
  /* ESPN pre-populates one entry per SLOT -- 12 teams x 15 rounds = 180 -- from
     the moment the room opens. An unmade slot carries no real playerId, so
     counting rows here counts the draft board, not the draft. */
  const picks = all.filter(p => p && p.playerId > 0);
  if (!picks.length) throw new Error("draft has not started");
  const map = await loadPlayerMap();
  const mineTeam = myTeamId();
  const out = [];
  for (const p of picks.sort((a,b) =>
        (a.overallPickNumber||0) - (b.overallPickNumber||0))){
    const k = "api:" + p.overallPickNumber;
    if (seen.has(k)) continue;
    const nm = map[p.playerId];
    if (!nm) continue;
    seen.add(k);
    /* the payload knows whose pick it was, so your own roster fills itself --
       and `teamId` is equally what lets the panel rebuild every OPPONENT's
       roster live, at no extra request */
    out.push({name:nm, pick:p.overallPickNumber, team:p.teamId,
              mine: mineTeam != null && p.teamId === mineTeam});
  }
  if (mineTeam != null && !TEAM_SENT){
    TEAM_SENT = true;
    send({type:"myteam", team: mineTeam});
  }
  /* round 1 position of your team IS your draft slot -- no need to set it */
  if (mineTeam != null && !SLOT_SENT){
    const r1 = all.find(p => p.roundId === 1 && p.teamId === mineTeam);
    if (r1 && r1.roundPickNumber){
      SLOT_SENT = true;
      send({type:"slot", slot: r1.roundPickNumber});
    }
  }
  /* The AUTHORITATIVE pick number. The panel used to derive it as
     `state.picks.length + 1`, which counts only picks it could match to a
     name on a 235-player board -- so every time the room drafted a kicker, a
     defence, or a deep bench player, the panel fell one pick further behind
     ESPN, permanently. That silently corrupts the horizon and therefore every
     conditional availability number, and it drifts in the direction that
     matters: the panel thinks you have more time than you do. */
  const made = picks.reduce((m, p) => Math.max(m, p.overallPickNumber || 0), 0);
  send({type:"pickno", pick: made + 1});
  return out;
}

/* ---------- path 2: DOM scan -------------------------------------------- */
/* Only ever called inside a draft room. Requires an explicit round/pick token
   tight against the name -- "R4, P11" or "Pick 37" within ~25 characters. The
   previous version accepted any two digits within 60 characters, which is why
   a lobby table full of clock times registered as a draft. */
const PICK_TOKEN = /(\bR\s?\d{1,2}\s*[,.]?\s*P\s?\d{1,2}\b|\bpick\s+\d{1,3}\b)/i;
function scanDom(){
  if (!NAMES.length || !inDraftRoom()) return [];
  const text = document.body.innerText || "";
  const out = [];
  for (const nm of NAMES){
    if (seen.has("dom:" + nm)) continue;
    const i = text.indexOf(nm);
    if (i < 0) continue;
    if (PICK_TOKEN.test(text.slice(Math.max(0, i - 25), i + nm.length + 25))){
      seen.add("dom:" + nm);
      out.push({name:nm});
    }
  }
  return out;
}

/* ---------- loop --------------------------------------------------------- */
/* The API is never permanently abandoned. It used to be: one failed poll set
   `apiDead = true` for the life of the page, and the extension spent the rest
   of the draft on the DOM scanner. That is close to useless INSIDE the draft
   room -- ESPN's visible list is who is AVAILABLE, and the pick ticker carries
   team names ("Team 11", "monkey"), not player names -- so a single transient
   429 from polling every 2.5s looked exactly like the tool quietly dying
   around round 11. It was still running; it had just blinded itself.

   Failures now back off and retry, forever. Recovery repairs itself: a pick
   only enters `seen` when it is actually emitted, so every pick missed during
   an outage is re-emitted on the first poll that succeeds. */
let apiFails = 0, skipTicks = 0;
async function tick(){
  if (!inDraftRoom()){
    if (mode !== "idle"){
      mode = "idle";
      status("not a draft room — sync off (manual still works)", "dim");
    }
    return;
  }
  let picks = [], source = mode, apiOk = false;
  if (skipTicks > 0){
    skipTicks--;
  } else {
    try{
      picks = await pollApi();
      apiOk = true;
      source = "API";
      if (mode !== "API" || apiFails){
        mode = "API"; apiFails = 0;
        status("synced via ESPN API", "pos-n");
      }
    }catch(e){
      /* "not started" is a state, not a failure -- do not back off for it */
      if (/not started/.test(e.message)){
        if (mode !== "waiting"){
          mode = "waiting";
          status("in the draft room — waiting for the first pick", "dim");
        }
        return;
      }
      apiFails++;
      skipTicks = Math.min(2 ** apiFails, 24);        // cap the wait at ~60s
      mode = "retry";
      status(`ESPN API failed (${e.message}) — retry ${apiFails} in `
             + `${Math.round(skipTicks * POLL_MS / 1000)}s · DOM fallback `
             + `cannot see picks in the draft room`, "warn");
    }
  }
  if (!apiOk){
    const dom = scanDom();
    if (dom.length){ picks = dom; source = "DOM"; }
  }
  if (picks.length) send({type:"picks", picks, source});
}

window.__draftProbe = async function(){
  const out = {url: location.href, leagueId: leagueId(),
               panelMounted: !!PANEL, mode, namesLoaded: NAMES.length,
               appliedSoFar: seen.size};
  try{
    const lid = leagueId();
    const r = await fetch(`https://lm-api-reads.fantasy.espn.com/apis/v3/`
      + `games/ffl/seasons/${SEASON}/segments/0/leagues/${lid}`
      + `?view=mDraftDetail`, {credentials:"include"});
    out.apiStatus = r.status;
    const j = await r.json();
    const all = ((j.draftDetail||{}).picks||[]);
    const picks = all.filter(p => p && p.playerId > 0);
    out.apiSlotCount = all.length;          // every slot, made or not
    out.apiPickCount = picks.length;        // actually drafted
    out.apiSample = picks.slice(0,2);
    out.myTeamId = myTeamId();
    /* the number that actually matters: of the picks the API reports, how many
       land on a player the board knows about */
    try{
      const map = await loadPlayerMap();
      const named = picks.map(p => map[p.playerId]).filter(Boolean);
      out.apiMinePicks = picks.filter(p => p.teamId === myTeamId()).length;
      out.apiNamesResolved = named.length;
      out.apiNamesSample = named.slice(0, 6);
    }catch(e){ out.apiNameError = String(e); }
  }catch(e){ out.apiError = String(e); }

  /* the suspect: how many player IDs actually resolve to a name */
  try{
    const map = await loadPlayerMap();
    const keys = Object.keys(map);
    out.playerMapSize = keys.length;
    out.playerMapSample = keys.slice(0,3).map(k => k + " -> " + map[k]);
    out.playerMapUndefined = keys.filter(k => !map[k]).length;
  }catch(e){ out.playerMapError = String(e); }

  /* raw shape of the players payload, so the map can be built correctly */
  try{
    const r2 = await fetch(`https://lm-api-reads.fantasy.espn.com/apis/v3/`
      + `games/ffl/seasons/${SEASON}/players?scoringPeriodId=0&view=players_wl`,
      {credentials:"include",
       headers:{"x-fantasy-filter": JSON.stringify({players:{limit:5}})}});
    out.playersStatus = r2.status;
    const j2 = await r2.json();
    out.playersIsArray = Array.isArray(j2);
    out.playersLen = Array.isArray(j2) ? j2.length : Object.keys(j2||{}).length;
    out.playersRawFirst = Array.isArray(j2) ? j2[0] : j2;
  }catch(e){ out.playersError = String(e); }

  const txt = document.body.innerText || "";
  out.domNameHits = NAMES.filter(n => txt.includes(n)).length;
  console.log("[draft probe]", out);
  return out;
};

/* Finds the DOM node that actually holds the pick list, by looking for the
   element with the most board names in it and the fewest children -- i.e. the
   tightest container around real picks. Reports enough structure to write an
   exact scraper without guessing at class names. */
window.__domDump = function(){
  if (!NAMES.length) return "panel not ready";
  const hits = [];
  document.querySelectorAll("div,section,ul,ol,table,tbody").forEach(el => {
    const t = el.innerText || "";
    if (t.length > 20000) return;
    const n = NAMES.filter(nm => t.includes(nm)).length;
    if (n >= 3) hits.push({n, el, depth: t.length});
  });
  hits.sort((a,b) => (b.n / Math.log(b.depth+10)) - (a.n / Math.log(a.depth+10)));
  const top = hits.slice(0, 5).map(h => ({
    names: h.n,
    tag: h.el.tagName,
    cls: (h.el.className || "").toString().slice(0, 90),
    id: h.el.id || "",
    childTags: [...h.el.children].slice(0,4).map(c => c.tagName + "." +
      (c.className||"").toString().split(" ")[0]).join(", "),
    sample: (h.el.innerText || "").slice(0, 220).replace(/\n/g, " | ")
  }));
  console.log("[dom dump]", top);
  return top;
};

window.addEventListener("message", async ev => {
  const m = ev.data;
  if (!m || m.__draft !== 1) return;
  if (m.type === "ready"){
    NAMES = m.names || [];
    status("panel ready — watching for picks", "dim");
    tick(); setInterval(tick, POLL_MS);
  } else if (m.type === "probe"){
    /* The panel is sandboxed in an iframe and cannot reach the ESPN page, so
       it asks us. Run both probes and hand back plain text to paste. */
    let text;
    try{
      const probe = await window.__draftProbe();
      const dump = window.__domDump();
      text = JSON.stringify({probe, dom: dump}, null, 1);
    }catch(e){ text = "probe failed: " + String(e) + "\n" + (e && e.stack); }
    send({type:"probeResult", text});
  }
});
mountPanel();
"""


def write_extension(app_js: str) -> None:
    """Unpacked MV3 extension. JS goes in SEPARATE FILES -- extension pages
    block inline <script> under the default MV3 CSP, so an inline build paints
    the markup and silently runs nothing."""
    if EXT.exists():
        shutil.rmtree(EXT)
    EXT.mkdir(parents=True)
    (EXT / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
    (EXT / "panel.html").write_text(
        MARKUP + '<script src="app.js"></script>\n'
                 '<script src="bridge.js"></script>\n')
    (EXT / "app.js").write_text(app_js)
    (EXT / "bridge.js").write_text(BRIDGE_JS)
    (EXT / "content.js").write_text(
        CONTENT_JS.replace("__SEASON__", str(SEASON))
                  .replace("__PANEL_W__", str(PANEL_W)))


def load_opponents() -> dict:
    """teamId -> {name, lean} for the 2026 league.

    Joins `data/league_history/teams_2026.json` (teamId -> owner GUID) to
    `data/league_tendencies.json` (owner GUID -> shrunk tendencies). A manager
    with no history -- the two new members -- gets a name and no lean, which
    is the correct output, not a gap to be filled in.
    """
    hist = league.teams_path()
    tend = league.tendencies_path()
    if not hist.exists():
        print("no teams_2026.json -- opponent view will show team ids only")
        return {}
    teams = json.loads(hist.read_text())
    mgrs, share = {}, {}
    if tend.exists():
        t = json.loads(tend.read_text())
        mgrs = t.get("managers", {})
        share = t.get("league_pos_share", {})

    prof = {}
    pf = league.opponents_csv()
    if pf.exists():
        import csv as _csv
        txt = "\n".join(l for l in pf.read_text().splitlines()
                        if not l.lstrip().startswith("#"))
        for row in _csv.DictReader(txt.splitlines()):
            rec = {"slot": (row.get("slot") or "").strip(),
                   "label": (row.get("team") or "").strip(),
                   "experience": (row.get("experience") or "vet").strip(),
                   "note": (row.get("note") or "").strip()}
            tid = (row.get("espn_team_id") or "").strip()
            if tid.isdigit():
                prof[int(tid)] = rec          # stable key
            prof[rec["label"]] = rec          # name fallback

    out = {}
    for tm in teams.get("teams", []):
        owner = (tm.get("owners") or [None])[0]
        m = mgrs.get(owner) if owner else None
        bits = []
        if m and share:
            for pos in ("QB", "RB", "WR", "TE"):
                lg = share.get(pos)
                mine = m.get("pos_share_shrunk", {}).get(pos)
                if not lg or mine is None:
                    continue
                if mine / lg >= 1.12:
                    bits.append(pos + "-heavy")
                elif mine / lg <= 0.88:
                    bits.append(pos + "-light")
            q = m.get("first_qb_frac")
            if q is not None:
                bits.append(f"QB~R{max(1, round(q * ROUNDS_N))}")
            r = m.get("reach_shrunk", 0)
            if r <= -0.3:
                bits.append("reaches")
            elif r >= 0.3:
                bits.append("waits")
        name = tm.get("name", f"team {tm['id']}")
        rec = {"name": name, "lean": " · ".join(bits)}

        # Hand-maintained profile: experience and anything the user knows that the
        # pick history cannot see. THREE of twelve 2026 seats have no history
        # at all, so this file is the only source for a quarter of the room.
        p = prof.get(tm["id"]) or prof.get(name)
        if p:
            rec["exp"] = p["experience"]
            # prefer the label the user chose over ESPN's placeholder team name
            if p.get("label"):
                rec["name"] = p["label"]
            rec["note"] = p["note"]
            if p["experience"] == "beginner" and not bits:
                # A first-timer has no rankings of his own, so he takes what is
                # top of his ESPN screen. That is the PUBLISHED queue rank, not
                # ESPN's ADP, and the two disagree wildly. The queue ranks the
                # first TE at #17 and the first QB at #36, with K at #251 and
                # DST at #234 -- so expect WR/RB-heavy, TE-light, and NOT the
                # early kicker a beginner is usually assumed to take.
                # ⚠️ Assumption, not measurement: untestable at n=0 history,
                # and ESPN's market was rejected league-wide (replay_market.py).
                rec["lean"] = ("first-timer · likely drafts off ESPN's queue "
                               "(TE-light, no early K/DST)")
                rec["assumed"] = 1
        out[tm["id"]] = rec

    known = sum(1 for v in out.values() if v["lean"])
    npro = sum(1 for v in out.values() if v.get("exp"))
    print(f"opponent view: {len(out)} teams, {known} with a lean, "
          f"{npro} with a hand profile")
    missing = [v["name"] for v in out.values() if not v.get("exp")]
    if missing:
        print(f"  !! no opponents_2026.csv row for: {', '.join(missing)}")
    return out


ROUNDS_N = league.settings()["rounds"]


def main() -> None:
    _L = league.settings()
    for w in league.warnings():
        print("  !!", w)
    data, platform_meta = load_board()
    app_js = (APP_JS.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                    .replace("__OPPONENTS__",
                             json.dumps(load_opponents(), separators=(",", ":")))
                    .replace("__PORT__", str(STATE_PORT))
                    .replace("__TEAMS__", str(_L["teams"]))
                    .replace("__ROUNDS__", str(_L["rounds"]))
                    .replace("__LINEUP__", json.dumps(_L["lineup"]))
                    .replace("__FLEX__", str(_L["flex"]))
                    .replace("__PLATFORMS__",
                             json.dumps(platform_meta, separators=(",", ":"))))

    dest = OUT / "draft_day.html"
    dest.write_text(MARKUP + "<script>\n" + app_js + "\n</script>\n")
    print(f"wrote {dest}  ({len(data)} players, "
          f"{dest.stat().st_size/1024:.0f} KB)")

    write_extension(app_js)
    print(f"wrote {EXT}/  (manifest, panel.html, app.js, bridge.js, content.js)")
    print("\nstandalone:  open " + str(dest))
    print("extension:   reload it at chrome://extensions (circular arrow), "
          "then reload the ESPN tab")


if __name__ == "__main__":
    main()
