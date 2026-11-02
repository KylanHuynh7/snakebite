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
    for name, r in zip(names, rows):
        r["pl"] = per_platform.get(name, {})
    return rows, platform_meta


# --------------------------------------------------------------------- markup

MARKUP = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Draft Day 2026</title>
<style>
:root{
  --bg:#0f1115; --panel:#171a21; --line:#252a34; --ink:#e6e9ef; --dim:#8b93a3;
  --pos:#3fb950; --neg:#f85149; --warn:#d29922; --acc:#58a6ff;
  --qb:#c2410c; --rb:#15803d; --wr:#1d4ed8; --te:#7e22ce;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.4 ui-sans-serif,-apple-system,"SF Pro Text",system-ui,sans-serif}
#wrap{padding:8px;max-width:1100px;margin:0 auto}
#search{width:100%;padding:9px 10px;font-size:15px;background:#0b0d11;
  color:var(--ink);border:1px solid var(--line);border-radius:7px;outline:none}
#search:focus{border-color:var(--acc)}
#meta{display:flex;gap:6px;align-items:center;flex-wrap:wrap;
  margin:7px 0;font-size:12px;color:var(--dim)}
#meta b{color:var(--ink)}
button{background:#20242e;color:var(--ink);border:1px solid var(--line);
  border-radius:6px;padding:4px 8px;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}
button.on{background:var(--acc);border-color:var(--acc);color:#04121f;
  font-weight:600}
input[type=number]{width:44px;background:#0b0d11;color:var(--ink);
  border:1px solid var(--line);border-radius:6px;padding:3px 5px;font-size:12px}
#rec{background:linear-gradient(180deg,#16241a,#141821);
  border:1px solid #2b4133;border-radius:9px;padding:9px 10px;margin-bottom:8px}
#rec.thin{background:#171a21;border-color:var(--line)}
.rlab{font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--pos);font-weight:700;margin-bottom:3px}
#rec.thin .rlab{color:var(--dim)}
.rname{font-size:16px;font-weight:700}
.rwhy{font-size:11.5px;color:var(--dim);margin-top:3px}
.alt{font-size:11.5px;color:var(--dim);margin-top:5px;
  border-top:1px solid var(--line);padding-top:5px}
#ctx{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:7px}
.pill{display:inline-block;background:#20242e;border:1px solid var(--line);
  border-radius:20px;padding:1px 7px;font-size:11px;color:var(--dim)}
.need{border-color:var(--warn);color:var(--warn)}
.done{border-color:var(--pos);color:var(--pos)}
#bar{display:flex;gap:4px;align-items:center;margin-bottom:6px}
#plat{display:flex;gap:2px;flex-wrap:wrap;margin:2px 0 4px;
  border-bottom:1px solid var(--line)}
#plat button{background:none;border:0;border-bottom:2px solid transparent;
  border-radius:0;color:var(--dim);padding:5px 9px;cursor:pointer;font-size:12px}
#plat button.on{color:var(--ink);border-bottom-color:var(--acc)}
#plat button:disabled{opacity:.4;cursor:not-allowed}
#platnote{font-size:11px;color:var(--dim);margin:0 0 6px}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--dim);font-weight:600;padding:3px 4px;
  border-bottom:1px solid var(--line)}
td{padding:4px;border-bottom:1px solid #1c202a;vertical-align:top}
tr.row:hover{background:#1c2029;cursor:pointer}
.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.nm{font-weight:600}
.pos{display:inline-block;min-width:22px;text-align:center;border-radius:3px;
  padding:0 4px;font-size:10px;font-weight:700;color:#fff}
.QB{background:var(--qb)}.RB{background:var(--rb)}
.WR{background:var(--wr)}.TE{background:var(--te)}
.note{font-size:11px;margin-top:1px}
/* team scoring-environment forecast -- deliberately small and inline, it is
   context on the offence, never a reason to move a player by itself */
.offtag{font-size:10px;opacity:.75;white-space:nowrap}
.gap{font-weight:600;padding:0 4px;border-radius:3px}
.gap.up{color:var(--pos);background:rgba(63,185,80,.12)}
.gap.dn{color:var(--neg);background:rgba(248,81,73,.12)}
.mkt{color:var(--dim)}
.exp{font-size:10px;padding:1px 4px;border-radius:3px;margin-left:4px;
  background:var(--line);color:var(--dim)}
.exp.beginner{background:rgba(210,153,34,.18);color:var(--warn)}
/* opponent view: who picks before you do, and what they still need */
#opp{margin-bottom:7px;font-size:11px}
#opp .hd{font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--dim);font-weight:700;margin-bottom:3px;cursor:pointer}
#opp table{width:100%}
#opp td{padding:2px 4px;border-bottom:1px solid #1c202a;font-size:11px}
#opp .tm{font-weight:600}
#opp .lean{color:var(--dim);font-size:10px}
#opp .cnt{font-variant-numeric:tabular-nums;color:var(--dim);white-space:nowrap}
.neg{color:var(--neg)}.pos-n{color:var(--pos)}.warn{color:var(--warn)}
.dim{color:var(--dim)}
.small{font-size:11px;color:var(--dim)}
.tierbreak td{border-top:2px solid #313847}
#hints{display:none;font-size:11px;color:var(--dim);margin:4px 0}
#log{display:none;max-height:260px;overflow:auto;font-size:10.5px;
  margin-top:6px;color:var(--dim);white-space:pre-wrap;word-break:break-all;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  background:#0b0d11;border:1px solid var(--line);border-radius:6px;padding:6px;
  user-select:all}
kbd{background:#0b0d11;border:1px solid var(--line);border-radius:3px;
  padding:0 4px;font-size:10px;font-family:ui-monospace,monospace}
</style>

<div id="wrap">
  <input id="search" placeholder="name → Enter = gone · Shift+Enter = mine"
         autocomplete="off" spellcheck="false">
  <div id="hints">
    <kbd>Enter</kbd> mark top match gone · <kbd>Shift</kbd>+<kbd>Enter</kbd>
    my pick · <kbd>Esc</kbd> clear · <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo ·
    click a row to mark it gone
  </div>
  <div id="meta">
    <span>pick <b id="pick">1</b></span>
    <span>· slot <input type="number" id="slot" min="1" max="12" value="3"></span>
    <span id="nextpk"></span>
    <span style="flex:1"></span>
    <button id="undo">undo</button>
    <button id="more">···</button>
    <button id="diag" title="run ESPN sync diagnostics">diag</button>
  </div>
  <div id="sync" class="small" style="margin-bottom:6px">manual mode</div>
  <div id="log"></div>

  <div id="plat" title="whose ADP drives every availability number"></div>
  <div id="platnote"></div>
  <div id="rec"></div>
  <div id="ctx"></div>
  <div id="opp"></div>

  <div id="bar">
    <button data-f="ALL" class="on">All</button>
    <button data-f="QB">QB</button><button data-f="RB">RB</button>
    <button data-f="WR">WR</button><button data-f="TE">TE</button>
    <button data-f="WATCH" title="only my watchlist">★ mine</button>
    <span style="flex:1"></span>
    <button id="reset">reset</button>
  </div>
  <table><thead><tr>
    <th class="num">#</th><th>Player</th><th></th>
    <th class="num">VOR</th>
    <th class="num" title="Odds he is still there at your next pick, given he is here now">→nxt</th>
  </tr></thead><tbody id="rows"></tbody></table>
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
const TEAMS = 12, ROUNDS = 15;
const LINEUP = {QB:1, RB:2, WR:2, TE:1};   // + 2 FLEX (RB/WR/TE)
const FLEX = 2;
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
let OPP_OPEN = true;
function renderOpponents(now, nxt){
  const el = document.getElementById("opp");
  if (!el) return;
  const known = state.picks.filter(p => p.team != null).length;
  if (!known){
    el.innerHTML = `<div class="hd">opponents</div><div class="dim">waiting `
      + `for the ESPN feed — manual picks carry no team, so rosters cannot `
      + `be attributed</div>`;
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
    const nd = need.length ? need.join(" · ") : "starters full";
    /* An "assumed" lean is reasoning from a profile, not from picks -- it must
       never look like the measured ones sitting next to it. */
    const lean = info.lean
      ? `<span class="lean${info.assumed ? " dim" : ""}"> ${info.lean}`
        + `${info.assumed ? " (assumed)" : ""}</span>` : "";
    const exp = (info.exp && info.exp !== "vet")
      ? `<span class="exp ${info.exp}">${info.exp}</span>` : "";
    const note = info.note
      ? `<div class="lean dim">${info.note}</div>` : "";
    return `<tr><td class="cnt">${b.picks.join(",")}</td>`
         + `<td><span class="tm">${info.name || ("team " + b.team)}</span>${exp}${lean}`
         + `<div class="lean">${c.n} picked · needs ${nd}</div>${note}</td></tr>`;
  }).join("");
  el.innerHTML = `<div class="hd" id="opphd">opponents before your pick`
    + `${nxt ? " " + nxt : ""} — ${between.length} manager`
    + `${between.length === 1 ? "" : "s"} ${OPP_OPEN ? "▾" : "▸"}</div>`
    + (OPP_OPEN
        ? (rows ? `<table>${rows}</table>`
                : `<div class="dim">nobody picks before you</div>`)
        : "");
  const hd = document.getElementById("opphd");
  if (hd) hd.onclick = () => { OPP_OPEN = !OPP_OPEN; render(); };
}

function render(){
  const gone = goneSet(), now = curPick();
  const mine = nextOfMine(), nxt = horizon();
  const onClock = mine === now;
  const q = document.getElementById("search").value.trim().toLowerCase();
  const live = BOARD.filter(p => !gone.has(p.name));

  document.getElementById("pick").textContent = now;
  document.getElementById("nextpk").innerHTML = !mine ? "· done"
    : onClock
      ? `· <b style="color:var(--pos)">YOU ARE UP</b>` +
        (nxt ? ` <span class="dim">· back at ${nxt}</span>` : "")
      : `· yours <b>${mine}</b> <span class="dim">(${mine - now} away)</span>`;

  /* recommendation */
  const rec = document.getElementById("rec");
  const rows = recommend(live, nxt, now);
  if (!rows.length){ rec.innerHTML = ""; }
  else {
    const t = rows[0];
    const a = nxt ? availAt(t.best, nxt, now) : null;
    const aTxt = !nxt || a == null ? ""
      : `${a.est ? "~" : ""}${Math.round(a.v*100)}% to last to ${nxt}` +
        (a.est ? " (already sliding past ADP)" : "");
    const thin = t.cost < 0.35;
    rec.className = thin ? "thin" : "";
    rec.innerHTML = `
      <div class="rlab">${thin ? "no urgency — best available" : "take now"}</div>
      <div class="rname">${t.best.name}
        <span class="pos ${t.best.pos}">${t.best.pos}</span>
        <span class="dim" style="font-size:12px">${t.best.team}</span></div>
      ${t.best.note ? `<div class="note ${
          t.best.flag==="neg"?"neg":t.best.flag==="pos"?"pos-n":"warn"}">${
          t.best.flag==="neg"?"! ":""}${t.best.note}</div>` : ""}
      <div class="rwhy">${
        thin
        ? `${t.pos} holds up — expect ~${t.exp.toFixed(1)} VOR still there at ${nxt||"?"}. ${aTxt}`
        : `waiting costs <b>${t.cost.toFixed(1)} VOR</b> at ${t.pos}: ${
            t.best.vor.toFixed(1)} now vs ~${t.exp.toFixed(1)} expected at ${nxt}. ${aTxt}`}
      </div>
      <div class="alt">then: ${rows.slice(1,4).map(r =>
        `<b>${r.pos}</b> ${r.best.name.split(" ").slice(-1)[0]} (−${
          r.cost.toFixed(1)})`).join(" · ")}</div>`;
  }

  /* context: needs + what the room has been doing */
  const c = rosterCounts();
  let pills = "";
  for (const [pos, n] of Object.entries(LINEUP))
    pills += `<span class="pill ${c[pos]>=n?"done":"need"}">${pos} ${c[pos]}/${n}</span>`;
  const flexHave = ["RB","WR","TE"].reduce(
    (a,k) => a + Math.max(0, c[k] - LINEUP[k]), 0);
  pills += `<span class="pill ${flexHave>=FLEX?"done":"need"}">FLX ${
    Math.min(flexHave,FLEX)}/${FLEX}</span>`;
  const recent = state.picks.slice(-12);
  const tally = {};
  recent.forEach(p => { const pl = byName[p.name];
                        if (pl) tally[pl.pos] = (tally[pl.pos]||0)+1; });
  const runTxt = Object.entries(tally).sort((a,b)=>b[1]-a[1])
    .map(([k,v]) => `${k} ${v}`).join(" ");
  /* Counts, NOT momentum. Over this league's 391 historical picks a run does
     not predict more of that position: 2-of-last-3 RB is followed by an RB at
     lift 0.92, 4-of-last-5 WR at 0.90 -- both inside noise of 1.0 and if
     anything negative. A WR run looks like a signal only because WR is 42% of
     every skill pick here. A pill that implies "get ahead of the run" is the
     tool talking you into a reach. */
  if (runTxt) pills += `<span class="pill" title="counts only -- runs have not
predicted more of that position in this league (lift 0.90-1.08 over 391 picks)"
>last 12: ${runTxt}</span>`;
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
  let lastTier = null, lastPos = null;
  for (const p of list){
    const tr = document.createElement("tr");
    tr.className = "row";
    if (state.filter !== "ALL" && lastTier !== null
        && p.tier !== lastTier && p.pos === lastPos) tr.className += " tierbreak";
    lastTier = p.tier; lastPos = p.pos;
    const a = nxt ? availAt(p, nxt, now) : null;
    const noteCls = p.flag === "neg" ? "neg" : p.flag === "pos" ? "pos-n"
                  : p.flag === "warn" ? "warn" : "dim";
    tr.innerHTML = `
      <td class="num dim">${p.gap ? `<span class="gap ${p.gap>0?"up":"dn"}"
        title="model disagrees with the market by ${Math.abs(p.gap)} ranks -- only
gaps of 24+ beat their price in the backtest">${p.gap>0?"+":""}${p.gap}</span>`
        : `<span class="mkt">${p.adp?Math.round(p.adp):"—"}</span>`}</td>
      <td>${p.watch ? '<span style="color:var(--acc)">★ </span>' : ""}<span class="nm">${p.name}</span>
          <span class="dim"> ${p.team}</span>
          ${p.off ? `<span class="offtag ${
            p.off.startsWith("OFF--") ? "neg" :
            p.off.startsWith("OFF-") ? "warn" :
            p.off.startsWith("OFF+") ? "pos-n" : "dim"}"> ${p.off}</span>` : ""}
          ${p.tdd ? `<span class="offtag ${
            parseFloat(p.tdd.split(" ")[2]) <= -1 ? "warn" : "dim"
            }"> ${p.tdd}</span>` : ""}
          ${p.stat ? `<div class="note ${
            ["REHAB","OUT","LEGAL"].includes(p.stat) ? "warn" : "dim"
            }" title="${(p.stn || "").replace(/"/g, "&quot;")}">${p.stat}${
            /* The status notes are dated, sourced paragraphs by design -- that
               is what makes them re-verifiable. But five of them on screen at
               once is why the panel needed constant scrolling, so the row shows
               the first sentence and the whole note lives in the tooltip. */
            p.stn ? " — " + (p.stn.length > 110
              ? p.stn.slice(0, 110).replace(/\s+\S*$/, "") + "…"
              : p.stn) : ""}</div>` : ""}
          ${p.watch && p.wnote ? `<div class="note" style="color:var(--acc)">${p.wnote}</div>` : ""}
          ${p.ctx ? `<div class="note dim">${p.ctx}</div>` : ""}
          ${p.qbd ? `<div class="note ${/high|unknown/.test(p.qbd)
            ? "warn" : "dim"}" title="41% of team-seasons lose 3+ starts from
their week-1 QB; that costs a true #1 target about 1.3 PPG. A variance label,
not a projection adjustment -- the model averages over whoever was throwing."
            >${p.qbd}</div>` : ""}
          ${p.note ? `<div class="note ${noteCls}">${
            p.flag==="neg"?"! ":""}${p.note}</div>` : ""}</td>
      <td><span class="pos ${p.pos}">${p.pos}</span></td>
      <td class="num">${p.vor == null ? "–" : p.vor.toFixed(1)}</td>
      <td class="num" title="${a && a.est ? "re-estimated: he has already fallen past his ADP" : ""}"
          style="color:${
        a == null ? "var(--dim)"
        : a.v >= .7 ? "var(--pos)" : a.v >= .35 ? "var(--warn)" : "var(--neg)"}">${
        a == null ? "–" : (a.est ? "~" : "") + Math.round(a.v*100)+"%"}</td>`;
    tr.onclick = () => take(p.name, false);
    body.appendChild(tr);
  }

  document.getElementById("log").innerHTML = state.picks.slice().reverse()
    .map((p,i) => `${state.picks.length-i}. ${p.name}${p.mine?" ← me":""}`)
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
    const first = document.querySelector("#rows tr .nm");
    if (first) take(first.textContent, e.shiftKey);
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
  state.slot = Math.min(12, Math.max(1, +e.target.value || 3));
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
    hist = ROOT / "data" / "league_history" / "teams_2026.json"
    tend = ROOT / "data" / "league_tendencies.json"
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
    pf = ROOT / "data" / "opponents_2026.csv"
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


ROUNDS_N = 15


def main() -> None:
    data, platform_meta = load_board()
    app_js = (APP_JS.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                    .replace("__OPPONENTS__",
                             json.dumps(load_opponents(), separators=(",", ":")))
                    .replace("__PORT__", str(STATE_PORT))
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
