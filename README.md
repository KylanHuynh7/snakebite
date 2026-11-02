# 🐍 Snakebite

**Will the player you want still be there at your next pick?**

Snakebite is a live draft-room tool for snake fantasy drafts. It combines a
platform's published ADP with each opponent's own draft history to estimate,
pick by pick, whether a target will survive until you are back on the clock.

> Not "who is best?" — but **"who will not be here when I come back?"**

```bash
pip install -r requirements.txt
python scripts/demo.py            # simulated 27-pick draft + full briefing
python scripts/demo.py --panel    # ...and replay it in the browser panel
```

No league, account, API key or network needed: the demo runs on the committed
board and a synthetic 12-manager league. Picks are simulated off the platform's
ADP (`--platform espn|ffc`), so it shows the mechanics, not the model's accuracy
— the accuracy evidence is `scripts/replay_brier.py`, which scores the same
probabilities against real drafts.

What makes it different from a ranking: the project's own model ranking *lost*
to platform ADP in backtests, so ADP is the prior and the model's job is the
thing ADP cannot do — conditional availability, calibrated against a league's
real pick history and updated as picks land.

---

## Platforms

The panel has one tab per platform, and the tab picks whose ADP prices every
player. Switching re-sorts the board and recomputes every availability number.

| tab | source | status |
|---|---|---|
| Consensus | FantasyFootballCalculator, pooled across sites | built in; the only one with a measured per-player spread, and the one the league calibration was validated on |
| ESPN | ESPN public ADP | built in |
| Sleeper | Sleeper projections API (`adp_ppr`) | built in; refresh with `python scripts/platforms.py fetch sleeper` |
| Yahoo, NFL.com, CBS, Underdog | no public ADP endpoint | drop a CSV (`name,adp`) in `data/adp_import/<site>.csv` and the tab lights up — see `TEMPLATE.csv` |

Any other site: add a row to `PLATFORMS` in `scripts/platforms.py`.

**Calibration caveat.** The spread and chalk-drift curves were fitted on a
league's FFC-space draft history. For Consensus that is the validated model; on
every other tab the same curve is applied in that platform's own rank space
(borrowing Consensus's measured per-player stdev), which is an approximation.
The tab says so. To do better, run `league_tendencies.py` on a league that
actually drafted on that platform.

---

## The idea

Every player carries a conditional availability curve: `avail_pk3` … `avail_pk171`,
one column per pick you own. Preseason, `avail_pk30` answers *"what were the odds
he lasts to pick 30?"*. Once picks are happening, the useful question is *"he is
STILL here at 18 — will he last to 30?"*, which is `P(avail@30) / P(avail@18)`.

The panel leads with a **recommendation**, not a ranked list, because you already
have a ranked list on paper. The recommendation is value-over-next-available: for
each position, compare the best man on the board *now* against the value you can
expect to still be there at your next pick, **given who has actually gone**. The
position with the most to lose by waiting is the position to draft.

---

## Layout

```
snakebite/
├── scripts/
│   │  ── the live tool ───────────────────────────────────
│   ├── build_draft_app.py      bakes the board into draft_day.html + the extension
│   ├── draft_state_server.py   localhost:8788 mirror of live draft state
│   ├── draft_advise.py         full briefing from live state, ~1s, no model
│   ├── test_ingestion.mjs      29 tests over the pick-ingestion logic
│   │
│   │  ── the probability engine ───────────────────────────
│   ├── phase1_baseline.py      nflverse pull → full-PPR VOR baseline
│   ├── phase2_projections.py   LightGBM 2026 PPG projections
│   ├── phase3_draft_board.py   veterans + rookies + ADP availability → the board
│   ├── rookie_features.py      rookie model (draft capital and what beats it)
│   ├── availability_model.py   is proj_games better learned than hand-set?
│   ├── calibrate_quantiles.py  are floor/ceil the percentiles they claim to be?
│   │
│   │  ── the market ────────────────────────────────────────
│   ├── espn_adp.py             ESPN ADP — the market this league drafts in
│   ├── espn_translate.py       isotonic FFC-rank → ESPN-rank map (reliable to ~120)
│   │
│   │  ── opponent modelling + validation ───────────────────
│   ├── league_tendencies.py    per-manager tendencies from 2023-25 ESPN history
│   ├── replay_brier.py         score P(available) against the real league drafts
│   ├── replay_market.py        does an ESPN-shaped prior beat the FFC one?
│   ├── need_redesign.py        second attempt at opponent-aware availability
│   ├── scan_opponent_weight.py how much opponent info belongs in the numbers
│   ├── board_backtest.py       would this board have beaten the market in 24/25?
│   ├── validate_board.py       sanity checks — run after every rebuild
│   └── qb_context.py           QB dependency → board annotation
│
├── data/     market pulls, league history, hand-maintained inputs
├── output/   draft_board_2026.csv, draft_day.html, espn_extension/, draft_state.json
└── .venv/    Python 3.13 — lightgbm, polars, pandas, sklearn, nflreadpy
```

---

## Running it

Everything uses the project venv. **System `python3` has no lightgbm.**

### Rebuild the board after an ADP re-pull

```bash
.venv/bin/python scripts/phase2_projections.py
.venv/bin/python scripts/phase3_draft_board.py 3      # 3 = your draft slot
.venv/bin/python scripts/validate_board.py
.venv/bin/python scripts/build_draft_app.py
node scripts/test_ingestion.mjs
```

`build_draft_app.py` **bakes the board in at build time** — rerun it after every
re-pull or the panel shows stale numbers.

### Draft night

```bash
# 1. state mirror, leave running in its own terminal
.venv/bin/python scripts/draft_state_server.py

# 2. load the extension
#    chrome://extensions → Developer mode → Load unpacked → output/espn_extension/
#    (already loaded? just hit the circular reload arrow, then reload the ESPN tab)

# 3. briefing between picks
.venv/bin/python scripts/draft_advise.py --slot 3
```

No-extension fallback — one self-contained file, no server, nothing to be down:

```bash
open output/draft_day.html
```

Type a name + <kbd>Enter</kbd> to mark a player gone, <kbd>Shift</kbd>+<kbd>Enter</kbd>
for your own pick.

---

## How picks get ingested

Three tiers, in order:

1. **ESPN JSON API** — `mDraftDetail` on the league endpoint, league id read from
   the page URL, your existing session cookies. Structured, and survives a UI
   redesign. Your own picks self-mark (each pick carries `teamId`) and your draft
   slot auto-detects from the round-1 `roundPickNumber`.
2. **DOM scan** — fallback. Matches on **known player names**, not CSS selectors,
   because ESPN's class names are obfuscated and change between seasons while
   player names don't.
3. **Neither working** → it says so and stays in manual mode.

Every auto-detected pick is announced in the panel and is undoable. Unmatched
names get listed, never dropped silently.

## The latency fix

An agent driving a browser pays, per step: snapshot → serialise the a11y tree →
model reads it → tool call → repeat. Several seconds before anything is *thought*,
and it scales with page complexity. That does not fit inside a 90-second pick clock.

Almost none of that work is necessary. Draft state is small — a pick number, a
slot, a list of names. So the page POSTs state to `127.0.0.1:8788`,
`draft_state_server.py` writes `output/draft_state.json`, and `draft_advise.py`
turns it into a briefing in ~1s with **no model involved**: best available,
conditional availability, tier cliffs, roster needs, positional run. A live
question then costs one inference instead of a snapshot/parse loop.

---

## Hand-maintained inputs

These are yours to edit; nothing regenerates them.

| file | what it is |
|---|---|
| `data/watchlist_2026.csv` | neutral demo watchlist. Put your own targets in `data/watchlist_private.csv` (gitignored) and it is used instead |
| `data/player_status_2026.csv` | dated injury/status lines — the model has no injury input on its own |
| `data/opponents_2026.csv` | profiles for the other 11 managers (synthetic in this repo — replace with your league) |
| `data/stances_private.py` | optional, gitignored: a `STANCES = {"Player Name": "label"}` dict of your own per-player calls, shown on the board |
| `data/ol_ratings_2026.csv`, `data/playcallers_2026.csv` | third-party OL/playcaller tables |

## Known biases — read before trusting a number

* **The board runs ~7% pessimistic on availability.** It expects more players
  gone than can be, which biases the panel toward taking players *early*.
  `validate_board.py` reports this on every run.
* **`espn_translate` collapses past FFC rank ~120.** `RELIABLE_TO = 120` is a real
  horizon, not a tuning knob. Outside it, sigma is widened so those players read
  as *unknown* rather than as confident predictions. QB carries ~14 ranks of
  residual even inside the horizon, so QB translations stay deliberately vague.
* **Run detection deliberately does NOT adjust ADP.** Showing the positional split
  of recent picks is honest; folding it into the prior would be inventing precision.
* **A player who outlived his ADP reads `fell`, not `0%`.** Once the market
  contradicts the prior, that number is dead information.
