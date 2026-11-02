/* Replay harness for the ESPN ingestion path.
 *
 *     node scripts/test_ingestion.mjs
 *
 * Loads the REAL output/espn_extension/content.js in a vm sandbox with stubbed
 * location / document / chrome / fetch, then feeds it synthetic mDraftDetail
 * payloads. Everything is exercised except the network call itself: the
 * 180-slot filter, name resolution, dedupe, self-marking picks, slot
 * auto-detection, the draft-room gate, and the DOM-scan pick regex.
 *
 * Testing the shipped file rather than a copy is the point -- a copy drifts,
 * and every one of these tests corresponds to a bug that actually shipped.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CONTENT_JS = join(ROOT, "output", "espn_extension", "content.js");
const SEASON = 2026;

/* ---------- fixtures ------------------------------------------------------ */

/* ESPN pre-populates one row per pick SLOT the moment the room opens, so a
   12x15 draft reports 180 rows before anyone has picked. Unmade slots carry
   playerId 0 or -1. Reproducing that exactly is the whole point of fixture. */
function draftDetail({ teams = 12, rounds = 15, made = 0, myTeamId = 3 } = {}) {
  const picks = [];
  for (let r = 1; r <= rounds; r++) {
    for (let i = 1; i <= teams; i++) {
      const overall = (r - 1) * teams + i;
      // snake: even rounds reverse, so team order flips
      const teamId = r % 2 ? i : teams - i + 1;
      const isMade = overall <= made;
      picks.push({
        overallPickNumber: overall,
        roundId: r,
        roundPickNumber: i,
        teamId,
        playerId: isMade ? 1000 + overall : 0,
      });
    }
  }
  return { draftDetail: { picks }, _myTeamId: myTeamId };
}

const playerList = (n = 200) =>
  Array.from({ length: n }, (_, i) => ({
    id: 1000 + i + 1,
    fullName: `Player ${i + 1}`,
  }));

/* Text off the real mock-draft lobby -- room names, clock times, member counts.
   This is what produced 154 phantom picks. */
const LOBBY_TEXT = `Mock Draft Lobby
Drafting Rooms: 191   Full Rooms: 150
Beginner 8-Team H2H Points Mock Salary Cap   Salary Cap   11:20 PM   0/8
Pro 10-Team H2H Points PPR Mock Salary Cap   Salary Cap   11:20 PM   0/10
Beginner 12-Team H2H Points Mock Salary Cap  Salary Cap   11:20 PM   0/12
Player 1   Player 2   Player 3   Player 4   Player 5`;

const DRAFT_ROOM_TEXT = `Round 4 Pick 11
R4, P11  Player 1  RB
Pick 38  Player 2  WR
On the clock: Player 99`;

/* ---------- sandbox ------------------------------------------------------- */

function boot({
  pathname = "/football/draft",
  search = "?leagueId=123456&teamId=3",
  detail = draftDetail({ made: 24 }),
  players = playerList(),
  bodyText = "",
  failApi = false,
} = {}) {
  const messages = [];
  const listeners = {};
  const href = `https://fantasy.espn.com${pathname}${search}`;

  const el = () => ({
    style: { cssText: "" },
    appendChild() {},
    addEventListener() {},
    set textContent(_v) {},
    set title(_v) {},
    set src(_v) {},
    set id(_v) {},
    contentWindow: { postMessage: (m) => messages.push(m) },
  });

  const sandbox = {
    console,
    URL,
    setInterval: () => 0,
    setTimeout: () => 0,
    location: { href, pathname, search },
    document: {
      createElement: el,
      getElementById: () => null,
      querySelectorAll: () => [],
      body: { appendChild() {}, innerText: bodyText },
    },
    window: {
      innerWidth: 1600,
      addEventListener(type, fn) {
        (listeners[type] ||= []).push(fn);
      },
    },
    chrome: {
      runtime: { getURL: (p) => `chrome-extension://test/${p}` },
      storage: { local: { get: async () => ({}), set: async () => {} } },
    },
    async fetch(url) {
      if (failApi) throw new Error("network down");
      if (url.includes("view=mDraftDetail"))
        return { ok: true, status: 200, json: async () => detail };
      if (url.includes("view=players_wl"))
        return { ok: true, status: 200, json: async () => players };
      return { ok: false, status: 404, json: async () => ({}) };
    },
  };
  sandbox.globalThis = sandbox;

  const ctx = vm.createContext(sandbox);
  vm.runInContext(readFileSync(CONTENT_JS, "utf8"), ctx, { filename: "content.js" });

  const fire = (data) =>
    Promise.all((listeners.message || []).map((fn) => fn({ data })));

  return { ctx, messages, fire };
}

/* ---------- runner -------------------------------------------------------- */

let pass = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    console.log(`  PASS  ${name}`);
    pass++;
  } catch (e) {
    console.log(`  FAIL  ${name}\n          ${e.message}`);
    failures.push(name);
  }
}
function eq(actual, expected, what) {
  const a = JSON.stringify(actual), b = JSON.stringify(expected);
  if (a !== b) throw new Error(`${what}: expected ${b}, got ${a}`);
}
function ok(cond, what) {
  if (!cond) throw new Error(what);
}

/* ---------- tests --------------------------------------------------------- */

console.log("\nESPN ingestion replay tests\n");

/* Bug #2: mDraftDetail returns every pick SLOT, not every pick. A 12x15 draft
   reports 180 rows from the moment the room opens. Counting rows counted the
   draft board, not the draft, and 1 of 36 picks registered. */
await test("180 slots with 24 made yields 24 picks, not 180", async () => {
  const { ctx } = boot({ detail: draftDetail({ made: 24 }) });
  const picks = await ctx.pollApi();
  eq(picks.length, 24, "pick count");
  eq(picks[0].pick, 1, "first overall pick number");
  eq(picks[23].pick, 24, "last overall pick number");
});

await test("an opened but unstarted room reports no picks", async () => {
  const { ctx } = boot({ detail: draftDetail({ made: 0 }) });
  await ctx.pollApi().then(
    () => { throw new Error("should have thrown"); },
    (e) => ok(/not started/.test(e.message), `wrong error: ${e.message}`),
  );
});

await test("picks resolve to names via the player map", async () => {
  const { ctx } = boot({ detail: draftDetail({ made: 5 }) });
  const picks = await ctx.pollApi();
  eq(picks.map((p) => p.name), ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"],
     "resolved names");
});

await test("an unmapped playerId is skipped, not crashed on", async () => {
  const { ctx } = boot({
    detail: draftDetail({ made: 5 }),
    players: playerList().filter((p) => p.id !== 1003),
  });
  const picks = await ctx.pollApi();
  eq(picks.length, 4, "picks surviving an unresolvable id");
  ok(!picks.some((p) => p.pick === 3), "pick 3 should be absent");
});

await test("re-polling the same state yields no duplicates", async () => {
  const { ctx } = boot({ detail: draftDetail({ made: 12 }) });
  eq((await ctx.pollApi()).length, 12, "first poll");
  eq((await ctx.pollApi()).length, 0, "second poll");
});

await test("only newly made picks are emitted as the draft advances", async () => {
  const detail = draftDetail({ made: 10 });
  const { ctx } = boot({ detail });
  eq((await ctx.pollApi()).length, 10, "initial");
  for (let i = 10; i < 13; i++) detail.draftDetail.picks[i].playerId = 1000 + i + 1;
  const next = await ctx.pollApi();
  eq(next.length, 3, "incremental");
  eq(next.map((p) => p.pick), [11, 12, 13], "incremental pick numbers");
});

/* Free from the payload: each pick carries teamId, so your roster fills itself
   with no Shift+Enter. */
await test("your own picks self-mark from teamId", async () => {
  const { ctx } = boot({ detail: draftDetail({ made: 24, myTeamId: 3 }) });
  const picks = await ctx.pollApi();
  const mine = picks.filter((p) => p.mine).map((p) => p.pick);
  eq(mine, [3, 22], "your picks in the first two rounds from slot 3");
});

await test("your draft slot auto-detects from round-1 roundPickNumber", async () => {
  const { ctx, messages } = boot({
    search: "?leagueId=123456&teamId=7",
    detail: draftDetail({ made: 24 }),
  });
  await ctx.pollApi();
  const slot = messages.filter((m) => m.type === "slot");
  eq(slot.length, 1, "one slot message");
  eq(slot[0].slot, 7, "detected slot");
  await ctx.pollApi();
  eq(messages.filter((m) => m.type === "slot").length, 1, "slot sent only once");
});

/* The opponent view is built entirely from data already in the payload:
   every pick carries teamId, and a snake seat is arithmetic from the pick
   number. If teamId ever stops riding along, opponent rosters go blank --
   which is why it is asserted rather than assumed. */
await test("every emitted pick carries its teamId", async () => {
  const { ctx } = boot({ detail: draftDetail({ made: 24, myTeamId: 3 }) });
  const picks = await ctx.pollApi();
  eq(picks.every((p) => typeof p.team === "number"), true, "team on every pick");
  const first = picks.find((p) => p.pick === 1);
  eq(first.team, 1, "pick 1 belongs to seat 1");
});

await test("my teamId is announced once, for excluding me from the opponent view",
  async () => {
    const { ctx, messages } = boot({
      search: "?leagueId=123456&teamId=7",
      detail: draftDetail({ made: 24 }),
    });
    await ctx.pollApi();
    await ctx.pollApi();
    const mt = messages.filter((m) => m.type === "myteam");
    eq(mt.length, 1, "sent exactly once");
    eq(mt[0].team, 7, "my teamId");
  });

await test("leagueId parses from query string and from path", async () => {
  eq(boot({ search: "?leagueId=987&teamId=1" }).ctx.leagueId(), "987", "query form");
  eq(boot({ pathname: "/football/draft/leagueId/555/", search: "" }).ctx.leagueId(),
     "555", "path form");
});

/* Bug #3: the mock-draft LOBBY listed rooms with clock times and member counts,
   the DOM scan read them as picks, and 154 phantom picks went into
   localStorage and corrupted every recommendation downstream. */
await test("the mock draft lobby is not a draft room", async () => {
  const { ctx } = boot({ pathname: "/football/mockdraftlobby", search: "" });
  eq(ctx.inDraftRoom(), false, "lobby must not be a draft room");
});

await test("the real draft room is a draft room", async () => {
  eq(boot({ pathname: "/football/draft" }).ctx.inDraftRoom(), true, "draft room");
});

await test("draftrecap is not mistaken for the live draft room", async () => {
  const { ctx } = boot({ pathname: "/football/draftrecap", search: "?leagueId=1" });
  eq(ctx.inDraftRoom(), false, "draftrecap is a different page");
});

/* These two must assert on the messages the panel actually receives, NOT on a
   second scanDom() call. The `ready` handler runs a tick, which scans and moves
   every match into `seen`; a later scanDom() then returns [] whether the first
   scan found nothing or found 154 phantom picks. Checking the second call is a
   test that cannot fail. */
const emitted = (messages) =>
  messages.filter((m) => m.type === "picks").flatMap((m) => m.picks.map((p) => p.name));

await test("lobby text produces zero picks even if the gate is bypassed", async () => {
  const { messages, fire } = boot({
    pathname: "/football/draft",     // gate deliberately open
    bodyText: LOBBY_TEXT,
    failApi: true,                    // force the DOM path
  });
  await fire({ __draft: 1, type: "ready", names: ["Player 1", "Player 2", "Player 3"] });
  eq(emitted(messages), [], "lobby room listings must not read as picks");
});

await test("real draft-room text still produces picks", async () => {
  const { messages, fire } = boot({
    bodyText: DRAFT_ROOM_TEXT,
    failApi: true,
  });
  await fire({ __draft: 1, type: "ready", names: ["Player 1", "Player 2", "Player 99"] });
  const found = emitted(messages);
  ok(found.includes("Player 1"), `R4, P11 token should match -- got ${JSON.stringify(found)}`);
  ok(found.includes("Player 2"), `Pick 38 token should match -- got ${JSON.stringify(found)}`);
  ok(!found.includes("Player 99"), "'on the clock' is not a pick");
});

await test("ingestion stays off outside a draft room", async () => {
  const { ctx, messages } = boot({
    pathname: "/football/mockdraftlobby",
    search: "",
    bodyText: LOBBY_TEXT,
  });
  await ctx.tick();
  eq(messages.filter((m) => m.type === "picks").length, 0, "no picks emitted");
  ok(messages.some((m) => m.type === "status" && /not a draft room/.test(m.text)),
     "should say why sync is off");
});

/* ---------- API resilience ------------------------------------------------ */

/* The bug that made the tool "fizzle out around round 11": `apiDead` was a
   one-way latch, so a single transient failure abandoned the API for the life
   of the page and left the panel on a DOM scanner that cannot see picks in the
   draft room at all. */
await test("a transient API failure does not kill the API path forever", async () => {
  const detail = draftDetail({ made: 20 });
  let calls = 0;
  const b = boot({ detail });
  const realFetch = b.ctx.fetch;
  b.ctx.fetch = async (url) => {
    if (url.includes("mDraftDetail") && ++calls === 2) throw new Error("HTTP 429");
    return realFetch(url);
  };
  await b.ctx.tick();                                   // succeeds, 20 picks
  eq(emitted(b.messages).length, 20, "first poll");
  await b.ctx.tick();                                   // fails -> backoff
  ok(b.messages.some((m) => m.type === "status" && /429/.test(m.text)),
     "the failure must be reported");
  for (let i = 0; i < 6; i++) await b.ctx.tick();       // ride out the backoff
  for (let i = 20; i < 26; i++) detail.draftDetail.picks[i].playerId = 1000 + i + 1;
  for (let i = 0; i < 8; i++) await b.ctx.tick();
  eq(emitted(b.messages).length, 26, "API must recover and catch every pick");
});

await test("picks missed during an outage are re-emitted on recovery", async () => {
  const detail = draftDetail({ made: 5 });
  let fail = false;
  const b = boot({ detail });
  const realFetch = b.ctx.fetch;
  b.ctx.fetch = async (url) => {
    if (fail && url.includes("mDraftDetail")) throw new Error("HTTP 503");
    return realFetch(url);
  };
  await b.ctx.tick();
  eq(emitted(b.messages).length, 5, "before the outage");
  fail = true;
  for (let i = 5; i < 15; i++) detail.draftDetail.picks[i].playerId = 1000 + i + 1;
  await b.ctx.tick();
  eq(emitted(b.messages).length, 5, "nothing arrives while the API is down");
  fail = false;
  for (let i = 0; i < 10; i++) await b.ctx.tick();
  eq(emitted(b.messages).length, 15, "the gap is repaired, not skipped");
});

await test("an unstarted draft is a state, not a failure to back off from", async () => {
  const b = boot({ detail: draftDetail({ made: 0 }) });
  await b.ctx.tick();
  ok(b.messages.some((m) => m.type === "status" && /waiting for the first pick/.test(m.text)),
     "should report waiting, not an API failure");
  ok(!b.messages.some((m) => m.type === "status" && /retry/.test(m.text)),
     "must not enter backoff for an empty draft");
});

/* The panel derived the current pick as `state.picks.length + 1`, which counts
   only picks it could match on a 235-player board -- so every kicker, defence
   or deep bench player drafted pushed it one pick further behind ESPN. */
await test("the authoritative pick number comes from ESPN, not from our count", async () => {
  const { ctx, messages } = boot({ detail: draftDetail({ made: 123 }) });
  await ctx.pollApi();
  const no = messages.filter((m) => m.type === "pickno");
  ok(no.length >= 1, "a pickno message must be sent");
  eq(no.at(-1).pick, 124, "123 picks made means pick 124 is on the clock");
});

await test("the pick number ignores players the board cannot match", async () => {
  // 20 made, but 6 of them resolve to nobody on the board -- ESPN is still at 21.
  const { ctx, messages } = boot({
    detail: draftDetail({ made: 20 }),
    players: playerList().filter((p) => ![1003, 1007, 1009, 1012, 1015, 1018].includes(p.id)),
  });
  const picks = await ctx.pollApi();
  eq(picks.length, 14, "only matchable picks are emitted");
  eq(messages.filter((m) => m.type === "pickno").at(-1).pick, 21,
     "but the pick number must still be 21, not 15");
});

/* ---------- panel state schema ------------------------------------------- */

/* v1.4 logged 154 phantom picks off the lobby. v1.5 fixed the ingestion, but
   the bad picks were already in localStorage and neither reloading the
   extension nor reloading the page clears it -- so v1.5 opened on a poisoned
   board showing "pick 157" and no top-50 players. A schema stamp makes the
   upgrade itself throw the bad state away. */
const APP_JS = join(ROOT, "output", "espn_extension", "app.js");

function bootPanel(stored) {
  const anyEl = () =>
    new Proxy(function () {}, {
      get: (_t, k) =>
        k === "style" ? {}
        : k === "classList" ? { toggle() {}, add() {}, remove() {} }
        : k === "dataset" ? {}
        : k === "value" ? ""
        : anyEl(),
      set: () => true,
      apply: () => anyEl(),
    });
  const store = new Map(stored ? [["draftday2026", JSON.stringify(stored)]] : []);
  const sandbox = {
    console,
    localStorage: {
      getItem: (k) => store.get(k) ?? null,
      setItem: (k, v) => store.set(k, v),
      removeItem: (k) => store.delete(k),
    },
    document: {
      getElementById: anyEl,
      createElement: anyEl,
      querySelectorAll: () => [],
      addEventListener() {},
      body: anyEl(),
    },
    window: { addEventListener() {}, scrollTo() {}, parent: { postMessage() {} } },
    fetch: () => ({ catch() {} }),
    confirm: () => true,
    setTimeout: () => 0,
  };
  const posted = [];
  sandbox.fetch = (_url, opts) => {
    try { posted.push(JSON.parse(opts.body)); } catch {}
    return { catch() {} };
  };
  sandbox.globalThis = sandbox;
  const ctx = vm.createContext(sandbox);
  vm.runInContext(readFileSync(APP_JS, "utf8"), ctx, { filename: "app.js" });
  const read = (expr) => vm.runInContext(expr, ctx);
  return {
    ctx, posted, read,
    picks: read("state.picks.length"),
    stale: read("STALE"),
    stored: store.get("draftday2026"),
  };
}

/* Regression for a bug introduced BY the pick-number fix: once the API sets
   `apiPick`, curPick() stops reading state.picks.length, so a manual take had
   to advance the counter itself or it would freeze the moment the feed died. */
await test("a manual take advances the pick number when the feed has stopped", () => {
  const b = bootPanel(null);
  b.read("window.DRAFTAPP.setPickNo(124)");
  eq(b.read("curPick()"), 124, "starts at the API's number");
  b.read("window.DRAFTAPP.take('Puka Nacua', false, false)");
  eq(b.read("curPick()"), 125, "a manual pick must move it");
  b.read("window.DRAFTAPP.undo()");
  eq(b.read("curPick()"), 124, "undo must move it back");
});

await test("an API pick does not double-advance the counter", () => {
  const b = bootPanel(null);
  b.read("window.DRAFTAPP.setPickNo(50)");
  b.read("window.DRAFTAPP.take('Puka Nacua', false, true)");   // auto = from API
  eq(b.read("curPick()"), 50, "the feed owns the numbering for its own picks");
});

await test("the state mirror reports the same pick number as the panel", () => {
  const b = bootPanel(null);
  b.read("window.DRAFTAPP.setPickNo(124)");
  const last = b.posted.at(-1);
  ok(last, "mirror must post");
  eq(last.pick, 124, "draft_state.json must not disagree with the panel");
});

await test("no computed pick falls outside a 15-round draft", () => {
  const b = bootPanel(null);
  for (let slot = 1; slot <= 12; slot++) {
    const picks = b.read(`myPicks(${slot})`);
    eq(picks.length, 15, `slot ${slot} pick count`);
    ok(Math.max(...picks) <= 180,
       `slot ${slot} produced pick ${Math.max(...picks)} in a 180-pick draft`);
    ok(new Set(picks).size === picks.length, `slot ${slot} has duplicate picks`);
  }
});

await test("state from an older schema is discarded, not loaded", () => {
  const r = bootPanel({ picks: Array.from({ length: 156 }, (_, i) => ({ name: `x${i}` })), slot: 2 });
  eq(r.picks, 0, "phantom picks must not survive the upgrade");
  eq(r.stale, 156, "and the panel must say how many it threw away");
});

/* Read SCHEMA out of app.js rather than hardcoding it. A literal here turns
   every legitimate schema bump into a spurious test failure, which trains you
   to "fix" the test -- and the one time the bump was wrong you would sail
   past it. */
const CURRENT_SCHEMA = Number(
  (readFileSync(APP_JS, "utf8").match(/const SCHEMA = (\d+)/) || [])[1]);

await test("state from the current schema still loads", () => {
  ok(Number.isFinite(CURRENT_SCHEMA), "SCHEMA must be readable from app.js");
  const r = bootPanel({ picks: [{ name: "Bijan Robinson" }], slot: 4,
                        schema: CURRENT_SCHEMA });
  eq(r.picks, 1, "good state must survive a reload");
  eq(r.stale, 0, "nothing discarded");
});

await test("a first run with no saved state is clean", () => {
  const r = bootPanel(null);
  eq(r.picks, 0, "no picks");
  eq(r.stale, 0, "nothing to discard");
});

await test("switching platform swaps adp/sig/shift and re-sorts the board", () => {
  const b = bootPanel(null);
  const first = (id) => b.read(`(window.DRAFTAPP.setPlatform(${JSON.stringify(id)}),
    BOARD[0].name + "|" + BOARD[0].adp)`);
  const espn = b.read("PLATFORMS.find(m => m.id === 'espn' && m.ready) ? 1 : 0");
  const sleeper = b.read("PLATFORMS.find(m => m.id === 'sleeper' && m.ready) ? 1 : 0");
  ok(espn && sleeper, "espn and sleeper must be ready in the committed data");
  const a = first("espn"), c = first("sleeper");
  ok(a !== c, "different markets must price the top player differently");
  eq(b.read("BOARD.every((p, i) => i === 0 || p.adp == null "
          + "|| BOARD[i-1].adp == null || BOARD[i-1].adp <= p.adp)"),
     true, "board must be sorted by the active platform's ADP");
});

await test("an import platform with no CSV is disabled and cannot be selected", () => {
  const b = bootPanel(null);
  const before = b.read("state.platform");
  b.read("PLATFORMS.filter(m => !m.ready).forEach(m => window.DRAFTAPP.setPlatform(m.id))");
  eq(b.read("state.platform"), before, "an unready tab must be ignored");
});

await test("shortlist odds are 1 - prod(1 - p) over the live targets", () => {
  const b = bootPanel(null);
  b.read("window.DRAFTAPP.setSlot(5); window.DRAFTAPP.setPickNo(1)");
  b.read("window.DRAFTAPP.toggleTarget('Jonathan Taylor')");
  b.read("window.DRAFTAPP.toggleTarget('Drake London')");
  const q = b.read("window.DRAFTAPP.shortlist(curPick(), horizon()).q");
  const want = b.read(`(() => { const n = horizon(), now = curPick();
    const f = nm => availAt(byName[nm], n, now).v;
    return 1 - (1 - f('Jonathan Taylor')) * (1 - f('Drake London')); })()`);
  ok(Math.abs(q - want) < 1e-12, `got ${q}, want ${want}`);
  ok(q > b.read("Math.max(availAt(byName['Jonathan Taylor'], horizon(), curPick()).v, "
              + "availAt(byName['Drake London'], horizon(), curPick()).v) - 1e-12"),
     "at-least-one must be at least as likely as the likeliest single target");
});

await test("a target that has been taken drops out of the shortlist", () => {
  const b = bootPanel(null);
  b.read("window.DRAFTAPP.setSlot(5); window.DRAFTAPP.setPickNo(1)");
  b.read("window.DRAFTAPP.toggleTarget('Jonathan Taylor')");
  b.read("window.DRAFTAPP.toggleTarget('Drake London')");
  b.read("window.DRAFTAPP.take('Jonathan Taylor', false, false)");
  eq(b.read("window.DRAFTAPP.shortlist(curPick(), horizon()).items.length"), 1,
     "only the live target remains");
});

await test("targets persist, and state saved before targets existed loads clean", () => {
  const b = bootPanel(null);
  b.read("window.DRAFTAPP.toggleTarget('Drake London')");
  const saved = JSON.parse(b.read("localStorage.getItem('draftday2026')"));
  eq(saved.targets, ["Drake London"], "targets are saved");
  const old = bootPanel({ picks: [], slot: 3, schema: CURRENT_SCHEMA });
  eq(old.read("state.targets.length"), 0, "missing targets default to none");
});

const total = pass + failures.length;
console.log(`\n${pass}/${total} passed`);
if (failures.length) {
  console.log(`failed: ${failures.join(", ")}`);
  process.exit(1);
}
