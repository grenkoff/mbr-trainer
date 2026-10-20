"use strict";

const POSITIONS = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"];
const RANKS = "AKQJT98765432";
const SUIT_SYMBOL = { s: "♠", h: "♥", d: "♦", c: "♣" };
const DIFFICULTY_HINT = {
  natural: "Карты раздаются равномерно, как за столом: все 1326 комбинаций с равной вероятностью.",
  easy: "Ошибка стоит больше 0.10 BI — очевидные решения.",
  medium: "Ошибка стоит 0.03–0.10 BI.",
  hard: "Ошибка стоит меньше 0.03 BI — руки у границы диапазона.",
  mix: "Случайная сложность для каждой раздачи.",
};
const DIFF_NAME = { natural: "Как в игре", easy: "Легко", medium: "Средне", hard: "Сложно" };
const PLAYERS_NAME = Object.fromEntries([3, 4, 5, 6, 7, 8, 9].map((n) => [n, `${n} ${n < 5 ? "игрока" : "игроков"}`]));
const KIND_NAME = { open: "Опен-пуш", call: "Колл пуша", overcall: "Оверколл" };
const CLOSE_EV = 0.005; // decisions this close in EV count as correct either way
const STATS_KEY = "mbr-trainer-stats-v1";
const SETTINGS_KEY = "mbr-trainer-settings-v1";

const $ = (id) => document.getElementById(id);

function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (e) {
    return fallback;
  }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* storage unavailable */ }
}

// player counts the trainer offers for now: the HRC model is solved for 3 players first
const ENABLED_PLAYERS = [3];
// models served from the library solved in advance (mbr.library); they always have overcalls
const LIBRARY_MODELS = ["hrc"];
const settings = Object.assign(
  { players: [3], position: "any", kind: "any", bbMin: 5, bbMax: 15, difficulty: "natural", model: "hrc", overcalls: true, stacks: "history" },
  load(SETTINGS_KEY, {}),
);
if (!Array.isArray(settings.players)) settings.players = [+settings.players || 3]; // saved before multi-select
settings.players = settings.players.filter((n) => ENABLED_PLAYERS.includes(n));
if (!settings.players.length) settings.players = [ENABLED_PLAYERS[0]];
let stats = load(STATS_KEY, []);
const history = []; // this session: {spot, choice, correct, cost}

let current = null;   // spot being shown
let answered = null;  // {choice, correct, cost} once answered
let gridMode = "freq";
let prefetch = null;  // {key, promise}
let spotSeq = 0;      // id of the latest nextSpot() call
let overcallsAvailable = true;
let historyCounts = {}; // players -> final-table hands in the user's hand histories
let libraryCounts = {}; // model -> players -> of those, solved in advance

const fromLibrary = () => LIBRARY_MODELS.includes(settings.model);
/** players -> history hands a spot can come from with the current model */
function usableHistory() {
  return fromLibrary() ? (libraryCounts[settings.model] || {}) : historyCounts;
}

/* ---------------- settings UI ---------------- */

function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  return m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14) ? few : many;
}
const handsWord = (n) => `${n} ${plural(n, "раздача", "раздачи", "раздач")}`;

function positionsFor(n) { return POSITIONS.slice(-n); }

const maxPlayers = () => Math.max(...settings.players);
const playersText = () => [...settings.players].sort((a, b) => a - b).join(", ");

/** Can Hero sit in the chosen position facing `kind` with n players? */
function fits(n, kind) {
  const pos = positionsFor(n);
  const idx = pos.indexOf(settings.position);
  if (settings.position !== "any" && idx === -1) return false;
  if (kind === "open") return idx !== n - 1;
  if (kind === "call") return idx !== 0;
  if (kind === "overcall") return settings.overcalls && (idx === -1 || idx >= 2);
  return true;
}

/** The selected player counts a spot can be dealt with under the current settings. */
function playableCounts() {
  return settings.players.filter((n) => fits(n, settings.kind) &&
    (settings.stacks !== "history" || usableHistory()[n]));
}

function bindSeg(id, getter, setter) {
  const el = $(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    setter(b.dataset.v);
    onSettingsChanged();
  });
  return () => {
    for (const b of el.querySelectorAll("button")) b.classList.toggle("active", b.dataset.v === String(getter()));
  };
}

function renderPositionButtons() {
  const el = $("setPosition");
  const opts = ["any", ...positionsFor(maxPlayers())];
  if (!opts.includes(settings.position)) settings.position = "any";
  el.innerHTML = opts.map((p) => `<button data-v="${p}">${p === "any" ? "Любая" : p}</button>`).join("");
}

function kindAllowed(kind) {
  return settings.players.some((n) => fits(n, kind));
}

function updateAvailability() {
  if (!overcallsAvailable) settings.overcalls = false;
  if (fromLibrary() && overcallsAvailable) settings.overcalls = true;
  const anyHistory = settings.players.some((n) => usableHistory()[n]);
  if (!anyHistory) settings.stacks = "random";
  $("setStacks").querySelector('[data-v="history"]').disabled = !anyHistory;
  for (const b of document.querySelectorAll("#setKind button, #setKind2 button")) b.disabled = !kindAllowed(b.dataset.v);
  if (!kindAllowed(settings.kind)) settings.kind = "any";
}

const refreshers = [];
function setupSettings() {
  for (const b of $("setPlayers").querySelectorAll("button")) {
    if (!ENABLED_PLAYERS.includes(+b.dataset.v)) {
      b.disabled = true;
      b.title = "Пока недоступно";
    }
  }
  $("playersHint").textContent = "Пока доступны только 3 игрока: остальные откроются позже.";
  renderPositionButtons();
  // several player counts can be on at once; each spot gets one of them at random
  $("setPlayers").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    const n = +b.dataset.v;
    if (settings.players.includes(n)) {
      if (settings.players.length === 1) return; // keep at least one
      settings.players = settings.players.filter((x) => x !== n);
    } else {
      settings.players = [...settings.players, n].sort((a, b) => a - b);
    }
    renderPositionButtons();
    onSettingsChanged();
  });
  refreshers.push(() => {
    for (const b of $("setPlayers").querySelectorAll("button")) b.classList.toggle("active", settings.players.includes(+b.dataset.v));
  });
  refreshers.push(bindSeg("setPosition", () => settings.position, (v) => { settings.position = v; }));
  refreshers.push(bindSeg("setKind", () => settings.kind, (v) => { settings.kind = v; }));
  refreshers.push(bindSeg("setKind2", () => settings.kind, (v) => { settings.kind = v; }));
  $("setOvercalls").addEventListener("change", (e) => { settings.overcalls = e.target.checked; onSettingsChanged(); });
  refreshers.push(bindSeg("setStacks", () => settings.stacks, (v) => { settings.stacks = v; }));
  refreshers.push(bindSeg("setDifficulty0", () => settings.difficulty, (v) => { settings.difficulty = v; }));
  refreshers.push(bindSeg("setDifficulty", () => settings.difficulty, (v) => { settings.difficulty = v; }));
  refreshers.push(bindSeg("setStackPreset", () => `${settings.bbMin}-${settings.bbMax}`, (v) => {
    const [a, b] = v.split("-").map(Number);
    settings.bbMin = a; settings.bbMax = b;
  }));
  $("setModel").addEventListener("change", (e) => { settings.model = e.target.value; onSettingsChanged(); });
  for (const id of ["bbMin", "bbMax"]) {
    $(id).addEventListener("change", () => {
      let a = parseFloat($("bbMin").value) || 1, b = parseFloat($("bbMax").value) || a;
      a = Math.max(1, Math.min(60, a)); b = Math.max(a, Math.min(60, b));
      settings.bbMin = a; settings.bbMax = b;
      onSettingsChanged();
    });
  }
  refreshSettingsUI();
}

function refreshSettingsUI() {
  updateAvailability();
  refreshers.forEach((r) => r());
  $("bbMin").value = settings.bbMin;
  $("bbMax").value = settings.bbMax;
  $("setModel").value = settings.model;
  $("difficultyHint").textContent = DIFFICULTY_HINT[settings.difficulty];
  const hands = settings.players.reduce((sum, n) => sum + (historyCounts[n] || 0), 0);
  const solved = settings.players.reduce((sum, n) => sum + (usableHistory()[n] || 0), 0);
  const who = `${playersText()} ${plural(maxPlayers(), "игрока", "игроков", "игроков")}`;
  $("stacksHint").textContent = !Object.keys(historyCounts).length
    ? "Нет истории рук в hh/hands."
    : settings.stacks === "history"
      ? `Блайнды и стеки ваших раздач финального стола: ${handsWord(hands)} на ${who}` +
        (fromLibrary() ? `, решено заранее ${solved}. ` : ". ") +
        "Вы можете сидеть на любом месте, которое подходит под позицию и стек."
      : (hands ? `Случайный уровень и стеки. В истории ${handsWord(hands)} на ${who}.`
        : `Случайный уровень и стеки. В истории нет раздач на ${who}.`) +
        (fromLibrary() ? " Каждый спот считается заново, это до минуты." : "");
  $("modelHint").textContent = settings.model === "hrc"
    ? "Как HoldemResources Calculator: пуш/фолд, до трёх игроков в олл-ине, модель FGS глубины 1 " +
      "(исход раздачи оценивается по следующей раздаче, потом ICM), баунти 0,945 BI, " +
      "карты сфолдивших не учитываются."
    : "";
  $("setOvercalls").checked = settings.overcalls;
  $("setOvercalls").disabled = !overcallsAvailable || fromLibrary();
  $("overcallsToggle").classList.toggle("disabled", !overcallsAvailable || fromLibrary());
  $("overcallsHint").textContent = !overcallsAvailable
    ? "Нет таблицы эквити на троих (build/equity3.f32), см. README."
    : fromLibrary() ? "В этой модели всегда до трёх игроков в олл-ине."
    : settings.overcalls
      ? "После пуша и колла игроки сзади могут оверколлить, до трёх игроков в олл-ине."
      : "Пуш получает максимум один колл.";
}

function onSettingsChanged() {
  refreshSettingsUI();
  save(SETTINGS_KEY, settings);
  prefetch = null;
  if (!answered && current) nextSpot(); // replace an unanswered spot with one matching the new settings
}

/* ---------------- fetching ---------------- */

function spotParams() {
  const diff = settings.difficulty === "mix"
    ? ["easy", "medium", "hard"][Math.floor(Math.random() * 3)]
    : settings.difficulty;
  const counts = playableCounts();
  const players = counts.length ? counts[Math.floor(Math.random() * counts.length)] : settings.players[0];
  return new URLSearchParams({
    players, position: settings.position, kind: settings.kind,
    bb_min: settings.bbMin, bb_max: settings.bbMax, difficulty: diff, model: settings.model,
    overcalls: settings.overcalls ? 1 : 0, stacks: settings.stacks,
  });
}

function settingsKey() { return JSON.stringify(settings); }

async function fetchSpot() {
  const res = await fetch(`/api/spot?${spotParams()}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function startPrefetch() {
  const key = settingsKey();
  const promise = fetchSpot();
  promise.catch(() => {});
  prefetch = { key, promise };
}

async function nextSpot() {
  $("loadingHint").textContent = settings.overcalls && maxPlayers() >= 6
    ? "С оверколлами на 6–9 игроках первый спот считается до минуты, дальше споты берутся из запаса. " +
      "Запас можно заполнить заранее: python -m trainer.fill"
    : "";
  $("loading").classList.remove("hidden");
  const seq = ++spotSeq;
  try {
    let spot, key;
    do { // settings may change while a spot is on its way: then fetch one for the new settings
      key = settingsKey();
      spot = prefetch && prefetch.key === key ? await prefetch.promise : await fetchSpot();
    } while (key !== settingsKey());
    if (seq !== spotSeq) return; // a newer request shows its own spot
    prefetch = null;
    showSpot(spot, null);
    startPrefetch();
  } catch (e) {
    if (seq === spotSeq) $("question").innerHTML = `<span style="color:var(--bad)">Ошибка: ${e.message}</span>`;
  } finally {
    if (seq === spotSeq) $("loading").classList.add("hidden");
  }
}

/* ---------------- table rendering ---------------- */

function fmtBB(x) { return (Math.round(x * 10) / 10).toString(); }

function cardHTML(card) {
  const r = card[0], s = card[1];
  return `<div class="card s-${s}"><span>${r}</span><span class="s">${SUIT_SYMBOL[s]}</span></div>`;
}

function seatPoint(rel, n, radius) {
  const angle = (90 + rel * 360 / n) * Math.PI / 180;
  return { x: 50 + radius * 50 * Math.cos(angle), y: 50 + radius * 50 * Math.sin(angle) };
}

function renderTable(spot) {
  const felt = $("felt");
  felt.querySelectorAll(".seat, .chips, .dealer").forEach((el) => el.remove());
  const n = spot.seats.length;
  const bb = spot.bb;
  const anteBB = spot.ante / bb;
  let pot = anteBB * n;

  spot.seats.forEach((seat, i) => {
    const rel = (i - spot.hero + n) % n;
    const p = seatPoint(rel, n, 1.08);
    const act = i < spot.hero ? spot.actions[i] : null;
    const el = document.createElement("div");
    el.className = "seat" + (seat.hero ? " hero" : "") + (act && act.action === "fold" ? " folded" : "");
    el.style.left = `${p.x}%`;
    el.style.top = `${p.y}%`;
    let tag = "";
    if (!seat.hero) {
      tag = seat.hero_covers
        ? `<span class="tag cover" title="Вы покрываете — баунти ваш при нокауте">покрываете</span>`
        : `<span class="tag covered" title="Этот игрок покрывает вас">покрывает вас</span>`;
    }
    el.innerHTML =
      (seat.hero ? `<div class="hero-cards">${spot.cards.map(cardHTML).join("")}</div>` : "") +
      `<div class="pos">${seat.position}${seat.hero ? " · вы" : ""}</div>` +
      `<div class="stack">${fmtBB(seat.stack_bb)} bb</div>` + tag;
    if (act) {
      const a = document.createElement("div");
      a.className = `act ${act.action}`;
      a.textContent = { fold: "фолд", allin: "ALL-IN", call: act.allin ? "КОЛЛ ALL-IN" : "КОЛЛ" }[act.action];
      a.style[p.y < 50 ? "bottom" : "top"] = "-20px";
      el.appendChild(a);
    }
    felt.appendChild(el);

    // posted blinds / shove chips
    let chips = 0, cls = "chips";
    if (seat.position === "SB") chips = spot.sb / bb;
    if (seat.position === "BB") chips = 1;
    if (act && act.action === "allin") { chips = seat.stack_bb - anteBB; cls += " allin"; }
    else if (act && act.action === "call") { chips = act.amount_bb - anteBB; cls += " call"; }
    else pot += chips;
    if (chips > 0) {
      const c = seatPoint(rel, n, rel === 0 ? 0.45 : 0.62);
      const ch = document.createElement("div");
      ch.className = cls;
      ch.style.left = `${c.x}%`;
      ch.style.top = `${c.y}%`;
      ch.textContent = `${fmtBB(chips)} bb`;
      felt.appendChild(ch);
    }
    if (seat.position === "BTN") {
      // beside the seat box on the side facing the middle of the table, clear of the action labels
      const de = document.createElement("div");
      de.className = "dealer";
      de.style.left = `calc(${p.x}% ${p.x < 50 ? "+" : "-"} ${el.offsetWidth / 2 + 14}px)`;
      de.style.top = `calc(${p.y}% - 12px)`;
      de.textContent = "D";
      felt.appendChild(de);
    }
  });
  $("pot").textContent = `Банк ${fmtBB(pot)} bb`;
  $("levelInfo").textContent = `Уровень ${spot.level}: ${spot.sb}/${spot.bb}, анте ${spot.ante}` +
    (spot.settings.overcalls ? " · с оверколлами" : "") +
    (spot.source && spot.source.hand_id ? ` · раздача #${spot.source.hand_id}` : "");
}

function renderQuestion(spot) {
  const hero = spot.seats[spot.hero];
  if (spot.kind === "open") {
    $("question").innerHTML = `Вы на <b>${hero.position}</b>, до вас все сфолдили. Пуш или фолд?`;
    $("btnPlay").firstChild.textContent = "Пуш ";
  } else if (spot.kind === "call") {
    const sh = spot.seats[spot.shover];
    const bounty = sh.hero_covers ? " Вы покрываете — на кону баунти." : " Он покрывает вас.";
    $("question").innerHTML = `<b>${sh.position}</b> пушит ${fmtBB(sh.stack_bb)} bb.${bounty} Колл или фолд?`;
    $("btnPlay").firstChild.textContent = "Колл ";
  } else {
    const sh = spot.seats[spot.shover], ca = spot.seats[spot.caller];
    const covered = [sh, ca].filter((x) => x.hero_covers).map((x) => x.position);
    const bounty = covered.length === 2 ? " Вы покрываете обоих — на кону два баунти."
      : covered.length === 1 ? ` Вы покрываете ${covered[0]}.` : " Оба покрывают вас.";
    $("question").innerHTML =
      `<b>${sh.position}</b> пушит ${fmtBB(sh.stack_bb)} bb, <b>${ca.position}</b> коллирует.${bounty} Оверколл или фолд?`;
    $("btnPlay").firstChild.textContent = "Колл ";
  }
}

/* ---------------- answering ---------------- */

function showSpot(spot, ans) {
  current = spot;
  answered = ans;
  renderTable(spot);
  renderQuestion(spot);
  $("btnFold").classList.toggle("chosen", !!ans && ans.choice === "fold");
  $("btnPlay").classList.toggle("chosen", !!ans && ans.choice === "play");
  $("btnFold").disabled = $("btnPlay").disabled = !!ans;
  $("btnNext").classList.toggle("hidden", !ans);
  if (ans) renderResult(); else $("result").classList.add("hidden");
}

function answer(choice) {
  if (!current || answered) return;
  const a = current.answer;
  const d = a.ev_diff[a.hand_class];
  const best = d > 0 ? "play" : "fold";
  const cost = choice === best ? 0 : Math.abs(d);
  const correct = choice === best || Math.abs(d) < CLOSE_EV;
  answered = { choice, correct, cost };
  const record = {
    ts: Date.now(), kind: current.kind, players: current.seats.length,
    position: current.seats[current.hero].position, difficulty: current.settings.difficulty,
    model: current.settings.model, overcalls: !!current.settings.overcalls, stacks: current.settings.stacks || "random",
    hand: current.hand, correct, cost,
  };
  stats.push(record);
  if (stats.length > 5000) stats = stats.slice(-5000);
  save(STATS_KEY, stats);
  history.unshift({ spot: current, ...answered });
  showSpot(current, answered);
  renderStats();
  renderHistory();
}

function renderResult() {
  const spot = current, a = spot.answer, ans = answered;
  const d = a.ev_diff[a.hand_class];
  const playWord = spot.kind === "open" ? "пуш" : "колл";
  const v = $("verdict");
  if (ans.choice === (d > 0 ? "play" : "fold")) {
    v.className = "verdict good";
    v.innerHTML = `Верно<small>${spot.hand}: ${d > 0 ? playWord : "фолд"}</small>`;
  } else if (ans.correct) {
    v.className = "verdict close";
    v.innerHTML = `Почти без разницы<small>Разница ${Math.abs(d).toFixed(3)} BI</small>`;
  } else {
    v.className = "verdict bad";
    v.innerHTML = `Ошибка — лучше ${d > 0 ? playWord : "фолд"}<small>Потеря ${ans.cost.toFixed(3)} BI</small>`;
  }
  const sign = (x) => (x >= 0 ? "+" : "") + x.toFixed(3);
  $("evTable").innerHTML = `
    <span>EV ${playWord}а</span><span>${a.ev_play.toFixed(3)} BI</span>
    <span>EV фолда</span><span>${a.ev_fold.toFixed(3)} BI</span>
    <span>Разница</span><span>${sign(d)} BI</span>
    <span>Солвер</span><span>${playWord} ${Math.round(a.solver_freq * 100)}%</span>`;
  const hero = spot.seats[spot.hero].position;
  const vs = spot.kind === "open" ? "" : spot.seats[spot.shover].position;
  $("gridTitle").innerHTML = {
    open: `Диапазон пуша <b>${hero}</b>`,
    call: `Колл <b>${hero}</b> против пуша <b>${vs}</b>`,
    overcall: `Оверколл <b>${hero}</b>: пуш <b>${vs}</b>, колл <b>${spot.seats[spot.caller]?.position}</b>`,
  }[spot.kind] + ` — ${a.range_pct.toFixed(1)}%`;
  renderGrid();
  $("result").classList.remove("hidden");
}

function classLabel(i) {
  const row = Math.floor(i / 13), col = i % 13;
  if (row === col) return RANKS[row] + RANKS[row];
  return row < col ? RANKS[row] + RANKS[col] + "s" : RANKS[col] + RANKS[row] + "o";
}

function renderGrid() {
  const a = current.answer;
  const playWord = current.kind === "open" ? "пуш" : "колл";
  const cells = [];
  for (let i = 0; i < 169; i++) {
    const f = a.freq[i], d = a.ev_diff[i];
    let bg;
    if (gridMode === "freq") {
      bg = `linear-gradient(90deg, rgb(var(--play)) ${f * 100}%, rgb(var(--fold)) ${f * 100}%)`;
    } else {
      const k = Math.min(Math.abs(d) / 0.3, 1);
      bg = d > 0 ? `rgba(63,191,127,${0.15 + 0.85 * k})` : `rgba(229,83,75,${0.15 + 0.85 * k})`;
    }
    const hero = i === a.hand_class ? " hero" : "";
    cells.push(`<div class="cell${hero}" data-i="${i}" style="background:${bg}">${classLabel(i)}</div>`);
  }
  const grid = $("grid");
  grid.innerHTML = cells.join("");
  grid.onmouseover = (e) => {
    const c = e.target.closest(".cell");
    if (!c) return;
    const i = +c.dataset.i;
    const d = a.ev_diff[i];
    $("gridTip").textContent =
      `${classLabel(i)}: ${playWord} ${Math.round(a.freq[i] * 100)}% · EV ${playWord}а − фолда ${(d >= 0 ? "+" : "") + d.toFixed(3)} BI`;
  };
  grid.onmouseleave = () => { $("gridTip").innerHTML = "&nbsp;"; };
}

/* ---------------- stats & history ---------------- */

function summarize(rows) {
  const n = rows.length;
  const ok = rows.filter((r) => r.correct).length;
  const loss = rows.reduce((s, r) => s + r.cost, 0);
  return { n, ok, pct: n ? (100 * ok / n) : 0, loss, avg: n ? loss / n : 0 };
}

function renderStats() {
  const all = summarize(stats);
  $("statsInline").innerHTML = all.n
    ? `Раздач <b>${all.n}</b> · верно <b>${all.pct.toFixed(0)}%</b> · потеря <b>${all.loss.toFixed(2)} BI</b>`
    : "Статистика появится после первой раздачи";
  const row = (label, s) =>
    `<div class="stat-row"><span>${label}</span><span>${s.n ? `${s.pct.toFixed(0)}% · ${s.n}` : "—"}</span></div>`;
  const group = (title, key, names, skipEmpty = false) => {
    const rows = Object.keys(names)
      .map((k) => [names[k], summarize(stats.filter((r) => String(r[key]) === k))])
      .filter(([, s]) => s.n || !skipEmpty);
    return `<div class="stat-group"><div class="title">${title}</div>` +
      rows.map(([label, s]) => row(label, s)).join("") + `</div>`;
  };
  const sess = summarize(history);
  $("statsBox").innerHTML = `
    <div class="stat-row"><span>Эта сессия</span><span>${sess.n ? `${sess.pct.toFixed(0)}% · ${sess.n}` : "—"}</span></div>
    <div class="stat-row"><span>Всего раздач</span><span>${all.n}</span></div>
    <div class="stat-row"><span>Верно</span><span>${all.pct.toFixed(1)}%</span></div>
    <div class="stat-row"><span>Потеря EV</span><span>${all.loss.toFixed(2)} BI</span></div>
    <div class="stat-row"><span>На раздачу</span><span>${(all.avg * 1000).toFixed(1)} mBI</span></div>
    ${group("По типу решения", "kind", KIND_NAME)}
    ${group("По сложности", "difficulty", DIFF_NAME)}
    ${group("По числу игроков", "players", PLAYERS_NAME, true)}`;
}

function renderHistory() {
  $("history").innerHTML = history.slice(0, 50).map((h, idx) => {
    const s = h.spot;
    const pos = s.seats[s.hero].position;
    const what = s.kind === "open" ? "опен"
      : s.kind === "call" ? `vs ${s.seats[s.shover].position}`
      : `vs ${s.seats[s.shover].position}+${s.seats[s.caller].position}`;
    return `<li data-idx="${idx}"><span class="${h.correct ? "ok" : "err"}">${h.correct ? "✔" : "✘"}</span>` +
      `<b>${s.hand}</b><span>${s.seats.length}p ${pos} ${what} ${fmtBB(s.seats[s.hero].stack_bb)}bb</span>` +
      `<span class="loss">${h.cost ? "−" + h.cost.toFixed(2) : ""}</span></li>`;
  }).join("");
}

$("history").addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  const h = history[+li.dataset.idx];
  showSpot(h.spot, { choice: h.choice, correct: h.correct, cost: h.cost });
});

$("btnReset").addEventListener("click", () => {
  stats = [];
  save(STATS_KEY, stats);
  renderStats();
});

/* ---------------- controls ---------------- */

$("btnFold").addEventListener("click", () => answer("fold"));
$("btnPlay").addEventListener("click", () => answer("play"));
$("btnNext").addEventListener("click", () => nextSpot());
$("gridMode").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  gridMode = b.dataset.v;
  for (const x of $("gridMode").querySelectorAll("button")) x.classList.toggle("active", x === b);
  if (answered) renderGrid();
});

document.addEventListener("keydown", (e) => {
  if ((e.target instanceof Element && e.target.matches("input, select")) || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "f" || k === "а") answer("fold");
  else if (k === "a" || k === "c" || k === "ф" || k === "с") answer("play");
  else if ((k === " " || k === "enter") && answered) { e.preventDefault(); nextSpot(); }
});

async function init() {
  try {
    const info = await (await fetch("/api/info")).json();
    overcallsAvailable = !!info.overcalls;
    historyCounts = info.history || {};
    libraryCounts = info.library || {};
  } catch (e) {
    overcallsAvailable = false;
  }
  setupSettings();
  renderStats();
  nextSpot();
}

init();
