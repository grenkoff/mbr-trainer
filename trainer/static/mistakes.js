"use strict";

/* ---------------- mistakes tab: Hero's final-table push/fold decisions against HRC ---------------- */

const mistakesEl = (id) => document.getElementById(id);
const MISTAKES_PAGE = 100;
const mistakesFilter = { all: false, players: "", kind: "", position: "", sort: "loss" };
let mistakesData = null;
let mistakesShown = MISTAKES_PAGE;
let mistakeGridMode = "freq";

const SUITS = { h: ["♥", "red"], d: ["♦", "red"], c: ["♣", ""], s: ["♠", ""] };
const cardsText = (cards) => String(cards || "").split(" ").map((c) =>
  `<span class="card-text ${SUITS[c[1]]?.[1] || ""}">${esc(c[0])}${SUITS[c[1]]?.[0] || esc(c[1] || "")}</span>`).join("");
const actionWord = (kind, a) => (a === "fold" ? "фолд" : kind === "open" ? "пуш" : "колл");

async function loadMistakes() {
  mistakesEl("mistakesPage").innerHTML = `<p class="hint">Загрузка…</p>`;
  try {
    const r = await fetch(`/api/decisions${mistakesFilter.all ? "?all=1" : ""}`);
    mistakesData = await r.json();
    if (!r.ok) throw new Error(mistakesData.error || r.status);
  } catch (e) {
    mistakesEl("mistakesPage").innerHTML = `<p class="bad">Раздачи не загрузились: ${esc(e.message)}</p>`;
    return;
  }
  mistakesShown = MISTAKES_PAGE;
  renderMistakes();
}

function pickedDecisions() {
  const f = mistakesFilter;
  const rows = mistakesData.decisions.filter((d) => (!f.players || d.players === +f.players)
    && (!f.kind || d.kind === f.kind) && (!f.position || d.position === f.position));
  return rows.sort(f.sort === "loss" ? (a, b) => b.loss - a.loss || (b.date || "").localeCompare(a.date || "")
    : (a, b) => (b.date || "").localeCompare(a.date || "") || b.hand_id.localeCompare(a.hand_id));
}

function mistakesFiltersHtml() {
  const ds = mistakesData.decisions, f = mistakesFilter;
  const opt = (value, label, current) => `<option value="${esc(value)}"${String(current) === String(value) ? " selected" : ""}>${esc(label)}</option>`;
  const players = [...new Set(ds.map((d) => d.players))].sort((a, b) => a - b);
  const positions = POSITIONS.filter((p) => ds.some((d) => d.position === p));
  return `<section class="panel filters">
    <label class="toggle"><input type="checkbox" id="mfAll"${f.all ? " checked" : ""}> все решения, не только ошибки</label>
    <label>Игроков <select id="mfPlayers">${opt("", "Все", f.players)}${players.map((n) => opt(n, n, f.players)).join("")}</select></label>
    <label>Решение <select id="mfKind">${opt("", "Все", f.kind)}${Object.entries(KIND_LABEL).map(([k, v]) => opt(k, v, f.kind)).join("")}</select></label>
    <label>Позиция <select id="mfPosition">${opt("", "Все", f.position)}${positions.map((p) => opt(p, p, f.position)).join("")}</select></label>
    <label>Порядок <select id="mfSort">${opt("loss", "Сначала дорогие", f.sort)}${opt("date", "Сначала новые", f.sort)}</select></label>
  </section>`;
}

function renderMistakes() {
  const d = mistakesData, rows = pickedDecisions();
  const errors = rows.filter((r) => r.loss > 0), loss = errors.reduce((a, r) => a + r.loss, 0);
  const pending = d.solved < d.needed || d.results.left
    ? `<p class="hint">Оценены только раздачи, чьи сетапы уже решены (${fmtInt(d.solved)} из ${fmtInt(d.needed)}),
        в турнирах с посчитанными результатами (${fmtInt(d.results.total - d.results.left)} из ${fmtInt(d.results.total)}).
        Остальные появятся по ходу фонового расчёта. <button class="link" id="mfReload">Обновить</button></p>` : "";
  const summary = mistakesFilter.all
    ? `Решений ${fmtInt(rows.length)}, из них ошибок ${fmtInt(errors.length)} (${pct(errors.length, rows.length)}), потеря ${loss.toFixed(2)} BI.`
    : `Ошибок ${fmtInt(rows.length)}, потеря ${loss.toFixed(2)} BI.`;
  const body = rows.slice(0, mistakesShown).map((r) => `<tr class="decision-row${r.loss > 0 ? "" : " ok"}" data-hand="${esc(r.hand_id)}">
      <td>${fmtDay(r.date)}</td><td>${esc(r.hand_id)}</td><td>${r.players}</td><td>${r.position}</td><td>${r.eff_bb}</td>
      <td>${cardsText(r.cards)} <span class="muted">${r.class}</span></td><td>${KIND_LABEL[r.kind]}</td>
      <td class="${r.loss > 0 ? "bad" : "good"}">${actionWord(r.kind, r.action)}</td>
      <td>${actionWord(r.kind, r.best)} <span class="muted">${Math.round(100 * (r.best === "play" ? r.freq : 1 - r.freq))}%</span></td>
      <td>${r.loss > 0 ? r.loss.toFixed(3) : "—"}</td></tr>`).join("");
  mistakesEl("mistakesPage").innerHTML = mistakesFiltersHtml() + `<section class="panel">
    <p>${summary}</p>
    <p class="hint">Решения Hero на финальном столе при эффективном стеке до 20 bb против HRC: 2–5 игроков с оверколлами, 6–9 без.
      «HRC» — что выгоднее и как часто HRC так играет эту руку. «Потеря» — разница EV в BI. Нажми на раздачу, чтобы увидеть её и диапазон HRC.</p>
    ${pending}
    ${rows.length ? `<div class="table-scroll"><table class="data decisions">
      <tr><th>Дата</th><th>Раздача</th><th>Игроков</th><th>Позиция</th><th>Эфф. bb</th><th>Рука</th><th>Решение</th><th>Hero</th><th>HRC</th><th>Потеря, BI</th></tr>
      ${body}</table></div>` : `<p class="muted">Нет раздач под этот фильтр.</p>`}
    ${rows.length > mistakesShown ? `<button class="btn small" id="mfMore">Показать ещё (${fmtInt(rows.length - mistakesShown)})</button>` : ""}
  </section>`;
  bindMistakes();
}

function bindMistakes() {
  const f = mistakesFilter;
  mistakesEl("mfAll").onchange = (e) => { f.all = e.target.checked; loadMistakes(); };
  for (const [id, key] of [["mfPlayers", "players"], ["mfKind", "kind"], ["mfPosition", "position"], ["mfSort", "sort"]]) {
    mistakesEl(id).onchange = (e) => { f[key] = e.target.value; mistakesShown = MISTAKES_PAGE; renderMistakes(); };
  }
  if (mistakesEl("mfReload")) mistakesEl("mfReload").onclick = loadMistakes;
  if (mistakesEl("mfMore")) mistakesEl("mfMore").onclick = () => { mistakesShown += MISTAKES_PAGE; renderMistakes(); };
  for (const tr of document.querySelectorAll("#mistakesPage .decision-row")) tr.onclick = () => toggleHand(tr);
}

async function toggleHand(tr) {
  const open = tr.nextElementSibling?.classList.contains("hand-detail");
  document.querySelectorAll("#mistakesPage .hand-detail").forEach((el) => el.remove());
  document.querySelectorAll("#mistakesPage .decision-row.open").forEach((el) => el.classList.remove("open"));
  if (open) return;
  tr.classList.add("open");
  const detail = document.createElement("tr");
  detail.className = "hand-detail";
  detail.innerHTML = `<td colspan="10"><p class="hint">Загрузка…</p></td>`;
  tr.after(detail);
  let hand;
  try {
    const r = await fetch(`/api/hand?id=${encodeURIComponent(tr.dataset.hand)}`);
    hand = await r.json();
    if (!r.ok) throw new Error(hand.error || r.status);
  } catch (e) {
    detail.firstChild.innerHTML = `<p class="bad">Раздача не загрузилась: ${esc(e.message)}</p>`;
    return;
  }
  renderHand(detail.firstChild, hand);
}

function renderHand(td, hand) {
  const d = hand.decision;
  if (!d) {
    td.innerHTML = `<p class="muted">Для этой раздачи нет решения HRC.</p><pre class="hh">${esc(hand.text)}</pre>`;
    return;
  }
  const play = actionWord(d.kind, "play");
  const verdict = d.loss > 0
    ? `<div class="verdict bad">Ошибка: ${actionWord(d.kind, d.action)}<small>HRC: ${actionWord(d.kind, d.best)}, потеря ${d.loss.toFixed(3)} BI</small></div>`
    : `<div class="verdict good">Верно: ${actionWord(d.kind, d.action)}<small>как у HRC</small></div>`;
  td.innerHTML = `<div class="hand-view">
    <div class="hand-side">
      ${verdict}
      <div class="ev-table">
        <span>Рука</span><span>${cardsText(d.cards)} ${d.class}</span>
        <span>Позиция</span><span>${d.position}, ${d.players} игр.</span>
        <span>Эфф. стек</span><span>${d.eff_bb} bb</span>
        <span>EV ${play}а</span><span>${d.ev_play.toFixed(3)} BI</span>
        <span>EV фолда</span><span>${d.ev_fold.toFixed(3)} BI</span>
        <span>HRC ${play}</span><span>${Math.round(100 * d.freq)}%</span>
      </div>
      <p class="hint">${d.overcalls ? "Модель с оверколлами" : "Модель без оверколлов"}. Турнир ${esc(hand.tid)}.</p>
    </div>
    <div>
      <div class="grid-head"><div class="grid-title">Диапазон HRC: <b>${KIND_LABEL[d.kind].toLowerCase()}</b></div>
        <div class="seg small" data-grid-mode>
          <button data-mode="freq"${mistakeGridMode === "freq" ? ' class="active"' : ""}>Частоты</button>
          <button data-mode="ev"${mistakeGridMode === "ev" ? ' class="active"' : ""}>EV</button>
        </div></div>
      <div class="grid"></div>
      <div class="grid-tip">&nbsp;</div>
    </div>
    <details class="hh-wrap"><summary>История раздачи</summary><pre class="hh">${esc(hand.text)}</pre></details>
  </div>`;
  const grid = td.querySelector(".grid"), tip = td.querySelector(".grid-tip");
  const draw = () => { grid.innerHTML = rangeCells(d.range); };
  draw();
  td.querySelector("[data-grid-mode]").onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    mistakeGridMode = b.dataset.mode;
    td.querySelectorAll("[data-grid-mode] button").forEach((x) => x.classList.toggle("active", x === b));
    draw();
  };
  grid.onmouseover = (e) => {
    const c = e.target.closest(".cell");
    if (!c) return;
    const i = +c.dataset.i, diff = d.range.ev_diff[i];
    tip.textContent = `${classLabel(i)}: ${play} ${Math.round(d.range.freq[i] * 100)}% · EV ${play}а − фолда ${(diff >= 0 ? "+" : "") + diff.toFixed(3)} BI`;
  };
  grid.onmouseleave = () => { tip.innerHTML = "&nbsp;"; };
}

// the same colours as the trainer's range grid
function rangeCells(range) {
  const cells = [];
  for (let i = 0; i < 169; i++) {
    const f = range.freq[i], diff = range.ev_diff[i];
    let bg;
    if (mistakeGridMode === "freq") {
      bg = `linear-gradient(90deg, rgb(var(--play)) ${f * 100}%, rgb(var(--fold)) ${f * 100}%)`;
    } else {
      const k = Math.min(Math.abs(diff) / 0.3, 1);
      bg = diff > 0 ? `rgba(63,191,127,${0.15 + 0.85 * k})` : `rgba(229,83,75,${0.15 + 0.85 * k})`;
    }
    cells.push(`<div class="cell${i === range.hand_class ? " hero" : ""}" data-i="${i}" style="background:${bg}">${classLabel(i)}</div>`);
  }
  return cells.join("");
}

document.addEventListener("tabshown", (e) => {
  if (e.detail === "mistakes") loadMistakes();
});
