"use strict";

/* ---------------- average opponent tab ---------------- */

const oppEl = (id) => document.getElementById(id);
const oppFilter = { from: "", to: "", buyin: "", stage: "ft", shownKind: "call", bucket: "" };
let oppData = null;
let oppPoll = null;

// combos of each of the 169 hand classes (grid order of classLabel): pairs 6, suited 4, offsuit 12
const CLASS_COMBOS = Array.from({ length: 169 }, (_, i) => {
  const row = Math.floor(i / 13), col = i % 13;
  return row === col ? 6 : row < col ? 4 : 12;
});
const bucketLow = (b) => parseFloat(b) || 0;

async function loadOpponents() {
  const q = new URLSearchParams({ from: oppFilter.from, to: oppFilter.to, buyin: oppFilter.buyin });
  try {
    const r = await fetch(`/api/opponents?${q}`);
    oppData = await r.json();
    if (!r.ok) throw new Error(oppData.error || r.status);
  } catch (e) {
    oppEl("opponentsPage").innerHTML = `<p class="bad">Данные не загрузились: ${esc(e.message)}</p>`;
    return;
  }
  renderOpponents();
  clearTimeout(oppPoll);
  if (oppData.results.opponents_left && !oppEl("tab-opponents").classList.contains("hidden")) {
    oppPoll = setTimeout(loadOpponents, 10000);
  }
}

function oppFiltersHtml() {
  const f = oppFilter;
  return `<div class="filters panel">
    <label>С <input type="date" id="oFrom" value="${f.from}"></label>
    <label>По <input type="date" id="oTo" value="${f.to}"></label>
    <label>Бай-ин <select id="oBuyin"><option value="">Все</option>${oppData.buyins.map((b) =>
      `<option value="${b}" ${String(b) === f.buyin ? "selected" : ""}>$${b}</option>`).join("")}</select></label>
    <div class="seg small" id="oStage">${["rush", "ft"].map((s) =>
      `<button data-v="${s}" class="${s === f.stage ? "active" : ""}">${STAGE_LABEL[s]}</button>`).join("")}</div>
    <button class="btn small" id="oReset">Сбросить</button>
  </div>`;
}

function renderOpponents() {
  const d = oppData, f = oppFilter, r = d.results;
  const progress = r.opponents_left
    ? `<section class="panel">${progressBar("Турниров посчитано", r.total - r.opponents_left, r.total,
        r.cooling ? "Пауза: процессор остывает." : "Считается в фоне после твоих результатов, страница дополняется по ходу.")}</section>` : "";
  const theirs = d.preflop[f.stage], mine = d.hero_preflop[f.stage];
  if (!theirs) {
    oppEl("opponentsPage").innerHTML = oppFiltersHtml() + progress +
      `<section class="panel"><p class="muted">Нет раздач: загрузи историю на вкладке «Загрузка» или измени фильтр.</p></section>`;
    bindOpponents();
    return;
  }
  const t = theirs.total, h = mine ? mine.total : null;
  const share = (a, b) => (b ? `${(100 * a / b).toFixed(1)}%` : "—");
  const cmp = (label, a, b, note) => tile(label, share(t[a], t[b]),
    `Hero ${h ? share(h[a], h[b]) : "—"} · ${note} ${fmtInt(t[b])}`);
  oppEl("opponentsPage").innerHTML = oppFiltersHtml() + progress + `
    <div class="tiles">
      ${cmp("VPIP", "vpip", "hands", "раздач")}
      ${cmp("PFR", "pfr", "hands", "раздач")}
      ${cmp("Первым в банк", "rfi", "rfi_opp", "возможностей")}
      ${cmp("Опен-пуш", "open_shove", "rfi_opp", "возможностей")}
      ${cmp("3-бет", "threebet", "threebet_opp", "возможностей")}
      ${cmp("Колл пуша", "vs_shove_call", "vs_shove_opp", "возможностей")}
    </div>
    <section class="panel">
      <h2>Префлоп среднего оппонента: ${STAGE_LABEL[f.stage]}</h2>
      <p class="hint">Все соперники из твоих раздач вместе, ${fmtInt(d.tournaments)} турниров. В плитках сверху — для сравнения то же у Hero.</p>
      ${preflopHtml(theirs, "opponents")}
    </section>
    <section class="panel">
      <h2>Пуш/фолд против HRC на финальном столе</h2>
      ${oppDecisionsHtml(d.decisions)}
    </section>
    <section class="panel">
      <h2>Какие руки показывают: ${STAGE_LABEL[f.stage]}</h2>
      ${shownHtml()}
    </section>`;
  bindOpponents();
}

function oppDecisionsHtml(d) {
  if (!d.decisions) return `<p class="muted">Нет решений с готовым решением HRC.</p>`;
  const pp = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(1)}`;
  const group = (rows, first) => `<div class="table-scroll"><table class="data kind-table">
    <tr><th>Решение</th><th>${first}</th><th>Решений</th><th>Играют</th><th>HRC играет</th><th>Разница, п.п.</th>
      <th>Лимп или рейз не олл-ин</th>
      <th>Вскрыто</th><th>Из них HRC фолдит</th><th>Потеря на вскрытое, mBI</th></tr>
    ${rows.filter(([, , g]) => g.decisions).map(([kind, k, g]) => {
      const played = 100 * g.plays / g.decisions, hrc = 100 * g.hrc / g.decisions;
      return `<tr${k === "all" ? ' class="total"' : ""}><td>${KIND_LABEL[kind]}</td><td>${k === "all" ? "<b>Всего</b>" : k}</td>
        <td>${fmtInt(g.decisions)}</td><td>${played.toFixed(1)}%</td><td>${hrc.toFixed(1)}%</td><td>${pp(played - hrc)}</td>
        <td>${g.other ? `${pct(g.other, g.decisions + g.other)}` : "—"}</td>
        <td>${g.shown ? fmtInt(g.shown) : "—"}</td><td>${pct(g.shown_hrc_folds, g.shown)}</td>
        <td>${g.shown ? (1000 * g.loss / g.shown).toFixed(1) : "—"}</td></tr>`;
    }).join("")}
  </table></div>`;
  return `<p class="hint">Решения соперников при эффективном стеке до 20 bb в раздачах с решённым сетапом (${fmtInt(d.decisions)}):
      2–5 игроков с оверколлами, 6–9 без. «Играют» — как часто они пушат или коллируют, «HRC играет» — какую долю всех рук HRC играет
      в тех же ситуациях. Плюс в разнице — оппоненты играют шире HRC, минус — туже. Лимпы и рейзы не в олл-ин в эти проценты не входят:
      их доля от всех решений — в отдельной колонке. Руки видны только у сыгранных и вскрытых решений:
      колл пуша вскрывается всегда, опен-пуш — только если его заколлировали.</p>
    ${group(d.by_position, "Позиция")}${group(d.by_bucket, "Эфф. стек")}`;
}

function shownHtml() {
  const f = oppFilter, prefix = `${f.stage} ${f.shownKind} `;
  const keys = Object.keys(oppData.shown).filter((k) => k.startsWith(prefix));
  const buckets = keys.map((k) => k.slice(prefix.length)).sort((a, b) => bucketLow(a) - bucketLow(b));
  const counts = Array(169).fill(0);
  for (const k of keys) {
    if (f.bucket && k !== prefix + f.bucket) continue;
    oppData.shown[k].forEach((n, i) => { counts[i] += n; });
  }
  const total = counts.reduce((a, b) => a + b, 0);
  // how often each hand turns up per combo, so offsuit hands do not look three times as common as pairs
  const rate = counts.map((n, i) => n / CLASS_COMBOS[i]);
  const top = Math.max(...rate) || 1;
  const cells = counts.map((n, i) => {
    const k = rate[i] / top;
    return `<div class="cell" data-i="${i}" style="background:${n ? `rgba(57,135,229,${0.12 + 0.88 * k})` : "rgb(var(--fold))"}">${classLabel(i)}</div>`;
  }).join("");
  return `<div class="seg small" id="oShownKind">
      <button data-v="call" class="${f.shownKind === "call" ? "active" : ""}">Колл пуша</button>
      <button data-v="shove" class="${f.shownKind === "shove" ? "active" : ""}">Опен-пуш (вскрытые)</button></div>
    <label class="field-inline">Эфф. стек <select id="oBucket"><option value="">Все</option>${buckets.map((b) =>
      `<option value="${b}" ${b === f.bucket ? "selected" : ""}>${b}</option>`).join("")}</select></label>
    <p class="hint">${f.shownKind === "call"
      ? "С какими руками оппоненты коллируют пуш (или отвечают олл-ином). Такой колл всегда вскрывается, так что это весь их диапазон колла."
      : "С какими руками оппоненты открываются олл-ином. Видны только заколлированные пуши, поэтому выборка смещена к рукам, против которых коллируют."}
      Цвет — как часто рука встречается в пересчёте на комбинацию, ярче — чаще. Всего рук: ${fmtInt(total)}.</p>
    ${total ? `<div class="grid shown-grid" id="oGrid">${cells}</div><div class="grid-tip" id="oGridTip">&nbsp;</div>`
      : `<p class="muted">Нет вскрытых рук под этот фильтр.</p>`}`;
}

function bindOpponents() {
  const f = oppFilter;
  const reload = (k, v) => { f[k] = v; loadOpponents(); };
  oppEl("oFrom").onchange = (e) => reload("from", e.target.value);
  oppEl("oTo").onchange = (e) => reload("to", e.target.value);
  oppEl("oBuyin").onchange = (e) => reload("buyin", e.target.value);
  oppEl("oReset").onclick = () => { Object.assign(f, { from: "", to: "", buyin: "" }); loadOpponents(); };
  const seg = (id, key) => {
    const el = oppEl(id);
    if (!el) return;
    el.onclick = (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      f[key] = b.dataset.v;
      f.bucket = "";
      renderOpponents();
    };
  };
  seg("oStage", "stage");
  seg("oShownKind", "shownKind");
  if (oppEl("oBucket")) oppEl("oBucket").onchange = (e) => { f.bucket = e.target.value; renderOpponents(); };
  const grid = oppEl("oGrid");
  if (grid) {
    const prefix = `${f.stage} ${f.shownKind} `;
    grid.onmouseover = (e) => {
      const c = e.target.closest(".cell");
      if (!c) return;
      const i = +c.dataset.i;
      let n = 0, total = 0;
      for (const [k, counts] of Object.entries(oppData.shown)) {
        if (!k.startsWith(prefix) || (f.bucket && k !== prefix + f.bucket)) continue;
        n += counts[i];
        total += counts.reduce((a, b) => a + b, 0);
      }
      oppEl("oGridTip").textContent = `${classLabel(i)}: ${fmtInt(n)} раз, ${total ? (100 * n / total).toFixed(1) : 0}% вскрытых рук`;
    };
    grid.onmouseleave = () => { oppEl("oGridTip").innerHTML = "&nbsp;"; };
  }
}

document.addEventListener("tabshown", (e) => {
  clearTimeout(oppPoll);
  if (e.detail === "opponents") loadOpponents();
});
