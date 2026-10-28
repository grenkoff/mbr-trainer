"use strict";

/* ---------------- statistics tab ---------------- */

const statsEl = (id) => document.getElementById(id);
// categorical slots of the dark chart palette, in fixed order
const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"];
const KIND_LABEL = { open: "Опен-пуш", call: "Колл пуша", overcall: "Оверколл" };
const STAGE_LABEL = { rush: "Rush", ft: "Финальный стол" };
const statsFilter = { from: "", to: "", buyin: "", stage: "ft" };
let statsData = null;
let statsPoll = null;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const signed = (x, d = 3) => (x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const pm = (s, d = 3) => (s ? `${signed(s[0], d)} <span class="muted">± ${s[1].toFixed(d)}</span>` : "—");
const pmPct = (s) => (s ? `${signed(100 * s[0], 1)}% <span class="muted">± ${(100 * s[1]).toFixed(1)}%</span>` : "—");
const pct = (a, b) => (b ? `${(100 * a / b).toFixed(1)}%` : "—");
const fmtDay = (s) => (s ? new Date(s).toLocaleDateString("ru-RU") : "");

async function loadStats() {
  const q = new URLSearchParams({ from: statsFilter.from, to: statsFilter.to, buyin: statsFilter.buyin });
  try {
    const r = await fetch(`/api/stats?${q}`);
    statsData = await r.json();
    if (!r.ok) throw new Error(statsData.error || r.status);
  } catch (e) {
    statsEl("statsPage").innerHTML = `<p class="bad">Статистика не загрузилась: ${esc(e.message)}</p>`;
    return;
  }
  renderStatsPage();
  // results still being computed: come back for them
  clearTimeout(statsPoll);
  if (statsData.results.left && !statsEl("tab-stats").classList.contains("hidden")) statsPoll = setTimeout(loadStats, 5000);
}

function renderStatsPage() {
  const d = statsData, t = d.totals;
  const left = d.results.left
    ? `<p class="hint">Считаются результаты: осталось ${d.results.left} турниров${d.results.cooling ? " (пауза: процессор остывает)" : ""}.</p>` : "";
  if (!t.tournaments) {
    statsEl("statsPage").innerHTML = filtersHtml() + left + `<section class="panel"><p class="muted">Нет турниров: загрузи историю на вкладке «Загрузка» или измени фильтр.</p></section>`;
    bindFilters();
    return;
  }
  statsEl("statsPage").innerHTML = filtersHtml() + left + `
    <div class="tiles">
      ${tile("Турниров", t.tournaments, `финальный стол ${pct(t.final_tables, t.tournaments)}, призы ${pct(t.itm, t.tournaments)}`)}
      ${tile("ROI фактически", t.roi.actual ? `${signed(100 * t.roi.actual[0], 1)}%` : "—", `± ${t.roi.actual ? (100 * t.roi.actual[1]).toFixed(1) : "—"}%, конверты как выпали`)}
      ${tile("ROI без удачи", t.roi.full ? `${signed(100 * t.roi.full[0], 1)}%` : "—", `± ${t.roi.full ? (100 * t.roi.full[1]).toFixed(1) : "—"}%, средние баунти и олл-ины по EV`)}
      ${tile("chipEV на Rush", t.rush.rush_ev ? signed(t.rush.rush_ev[0], 0) : "—", `фишек за турнир, фактически ${t.rush.rush_actual ? signed(t.rush.rush_actual[0], 0) : "—"}`)}
      ${tile("Нокаутов", fmtNum(t.kos), `баунти ${t.bounties.toFixed(2)} BI, ${t.bounty_per_ko == null ? "—" : t.bounty_per_ko.toFixed(3)} на нокаут`)}
    </div>
    <section class="panel">
      <h2>Результат, BI нарастающим итогом</h2>
      <div class="chart" id="chartResult"></div>
      <p class="hint">Каждая точка — турнир. «Средние баунти»: каждый нокаут по 0,945 BI вместо выпавшего конверта.
        «Олл-ины на финальном столе по EV»: раздачи, где все карты открыты до ривера, по ожиданию по оставшимся картам.
        «Без удачи»: то же на Rush, фишки после Rush по их средней стоимости (0,92 BI за 1000).</p>
    </section>
    <div class="two-col">
      <section class="panel">
        <h2>Rush: фишки нарастающим итогом</h2>
        <div class="chart" id="chartRush"></div>
      </section>
      <section class="panel">
        <h2>Места</h2>
        <div class="chart" id="chartPlaces"></div>
      </section>
    </div>
    <section class="panel">
      <h2>Итоги</h2>
      ${totalsHtml(t)}
    </section>
    <section class="panel">
      <h2>Префлоп</h2>
      <div class="seg small" id="statsStage">${["rush", "ft"].map((s) =>
        `<button data-v="${s}" class="${s === statsFilter.stage ? "active" : ""}">${STAGE_LABEL[s]}</button>`).join("")}</div>
      ${preflopHtml(d.preflop[statsFilter.stage])}
    </section>
    <section class="panel">
      <h2>Пуш/фолд против HRC</h2>
      ${decisionsHtml(d.decisions, t.tournaments)}
    </section>`;
  bindFilters();
  statsEl("statsStage").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    statsFilter.stage = b.dataset.v;
    renderStatsPage();
  });
  const rows = d.tournaments;
  // from 0 before the first tournament
  const cum = (key) => rows.reduce((acc, r) => (acc.push(acc.at(-1) + r[key]), acc), [0]);
  const tip = (i) => (i ? `Турнир ${i}: #${rows[i - 1].tid}, ${fmtDay(rows[i - 1].date)}${rows[i - 1].place ? `, ${rows[i - 1].place} место` : ""}` : "Старт");
  lineChart(statsEl("chartResult"), [
    { name: "Фактически", values: cum("actual") },
    { name: "Средние баунти", values: cum("mean") },
    { name: "+ олл-ины на финальном столе по EV", values: cum("ev") },
    { name: "Без удачи", values: cum("full") },
  ], { digits: 2, unit: " BI", tip });
  lineChart(statsEl("chartRush"), [
    { name: "Фактически", values: cum("rush_actual") },
    { name: "chipEV", values: cum("rush_ev") },
  ], { digits: 0, unit: "", tip });
  const places = Array.from({ length: 18 }, (_, i) => t.places[i + 1] || 0);
  barChart(statsEl("chartPlaces"), places.map((n, i) => ({ label: String(i + 1), value: n })),
    (b) => `${b.label} место: ${b.value} (${pct(b.value, t.tournaments)})`);
}

const fmtNum = (x) => (Number.isInteger(x) ? String(x) : x.toFixed(1));
const tile = (label, value, sub) => `<div class="panel tile"><div class="tile-label">${label}</div><div class="tile-value">${value}</div><div class="hint">${sub}</div></div>`;

function filtersHtml() {
  const buyins = (statsData && statsData.buyins) || [];
  return `<div class="filters panel">
    <label>С <input type="date" id="fFrom" value="${statsFilter.from}"></label>
    <label>По <input type="date" id="fTo" value="${statsFilter.to}"></label>
    <label>Бай-ин <select id="fBuyin"><option value="">Все</option>${buyins.map((b) =>
      `<option value="${b}" ${String(b) === statsFilter.buyin ? "selected" : ""}>$${b}</option>`).join("")}</select></label>
    <button class="btn small" id="fReset">Сбросить</button>
  </div>`;
}

function bindFilters() {
  const set = (k, v) => { statsFilter[k] = v; loadStats(); };
  statsEl("fFrom").addEventListener("change", (e) => set("from", e.target.value));
  statsEl("fTo").addEventListener("change", (e) => set("to", e.target.value));
  statsEl("fBuyin").addEventListener("change", (e) => set("buyin", e.target.value));
  statsEl("fReset").addEventListener("click", () => {
    Object.assign(statsFilter, { from: "", to: "", buyin: "" });
    loadStats();
  });
}

function totalsHtml(t) {
  const row = (label, value) => `<tr><td>${label}</td><td>${value}</td></tr>`;
  return `<div class="two-col"><div class="table-scroll"><table class="data">
    <tr><th>ROI за турнир, BI</th><th>среднее ± ошибка</th></tr>
    ${row("Фактически, конверты как выпали", pm(t.roi.actual))}
    ${row("Средние баунти", pm(t.roi.mean))}
    ${row("+ олл-ины на финальном столе по EV", pm(t.roi.ev))}
    ${row("Без удачи (и на Rush)", pm(t.roi.full))}
    ${row("Rush: фишки фактически", pm(t.rush.rush_actual, 0))}
    ${row("Rush: chipEV", pm(t.rush.rush_ev, 0))}
  </table></div><div class="table-scroll"><table class="data">
    <tr><th>Финальный стол</th><th></th></tr>
    ${row("Финальных столов", `${t.final_tables} (${pct(t.final_tables, t.tournaments)})`)}
    ${row("Стоимость на входе, BI", t.ft_value == null ? "—" : t.ft_value.toFixed(3))}
    ${row("Изменение фактически, BI", pm(t.ft_change.actual))}
    ${row("Изменение по EV, BI", pm(t.ft_change.ev))}
    ${row("Удача в олл-инах: свои / чужие", `${signed(t.ft_luck.hero)} / ${signed(t.ft_luck.others)}`)}
    ${row("Нокаутов: на финальном столе / в конце Rush", `${fmtNum(t.ft_kos)} / ${fmtNum(t.rush_kos)}`)}
    ${row("Баунти, BI: всего / на нокаут", `${t.bounties.toFixed(2)} / ${t.bounty_per_ko == null ? "—" : t.bounty_per_ko.toFixed(3)} (среднее 0,945)`)}
  </table></div></div>`;
}

function preflopHtml(p) {
  if (!p) return `<p class="muted">Нет раздач.</p>`;
  const cell = (a, b) => (b ? `${(100 * a / b).toFixed(0)}% <span class="muted">${b}</span>` : `<span class="muted">—</span>`);
  const line = (label, c) => `<tr><td>${label}</td><td>${c.hands}</td><td>${cell(c.vpip, c.hands)}</td><td>${cell(c.pfr, c.hands)}</td>
    <td>${cell(c.rfi, c.rfi_opp)}</td><td>${cell(c.open_shove, c.rfi_opp)}</td><td>${cell(c.limp, c.rfi_opp)}</td>
    <td>${cell(c.threebet, c.threebet_opp)}</td><td>${cell(c.vs_shove_call, c.vs_shove_opp)}</td></tr>`;
  const head = (first) => `<tr><th>${first}</th><th>Раздач</th><th>VPIP</th><th>PFR</th><th>Первым в банк</th><th>Опен-пуш</th><th>Лимп</th><th>3-бет</th><th>Колл пуша</th></tr>`;
  return `<p class="hint">Проценты от возможностей, мелким шрифтом — сколько их было. «Первым в банк», «Опен-пуш» и «Лимп» — когда до Hero все сбросили;
    «3-бет» — против одного рейза; «Колл пуша» — против одного олл-ина (колл или ответный олл-ин).</p>
    <div class="table-scroll"><table class="data">${head("Позиция")}${p.positions.map(([k, c]) => line(k, c)).join("")}${line("<b>Всего</b>", p.total)}</table></div>
    <div class="table-scroll"><table class="data">${head("Стек Hero")}${p.buckets.map(([k, c]) => line(k, c)).join("")}</table></div>`;
}

function decisionsHtml(d, tournaments) {
  if (!d.decisions) return `<p class="muted">Нет решений с готовым решением HRC.</p>`;
  const group = (rows, first) => `<div class="table-scroll"><table class="data">
    <tr><th>Решение</th><th>${first}</th><th>Решений</th><th>Ошибок</th><th>Потеря, BI</th><th>На решение, mBI</th><th>Hero играет</th><th>HRC играет</th></tr>
    ${rows.map(([kind, k, g]) => `<tr><td>${KIND_LABEL[kind]}</td><td>${k}</td><td>${g.decisions}</td><td>${pct(g.mistakes, g.decisions)}</td>
      <td>${g.loss.toFixed(2)}</td><td>${(1000 * g.loss / g.decisions).toFixed(1)}</td><td>${pct(g.plays, g.decisions)}</td><td>${pct(g.best_plays, g.decisions)}</td></tr>`).join("")}
  </table></div>`;
  const act = (kind, a) => (a === "fold" ? "фолд" : kind === "open" ? "пуш" : "колл");
  return `<p>Решений: ${d.decisions}, ошибок ${d.mistakes} (${pct(d.mistakes, d.decisions)}), потеря ${d.loss.toFixed(2)} BI —
      <b>${(d.loss / tournaments).toFixed(3)} BI за турнир</b>.</p>
    <p class="hint">Решения Hero на финальном столе при эффективном стеке до 20 bb против библиотеки HRC: 2–5 игроков с оверколлами, 6–9 без.
      Лимпы и неполные рейзы не оцениваются. «Hero играет» и «HRC играет» — как часто пуш/колл у Hero и у HRC в тех же раздачах:
      если у Hero меньше, он играет туже.</p>
    ${group(d.by_position, "Позиция")}${group(d.by_bucket, "Эфф. стек")}
    <h2>Самые дорогие ошибки</h2>
    <div class="table-scroll"><table class="data">
      <tr><th>Раздача</th><th>Игроков</th><th>Позиция</th><th>Эфф. bb</th><th>Рука</th><th>Hero</th><th>HRC</th><th>Потеря, BI</th></tr>
      ${d.worst.map((w) => `<tr><td>${esc(w.hand_id)}</td><td>${w.players}</td><td>${w.position}</td><td>${w.eff_bb}</td>
        <td>${esc(w.cards)} <span class="muted">${w.class}</span></td><td>${KIND_LABEL[w.kind]}: ${act(w.kind, w.action)}</td>
        <td>${act(w.kind, w.best)} <span class="muted">${Math.round(100 * (w.best === "play" ? w.freq : 1 - w.freq))}%</span></td><td>${w.loss.toFixed(3)}</td></tr>`).join("")}
    </table></div>`;
}

/* ---------------- charts (SVG) ---------------- */

const CHART_H = 260, PAD = { l: 52, r: 12, t: 12, b: 28 };

function niceTicks(lo, hi, count = 5) {
  const span = hi - lo || 1, step0 = span / count, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function chartTooltip(el) {
  let tipEl = el.querySelector(".chart-tip");
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "chart-tip hidden";
    el.appendChild(tipEl);
  }
  return tipEl;
}

/** Lines over the tournaments, one shared y axis, with a legend, a crosshair and a tooltip. */
function lineChart(el, series, { digits, unit, tip }) {
  const n = series[0].values.length;
  const w = Math.max(el.clientWidth, 300);
  const all = series.flatMap((s) => s.values).concat([0]);
  const ticks = niceTicks(Math.min(...all), Math.max(...all));
  const lo = Math.min(ticks[0], ...all), hi = Math.max(ticks.at(-1), ...all);
  const x = (i) => PAD.l + (n > 1 ? i / (n - 1) : 0.5) * (w - PAD.l - PAD.r);
  const y = (v) => PAD.t + (hi - v) / (hi - lo || 1) * (CHART_H - PAD.t - PAD.b);
  const xticks = niceTicks(0, n - 1, Math.min(8, n - 1)).filter((v) => Number.isInteger(v));
  el.innerHTML = `<div class="legend">${series.map((s, k) => `<span><i style="background:${SERIES[k]}"></i>${esc(s.name)}</span>`).join("")}</div>
    <svg width="${w}" height="${CHART_H}" role="img">
      ${ticks.map((v) => `<line class="grid${v === 0 ? " zero" : ""}" x1="${PAD.l}" x2="${w - PAD.r}" y1="${y(v)}" y2="${y(v)}"/>
        <text class="axis" x="${PAD.l - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(digits && Math.abs(ticks[1] - ticks[0]) < 1 ? digits : 0)}</text>`).join("")}
      ${xticks.map((v) => `<text class="axis" x="${x(v)}" y="${CHART_H - 8}" text-anchor="middle">${v}</text>`).join("")}
      ${series.map((s, k) => `<polyline class="series" stroke="${SERIES[k]}" points="${s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}"/>`).join("")}
      <line class="crosshair hidden" y1="${PAD.t}" y2="${CHART_H - PAD.b}"/>
      ${series.map((_, k) => `<circle class="dot hidden" r="4" fill="${SERIES[k]}"/>`).join("")}
      <rect class="hit" x="${PAD.l}" y="0" width="${w - PAD.l - PAD.r}" height="${CHART_H}"/>
    </svg>`;
  const svg = el.querySelector("svg"), tipEl = chartTooltip(el);
  const cross = svg.querySelector(".crosshair"), dots = [...svg.querySelectorAll(".dot")];
  svg.querySelector(".hit").addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(n - 1, Math.round((e.clientX - box.left - PAD.l) / (w - PAD.l - PAD.r) * (n - 1))));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.classList.remove("hidden");
    dots.forEach((d, k) => {
      d.setAttribute("cx", x(i));
      d.setAttribute("cy", y(series[k].values[i]));
      d.classList.remove("hidden");
    });
    tipEl.innerHTML = `<div>${esc(tip(i))}</div>` + series.map((s, k) =>
      `<div><i style="background:${SERIES[k]}"></i>${esc(s.name)}: <b>${signed(s.values[i], digits)}${unit}</b></div>`).join("");
    tipEl.classList.remove("hidden");
    const left = x(i) + 14 + tipEl.offsetWidth > w ? x(i) - 14 - tipEl.offsetWidth : x(i) + 14;
    tipEl.style.left = `${left}px`;
    tipEl.style.top = `${PAD.t + 24}px`;
  });
  svg.querySelector(".hit").addEventListener("mouseleave", () => {
    cross.classList.add("hidden");
    dots.forEach((d) => d.classList.add("hidden"));
    tipEl.classList.add("hidden");
  });
}

/** Vertical bars with a tooltip each. */
function barChart(el, bars, tip) {
  const w = Math.max(el.clientWidth, 300);
  const ticks = niceTicks(0, Math.max(4, ...bars.map((b) => b.value)), 4).filter(Number.isInteger); // counts
  const hi = Math.max(ticks.at(-1), ...bars.map((b) => b.value));
  const slot = (w - PAD.l - PAD.r) / bars.length, bw = Math.max(4, slot - 4);
  const y = (v) => PAD.t + (hi - v) / hi * (CHART_H - PAD.t - PAD.b);
  el.innerHTML = `<svg width="${w}" height="${CHART_H}" role="img">
    ${ticks.map((v) => `<line class="grid${v === 0 ? " zero" : ""}" x1="${PAD.l}" x2="${w - PAD.r}" y1="${y(v)}" y2="${y(v)}"/>
      <text class="axis" x="${PAD.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("")}
    ${bars.map((b, i) => {
      const x0 = PAD.l + i * slot + (slot - bw) / 2, top = y(b.value), h = y(0) - top;
      return `<path class="bar" fill="${SERIES[0]}" d="${h > 0 ? `M${x0},${y(0)}V${top + Math.min(4, h)}q0,-${Math.min(4, h)} ${Math.min(4, bw / 2)},-${Math.min(4, h)}H${x0 + bw - Math.min(4, bw / 2)}q${Math.min(4, bw / 2)},0 ${Math.min(4, bw / 2)},${Math.min(4, h)}V${y(0)}Z` : ""}"/>
        <text class="axis" x="${x0 + bw / 2}" y="${CHART_H - 8}" text-anchor="middle">${esc(b.label)}</text>
        <rect class="hit" data-i="${i}" x="${PAD.l + i * slot}" y="0" width="${slot}" height="${CHART_H}"/>`;
    }).join("")}
  </svg>`;
  const tipEl = chartTooltip(el);
  for (const r of el.querySelectorAll(".hit")) {
    r.addEventListener("mouseenter", () => {
      const i = +r.dataset.i;
      tipEl.textContent = tip(bars[i]);
      tipEl.classList.remove("hidden");
      const cx = PAD.l + (i + 0.5) * slot;
      tipEl.style.left = `${Math.min(cx + 10, w - tipEl.offsetWidth - 4)}px`;
      tipEl.style.top = `${PAD.t + 8}px`;
    });
    r.addEventListener("mouseleave", () => tipEl.classList.add("hidden"));
  }
}

document.addEventListener("tabshown", (e) => {
  clearTimeout(statsPoll);
  if (e.detail === "stats") loadStats();
});
window.addEventListener("resize", () => {
  if (statsData && !statsEl("tab-stats").classList.contains("hidden")) renderStatsPage();
});
