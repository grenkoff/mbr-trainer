"use strict";

/* ---------------- upload tab ---------------- */

const uploadEl = (id) => document.getElementById(id);
const fmtInt = (n) => Number(n || 0).toLocaleString("ru-RU");
const fmtDate = (s) => (s ? new Date(s).toLocaleString("ru-RU", { dateStyle: "short", timeStyle: "short" }) : "—");
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
let uploadPoll = null;

async function uploadFiles(files) {
  files = [...files].filter((f) => /\.(txt|zip)$/i.test(f.name));
  const log = uploadEl("uploadLog");
  for (const f of files) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escapeHtml(f.name)}</span><span class="muted">загружается…</span>`;
    log.prepend(li);
    try {
      const r = await fetch(`/api/upload?name=${encodeURIComponent(f.name)}`, { method: "POST", body: f });
      const res = await r.json();
      if (!r.ok) throw new Error(res.error || r.status);
      li.lastChild.textContent = res.duplicate ? "уже загружен"
        : `раздач ${fmtInt(res.hands)}, сводок ${fmtInt(res.summaries)}, сетапов в расчёт ${fmtInt(res.setups)}`;
      li.classList.add(res.duplicate ? "muted" : "good");
    } catch (e) {
      li.lastChild.textContent = `ошибка: ${e.message}`;
      li.classList.add("bad");
    }
  }
  if (files.length) refreshUploads();
}

function fmtDuration(sec) {
  const m = Math.round(sec / 60);
  if (m < 1) return "меньше минуты";
  if (m < 60) return `${m} мин`;
  const h = Math.floor(m / 60);
  return h < 48 ? `${h} ч ${m % 60} мин` : `${Math.floor(h / 24)} д ${h % 24} ч`;
}

/* a progress bar with "done of total" and a note under it; also used by the statistics tab */
function progressBar(label, done, total, note = "") {
  const share = total ? Math.min(1, done / total) : 1;
  // never rounds up to 100% before the end
  const percent = Math.floor(share * 1000) / 10;
  return `<div class="progress">
    <div class="progress-head"><span>${label}</span><b>${fmtInt(done)} из ${fmtInt(total)} · ${percent}%</b></div>
    <div class="progress-track" role="progressbar" aria-label="${label}" aria-valuemin="0" aria-valuemax="${total}"
      aria-valuenow="${done}"><div class="progress-fill${share >= 1 ? " full" : ""}" style="width:${share * 100}%"></div></div>
    ${note ? `<p class="hint">${note}</p>` : ""}</div>`;
}

function renderQueue(q) {
  const temp = q.temperature == null ? "" : ` · процессор ${Math.round(q.temperature)}°C`;
  const byPlayers = new Map();
  let done = 0, total = 0, failed = 0, seconds = 0;
  const unknown = new Set();
  for (const g of q.progress) {
    const left = g.total - g.done - g.failed;
    const p = byPlayers.get(g.players) || { done: 0, total: 0, seconds: 0, unknown: false };
    p.done += g.done; p.total += g.total;
    done += g.done; total += g.total; failed += g.failed;
    if (left > 0 && g.seconds == null) { p.unknown = true; unknown.add(g.players); }
    else if (left > 0) { p.seconds += left * g.seconds / q.workers; seconds += left * g.seconds / q.workers; }
    byPlayers.set(g.players, p);
  }
  const busy = done + failed < total;
  const state = !busy ? `всё посчитано${temp}`
    : q.paused ? `<span class="bad">пауза: перегрев, продолжит при ${q.resume_temp}°C</span>${temp}`
    : `<span class="good">считает</span>${temp}`;
  const eta = !busy ? "" : unknown.size
    ? `Осталось не меньше ${fmtDuration(seconds)}: время для ${[...unknown].join(", ")} игроков станет известно после их первых решений.`
    : `Осталось около ${fmtDuration(seconds)} при нынешней скорости, с паузами на остывание.`;
  const rows = [...byPlayers].map(([n, p]) => `<tr><td>${n}</td><td>${fmtInt(p.done)}</td><td>${fmtInt(p.total)}</td>
    <td class="bar-cell"><div class="progress-track small"><div class="progress-fill${p.done >= p.total ? " full" : ""}"
      style="width:${p.total ? p.done / p.total * 100 : 100}%"></div></div></td>
    <td>${fmtInt(p.total - p.done)}</td>
    <td>${p.done >= p.total ? "—" : p.unknown ? "?" : fmtDuration(p.seconds)}</td></tr>`);
  uploadEl("queueStatus").innerHTML = `<p>${state}</p>` + progressBar("Решено", done, total, eta) +
    (rows.length ? `<div class="table-scroll"><table class="data"><tr><th>Игроков</th><th>Решено</th><th>Всего</th>
      <th class="bar-cell"></th><th>Осталось</th><th>Время</th></tr>${rows.join("")}</table></div>` : "") +
    q.running.map((r) => `<p class="hint">Сейчас: ${r.hand_id}, ${r.variant.endsWith("-nooc") ? "без оверколлов" : "с оверколлами"}, ${r.seconds} с</p>`).join("") +
    (failed ? `<p class="hint bad">С ошибкой: ${fmtInt(failed)}</p>` : "") +
    (q.last_error ? `<p class="hint bad">Последняя ошибка: ${escapeHtml(q.last_error)}</p>` : "");
}

function resultsProgress(r) {
  return progressBar("Турниров посчитано", r.total - r.left, r.total,
    r.left ? (r.cooling ? "Пауза: процессор остывает." : "Считается в фоне, статистика дополняется по ходу.") : "");
}

function opponentsProgress(r) {
  return progressBar("Средний оппонент: турниров посчитано", r.total - r.opponents_left, r.total,
    r.opponents_left ? (r.left ? "Начнётся после твоих результатов." : r.cooling ? "Пауза: процессор остывает." : "Считается в фоне.") : "");
}

async function refreshUploads() {
  let info;
  try { info = await (await fetch("/api/uploads")).json(); } catch (e) { return; }
  const t = info.totals;
  uploadEl("dbTotals").innerHTML = `<div class="kv">
    <span>Турниров</span><b>${fmtInt(t.tournaments)}</b><span>со сводкой</span><b>${fmtInt(t.summaries)}</b>
    <span>Раздач</span><b>${fmtInt(t.hands)}</b><span>Сетапов финального стола</span><b>${fmtInt(t.setups)}</b>
    <span>Период</span><b>${fmtDate(t.first)} — ${fmtDate(t.last)}</b></div>`;
  renderQueue(info.queue);
  uploadEl("resultsStatus").innerHTML = resultsProgress(info.results) + opponentsProgress(info.results);
  uploadEl("uploadsTable").innerHTML = info.uploads.length
    ? `<tr><th>Файл</th><th>Загружен</th><th>Новых раздач</th><th>Сводок</th></tr>` + info.uploads.map((u) =>
      `<tr><td>${escapeHtml(u.name)}</td><td>${fmtDate(u.uploaded_at)}</td><td>${fmtInt(u.hands)}</td><td>${fmtInt(u.summaries)}</td></tr>`).join("")
    : `<tr><td class="muted">Пока ничего не загружено</td></tr>`;
}

uploadEl("pickFiles").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
uploadEl("pickFolder").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
const drop = uploadEl("dropzone");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  uploadFiles(e.dataTransfer.files);
});

document.addEventListener("tabshown", (e) => {
  clearInterval(uploadPoll);
  if (e.detail !== "upload") return;
  refreshUploads();
  uploadPoll = setInterval(refreshUploads, 3000);
});
