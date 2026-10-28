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

const PLAYER_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9];

function renderQueue(q) {
  const jobs = q.jobs || {};
  const sum = (s) => Object.values(jobs[s] || {}).reduce((a, b) => a + b, 0);
  const temp = q.temperature == null ? "" : ` · процессор ${Math.round(q.temperature)}°C`;
  const state = !sum("queued") && !q.running.length ? `всё посчитано${temp}`
    : q.paused ? `<span class="bad">пауза: перегрев, продолжит при ${q.resume_temp}°C</span>${temp}`
    : `<span class="good">считает</span>${temp}`;
  const rows = PLAYER_COUNTS.filter((n) => ["queued", "running", "done", "failed"].some((s) => (jobs[s] || {})[n]))
    .map((n) => `<tr><td>${n}</td>${["queued", "running", "done", "failed"].map((s) => `<td>${fmtInt((jobs[s] || {})[n])}</td>`).join("")}</tr>`);
  uploadEl("queueStatus").innerHTML = `<p>${state}</p>` +
    (rows.length ? `<div class="table-scroll"><table class="data"><tr><th>Игроков</th><th>В очереди</th><th>Считается</th><th>Готово</th><th>Ошибки</th></tr>${rows.join("")}</table></div>` : "") +
    q.running.map((r) => `<p class="hint">Сейчас: ${r.hand_id}, ${r.variant.endsWith("-nooc") ? "без оверколлов" : "с оверколлами"}, ${r.seconds} с</p>`).join("") +
    (q.last_error ? `<p class="hint bad">Последняя ошибка: ${escapeHtml(q.last_error)}</p>` : "");
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
