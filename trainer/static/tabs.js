"use strict";

// tabs by the URL hash: each tab's script listens for "tabshown" with its name
const TABS = ["upload", "stats", "mistakes", "opponents", "trainer"];

function showTab() {
  // "#trainer/<hand id>" opens that hand in the trainer
  const hash = location.hash.slice(1).split("/")[0];
  const name = TABS.includes(hash) ? hash : "trainer";
  for (const t of TABS) document.getElementById(`tab-${t}`).classList.toggle("hidden", t !== name);
  for (const a of document.querySelectorAll("#tabs a")) a.classList.toggle("active", a.dataset.tab === name);
  document.getElementById("statsInline").classList.toggle("hidden", name !== "trainer");
  document.dispatchEvent(new CustomEvent("tabshown", { detail: name }));
}

window.addEventListener("hashchange", showTab);
showTab();
