"use strict";

// tabs by the URL hash: each tab's script listens for "tabshown" with its name
const TABS = ["upload", "stats", "trainer"];

function showTab() {
  const name = TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "trainer";
  for (const t of TABS) document.getElementById(`tab-${t}`).classList.toggle("hidden", t !== name);
  for (const a of document.querySelectorAll("#tabs a")) a.classList.toggle("active", a.dataset.tab === name);
  document.getElementById("statsInline").classList.toggle("hidden", name !== "trainer");
  document.dispatchEvent(new CustomEvent("tabshown", { detail: name }));
}

window.addEventListener("hashchange", showTab);
showTab();
