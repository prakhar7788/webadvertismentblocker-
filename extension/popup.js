const $ = (id) => document.getElementById(id);

async function init() {
  const s = await chrome.storage.local.get({
    apiBase: "http://127.0.0.1:8000",
    enabled: true,
  });
  $("apiBase").value = s.apiBase;
  $("enabled").checked = s.enabled;
}

$("save").addEventListener("click", async () => {
  const apiBase = $("apiBase").value.trim().replace(/\/+$/, "");
  await chrome.storage.local.set({ apiBase, enabled: $("enabled").checked });
  $("status").textContent = "Saved. Reload open pages to apply.";
  setTimeout(() => ($("status").textContent = ""), 2500);
});

$("enabled").addEventListener("change", async () => {
  await chrome.storage.local.set({ enabled: $("enabled").checked });
  $("status").textContent = $("enabled").checked ? "Blocking on" : "Blocking off";
  setTimeout(() => ($("status").textContent = ""), 2000);
});

init();
