// MV3 service worker: classifies URLs by calling the local FastAPI backend.
const DEFAULT_API = "http://127.0.0.1:8000";
const cache = new Map(); // url -> {ad, combined, models}

async function getConfig() {
  return chrome.storage.local.get({ apiBase: DEFAULT_API, enabled: true });
}

async function classifyOne(apiBase, url) {
  if (cache.has(url)) return cache.get(url);
  let rec = null;
  try {
    const res = await fetch(`${apiBase}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (res.ok) {
      const data = await res.json();
      rec = {
        ad: data.combined === "ad",
        combined: data.combined,
        models: (data.results || []).map((r) => `${r.name}: ${r.label}`),
      };
    }
  } catch (e) {
    rec = null; // API unreachable / wrong host -> skip silently
  }
  cache.set(url, rec);
  return rec;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "classify") {
    (async () => {
      const { apiBase, enabled } = await getConfig();
      if (!enabled) {
        sendResponse({ ads: [] });
        return;
      }
      const urls = [...new Set(msg.urls || [])].filter((u) =>
        /^https?:\/\//i.test(u)
      );
      const ads = [];
      const limit = 6; // concurrency cap so we don't hammer the local API
      let idx = 0;
      async function worker() {
        while (idx < urls.length) {
          const u = urls[idx++];
          const r = await classifyOne(apiBase, u);
          if (r && r.ad) ads.push({ url: u, ...r });
        }
      }
      await Promise.all(
        Array.from({ length: Math.min(limit, urls.length) }, worker)
      );
      sendResponse({ ads });
    })();
    return true; // async response
  }

  if (msg.type === "badge") {
    const tabId = sender.tab && sender.tab.id;
    const text =
      msg.count > 0 ? (msg.count > 999 ? "999+" : String(msg.count)) : "";
    if (tabId != null) {
      chrome.action.setBadgeBackgroundColor({ color: "#dc2626", tabId });
      chrome.action.setBadgeText({ text, tabId });
    }
    sendResponse({ ok: true });
    return false;
  }
});
