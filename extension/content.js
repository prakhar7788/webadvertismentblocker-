// Content script: finds resource elements, asks the background to classify each
// URL, and removes the ones the models vote as ads. Runs at document_idle and
// watches for dynamically-added elements (MutationObserver).
const SELECTOR = "img, iframe, video, source, embed, object";
const seen = new Set();      // urls already sent for classification
const removed = new WeakSet(); // elements already replaced
let blockedCount = 0;

function srcOf(el) {
  return el.src || el.getAttribute("data") || el.getAttribute("src") || "";
}

function collectUrls(root) {
  const urls = new Set();
  (root.querySelectorAll ? root.querySelectorAll(SELECTOR) : []).forEach((el) => {
    const s = srcOf(el);
    if (s && /^https?:\/\//i.test(s)) urls.add(s);
  });
  return urls;
}

function removeMatching(adSet) {
  if (!adSet.size) return 0;
  let n = 0;
  document.querySelectorAll(SELECTOR).forEach((el) => {
    const s = srcOf(el);
    if (s && adSet.has(s) && !removed.has(el)) {
      removed.add(el);
      const ph = document.createElement("div");
      ph.className = "mladblock-ph";
      ph.textContent = "🛡️ Ad blocked";
      ph.title = "ML model classified this URL as an ad: " + s;
      el.replaceWith(ph);
      n++;
    }
  });
  return n;
}

function setBadge() {
  chrome.runtime.sendMessage({ type: "badge", count: blockedCount });
}

let toast;
function showToast() {
  if (!toast) {
    toast = document.createElement("div");
    toast.style.cssText =
      "position:fixed;bottom:14px;right:14px;z-index:2147483647;" +
      "background:#dc2626;color:#fff;font:600 13px/1.4 system-ui,sans-serif;" +
      "padding:9px 13px;border-radius:9px;box-shadow:0 4px 14px rgba(0,0,0,.3);";
    (document.body || document.documentElement).appendChild(toast);
  }
  toast.textContent = `🛡️ ML Ad Blocker: blocked ${blockedCount} ad${
    blockedCount === 1 ? "" : "s"
  }`;
  toast.style.display = "block";
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => {
    if (toast) toast.style.display = "none";
  }, 5000);
}

// inject a tiny style for the placeholder
(function () {
  const st = document.createElement("style");
  st.textContent =
    ".mladblock-ph{height:80px;display:flex;align-items:center;justify-content:center;" +
    "background:#fef2f2;color:#b91c1c;border:1px dashed #fca5a5;border-radius:8px;" +
    "font:600 13px system-ui,sans-serif;margin:4px 0}";
  (document.head || document.documentElement).appendChild(st);
})();

function scan(root) {
  const urls = collectUrls(root || document);
  const fresh = [...urls].filter((u) => !seen.has(u));
  fresh.forEach((u) => seen.add(u));
  if (!fresh.length) return;
  chrome.runtime.sendMessage({ type: "classify", urls: fresh }, (resp) => {
    if (chrome.runtime.lastError) return; // background unreachable
    const ads = (resp && resp.ads) || [];
    if (!ads.length) return;
    const adSet = new Set(ads.map((a) => a.url));
    const added = removeMatching(adSet);
    if (added) {
      blockedCount += added;
      setBadge();
      showToast();
    }
  });
}

scan(document);

const mo = new MutationObserver((mutations) => {
  for (const m of mutations) {
    for (const node of m.addedNodes) {
      if (node.nodeType === 1) scan(node);
    }
  }
});
mo.observe(document.documentElement, { childList: true, subtree: true });
