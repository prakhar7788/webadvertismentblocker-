const $ = (id) => document.getElementById(id);

async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();
    $("total").textContent = s.total_checked;
    $("ads").textContent = s.ads_detected;
    $("blocked-count").textContent = s.blocked_count;

    // status / demo banners
    if (s.demo_mode) {
      $("demo-banner").classList.remove("hidden");
      $("demo-banner").innerHTML =
        "<b>DEMO MODE — no trained models loaded.</b> Predictions come from a " +
        "keyword heuristic, <u>not</u> from trained models. See the README to add " +
        "model files to <code>ml/artifacts/</code>.";
    } else {
      $("status-banner").classList.remove("hidden");
      const loaded = s.models_loaded.map((m) => `<li><b>${m}</b> ✓</li>`).join("");
      const missing = s.missing_models.length
        ? s.missing_models.map((m) => `<li>${m} — not loaded</li>`).join("")
        : "";
      $("status-banner").innerHTML =
        `<b>Models loaded (${s.models_loaded.length}/4):</b><ul>${loaded}</ul>` +
        (missing ? `<b>Missing:</b><ul>${missing}</ul>` : "");
    }
  } catch (e) {
    console.error(e);
  }
}

function renderResult(data) {
  $("result").classList.remove("hidden");
  $("result-url").textContent = data.url;

  // combined decision
  const cb = $("combined");
  cb.className = "combined " + data.combined;
  const voteAd = data.results.filter((r) => r.ad).length;
  const voteNonad = data.results.length - voteAd;
  cb.innerHTML =
    `<span class="badge">${data.combined === "ad" ? "BLOCK" : "ALLOW"}</span>` +
    `<span class="verdict">Combined decision: <b>${data.combined}</b>` +
    (data.demo_mode ? " (demo heuristic)" : ` &nbsp;·&nbsp; vote ${voteAd} ad / ${voteNonad} nonad`) +
    `</span>`;

  // per-model cards
  $("models").innerHTML = data.results.map((r) => {
    const cls = r.source === "demo" ? "model demo" : "model";
    const scoreTxt =
      r.score_func === "demo_keyword_match"
        ? `matched: ${(r.matched_keywords || []).join(", ") || "—"}`
        : `score ${r.score} ≥ ${r.threshold} (${r.score_func})`;
    return `<div class="${cls}">
      <span class="mname">${r.name}</span>
      <span class="mscore">${scoreTxt}</span>
      <span></span>
      <span class="mlabel ${r.label}">${r.label}</span>
    </div>`;
  }).join("");

  // meta line
  const f = data.features || {};
  const parts = [
    data.demo_mode ? "source: <b>demo heuristic</b>" : `source: <b>trained models</b>`,
    f.n_features_fired != null ? `features fired: <b>${f.n_features_fired}</b>/1555` : null,
    data.missing_models.length ? `not loaded: ${data.missing_models.join(", ")}` : null,
  ].filter(Boolean);
  $("meta").innerHTML = parts.join(" &nbsp;·&nbsp; ");
}

async function checkUrl(url) {
  $("error").classList.add("hidden");
  $("check").disabled = true;
  $("check").textContent = "Checking…";
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    renderResult(data);
    loadStats();
  } catch (e) {
    $("error").textContent = "Error: " + e.message;
    $("error").classList.remove("hidden");
  } finally {
    $("check").disabled = false;
    $("check").textContent = "Check";
  }
}

$("check").addEventListener("click", () => {
  const url = $("url").value.trim();
  if (url) checkUrl(url);
});
$("url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("check").click();
});
document.querySelectorAll(".sample").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("url").value = btn.dataset.url;
    checkUrl(btn.dataset.url);
  });
});

loadStats();
