"""FastAPI application: load models, predict per model, majority vote, stats."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .features import feature_summary
from .models import load_models
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("adblocker.main")

ROOT = Path(__file__).resolve().parent.parent  # project root
STATIC_DIR = ROOT / "static"

app = FastAPI(title="ML Advertisement Blocker — Demo", version="1.0")

# --- load models once at startup ---------------------------------------------
bundles, missing = load_models(config.ARTIFACTS_DIR)
DEMO_MODE = len(bundles) == 0
if DEMO_MODE:
    log.warning(
        "DEMO MODE: no trained models could be loaded from %s. "
        "Predictions will come from a clearly-labeled keyword heuristic, "
        "NOT from trained models. Missing: %s", config.ARTIFACTS_DIR, missing,
    )
else:
    log.info("Loaded %d model(s): %s", len(bundles), [b.name for b in bundles])
    if missing:
        log.info("Missing models (will not run): %s", missing)

store = Store()


# --- request / response models -----------------------------------------------
class PredictRequest(BaseModel):
    url: str


def _demo_predict(url: str) -> dict:
    """Keyword heuristic used ONLY when no trained models are loaded.

    This is clearly flagged with source='demo' so it is never presented as a
    result from a trained model.
    """
    lowered = url.lower()
    hits = [kw for kw in config.DEMO_KEYWORDS if kw in lowered]
    is_ad = len(hits) > 0
    return {
        "name": "Demo heuristic (no trained models loaded)",
        "label": "ad" if is_ad else "nonad",
        "ad": is_ad,
        "score": round(len(hits) / max(len(config.DEMO_KEYWORDS), 1), 4),
        "threshold": 0.5,
        "score_func": "demo_keyword_match",
        "source": "demo",
        "matched_keywords": hits,
    }


def _majority_vote(results: list[dict]) -> str:
    ad_n = sum(1 for r in results if r["ad"])
    nonad_n = len(results) - ad_n
    # Conservative on ties: if votes are split evenly, do NOT block (nonad).
    return "ad" if ad_n > nonad_n else "nonad"


# --- endpoints ----------------------------------------------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="`url` is required.")

    if DEMO_MODE:
        result = _demo_predict(url)
        results = [result]
        combined = result["label"]
    else:
        results = [b.predict(url) for b in bundles]
        combined = _majority_vote(results)

    models_loaded = [b.name for b in bundles]
    store.add(url, results, combined, DEMO_MODE, models_loaded)

    return {
        "url": url,
        "combined": combined,
        "results": results,
        "demo_mode": DEMO_MODE,
        "models_loaded": models_loaded,
        "missing_models": missing,
        "features": feature_summary(url),
    }


@app.get("/api/stats")
def stats():
    data = store.stats()
    data["demo_mode"] = DEMO_MODE
    data["models_loaded"] = [b.name for b in bundles]
    data["missing_models"] = missing
    return data


@app.get("/api/blocked")
def blocked():
    items = store.blocked()
    return {"count": len(items), "blocked": items}


@app.get("/api/recent")
def recent():
    return {"recent": store.recent()}


# --- pages -------------------------------------------------------------------
@app.get("/")
def index_page():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail=f"static/index.html missing at {index}")
    return FileResponse(index)


@app.get("/blocked")
def blocked_page():
    page = STATIC_DIR / "blocked.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail=f"static/blocked.html missing at {page}")
    return FileResponse(page)


@app.get("/demo")
def demo_page():
    page = STATIC_DIR / "demo.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail=f"static/demo.html missing at {page}")
    return FileResponse(page)


# mount static assets (css/js) if the directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    log.warning("Static directory %s does not exist; CSS/JS will 404.", STATIC_DIR)
