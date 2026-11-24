"""Configuration for the ad-blocker demo backend.

All paths can be overridden with environment variables so the project is easy to
reconfigure on another machine.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root = the folder that contains this package's parent.
ROOT = Path(__file__).resolve().parent.parent

# Where trained models (.joblib) are stored.
ML_DIR = Path(os.getenv("ML_DIR", str(ROOT / "ml")))
ARTIFACTS_DIR = Path(os.getenv("MODELS_DIR", str(ML_DIR / "artifacts")))

# The ad.names file lists the 1558 feature attributes in order.
NAMES_PATH = Path(os.getenv("AD_NAMES", str(ML_DIR / "dataset" / "ad.names")))

# The three continuous columns the models were trained on (the only ones scaled).
CONTINUOUS_FEATURES = ["height", "width", "aratio"]

# Total feature count the models expect (3 continuous + 1555 binary).
N_FEATURES = 1558

# Keyword heuristic used ONLY in demo mode (no trained models loaded).
# These mirror the strongest ad-indicator tokens found in the dataset's
# anchor-URL vocabulary (adclick, adid, familyid, groupid, ...).
DEMO_KEYWORDS = [
    "adclick", "adid", "familyid", "groupid", "tagvalues", "cid=",
    "doubleclick", "adserver", "/ads/", "advertising", "banner", "redirect",
    "memberbanners", "click?", "adurl",
]
