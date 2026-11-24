"""Convert a single URL into the 1558-feature vector the trained models expect.

The models were trained on the UCI Internet Advertisements data, whose features are:

    height, width, aratio          (3 continuous image dimensions)
    local                          (1 binary: image host == page host)
    url*<token>       (457)        (binary word flags from the image URL)
    origurl*<token>   (495)        (binary word flags from the page URL)
    alt*<token>       (111)        (binary word flags from the image alt text)
    caption*<token>   (19)         (binary word flags from the image caption)
    ancurl*<token>    (472)        (binary word flags from the anchor <a href> URL)

For a demo where the user enters a single URL we do NOT have image dimensions,
alt text, a page URL or a separate anchor URL.  We therefore:

  * set height/width/aratio to NaN  -> the pipeline's SimpleImputer fills them with
    the training-set mean (i.e. a neutral geometry);
  * populate the `url*` and `ancurl*` features from the entered URL (the two
    groups that carry the strongest ad signal);
  * set `origurl*`, `alt*`, `caption*` and `local` to 0 (not available from a URL).

The entered URL is treated as BOTH the image URL and its anchor URL.  This is a
demo assumption, clearly documented in the README and shown on the dashboard.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from . import config

_ATTR_RE = re.compile(r"^\s*(.+?):\s*(?:continuous|0,1)\.")


def _compile_token_pattern(token: str) -> re.Pattern:
    """Build a regex that matches a feature token inside a URL.

    Feature tokens use `+` to join consecutive words (e.g. `ad+gif`, `http+www`).
    Dots are literal (e.g. `www.slake.com`).  Words are matched at word boundaries
    so short tokens like `ads` do not fire inside `uploads`.
    """
    parts = token.split("+")
    inner = r"[^a-z0-9]+".join(re.escape(p) for p in parts)
    return re.compile(r"\b" + inner + r"\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _feature_spec():
    """Return (column_names, binary_instructions).

    binary_instructions is a list of (column_name, compiled_pattern_or_None) for
    each of the 1555 binary columns, in order.  A None pattern means "always 0".
    """
    if not config.NAMES_PATH.exists():
        raise FileNotFoundError(
            f"ad.names not found at {config.NAMES_PATH}. "
            f"Set the AD_NAMES environment variable to its path."
        )

    attr_names: list[str] = []
    for line in config.NAMES_PATH.read_text(encoding="latin-1").splitlines():
        m = _ATTR_RE.match(line)
        if m:
            attr_names.append(m.group(1).strip())

    if len(attr_names) != config.N_FEATURES:
        raise ValueError(
            f"ad.names parsed {len(attr_names)} attributes, "
            f"expected {config.N_FEATURES}."
        )

    # Column names exactly as the trained pipeline expects them.
    columns = config.CONTINUOUS_FEATURES + [
        f"binary_{i}" for i in range(1, config.N_FEATURES - 2)
    ]  # 3 + 1555 = 1558

    instructions: list[tuple[str, re.Pattern | None]] = []
    # Columns 0..2 are continuous.  Columns 3..1557 are binary; column j maps to
    # attribute attr_names[j].
    for j in range(3, config.N_FEATURES):
        col = columns[j]
        attr = attr_names[j]
        if "*" not in attr:
            # 'local' — no page host to compare against -> 0
            instructions.append((col, None))
            continue
        field, token = attr.split("*", 1)
        if field in ("url", "ancurl"):
            instructions.append((col, _compile_token_pattern(token)))
        else:  # origurl, alt, caption -> not available from a URL -> 0
            instructions.append((col, None))

    return columns, instructions


def url_to_features(url: str) -> pd.DataFrame:
    """Build a 1-row DataFrame with the 1558 model-input columns for a URL."""
    columns, instructions = _feature_spec()
    row: dict = {c: float("nan") for c in config.CONTINUOUS_FEATURES}
    for col, pattern in instructions:
        row[col] = 1.0 if (pattern is not None and pattern.search(url)) else 0.0
    return pd.DataFrame([row], columns=columns)


def feature_summary(url: str) -> dict:
    """Return a small summary of which features fired (for debugging / display)."""
    columns, instructions = _feature_spec()
    fired = []
    for col, pattern in instructions:
        if pattern is not None and pattern.search(url):
            fired.append(col)
    return {"n_features_fired": len(fired), "features_fired": fired[:25]}
