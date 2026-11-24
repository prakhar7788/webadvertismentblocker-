"""Tiny in-memory store of checked URLs and detected ads.

Everything is kept in process memory, so counts reset when the server restarts.
This is intentional for a demo — no database is needed.
"""
from __future__ import annotations

from threading import Lock
from datetime import datetime


class Store:
    def __init__(self, keep_recent: int = 100):
        self._lock = Lock()
        self._total = 0
        self._ads = 0
        self._blocked: list[dict] = []     # urls classified as ad (newest last)
        self._recent: list[dict] = []
        self._keep_recent = keep_recent

    def add(self, url, results, combined, demo_mode, models_loaded):
        record = {
            "url": url,
            "combined": combined,
            "results": results,
            "demo_mode": demo_mode,
            "models_loaded": models_loaded,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            self._total += 1
            if combined == "ad":
                self._ads += 1
                self._blocked.append({
                    "url": url,
                    "time": record["time"],
                    "combined": combined,
                    "results": results,
                    "demo_mode": demo_mode,
                })
            self._recent.append(record)
            if len(self._recent) > self._keep_recent:
                self._recent = self._recent[-self._keep_recent:]

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_checked": self._total,
                "ads_detected": self._ads,
                "blocked_count": len(self._blocked),
            }

    def blocked(self) -> list[dict]:
        with self._lock:
            # newest first, copy so callers can't mutate internal state
            return list(reversed(self._blocked))

    def recent(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._recent))
