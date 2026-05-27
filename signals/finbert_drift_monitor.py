"""
finbert_drift_monitor

Monitors FinBERT output distribution for drift by analysing the recent
sentiment_cache.jsonl window. Runs daily as a scheduler job (job 9 of 9).

Drift is flagged when any of the following are true:
    1. Stuck class: >95% of readings share the same dominant_class
       (model predictions have collapsed to one label)
    2. Confidence collapse: mean dominant_prob < 0.55
       (model is uncertain across all readings — gate will rarely open)
    3. Low coverage: fewer than 50% of expected hourly readings present
       (GDELT fetches are failing silently)

Public API:
    run_batch_check(window_days=7) -> dict  — called by job_finbert_drift in main.py
    send_telegram(message)                  — from alerts.telegram_util; usable standalone

Usage (standalone test):
    python -c "from signals.finbert_drift_monitor import run_batch_check; \
import json; print(json.dumps(run_batch_check(), indent=2))"
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
_LOG_DIR = _HERE / "logs"
_SENTIMENT_JSONL = _LOG_DIR / "sentiment_cache.jsonl"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from alerts.telegram_util import send_telegram  # noqa: E402

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
_MIN_RECORDS = 12             # fewer records → INSUFFICIENT_DATA (< ~0.5 days)
_STUCK_CLASS_THRESHOLD = 0.95  # >95% same dominant_class → stuck predictions
_MIN_MEAN_PROB = 0.55         # mean dominant_prob below this → confidence collapse
_MIN_COVERAGE_PCT = 0.50      # <50% expected hourly readings → GDELT gaps

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("finbert_drift")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
    fmt.converter = time.gmtime

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(_LOG_DIR / "process.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _setup_logging()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_window(window_days: int) -> list[dict]:
    """
    Read sentiment_cache.jsonl and return records within the last window_days.

    Fields per record (confirmed from live file):
        timestamp, headline_count, dominant_class, dominant_prob,
        sentiment_score, signal, logged_utc
    """
    if not _SENTIMENT_JSONL.exists():
        log.warning("sentiment_cache.jsonl not found at %s", _SENTIMENT_JSONL)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    records: list[dict] = []

    try:
        with open(_SENTIMENT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = datetime.fromisoformat(rec["timestamp"])
                    if ts >= cutoff:
                        records.append(rec)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    except OSError as exc:
        log.error("Cannot read sentiment_cache.jsonl: %s", exc)
        return []

    log.info("Loaded %d sentiment records for %d-day window", len(records), window_days)
    return records


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def _compute_metrics(records: list[dict], window_days: int) -> dict:
    """
    Compute distribution statistics over the record window.

    coverage_pct: fraction of expected hourly readings that are present.
    Expected = window_days * 24 (one reading per hour).
    """
    n = len(records)
    expected = window_days * 24

    classes = [r["dominant_class"] for r in records]
    class_counts = Counter(classes)
    total = len(classes)
    class_dist = {
        "positive": round(class_counts.get("positive", 0) / total, 4),
        "negative": round(class_counts.get("negative", 0) / total, 4),
        "neutral": round(class_counts.get("neutral", 0) / total, 4),
    }

    mean_dominant_prob = round(
        sum(r["dominant_prob"] for r in records) / n, 4
    )
    mean_sentiment_score = round(
        sum(r["sentiment_score"] for r in records) / n, 4
    )
    coverage_pct = round(n / expected, 4)

    return {
        "class_dist": class_dist,
        "mean_dominant_prob": mean_dominant_prob,
        "mean_sentiment_score": mean_sentiment_score,
        "coverage_pct": coverage_pct,
        "drift_reasons": [],  # populated by _check_drift
    }


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
def _check_drift(metrics: dict) -> list[str]:
    """Evaluate threshold conditions. Returns list of triggered reason strings."""
    reasons: list[str] = []

    # Stuck class check
    max_class_frac = max(metrics["class_dist"].values())
    if max_class_frac > _STUCK_CLASS_THRESHOLD:
        dominant = max(metrics["class_dist"], key=metrics["class_dist"].__getitem__)
        reasons.append(
            f"Stuck class: '{dominant}' = {max_class_frac:.0%} of readings "
            f"(threshold: >{_STUCK_CLASS_THRESHOLD:.0%})"
        )

    # Confidence collapse check
    if metrics["mean_dominant_prob"] < _MIN_MEAN_PROB:
        reasons.append(
            f"Confidence collapse: mean_dominant_prob={metrics['mean_dominant_prob']:.4f} "
            f"(floor: {_MIN_MEAN_PROB})"
        )

    # Coverage check
    if metrics["coverage_pct"] < _MIN_COVERAGE_PCT:
        reasons.append(
            f"Low coverage: {metrics['coverage_pct']:.0%} of expected hourly readings "
            f"(floor: {_MIN_COVERAGE_PCT:.0%})"
        )

    return reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_batch_check(window_days: int = 7) -> dict:
    """
    Check FinBERT output distribution for drift over the last window_days.

    Returns a dict with consistent shape regardless of outcome:
        {
            "ts": str,          ISO 8601 UTC timestamp
            "job": "finbert_drift",
            "status": str,      OK | DRIFT_DETECTED | INSUFFICIENT_DATA
            "n": int,           number of records in window
            "metrics": dict | None
        }
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    records = _load_window(window_days)
    n = len(records)

    if n < _MIN_RECORDS:
        log.warning(
            "Insufficient data for drift check: %d records (need >= %d)",
            n, _MIN_RECORDS,
        )
        return {
            "ts": ts,
            "job": "finbert_drift",
            "status": "INSUFFICIENT_DATA",
            "n": n,
            "metrics": None,
        }

    metrics = _compute_metrics(records, window_days)
    drift_reasons = _check_drift(metrics)
    metrics["drift_reasons"] = drift_reasons

    if drift_reasons:
        reasons_text = "\n".join(f"- {r}" for r in drift_reasons)
        msg = (
            f"DRIFT DETECTED in FinBERT outputs ({window_days}-day window, n={n})\n"
            f"{reasons_text}"
        )
        log.warning("FinBERT drift detected: %s", drift_reasons)
        send_telegram(f"[SYNK DRIFT] {msg}")
        status = "DRIFT_DETECTED"
    else:
        log.info(
            "FinBERT drift check OK | n=%d | mean_prob=%.4f | coverage=%.0f%%",
            n, metrics["mean_dominant_prob"], metrics["coverage_pct"] * 100,
        )
        status = "OK"

    return {
        "ts": ts,
        "job": "finbert_drift",
        "status": status,
        "n": n,
        "metrics": metrics,
    }
