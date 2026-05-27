"""
health_monitor

Operational health checks for the Synk bot. Validates data freshness by
inspecting file modification times — no Alpaca API calls, no network I/O.
Called inside main.py's hourly cycle; results appended to logs/health.jsonl.

Checks:
    1. Sentiment cache     < 90 min old  (covers GDELT fetch + FinBERT run)
    2. Price cache         < 25 h old    (one file per symbol, checks GLD as proxy)
    3. GPR daily file      < 25 h old
    4. Kill switch state   readable and not HALTED

Usage:
    from alerts.health_monitor import check_health, HealthStatus
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
_LOG_DIR = _HERE / "logs"
_HEALTH_JSONL = _LOG_DIR / "health.jsonl"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv  # noqa: E402

from alerts.telegram_util import send_telegram  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Tuneable thresholds (seconds)
# ---------------------------------------------------------------------------
_SENTIMENT_MAX_AGE = 90 * 60       # 90 min: sentiment + GDELT freshness proxy
_PRICE_MAX_AGE = 25 * 3600         # 25 h:   price cache (daily bars)
_GPR_MAX_AGE = 25 * 3600           # 25 h:   GPR XLS file


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("health_monitor")
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
# Domain type
# ---------------------------------------------------------------------------
@dataclass
class HealthStatus:
    timestamp: str
    all_healthy: bool
    issues: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _file_age_seconds(path: Path) -> float | None:
    """Return seconds since path was last modified, or None if missing."""
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _check_sentiment_freshness(issues: list[str]) -> None:
    age = _file_age_seconds(_LOG_DIR / "sentiment_cache.jsonl")
    if age is None:
        issues.append("sentiment_cache.jsonl missing — sentiment job has not run")
    elif age > _SENTIMENT_MAX_AGE:
        issues.append(
            f"Sentiment cache stale: {age/60:.0f} min old (limit: {_SENTIMENT_MAX_AGE//60} min)"
        )


def _check_price_freshness(issues: list[str]) -> None:
    # GLD is the anchor — if it's fresh, the batch fetch ran
    gld_path = _HERE / "data" / "price_cache" / "GLD.csv"
    age = _file_age_seconds(gld_path)
    if age is None:
        issues.append("Price cache missing — price_feed job has not run")
    elif age > _PRICE_MAX_AGE:
        issues.append(
            f"Price cache stale: {age/3600:.1f} h old (limit: {_PRICE_MAX_AGE//3600} h)"
        )


def _check_gpr_freshness(issues: list[str]) -> None:
    age = _file_age_seconds(_HERE / "data" / "gpr_daily_recent.xls")
    if age is None:
        issues.append("GPR file missing — download_gpr_daily() has not run")
    elif age > _GPR_MAX_AGE:
        issues.append(
            f"GPR file stale: {age/3600:.1f} h old (limit: {_GPR_MAX_AGE//3600} h)"
        )


def _check_kill_switch(issues: list[str]) -> None:
    ks_path = _LOG_DIR / "kill_switch_state.json"
    if not ks_path.exists():
        # State file absent = kill switch never initialised — not an error at startup
        return
    try:
        state = json.loads(ks_path.read_text(encoding="utf-8"))
        if state.get("state") == "HALTED":
            reason = state.get("halt_reason", "unknown")
            issues.append(f"Kill switch HALTED: {reason}")
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(f"Kill switch state file unreadable: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_health() -> HealthStatus:
    """
    Run all health checks and return a HealthStatus.
    Never raises — individual check failures are captured as issues.
    """
    issues: list[str] = []
    _check_sentiment_freshness(issues)
    _check_price_freshness(issues)
    _check_gpr_freshness(issues)
    _check_kill_switch(issues)

    status = HealthStatus(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        all_healthy=len(issues) == 0,
        issues=issues,
    )

    if status.all_healthy:
        log.info("Health check: OK")
    else:
        for issue in issues:
            log.warning("Health issue: %s", issue)
        issues_text = "\n".join(f"- {i}" for i in issues)
        send_telegram(f"\u26a0\ufe0f *SYNK HEALTH ALERT*\n{issues_text}")

    return status


def append_to_jsonl(status: HealthStatus, path: Path = _HEALTH_JSONL) -> None:
    """Append a HealthStatus record to logs/health.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(status.as_dict()) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    status = check_health()
    print(f"\nHealth: {'OK' if status.all_healthy else 'ISSUES FOUND'}")
    if status.issues:
        for issue in status.issues:
            print(f"  - {issue}")
    else:
        print("  All checks passed.")
    append_to_jsonl(status)
