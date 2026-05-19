"""
weekly_review

Standalone Sunday report. Reads existing .jsonl log files, computes the week's
performance and system health, and sends one structured Telegram message.

No Alpaca interaction. No scheduler dependency. Read-only — writes nothing.

Run:
    python weekly_review.py            # send Telegram report
    python weekly_review.py --dry-run  # print to console, no Telegram

Schedule: Sunday 19:00 local time via Task Scheduler.
Deps: requests, python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved from this file so Task Scheduler cwd doesn't matter
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # synk/ root
_LOG_DIR = _HERE / "logs"

# Required so `from signals.finbert_drift_monitor import ...` resolves
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Constants — all thresholds live here, nowhere else
# ---------------------------------------------------------------------------
_WINDOW_DAYS = 7
_GLD_WR_TARGET = 0.40
_FXY_WR_TARGET = 0.35
_PF_TARGET = 1.5
_MAX_DD_LIMIT = 0.30
_DAILY_LOSS_LIMIT = 0.05
_GDELT_LAG_WARN_MIN = 20
_TRADE_COUNT_TARGET = 50
_HEARTBEAT_STALE_MIN = 15   # heartbeat older than this → watchdog warning
_HEALTH_STALE_H = 4         # no health check within this window → bot likely down
_TELEGRAM_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Logging — console only (script runs once and exits, no file needed)
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("weekly_review")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
    fmt.converter = time.gmtime
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


log = _setup_logging()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_ts(ts: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp string; return UTC-aware datetime or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _cutoff(window_days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=window_days)


def is_last_sunday_of_month(date: datetime) -> bool:
    """True if date is a Sunday and the last Sunday of its calendar month."""
    if date.weekday() != 6:  # 6 = Sunday
        return False
    return (date + timedelta(days=7)).month != date.month


def _compute_profit_factor(closed: list[dict]) -> float:
    """Gross profit / gross loss across closed trades. Returns inf if no losses."""
    if not closed:
        return 0.0
    gross_profit = sum(float(t["pnl_pct"]) for t in closed if float(t["pnl_pct"]) > 0)
    gross_loss = sum(abs(float(t["pnl_pct"])) for t in closed if float(t["pnl_pct"]) < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _portfolio_impact(trade: dict) -> float:
    """
    Portfolio-level return contribution of a single closed trade.
    allocation_pct is stored as decimal (0.04 = 4%).
    pnl_pct is stored as percentage-points (2.5 = 2.5% of position value).
    Result: decimal fraction of NAV (e.g. 0.001 = 0.1% NAV).
    """
    alloc = float(trade.get("allocation_pct") or 0.04)
    pnl = float(trade["pnl_pct"])
    return alloc * pnl / 100


# ---------------------------------------------------------------------------
# Data loaders — never raise; return [] on missing or corrupt file
# ---------------------------------------------------------------------------
def load_trades(window_days: int = _WINDOW_DAYS) -> list[dict]:
    """
    Load trades from logs/trades.jsonl.

    Includes a trade if its exit_timestamp falls within the window,
    OR if the trade is still open and was entered within the window.
    """
    path = _LOG_DIR / "trades.jsonl"
    if not path.exists():
        return []
    cut = _cutoff(window_days)
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    exit_ts = _parse_ts(rec.get("exit_timestamp"))
                    entry_ts = _parse_ts(rec.get("timestamp"))
                    if exit_ts is not None and exit_ts >= cut:
                        records.append(rec)
                    elif exit_ts is None and entry_ts is not None and entry_ts >= cut:
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        log.warning("Could not read trades.jsonl: %s", exc)
    return records


def load_signals(window_days: int = _WINDOW_DAYS) -> list[dict]:
    """
    Load signal evaluation records from logs/signals.jsonl.
    Not yet implemented — signals.jsonl not written by the live pipeline.
    Returns empty list to satisfy the function contract.
    """
    path = _LOG_DIR / "signals.jsonl"
    if not path.exists():
        return []
    cut = _cutoff(window_days)
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = _parse_ts(rec.get("timestamp") or rec.get("logged_utc"))
                    if ts is None or ts >= cut:
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        log.warning("Could not read signals.jsonl: %s", exc)
    return records


def load_gdelt_events(window_days: int = _WINDOW_DAYS) -> list[dict]:
    """
    Load GDELT lag records from logs/gdelt_lag_results.json.

    This file is a JSON array (not JSONL). Field: lag_minutes (already in minutes).
    """
    path = _LOG_DIR / "gdelt_lag_results.json"
    if not path.exists():
        return []
    cut = _cutoff(window_days)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [
            rec for rec in data
            if (ts := _parse_ts(rec.get("check_time_utc"))) is None or ts >= cut
        ]
    except Exception as exc:
        log.warning("Could not read gdelt_lag_results.json: %s", exc)
        return []


def load_health_records(window_days: int = _WINDOW_DAYS) -> list[dict]:
    """Load health monitor records from logs/health.jsonl."""
    path = _LOG_DIR / "health.jsonl"
    if not path.exists():
        return []
    cut = _cutoff(window_days)
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = _parse_ts(rec.get("timestamp"))
                    if ts is None or ts >= cut:
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        log.warning("Could not read health.jsonl: %s", exc)
    return records


# ---------------------------------------------------------------------------
# Compute functions — each returns a dict with a 'status' key
# ---------------------------------------------------------------------------
def compute_performance(trades: list[dict]) -> dict:
    """
    Compute weekly performance from trade records.

    pnl_pct is stored as percentage-points (2.5 = +2.5% position return).
    allocation_pct is stored as decimal (0.04 = 4% of NAV).
    All returned values use decimal fractions for portfolio-level figures.
    """
    _empty: dict = {
        "status": "NO_DATA",
        "trade_count": 0,
        "win_count": 0,
        "win_rate": 0.0,
        "total_portfolio_pnl": 0.0,
        "max_drawdown": 0.0,
        "worst_day_pnl": 0.0,
        "sh_hostile_exits": 0,
        "ks_exits": 0,
        "gld_win_rate": 0.0,
        "fxy_win_rate": 0.0,
    }

    if not trades:
        return _empty

    closed = [
        t for t in trades
        if t.get("exit_timestamp") and t.get("pnl_pct") is not None
    ]

    if not closed:
        return {**_empty, "status": "NO_CLOSED_TRADES", "trade_count": len(trades)}

    impacts = [_portfolio_impact(t) for t in closed]
    wins = [p for p in impacts if p > 0]
    total_portfolio_pnl = sum(impacts)

    # Peak-to-trough drawdown on running cumulative equity curve
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for impact in impacts:
        equity += impact
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Worst single-day portfolio loss (sum of impacts per exit day)
    daily: dict[str, float] = {}
    for t, impact in zip(closed, impacts):
        day = (t.get("exit_timestamp") or "")[:10]
        daily[day] = daily.get(day, 0.0) + impact
    worst_day_pnl = min(daily.values()) if daily else 0.0

    # Exit reason counts — requires exit_reason field (written since order_executor fix)
    def _reason(t: dict) -> str:
        return (t.get("exit_reason") or "").lower()

    sh_exits = sum(1 for t in closed if _reason(t) == "sh_hostile_regime")
    ks_exits = sum(1 for t in closed if "kill" in _reason(t) or "halt" in _reason(t))

    # Per-symbol win rates
    def _sym_wr(sym: str) -> float:
        sym_closed = [t for t in closed if t.get("symbol") == sym]
        if not sym_closed:
            return 0.0
        return sum(1 for t in sym_closed if float(t["pnl_pct"]) > 0) / len(sym_closed)

    return {
        "status": "OK",
        "trade_count": len(closed),
        "win_count": len(wins),
        "win_rate": len(wins) / len(closed),
        "total_portfolio_pnl": total_portfolio_pnl,
        "max_drawdown": max_dd,
        "worst_day_pnl": worst_day_pnl,
        "sh_hostile_exits": sh_exits,
        "ks_exits": ks_exits,
        "gld_win_rate": _sym_wr("GLD"),
        "fxy_win_rate": _sym_wr("FXY"),
    }


def compute_gdelt_lag(events: list[dict]) -> dict:
    """Compute median GDELT lag (in minutes) for the window."""
    _empty: dict = {"status": "NO_DATA", "median_lag_min": 0.0, "record_count": 0, "flagged": False}

    if not events:
        return _empty

    lags = sorted(float(e["lag_minutes"]) for e in events if "lag_minutes" in e)
    if not lags:
        return _empty

    mid = len(lags) // 2
    median = lags[mid] if len(lags) % 2 == 1 else (lags[mid - 1] + lags[mid]) / 2

    return {
        "status": "OK",
        "median_lag_min": round(median, 1),
        "record_count": len(lags),
        "flagged": median > _GDELT_LAG_WARN_MIN,
    }


def compute_system_health() -> dict:
    """
    Read heartbeat.json, kill_switch_state.json, watchdog_state.json.
    Returns health state dict. Never raises.
    """
    result: dict = {
        "status": "OK",
        "heartbeat_age_min": None,
        "watchdog_ok": False,
        "kill_switch_state": "UNKNOWN",
        "halt_reason": None,
        "peak_equity": 0.0,
        "health_issues": [],
    }

    # Heartbeat freshness
    hb_path = _LOG_DIR / "heartbeat.json"
    if hb_path.exists():
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            last_alive = _parse_ts(hb.get("last_alive"))
            if last_alive:
                age_min = (datetime.now(timezone.utc) - last_alive).total_seconds() / 60
                result["heartbeat_age_min"] = round(age_min, 1)
                result["watchdog_ok"] = age_min <= _HEARTBEAT_STALE_MIN
        except Exception as exc:
            log.warning("Could not read heartbeat.json: %s", exc)

    # Kill switch state
    ks_path = _LOG_DIR / "kill_switch_state.json"
    if ks_path.exists():
        try:
            ks = json.loads(ks_path.read_text(encoding="utf-8"))
            result["kill_switch_state"] = ks.get("state", "UNKNOWN")
            result["halt_reason"] = ks.get("halt_reason")
            result["peak_equity"] = float(ks.get("peak_equity") or 0)
        except Exception as exc:
            log.warning("Could not read kill_switch_state.json: %s", exc)

    return result


def get_finbert_drift(window_days: int = _WINDOW_DAYS) -> dict:
    """
    Call check_drift() from signals/finbert_drift_monitor.py if the module exists.
    Returns a status dict in all cases — never raises.
    """
    try:
        from signals.finbert_drift_monitor import check_drift  # type: ignore[import]
        result = check_drift(window_days=window_days)
        if not isinstance(result, dict):
            return {"status": "INVALID_RETURN", "alerts": []}
        return result
    except ImportError:
        log.info("finbert_drift_monitor not found — FinBERT drift check unavailable")
        return {"status": "MODULE_NOT_FOUND", "alerts": []}
    except Exception as exc:
        log.warning("check_drift() error: %s", exc)
        return {"status": "ERROR", "alerts": [str(exc)]}


# ---------------------------------------------------------------------------
# Current health helper
# ---------------------------------------------------------------------------
def _current_health_issues(health_records: list[dict], now: datetime) -> list[str]:
    """
    Return health issues reflecting the bot's current state — not the full
    7-day history. Checks only the most recent record; warns if no record
    within _HEALTH_STALE_H hours (bot likely not running).
    """
    if not health_records:
        return ["No health records — bot has not run this week"]
    most_recent = max(
        health_records,
        key=lambda r: _parse_ts(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    last_ts = _parse_ts(most_recent.get("timestamp"))
    if last_ts and (now - last_ts).total_seconds() > _HEALTH_STALE_H * 3600:
        age_h = (now - last_ts).total_seconds() / 3600
        return [f"Health monitor last seen {age_h:.1f}h ago — bot may not be running"]
    return list(most_recent.get("issues") or [])


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def _compute_verdict(
    perf: dict,
    drift: dict,
    gdelt: dict,
    health: dict,
    health_records: list[dict],
) -> str:
    """Overall one-line verdict. Priority: RED > AMBER > GREEN."""
    # RED conditions
    if health["kill_switch_state"] == "HALTED":
        return "🔴 RED — Kill switch HALTED"
    if perf.get("ks_exits", 0) > 0:
        return f"🔴 RED — Kill switch exits: {perf['ks_exits']}"
    worst_day = abs(perf.get("worst_day_pnl", 0.0))
    if worst_day >= _DAILY_LOSS_LIMIT * 0.5 and worst_day > 0:
        return f"🔴 RED — Daily loss {worst_day:.1%} (≥50% of 5% limit)"
    if perf.get("max_drawdown", 0.0) >= _MAX_DD_LIMIT * 0.5:
        return f"🔴 RED — Drawdown {perf['max_drawdown']:.1%} (≥50% of 30% limit)"
    if drift.get("status") == "DRIFT_DETECTED":
        return "🔴 RED — FinBERT drift detected"

    # AMBER conditions
    has_health_issues = bool(_current_health_issues(health_records, datetime.now(timezone.utc)))
    drift_status = drift.get("status", "UNKNOWN")
    if (
        perf.get("sh_hostile_exits", 0) > 0
        or not health.get("watchdog_ok", True)
        or gdelt.get("flagged")
        or has_health_issues
        or (drift.get("alerts") or [])
        or drift_status not in ("OK", "INSUFFICIENT_DATA", "MODULE_NOT_FOUND", "NO_DATA")
    ):
        return "🟡 AMBER — Review attention items"

    return "🟢 GREEN — System nominal"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def build_month_end_block(all_trades: list[dict]) -> str:
    """Build the RAG month-end checklist from the full trade history."""
    closed = [
        t for t in all_trades
        if t.get("exit_timestamp") and t.get("pnl_pct") is not None
    ]

    if not closed:
        return (
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "📋 *MONTH-END CHECKLIST*\n"
            "No closed trades — cannot assess."
        )

    perf = compute_performance(closed)
    pf = _compute_profit_factor(closed)

    def _rag(value: float, target: float, higher_is_better: bool = True) -> str:
        if higher_is_better:
            if value >= target:
                return "🟢"
            return "🟡" if value >= target * 0.85 else "🔴"
        else:
            if value <= target * 0.5:
                return "🟢"
            return "🟡" if value <= target * 0.75 else "🔴"

    gld_wr = perf["gld_win_rate"]
    fxy_wr = perf["fxy_win_rate"]
    max_dd = perf["max_drawdown"]
    n = perf["trade_count"]
    pf_display = f"{pf:.2f}" if pf != float("inf") else "∞"

    rows = [
        (f"GLD WR {gld_wr:.1%} vs 40%", _rag(gld_wr, _GLD_WR_TARGET)),
        (f"FXY WR {fxy_wr:.1%} vs 35%", _rag(fxy_wr, _FXY_WR_TARGET)),
        (f"Profit factor {pf_display} vs 1.5", _rag(pf if pf != float("inf") else 999, _PF_TARGET)),
        (f"Max DD {max_dd:.1%} vs 30%", _rag(max_dd, _MAX_DD_LIMIT, higher_is_better=False)),
        (f"Trades {n} vs 50", _rag(n, _TRADE_COUNT_TARGET)),
    ]

    reds = sum(1 for _, r in rows if r == "🔴")
    yellows = sum(1 for _, r in rows if r == "🟡")
    if reds == 0 and yellows <= 1:
        verdict = "✅ PASS"
    elif reds == 0:
        verdict = "🟡 WATCHLIST"
    else:
        verdict = "🔴 FAIL"

    lines = ["━━━━━━━━━━━━━━━━━━━━━━", "📋 *MONTH-END CHECKLIST*"]
    lines += [f"{label}: {rag}" for label, rag in rows]
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines)


def build_report(
    perf: dict,
    drift: dict,
    gdelt: dict,
    health: dict,
    health_records: list[dict],
    month_end_block: str,
    today: datetime,
) -> str:
    date_str = today.strftime("%Y-%m-%d")
    sep = "━━━━━━━━━━━━━━━━━━━━━━"

    # --- Section 1: Performance ---
    if perf["status"] == "NO_DATA":
        perf_block = "No trades data — logs/trades.jsonl not found yet."
    elif perf["status"] == "NO_CLOSED_TRADES":
        perf_block = f"Open trades: {perf['trade_count']} | No closed trades this week."
    else:
        worst_day_used = abs(perf["worst_day_pnl"]) / _DAILY_LOSS_LIMIT
        # Account drawdown: without Alpaca, approximate from kill switch peak equity
        # Positive max_drawdown = how far below running peak we fell intra-week
        acct_dd = perf["max_drawdown"]
        perf_block = (
            f"Trades: {perf['trade_count']} | Wins: {perf['win_count']} | WR: {perf['win_rate']:.1%}\n"
            f"PnL: {perf['total_portfolio_pnl']:+.2%} | Max DD: {perf['max_drawdown']:.2%}\n"
            f"Kill switch: Daily {worst_day_used:.1%}/5% | Account {acct_dd:.1%}/30%"
        )

    # --- Section 2: System Health ---
    drift_status = drift.get("status", "UNKNOWN")
    drift_alerts = drift.get("alerts") or []
    if drift_status == "MODULE_NOT_FOUND":
        drift_line = "finbert_drift_monitor.py not built yet"
    elif drift_status == "INSUFFICIENT_DATA":
        drift_line = "INSUFFICIENT_DATA"
    else:
        drift_line = drift_status
        if drift_alerts:
            drift_line += f" | {drift_alerts[0]}"

    age_min = health.get("heartbeat_age_min")
    age_str = f"{age_min:.0f} min" if age_min is not None else "unknown"

    if gdelt["status"] == "OK":
        lag_flag = "⚠️ ABOVE THRESHOLD" if gdelt["flagged"] else "✅"
        lag_line = f"{gdelt['median_lag_min']:.1f} min {lag_flag} (n={gdelt['record_count']})"
    else:
        lag_line = "NO_DATA"

    watchdog_str = "✅ OK" if health["watchdog_ok"] else f"⚠️ STALE (>{_HEARTBEAT_STALE_MIN} min)"
    ks_state = health["kill_switch_state"]
    ks_str = "✅ ACTIVE" if ks_state == "ACTIVE" else f"🔴 {ks_state}"
    if health.get("halt_reason"):
        ks_str += f" ({health['halt_reason']})"

    health_block = (
        f"FinBERT: {drift_line}\n"
        f"Heartbeat age: {age_str}\n"
        f"GDELT lag (7d median): {lag_line}\n"
        f"Watchdog: {watchdog_str} | Kill switch: {ks_str}"
    )

    # --- Section 3: Attention items ---
    attention_lines: list[str] = []

    if health["kill_switch_state"] == "HALTED":
        reason = health.get("halt_reason") or "unknown"
        attention_lines.append(f"🔴 Kill switch HALTED: {reason}")
    elif perf.get("ks_exits", 0) > 0:
        attention_lines.append(f"🔴 Kill switch exits this week: {perf['ks_exits']}")

    if perf.get("sh_hostile_exits", 0) > 0:
        attention_lines.append(f"⚠️ SH_HOSTILE_REGIME exits: {perf['sh_hostile_exits']}")

    for alert in drift_alerts:
        attention_lines.append(f"⚠️ FinBERT drift: {alert}")

    for issue in _current_health_issues(health_records, today)[:3]:
        attention_lines.append(f"⚠️ Health: {issue}")

    if not health["watchdog_ok"] and age_min is not None:
        attention_lines.append(f"⚠️ Heartbeat stale: {age_str}")

    if gdelt.get("flagged"):
        attention_lines.append(
            f"⚠️ GDELT lag elevated: {gdelt['median_lag_min']:.1f} min "
            f"(threshold: {_GDELT_LAG_WARN_MIN} min)"
        )

    attention_block = "\n".join(attention_lines) if attention_lines else "✅ No attention items"

    # --- Assemble ---
    parts = [
        f"⚙️ *Synk Weekly Review — {date_str}*",
        f"\n{sep}",
        "📊 *PERFORMANCE*",
        perf_block,
        f"\n{sep}",
        "🔧 *SYSTEM HEALTH*",
        health_block,
        f"\n{sep}",
        "⚠️ *ATTENTION ITEMS*",
        attention_block,
    ]

    if month_end_block:
        parts.append(f"\n{month_end_block}")

    verdict = _compute_verdict(perf, drift, gdelt, health, health_records)
    parts.append(f"\n{sep}")
    parts.append(f"🏁 *VERDICT*\n{verdict}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Telegram sender
# ---------------------------------------------------------------------------
def send_telegram(message: str) -> bool:
    """Send message to Telegram. Returns True on success."""
    import requests  # deferred: only needed when actually sending

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.error("Telegram not configured — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=_TELEGRAM_TIMEOUT,
        )
        resp.raise_for_status()
        log.info("Telegram report sent (%d chars)", len(message))
        return True
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> None:
    today = datetime.now(timezone.utc)
    log.info(
        "Weekly review starting | %s | dry_run=%s",
        today.strftime("%Y-%m-%d %H:%M UTC"), dry_run,
    )

    # Load data
    trades_7d = load_trades(_WINDOW_DAYS)
    gdelt_events = load_gdelt_events(_WINDOW_DAYS)
    health_records = load_health_records(_WINDOW_DAYS)

    log.info(
        "Loaded: %d trades | %d GDELT events | %d health records",
        len(trades_7d), len(gdelt_events), len(health_records),
    )

    # Compute metrics
    perf = compute_performance(trades_7d)
    gdelt = compute_gdelt_lag(gdelt_events)
    health = compute_system_health()
    drift = get_finbert_drift(_WINDOW_DAYS)

    log.info(
        "Metrics: perf=%s | gdelt=%s | ks=%s | drift=%s",
        perf["status"], gdelt["status"],
        health["kill_switch_state"], drift["status"],
    )

    # Month-end block — only on last Sunday of month
    month_end_block = ""
    if is_last_sunday_of_month(today):
        log.info("Last Sunday of month — building month-end checklist")
        all_trades = load_trades(365)  # full history for month-end assessment
        month_end_block = build_month_end_block(all_trades)

    report = build_report(perf, drift, gdelt, health, health_records, month_end_block, today)

    if dry_run:
        print("\n" + "=" * 52)
        print("DRY RUN — report not sent to Telegram")
        print("=" * 52)
        print(report)
        print("=" * 52 + "\n")
        log.info("Dry run complete — %d chars", len(report))
    else:
        if not send_telegram(report):
            log.error("Failed to send weekly review — check Telegram credentials")
            sys.exit(1)

    log.info("Weekly review complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Force UTF-8 on Windows console so emoji characters don't crash dry-run output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Synk weekly performance and health review")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report to console instead of sending Telegram",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(_HERE / ".env")
    except ImportError:
        log.warning("python-dotenv not installed — reading Telegram creds from environment")

    main(dry_run=args.dry_run)
