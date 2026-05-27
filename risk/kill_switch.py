"""
kill_switch

Enforces Synk's three hard risk limits. Checked every 15 minutes by the
watchdog daemon and queried by synk_strategy.py before any TradeInstruction
is acted on.

Three triggers (hard limits, not soft warnings):
    1. Per-trade stop-loss:  2% of portfolio NAV on any single position
    2. Daily loss limit:     5% of NAV (equity vs last_equity from Alpaca)
    3. Peak drawdown limit: 30% from high-water mark (tracked in state file)

State persistence:
    logs/kill_switch_state.json — atomic write via os.replace(), never partial.
    Fields: state (ACTIVE/HALTED), peak_equity, halted_at, halt_reason,
            last_checked_utc.

On trigger:
    - State set to HALTED (persisted immediately)
    - Telegram alert sent if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set
    - All further kill_switch_active() calls return True, blocking strategy

Recovery:
    Manual only. Delete or edit kill_switch_state.json to reset to ACTIVE.
    No auto-reset — intentional.

Usage:
    from risk.kill_switch import kill_switch_active, check_triggers
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
_STATE_PATH = _LOG_DIR / "kill_switch_state.json"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv  # noqa: E402

from alerts.telegram_util import send_telegram  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Tuneable constants — hard limits, change only with deliberate intent
# ---------------------------------------------------------------------------
_STOP_LOSS_PCT = 0.02       # trigger 1: 2% max loss per trade position
_DAILY_LOSS_PCT = 0.05      # trigger 2: 5% daily NAV drawdown
_PEAK_DRAWDOWN_PCT = 0.30   # trigger 3: 30% from portfolio high-water mark

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kill_switch")
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
# Domain types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KillStatus:
    triggered: bool
    reason: str | None      # None when not triggered
    equity: float
    peak_equity: float
    daily_pnl_pct: float    # (equity - last_equity) / last_equity
    drawdown_pct: float     # (peak_equity - equity) / peak_equity


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------
_DEFAULT_STATE: dict = {
    "state": "ACTIVE",
    "peak_equity": None,    # populated on first check
    "halted_at": None,
    "halt_reason": None,
    "last_checked_utc": None,
}


def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return dict(_DEFAULT_STATE)
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to read kill switch state: %s — treating as ACTIVE", exc)
        return dict(_DEFAULT_STATE)


def _save_state(state: dict) -> None:
    """Atomic write via os.replace() — never leaves a partial file."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    # Retry on OSError — OneDrive can briefly lock the .tmp file during sync
    for delay in (0.2, 0.5, 1.0, 2.0):
        try:
            os.replace(tmp, _STATE_PATH)
            return
        except OSError:
            time.sleep(delay)
    os.replace(tmp, _STATE_PATH)  # final attempt; raises if still locked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def kill_switch_active() -> bool:
    """
    Return True if the kill switch is in HALTED state.
    Called by synk_strategy.py before acting on any TradeInstruction.
    Fast path — reads state file only, no Alpaca API call.
    """
    state = _load_state()
    return state.get("state") == "HALTED"


def check_triggers(config=None) -> KillStatus:
    """
    Evaluate all three risk triggers against live Alpaca account data.

    If any trigger fires:
        - State is set to HALTED immediately (atomic write)
        - Telegram alert is sent
        - KillStatus.triggered = True is returned

    If no trigger fires:
        - peak_equity is updated if current equity is a new high
        - State written with last_checked_utc timestamp
        - KillStatus.triggered = False is returned

    Raises EnvironmentError (via get_config) if Alpaca keys are missing.
    """
    if config is None:
        from config import get_config  # noqa: PLC0415
        config = get_config()

    from alpaca.trading.client import TradingClient  # noqa: PLC0415
    client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.PAPER)
    acct = client.get_account()

    equity = float(acct.equity)
    last_equity = float(acct.last_equity)
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Load persisted state
    state = _load_state()

    # Initialise peak_equity on first run
    peak_equity = float(state["peak_equity"]) if state["peak_equity"] is not None else equity
    if equity > peak_equity:
        peak_equity = equity

    # Compute metrics
    daily_pnl_pct = (equity - last_equity) / last_equity if last_equity else 0.0
    drawdown_pct = (peak_equity - equity) / peak_equity if peak_equity else 0.0

    log.info(
        "Kill switch check | equity=%.2f | peak=%.2f | daily_pnl=%.2f%% | drawdown=%.2f%%",
        equity, peak_equity, daily_pnl_pct * 100, drawdown_pct * 100,
    )

    # Evaluate triggers (checked in severity order)
    halt_reason: str | None = None

    if daily_pnl_pct <= -_DAILY_LOSS_PCT:
        halt_reason = (
            f"Daily loss limit breached: {daily_pnl_pct*100:.2f}% "
            f"(limit: -{_DAILY_LOSS_PCT*100:.0f}%)"
        )
    elif drawdown_pct >= _PEAK_DRAWDOWN_PCT:
        halt_reason = (
            f"Peak drawdown limit breached: {drawdown_pct*100:.2f}% "
            f"(limit: {_PEAK_DRAWDOWN_PCT*100:.0f}%)"
        )

    # Per-trade stop-loss (trigger 1) is enforced at order time via stop-loss
    # bracket parameters, not here. This module handles account-level limits.

    if halt_reason:
        log.critical("KILL SWITCH TRIGGERED: %s", halt_reason)
        new_state = {
            "state": "HALTED",
            "peak_equity": peak_equity,
            "halted_at": now_utc,
            "halt_reason": halt_reason,
            "last_checked_utc": now_utc,
        }
        _save_state(new_state)
        send_telegram(f"[SYNK KILL SWITCH] {halt_reason}")
        return KillStatus(
            triggered=True,
            reason=halt_reason,
            equity=equity,
            peak_equity=peak_equity,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=drawdown_pct,
        )

    # No trigger — update peak and timestamp
    state["peak_equity"] = peak_equity
    state["last_checked_utc"] = now_utc
    _save_state(state)

    return KillStatus(
        triggered=False,
        reason=None,
        equity=equity,
        peak_equity=peak_equity,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_pct=drawdown_pct,
    )


def reset_to_active(reason: str = "Manual reset") -> None:
    """
    Reset kill switch state to ACTIVE. Manual use only — not called by bot.
    Logs a WARNING so every reset is traceable in process.log.
    """
    state = _load_state()
    state["state"] = "ACTIVE"
    state["halted_at"] = None
    state["halt_reason"] = None
    state["last_checked_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_state(state)
    log.warning("Kill switch manually reset to ACTIVE. Reason: %s", reason)


# ---------------------------------------------------------------------------
# Entry point — run one check cycle and print status
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    status = check_triggers()

    print("\n--- Kill Switch Status ---")
    print(f"Triggered:    {status.triggered}")
    print(f"Equity:       ${status.equity:,.2f}")
    print(f"Peak equity:  ${status.peak_equity:,.2f}")
    print(f"Daily P&L:    {status.daily_pnl_pct*100:+.3f}% (limit: -{_DAILY_LOSS_PCT*100:.0f}%)")
    print(f"Drawdown:     {status.drawdown_pct*100:.3f}% (limit: {_PEAK_DRAWDOWN_PCT*100:.0f}%)")
    if status.reason:
        print(f"Halt reason:  {status.reason}")
