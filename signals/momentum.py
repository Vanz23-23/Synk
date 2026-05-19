"""
momentum

Gate 2 of the Synk three-factor signal stack: price momentum confirmation.

Computes two indicators on a price DataFrame supplied by data/price_feed.py
and requires BOTH conditions to be True for the gate to open:

    1. ROC(20) > 0   — 20-day rate of change is positive
    2. Close > SMA(20) — price is above its 20-day simple moving average

ROC(20) = (close_today - close_20d_ago) / close_20d_ago

Neither indicator is pre-computed by price_feed — this module owns them.
Rows in the 20-bar burn-in period (NaN in either indicator) are excluded.

Usage (standalone):
    python signals/momentum.py

Deps: data/price_feed.py, pandas
Run from synk/ root so that data.price_feed is importable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
_LOG_DIR = _HERE / "logs"
_MOMENTUM_JSONL = _LOG_DIR / "momentum.jsonl"

# Ensure synk/ root is on sys.path for `from data.price_feed import ...`
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
_ROC_PERIOD = 20    # rate-of-change lookback (trading days)
_SMA_PERIOD = 20    # simple moving average window (trading days)

# ---------------------------------------------------------------------------
# Logging — stdout + logs/process.log, UTC timestamps
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("momentum")
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
class MomentumSignal:
    date: str
    symbol: str
    roc_20: float   # rate of change over 20 days (e.g. 0.05 = +5%)
    sma_20: float   # 20-day simple moving average of close
    close: float
    signal: bool    # True = both conditions met, gate open

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append roc_20 and sma_20 columns to a price DataFrame.

    Expects a DatetimeIndex and a 'close' column. Burn-in rows produce NaN
    and are excluded by build_signals(). Returns a copy.
    """
    df = df.copy()
    df["sma_20"] = df["close"].rolling(window=_SMA_PERIOD).mean()
    close_20d_ago = df["close"].shift(_ROC_PERIOD)
    df["roc_20"] = (df["close"] - close_20d_ago) / close_20d_ago
    return df


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def build_signals(df: pd.DataFrame, symbol: str) -> list[MomentumSignal]:
    """
    Convert an indicator-enriched DataFrame into MomentumSignal objects.
    Burn-in rows (NaN in roc_20 or sma_20) are excluded.
    Both conditions must be True for signal=True — no partial credit.
    """
    valid = df.dropna(subset=["roc_20", "sma_20"])
    signals = []
    for row in valid.itertuples():
        gate = bool(row.roc_20 > 0 and row.close > row.sma_20)
        signals.append(
            MomentumSignal(
                date=str(row.Index.date()),
                symbol=symbol,
                roc_20=round(float(row.roc_20), 6),
                sma_20=round(float(row.sma_20), 4),
                close=round(float(row.close), 4),
                signal=gate,
            )
        )
    return signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_latest_momentum(symbol: str, df: pd.DataFrame) -> MomentumSignal:
    """
    Compute indicators and return the most recent MomentumSignal for symbol.
    Raises ValueError if data is insufficient for the burn-in period.
    """
    df = compute_indicators(df)
    signals = build_signals(df, symbol)
    if not signals:
        raise ValueError(
            f"{symbol}: no valid momentum signals — need > {_ROC_PERIOD} bars, "
            f"got {len(df)}."
        )
    latest = signals[-1]
    log.info(
        "%s | date=%s | close=%.2f | sma20=%.2f | roc20=%+.4f | gate=%s",
        symbol, latest.date, latest.close, latest.sma_20, latest.roc_20,
        "OPEN" if latest.signal else "CLOSED",
    )
    return latest


def get_momentum_series(symbol: str, df: pd.DataFrame) -> list[MomentumSignal]:
    """Return the full historical list of MomentumSignals for symbol."""
    df = compute_indicators(df)
    return build_signals(df, symbol)


def is_gate_open(signal: MomentumSignal) -> bool:
    """True if both ROC(20) > 0 and close > SMA(20)."""
    return signal.signal


# ---------------------------------------------------------------------------
# JSONL persistence — append-only (third logging tier)
# ---------------------------------------------------------------------------
def append_to_jsonl(signal: MomentumSignal, path: Path = _MOMENTUM_JSONL) -> None:
    """Append a single MomentumSignal as a newline-delimited JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = signal.as_dict()
    record["logged_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info("Momentum signal appended to %s", path)


# ---------------------------------------------------------------------------
# Entry point — last 5 signals per symbol
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data.price_feed import get_prices  # deferred: only needed as a script

    prices = get_prices()

    for sym, df in prices.items():
        signals = get_momentum_series(sym, df)

        if not signals:
            print(f"\n{sym}: insufficient data for momentum signals")
            continue

        last_5 = signals[-5:]
        latest = signals[-1]

        print(f"\n--- {sym} | Last 5 Momentum Signals ---")
        header = (
            f"{'Date':<12}  {'Close':>8}  {'SMA20':>8}  "
            f"{'ROC20':>8}  {'Gate':<6}"
        )
        print(header)
        print("-" * len(header))
        for s in last_5:
            gate = "OPEN " if s.signal else "CLOSED"
            print(
                f"{s.date:<12}  {s.close:>8.2f}  {s.sma_20:>8.2f}  "
                f"{s.roc_20:>+8.4f}  {gate}"
            )

        print(
            f"  Gate (latest): {'OPEN' if is_gate_open(latest) else 'CLOSED'} | "
            f"ROC={latest.roc_20:+.4f} | Close={latest.close:.2f} vs SMA={latest.sma_20:.2f}"
        )
        append_to_jsonl(latest)
