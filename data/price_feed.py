"""
price_feed

Fetches daily OHLCV bars for Synk's watchlist via Alpaca's historical data API.
Regular market session only (DataFeed.IEX), split/dividend adjusted.

Results are cached to data/price_cache/<SYMBOL>.csv with a 24h staleness window —
the same pattern as download_gpr_daily() in signals/regime_filter.py.
A stale or missing cache triggers a live Alpaca fetch; a fresh cache is returned
directly without an API call.

Symbols (verified 2026-04-22):
    GLD  — SPDR Gold Shares (macro hedge leg)
    FXY  — Invesco CurrencyShares Japanese Yen (FX safe-haven leg)
    LMT  — Lockheed Martin (defence leg)
    NOC  — Northrop Grumman (defence leg)
    ITA  — iShares US Aerospace & Defense ETF (defence leg; ~4x volume of XAR)

Usage (standalone):
    python data/price_feed.py

Deps: alpaca-py, pandas, config.py
Run from synk/ root so that config.py is importable via the working directory.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the module works from any cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
_LOG_DIR = _HERE / "logs"
_CACHE_DIR = _HERE / "data" / "price_cache"

# Ensure synk/ root is on sys.path so `from config import ...` works whether
# this file is run as a script (python data/price_feed.py) or imported.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
SYMBOLS: list[str] = ["GLD", "FXY", "LMT", "NOC", "ITA"]  # public for momentum.py
_LOOKBACK_DAYS = 400        # 400 calendar days ≈ 275 trading days; ensures ≥252 bars for check_fxy_viability() 52-week low
_CACHE_MAX_AGE_SECONDS = 86400  # re-fetch if cache is >= 24h old
_FEED = DataFeed.IEX        # regular session only — no pre/post market bleed
_ADJUSTMENT = Adjustment.ALL  # split + dividend adjusted

# ---------------------------------------------------------------------------
# Logging — stdout + logs/process.log, UTC timestamps
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("price_feed")
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
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(symbol: str) -> Path:
    return _CACHE_DIR / f"{symbol}.csv"


def _cache_age_hours(symbol: str) -> float:
    p = _cache_path(symbol)
    if not p.exists():
        return float("inf")
    return (time.time() - p.stat().st_mtime) / 3600


def _is_cache_fresh(symbol: str) -> bool:
    return _cache_age_hours(symbol) * 3600 < _CACHE_MAX_AGE_SECONDS


def _load_from_cache(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(_cache_path(symbol), index_col="date", parse_dates=True)
    log.info(
        "%s: loaded %d rows from cache (%.1fh old)",
        symbol, len(df), _cache_age_hours(symbol),
    )
    return df


def _save_to_cache(symbol: str, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(symbol)
    df.to_csv(p)
    log.info("Cached %s -> %s (%d rows)", symbol, p.name, len(df))


# ---------------------------------------------------------------------------
# Alpaca fetch
# ---------------------------------------------------------------------------
def _fetch_bars(
    symbols: list[str],
    client: StockHistoricalDataClient,
    lookback_days: int = _LOOKBACK_DAYS,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily bars from Alpaca for all symbols in a single request.

    Uses DataFeed.IEX (regular session) and Adjustment.ALL (split + dividend).
    Returns a dict of clean DataFrames keyed by symbol.
    Raises ValueError for any symbol that returns empty bars.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    log.info(
        "Fetching bars for %s | %s to %s | feed=%s",
        symbols, start.date(), end.date(), _FEED.value,
    )

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=_ADJUSTMENT,
        feed=_FEED,
    )
    raw = client.get_stock_bars(req).df  # MultiIndex: (symbol, timestamp)

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        present = raw.index.get_level_values("symbol")
        if sym not in present:
            raise ValueError(
                f"Alpaca returned no bars for '{sym}'. "
                "Check symbol spelling and IEX feed eligibility."
            )

        sym_df = raw.xs(sym, level="symbol").copy()

        # Timestamps arrive as tz-aware (05:00 UTC = midnight ET).
        # Normalise to tz-naive date-only index for downstream indicator maths.
        sym_df.index = pd.to_datetime(sym_df.index).tz_localize(None).normalize()
        sym_df.index.name = "date"

        sym_df = sym_df[["open", "high", "low", "close", "volume"]].sort_index()

        if len(sym_df) == 0:
            raise ValueError(
                f"Bar slice for '{sym}' is empty after filtering. "
                "Check date range or feed availability."
            )

        log.info(
            "%s: %d bars | %s to %s | close range %.2f - %.2f",
            sym, len(sym_df),
            sym_df.index[0].date(), sym_df.index[-1].date(),
            sym_df["close"].min(), sym_df["close"].max(),
        )
        result[sym] = sym_df

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_prices(
    symbols: list[str] | None = None,
    lookback_days: int = _LOOKBACK_DAYS,
) -> dict[str, pd.DataFrame]:
    """
    Return a dict of OHLCV DataFrames keyed by symbol.

    Each DataFrame has a tz-naive DatetimeIndex named 'date' and columns
    [open, high, low, close, volume]. Bars are regular session only (IEX feed),
    split and dividend adjusted.

    Fresh cache (< 24h old) is returned without an API call. Stale or missing
    cache triggers a live fetch and updates the cache.

    Raises ValueError if any requested symbol returns empty bars.
    Raises EnvironmentError (via get_config) if Alpaca keys are missing.
    """
    if symbols is None:
        symbols = SYMBOLS

    fresh = [s for s in symbols if _is_cache_fresh(s)]
    stale = [s for s in symbols if not _is_cache_fresh(s)]

    result: dict[str, pd.DataFrame] = {}

    for sym in fresh:
        result[sym] = _load_from_cache(sym)

    if stale:
        # Import here to avoid a hard dependency when price_feed is imported
        # by other modules that supply their own Config.
        from config import get_config  # noqa: PLC0415
        cfg = get_config()
        client = StockHistoricalDataClient(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY)
        fetched = _fetch_bars(stale, client, lookback_days)
        for sym, df in fetched.items():
            _save_to_cache(sym, df)
            result[sym] = df

    return result


# ---------------------------------------------------------------------------
# Entry point — fetch all symbols and print last 5 bars each
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prices = get_prices()

    for sym, df in prices.items():
        print(f"\n--- {sym} | {len(df)} bars ---")
        header = (
            f"{'Date':<12}  {'Open':>8}  {'High':>8}  "
            f"{'Low':>8}  {'Close':>8}  {'Volume':>12}"
        )
        print(header)
        print("-" * len(header))
        for row in df.tail(5).itertuples():
            print(
                f"{str(row.Index.date()):<12}  "
                f"{row.open:>8.2f}  {row.high:>8.2f}  {row.low:>8.2f}  "
                f"{row.close:>8.2f}  {int(row.volume):>12,}"
            )
