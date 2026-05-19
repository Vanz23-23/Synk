"""
regime_filter

Gate 1 of the Synk three-factor signal stack: GPR regime classification.

Loads the Caldara-Iacoviello GPR Index daily data from a local .xls file,
computes a rolling 252-day z-score of the GPRD column, and classifies each
day into one of four regime states:

    NORMAL   (z < 0.5)  — no signals should fire
    ELEVATED (0.5–1.5)  — GLD/FXY leg armed
    HIGH     (1.5–2.0)  — full signal gate active
    EXTREME  (z > 2.0)  — full signal gate active (heightened caution)

All three gates (regime_filter, momentum, sentiment) must pass before a trade
is submitted.

Data file (verified 2026-04-22):
    URL:     https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls
    Format:  .xls (xlrd required)
    Columns: DAY, N10D, GPRD, GPRD_ACT, GPRD_THREAT, date, GPRD_MA30, GPRD_MA7
    Use:     date (YYYY-MM-DD)  +  GPRD (Daily GPR, index 1985:2019=100)
    WARNING: data_gpr_export.xls is the MONTHLY series — do not use for intraday.

Usage (standalone):
    python signals/regime_filter.py                     # auto-downloads to data/
    python signals/regime_filter.py <path/to/file.xls>  # use local file

Data source:
    Caldara & Iacoviello (2022), Measuring Geopolitical Risk.
    https://www.matteoiacoviello.com/gpr.htm
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the script works from any cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
_LOG_DIR = _HERE / "logs"
_DATA_DIR = _HERE / "data"
_DEFAULT_GPR_PATH = _DATA_DIR / "gpr_daily_recent.xls"
_REGIME_JSONL = _LOG_DIR / "regime.jsonl"       # append-only signal log

# ---------------------------------------------------------------------------
# Tuneable constants — all magic numbers live here, nowhere else
# ---------------------------------------------------------------------------
_ROLLING_WINDOW = 252       # trading days used for z-score denominator
_Z_ELEVATED = 0.5           # z ≥ this → ELEVATED
_Z_HIGH = 1.5               # z ≥ this → HIGH
_Z_EXTREME = 2.0            # z ≥ this → EXTREME
_MIN_ROWS_REQUIRED = _ROLLING_WINDOW + 1  # need window+1 rows for any valid z
_GPR_DAILY_URL = (
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
)
_DOWNLOAD_TIMEOUT = 30      # seconds
_GPR_MAX_AGE_SECONDS = 86400  # re-download if file is older than 24 h

# Safe-haven confirmation filter (GLD only)
_SH_LOOKBACK_DAYS = 5
_SH_HOSTILE_THRESHOLD = -0.03   # 5-day return below this → hostile regime (gold falling in GPR spike)

# FXY 52-week low proximity filter
_FXY_52W_LOW_BUFFER = 0.05      # block FXY entries within 5% of 52-week low
_FXY_52W_MIN_BARS = 252         # bars needed for true 52w low; logs warning if fewer available

# ---------------------------------------------------------------------------
# Logging — dual output: stdout + logs/process.log, all timestamps UTC
# Same pattern as gdelt_fetcher.py so both modules share the same process.log
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("regime_filter")
    if logger.handlers:
        return logger  # already configured (e.g. re-imported in tests)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
    fmt.converter = time.gmtime  # force UTC in log timestamps

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
class RegimeState(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass(frozen=True)
class RegimeSignal:
    date: str           # ISO 8601 date (YYYY-MM-DD)
    gpr_value: float
    z_score: float
    regime: RegimeState

    def as_dict(self) -> dict:
        d = asdict(self)
        d["regime"] = self.regime.value  # asdict leaves enums as objects
        return d


# ---------------------------------------------------------------------------
# GPR file download
# ---------------------------------------------------------------------------
def download_gpr_daily(dest: Path = _DEFAULT_GPR_PATH) -> Path:
    """
    Download data_gpr_daily_recent.xls from Iacoviello's server and save to dest.

    Skips the download if the file exists AND is less than 24 hours old.
    Re-downloads if the file is stale (>= 24 h) or missing entirely.
    Returns the path to the local file.
    """
    if dest.exists():
        age_seconds = time.time() - dest.stat().st_mtime
        if age_seconds < _GPR_MAX_AGE_SECONDS:
            log.info(
                "GPR file is fresh (%.1f h old) — skipping download: %s",
                age_seconds / 3600,
                dest,
            )
            return dest
        log.info(
            "GPR file is stale (%.1f h old) — re-downloading",
            age_seconds / 3600,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading GPR daily data from %s ...", _GPR_DAILY_URL)
    resp = requests.get(
        _GPR_DAILY_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=_DOWNLOAD_TIMEOUT,
    )
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info("Saved %.1f KB -> %s", len(resp.content) / 1024, dest)
    return dest


# ---------------------------------------------------------------------------
# GPR data loading — handles .xls, .xlsx, and .csv
# ---------------------------------------------------------------------------
def load_gpr(path: Path) -> pd.DataFrame:
    """
    Load a Caldara-Iacoviello GPR file into a clean two-column DataFrame.

    Supports .xls, .xlsx, and .csv. For the official daily file
    (data_gpr_daily_recent.xls) the loader selects:
        date column  : 'date'  (YYYY-MM-DD strings)
        value column : 'GPRD'  (daily GPR, NOT the monthly 'GPR' column)

    Column selection logic:
      - Date: first column whose name contains date/month/year/time; else col 0.
      - GPR:  prefers columns starting with 'gprd' (daily); falls back to any
              column containing 'gpr'; else col 1. This ensures the monthly
              'GPR' column is never chosen when 'GPRD' is present.

    Returns DataFrame with columns ['date' (str YYYY-MM-DD), 'gpr' (float)],
    sorted ascending, NaN rows dropped.
    Raises FileNotFoundError, ValueError.
    """
    if not path.exists():
        raise FileNotFoundError(f"GPR file not found: {path}")

    log.info("Loading GPR data from %s", path)
    suffix = path.suffix.lower()

    if suffix in (".xls", ".xlsx"):
        engine = "xlrd" if suffix == ".xls" else "openpyxl"
        df = pd.read_excel(path, engine=engine)
    else:
        df = pd.read_csv(path)

    log.info(
        "Raw shape: %d rows x %d cols | columns: %s",
        len(df), len(df.columns), list(df.columns),
    )

    # Locate date column
    date_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ("date", "month", "year", "time"))
    ]
    date_col = date_candidates[0] if date_candidates else df.columns[0]
    log.info("Using date column: '%s'", date_col)

    # Locate GPR value column — GPRD (daily) takes priority over GPR (monthly)
    gpr_candidates = [c for c in df.columns if "gpr" in c.lower() and c != date_col]
    daily_candidates = [c for c in gpr_candidates if c.lower().startswith("gprd")]
    if daily_candidates:
        gpr_col = daily_candidates[0]   # e.g. 'GPRD'
    elif gpr_candidates:
        gpr_col = gpr_candidates[0]     # fallback: 'GPR' (monthly)
    else:
        gpr_col = df.columns[1]
    log.info("Using GPR value column: '%s'", gpr_col)

    out = df[[date_col, gpr_col]].copy()
    out.columns = ["date", "gpr"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["gpr"] = pd.to_numeric(out["gpr"], errors="coerce")
    out = out.dropna(subset=["date", "gpr"]).sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    if len(out) < _MIN_ROWS_REQUIRED:
        raise ValueError(
            f"GPR data has only {len(out)} valid rows after cleaning; "
            f"need >= {_MIN_ROWS_REQUIRED} for a {_ROLLING_WINDOW}-day z-score."
        )

    log.info(
        "Loaded %d valid GPR rows. Date range: %s -> %s",
        len(out), out["date"].iloc[0], out["date"].iloc[-1],
    )
    return out


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------
def compute_z_scores(df: pd.DataFrame, window: int = _ROLLING_WINDOW) -> pd.DataFrame:
    """
    Append a 'z_score' column: rolling (window)-day z-score of 'gpr'.

    z = (gpr - rolling_mean) / rolling_std  [ddof=1, sample std]

    Rows inside the burn-in period (fewer than window prior observations)
    produce NaN and are excluded by build_signals().
    """
    df = df.copy()
    roll = df["gpr"].rolling(window=window)
    df["z_score"] = (df["gpr"] - roll.mean()) / roll.std(ddof=1)
    valid = int(df["z_score"].notna().sum())
    log.info("Z-scores computed. Valid (non-NaN) rows: %d / %d", valid, len(df))
    return df


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def classify_regime(z: float) -> RegimeState:
    """Map a z-score to a RegimeState. Boundaries are inclusive at the top."""
    if z >= _Z_EXTREME:
        return RegimeState.EXTREME
    if z >= _Z_HIGH:
        return RegimeState.HIGH
    if z >= _Z_ELEVATED:
        return RegimeState.ELEVATED
    return RegimeState.NORMAL


def build_signals(df: pd.DataFrame) -> list[RegimeSignal]:
    """
    Convert a z-scored DataFrame into RegimeSignal objects.
    Rows where z_score is NaN (burn-in period) are silently excluded.
    """
    valid = df.dropna(subset=["z_score"])
    return [
        RegimeSignal(
            date=str(row["date"]),
            gpr_value=round(float(row["gpr"]), 4),
            z_score=round(float(row["z_score"]), 4),
            regime=classify_regime(float(row["z_score"])),
        )
        for _, row in valid.iterrows()
    ]


# ---------------------------------------------------------------------------
# Public API — entry points for synk_strategy.py
# ---------------------------------------------------------------------------
def get_latest_regime(gpr_path: Path = _DEFAULT_GPR_PATH) -> RegimeSignal:
    """
    Load GPR data, compute z-scores, return the single most recent RegimeSignal.
    Raises FileNotFoundError or ValueError if data is missing or insufficient.
    """
    df = load_gpr(gpr_path)
    df = compute_z_scores(df)
    signals = build_signals(df)
    if not signals:
        raise ValueError("No valid regime signals produced — check GPR data quality.")
    latest = signals[-1]
    log.info(
        "Latest regime: %s | date=%s | gpr=%.2f | z=%.3f",
        latest.regime.value, latest.date, latest.gpr_value, latest.z_score,
    )
    return latest


def get_regime_series(gpr_path: Path = _DEFAULT_GPR_PATH) -> list[RegimeSignal]:
    """Return the full historical list of RegimeSignal objects (burn-in excluded)."""
    df = load_gpr(gpr_path)
    df = compute_z_scores(df)
    return build_signals(df)


def is_gate_open(signal: RegimeSignal) -> bool:
    """
    True when regime warrants trading: ELEVATED, HIGH, or EXTREME.
    NORMAL returns False — all downstream gates are bypassed.
    """
    return signal.regime != RegimeState.NORMAL


# ---------------------------------------------------------------------------
# JSONL persistence — append-only, one signal per line (third logging tier)
# ---------------------------------------------------------------------------
def append_to_jsonl(signal: RegimeSignal, path: Path = _REGIME_JSONL) -> None:
    """Append a single RegimeSignal as a newline-delimited JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = signal.as_dict()
    record["logged_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info("Regime signal appended to %s", path)


# ---------------------------------------------------------------------------
# Safe-haven confirmation (Change 1)
# ---------------------------------------------------------------------------
def check_safe_haven_confirmation(
    prices_df: pd.DataFrame,
    lookback_days: int = _SH_LOOKBACK_DAYS,
    hostile_threshold: float = _SH_HOSTILE_THRESHOLD,
) -> dict:
    """
    Test whether GLD is behaving as a safe-haven asset.

    Computes the lookback_days-day close return. If the return falls below
    hostile_threshold (default -3%), gold is falling during a high-GPR period —
    indicating an inflationary or dollar-strength regime where the safe-haven
    relationship has broken down (e.g. Iran war 2026: GLD -23% during extreme GPR).

    Returns: {confirmed: bool, return_5d: float, regime: str}
    regime: 'SAFE_HAVEN' | 'HOSTILE' | 'INSUFFICIENT_DATA'
    confirmed=False should block GLD entry and trigger exit of open GLD positions.
    """
    insufficient: dict = {"confirmed": False, "return_5d": 0.0, "regime": "INSUFFICIENT_DATA"}

    if prices_df is None or prices_df.empty or "close" not in prices_df.columns:
        log.warning("check_safe_haven_confirmation: empty or invalid prices_df")
        return insufficient

    if len(prices_df) < lookback_days + 1:
        log.warning(
            "check_safe_haven_confirmation: only %d bars available, need %d",
            len(prices_df), lookback_days + 1,
        )
        return insufficient

    close = prices_df["close"]
    ret = float((close.iloc[-1] - close.iloc[-(lookback_days + 1)]) / close.iloc[-(lookback_days + 1)])

    if ret < hostile_threshold:
        regime, confirmed = "HOSTILE", False
    else:
        regime, confirmed = "SAFE_HAVEN", True

    log.info(
        "Safe-haven check: return_%dd=%+.2f%% | regime=%s",
        lookback_days, ret * 100, regime,
    )
    return {"confirmed": confirmed, "return_5d": round(ret, 6), "regime": regime}


# ---------------------------------------------------------------------------
# FXY 52-week low proximity filter (Change 3)
# ---------------------------------------------------------------------------
def check_fxy_viability(
    fxy_prices_df: pd.DataFrame,
    buffer: float = _FXY_52W_LOW_BUFFER,
) -> dict:
    """
    Test whether FXY is too close to its 52-week low to enter a position.

    Within buffer (default 5%) of the 52-week low signals a structural BoJ
    divergence / dollar-strength regime — entering FXY there is unsound.

    Note: uses all available bars in fxy_prices_df as the low reference.
    Logs a warning when fewer than _FXY_52W_MIN_BARS are available; the
    90-day default cache gives a shorter-period low but is still useful signal.

    Returns: {viable: bool, current: float, low_52w: float, distance_from_low: float}
    """
    empty: dict = {"viable": False, "current": 0.0, "low_52w": 0.0, "distance_from_low": 0.0}

    if fxy_prices_df is None or fxy_prices_df.empty or "close" not in fxy_prices_df.columns:
        log.warning("check_fxy_viability: empty or invalid prices_df")
        return empty

    available = len(fxy_prices_df)
    if available < _FXY_52W_MIN_BARS:
        log.warning(
            "check_fxy_viability: %d bars available (need %d for true 52w low) — "
            "using %d-bar low as proxy",
            available, _FXY_52W_MIN_BARS, available,
        )

    low_52w = float(fxy_prices_df["close"].min())
    current = float(fxy_prices_df["close"].iloc[-1])

    if low_52w <= 0:
        log.error("check_fxy_viability: low_52w is zero or negative — returning not viable")
        return empty

    distance = (current - low_52w) / low_52w
    viable = distance > buffer

    log.info(
        "FXY viability: current=%.4f | low=%.4f | dist=%.2f%% | buffer=%.0f%% | viable=%s",
        current, low_52w, distance * 100, buffer * 100, viable,
    )
    return {
        "viable": viable,
        "current": round(current, 4),
        "low_52w": round(low_52w, 4),
        "distance_from_low": round(distance, 4),
    }


# ---------------------------------------------------------------------------
# Entry point — prints last 5 regime readings for manual verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use explicit path if given; otherwise auto-download to data/gpr_daily_recent.xls
    if len(sys.argv) >= 2:
        gpr_path = Path(sys.argv[1])
    else:
        try:
            gpr_path = download_gpr_daily()
        except Exception as exc:
            print(f"Auto-download failed: {exc}")
            print("Run: python signals/regime_filter.py <path/to/file.xls>")
            sys.exit(1)

    try:
        signals = get_regime_series(gpr_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    last_5 = signals[-5:]
    latest = signals[-1]

    print("\n--- Last 5 GPR Regime Readings ---")
    header = f"{'Date':<12}  {'GPR Value':>10}  {'Z-Score':>9}  {'Regime':<10}  Gate"
    print(header)
    print("-" * len(header))
    for s in last_5:
        gate = "OPEN " if is_gate_open(s) else "CLOSED"
        print(
            f"{s.date:<12}  {s.gpr_value:>10.2f}  {s.z_score:>9.3f}"
            f"  {s.regime.value:<10}  {gate}"
        )

    print(f"\nTotal signals in series: {len(signals)}")
    print(f"Gate (latest):           {'OPEN' if is_gate_open(latest) else 'CLOSED'}")
    print(f"Regime (latest):         {latest.regime.value}")

    append_to_jsonl(latest)
