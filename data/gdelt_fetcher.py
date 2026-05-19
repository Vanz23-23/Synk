"""
gdelt_fetcher

Measures real-world GDELT 2.0 end-to-end latency. Fetches the GDELT
masterfile list, extracts the most recent file timestamp, and computes
lag vs current UTC time. Runs 5 checks one minute apart, saves results,
and prints a strategy verdict based on average lag.

Usage:
    python data/gdelt_fetcher.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
_LOG_DIR = _HERE / "logs"
_RESULTS_PATH = _LOG_DIR / "gdelt_lag_results.json"
_NOTES_PATH = _LOG_DIR / "gdelt_lag_notes.txt"
_MASTERFILE_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# ---------------------------------------------------------------------------
# Tuneable constants — all magic numbers live here, nowhere else
# ---------------------------------------------------------------------------
_MAX_FETCH_RETRIES = 3           # retry attempts per masterfile fetch
_RETRY_BACKOFF_SECONDS = 5       # wait between retry attempts
_FETCH_TIMEOUT_SECONDS = 15      # HTTP request timeout
_MASTERFILE_TAIL_BYTES = 2000    # tail size to fetch (~10-12 lines at ~180 chars each)
_MAX_LINES_TO_SCAN = 10          # max lines to walk back when filtering future entries
_LAG_SWING_THRESHOLD_MIN = 30    # avg lag above this → SWING ONLY verdict
_LAG_BORDERLINE_THRESHOLD_MIN = 20  # avg lag above this → BORDERLINE verdict
_DEFAULT_N_CHECKS = 5            # checks to run in a standard measurement session
_DEFAULT_INTERVAL_SECONDS = 60   # wait between checks

# ---------------------------------------------------------------------------
# Logging — dual output: stdout + logs/process.log, all timestamps UTC
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gdelt_fetcher")
    if logger.handlers:
        return logger  # already configured (e.g. re-imported)
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
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class GDELTLagResult:
    check_time_utc: str         # ISO 8601
    latest_record_utc: str      # ISO 8601
    lag_minutes: float
    raw_filename: str           # basename extracted from masterfile URL
    skipped_future_entries: int = 0  # entries skipped due to GDELT pre-publication


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------
def _fetch_latest_timestamp(
    session: requests.Session,
    now_utc: datetime,
) -> tuple[datetime, str, int]:
    """
    Fetch the tail of the GDELT masterfile (HTTP Range header) to avoid
    downloading ~60 MB. Walk backwards through the last 10 lines and return
    the most recent entry whose timestamp is <= now_utc.

    GDELT occasionally pre-publishes entries for files not yet available;
    those show up as future timestamps and produce negative lag values.
    Each skipped future entry is logged at INFO level.

    Returns (timestamp_utc: datetime, raw_filename: str, skipped: int).
    Raises RuntimeError if all retries are exhausted or no valid entry is found.
    """
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, _MAX_FETCH_RETRIES + 1):
        try:
            # Fetch only the tail — see _MASTERFILE_TAIL_BYTES for sizing rationale.
            headers = {"Range": f"bytes=-{_MASTERFILE_TAIL_BYTES}"}
            resp = session.get(_MASTERFILE_URL, headers=headers, timeout=_FETCH_TIMEOUT_SECONDS)
            resp.raise_for_status()

            lines = [ln for ln in resp.text.strip().splitlines() if ln.strip()]
            if not lines:
                raise ValueError("Masterfile tail returned no usable lines")

            skipped = 0

            # Walk backwards through the most recent lines looking for the
            # first entry that is not in the future.
            for line in reversed(lines[-_MAX_LINES_TO_SCAN:]):
                parts = line.split()
                if len(parts) < 3:
                    continue  # malformed line — skip silently

                url = parts[2]
                raw_filename = url.split("/")[-1]  # e.g. 20240120153000.export.CSV.zip

                # First 14 chars of basename are YYYYMMDDHHMMSS
                ts_str = raw_filename[:14]
                if len(ts_str) != 14 or not ts_str.isdigit():
                    continue  # unrecognised filename pattern — skip silently

                timestamp = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(
                    tzinfo=timezone.utc
                )

                if timestamp > now_utc:
                    delta_min = (timestamp - now_utc).total_seconds() / 60.0
                    log.info(
                        "Skipped future-dated entry: %s (%.1f min ahead)",
                        raw_filename,
                        delta_min,
                    )
                    skipped += 1
                    continue

                # Valid entry found — return immediately
                return timestamp, raw_filename, skipped

            raise ValueError(
                "No valid (non-future) entry found in the last 10 masterfile lines"
            )

        except Exception as exc:
            last_exc = exc
            log.warning(
                "Attempt %d/%d failed fetching masterfile: %s",
                attempt,
                _MAX_FETCH_RETRIES,
                exc,
            )
            if attempt < _MAX_FETCH_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS)

    raise RuntimeError(
        f"All 3 attempts to fetch GDELT masterfile failed. Last error: {last_exc}"
    )


def _single_lag_check(session: requests.Session) -> GDELTLagResult:
    """Run one lag check and return a GDELTLagResult."""
    check_time = datetime.now(timezone.utc)
    latest_ts, raw_filename, skipped = _fetch_latest_timestamp(session, check_time)
    lag_minutes = (check_time - latest_ts).total_seconds() / 60.0
    return GDELTLagResult(
        check_time_utc=check_time.isoformat(timespec="seconds"),
        latest_record_utc=latest_ts.isoformat(timespec="seconds"),
        lag_minutes=round(lag_minutes, 1),
        raw_filename=raw_filename,
        skipped_future_entries=skipped,
    )


# ---------------------------------------------------------------------------
# Main measurement loop
# ---------------------------------------------------------------------------
def run_lag_measurement(
    n_checks: int = 5, interval_seconds: int = 60
) -> list[GDELTLagResult]:
    """
    Run n_checks lag checks spaced interval_seconds apart.
    Failed checks are logged and skipped; never raises.
    Returns list of successful GDELTLagResult objects.
    """
    log.info(
        "Starting GDELT lag measurement: %d checks, %ds interval (~%d min total)",
        n_checks,
        interval_seconds,
        (n_checks * interval_seconds) // 60,
    )
    results: list[GDELTLagResult] = []

    with requests.Session() as session:
        for i in range(1, n_checks + 1):
            log.info("Check %d/%d ...", i, n_checks)
            try:
                result = _single_lag_check(session)
                results.append(result)
                log.info(
                    "Check %d done — lag=%.1f min | file=%s",
                    i,
                    result.lag_minutes,
                    result.raw_filename,
                )
            except Exception as exc:
                log.error("Check %d FAILED — skipping: %s", i, exc)

            if i < n_checks:
                log.info("Waiting %ds before next check...", interval_seconds)
                time.sleep(interval_seconds)

    log.info("Measurement complete. %d/%d checks succeeded.", len(results), n_checks)
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_results(
    results: list[GDELTLagResult],
    path: os.PathLike | str = _RESULTS_PATH,
) -> None:
    """Atomic write via os.replace(). Creates logs/ dir if missing."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    payload = [asdict(r) for r in results]
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, dest)
    log.info("Results saved to %s", dest)


def append_notes(
    results: list[GDELTLagResult],
    total_checks: int,
    path: os.PathLike | str = _NOTES_PATH,
) -> None:
    """
    Append a one-line human-readable summary to logs/gdelt_lag_notes.txt.
    Uses append mode — never overwrites previous runs.
    """
    if not results:
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lags = [r.lag_minutes for r in results]
    avg_lag = sum(lags) / len(lags)
    verdict_short = _compute_verdict(avg_lag).split(":")[0]  # e.g. "INTRADAY VIABLE"
    now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = (
        f"[{now_str} UTC] Run complete. "
        f"Avg lag: {avg_lag:.1f} min. "
        f"Verdict: {verdict_short}. "
        f"Checks: {len(results)}/{total_checks}. "
        "Note: rerun during market hours (Tue/Wed 14:00-15:00 UTC) for representative reading.\n"
    )
    with open(dest, "a", encoding="utf-8") as f:
        f.write(line)
    log.info("Notes appended to %s", dest)


# ---------------------------------------------------------------------------
# Summary + verdict
# ---------------------------------------------------------------------------
def _compute_verdict(avg_lag: float) -> str:
    """Return the strategy verdict string for a given average lag (minutes)."""
    if avg_lag > _LAG_SWING_THRESHOLD_MIN:
        return (
            "SWING ONLY: GDELT lag too high for intraday signals. "
            "GLD/FXY swing trading only."
        )
    elif avg_lag >= _LAG_BORDERLINE_THRESHOLD_MIN:
        return "BORDERLINE: Lag is marginal. Recommend swing only to be safe."
    else:
        return "INTRADAY VIABLE: Defence ETF leg possible."


def print_summary(results: list[GDELTLagResult]) -> None:
    """Print a formatted table, stats, and strategy verdict to stdout."""
    if not results:
        print("\nNo successful checks — cannot produce summary.")
        return

    lags = [r.lag_minutes for r in results]
    min_lag = min(lags)
    max_lag = max(lags)
    avg_lag = sum(lags) / len(lags)

    col_w = [5, 26, 26, 10, 40]
    header = (
        f"{'Check':<{col_w[0]}} | "
        f"{'Time (UTC)':<{col_w[1]}} | "
        f"{'Latest Record (UTC)':<{col_w[2]}} | "
        f"{'Lag (min)':<{col_w[3]}} | "
        f"{'File':<{col_w[4]}}"
    )
    divider = "-" * len(header)

    print()
    print(header)
    print(divider)
    for idx, r in enumerate(results, start=1):
        print(
            f"{idx:<{col_w[0]}} | "
            f"{r.check_time_utc:<{col_w[1]}} | "
            f"{r.latest_record_utc:<{col_w[2]}} | "
            f"{r.lag_minutes:<{col_w[3]}.1f} | "
            f"{r.raw_filename:<{col_w[4]}}"
        )

    print()
    print(f"Min lag: {min_lag:.1f} min  |  Max lag: {max_lag:.1f} min  |  Avg lag: {avg_lag:.1f} min")
    print()

    verdict = _compute_verdict(avg_lag)
    print(f"STRATEGY VERDICT: {verdict}")

    # Report any future-dated entries that were silently skipped
    total_skipped = sum(r.skipped_future_entries for r in results)
    checks_with_skips = sum(1 for r in results if r.skipped_future_entries > 0)
    if total_skipped > 0:
        print(
            f"\nNote: {total_skipped} future-dated "
            f"{'entry was' if total_skipped == 1 else 'entries were'} skipped "
            f"in {checks_with_skips} "
            f"{'check' if checks_with_skips == 1 else 'checks'} "
            "(GDELT pre-publication artefact)"
        )

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_lag_measurement(
        n_checks=_DEFAULT_N_CHECKS,
        interval_seconds=_DEFAULT_INTERVAL_SECONDS,
    )
    save_results(results)
    append_notes(results, total_checks=_DEFAULT_N_CHECKS)
    print_summary(results)
