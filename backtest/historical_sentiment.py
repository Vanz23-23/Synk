"""
backtest/historical_sentiment

Retroactive FinBERT sentiment scorer for the Synk historical backtest.

Scores one GDELT GKG file per trading day (US business-day calendar) at
14:00 UTC — post-EU open, mid-US morning. Reuses the live bot's FinBERT
pipeline so results are comparable to what the live gate would produce.

Output columns:
    date, headline_count, dominant_class, dominant_prob, sentiment_score,
    gate_at_p60_s30, gate_at_p55_s30, gate_at_p50_s30, gate_at_p45_s30,
    gate_at_p60_s20, gate_at_p55_s20, gate_at_p50_s20, gate_at_p45_s20,
    gate_at_p60_s15, gate_at_p55_s15, gate_at_p50_s15, gate_at_p45_s15,
    gate_at_p60_s10, gate_at_p55_s10, gate_at_p50_s10, gate_at_p45_s10

  gate_at_pXX_sYY = True when dominant_prob > 0.XX AND abs(sentiment_score) > 0.YY

Design notes:
    - Year-chunked: writes one parquet per calendar year (resumable across sessions).
    - Streaming: GKG zip is parsed in memory, never written to disk.
    - Disk footprint: final merged parquet ~50-100 MB for 6 years.
    - FinBERT model loads once (~25s) and stays resident for the full run.

Public API:
    score_date_range(start, end, output_path) -> pd.DataFrame

CLI usage (from synk/ root):
    # Validate against live cache (fast, ~35 days):
    python backtest/historical_sentiment.py --start 2026-04-22 --end 2026-05-27 --validate

    # Full Scan A (2020-2026, takes ~2-3 hours):
    python backtest/historical_sentiment.py --start 2020-01-01 --end 2026-01-01
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths and sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
_LOG_DIR = _HERE / "logs"
_RESULTS_DIR = _HERE / "backtest" / "results"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# csv field-size limit — must be set before any csv.reader call
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Logging — dedicated file so historical scan doesn't pollute process.log
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("historical_sentiment")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
    fmt.converter = time.gmtime

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(_LOG_DIR / "historical_sentiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _setup_logging()

# ---------------------------------------------------------------------------
# Import FinBERT pipeline from live sentiment module (reuses singleton)
# ---------------------------------------------------------------------------
from signals.sentiment import (  # noqa: E402
    score_headlines,
    _parse_tone,
    _has_conflict_theme,
    _url_to_slug,
    _MAX_HEADLINES,
    _CONFLICT_TONE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GKG_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
_TARGET_HOUR = 14        # 14:00 UTC for each daily reading
_FETCH_TIMEOUT = 30      # seconds — historical server can be slower than live

# Threshold grid: 4 p-values × 4 s-values = 16 gate columns
# Current live thresholds: p=0.60, s=0.30 → column "gate_at_p60_s30"
_P_THRESHOLDS = (0.45, 0.50, 0.55, 0.60)
_S_THRESHOLDS = (0.10, 0.15, 0.20, 0.30)


def _col(p: float, s: float) -> str:
    """Canonical column name for a (p_threshold, s_threshold) pair."""
    return f"gate_at_p{int(round(p * 100)):02d}_s{int(round(s * 100)):02d}"


# Stable ordered list of all 16 gate column names
GATE_COLUMNS: list[str] = [_col(p, s) for p in _P_THRESHOLDS for s in _S_THRESHOLDS]

# Convenience alias for the current live threshold pair
LIVE_GATE_COLUMN = _col(0.60, 0.30)  # "gate_at_p60_s30"


# ---------------------------------------------------------------------------
# GDELT historical URL construction
# ---------------------------------------------------------------------------
def _candidate_urls(d: date) -> list[str]:
    """
    Return GKG URLs to try for date d at 14:00 UTC.
    Primary: exactly 14:00:00. Fallbacks: 13:45 and 14:15 (±1 file interval).
    GDELT GKGv2 files are released every 15 minutes.
    """
    base = d.strftime("%Y%m%d")
    return [
        f"{_GKG_BASE_URL}/{base}140000.gkg.csv.zip",  # primary
        f"{_GKG_BASE_URL}/{base}134500.gkg.csv.zip",  # fallback −15 min
        f"{_GKG_BASE_URL}/{base}141500.gkg.csv.zip",  # fallback +15 min
    ]


# ---------------------------------------------------------------------------
# Per-date headline fetching
# ---------------------------------------------------------------------------
def _fetch_headlines_for_date(d: date) -> list[str]:
    """
    Download a historical GKG file and extract conflict headlines.
    Tries three candidate URLs. Returns [] on total failure.

    Parsing mirrors fetch_gdelt_headlines() in signals/sentiment.py exactly
    so results are directly comparable to the live gate.
    """
    for url in _candidate_urls(d):
        try:
            r = requests.get(url, timeout=_FETCH_TIMEOUT)
            if r.status_code == 404:
                continue
            r.raise_for_status()

            headlines: list[str] = []
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    reader = csv.reader(
                        io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                        delimiter="\t",
                    )
                    for row in reader:
                        if len(row) < 23:
                            continue

                        url_val = row[4]
                        themes = row[7]
                        tone_field = row[15]
                        quotations = row[22]

                        tone = _parse_tone(tone_field)
                        if tone is None or tone >= _CONFLICT_TONE_THRESHOLD:
                            continue
                        if not _has_conflict_theme(themes):
                            continue

                        # Prefer quoted text; fall back to URL slug
                        if quotations.strip():
                            first_quote = quotations.split("#")[0]
                            parts = first_quote.split("||", 1)
                            text = parts[1].strip() if len(parts) > 1 else ""
                        else:
                            text = _url_to_slug(url_val)

                        if len(text) > 10:
                            headlines.append(text)

                        if len(headlines) >= _MAX_HEADLINES:
                            break

            log.info("[%s] %d headlines from %s", d, len(headlines), url.split("/")[-1])
            return headlines

        except requests.exceptions.HTTPError:
            # 4xx/5xx from raise_for_status — try next candidate
            continue
        except Exception as exc:
            log.warning("[%s] Fetch failed (%s: %s)", d, type(exc).__name__, url.split("/")[-1])
            continue

    log.warning("[%s] No GKG file at any candidate URL — gap recorded", d)
    return []


# ---------------------------------------------------------------------------
# Per-date scoring
# ---------------------------------------------------------------------------
def _score_date(d: date) -> dict:
    """
    Fetch and score one trading day.
    Returns a row dict ready for pd.DataFrame construction.

    If fewer than 5 conflict headlines found (gap / quiet day / GDELT outage),
    all gate columns are False and dominant_class is 'neutral'.
    """
    headlines = _fetch_headlines_for_date(d)

    if len(headlines) < 5:
        return {
            "date": d,
            "headline_count": len(headlines),
            "dominant_class": "neutral",
            "dominant_prob": 0.0,
            "sentiment_score": 0.0,
            **{col: False for col in GATE_COLUMNS},
        }

    mean_probs = score_headlines(headlines)

    dominant_class = max(mean_probs, key=mean_probs.__getitem__)
    dominant_prob = round(mean_probs[dominant_class], 4)
    sentiment_score = round(mean_probs["positive"] - mean_probs["negative"], 4)

    gates = {
        _col(p, s): (dominant_prob > p and abs(sentiment_score) > s)
        for p in _P_THRESHOLDS
        for s in _S_THRESHOLDS
    }

    return {
        "date": d,
        "headline_count": len(headlines),
        "dominant_class": dominant_class,
        "dominant_prob": dominant_prob,
        "sentiment_score": sentiment_score,
        **gates,
    }


# ---------------------------------------------------------------------------
# Year-chunked helpers
# ---------------------------------------------------------------------------
def _chunk_path(output_path: Path, year: int) -> Path:
    """Path for a year's intermediate parquet, adjacent to the final output."""
    return output_path.parent / f"historical_sentiment_{year}.parquet"


def _load_chunk(path: Path) -> pd.DataFrame | None:
    """
    Load a year chunk parquet. Returns None if missing, unreadable, or empty.
    Partial chunks (mid-year interruption) are returned as-is; the caller
    computes which dates still need scoring.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if len(df) == 0:
            return None
        return df
    except Exception as exc:
        log.warning("Cannot read chunk %s (%s) — will re-score", path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def score_date_range(
    start: date,
    end: date,
    output_path: Path,
) -> pd.DataFrame:
    """
    Score all US business days in [start, end] and write to output_path.

    Year-chunked for resumability:
        - Each calendar year writes its own intermediate parquet.
        - A complete chunk (≥200 rows) is skipped on subsequent runs.
        - The final merged parquet is always rewritten from the year chunks.

    Returns the merged DataFrame (all dates, sorted).
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # All US business days in range (approximate — GDELT has data every day anyway)
    trading_days: list[date] = [ts.date() for ts in pd.bdate_range(str(start), str(end))]
    years = sorted({d.year for d in trading_days})

    log.info(
        "score_date_range: %s → %s | %d trading days | %d years",
        start, end, len(trading_days), len(years),
    )

    year_dfs: list[pd.DataFrame] = []

    for year in years:
        days_this_year = [d for d in trading_days if d.year == year]
        chunk_out = _chunk_path(output_path, year)
        existing_chunk = _load_chunk(chunk_out)

        # Compute which days still need scoring (handles partial chunks)
        if existing_chunk is not None:
            already_scored = set(existing_chunk["date"].dt.date)
            days_to_score = [d for d in days_this_year if d not in already_scored]
            if not days_to_score:
                log.info("[%d] Complete chunk loaded (%d rows) — skipping", year, len(existing_chunk))
                year_dfs.append(existing_chunk)
                continue
            log.info(
                "[%d] Resuming: %d already scored, %d remaining",
                year, len(existing_chunk), len(days_to_score),
            )
            existing_rows = existing_chunk.to_dict("records")
        else:
            days_to_score = days_this_year
            existing_rows = []

        log.info("[%d] Scoring %d days ...", year, len(days_to_score))

        new_rows = []
        t_year_start = time.time()
        for i, d in enumerate(days_to_score, 1):
            row = _score_date(d)
            new_rows.append(row)
            if i % 50 == 0:
                elapsed = time.time() - t_year_start
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(days_to_score) - i) / rate if rate > 0 else 0
                log.info(
                    "[%d] %d/%d scored | %.1f s/day | ~%.0f min remaining",
                    year, i, len(days_to_score),
                    elapsed / i,
                    remaining / 60,
                )

        all_rows = existing_rows + new_rows
        df_year = pd.DataFrame(all_rows)
        df_year["date"] = pd.to_datetime(df_year["date"])
        df_year = df_year.sort_values("date").reset_index(drop=True)

        df_year.to_parquet(chunk_out, index=False)
        year_elapsed = time.time() - t_year_start
        log.info(
            "[%d] Done: %d rows total | %.1f min this session | → %s",
            year, len(df_year), year_elapsed / 60, chunk_out.name,
        )
        year_dfs.append(df_year)

    if not year_dfs:
        log.warning("No data scored — returning empty DataFrame")
        return pd.DataFrame()

    df_merged = pd.concat(year_dfs, ignore_index=True)
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    df_merged.to_parquet(output_path, index=False)
    log.info("Merged: %d rows → %s", len(df_merged), output_path)

    return df_merged


# ---------------------------------------------------------------------------
# Validation: cross-check against live sentiment_cache.jsonl
# ---------------------------------------------------------------------------
def validate_against_cache(df: pd.DataFrame, cache_path: Path) -> int:
    """
    Compare scored dates against live sentiment_cache.jsonl.

    Mismatches are expected (live cache uses hourly GKG at random offsets;
    historical scorer uses fixed 14:00 UTC GKG) but big divergences flag
    pipeline errors. Reports dominant_class mismatches and score differences
    > 0.30.

    Returns number of mismatches found.
    """
    if not cache_path.exists():
        log.warning("Cache not found: %s", cache_path)
        return 0

    records = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        log.warning("Cache is empty")
        return 0

    log.info("Validating against %d live cache entries ...", len(records))

    # Index historical df by date for fast lookup
    df_indexed = df.copy()
    df_indexed["_date_key"] = df_indexed["date"].dt.date
    df_indexed = df_indexed.set_index("_date_key")

    mismatches = 0
    compared = 0

    for rec in records:
        ts_str = rec.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            d = ts.date()
        except (ValueError, AttributeError):
            continue

        if d not in df_indexed.index:
            continue

        compared += 1
        row = df_indexed.loc[d]
        h_class = row["dominant_class"]
        h_score = float(row["sentiment_score"])
        c_class = rec["dominant_class"]
        c_score = float(rec["sentiment_score"])

        score_diff = abs(h_score - c_score)
        class_match = h_class == c_class

        if not class_match or score_diff > 0.30:
            log.warning(
                "[%s] MISMATCH — historical: %s/%.3f | cache: %s/%.3f | Δscore=%.3f",
                d, h_class, h_score, c_class, c_score, score_diff,
            )
            mismatches += 1

    log.info(
        "Validation: %d compared, %d mismatches (%.0f%% match rate)",
        compared, mismatches,
        100 * (1 - mismatches / compared) if compared else 0,
    )
    return mismatches


# ---------------------------------------------------------------------------
# Distribution report
# ---------------------------------------------------------------------------
def print_distribution(df: pd.DataFrame, label: str) -> None:
    """Print sentiment distribution and gate-open rates to stdout."""
    print(f"\n{'='*60}")
    print(f"Sentiment distribution — {label}")
    print(f"{'='*60}")
    print(f"Total rows : {len(df)}")

    print("\ndominant_class:")
    print(df["dominant_class"].value_counts(normalize=True).map("{:.1%}".format).to_string())

    print(
        f"\ndominant_prob  : mean={df['dominant_prob'].mean():.4f}  "
        f"std={df['dominant_prob'].std():.4f}"
    )
    print(
        f"sentiment_score: mean={df['sentiment_score'].mean():.4f}  "
        f"std={df['sentiment_score'].std():.4f}"
    )

    print("\nGate-open rate (% of trading days):")
    for col in GATE_COLUMNS:
        if col in df.columns:
            rate = df[col].mean() * 100
            marker = " ◄ live threshold" if col == LIVE_GATE_COLUMN else ""
            print(f"  {col}: {rate:5.1f}%{marker}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score GDELT sentiment retroactively for the Synk backtest."
    )
    parser.add_argument(
        "--start",
        default="2026-04-22",
        metavar="YYYY-MM-DD",
        help="Start date (default: 2026-04-22 — live paper-phase start)",
    )
    parser.add_argument(
        "--end",
        default="2026-05-27",
        metavar="YYYY-MM-DD",
        help="End date inclusive (default: 2026-05-27)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cross-check against logs/sentiment_cache.jsonl after scoring",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output parquet path (default: backtest/results/historical_sentiment_START_END.parquet)",
    )
    args = parser.parse_args()

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = _RESULTS_DIR / f"historical_sentiment_{args.start}_{args.end}.parquet"

    log.info("=== historical_sentiment start ===")
    log.info("Range  : %s → %s", args.start, args.end)
    log.info("Output : %s", out_path)

    df = score_date_range(start_d, end_d, out_path)

    if df.empty:
        log.error("No data produced — check network and GDELT availability")
        sys.exit(1)

    print_distribution(df, f"{args.start} → {args.end}")

    # Quick sanity check: warn if > 5% of days have no data
    gap_pct = (df["headline_count"] == 0).mean() * 100
    if gap_pct > 5:
        log.warning("%.1f%% of days have 0 headlines — check GDELT coverage for this range", gap_pct)
    else:
        log.info("Data coverage OK: %.1f%% gap rate", gap_pct)

    if args.validate:
        cache = _HERE / "logs" / "sentiment_cache.jsonl"
        validate_against_cache(df, cache)

    log.info("=== historical_sentiment complete ===")
