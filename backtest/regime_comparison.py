"""
backtest/regime_comparison

Compare strategy performance across regime z-score thresholds: 0.0, 0.25, 0.5.
Base slippage scenario only. Patches REGIME_Z_THRESHOLD before each run so
synk_backtest.py is not modified.

Usage (from synk/ root):
    python backtest/regime_comparison.py

Runtime: ~5-15 min (Yahoo data cached by Lumibot after first run).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

if "ALPACA_API_SECRET" not in os.environ:
    os.environ["ALPACA_API_SECRET"] = os.environ.get("ALPACA_SECRET_KEY", "placeholder")

import backtest.synk_backtest as bt  # noqa: E402

THRESHOLDS: list[float] = [0.0, 0.25, 0.50]
_RESULTS_DIR = Path(__file__).parent / "results"


def run_all() -> dict[float, dict]:
    results: dict[float, dict] = {}
    for z in THRESHOLDS:
        bt.REGIME_Z_THRESHOLD = z
        label = f"z{z:.2f}_base"
        print(f"\n{'='*60}")
        print(f"THRESHOLD z >= {z:.2f}")
        print(f"{'='*60}")
        stats, _ = bt.run_scenario(label, bt.SLIPPAGE_SCENARIOS["base"])
        results[z] = stats
    bt.REGIME_Z_THRESHOLD = 0.50  # restore default
    return results


def _fmt(v: object, pct: bool = False) -> str:
    if v in (None, "N/A"):
        return "N/A"
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v * 100:.1f}%" if pct else f"{v:.4f}"
    return str(v)


def print_and_save(results: dict[float, dict]) -> None:
    run_date = datetime.now().strftime("%Y-%m-%d")
    labels = [f"z>={z:.2f}" for z in THRESHOLDS]
    col = 12

    def row(label: str, key: str, pct: bool = False) -> str:
        vals = "  ".join(
            f"{_fmt(results[z].get(key), pct):>{col}}" for z in THRESHOLDS
        )
        return f"  {label:<18}{vals}"

    header_vals = "  ".join(f"{lbl:>{col}}" for lbl in labels)
    divider = "  " + "-" * (18 + (col + 2) * len(THRESHOLDS))

    lines = [
        f"REGIME THRESHOLD COMPARISON — {run_date}",
        "Base slippage | Gate 3 (sentiment) OMITTED | 2020-01-01 to 2026-01-01",
        "",
        f"  {'':18}{header_vals}",
        divider,
        row("Total trades:",   "total_trades"),
        row("Win rate:",       "win_rate",     pct=True),
        row("Profit factor:",  "profit_factor"),
        row("Total return:",   "total_return", pct=True),
        row("Max drawdown:",   "max_drawdown", pct=True),
        row("Sharpe ratio:",   "sharpe_ratio"),
        row("Avg hold days:",  "avg_hold_days"),
        "",
        row("GLD trades:",     "gld_trades"),
        row("FXY trades:",     "fxy_trades"),
        row("Defence trades:", "def_trades"),
        "",
        "NOTE: GLD trades = 0 — Lumibot won't trade the benchmark asset (GLD).",
        "      In the live strategy GLD IS tradeable. See synk_backtest.py.",
        "NOTE: Sentiment gate omitted — all figures ~15-25% optimistic vs live.",
        "      Relative comparison between thresholds remains valid.",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    out_path = Path(__file__).parent / "regime_comparison_results.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("SYNK REGIME THRESHOLD COMPARISON")
    print(f"Thresholds: {THRESHOLDS}")
    print("Scenario:   base slippage")
    print("Sentiment gate OMITTED — results optimistic vs live.")
    print("=" * 60)

    results = run_all()
    print_and_save(results)
