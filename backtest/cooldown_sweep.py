"""
backtest/cooldown_sweep

Test whether the 3-day post-exit cooldown is a real constraint on trade
frequency. Sweeps COOLDOWN_TRADING_DAYS across [0, 1, 3, 5] using the full
3-gate stack at base slippage.

Patches bt.COOLDOWN_TRADING_DAYS before each run so synk_backtest.py is not
modified — same pattern as regime_comparison.py.

Read alongside the entry-loop skip diagnostic: if `cooldown` is a small share
of blocked opportunities and `occupied` dominates, the real bottleneck is
one-position-per-symbol, not the cooldown.

Usage (from synk/ root):
    python backtest/cooldown_sweep.py

Runtime: ~5-20 min (Yahoo data cached by Lumibot after first run).
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

COOLDOWNS: list[int] = [0, 1, 3, 5]


def run_all() -> dict[int, dict]:
    results: dict[int, dict] = {}
    orig = bt.COOLDOWN_TRADING_DAYS
    for cd in COOLDOWNS:
        bt.COOLDOWN_TRADING_DAYS = cd
        print(f"\n{'='*60}\nCOOLDOWN = {cd} trading days\n{'='*60}")
        results[cd], _ = bt.run_scenario(f"cooldown_{cd}", bt.SLIPPAGE_SCENARIOS["base"])
    bt.COOLDOWN_TRADING_DAYS = orig  # restore default
    return results


def _fmt(v: object, pct: bool = False) -> str:
    if v in (None, "N/A"):
        return "N/A"
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v * 100:.1f}%" if pct else f"{v:.4f}"
    return str(v)


def print_and_save(results: dict[int, dict]) -> None:
    run_date = datetime.now().strftime("%Y-%m-%d")
    labels = [f"cd={cd}" for cd in COOLDOWNS]
    col = 12

    def row(label: str, key: str, pct: bool = False) -> str:
        vals = "  ".join(
            f"{_fmt(results[cd].get(key), pct):>{col}}" for cd in COOLDOWNS
        )
        return f"  {label:<18}{vals}"

    header_vals = "  ".join(f"{lbl:>{col}}" for lbl in labels)
    divider = "  " + "-" * (18 + (col + 2) * len(COOLDOWNS))

    lines = [
        f"COOLDOWN SWEEP COMPARISON — {run_date}",
        f"Base slippage | 3-gate stack ({bt.LIVE_GATE_COLUMN}) | 2020-01-01 to 2026-01-01",
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
        "ENTRY-LOOP SKIP DIAGNOSTIC:",
        *[f"  cd={cd}: {bt.format_skip_diag(results[cd].get('skip_diag', {}))}" for cd in COOLDOWNS],
        "",
        "NOTE: if `occupied` >> `cooldown` across all columns, the binding",
        "      constraint is one-position-per-symbol, not the cooldown.",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    out_path = Path(__file__).parent / "results_summary_cooldown.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SYNK COOLDOWN SWEEP")
    print(f"Cooldowns: {COOLDOWNS} trading days")
    print("Scenario:  base slippage | 3-gate stack")
    print("=" * 60)

    results = run_all()
    print_and_save(results)
