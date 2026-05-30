"""
backtest/universe_expansion

Test whether expanding the tradeable universe increases trade frequency
without diluting the per-signal edge.

Baseline : GLD, FXY, LMT, NOC, ITA              (current 5)
Expanded : baseline + GD, RTX, LHX              (defence primes)
                    + TLT, IEF                  (treasury havens)

Both runs use the full 3-gate stack (regime + momentum + sentiment) at
base slippage. Patches bt.SYMBOLS / bt._DEFENCE / slippage before each run
so synk_backtest.py is not modified — same pattern as regime_comparison.py.

Backtest-only: live config (price_feed.SYMBOLS, synk_strategy.py) is untouched.

NOTE: the backtest gates all symbols at z >= 0.50. Live gates defence names
at z >= 1.0 (synk_strategy._DEFENCE_Z_THRESHOLD). New defence-prime trade
counts here are therefore an upper bound vs live behaviour.

Usage (from synk/ root):
    python backtest/universe_expansion.py

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

# New instruments (user-selected): defence primes + treasury havens
NEW_DEFENCE: list[str] = ["GD", "RTX", "LHX"]
NEW_HAVEN: list[str] = ["TLT", "IEF"]

# Slippage (base scenario): primes match existing defence tier (10 bps);
# treasuries are highly liquid ETFs -> GLD-like tier (5 bps).
_NEW_SLIPPAGE_BASE: dict[str, float] = {
    "GD": 10.0, "RTX": 10.0, "LHX": 10.0,
    "TLT": 5.0, "IEF": 5.0,
}


def run_all() -> dict[str, dict]:
    """Run baseline then expanded universe; return {label: stats}."""
    results: dict[str, dict] = {}

    # Preserve originals for restore
    orig_symbols = list(bt.SYMBOLS)
    orig_defence = set(bt._DEFENCE)

    # --- Baseline (current 5 symbols) ---
    print(f"\n{'='*60}\nBASELINE universe: {orig_symbols}\n{'='*60}")
    results["baseline"], _ = bt.run_scenario(
        "universe_baseline", bt.SLIPPAGE_SCENARIOS["base"]
    )

    # --- Expanded universe ---
    bt.SYMBOLS = orig_symbols + NEW_DEFENCE + NEW_HAVEN
    bt._DEFENCE = orig_defence | set(NEW_DEFENCE)
    expanded_slippage = {**bt.SLIPPAGE_SCENARIOS["base"], **_NEW_SLIPPAGE_BASE}
    print(f"\n{'='*60}\nEXPANDED universe: {bt.SYMBOLS}\n{'='*60}")
    results["expanded"], _ = bt.run_scenario("universe_expanded", expanded_slippage)

    # Restore module state
    bt.SYMBOLS = orig_symbols
    bt._DEFENCE = orig_defence
    return results


def _fmt(v: object, pct: bool = False) -> str:
    if v in (None, "N/A"):
        return "N/A"
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v * 100:.1f}%" if pct else f"{v:.4f}"
    return str(v)


def print_and_save(results: dict[str, dict]) -> None:
    run_date = datetime.now().strftime("%Y-%m-%d")
    labels = ["baseline", "expanded"]
    col = 14

    def row(label: str, key: str, pct: bool = False) -> str:
        vals = "  ".join(
            f"{_fmt(results[lbl].get(key), pct):>{col}}" for lbl in labels
        )
        return f"  {label:<18}{vals}"

    header_vals = "  ".join(f"{lbl:>{col}}" for lbl in labels)
    divider = "  " + "-" * (18 + (col + 2) * len(labels))

    lines = [
        f"UNIVERSE EXPANSION COMPARISON — {run_date}",
        f"Base slippage | 3-gate stack ({bt.LIVE_GATE_COLUMN}) | 2020-01-01 to 2026-01-01",
        f"Added: {NEW_DEFENCE} (defence primes) + {NEW_HAVEN} (treasury havens)",
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
        "ENTRY-LOOP SKIP DIAGNOSTIC:",
        f"  baseline: {bt.format_skip_diag(results['baseline'].get('skip_diag', {}))}",
        f"  expanded: {bt.format_skip_diag(results['expanded'].get('skip_diag', {}))}",
        "",
        "NOTE: GLD trades = 0 — Lumibot won't trade the benchmark asset (GLD).",
        "NOTE: backtest gates defence at z>=0.50; live uses z>=1.0. New-prime",
        "      counts are an upper bound vs live. Backtest-only experiment.",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    out_path = Path(__file__).parent / "results_summary_universe.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SYNK UNIVERSE EXPANSION TEST")
    print(f"Adding: {NEW_DEFENCE + NEW_HAVEN}")
    print("Scenario: base slippage | 3-gate stack")
    print("=" * 60)

    results = run_all()
    print_and_save(results)
