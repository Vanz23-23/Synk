"""
backtest/universe_validate_live

Validate the defence-prime universe expansion under LIVE-faithful gating.

Two fidelity corrections vs universe_expansion.py:
  1. Defence names gated at z >= 1.0 (DEFENCE_Z_THRESHOLD), matching live
     synk_strategy._DEFENCE_Z_THRESHOLD — not the loose backtest z >= 0.50.
  2. Treasury havens (TLT, IEF) dropped — they added ~10 trades and were
     near-dead weight in the loose-gate run.

Compares, both at z>=1.0 defence gating + 3-gate stack + base slippage:
  Baseline : GLD, FXY, LMT, NOC, ITA
  Expanded : baseline + GD, RTX, LHX  (defence primes)

This answers: does the defence-prime edge survive the real live gate, or was
the +75% trades / +8.5% return in universe_expansion.py an artefact of the
loose z>=0.50 defence gate?

Patches bt.DEFENCE_Z_THRESHOLD / bt.SYMBOLS / bt._DEFENCE before each run so
synk_backtest.py is not modified. Backtest-only.

CAVEAT this CANNOT remove: defence stocks rode a 2022-2025 secular rearmament
bull. Surviving the z>=1.0 gate confirms the gate fidelity, not that the edge
is geopolitical-drift alpha vs defence beta. (See haven-only backtest for the
beta-isolation lens.)

Usage (from synk/ root):
    python backtest/universe_validate_live.py

Runtime: ~5-10 min (Yahoo data cached by Lumibot after first run).
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

NEW_DEFENCE: list[str] = ["GD", "RTX", "LHX"]
LIVE_DEFENCE_Z: float = 1.0  # matches synk_strategy._DEFENCE_Z_THRESHOLD

# Defence-prime slippage (base scenario), same tier as LMT/NOC/ITA
_NEW_SLIPPAGE_BASE: dict[str, float] = {"GD": 10.0, "RTX": 10.0, "LHX": 10.0}


def run_all() -> dict[str, dict]:
    results: dict[str, dict] = {}

    orig_symbols = list(bt.SYMBOLS)
    orig_defence = set(bt._DEFENCE)
    orig_def_z = bt.DEFENCE_Z_THRESHOLD

    # Apply live-faithful defence gate to BOTH runs
    bt.DEFENCE_Z_THRESHOLD = LIVE_DEFENCE_Z

    # --- Baseline (current 5) at z>=1.0 defence ---
    print(f"\n{'='*60}\nBASELINE @ z>=1.0 defence: {orig_symbols}\n{'='*60}")
    results["baseline_z1.0"], _ = bt.run_scenario(
        "validate_baseline_z1.0", bt.SLIPPAGE_SCENARIOS["base"]
    )

    # --- Expanded (+ defence primes) at z>=1.0 defence ---
    bt.SYMBOLS = orig_symbols + NEW_DEFENCE
    bt._DEFENCE = orig_defence | set(NEW_DEFENCE)
    expanded_slippage = {**bt.SLIPPAGE_SCENARIOS["base"], **_NEW_SLIPPAGE_BASE}
    print(f"\n{'='*60}\nEXPANDED @ z>=1.0 defence: {bt.SYMBOLS}\n{'='*60}")
    results["expanded_z1.0"], _ = bt.run_scenario(
        "validate_expanded_z1.0", expanded_slippage
    )

    # Restore module state
    bt.SYMBOLS = orig_symbols
    bt._DEFENCE = orig_defence
    bt.DEFENCE_Z_THRESHOLD = orig_def_z
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
    labels = ["baseline_z1.0", "expanded_z1.0"]
    col = 16

    def row(label: str, key: str, pct: bool = False) -> str:
        vals = "  ".join(
            f"{_fmt(results[lbl].get(key), pct):>{col}}" for lbl in labels
        )
        return f"  {label:<18}{vals}"

    header_vals = "  ".join(f"{lbl:>{col}}" for lbl in labels)
    divider = "  " + "-" * (18 + (col + 2) * len(labels))

    lines = [
        f"UNIVERSE VALIDATION @ LIVE GATE (z>=1.0 defence) — {run_date}",
        f"Base slippage | 3-gate stack ({bt.LIVE_GATE_COLUMN}) | 2020-01-01 to 2026-01-01",
        f"Added: {NEW_DEFENCE} (defence primes). Treasuries dropped.",
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
        f"  baseline: {bt.format_skip_diag(results['baseline_z1.0'].get('skip_diag', {}))}",
        f"  expanded: {bt.format_skip_diag(results['expanded_z1.0'].get('skip_diag', {}))}",
        "",
        "NOTE: defence gated at z>=1.0 (live-faithful). regime_defence_closed",
        "      counts entries blocked by that gate in the 0.5<=z<1.0 band.",
        "CAVEAT: surviving this gate confirms fidelity, NOT that defence-prime",
        "        return is alpha vs 2022-2025 secular beta.",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    out_path = Path(__file__).parent / "results_summary_universe_validate.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SYNK UNIVERSE VALIDATION @ LIVE GATE")
    print(f"Adding: {NEW_DEFENCE} | defence gate z>={LIVE_DEFENCE_Z}")
    print("Scenario: base slippage | 3-gate stack | treasuries dropped")
    print("=" * 60)

    results = run_all()
    print_and_save(results)
