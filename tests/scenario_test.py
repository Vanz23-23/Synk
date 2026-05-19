"""
scenario_test

Runs three synthetic gate evaluations against the real Synk strategy logic.
Does NOT submit orders, touch live files, call Alpaca, or modify anything.

Scenarios:
    BAD    - all three gates closed, bot sits idle
    NORMAL - regime and momentum open, sentiment just misses
    GREAT  - all gates open, full trade instruction generated

Run from synk/ root:
    python tests/scenario_test.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Make synk/ importable regardless of working directory
# ---------------------------------------------------------------------------
_SYNK_ROOT = Path(__file__).resolve().parent.parent
if str(_SYNK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SYNK_ROOT))

from signals.regime_filter import RegimeSignal, RegimeState
from signals.regime_filter import is_gate_open as regime_gate_open
from signals.momentum import MomentumSignal
from signals.momentum import is_gate_open as momentum_gate_open
from signals.sentiment import SentimentSignal
from signals.sentiment import is_gate_open as sentiment_gate_open
from strategy.synk_strategy import (
    GateResult,
    build_trade_instruction,
)

# ---------------------------------------------------------------------------
# Synthetic NAV (matches paper account starting capital)
# ---------------------------------------------------------------------------
_NAV = 100_000.0
_SYMBOL = "GLD"
_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
_NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict] = {
    "BAD": {
        "label": "BAD - no signal, all gates closed",
        "regime": RegimeSignal(
            date=_TODAY,
            gpr_value=95.0,
            z_score=0.05,           # flat, well below ELEVATED threshold (0.5)
            regime=RegimeState.NORMAL,
        ),
        "momentum": MomentumSignal(
            date=_TODAY,
            symbol=_SYMBOL,
            roc_20=-0.052,          # negative momentum
            sma_20=430.0,
            close=408.5,            # well below SMA
            signal=False,
        ),
        "sentiment": SentimentSignal(
            timestamp=_NOW,
            headline_count=50,
            dominant_class="neutral",
            dominant_prob=0.45,     # below 0.6 threshold
            sentiment_score=-0.10,  # abs < 0.3 threshold
            signal=False,
        ),
        "safe_haven": {"confirmed": True, "return_5d": 0.005, "regime": "SAFE_HAVEN"},
    },
    "NORMAL": {
        "label": "NORMAL - regime + momentum open, sentiment just misses",
        "regime": RegimeSignal(
            date=_TODAY,
            gpr_value=145.0,
            z_score=0.9,            # ELEVATED - gate 1 open
            regime=RegimeState.ELEVATED,
        ),
        "momentum": MomentumSignal(
            date=_TODAY,
            symbol=_SYMBOL,
            roc_20=0.031,           # positive ROC - gate 3 open
            sma_20=430.0,
            close=434.2,            # above SMA - gate 3 open
            signal=True,
        ),
        "sentiment": SentimentSignal(
            timestamp=_NOW,
            headline_count=50,
            dominant_class="neutral",
            dominant_prob=0.58,     # just below 0.6 - gate 2 CLOSED
            sentiment_score=-0.27,  # abs just below 0.3 - gate 2 CLOSED
            signal=False,
        ),
        "safe_haven": {"confirmed": True, "return_5d": 0.012, "regime": "SAFE_HAVEN"},
    },
    "GREAT": {
        "label": "GREAT - all gates open, trade instruction generated",
        "regime": RegimeSignal(
            date=_TODAY,
            gpr_value=210.0,
            z_score=1.8,            # HIGH - gate 1 open
            regime=RegimeState.HIGH,
        ),
        "momentum": MomentumSignal(
            date=_TODAY,
            symbol=_SYMBOL,
            roc_20=0.072,           # strong positive ROC - gate 3 open
            sma_20=430.0,
            close=447.5,            # well above SMA - gate 3 open
            signal=True,
        ),
        "sentiment": SentimentSignal(
            timestamp=_NOW,
            headline_count=50,
            dominant_class="negative",
            dominant_prob=0.74,     # above 0.6 - gate 2 open
            sentiment_score=-0.56,  # abs > 0.3 - gate 2 open
            signal=True,
        ),
        "safe_haven": {"confirmed": True, "return_5d": 0.024, "regime": "SAFE_HAVEN"},
    },
}


# ---------------------------------------------------------------------------
# Gate evaluation (mirrors _build_reason logic without importing private fn)
# ---------------------------------------------------------------------------
def evaluate_scenario(name: str, s: dict) -> GateResult:
    regime = s["regime"]
    momentum = s["momentum"]
    sentiment = s["sentiment"]
    safe_haven = s["safe_haven"]

    closed: list[str] = []

    if not regime_gate_open(regime):
        closed.append(
            f"REGIME=CLOSED  z={regime.z_score:+.3f} ({regime.regime.value}) - needs z >= 0.5"
        )
    if not momentum_gate_open(momentum):
        closed.append(
            f"MOMENTUM=CLOSED  roc={momentum.roc_20:+.4f}, "
            f"close={momentum.close:.2f} vs sma={momentum.sma_20:.2f}"
        )
    if not sentiment_gate_open(sentiment):
        closed.append(
            f"SENTIMENT=CLOSED  class={sentiment.dominant_class}, "
            f"prob={sentiment.dominant_prob:.3f} (need >0.6), "
            f"score={sentiment.sentiment_score:+.3f} (need abs >0.3)"
        )
    if not safe_haven["confirmed"]:
        closed.append(
            f"SH_HOSTILE_REGIME  return_5d={safe_haven['return_5d']:+.2%}"
        )

    all_open = len(closed) == 0

    if all_open:
        reason = (
            f"All gates OPEN | "
            f"regime={regime.regime.value} z={regime.z_score:+.3f} | "
            f"roc={momentum.roc_20:+.4f} close={momentum.close:.2f}>{momentum.sma_20:.2f} | "
            f"sentiment={sentiment.dominant_class} prob={sentiment.dominant_prob:.3f} "
            f"score={sentiment.sentiment_score:+.3f}"
        )
    else:
        reason = "Gate(s) closed:\n      " + "\n      ".join(closed)

    return GateResult(
        timestamp=_NOW,
        symbol=_SYMBOL,
        regime_signal=regime,
        momentum_signal=momentum,
        sentiment_signal=sentiment,
        all_open=all_open,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_scenario(name: str, s: dict) -> None:
    result = evaluate_scenario(name, s)

    print(f"\n{'-' * 60}")
    print(f"  SCENARIO: {s['label']}")
    print(f"{'-' * 60}")

    # Gate status table
    regime = s["regime"]
    momentum = s["momentum"]
    sentiment = s["sentiment"]

    g1 = "OPEN  [OK]" if regime_gate_open(regime) else "CLOSED [--]"
    g2 = "OPEN  [OK]" if sentiment_gate_open(sentiment) else "CLOSED [--]"
    g3 = "OPEN  [OK]" if momentum_gate_open(momentum) else "CLOSED [--]"

    print(f"\n  Gate 1 - GPR Regime    [{g1}]"
          f"  z={regime.z_score:+.3f}  ({regime.regime.value})")
    print(f"  Gate 2 - FinBERT       [{g2}]"
          f"  class={sentiment.dominant_class}  "
          f"prob={sentiment.dominant_prob:.3f}  score={sentiment.sentiment_score:+.3f}")
    print(f"  Gate 3 - Momentum      [{g3}]"
          f"  roc={momentum.roc_20:+.4f}  "
          f"close={momentum.close:.2f}  sma={momentum.sma_20:.2f}")

    print(f"\n  Result: {'ALL OPEN -> WOULD TRADE' if result.all_open else 'BLOCKED -> NO TRADE'}")

    if not result.all_open:
        for line in result.reason.split("\n"):
            print(f"  {line}")
    else:
        try:
            instr = build_trade_instruction(result, _NAV)
            print(f"\n  {'-' * 40}")
            print(f"  TRADE INSTRUCTION ({_SYMBOL})")
            print(f"  {'-' * 40}")
            print(f"  Direction:  {instr.direction}")
            print(f"  Quantity:   {instr.quantity} shares")
            print(f"  Entry:      ${instr.entry_price:.2f}")
            print(f"  Allocation: {instr.allocation_pct * 100:.2f}% of NAV  "
                  f"(${instr.allocation_pct * _NAV:,.0f})")
            print(f"  Kelly (QK): {instr.kelly_fraction:.4f}")
            print(f"  Stop loss:  ${instr.entry_price * 0.92:.2f}  (-8%)")
            print(f"  *** ORDER NOT SUBMITTED - test only ***")
        except ValueError as exc:
            print(f"  TradeInstruction error: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"  SYNK SCENARIO TEST")
    print(f"  Symbol: {_SYMBOL}  |  Paper NAV: GBP{_NAV:,.0f}")
    print(f"  {_NOW}")
    print(f"{'=' * 60}")

    for name, scenario in SCENARIOS.items():
        print_scenario(name, scenario)

    print(f"\n{'=' * 60}")
    print("  Test complete. No orders submitted. No files modified.")
    print(f"{'=' * 60}\n")
