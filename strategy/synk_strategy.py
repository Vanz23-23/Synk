"""
synk_strategy

Top-level strategy orchestrator for Synk. Evaluates all three signal gates
for each symbol in the watchlist and, when all gates agree, computes a
TradeInstruction using Quarter-Kelly position sizing.

Gate logic: hard AND — all three must return signal=True simultaneously.
    Gate 1 (Regime):   GPR z-score via signals/regime_filter.py
    Gate 2 (Momentum): ROC(20) > 0 AND Close > SMA(20) via signals/momentum.py
    Gate 3 (Sentiment): FinBERT dominant_prob > 0.6 AND abs(score) > 0.3

This module produces TradeInstruction objects only — it does NOT submit orders.
Order submission is the next structural layer (after kill switch wiring is done).

Kill switch check: kill_switch_active() is called before any TradeInstruction
is returned. If HALTED, the instruction is suppressed and GateResult explains why.

Position sizing (Quarter-Kelly):
    kelly  = (win_rate * avg_win_ratio - (1 - win_rate)) / avg_win_ratio
    size   = kelly / 4
    capped to [POSITION_FLOOR_PCT, POSITION_CAP_PCT] of current NAV
    Default params for paper phase (hardcoded until 50 trades logged):
        win_rate = 0.5, avg_win_ratio = 1.5 → kelly ~16.7% → QK ~4.2% → capped at 4%

Usage (standalone):
    python strategy/synk_strategy.py

Deps: signals/*, risk/kill_switch.py, data/price_feed.py, config.py
Run from synk/ root.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from signals.regime_filter import (
    RegimeSignal,
    get_latest_regime,
    is_gate_open as regime_gate_open,
    _DEFAULT_GPR_PATH,
    check_safe_haven_confirmation,
    check_fxy_viability,
)
from signals.momentum import (
    MomentumSignal,
    get_latest_momentum,
    is_gate_open as momentum_gate_open,
)
from signals.sentiment import (
    SentimentSignal,
    get_latest_cached as get_latest_sentiment,
    is_gate_open as sentiment_gate_open,
)
from data.price_feed import get_prices, SYMBOLS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_LOG_DIR = _HERE / "logs"

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
# Quarter-Kelly paper-phase defaults — replace once 50 trades are logged
_DEFAULT_WIN_RATE = 0.50
_DEFAULT_AVG_WIN_RATIO = 1.50   # avg_win / avg_loss

POSITION_FLOOR_PCT = 0.02       # minimum allocation: 2% of NAV
POSITION_CAP_PCT = 0.04         # maximum allocation: 4% of NAV

# Defence leg requires stronger regime signal to filter secular re-rating noise.
# Backtest (2020–2026) showed 47% of defence trades fired at z=0.5–1.0, the band
# most likely to reflect structural rearmament spending rather than acute GPR spikes.
_DEFENCE_SYMBOLS: frozenset[str] = frozenset({"LMT", "NOC", "ITA"})
_DEFENCE_Z_THRESHOLD: float = 1.0  # raised from the global 0.5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synk_strategy")
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
class GateResult:
    timestamp: str
    symbol: str
    regime_signal: RegimeSignal
    momentum_signal: MomentumSignal
    sentiment_signal: SentimentSignal | None  # None if cache is empty
    all_open: bool
    reason: str     # human-readable: what fired or why it closed


@dataclass(frozen=True)
class TradeInstruction:
    symbol: str
    direction: str          # 'BUY' (shorting not implemented in paper phase)
    quantity: int
    entry_price: float
    kelly_fraction: float   # raw QK fraction before floor/cap
    allocation_pct: float   # actual allocation as % of NAV (post floor/cap)
    rationale: str


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def compute_kelly(
    win_rate: float = _DEFAULT_WIN_RATE,
    avg_win_ratio: float = _DEFAULT_AVG_WIN_RATIO,
) -> float:
    """
    Compute Quarter-Kelly fraction.
    kelly = (win_rate * avg_win_ratio - (1 - win_rate)) / avg_win_ratio
    Returns kelly / 4.
    """
    full_kelly = (win_rate * avg_win_ratio - (1.0 - win_rate)) / avg_win_ratio
    return max(full_kelly / 4.0, 0.0)  # clamp to 0 if Kelly goes negative


def compute_position(
    nav: float,
    entry_price: float,
    kelly_fraction: float,
) -> tuple[int, float]:
    """
    Convert a Kelly fraction to (quantity, actual_allocation_pct).
    Applies floor/cap constraints. Returns (0, 0.0) if entry_price is 0.
    """
    if entry_price <= 0:
        return 0, 0.0
    raw_allocation = kelly_fraction * nav
    allocation = max(raw_allocation, nav * POSITION_FLOOR_PCT)
    allocation = min(allocation, nav * POSITION_CAP_PCT)
    quantity = int(allocation / entry_price)
    actual_pct = (quantity * entry_price) / nav if nav > 0 else 0.0
    return quantity, actual_pct


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def _build_reason(
    regime: RegimeSignal,
    momentum: MomentumSignal,
    sentiment: SentimentSignal | None,
    kill_halted: bool,
    symbol: str = "",
    safe_haven: dict | None = None,
    fxy_viability: dict | None = None,
) -> tuple[bool, str]:
    """
    Evaluate hard AND gate logic. Returns (all_open: bool, reason: str).
    """
    if kill_halted:
        return False, "Kill switch HALTED — all trading suspended"

    closed: list[str] = []

    if not regime_gate_open(regime):
        closed.append(
            f"REGIME=CLOSED (state={regime.regime.value}, z={regime.z_score:+.3f})"
        )
    elif symbol in _DEFENCE_SYMBOLS and regime.z_score < _DEFENCE_Z_THRESHOLD:
        closed.append(
            f"REGIME=CLOSED (defence threshold: z={regime.z_score:+.3f} < {_DEFENCE_Z_THRESHOLD})"
        )
    if not momentum_gate_open(momentum):
        closed.append(
            f"MOMENTUM=CLOSED "
            f"(roc={momentum.roc_20:+.4f}, close={momentum.close:.2f} vs sma={momentum.sma_20:.2f})"
        )
    if sentiment is None:
        closed.append("SENTIMENT=CLOSED (no cached signal — run sentiment.py first)")
    elif not sentiment_gate_open(sentiment):
        closed.append(
            f"SENTIMENT=CLOSED "
            f"(class={sentiment.dominant_class}, prob={sentiment.dominant_prob:.3f}, "
            f"score={sentiment.sentiment_score:+.3f})"
        )

    # GLD-only: safe-haven confirmation — blocks entry if gold is falling during GPR spike
    if symbol == "GLD" and safe_haven is not None:
        if not safe_haven["confirmed"]:
            closed.append(
                f"SH_HOSTILE_REGIME (return_5d={safe_haven['return_5d']:+.2%}, "
                f"regime={safe_haven['regime']})"
            )

    # FXY-only: 52-week low proximity — blocks entry in dollar-strength / BoJ divergence regime
    if symbol == "FXY" and fxy_viability is not None:
        if not fxy_viability["viable"]:
            closed.append(
                f"FXY_52W_LOW_GATE (current={fxy_viability['current']:.4f}, "
                f"low={fxy_viability['low_52w']:.4f}, "
                f"dist={fxy_viability['distance_from_low']:.2%})"
            )

    if closed:
        return False, "Gate(s) closed: " + " | ".join(closed)

    return True, (
        f"All gates open | "
        f"regime={regime.regime.value} z={regime.z_score:+.3f} | "
        f"roc={momentum.roc_20:+.4f} close>{momentum.sma_20:.2f} | "
        f"sentiment={sentiment.dominant_class} prob={sentiment.dominant_prob:.3f}"  # type: ignore[union-attr]
    )


def check_all_gates(
    symbol: str,
    gpr_path: Path = _DEFAULT_GPR_PATH,
    price_df=None,
    regime_sig: RegimeSignal | None = None,
) -> GateResult:
    """
    Evaluate all three gates for a single symbol.

    Args:
        symbol:     ticker to evaluate momentum for
        gpr_path:   path to GPR daily XLS (defaults to data/gpr_daily_recent.xls)
        price_df:   pre-loaded price DataFrame for the symbol; fetched if None
        regime_sig: pre-computed RegimeSignal; loaded from GPR if None.
                    Pass this when evaluating multiple symbols to avoid
                    re-reading the 15k-row XLS on every call.

    Returns a GateResult. Never raises — individual gate failures are caught
    and surfaced in GateResult.reason so the loop can continue.
    """
    from risk.kill_switch import kill_switch_active  # noqa: PLC0415

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    kill_halted = kill_switch_active()

    # Gate 1: Regime — use pre-loaded signal if available
    if regime_sig is None:
        try:
            regime_sig = get_latest_regime(gpr_path)
        except Exception as exc:
            log.error("Regime gate error for %s: %s", symbol, exc)
            regime_sig = RegimeSignal(
                date=now[:10], gpr_value=0.0, z_score=0.0,
                regime=__import__("signals.regime_filter", fromlist=["RegimeState"]).RegimeState.NORMAL,
            )

    # Gate 2: Momentum — load price data if not supplied
    if price_df is None:
        prices = get_prices([symbol])
        price_df = prices[symbol]

    try:
        momentum_sig = get_latest_momentum(symbol, price_df)
    except Exception as exc:
        log.error("Momentum gate error for %s: %s", symbol, exc)
        momentum_sig = MomentumSignal(
            date=now[:10], symbol=symbol,
            roc_20=0.0, sma_20=0.0, close=0.0, signal=False,
        )

    # Gate 3: Sentiment (reads from cache — never blocks)
    sentiment_sig = get_latest_sentiment()

    # Instrument-specific filters — computed from the same price_df already loaded above
    safe_haven_result: dict | None = None
    fxy_viability_result: dict | None = None
    if symbol == "GLD":
        safe_haven_result = check_safe_haven_confirmation(price_df)
    elif symbol == "FXY":
        fxy_viability_result = check_fxy_viability(price_df)

    all_open, reason = _build_reason(
        regime_sig, momentum_sig, sentiment_sig, kill_halted, symbol,
        safe_haven=safe_haven_result,
        fxy_viability=fxy_viability_result,
    )

    log.info(
        "%s | all_open=%s | %s",
        symbol, all_open, reason[:120],
    )

    return GateResult(
        timestamp=now,
        symbol=symbol,
        regime_signal=regime_sig,
        momentum_signal=momentum_sig,
        sentiment_signal=sentiment_sig,
        all_open=all_open,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Trade instruction generation
# ---------------------------------------------------------------------------
def build_trade_instruction(
    gate_result: GateResult,
    nav: float,
    win_rate: float = _DEFAULT_WIN_RATE,
    avg_win_ratio: float = _DEFAULT_AVG_WIN_RATIO,
) -> TradeInstruction:
    """
    Compute a TradeInstruction from an open GateResult.
    Raises ValueError if gate_result.all_open is False or quantity would be 0.
    """
    if not gate_result.all_open:
        raise ValueError(
            f"Cannot build TradeInstruction: gate is not open for {gate_result.symbol}. "
            f"Reason: {gate_result.reason}"
        )

    entry_price = gate_result.momentum_signal.close
    kelly_frac = compute_kelly(win_rate, avg_win_ratio)
    quantity, actual_pct = compute_position(nav, entry_price, kelly_frac)

    if quantity == 0:
        raise ValueError(
            f"{gate_result.symbol}: position quantity is 0 — "
            f"entry price ${entry_price:.2f} too high for {POSITION_CAP_PCT*100:.0f}% of NAV ${nav:,.0f}"
        )

    rationale = (
        f"QK={kelly_frac:.4f} (win_rate={win_rate}, ratio={avg_win_ratio}) | "
        f"alloc={actual_pct*100:.2f}% of NAV | "
        f"{gate_result.reason}"
    )

    return TradeInstruction(
        symbol=gate_result.symbol,
        direction="BUY",
        quantity=quantity,
        entry_price=entry_price,
        kelly_fraction=kelly_frac,
        allocation_pct=actual_pct,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------
def evaluate_all_symbols(
    symbols: list[str] = SYMBOLS,
    gpr_path: Path = _DEFAULT_GPR_PATH,
    nav: float | None = None,
) -> list[GateResult]:
    """
    Run check_all_gates for every symbol. Returns all GateResults.
    If nav is not supplied, fetches it from Alpaca.
    """
    if nav is None:
        from config import get_config          # noqa: PLC0415
        from alpaca.trading.client import TradingClient  # noqa: PLC0415
        cfg = get_config()
        client = TradingClient(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, paper=cfg.PAPER)
        nav = float(client.get_account().equity)
        log.info("NAV from Alpaca: $%.2f", nav)

    # Load regime once — it's macro-level, not per-symbol
    shared_regime = get_latest_regime(gpr_path)

    # Fetch all price data in one call to avoid redundant API hits
    prices = get_prices(symbols)

    results: list[GateResult] = []
    for sym in symbols:
        result = check_all_gates(sym, gpr_path, price_df=prices[sym], regime_sig=shared_regime)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Post-exit cooldown — persisted to data/last_exit_state.json
# ---------------------------------------------------------------------------
_EXIT_STATE_PATH = _HERE / "data" / "last_exit_state.json"
_EXPECTED_POSITIONS_PATH = _HERE / "data" / "expected_positions.json"
_COOLDOWN_TRADING_DAYS = 3  # block re-entry this many trading days after exit


def load_exit_state() -> dict[str, str | None]:
    """
    Load last exit timestamps from disk.
    Returns {symbol: ISO_timestamp_str | None} for all symbols in SYMBOLS.
    Missing file → all None (no cooldown active).
    """
    default: dict[str, str | None] = {sym: None for sym in SYMBOLS}
    if not _EXIT_STATE_PATH.exists():
        return default
    try:
        data = json.loads(_EXIT_STATE_PATH.read_text(encoding="utf-8"))
        return {sym: data.get(sym) for sym in SYMBOLS}
    except Exception as exc:
        log.warning("Could not load exit state (using empty): %s", exc)
        return default


def save_exit_state(state: dict[str, str | None]) -> None:
    """Write exit state atomically via os.replace()."""
    _EXIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _EXIT_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    # Retry on OSError — OneDrive can briefly lock the .tmp file during sync
    for delay in (0.2, 0.5, 1.0, 2.0):
        try:
            os.replace(tmp, _EXIT_STATE_PATH)
            return
        except OSError:
            time.sleep(delay)
    os.replace(tmp, _EXIT_STATE_PATH)  # final attempt; raises if still locked


def record_exit(symbol: str) -> None:
    """Record an exit event for symbol. Persisted atomically."""
    state = load_exit_state()
    state[symbol] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_exit_state(state)
    log.info("Cooldown: exit recorded for %s — blocked %d trading days", symbol, _COOLDOWN_TRADING_DAYS)


def is_in_cooldown(symbol: str, exit_state: dict[str, str | None] | None = None) -> bool:
    """
    Return True if symbol is within _COOLDOWN_TRADING_DAYS trading days of its
    last recorded exit. Pass exit_state if already loaded to avoid double I/O.
    """
    import pandas as pd  # noqa: PLC0415
    from pandas.tseries.offsets import BDay  # noqa: PLC0415

    if exit_state is None:
        exit_state = load_exit_state()
    last_exit_str = exit_state.get(symbol)
    if last_exit_str is None:
        return False
    last_exit = pd.Timestamp(last_exit_str).normalize()
    cooldown_end = last_exit + BDay(_COOLDOWN_TRADING_DAYS)
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    in_cd = now <= cooldown_end
    if in_cd:
        log.debug(
            "Cooldown active: %s | exit=%s cooldown_end=%s",
            symbol, last_exit.date(), cooldown_end.date(),
        )
    return in_cd


def load_expected_positions() -> set[str]:
    """Load the set of symbols we expect to currently be open in Alpaca."""
    if not _EXPECTED_POSITIONS_PATH.exists():
        return set()
    try:
        data = json.loads(_EXPECTED_POSITIONS_PATH.read_text(encoding="utf-8"))
        return set(data) if isinstance(data, list) else set()
    except Exception as exc:
        log.warning("Could not load expected positions (using empty set): %s", exc)
        return set()


def save_expected_positions(positions: set[str]) -> None:
    """Persist expected open positions atomically."""
    _EXPECTED_POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _EXPECTED_POSITIONS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(positions), indent=2), encoding="utf-8")
    # Retry on OSError — OneDrive can briefly lock the .tmp file during sync
    for delay in (0.2, 0.5, 1.0, 2.0):
        try:
            os.replace(tmp, _EXPECTED_POSITIONS_PATH)
            return
        except OSError:
            time.sleep(delay)
    os.replace(tmp, _EXPECTED_POSITIONS_PATH)  # final attempt; raises if still locked


# ---------------------------------------------------------------------------
# Entry point — evaluate all symbols and print gate status + any instructions
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import get_config
    from alpaca.trading.client import TradingClient

    cfg = get_config()
    client = TradingClient(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, paper=cfg.PAPER)
    nav = float(client.get_account().equity)

    results = evaluate_all_symbols(nav=nav)

    print(f"\n{'='*60}")
    print(f"SYNK STRATEGY EVALUATION  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"NAV: ${nav:,.2f}")
    print(f"{'='*60}")

    open_gates = [r for r in results if r.all_open]
    closed_gates = [r for r in results if not r.all_open]

    for result in results:
        status = "OPEN " if result.all_open else "CLOSED"
        print(f"\n  [{status}] {result.symbol}")
        print(f"    {result.reason}")

    if open_gates:
        print(f"\n{'='*60}")
        print("TRADE INSTRUCTIONS")
        print(f"{'='*60}")
        for result in open_gates:
            try:
                instr = build_trade_instruction(result, nav)
                print(f"\n  {instr.symbol}: {instr.direction} {instr.quantity} shares")
                print(f"    Entry:  ${instr.entry_price:.2f}")
                print(f"    Alloc:  {instr.allocation_pct*100:.2f}% of NAV")
                print(f"    Kelly:  {instr.kelly_fraction:.4f} (QK)")
                print(f"    Note:   ORDER NOT SUBMITTED — wiring pending")
            except ValueError as exc:
                print(f"\n  {result.symbol}: {exc}")
    else:
        print("\nNo gates open. No trade instructions generated.")
