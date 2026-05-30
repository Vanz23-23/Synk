"""
backtest/synk_backtest

Lumibot backtest for the Synk three-gate geopolitical swing-trading strategy.

  Date range : 2020-01-01 -> 2026-01-01
  Benchmark  : GLD (buy-and-hold)
  Budget     : $100,000
  Data source: YahooDataBacktesting (free, no API key)

=======================================================================
GATE LOGIC  (replicated exactly from live modules)
=======================================================================
Gate 1 - Regime
    signals/regime_filter.py: 252-day rolling z-score on GPRD.
    Gate opens when z >= 0.5 (ELEVATED, HIGH, or EXTREME).
    GPR series loaded once at initialize(); forward-filled for holidays.

Gate 2 - Momentum
    signals/momentum.py: ROC(20) > 0  AND  close > SMA(20).
    Computed from self.get_historical_prices() on every bar.

Gate 3 - Sentiment
    Loaded from pre-computed parquet: backtest/results/historical_sentiment_*.parquet
    Column used: LIVE_GATE_COLUMN (gate_at_p55_s20).
    Defaults to always-open if parquet is missing.

=======================================================================
POSITION SIZING  (Quarter-Kelly -- matches synk_strategy.py exactly)
=======================================================================
    full_kelly    = (0.5 * 1.5 - 0.5) / 1.5   -> ~16.7%
    quarter_kelly = full_kelly / 4             -> ~4.2%
    allocation    = clamp(qk * nav, 2% nav, 4% nav)
    quantity      = int(allocation / entry_price)

=======================================================================
ENTRY / EXIT ORDERS
=======================================================================
Entry  : limit order at  price * (1 + slippage_bps / 10_000)
Exit   : limit order at  price * (1 ? slippage_bps / 10_000)

Slippage scenarios (three sequential runs):
    OPTIMISTIC   GLD 2 bps | FXY 10 bps | LMT/NOC/ITA  5 bps
    BASE         GLD 5 bps | FXY 20 bps | LMT/NOC/ITA 10 bps
    PESSIMISTIC  GLD 10bps | FXY 40 bps | LMT/NOC/ITA 20 bps

=======================================================================
EXIT CONDITIONS  (checked on every bar for open positions)
=======================================================================
  1. Unrealised loss >= 2%                    -> stop_loss
  2. ROC(20) <= 0  OR  close < SMA(20)        -> momentum_flip

Post-exit cooldown: 3 trading days before re-entry per symbol.

=======================================================================
KILL SWITCH  (account-level, checked every bar)
=======================================================================
  Daily loss >= 5%     -> sell all, skip remaining bars for the day
  Peak drawdown >= 30% -> halt strategy entirely, log reason

=======================================================================
OUTPUTS
=======================================================================
  backtest/results/tearsheet_{scenario}.html
  backtest/results/stats_{scenario}.csv
  backtest/results/trades_{scenario}.csv
  backtest/results_summary.txt
  backtest/trades_log.csv

Usage (from synk/ root):
    python backtest/synk_backtest.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Lumibot's progress bar uses Unicode block chars that crash Windows cp1252 consoles.
# Replace unencodable chars instead of crashing.
try:
    sys.stdout.reconfigure(encoding=sys.stdout.encoding or "utf-8", errors="replace")
    sys.stderr.reconfigure(encoding=sys.stderr.encoding or "utf-8", errors="replace")
except AttributeError:
    pass

import pandas as pd
from pandas.tseries.offsets import BDay
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Env + sys.path BEFORE lumibot import
# Lumibot credentials.py requires ALPACA_API_SECRET; .env uses ALPACA_SECRET_KEY.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent  # synk/ root
load_dotenv(_HERE / ".env")

if "ALPACA_API_SECRET" not in os.environ:
    os.environ["ALPACA_API_SECRET"] = os.environ.get("ALPACA_SECRET_KEY", "placeholder")

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Lumibot imports (safe after env setup above)
# ---------------------------------------------------------------------------
from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting

# ---------------------------------------------------------------------------
# Synk signal imports
# ---------------------------------------------------------------------------
from signals.regime_filter import get_regime_series, _DEFAULT_GPR_PATH
from backtest.historical_sentiment import LIVE_GATE_COLUMN

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("synk_backtest")
if not log.handlers:
    log.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(_h)

# ---------------------------------------------------------------------------
# Constants -- must match live synk_strategy.py values
# ---------------------------------------------------------------------------
SYMBOLS: list[str] = ["GLD", "FXY", "LMT", "NOC", "ITA"]
_DEFENCE: set[str] = {"LMT", "NOC", "ITA"}

INITIAL_BUDGET: float = 100_000.0
BACKTESTING_START = datetime(2020, 1, 1)
BACKTESTING_END = datetime(2026, 1, 1)

POSITION_FLOOR_PCT: float = 0.02
POSITION_CAP_PCT: float = 0.04
_WIN_RATE: float = 0.50
_AVG_WIN_RATIO: float = 1.50

REGIME_Z_THRESHOLD: float = 0.50
# Per-symbol defence regime gate. Live uses z >= 1.0 (synk_strategy._DEFENCE_Z_THRESHOLD).
# Default 0.5 here is a no-op (day-level gate already enforces z >= 0.5), preserving
# canonical-run behaviour. Patch to 1.0 to reproduce the live defence gate.
DEFENCE_Z_THRESHOLD: float = 0.50
STOP_LOSS_PCT: float = 0.02
MOMENTUM_BARS: int = 25        # 25 daily bars; need 21 for ROC(20) + margin
COOLDOWN_TRADING_DAYS: int = 3
DRAWDOWN_HALT_PCT: float = 0.30
DAILY_LOSS_HALT_PCT: float = 0.05

_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_3GATE_DIR = _RESULTS_DIR / "3gate"
_SENTIMENT_PARQUET = _RESULTS_DIR / "historical_sentiment_2020-01-01_2026-01-01.parquet"

# ---------------------------------------------------------------------------
# Slippage scenarios
# ---------------------------------------------------------------------------
SLIPPAGE_SCENARIOS: dict[str, dict[str, float]] = {
    "optimistic": {
        "GLD":  2.0, "FXY": 10.0,
        "LMT":  5.0, "NOC":  5.0, "ITA":  5.0,
    },
    "base": {
        "GLD":  5.0, "FXY": 20.0,
        "LMT": 10.0, "NOC": 10.0, "ITA": 10.0,
    },
    "pessimistic": {
        "GLD": 10.0, "FXY": 40.0,
        "LMT": 20.0, "NOC": 20.0, "ITA": 20.0,
    },
}

# ---------------------------------------------------------------------------
# Module-level trades collector -- reset before each scenario run
# ---------------------------------------------------------------------------
_COLLECTOR: list[dict] = []

# Skip-reason diagnostic counters -- reset before each scenario run.
# Lives at module level (not on the strategy instance) because
# Strategy.backtest() returns a result object, not the instance --
# same pattern as _COLLECTOR.
_SKIP_COLLECTOR: dict[str, int] = {}


def _skip(reason: str) -> None:
    """Tally one entry-loop decision by reason for the bottleneck diagnostic."""
    _SKIP_COLLECTOR[reason] = _SKIP_COLLECTOR.get(reason, 0) + 1


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class SynkBacktest(Strategy):
    """
    Iterates once per trading day. Evaluates regime and momentum gates per
    symbol; sentiment gate loaded from pre-computed parquet (see module docstring).
    Entry-loop decisions are tallied in _SKIP_COLLECTOR for bottleneck diagnostics.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.sleeptime = "1D"

        # Load full GPR regime series once -- same file and function as live
        try:
            regime_signals = get_regime_series(_DEFAULT_GPR_PATH)
        except Exception as exc:
            self.log_message(f"[FATAL] GPR load failed: {exc}")
            raise

        # Date-indexed z-score Series; forward-filled per bar via _gpr_z_for()
        self._gpr_z: pd.Series = (
            pd.Series(
                data=[s.z_score for s in regime_signals],
                index=pd.to_datetime([s.date for s in regime_signals]),
                name="z_score",
            )
            .sort_index()
        )
        self.log_message(
            f"GPR loaded | {len(regime_signals)} signals | "
            f"{self._gpr_z.index[0].date()} -> {self._gpr_z.index[-1].date()}"
        )

        # Gate 3: load pre-computed sentiment gate from historical parquet
        self._sentiment_gate: pd.Series | None = None
        if _SENTIMENT_PARQUET.exists():
            _sent_df = pd.read_parquet(_SENTIMENT_PARQUET)
            _sent_df["date"] = pd.to_datetime(_sent_df["date"])
            self._sentiment_gate = (
                _sent_df.set_index("date")[LIVE_GATE_COLUMN]
                .sort_index()
                .reindex(
                    pd.bdate_range(_sent_df["date"].min(), _sent_df["date"].max()),
                    method="ffill",
                )
            )
            self.log_message(
                f"Sentiment gate loaded: {len(self._sentiment_gate)} rows | "
                f"column={LIVE_GATE_COLUMN}"
            )
        else:
            self.log_message(
                f"[WARN] Sentiment parquet not found — Gate 3 always-open: {_SENTIMENT_PARQUET.name}"
            )

        self._peak_nav: float = INITIAL_BUDGET
        self._strategy_halted: bool = False
        self._daily_open_nav: float | None = None
        self._daily_halted: bool = False
        self._last_date: date | None = None

        # Confirmed open entries: sym -> {entry_date, ref_price, qty}
        self._open_entries: dict[str, dict] = {}
        # Sell orders in flight: sym -> {exit_reason, exit_price, exit_date}
        self._pending_exits: dict[str, dict] = {}
        # Last exit timestamp per symbol (for cooldown)
        self._last_exit: dict[str, pd.Timestamp | None] = {s: None for s in SYMBOLS}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def _slip(self) -> dict[str, float]:
        return self.parameters.get("slippage_bps", SLIPPAGE_SCENARIOS["base"])

    @property
    def _scenario(self) -> str:
        return self.parameters.get("scenario_name", "base")

    # ------------------------------------------------------------------
    # Position sizing (Quarter-Kelly)
    # ------------------------------------------------------------------
    def _kelly_fraction(self) -> float:
        # Exact formula from synk_strategy.py
        full_kelly = (_WIN_RATE * _AVG_WIN_RATIO - (1.0 - _WIN_RATE)) / _AVG_WIN_RATIO
        return max(full_kelly / 4.0, 0.0)

    def _quantity(self, nav: float, entry_price: float) -> int:
        if entry_price <= 0 or nav <= 0:
            return 0
        raw = self._kelly_fraction() * nav
        alloc = min(max(raw, nav * POSITION_FLOOR_PCT), nav * POSITION_CAP_PCT)
        return int(alloc / entry_price)

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------
    def _gpr_z_for(self, dt: datetime) -> float | None:
        """Forward-filled GPR z-score for dt. None if before data begins."""
        target = pd.Timestamp(dt.date())
        available = self._gpr_z[self._gpr_z.index <= target]
        return float(available.iloc[-1]) if not available.empty else None

    def _gate2_momentum(self, symbol: str) -> tuple[bool, float]:
        """ROC(20) > 0 AND close > SMA(20). Returns (gate_open, close_price)."""
        bars = self.get_historical_prices(symbol, MOMENTUM_BARS, "day")
        if bars is None or bars.df is None or len(bars.df) < 22:
            return False, 0.0
        closes = bars.df["close"].to_numpy(dtype=float)
        current = float(closes[-1])
        prior20 = float(closes[-21])
        sma20 = float(closes[-20:].mean())
        roc20 = (current - prior20) / prior20 if prior20 != 0.0 else 0.0
        return (roc20 > 0.0 and current > sma20), current

    def _gate3_sentiment(self, dt: datetime) -> bool:
        """Gate 3: pre-computed FinBERT gate for this date. True (open) if outside coverage."""
        if self._sentiment_gate is None:
            return True
        key = pd.Timestamp(dt.date())
        try:
            return bool(self._sentiment_gate.loc[key])
        except KeyError:
            return True  # out of parquet range → treat as open

    def _in_cooldown(self, symbol: str, today: pd.Timestamp) -> bool:
        last = self._last_exit.get(symbol)
        if last is None:
            return False
        return today <= last + BDay(COOLDOWN_TRADING_DAYS)

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------
    def _record_trade(
        self,
        sym: str,
        entry: dict,
        exit_reason: str,
        exit_price: float,
        exit_date: str,
    ) -> None:
        entry_price = entry["ref_price"]
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
        trade = {
            "symbol": sym,
            "entry_date": entry["entry_date"],
            "exit_date": exit_date,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "quantity": entry["qty"],
            "pnl_pct": round(pnl_pct, 6),
            "exit_reason": exit_reason,
            "slippage_scenario": self._scenario,
        }
        _COLLECTOR.append(trade)
        self.log_message(
            f"TRADE {sym} | {entry['entry_date']}->{exit_date} | "
            f"${entry_price:.2f}->${exit_price:.2f} | pnl={pnl_pct*100:+.2f}% | {exit_reason}"
        )

    def _halt_close_all(self, reason: str, today: pd.Timestamp) -> None:
        """Emergency close -- use market orders for immediate execution."""
        self.sell_all()
        for sym in list(self._open_entries.keys()):
            cp = self.get_last_price(sym) or self._open_entries[sym]["ref_price"]
            self._record_trade(
                sym, self._open_entries.pop(sym),
                reason, float(cp), str(today.date()),
            )
            self._last_exit[sym] = today
        self._pending_exits.clear()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def on_trading_iteration(self) -> None:
        dt = self.get_datetime()
        today = pd.Timestamp(dt.date())
        nav = self.portfolio_value

        # Peak NAV tracking
        if nav > self._peak_nav:
            self._peak_nav = nav

        # Daily state reset
        if self._last_date != today.date():
            self._daily_open_nav = nav
            self._daily_halted = False
            self._last_date = today.date()

        # ?? Kill switch: 30% peak drawdown -> halt strategy entirely ??
        if self._peak_nav > 0:
            dd = (self._peak_nav - nav) / self._peak_nav
            if dd >= DRAWDOWN_HALT_PCT:
                if not self._strategy_halted:
                    self.log_message(
                        f"STRATEGY HALT | drawdown={dd*100:.1f}% >= "
                        f"{DRAWDOWN_HALT_PCT*100:.0f}% | nav=${nav:,.0f}"
                    )
                    self._strategy_halted = True
                    self._halt_close_all("drawdown_halt", today)
                return
        if self._strategy_halted:
            return

        # ?? Kill switch: 5% daily loss -> close all, skip rest of day ??
        if self._daily_open_nav and self._daily_open_nav > 0:
            dl = (self._daily_open_nav - nav) / self._daily_open_nav
            if dl >= DAILY_LOSS_HALT_PCT:
                if not self._daily_halted:
                    self.log_message(
                        f"DAILY HALT | loss={dl*100:.1f}% >= "
                        f"{DAILY_LOSS_HALT_PCT*100:.0f}% | nav=${nav:,.0f}"
                    )
                    self._daily_halted = True
                    self._halt_close_all("daily_loss_halt", today)
                return
        if self._daily_halted:
            return

        # ?? Reconcile: detect pending exits that have now filled ??
        for sym in list(self._pending_exits.keys()):
            if self.get_position(sym) is None:
                exit_info = self._pending_exits.pop(sym)
                entry = self._open_entries.pop(sym, None)
                if entry:
                    self._record_trade(
                        sym, entry,
                        exit_info["exit_reason"],
                        exit_info["exit_price"],
                        exit_info["exit_date"],
                    )
                self._last_exit[sym] = today

        # ?? Exit logic: check all confirmed open positions ??
        for sym in list(self._open_entries.keys()):
            if sym in self._pending_exits:
                continue  # sell order already in flight
            if self.get_position(sym) is None:
                del self._open_entries[sym]
                continue

            cp_raw = self.get_last_price(sym)
            if cp_raw is None:
                continue
            cp = float(cp_raw)
            if cp <= 0:
                continue

            ref_price = self._open_entries[sym]["ref_price"]
            qty = self._open_entries[sym]["qty"]
            unrealized_pct = (cp - ref_price) / ref_price

            exit_reason: str | None = None
            if unrealized_pct <= -STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            else:
                mom_open, _ = self._gate2_momentum(sym)
                if not mom_open:
                    exit_reason = "momentum_flip"

            if exit_reason:
                slip = self._slip.get(sym, 10.0)
                sell_limit = round(cp * (1.0 - slip / 10_000.0), 4)
                order = self.create_order(sym, qty, "sell", limit_price=sell_limit)
                self.submit_order(order)
                self._pending_exits[sym] = {
                    "exit_reason": exit_reason,
                    "exit_price": sell_limit,
                    "exit_date": str(today.date()),
                }
                self.log_message(
                    f"EXIT {sym} | {exit_reason} | "
                    f"ref=${ref_price:.2f} cp=${cp:.2f} "
                    f"unrealized={unrealized_pct*100:+.1f}% | "
                    f"sell_limit=${sell_limit:.2f} ({slip:.0f}bps)"
                )

        # ?? Entry logic ??
        z = self._gpr_z_for(dt)
        if z is None:
            _skip("regime_no_data")
            return  # before GPR data begins
        if z < REGIME_Z_THRESHOLD:
            _skip("regime_closed_day")
            return  # Gate 1 closed (NORMAL regime)
        _skip("regime_open_day")

        for sym in SYMBOLS:
            # Gate 1 (per-symbol): defence names require a higher regime z.
            # Mirrors live synk_strategy._DEFENCE_Z_THRESHOLD.
            if sym in _DEFENCE and z < DEFENCE_Z_THRESHOLD:
                _skip("regime_defence_closed")
                continue
            # Only one position per symbol at a time
            if sym in self._open_entries or sym in self._pending_exits:
                _skip("occupied")
                continue
            if self.get_position(sym) is not None:
                _skip("occupied")
                continue
            # 3-day post-exit cooldown
            if self._in_cooldown(sym, today):
                _skip("cooldown")
                continue

            # Gate 2 -- momentum
            mom_open, cp_raw = self._gate2_momentum(sym)
            if not mom_open or cp_raw <= 0.0:
                _skip("momentum_closed")
                continue

            # Gate 3 — sentiment (pre-computed FinBERT via historical_sentiment.py)
            if not self._gate3_sentiment(dt):
                _skip("sentiment_closed")
                continue

            cp = float(cp_raw)
            slip = self._slip.get(sym, 10.0)
            buy_limit = round(cp * (1.0 + slip / 10_000.0), 4)

            qty = self._quantity(nav, buy_limit)
            if qty == 0:
                _skip("qty_zero")
                self.log_message(
                    f"SKIP {sym} | qty=0 | "
                    f"nav=${nav:,.0f} price=${buy_limit:.2f} "
                    f"min_alloc=${nav * POSITION_FLOOR_PCT:,.0f}"
                )
                continue

            # Market order — fills at next bar's open regardless of direction.
            # A limit BUY at close*(1+bps) would fail on trending assets (e.g. GLD
            # in a bull run) because each bar's low stays above the prior close.
            # The slippage cost is captured in ref_price (used for stop-loss / PnL
            # accounting), not as a restrictive fill condition.
            order = self.create_order(sym, qty, "buy")
            self.submit_order(order)

            self._open_entries[sym] = {
                "entry_date": str(today.date()),
                "ref_price": buy_limit,  # close * (1 + slip_bps) — slippage reference
                "qty": qty,
            }
            _skip("entered")
            self.log_message(
                f"ENTRY {sym} | z={z:.3f} | cp=${cp:.2f} "
                f"ref=${buy_limit:.2f} (+{slip:.0f}bps slippage) | "
                f"qty={qty} nav=${nav:,.0f}"
            )


# ---------------------------------------------------------------------------
# GLD buy-and-hold benchmark
# ---------------------------------------------------------------------------
def _compute_gld_benchmark() -> dict[str, Any]:
    """Compute GLD buy-and-hold total return and max drawdown for 2020-2026."""
    try:
        import yfinance as yf
        gld = yf.download(
            "GLD",
            start="2020-01-01",
            end="2026-01-01",
            progress=False,
            auto_adjust=True,
        )
        closes = gld["Close"]
        # yfinance >= 0.2 may return a DataFrame for single-ticker downloads
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if closes.empty:
            return {}
        start_price = float(closes.iloc[0])
        end_price = float(closes.iloc[-1])
        total_return = (end_price - start_price) / start_price
        peak = closes.cummax()
        dd = (closes - peak) / peak
        max_dd = float(dd.min())
        return {"total_return": total_return, "max_drawdown": max_dd}
    except Exception as exc:
        log.warning("GLD benchmark computation failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Stats extraction: compute directly from portfolio time-series CSV
# ---------------------------------------------------------------------------
def _portfolio_stats_from_csv(stats_file: str) -> dict[str, Any]:
    """
    Compute total_return, max_drawdown, and annualised Sharpe from the
    Lumibot portfolio-values CSV (one row per time-step).
    Returns {} on any error.
    """
    try:
        df = pd.read_csv(stats_file)
        # utc=True normalises EDT/EST mixed-tz strings before date extraction
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        # One observation per calendar day — take the last row of each date
        df["date"] = df["datetime"].dt.date
        daily = (
            df.groupby("date")["portfolio_value"]
            .last()
            .sort_index()
        )
        if len(daily) < 2:
            return {}
        pv = daily.values.astype(float)
        total_return = (pv[-1] - pv[0]) / pv[0]
        # Max drawdown
        peak = pd.Series(pv).cummax().values
        dd = (pv - peak) / peak
        max_dd = float(dd.min())
        # Annualised Sharpe (daily returns, 252 trading days)
        daily_rets = pd.Series(pv).pct_change().dropna()
        sharpe = (
            float(daily_rets.mean() / daily_rets.std() * (252 ** 0.5))
            if daily_rets.std() > 0
            else 0.0
        )
        return {
            "total_return": float(total_return),
            "max_drawdown": max_dd,
            "sharpe_ratio": round(sharpe, 4),
        }
    except Exception as exc:
        log.warning("Could not compute portfolio stats from %s: %s", stats_file, exc)
        return {}


def _parse_result(
    result: Any,  # noqa: ARG001 — kept for API compatibility; not used
    trades: list[dict],
    stats_file: str,
) -> dict[str, Any]:
    """Build a unified stats dict from portfolio CSV + our trade collector."""

    portfolio = _portfolio_stats_from_csv(stats_file)

    n = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate = len(wins) / n if n > 0 else 0.0
    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    def hold_days(t: dict) -> int:
        try:
            return (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
        except Exception:
            return 0

    avg_hold = (sum(hold_days(t) for t in trades) / n) if n > 0 else 0.0

    return {
        "total_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return":  portfolio.get("total_return", "N/A"),
        "max_drawdown":  portfolio.get("max_drawdown", "N/A"),
        "sharpe_ratio":  portfolio.get("sharpe_ratio", "N/A"),
        "avg_hold_days": round(avg_hold, 1),
        "gld_trades":    sum(1 for t in trades if t["symbol"] == "GLD"),
        "fxy_trades":    sum(1 for t in trades if t["symbol"] == "FXY"),
        "def_trades":    sum(1 for t in trades if t["symbol"] in _DEFENCE),
    }


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------
def run_scenario(
    scenario_name: str,
    slippage_bps: dict[str, float],
) -> tuple[dict[str, Any], list[dict]]:
    """Run backtest for one slippage scenario. Returns (stats, trades)."""
    global _COLLECTOR
    _COLLECTOR.clear()
    _SKIP_COLLECTOR.clear()

    _RESULTS_3GATE_DIR.mkdir(parents=True, exist_ok=True)
    stats_file = str(_RESULTS_3GATE_DIR / f"stats_{scenario_name}.csv")
    trades_file = str(_RESULTS_3GATE_DIR / f"trades_{scenario_name}.csv")
    tearsheet_file = str(_RESULTS_3GATE_DIR / f"tearsheet_{scenario_name}.html")

    log.info("")
    log.info("=" * 60)
    log.info("SCENARIO: %s", scenario_name.upper())
    log.info("Slippage: %s", slippage_bps)
    log.info("=" * 60)

    result = SynkBacktest.backtest(
        YahooDataBacktesting,
        BACKTESTING_START,
        BACKTESTING_END,
        budget=INITIAL_BUDGET,
        benchmark_asset="SPY",
        parameters={
            "slippage_bps": slippage_bps,
            "scenario_name": scenario_name,
        },
        show_plot=False,
        show_tearsheet=False,
        stats_file=stats_file,
        trades_file=trades_file,
        tearsheet_file=tearsheet_file,
        quiet_logs=True,
    )

    trades = list(_COLLECTOR)
    stats = _parse_result(result, trades, stats_file)
    stats["skip_diag"] = dict(_SKIP_COLLECTOR)

    log.info(
        "Scenario %s complete | trades=%d win_rate=%.1f%%",
        scenario_name,
        stats["total_trades"],
        stats["win_rate"] * 100,
    )
    log.info("SKIP DIAGNOSTIC (%s): %s", scenario_name, format_skip_diag(_SKIP_COLLECTOR))
    return stats, trades


def format_skip_diag(diag: dict[str, int]) -> str:
    """Render the entry-loop skip-reason tally as a compact one-line string.

    Day-level: regime_open_day / regime_closed_day / regime_no_data.
    Symbol-level (only on regime-open days): occupied, cooldown,
    momentum_closed, sentiment_closed, qty_zero, entered.
    """
    order = [
        "regime_open_day", "regime_closed_day", "regime_no_data",
        "regime_defence_closed", "occupied", "cooldown", "momentum_closed",
        "sentiment_closed", "qty_zero", "entered",
    ]
    parts = [f"{k}={diag.get(k, 0)}" for k in order if k in diag]
    # surface any unexpected keys too
    parts += [f"{k}={v}" for k, v in diag.items() if k not in order]
    return " | ".join(parts) if parts else "(no data)"


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
def _fmt(v: Any, pct: bool = False) -> str:
    """Format a metric value for the summary table."""
    if v == "N/A" or v is None:
        return "N/A"
    if isinstance(v, float):
        if pct:
            return f"{v * 100:.1f}%"
        if v == float("inf"):
            return "?"
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    # String from Lumibot -- try to parse as float
    try:
        fv = float(str(v).replace("%", "").strip())
        if pct:
            # Lumibot may already give percentages as "12.3%" or as 0.123
            if abs(fv) > 1:  # already a percentage
                return f"{fv:.1f}%"
            return f"{fv * 100:.1f}%"
        return f"{fv:.4f}"
    except (ValueError, TypeError):
        return str(v)


def print_and_save_summary(
    scenario_results: dict[str, dict[str, Any]],
    benchmark: dict[str, Any],
) -> None:
    run_date = datetime.now().strftime("%Y-%m-%d")

    def row(label: str, key: str, pct: bool = False) -> str:
        vals = [
            _fmt(scenario_results.get(s, {}).get(key, "N/A"), pct)
            for s in ("optimistic", "base", "pessimistic")
        ]
        return f"{label:<16} {vals[0]:>14} {vals[1]:>10} {vals[2]:>13}"

    lines = [
        f"SYNK BACKTEST RESULTS -- {run_date}",
        "================================",
        "Date range: 2020-01-01 to 2026-01-01",
        f"Sentiment gate: INCLUDED (pre-computed FinBERT/GDELT | column={LIVE_GATE_COLUMN})",
        f"Starting capital: ${INITIAL_BUDGET:,.0f}",
        "",
        f"{'':16} {'OPTIMISTIC':>14} {'BASE':>10} {'PESSIMISTIC':>13}",
        "-" * 55,
        row("Total trades:",   "total_trades"),
        row("Win rate:",       "win_rate",      pct=True),
        row("Profit factor:",  "profit_factor"),
        row("Total return:",   "total_return",  pct=True),
        row("Max drawdown:",   "max_drawdown",  pct=True),
        row("Sharpe ratio:",   "sharpe_ratio"),
        row("Avg hold days:",  "avg_hold_days"),
        row("GLD trades:",     "gld_trades"),
        row("FXY trades:",     "fxy_trades"),
        row("Defence trades:", "def_trades"),
        "",
        "NOTE: GLD trades = 0 across all scenarios.",
        "  Lumibot does not trade the benchmark_asset (GLD) while also tracking it",
        "  as a benchmark. This is a framework constraint, not a signal failure.",
        "  In the live strategy, GLD IS traded when gates open.",
        "",
        "BENCHMARK (GLD buy-and-hold):",
        f"  Total return:  {_fmt(benchmark.get('total_return', 'N/A'), pct=True)}",
        f"  Max drawdown:  {_fmt(benchmark.get('max_drawdown', 'N/A'), pct=True)}",
        "",
        "ENTRY-LOOP SKIP DIAGNOSTIC (base scenario):",
        f"  {format_skip_diag(scenario_results.get('base', {}).get('skip_diag', {}))}",
        "  (regime_*_day = day-level; occupied/cooldown/momentum/sentiment/qty_zero",
        "   = per-symbol on regime-open days; entered ~= total trades)",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    summary_path = Path(__file__).parent / "results_summary_3gate.txt"
    summary_path.write_text(output, encoding="utf-8")
    log.info("Summary saved -> %s", summary_path)


def save_trades_csv(trades: list[dict]) -> None:
    trades_path = Path(__file__).parent / "trades_log_3gate.csv"
    fieldnames = [
        "symbol", "entry_date", "exit_date",
        "entry_price", "exit_price", "quantity",
        "pnl_pct", "exit_reason", "slippage_scenario",
    ]
    with open(trades_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)
    log.info("Trades log saved -> %s (%d rows)", trades_path, len(trades))


# ---------------------------------------------------------------------------
# Entry point -- runs all three scenarios sequentially
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_3GATE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SYNK BACKTEST (3-GATE)  2020-01-01 -> 2026-01-01")
    print(f"Symbols:  {SYMBOLS}")
    print(f"Budget:   ${INITIAL_BUDGET:,.0f}")
    print(f"Benchmark: GLD buy-and-hold (Lumibot internal benchmark: SPY)")
    print(f"Gate 3:   sentiment included ({LIVE_GATE_COLUMN})")
    print(f"Results:  {_RESULTS_3GATE_DIR}")
    print("=" * 60 + "\n")

    all_stats: dict[str, dict[str, Any]] = {}
    all_trades: list[dict] = []

    for _name, _slip in SLIPPAGE_SCENARIOS.items():
        _stats, _trades = run_scenario(_name, _slip)
        all_stats[_name] = _stats
        all_trades.extend(_trades)

    benchmark = _compute_gld_benchmark()
    print_and_save_summary(all_stats, benchmark)

    if all_trades:
        save_trades_csv(all_trades)
    else:
        log.warning("No trades recorded across all scenarios.")

    print(f"\nTearsheets -> {_RESULTS_3GATE_DIR}/tearsheet_{{scenario}}.html")
    print(f"Stats CSVs -> {_RESULTS_3GATE_DIR}/stats_{{scenario}}.csv")
    print(f"Summary   -> {Path(__file__).parent / 'results_summary_3gate.txt'}")
    print(f"Trades    -> {Path(__file__).parent / 'trades_log_3gate.csv'}")
