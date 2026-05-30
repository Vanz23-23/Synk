"""
backtest/synk_backtest_haven

Haven-leg isolation: GLD + FXY only (defence excluded).

Purpose: test whether the core geopolitical haven thesis holds
independently of the defence secular re-rating (2022-2025).
Same gates, sizing, kill-switch, and slippage scenarios as the
full backtest — only SYMBOLS changes.

Usage (from synk/ root):
    python backtest/synk_backtest_haven.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding=sys.stdout.encoding or "utf-8", errors="replace")
    sys.stderr.reconfigure(encoding=sys.stderr.encoding or "utf-8", errors="replace")
except AttributeError:
    pass

import pandas as pd
from pandas.tseries.offsets import BDay
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent.parent
load_dotenv(_HERE / ".env")

if "ALPACA_API_SECRET" not in os.environ:
    os.environ["ALPACA_API_SECRET"] = os.environ.get("ALPACA_SECRET_KEY", "placeholder")

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting
from signals.regime_filter import get_regime_series, _DEFAULT_GPR_PATH
from backtest.historical_sentiment import LIVE_GATE_COLUMN

log = logging.getLogger("synk_backtest_haven")
if not log.handlers:
    log.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(_h)

# Haven-only: defence excluded
SYMBOLS: list[str] = ["GLD", "FXY"]

INITIAL_BUDGET: float = 100_000.0
BACKTESTING_START = datetime(2020, 1, 1)
BACKTESTING_END = datetime(2026, 1, 1)

POSITION_FLOOR_PCT: float = 0.02
POSITION_CAP_PCT: float = 0.04
_WIN_RATE: float = 0.50
_AVG_WIN_RATIO: float = 1.50

REGIME_Z_THRESHOLD: float = 0.50
STOP_LOSS_PCT: float = 0.02
MOMENTUM_BARS: int = 25
COOLDOWN_TRADING_DAYS: int = 3
DRAWDOWN_HALT_PCT: float = 0.30
DAILY_LOSS_HALT_PCT: float = 0.05

_RESULTS_DIR = Path(__file__).parent / "results" / "haven"
_RESULTS_3GATE_DIR = _RESULTS_DIR / "3gate"
_SENTIMENT_PARQUET = Path(__file__).parent / "results" / "historical_sentiment_2020-01-01_2026-01-01.parquet"

SLIPPAGE_SCENARIOS: dict[str, dict[str, float]] = {
    "optimistic": {"GLD":  2.0, "FXY": 10.0},
    "base":       {"GLD":  5.0, "FXY": 20.0},
    "pessimistic":{"GLD": 10.0, "FXY": 40.0},
}

_COLLECTOR: list[dict] = []


class SynkHavenBacktest(Strategy):

    def initialize(self) -> None:
        self.sleeptime = "1D"
        try:
            regime_signals = get_regime_series(_DEFAULT_GPR_PATH)
        except Exception as exc:
            self.log_message(f"[FATAL] GPR load failed: {exc}")
            raise

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
            self.log_message(f"Sentiment gate loaded: {len(self._sentiment_gate)} rows | column={LIVE_GATE_COLUMN}")
        else:
            self.log_message(f"[WARN] Sentiment parquet not found — Gate 3 always-open: {_SENTIMENT_PARQUET.name}")

        self._peak_nav: float = INITIAL_BUDGET
        self._strategy_halted: bool = False
        self._daily_open_nav: float | None = None
        self._daily_halted: bool = False
        self._last_date = None
        self._open_entries: dict[str, dict] = {}
        self._pending_exits: dict[str, dict] = {}
        self._last_exit: dict[str, pd.Timestamp | None] = {s: None for s in SYMBOLS}

    @property
    def _slip(self) -> dict[str, float]:
        return self.parameters.get("slippage_bps", SLIPPAGE_SCENARIOS["base"])

    @property
    def _scenario(self) -> str:
        return self.parameters.get("scenario_name", "base")

    def _kelly_fraction(self) -> float:
        full_kelly = (_WIN_RATE * _AVG_WIN_RATIO - (1.0 - _WIN_RATE)) / _AVG_WIN_RATIO
        return max(full_kelly / 4.0, 0.0)

    def _quantity(self, nav: float, entry_price: float) -> int:
        if entry_price <= 0 or nav <= 0:
            return 0
        raw = self._kelly_fraction() * nav
        alloc = min(max(raw, nav * POSITION_FLOOR_PCT), nav * POSITION_CAP_PCT)
        return int(alloc / entry_price)

    def _gpr_z_for(self, dt: datetime) -> float | None:
        target = pd.Timestamp(dt.date())
        available = self._gpr_z[self._gpr_z.index <= target]
        return float(available.iloc[-1]) if not available.empty else None

    def _gate2_momentum(self, symbol: str) -> tuple[bool, float]:
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
            return True

    def _in_cooldown(self, symbol: str, today: pd.Timestamp) -> bool:
        last = self._last_exit.get(symbol)
        if last is None:
            return False
        return today <= last + BDay(COOLDOWN_TRADING_DAYS)

    def _record_trade(self, sym: str, entry: dict, exit_reason: str, exit_price: float, exit_date: str) -> None:
        entry_price = entry["ref_price"]
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
        _COLLECTOR.append({
            "symbol": sym,
            "entry_date": entry["entry_date"],
            "exit_date": exit_date,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "quantity": entry["qty"],
            "pnl_pct": round(pnl_pct, 6),
            "exit_reason": exit_reason,
            "slippage_scenario": self._scenario,
        })

    def _halt_close_all(self, reason: str, today: pd.Timestamp) -> None:
        self.sell_all()
        for sym in list(self._open_entries.keys()):
            cp = self.get_last_price(sym) or self._open_entries[sym]["ref_price"]
            self._record_trade(sym, self._open_entries.pop(sym), reason, float(cp), str(today.date()))
            self._last_exit[sym] = today
        self._pending_exits.clear()

    def on_trading_iteration(self) -> None:
        dt = self.get_datetime()
        today = pd.Timestamp(dt.date())
        nav = self.portfolio_value

        if nav > self._peak_nav:
            self._peak_nav = nav

        if self._last_date != today.date():
            self._daily_open_nav = nav
            self._daily_halted = False
            self._last_date = today.date()

        if self._peak_nav > 0:
            dd = (self._peak_nav - nav) / self._peak_nav
            if dd >= DRAWDOWN_HALT_PCT:
                if not self._strategy_halted:
                    self._strategy_halted = True
                    self._halt_close_all("drawdown_halt", today)
                return
        if self._strategy_halted:
            return

        if self._daily_open_nav and self._daily_open_nav > 0:
            dl = (self._daily_open_nav - nav) / self._daily_open_nav
            if dl >= DAILY_LOSS_HALT_PCT:
                if not self._daily_halted:
                    self._daily_halted = True
                    self._halt_close_all("daily_loss_halt", today)
                return
        if self._daily_halted:
            return

        for sym in list(self._pending_exits.keys()):
            if self.get_position(sym) is None:
                exit_info = self._pending_exits.pop(sym)
                entry = self._open_entries.pop(sym, None)
                if entry:
                    self._record_trade(sym, entry, exit_info["exit_reason"], exit_info["exit_price"], exit_info["exit_date"])
                self._last_exit[sym] = today

        for sym in list(self._open_entries.keys()):
            if sym in self._pending_exits:
                continue
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

        z = self._gpr_z_for(dt)
        if z is None:
            return
        if z < REGIME_Z_THRESHOLD:
            return

        for sym in SYMBOLS:
            if sym in self._open_entries or sym in self._pending_exits:
                continue
            if self.get_position(sym) is not None:
                continue
            if self._in_cooldown(sym, today):
                continue

            mom_open, cp_raw = self._gate2_momentum(sym)
            if not mom_open or cp_raw <= 0.0:
                continue

            # Gate 3 — sentiment (pre-computed FinBERT via historical_sentiment.py)
            if not self._gate3_sentiment(dt):
                continue

            cp = float(cp_raw)
            slip = self._slip.get(sym, 10.0)
            buy_limit = round(cp * (1.0 + slip / 10_000.0), 4)

            qty = self._quantity(nav, buy_limit)
            if qty == 0:
                continue

            order = self.create_order(sym, qty, "buy")
            self.submit_order(order)
            self._open_entries[sym] = {
                "entry_date": str(today.date()),
                "ref_price": buy_limit,
                "qty": qty,
            }
            self.log_message(f"ENTRY {sym} | z={z:.3f} | cp=${cp:.2f} ref=${buy_limit:.2f} | qty={qty}")


def _compute_gld_benchmark() -> dict[str, Any]:
    try:
        import yfinance as yf
        gld = yf.download("GLD", start="2020-01-01", end="2026-01-01", progress=False, auto_adjust=True)
        closes = gld["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if closes.empty:
            return {}
        total_return = (float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0])
        peak = closes.cummax()
        max_dd = float(((closes - peak) / peak).min())
        return {"total_return": total_return, "max_drawdown": max_dd}
    except Exception as exc:
        log.warning("GLD benchmark failed: %s", exc)
        return {}


def _portfolio_stats_from_csv(stats_file: str) -> dict[str, Any]:
    try:
        df = pd.read_csv(stats_file)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["date"] = df["datetime"].dt.date
        daily = df.groupby("date")["portfolio_value"].last().sort_index()
        if len(daily) < 2:
            return {}
        pv = daily.values.astype(float)
        total_return = (pv[-1] - pv[0]) / pv[0]
        peak = pd.Series(pv).cummax().values
        max_dd = float(((pv - peak) / peak).min())
        daily_rets = pd.Series(pv).pct_change().dropna()
        sharpe = float(daily_rets.mean() / daily_rets.std() * (252 ** 0.5)) if daily_rets.std() > 0 else 0.0
        return {"total_return": float(total_return), "max_drawdown": max_dd, "sharpe_ratio": round(sharpe, 4)}
    except Exception as exc:
        log.warning("Portfolio stats failed for %s: %s", stats_file, exc)
        return {}


def _parse_result(result: Any, trades: list[dict], stats_file: str) -> dict[str, Any]:
    portfolio = _portfolio_stats_from_csv(stats_file)
    n = len(trades)
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n if n > 0 else 0.0
    gross_profit = sum(t["pnl_pct"] for t in wins)   if wins   else 0.0
    gross_loss   = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    def hold_days(t: dict) -> int:
        try:
            return (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
        except Exception:
            return 0

    avg_hold = (sum(hold_days(t) for t in trades) / n) if n > 0 else 0.0
    return {
        "total_trades":  n,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
        "total_return":  portfolio.get("total_return", "N/A"),
        "max_drawdown":  portfolio.get("max_drawdown", "N/A"),
        "sharpe_ratio":  portfolio.get("sharpe_ratio", "N/A"),
        "avg_hold_days": round(avg_hold, 1),
        "gld_trades":    sum(1 for t in trades if t["symbol"] == "GLD"),
        "fxy_trades":    sum(1 for t in trades if t["symbol"] == "FXY"),
    }


def run_scenario(scenario_name: str, slippage_bps: dict[str, float]) -> tuple[dict[str, Any], list[dict]]:
    global _COLLECTOR
    _COLLECTOR.clear()

    _RESULTS_3GATE_DIR.mkdir(parents=True, exist_ok=True)
    stats_file     = str(_RESULTS_3GATE_DIR / f"stats_{scenario_name}.csv")
    trades_file    = str(_RESULTS_3GATE_DIR / f"trades_{scenario_name}.csv")
    tearsheet_file = str(_RESULTS_3GATE_DIR / f"tearsheet_{scenario_name}.html")

    log.info("")
    log.info("=" * 60)
    log.info("HAVEN SCENARIO: %s", scenario_name.upper())
    log.info("Slippage: %s", slippage_bps)
    log.info("=" * 60)

    result = SynkHavenBacktest.backtest(
        YahooDataBacktesting,
        BACKTESTING_START,
        BACKTESTING_END,
        budget=INITIAL_BUDGET,
        benchmark_asset="SPY",
        parameters={"slippage_bps": slippage_bps, "scenario_name": scenario_name},
        show_plot=False,
        show_tearsheet=False,
        stats_file=stats_file,
        trades_file=trades_file,
        tearsheet_file=tearsheet_file,
        quiet_logs=True,
    )

    trades = list(_COLLECTOR)
    stats  = _parse_result(result, trades, stats_file)
    log.info("Scenario %s done | trades=%d win_rate=%.1f%%", scenario_name, stats["total_trades"], stats["win_rate"] * 100)
    return stats, trades


def _fmt(v: Any, pct: bool = False) -> str:
    if v == "N/A" or v is None:
        return "N/A"
    if isinstance(v, float):
        if pct:
            return f"{v * 100:.1f}%"
        if v == float("inf"):
            return "inf"
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    try:
        fv = float(str(v).replace("%", "").strip())
        if pct:
            return f"{fv:.1f}%" if abs(fv) > 1 else f"{fv * 100:.1f}%"
        return f"{fv:.4f}"
    except (ValueError, TypeError):
        return str(v)


def print_and_save_summary(scenario_results: dict[str, dict[str, Any]], benchmark: dict[str, Any]) -> None:
    run_date = datetime.now().strftime("%Y-%m-%d")

    def row(label: str, key: str, pct: bool = False) -> str:
        vals = [_fmt(scenario_results.get(s, {}).get(key, "N/A"), pct) for s in ("optimistic", "base", "pessimistic")]
        return f"{label:<16} {vals[0]:>14} {vals[1]:>10} {vals[2]:>13}"

    lines = [
        f"SYNK HAVEN-ONLY BACKTEST -- {run_date}",
        "========================================",
        "Symbols:    GLD + FXY only (defence excluded)",
        "Date range: 2020-01-01 to 2026-01-01",
        f"Sentiment gate: INCLUDED (pre-computed FinBERT/GDELT | column={LIVE_GATE_COLUMN})",
        f"Starting capital: ${INITIAL_BUDGET:,.0f}",
        "",
        f"{'':16} {'OPTIMISTIC':>14} {'BASE':>10} {'PESSIMISTIC':>13}",
        "-" * 55,
        row("Total trades:",  "total_trades"),
        row("Win rate:",      "win_rate",      pct=True),
        row("Profit factor:", "profit_factor"),
        row("Total return:",  "total_return",  pct=True),
        row("Max drawdown:",  "max_drawdown",  pct=True),
        row("Sharpe ratio:",  "sharpe_ratio"),
        row("Avg hold days:", "avg_hold_days"),
        row("GLD trades:",    "gld_trades"),
        row("FXY trades:",    "fxy_trades"),
        "",
        "BENCHMARK (GLD buy-and-hold):",
        f"  Total return:  {_fmt(benchmark.get('total_return', 'N/A'), pct=True)}",
        f"  Max drawdown:  {_fmt(benchmark.get('max_drawdown', 'N/A'), pct=True)}",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    summary_path = Path(__file__).parent / "results_summary_haven_3gate.txt"
    summary_path.write_text(output, encoding="utf-8")
    log.info("Summary saved -> %s", summary_path)


def save_trades_csv(trades: list[dict]) -> None:
    trades_path = Path(__file__).parent / "trades_log_haven_3gate.csv"
    fieldnames = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "quantity", "pnl_pct", "exit_reason", "slippage_scenario"]
    with open(trades_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)
    log.info("Trades log saved -> %s (%d rows)", trades_path, len(trades))


if __name__ == "__main__":
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_3GATE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SYNK HAVEN-ONLY BACKTEST (3-GATE)  2020-01-01 -> 2026-01-01")
    print(f"Symbols:  {SYMBOLS}  (defence excluded)")
    print(f"Budget:   ${INITIAL_BUDGET:,.0f}")
    print(f"Gate 3:   sentiment included ({LIVE_GATE_COLUMN})")
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
        log.warning("No trades recorded.")

    print(f"\nResults -> {Path(__file__).parent / 'results_summary_haven_3gate.txt'}")
    print(f"Trades  -> {Path(__file__).parent / 'trades_log_haven_3gate.csv'}")
