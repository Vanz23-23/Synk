"""
backtest/defence_beta_isolation

Quantify how much of the defence-prime contribution is TIMING ALPHA vs
secular BETA (the 2022-2025 rearmament bull).

Method (per-trade timing alpha):
    For each defence trade, compute the stock's raw close-to-close return over
    the actual holding window (entry -> exit, in trading days). Compare it to
    the MEAN return of ALL same-length windows of that stock across 2020-2026
    (the "random same-length hold" benchmark = pure beta).

        alpha_trade = trade_window_return - mean_same_length_window_return

    If mean alpha > 0 (statistically), the gates select better-than-random
    windows -> timing alpha. If alpha ~= 0, the strategy merely captured the
    average defence drift -> beta.

    Raw close returns (no slippage) are used on BOTH sides so the comparison
    isolates entry/exit timing, not transaction cost.

Stats reported: n, mean alpha, 95% CI, one-sample t-test (H0: alpha=0),
Cohen's d, % positive, sign test. Plus defence-prime basket buy-and-hold
(equal-weight) as beta context.

Runs the live-faithful expanded config (z>=1.0 defence, primes added,
treasuries dropped) to capture the exact trades, then analyses them.

Usage (from synk/ root):
    python backtest/defence_beta_isolation.py

Runtime: ~4-6 min (one backtest + yfinance downloads).
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

import backtest.synk_backtest as bt  # noqa: E402

NEW_DEFENCE = ["GD", "RTX", "LHX"]
ALL_DEFENCE = ["LMT", "NOC", "ITA", "GD", "RTX", "LHX"]
LIVE_DEFENCE_Z = 1.0
START, END = "2020-01-01", "2026-01-01"
_NEW_SLIPPAGE_BASE = {"GD": 10.0, "RTX": 10.0, "LHX": 10.0}


def capture_expanded_trades() -> list[dict]:
    """Run the live-faithful expanded config and return clean per-trade records."""
    orig_symbols = list(bt.SYMBOLS)
    orig_defence = set(bt._DEFENCE)
    orig_def_z = bt.DEFENCE_Z_THRESHOLD

    bt.DEFENCE_Z_THRESHOLD = LIVE_DEFENCE_Z
    bt.SYMBOLS = orig_symbols + NEW_DEFENCE
    bt._DEFENCE = orig_defence | set(NEW_DEFENCE)
    slippage = {**bt.SLIPPAGE_SCENARIOS["base"], **_NEW_SLIPPAGE_BASE}

    _, trades = bt.run_scenario("beta_isolation_expanded", slippage)

    bt.SYMBOLS = orig_symbols
    bt._DEFENCE = orig_defence
    bt.DEFENCE_Z_THRESHOLD = orig_def_z
    return trades


def load_closes(symbols: list[str]) -> dict[str, pd.Series]:
    """Download adjusted close series per symbol, trading-day indexed."""
    import yfinance as yf
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        df = yf.download(sym, start=START, end=END, progress=False, auto_adjust=True)
        closes = df["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        out[sym] = closes.dropna()
    return out


def per_trade_alpha(trades: list[dict], closes: dict[str, pd.Series], symbols: set[str]) -> pd.DataFrame:
    """Build a per-trade dataframe of trade-window return, same-length-window mean, alpha."""
    rows = []
    for t in trades:
        sym = t["symbol"]
        if sym not in symbols or sym not in closes:
            continue
        s = closes[sym]
        entry_ts = pd.Timestamp(t["entry_date"])
        exit_ts = pd.Timestamp(t["exit_date"])

        entry_idx = s.index.searchsorted(entry_ts, side="left")
        exit_idx = s.index.searchsorted(exit_ts, side="right") - 1
        if entry_idx >= len(s) or exit_idx <= entry_idx:
            continue
        n = exit_idx - entry_idx  # holding length in trading days
        px = s.to_numpy(dtype=float)

        trade_ret = px[exit_idx] / px[entry_idx] - 1.0
        # All same-length windows across the full series = beta benchmark
        if n >= len(px):
            continue
        window_rets = px[n:] / px[:-n] - 1.0
        mean_bench = float(np.mean(window_rets))

        rows.append({
            "symbol": sym,
            "hold_days": n,
            "trade_ret": trade_ret,
            "bench_mean_ret": mean_bench,
            "alpha": trade_ret - mean_bench,
        })
    return pd.DataFrame(rows)


def basket_buy_and_hold(closes: dict[str, pd.Series], symbols: list[str]) -> dict[str, float]:
    """Equal-weight buy-and-hold of the given symbols: return, max DD, Sharpe."""
    norm = []
    for sym in symbols:
        s = closes[sym]
        norm.append(s / s.iloc[0])
    basket = pd.concat(norm, axis=1).dropna().mean(axis=1)
    total_return = float(basket.iloc[-1] / basket.iloc[0] - 1.0)
    peak = basket.cummax()
    max_dd = float(((basket - peak) / peak).min())
    rets = basket.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0
    return {"total_return": total_return, "max_drawdown": max_dd, "sharpe": sharpe}


def analyse(df: pd.DataFrame, label: str) -> list[str]:
    """One-sample alpha stats vs H0: mean alpha = 0."""
    n = len(df)
    if n < 3:
        return [f"{label}: n={n} (too few trades to test)"]
    a = df["alpha"].to_numpy(dtype=float)
    mean_a = float(np.mean(a))
    std_a = float(np.std(a, ddof=1))
    se = std_a / np.sqrt(n)
    t_stat, p_val = scipy_stats.ttest_1samp(a, 0.0)
    ci_lo, ci_hi = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_a, scale=se)
    cohen_d = mean_a / std_a if std_a > 0 else 0.0
    pos = int((a > 0).sum())
    # sign test (binomial, H0: p=0.5)
    sign_p = scipy_stats.binomtest(pos, n, 0.5).pvalue

    return [
        f"{label}  (n={n} trades)",
        f"  Mean trade-window return : {df['trade_ret'].mean()*100:+.2f}%",
        f"  Mean same-length benchmark: {df['bench_mean_ret'].mean()*100:+.2f}%",
        f"  Mean ALPHA               : {mean_a*100:+.2f}%  (95% CI [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%])",
        f"  t({n-1})={t_stat:+.3f}  p={p_val:.4f}  Cohen's d={cohen_d:+.3f}",
        f"  Positive-alpha trades    : {pos}/{n} ({100*pos/n:.0f}%)  sign-test p={sign_p:.4f}",
    ]


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DEFENCE BETA-ISOLATION  (timing alpha vs secular beta)")
    print("=" * 60)

    trades = capture_expanded_trades()
    closes = load_closes(ALL_DEFENCE)

    df_primes = per_trade_alpha(trades, closes, set(NEW_DEFENCE))
    df_all = per_trade_alpha(trades, closes, set(ALL_DEFENCE))

    prime_bh = basket_buy_and_hold(closes, NEW_DEFENCE)

    run_date = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"DEFENCE BETA-ISOLATION — {run_date}",
        f"Expanded live-faithful config (z>={LIVE_DEFENCE_Z} defence) | {START} -> {END}",
        "Raw close-to-close returns (no slippage) on both sides.",
        "",
        "Q: do gated entries beat random same-length holds (alpha) or just",
        "   capture the average defence drift (beta)?",
        "",
        "-" * 58,
        *analyse(df_primes, "DEFENCE PRIMES (GD/RTX/LHX)"),
        "",
        *analyse(df_all, "ALL DEFENCE (LMT/NOC/ITA/GD/RTX/LHX)"),
        "",
        "-" * 58,
        "BETA CONTEXT — defence-prime basket buy-and-hold (equal weight):",
        f"  Total return : {prime_bh['total_return']*100:+.1f}%",
        f"  Max drawdown : {prime_bh['max_drawdown']*100:+.1f}%",
        f"  Sharpe       : {prime_bh['sharpe']:.3f}",
        "",
        "READ:",
        "  alpha p<0.05 & CI excludes 0  -> gates have real timing skill.",
        "  alpha ~0 / p>0.05             -> contribution is beta; the bot just",
        "                                   rode the rearmament drift.",
        "  Compare strategy (intermittent, low-DD) vs basket B&H (always-in,",
        "  high return, high DD) for the risk-adjusted trade-off.",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    out_path = Path(__file__).parent / "results_defence_beta_isolation.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved -> {out_path}")
