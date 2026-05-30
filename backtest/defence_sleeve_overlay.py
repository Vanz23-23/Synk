"""
backtest/defence_sleeve_overlay

Prototype: a SEPARATE buy-and-hold defence sleeve held OUTSIDE Synk's gated
logic and kill-switch, blended with the gated strategy as a conscious
portfolio tilt.

Rationale: the frequency/alpha investigation (2026-05-30) showed the defence
contribution is BETA, not timing alpha. Rather than gate defence single-names
(mislabeled beta), take the beta deliberately as a passive sleeve and measure
the COMBINED risk-return — the value of a separate, low-correlation sleeve is
portfolio-level, not per-trade.

Model:
    Synk leg  : gated baseline equity curve (5-symbol, 3-gate, base slippage),
                read from results/3gate/stats_universe_baseline.csv (no re-run).
    Sleeve    : buy-and-hold defence ETF (ITA primary; PPA/XAR for comparison).
    Blend     : combined = (1-w)*Synk + w*Sleeve, ANNUAL rebalance to target w.
    Sweep     : w in {0, 5, 10, 15, 20, 30, 100}%.

Reports total return, CAGR, max drawdown, Sharpe per weight, plus the
Synk<->sleeve daily-return correlation (diversification benefit).

This is a PROTOTYPE / analysis only — it does not touch live config or the
live strategy. Use it to choose a tolerable allocation before any real change.

Usage (from synk/ root):
    python backtest/defence_sleeve_overlay.py

Runtime: ~1-2 min (reads one CSV + yfinance downloads; no backtest re-run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

_SYNK_EQUITY_CSV = _HERE / "backtest" / "results" / "3gate" / "stats_universe_baseline.csv"
START, END = "2020-01-01", "2026-01-01"

SLEEVE_PRIMARY = "PPA"  # chosen vehicle (best return/DD/Sharpe of the three)
SLEEVE_COMPARE = ["ITA", "PPA", "XAR"]
WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.00]
_TRADING_DAYS = 252


def load_synk_equity() -> pd.Series:
    """Daily Synk portfolio value (one obs per date) from the backtest stats CSV."""
    if not _SYNK_EQUITY_CSV.exists():
        raise FileNotFoundError(
            f"Synk equity CSV not found: {_SYNK_EQUITY_CSV}\n"
            "Run `python backtest/universe_expansion.py` first to generate it."
        )
    df = pd.read_csv(_SYNK_EQUITY_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date")["portfolio_value"].last().sort_index()
    daily.index = pd.to_datetime(daily.index)
    return daily


def load_etf_close(symbol: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=True)
    closes = df["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes = closes.dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes


def _metrics(value: pd.Series) -> dict[str, float]:
    """Total return, CAGR, max drawdown, annualised Sharpe from a value series."""
    v = value.to_numpy(dtype=float)
    total_return = v[-1] / v[0] - 1.0
    years = (value.index[-1] - value.index[0]).days / 365.25
    cagr = (v[-1] / v[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    peak = pd.Series(v).cummax().to_numpy()
    max_dd = float(((v - peak) / peak).min())
    rets = pd.Series(v).pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(_TRADING_DAYS)) if rets.std() > 0 else 0.0
    return {"total_return": total_return, "cagr": cagr, "max_drawdown": max_dd, "sharpe": sharpe}


def blend(synk_ret: pd.Series, sleeve_ret: pd.Series, w: float) -> pd.Series:
    """Combined value series: (1-w) Synk + w sleeve, annual rebalance to target.

    Weights drift intra-year with returns; reset to target at each year boundary.
    """
    idx = synk_ret.index
    years = idx.year
    v_synk = (1.0 - w)          # capital in Synk leg
    v_sleeve = w                # capital in sleeve leg
    combined = []
    prev_year = years[0]
    for i, dt in enumerate(idx):
        # rebalance to target at the start of a new calendar year
        if years[i] != prev_year:
            total = v_synk + v_sleeve
            v_synk = total * (1.0 - w)
            v_sleeve = total * w
            prev_year = years[i]
        v_synk *= (1.0 + synk_ret.iloc[i])
        v_sleeve *= (1.0 + sleeve_ret.iloc[i])
        combined.append(v_synk + v_sleeve)
    return pd.Series(combined, index=idx)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DEFENCE SLEEVE OVERLAY  (separate buy-and-hold tilt)")
    print("=" * 60)

    synk = load_synk_equity()
    synk_norm = synk / synk.iloc[0]

    # --- Standalone ETF comparison (vehicle choice) ---
    etf_closes: dict[str, pd.Series] = {}
    vehicle_lines = ["Standalone defence-ETF buy-and-hold (vehicle comparison):"]
    for sym in SLEEVE_COMPARE:
        c = load_etf_close(sym)
        etf_closes[sym] = c
        m = _metrics(c)
        vehicle_lines.append(
            f"  {sym:<4} return={_fmt_pct(m['total_return']):>8}  CAGR={_fmt_pct(m['cagr']):>7}  "
            f"maxDD={_fmt_pct(m['max_drawdown']):>7}  Sharpe={m['sharpe']:.3f}"
        )

    # --- Align Synk + primary sleeve on common dates ---
    sleeve_close = etf_closes[SLEEVE_PRIMARY]
    common = synk_norm.index.intersection(sleeve_close.index)
    synk_ret = synk_norm.reindex(common).pct_change().dropna()
    sleeve_ret = (sleeve_close.reindex(common) / sleeve_close.reindex(common).iloc[0]).pct_change().dropna()
    common_ret = synk_ret.index.intersection(sleeve_ret.index)
    synk_ret = synk_ret.reindex(common_ret)
    sleeve_ret = sleeve_ret.reindex(common_ret)

    corr = float(synk_ret.corr(sleeve_ret))

    # --- Weight sweep ---
    rows = []
    for w in WEIGHTS:
        combined = blend(synk_ret, sleeve_ret, w)
        m = _metrics(combined)
        rows.append((w, m))

    # --- Report ---
    col = 12
    header = "  " + f"{'weight':<10}" + "".join(
        f"{_fmt_pct(w):>{col}}" for w, _ in rows
    )
    divider = "  " + "-" * (10 + col * len(rows))

    def row(label: str, key: str) -> str:
        return "  " + f"{label:<10}" + "".join(
            f"{_fmt_pct(m[key]):>{col}}" for _, m in rows
        )

    def row_sharpe() -> str:
        return "  " + f"{'Sharpe':<10}" + "".join(
            f"{m['sharpe']:>{col}.3f}" for _, m in rows
        )

    from datetime import datetime as _dt
    lines = [
        f"DEFENCE SLEEVE OVERLAY — {_dt.now():%Y-%m-%d}",
        f"Synk leg: gated baseline (5-symbol 3-gate, base slippage) | sleeve: {SLEEVE_PRIMARY}",
        f"Blend: (1-w) Synk + w sleeve, annual rebalance | {START} -> {END}",
        "",
        *vehicle_lines,
        "",
        f"Synk<->{SLEEVE_PRIMARY} daily-return correlation: {corr:+.3f}",
        "  (lower = more diversification benefit from a separate sleeve)",
        "",
        f"BLEND SWEEP (w = sleeve weight; w=0 is pure Synk, w=100% is pure {SLEEVE_PRIMARY}):",
        header,
        divider,
        row("Total ret:", "total_return"),
        row("CAGR:",      "cagr"),
        row("Max DD:",    "max_drawdown"),
        row_sharpe(),
        "",
        "READ: pick the largest weight whose Max DD stays within your tolerance",
        "      while CAGR/Sharpe improve. This sleeve sits OUTSIDE Synk's gated",
        "      logic + kill-switch (a manual portfolio tilt), so its drawdown is",
        "      NOT capped by the 30% halt. PROTOTYPE ONLY — no live config touched.",
    ]

    output = "\n".join(lines)
    print("\n" + output + "\n")

    out_path = Path(__file__).parent / "results_defence_sleeve_overlay.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved -> {out_path}")
