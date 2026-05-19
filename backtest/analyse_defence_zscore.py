"""
Analyse GPR z-score distribution at defence entry points.
Answers: how many defence trades fired at z=0.5-1.0 vs z>1.5?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "backtest" / "trades_log.csv"
GPR_PATH = ROOT / "data" / "gpr_daily_recent.xls"
ROLLING_WINDOW = 252

DEFENCE_SYMBOLS = {"LMT", "NOC", "ITA"}

# ── Load GPR z-scores ──────────────────────────────────────────────────────────
gpr_raw = pd.read_excel(GPR_PATH)
gpr_raw.columns = [c.strip() for c in gpr_raw.columns]

date_col = next(c for c in gpr_raw.columns if "date" in c.lower())
gpr_col  = next(c for c in gpr_raw.columns if "GPRD" in c or c == "GPRD")

gpr = (
    gpr_raw[[date_col, gpr_col]]
    .rename(columns={date_col: "date", gpr_col: "gprd"})
    .dropna(subset=["gprd"])
    .copy()
)
gpr["date"] = pd.to_datetime(gpr["date"])
gpr = gpr.sort_values("date").reset_index(drop=True)

gpr["zscore"] = (
    (gpr["gprd"] - gpr["gprd"].rolling(ROLLING_WINDOW).mean())
    / gpr["gprd"].rolling(ROLLING_WINDOW).std()
)
gpr = gpr.dropna(subset=["zscore"]).set_index("date")

# ── Load trades ────────────────────────────────────────────────────────────────
trades = pd.read_csv(TRADES, parse_dates=["entry_date"])

# Use only base scenario to avoid triple-counting
defence = trades[
    (trades["symbol"].isin(DEFENCE_SYMBOLS)) &
    (trades["slippage_scenario"] == "base")
].copy()

# Map entry date -> nearest GPR z-score (GPR is daily but may not align exactly)
def lookup_z(entry_date):
    idx = gpr.index.searchsorted(entry_date, side="right") - 1
    if idx < 0:
        return np.nan
    return float(gpr["zscore"].iloc[idx])

defence["zscore_at_entry"] = defence["entry_date"].apply(lookup_z)

# ── Bin z-scores ───────────────────────────────────────────────────────────────
bins   = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
labels = ["<0.5 (below gate)", "0.5–1.0 (elevated)", "1.0–1.5 (mid)", "1.5–2.0 (high)", ">2.0 (extreme)"]

defence["z_bin"] = pd.cut(defence["zscore_at_entry"], bins=bins, labels=labels)

print("\nDEFENCE TRADES — GPR z-score at entry (base scenario)")
print("=" * 58)
dist = defence["z_bin"].value_counts().reindex(labels, fill_value=0)
total = len(defence)
for label, count in dist.items():
    bar = "#" * count
    print(f"  {label:<28}  {count:>3}  ({100*count/total:4.1f}%)  {bar}")

print(f"\n  Total defence trades: {total}")

# ── Per-symbol breakdown ───────────────────────────────────────────────────────
print("\nPer-symbol breakdown:")
print(f"  {'Symbol':<6}  {'<0.5':>4}  {'0.5-1.0':>7}  {'1.0-1.5':>7}  {'1.5-2.0':>7}  {'>2.0':>5}")
for sym in ["LMT", "NOC", "ITA"]:
    sub = defence[defence["symbol"] == sym]["z_bin"].value_counts().reindex(labels, fill_value=0)
    print(f"  {sym:<6}  {sub.iloc[0]:>4}  {sub.iloc[1]:>7}  {sub.iloc[2]:>7}  {sub.iloc[3]:>7}  {sub.iloc[4]:>5}")

# ── Summary stat ──────────────────────────────────────────────────────────────
loose = dist.iloc[1]   # 0.5-1.0
tight = dist.iloc[2] + dist.iloc[3] + dist.iloc[4]  # >=1.0
print(f"\n  Fired at z=0.5–1.0 (loose threshold):  {loose}  ({100*loose/total:.1f}%)")
print(f"  Fired at z>=1.0   (tighter threshold): {tight}  ({100*tight/total:.1f}%)")
print(f"\n  Median z-score at defence entry: {defence['zscore_at_entry'].median():.3f}")
print(f"  Mean   z-score at defence entry: {defence['zscore_at_entry'].mean():.3f}")
