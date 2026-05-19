"""
gpr_loader

Loads the Caldara-Iacoviello Geopolitical Risk (GPR) Index from the official
source (https://www.matteoiacoviello.com/gpr.htm). Parses the monthly GPR
values, computes a rolling z-score, and exposes a function that returns the
current GPR regime label ('elevated' / 'normal') based on whether the z-score
exceeds the configured threshold (default 0.5). Used by signals/regime_filter.py
as the first gate in the three-factor signal stack.

Inputs:  none (downloads data directly from the web)
Outputs: pd.DataFrame with columns [date, gpr_raw, gpr_zscore, regime]
Deps:    requests, pandas
"""
