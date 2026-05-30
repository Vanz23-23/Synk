# Investigation: Can Synk Trade More Often and Stay Profitable?

**Date:** 2026-05-30
**Question:** The live sentiment gate (`p60/s30`) opened only 0.2% of days — effectively a permanent off-switch. Loosening it raised a broader question: can the bot trade *more often* while staying profitable? This investigation tested four candidate levers and ran a beta-isolation check on the most promising one.

**Headline conclusion:** No lever produced genuine timing alpha. The one lever that increased both frequency and return (universe expansion with defence primes) was shown to be **beta** — riding the 2022–2025 rearmament drift — not selection skill. The strategy's real value remains low-drawdown, selective tail-hedge participation.

All experiments were **backtest-only** (2020-01-01 → 2026-01-01, base slippage, 3-gate stack with sentiment `gate_at_p55_s20`). The only live-code change was the sentiment threshold (see below).

---

## Lever 1 — Sentiment threshold (LIVE CHANGE)

Old `p60/s30` opened 0.2% of days (mean dominant_prob 0.589 vs 0.60 floor; mean score −0.24 vs −0.30). Changed to **`p55/s20`** → 55.3% gate-open rate. This fixed a broken gate.

But the 3-gate backtest showed sentiment is a **weak, marginally dilutive** filter:

| Metric | 2-gate | 3-gate (p55_s20) |
|---|---|---|
| Total trades | 164 | 150 |
| Profit factor | 1.82 | 1.76 |
| Total return | +4.9% | +4.4% |

**Open question:** whether Gate 3 earns its place in the live stack at all. Deferred.

## Lever 2 — Regime threshold (rejected)

Sweeping z≥0.50 → 0.25 → 0.00 *reduced* trades and degraded risk-adjusted return:

| | z≥0.00 | z≥0.25 | z≥0.50 (current) |
|---|---|---|---|
| Trades | 118 | 134 | **150** |
| Profit factor | 1.88 | 1.53 | 1.76 |
| Total return | +3.0% | +3.3% | **+4.4%** |
| Sharpe | 0.24 | 0.27 | **0.43** |

Looser = worse, because of the structural bottleneck (Lever 3).

## Lever 3 — Cooldown / one-position-per-symbol (diagnosed)

Skip-reason instrumentation added to `synk_backtest.py`. Cooldown sweep {0,1,3,5} barely moved trade count (149/151/150/139). The diagnostic showed why:

```
occupied = 1544   (entries blocked: symbol already held)
cooldown =  276
```

**`occupied` outnumbers `cooldown` ~6:1.** The binding constraint is **one-position-per-symbol**, not the cooldown. This also explains Lever 2: more open days → positions held more often → more "occupied" blocks → fewer fresh entries. Each *symbol* is an independent position slot — which is why Lever 4 worked.

## Lever 4 — Universe expansion (works in backtest, but = beta)

Added GD/RTX/LHX (defence primes) + TLT/IEF (treasuries). Treasuries were dead weight (~10 trades). At the **live-faithful z≥1.0 defence gate**, baseline vs +primes:

| Metric | Baseline | +Primes |
|---|---|---|
| Total trades | 137 | **222 (+62%)** |
| Profit factor | 1.67 | **1.93** |
| Total return | +4.6% | **+10.0%** |
| Sharpe | 0.52 | **0.66** |
| Max drawdown | −2.1% | −3.4% |

Secondary finding: tightening defence to z≥1.0 *raised* Sharpe — the live gate is well-justified.

## Beta-isolation check — verdict: it's BETA, not alpha

Per-trade test: did gated entries beat a random same-length hold of the same stock (raw close returns)?

| | Primes (n=85) | All defence (n=154) |
|---|---|---|
| Mean trade return | +0.41% | +0.20% |
| Mean random-hold benchmark | +0.63% | +0.59% |
| **Mean alpha** | **−0.22%** | **−0.39%** |
| t-test p | 0.62 | 0.23 |
| Positive-alpha trades | 34% | 36% |
| Sign-test p | 0.0045 | 0.0005 |

- **No timing alpha:** mean alpha ≈ 0 (CI straddles zero, p>0.05).
- **Sign test damning:** only ~35% of trades beat random entry (highly significant) — the *median* trade underperforms a random hold; the mean is rescued only by a few tail winners.
- **Beta context:** defence-prime basket buy-and-hold returned **+99.8%** (Sharpe 0.60) vs the strategy's Sharpe 0.66 — same risk-adjusted quality. The strategy's only edge is low drawdown (−3.4% vs −42.8%), from small intermittent exposure, not timing.

**Conclusion:** the +10% was the bot riding rearmament beta. Promoting GD/RTX/LHX as a "validated edge" would be mislabeled beta. If defence beta is wanted, take it deliberately and cheaply (small permanent ETF allocation), not via gated single-names.

---

## Tooling added (ships in repo)

- `backtest/synk_backtest.py` — sentiment gate integrated (3-gate), skip-reason diagnostic, patchable `DEFENCE_Z_THRESHOLD`.
- `backtest/universe_expansion.py` — baseline vs expanded universe.
- `backtest/universe_validate_live.py` — expanded at live z≥1.0 defence gate.
- `backtest/cooldown_sweep.py` — cooldown ∈ {0,1,3,5}.
- `backtest/defence_beta_isolation.py` — per-trade alpha vs same-length-window beta.

Result `.txt`/`.csv` outputs are gitignored per repo convention ("scripts ship; results don't").

## Open decisions

1. Does Gate 3 (sentiment) belong in the live stack? (weak/marginally dilutive)
2. Deliberate defence-beta allocation as a conscious portfolio tilt — design pending.
3. One-position-per-symbol refactor — confirmed bottleneck, but no lever showed alpha, so frequency-for-its-own-sake is not advised.
