
======================================================================
A/B COMPARISON: v6.2 vs v7.2 (OUT-OF-SAMPLE TEST)
======================================================================

Metric               |            v6.2 |            v7.2 |         Diff |   Winner
---------------------------------------------------------------------------
Total Return         |         108.93% |         154.47% |      +45.55% |     v7.2
CAGR                 |          13.15% |          16.95% |       +3.80% |     v7.2
Max Drawdown         |          19.76% |          16.09% |       +3.67% |     v7.2
Sharpe Ratio         |           2.03 |           2.54 |        +0.50 |     v7.2
Win Rate             |          63.64% |          71.70% |       +8.06% |     v7.2
Total Trades         |          44.00 |          53.00 |        +9.00 |     v7.2
---------------------------------------------------------------------------

OVERALL WINNER: v7.2 (6/6 metrics)

======================================================================
RISK-ADJUSTED ANALYSIS
======================================================================
Calmar Ratio (CAGR/MaxDD):
  v6.2: 0.67
  v7.2: 1.05
  Winner: v7.2

======================================================================
v7.2 FEATURE IMPORTANCE (Did Pine features help?)
======================================================================

Pine Script features contribution: 4.6% of total importance

Pine feature rankings:
  #34: BB_20_crossunder (0.0056)
  #49: BB_50_crossunder (0.0025)
  #68: BB_20_crossover (0.0000)
  #69: BB_50_crossover (0.0000)
  #11: DAILY_RETURN_PANIC (0.0099)
  #12: DAILY_RETURN_CRASH (0.0099)
  #47: PINE_ENTRY_SIGNAL (0.0027)
  #15: MULTI_CROSSUNDER (0.0075)
  #14: DAILY_RETURN_EXTREME (0.0079)
  #89: DAILY_RETURN_SURGE (0.0000)

WARNING: Pine features contribute <5% importance.
   The ML model is not finding them useful.
   v7.2 may be adding noise, not signal.

======================================================================
VERDICT
======================================================================
v7.2 is BETTER: Higher Sharpe AND lower drawdown
  -> Upgrade to v7.2