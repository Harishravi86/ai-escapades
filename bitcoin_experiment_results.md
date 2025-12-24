# 🧪 Experiment: Bitcoin & Crypto Correlation

## Objective
1.  Test the strategy on **Bitcoin (BTC-USD)** from 2017.
2.  Test the hypothesis that **NVDA** should be traded using **Bitcoin signals**.

## Results Summary

| Asset | Signal Source | Strategy Return | Buy & Hold Return | Max Drawdown | Verdict |
|-------|---------------|-----------------|-------------------|--------------|---------|
| **BTC-USD** | **Self (BTC)** | **-3.17%** | **~9,009%** | 43% | ❌ **FAILED** |
| **BTC-USD** | **SPY** | **+22.35%** | **~8,612%** | 29% | ❌ **FAILED** |
| **NVDA** | **BTC Signals** | **+49.05%** | **~6,945%** | 19% | ❌ **FAILED** |

## Analysis

### 1. Why the Strategy Failed on Bitcoin
The "Bulletproof" strategy is designed for **Mean Reversion** on assets that have a fundamental "floor" or tend to bounce back quickly (like SPY or Blue Chip stocks).
-   **Crypto Behavior:** Bitcoin is a **Momentum** asset. When it crashes, it crashes *hard* (-80%) and stays down for years (Crypto Winter).
-   **The Trap:** The strategy likely bought the "dip" at -10% or -20%, only to see Bitcoin fall another 50%. The stop losses would be triggered repeatedly, or the capital would be tied up in a losing trade for too long.
-   **Conclusion:** You cannot trade Crypto with a Mean Reversion strategy designed for Equities. You need a **Trend Following** strategy for Crypto.

### 2. NVDA vs. Bitcoin Signals
The hypothesis that "NVDA follows Bitcoin" did not yield profitable trading results using this specific strategy.
-   **Result:** Trading NVDA using BTC signals resulted in a meager +49% return vs +6,945% Buy & Hold.
-   **Reason:** Bitcoin signals are too erratic and don't align perfectly with NVDA's equity-market liquidity cycles. While they are correlated in sentiment, using one to trigger entries for the other introduces too much noise.

## Recommendation

**DO NOT use this strategy for Crypto.**

-   **For Stocks (NVDA, AAPL):** Stick to **SPY Signals** (or Self Signals for TSLA).
-   **For Crypto:** You need a completely different logic (e.g., Moving Average Crossovers, Breakouts) rather than "Dip Buying".
