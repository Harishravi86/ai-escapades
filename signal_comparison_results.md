# 🔬 Experiment: SPY Signals vs. Stock's Own Signals

## Objective
Test the hypothesis that trading a stock using its **own panic signals** (e.g., NVDA chart) is superior to using **SPY's panic signals**.

## Results Summary

| Ticker | Winner 🏆 | SPY Signals Return | Self Signals Return | Insight |
|--------|----------|--------------------|---------------------|---------|
| **NVDA** | **SPY Signals** | **217%** | 162% | NVDA crashes are systemic; SPY captures them better. |
| **AAPL** | **SPY Signals** | **451%** | 129% | High correlation with market. |
| **MSFT** | **SPY Signals** | **161%** | 75% | High correlation with market. |
| **GOOGL**| **SPY Signals** | **169%** | 150% | High correlation with market. |
| **TSLA** | **Self Signals** | 14% | **97%** | **Idiosyncratic Risk.** TSLA has unique crashes (Elon, recalls) that SPY misses. |
| **AMZN** | **Self Signals** | 161% | **333%** | AMZN has distinct deep value cycles. |

## Analysis

### 1. The "Market Beta" Effect (NVDA, AAPL, MSFT)
For most mega-cap tech stocks, **SPY signals are superior**.
-   **Reason:** These stocks *are* the market. When they crash, the market crashes. SPY signals filter out "noise" (like a single bad earnings report) and only trigger on true systemic washouts, which are the best buying opportunities.
-   **NVDA Finding:** Contrary to the hypothesis, **NVDA performed better with SPY signals**. This suggests NVDA's best rallies come after *market-wide* fear, not just NVDA-specific dips.

### 2. The "Idiosyncratic" Effect (TSLA, AMZN)
**TSLA** and **AMZN** benefited significantly from using their **own signals**.
-   **TSLA:** Tesla often crashes when the market is flat (e.g., bad delivery numbers, CEO antics). SPY signals miss these "sale" opportunities. Using TSLA's own chart captured these unique dips.
-   **AMZN:** Amazon has had long periods of underperformance (2021-2022) where it dipped into deep value while SPY remained high. Self-signals captured these entries.

## Conclusion & Recommendation

1.  **Default to SPY Signals:** For most diversified "Blue Chip" trading (NVDA, AAPL, MSFT), rely on SPY. It provides a cleaner, higher-conviction signal.
2.  **Exception for Mavericks:** For stocks with high "idiosyncratic risk" or unique cycles (like **TSLA** or crypto-proxies), use the **Stock's Own Signals**.

### Proposed "Hybrid" Strategy
For a universal stock strategy, we could implement a **Hybrid Trigger**:
-   **Primary:** SPY Panic (captures systemic crashes)
-   **Secondary:** Stock Panic (captures unique discounts for stocks like TSLA)
-   **Condition:** If `Stock_Beta < 1.0` OR `Stock_Volatility > 2x SPY`, check Self Signals.
