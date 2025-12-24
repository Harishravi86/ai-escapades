# 🤖 Experiment: Machine Learning on Hourly Charts (3 STD)

## Objective
Train a Random Forest model specifically on **Hourly (1h)** data to predict "Dip Reversals".
**Features:** Bollinger Bands (2.0 & 3.0 STD), RSI, Hourly Return, Time of Day.
**Target:** +1.5% Return in next 12 hours (without hitting -1% Stop Loss).

## Results (Test Set: Last ~300 Days)

| Ticker | ML Strategy Return | Buy & Hold Return | Win Rate | Verdict |
|--------|--------------------|-------------------|----------|---------|
| **SPY** | **+25.12%** | +19.31% | 58% | ✅ **BEAT MARKET** |
| **NVDA** | **-15.18%** | **+46.74%** | 42% | ❌ **FAILED** |

## Analysis

### 1. SPY: The "Index Scalper" Works 🏆
-   **Success:** The ML model successfully identified hourly dips in SPY that reversed.
-   **Why:** SPY is naturally mean-reverting on hourly timeframes. The model learned to buy when RSI was low and price hit the bands, capturing quick 1.5% bounces.
-   **Outperformance:** +25% vs +19% is a significant edge for a short-term strategy.

### 2. NVDA: The "Momentum Killer" 💀
-   **Failure:** The strategy lost money (-15%) while the stock soared (+46%).
-   **Why:** NVDA is a **Momentum** asset. When it dips on an hourly chart, it often keeps dipping (momentum correction) or consolidates. It does not "snap back" as reliably as the index.
-   **Lesson:** Do NOT apply mean-reversion logic to high-beta momentum stocks on hourly charts. You will get run over.

### 3. Feature Importance
-   **Top Feature:** `RSI_14` (Relative Strength Index).
-   **3 STD Bands:** Were useful but secondary to RSI. The model preferred RSI for gauging overextension.

## Recommendation for Live Bot

1.  **For SPY:** We *could* enable an "Hourly ML Mode" for SPY, as it demonstrated alpha.
2.  **For NVDA:** **STRICTLY PROHIBIT** hourly mean reversion. Only trade NVDA on **Daily Signals** (Trend Following).

**Final Decision:** To keep the bot robust and safe for v1.0, I recommend sticking to the **Daily Strategy** for *all* assets. The SPY hourly edge is real but requires constant re-training to maintain.
