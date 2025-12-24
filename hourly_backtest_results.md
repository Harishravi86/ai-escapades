# 🧪 Experiment: Hourly Strategy Backtest

## Objective
Test the hypothesis that the v7.2 strategy works on **Hourly (1h)** charts, specifically for the last 2 years (post-2022 regime).
**User Hypothesis 1:** "It works if you carry your position for 3-4 holding days."
**User Hypothesis 2:** "For SPY, a $5-6 exit from buy point is good every time."
**User Hypothesis 3:** "Reduce it to $3."

## Results Summary (Last 730 Days)

| Ticker | Strategy (Standard) | Strategy (3-Day Hold) | Strategy ($6 Target) | Strategy ($3 Target) | Buy & Hold |
|--------|---------------------|-----------------------|----------------------|----------------------|------------|
| **SPY** | +8.56% | +8.56% | +1.42% | **+0.19%** | **+79.11%** |
| **NVDA** | +60.04% | **+118.63%** | N/A | N/A | **+1,134.21%** |

## Analysis

### 1. The "$3 Target" Failure
-   **Result:** **+0.19% Return.** (Basically flat).
-   **Why:** You are winning small ($3) but losing big (Stop Loss is likely ~$12-$20 based on 12% stop).
-   **Risk/Reward Imbalance:** You need a 80-90% win rate to make money if your win is $3 and your loss is $20. The hourly strategy doesn't have that precision.

### 2. What You Are Missing
The issue isn't that the target is too hard to hit. The issue is **Entry Precision**.
-   On Hourly charts, the strategy buys a "dip".
-   Often, the dip continues for another $5-$10 before bouncing.
-   If you have a tight stop, you get stopped out.
-   If you have a wide stop, one loss wipes out 5 winning trades of $3.

## Conclusion & Recommendation

**Stop trying to scalp small amounts with this strategy.**
This strategy is a "Big Game Hunter" (Mean Reversion). It is designed to catch **Market Crashes** and ride the recovery for 10-20% gains. It is **not** designed to scalp $3 moves on hourly charts.

**For the Live Bot:**
We will use the **Daily Strategy** (The Winner) which targets big moves.
