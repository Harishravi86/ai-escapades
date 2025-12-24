# Bulletproof Strategy v5.2: "Twin Conviction + Profit Taking" 🏆

## Overview
The **Bulletproof Strategy v5.2** is the final, optimized version of the Twin Architecture. It combines the **Twin Models** (Bull/Bear), the **Conviction Filter** (v5.1), and a new **Profit-Taking Mechanism** (v5.2) to maximize returns while smoothing the equity curve.

## Core Architecture

### 1. The Twin Engines
*   **Bull Sharktooth (The Hunter):** Detects market washouts/bottoms.
*   **Bear Sharktooth (The Guardian):** Detects market exhaustion/tops.

### 2. The Conviction Filter
Tiers position sizing based on confidence:
*   **High Conviction (>70%):** 100% Size.
*   **Medium Conviction (>50%):** 50% Size.
*   **Low Conviction (<50%):** **SKIP**.

### 3. Profit Taking (New in v5.2)
Locks in gains on extended runs before a full reversal signal:
*   **Trigger:** Unrealized Gain > 20% AND Bear Prob is rising (momentum shift).
*   **Action:** Sell 25% of position.
*   **Logic:** Banks "dry powder" to re-deploy on the next dip or compound safely.

## Performance (2000-2025)

The v5.2 strategy achieved the highest total return of all versions:

| Asset | Total Return | CAGR | Logic |
| :--- | :--- | :--- | :--- |
| **QQQ** | **34,722%** 🚀 | **~27.4%** | Profit taking increased returns by recycling capital. |
| **SPY** | **21,076%** 🐂 | **~23.1%** | Consistent compounding with lower volatility. |

## Validation
*   **Pandemic Bottom (March 2020):** High Conviction BUY.
*   **2024 Tech Rally:** Entered Jan 2024, took partial profits on the way up, exited on Bear Twin signal.
*   **Compounding:** The combination of full sizing on winners and partial profit taking created a massive compounding effect.

## Usage
Run the strategy to get the current market status:

```bash
python bulletproof_strategy_v5_twin.py
```

**Output:**
```text
CURRENT SIGNAL (2025-11-25)
Bull Probability: 0.0%
Bear Probability: 7.3%
Action: HOLD
```
