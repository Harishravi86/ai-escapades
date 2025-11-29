# 🧪 Strategy Experiment: Pine Script Alignment

## Objective
Align the Python implementation with the original Pine Script to capture the "exact moment" of crashes using crossover detection, hoping to improve performance.

## Experiments & Results

| Version | Logic | Return (QQQ) | Win Rate | Status |
|---------|-------|--------------|----------|--------|
| **v7.2 ML Integrated** | **Level Core + Pine Features** | **~53,791%** | **~75%** | 🏆 **NEW CHAMPION** |
| v6.2 Final | Level Detection (ML Core) | ~38,095% | 73.1% | 🥈 Strong Baseline |
| v7.1 Hybrid | Level Core + Pine Boost | ~24,966% | ~75% | 🥉 Good |
| v7.0 Retrofit | Pine Entry / Pine Exit | ~15,230% | 83.8% | ❌ Failed |

## Analysis

### Why v7.2 Won (The "ML Integrated" Approach)
Instead of hardcoding rules (v7.0) or probability boosts (v7.1), we fed the Pine Script signals (`BB_20_crossunder`, `DAILY_RETURN_PANIC`) directly into the ML model as **features**.

-   **Result**: The ML model learned that `DAILY_RETURN_PANIC` is highly predictive, ranking it as the **#7 most important feature**.
-   **Impact**: This allowed the model to dynamically weight the "panic" signal alongside its existing "level" signals, resulting in a massive performance jump to **~53,791%**.

### Why v7.0 Failed
The "Retrofit" replaced **Level Detection** (`%B < -0.1`) with **Crossover Detection** (`%B` crosses `-0.06`).
- **Problem**: The ML "Conviction Filter" relies on **accumulated counts** of extreme days. Crossovers only fire once, breaking this accumulation.

### Why v7.1 Hybrid Was Just "Okay"
The "Hybrid" used manual boosts. While better than v7.0, manual tuning is rarely as optimal as letting the Gradient Boosting model find the perfect weights itself.

## Final Verdict
**Switch to v7.2.**
The "ML Integrated" approach successfully combined the robustness of v6.2's level detection with the precision of Pine Script's panic signals. The ML model validated that the Pine Script signals *are* valuable, but only when integrated correctly as features.

## Files
- `bulletproof_strategy_v7_2_ml.py`: The **NEW CHAMPION**.
- `bulletproof_strategy_v6_final.py`: The previous baseline (archived).
