"""
SHARKTOOTH PATTERN REFERENCE
=============================

Your trusted "Sharktooth" concept: Extreme oversold → Recovery snap

This file documents ALL the sharktooth patterns the ML model learns from.

WHAT IS A SHARKTOOTH?
---------------------
A sharktooth is a washout pattern where:
1. An indicator hits extreme oversold territory
2. The indicator then starts recovering (turning up)

This creates a "shark tooth" shape on the chart - sharp point down, then bounce up.

KEY INSIGHT:
When MULTIPLE indicators show sharktooth patterns simultaneously,
the probability of a genuine bottom increases dramatically.
"""

# =============================================================
# SHARKTOOTH DEFINITIONS BY INDICATOR
# =============================================================

SHARKTOOTH_PATTERNS = {
    
    # -----------------------------------------------------------
    # RSI SHARKTOOTH
    # -----------------------------------------------------------
    "RSI": {
        "oversold_threshold": 30,
        "extreme_threshold": 20,
        "sharktooth_condition": "RSI was < 30 yesterday AND RSI is rising today",
        "strong_sharktooth": "RSI was < 20 yesterday AND RSI is rising today",
        "formula": "(RSI[t-1] < 30) AND (RSI[t] > RSI[t-1])",
        "timeframes_used": [5, 7, 14, 21],
        "interpretation": """
            RSI < 30 = oversold
            RSI < 20 = extremely oversold
            Sharktooth = oversold + turning up = potential bounce
        """,
    },
    
    # -----------------------------------------------------------
    # %B (BOLLINGER BANDS) SHARKTOOTH - YOUR PRIMARY SIGNAL
    # -----------------------------------------------------------
    "PERCENT_B": {
        "oversold_threshold": 0,
        "extreme_threshold": -0.10,
        "washout_threshold": -0.20,
        "sharktooth_condition": "%B was < 0 yesterday AND %B is rising today",
        "strong_sharktooth": "%B was < -0.10 yesterday AND %B is rising today",
        "formula": "(%B[t-1] < 0) AND (%B[t] > %B[t-1])",
        "settings_used": [
            {"length": 10, "std": 2.0},
            {"length": 20, "std": 2.0},
            {"length": 20, "std": 2.5},
            {"length": 28, "std": 2.61},  # Your optimized setting
        ],
        "interpretation": """
            %B < 0 = price below lower Bollinger Band (oversold)
            %B < -0.10 = significantly below lower band
            %B < -0.20 = extreme washout
            
            YOUR SHARKTOOTH: %B goes deeply negative, then starts recovering.
            This is the pattern you trust - the "washout bounce".
            
            Multiple BB settings capture different volatility regimes.
        """,
    },
    
    # -----------------------------------------------------------
    # MACD SHARKTOOTH
    # -----------------------------------------------------------
    "MACD": {
        "oversold_threshold": "histogram < 0",
        "sharktooth_condition": "Histogram was negative AND histogram improving",
        "bullish_cross": "MACD line crosses above signal line",
        "formula": "(Histogram[t-1] < 0) AND (Histogram[t] > Histogram[t-1])",
        "settings_used": [
            {"fast": 8, "slow": 21, "signal": 9},
            {"fast": 12, "slow": 26, "signal": 9},  # Standard
            {"fast": 5, "slow": 35, "signal": 5},   # Longer-term
        ],
        "interpretation": """
            Negative histogram = bearish momentum
            Histogram improving = momentum shift
            Bullish cross = trend reversal signal
            
            MACD sharktooth = momentum was bearish, now improving
        """,
    },
    
    # -----------------------------------------------------------
    # STOCHASTIC SHARKTOOTH
    # -----------------------------------------------------------
    "STOCHASTIC": {
        "oversold_threshold": 20,
        "extreme_threshold": 10,
        "sharktooth_condition": "%K was < 20 yesterday AND %K is rising",
        "formula": "(StochK[t-1] < 20) AND (StochK[t] > StochK[t-1])",
        "settings_used": [
            {"k": 5, "d": 3},   # Fast
            {"k": 14, "d": 3},  # Standard
            {"k": 21, "d": 5},  # Slow
        ],
        "interpretation": """
            Stochastic < 20 = oversold
            Stochastic < 10 = extremely oversold
            
            Sharktooth = was oversold, now turning up
        """,
    },
    
    # -----------------------------------------------------------
    # WILLIAMS %R SHARKTOOTH
    # -----------------------------------------------------------
    "WILLIAMS_R": {
        "oversold_threshold": -80,
        "extreme_threshold": -90,
        "sharktooth_condition": "%R was < -80 yesterday AND %R is rising",
        "formula": "(WillR[t-1] < -80) AND (WillR[t] > WillR[t-1])",
        "timeframes_used": [10, 14, 21],
        "interpretation": """
            Williams %R < -80 = oversold (note: scale is -100 to 0)
            Williams %R < -90 = extremely oversold
            
            Similar to Stochastic but inverted scale
        """,
    },
    
    # -----------------------------------------------------------
    # CCI SHARKTOOTH
    # -----------------------------------------------------------
    "CCI": {
        "oversold_threshold": -100,
        "extreme_threshold": -200,
        "sharktooth_condition": "CCI was < -100 yesterday AND CCI is rising",
        "formula": "(CCI[t-1] < -100) AND (CCI[t] > CCI[t-1])",
        "timeframes_used": [10, 20, 40],
        "interpretation": """
            CCI < -100 = oversold
            CCI < -200 = extreme oversold (rare)
            
            CCI measures deviation from statistical mean
            Extreme readings often precede reversals
        """,
    },
    
    # -----------------------------------------------------------
    # MFI (MONEY FLOW INDEX) SHARKTOOTH
    # -----------------------------------------------------------
    "MFI": {
        "oversold_threshold": 20,
        "extreme_threshold": 10,
        "sharktooth_condition": "MFI was < 20 yesterday AND MFI is rising",
        "formula": "(MFI[t-1] < 20) AND (MFI[t] > MFI[t-1])",
        "timeframes_used": [10, 14, 21],
        "interpretation": """
            MFI = Volume-weighted RSI
            MFI < 20 = oversold with selling volume exhaustion
            
            MFI sharktooth = selling pressure exhausting, buyers stepping in
            More reliable than RSI because it incorporates volume
        """,
    },
}


# =============================================================
# COMPOSITE SHARKTOOTH SCORING
# =============================================================

COMPOSITE_SCORING = """
SHARKTOOTH COUNT INTERPRETATION
================================

The ML model counts how many indicators show sharktooth patterns:

Count | Interpretation | Action
------|----------------|--------
0-1   | Normal market  | No signal
2-3   | Mild oversold  | Watch closely
4-5   | Significant    | Consider entry
6-8   | Strong washout | High probability bottom
9+    | Extreme        | Rare, very high conviction

OVERSOLD COUNT vs SHARKTOOTH COUNT
===================================

OVERSOLD_COUNT = How many indicators are currently oversold
SHARKTOOTH_COUNT = How many indicators were oversold AND are now recovering

Key insight: SHARKTOOTH_COUNT is more actionable because it shows
the TURN is happening, not just that conditions are bad.

Example:
- Day 1: OVERSOLD_COUNT = 8, SHARKTOOTH_COUNT = 0 (still falling)
- Day 2: OVERSOLD_COUNT = 6, SHARKTOOTH_COUNT = 5 (SHARKTOOTH SIGNAL)

Day 2 is the entry point - the turn has begun.
"""


# =============================================================
# ML FEATURE IMPORTANCE (TYPICAL RESULTS)
# =============================================================

TYPICAL_IMPORTANT_FEATURES = """
TOP FEATURES FROM ML TRAINING (typical results)
================================================

Based on historical training, these features typically rank highest:

1. DD_20d (20-day drawdown)
   - How far price has fallen from 20-day high
   - Deeper drawdowns = higher bottom probability

2. SHARKTOOTH_COUNT
   - Number of indicators showing sharktooth pattern
   - Your composite signal

3. BB_28_2_61_sharktooth  
   - Your optimized %B setting showing sharktooth
   - The pattern you trust most

4. RSI_14_sharktooth
   - Classic RSI showing recovery from oversold

5. WASHOUT_SCORE
   - Composite of oversold + extreme + volume
   - Higher = more washed out

6. OVERSOLD_COUNT
   - Raw count of oversold indicators
   
7. STOCH_14_3_sharktooth
   - Stochastic showing recovery

8. CCI_20_sharktooth
   - CCI showing recovery

9. MFI_14_sharktooth
   - Volume-weighted RSI showing recovery

10. VOL_ratio
    - Volume vs average
    - High volume on sharktooth = conviction
"""


# =============================================================
# SIGNAL INTERPRETATION GUIDE
# =============================================================

SIGNAL_GUIDE = """
HOW TO INTERPRET THE ML OUTPUT
==============================

1. ML Bottom Probability (0-100%)
   - < 30%: No signal, normal market
   - 30-45%: Elevated odds, watch for confirmation
   - 45-55%: Moderate signal, consider small entry
   - 55-70%: Strong signal, full entry
   - > 70%: Very strong, rare occurrence

2. Sharktooth Count
   - Think of this as "confirmation count"
   - More sharktooths = more confidence

3. Key Indicator Readings
   - RSI_14: Classic momentum (< 30 oversold)
   - %B: Your trusted signal (< 0 oversold)
   - DD_20d: How much carnage (< -5% significant)

ENTRY DECISION MATRIX
=====================

ML Prob | Sharktooth | Celestial | VIX  | Action
--------|------------|-----------|------|--------
> 50%   | >= 3       | Any       | Any  | FULL ENTRY
> 45%   | >= 4       | Any       | Any  | FULL ENTRY
> 40%   | >= 2       | Moon Opp  | <28  | ENTRY (quiet bottom)
> 40%   | >= 5       | Any       | Any  | ENTRY
< 40%   | >= 6       | Any       | Any  | SMALL ENTRY
< 40%   | < 3        | Any       | Any  | NO ENTRY

Remember: High Sharktooth Count can override lower ML probability
because it represents your trusted pattern across multiple indicators.
"""


if __name__ == "__main__":
    print("=" * 60)
    print("SHARKTOOTH PATTERN REFERENCE")
    print("=" * 60)
    
    print("\nSHARKTOOTH PATTERNS DETECTED:")
    print("-" * 40)
    
    for indicator, details in SHARKTOOTH_PATTERNS.items():
        print(f"\n{indicator}:")
        print(f"  Oversold at: {details['oversold_threshold']}")
        print(f"  Pattern: {details['sharktooth_condition']}")
    
    print("\n" + COMPOSITE_SCORING)
    print(TYPICAL_IMPORTANT_FEATURES)
    print(SIGNAL_GUIDE)
