#!/usr/bin/env python3
"""
================================================================================
RESEARCH: Day 5 of Month Effect
================================================================================

User's Hypothesis: The 5th of each month has special significance.
Notable dates:
- August 5, 2024: Japan Carry Trade Crash (-3% SPY)
- February 5, 2018: Volmageddon (-4.1% SPY)

Analysis: Does Day 5 (or nearby days) consistently produce larger moves?

================================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy import stats

def safe_series(col):
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col

def analyze_day_5():
    print("="*70)
    print("RESEARCH: Day 5 of Month Effect")
    print("="*70)
    
    # Download SPY data
    print("\nLoading SPY data (2000-2025)...")
    spy = yf.download("SPY", start="2000-01-01", end="2025-12-31", progress=False)
    
    close = safe_series(spy['Close'])
    spy['Return'] = close.pct_change() * 100
    spy['AbsReturn'] = spy['Return'].abs()
    spy['DayOfMonth'] = spy.index.day
    spy['Month'] = spy.index.month
    spy['Year'] = spy.index.year
    spy = spy.dropna()
    
    print(f"Total trading days: {len(spy)}")
    
    # ==========================================================================
    # ANALYSIS 1: Day 5 vs All Days
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: Day 5 vs All Days")
    print("="*70)
    
    day_5 = spy[spy['DayOfMonth'] == 5]['Return']
    all_days = spy['Return']
    
    print(f"\nDay 5 of Month:")
    print(f"  Count: {len(day_5)}")
    print(f"  Mean Return: {day_5.mean():.4f}%")
    print(f"  Mean Absolute Return: {spy[spy['DayOfMonth'] == 5]['AbsReturn'].mean():.4f}%")
    print(f"  Win Rate: {(day_5 > 0).sum() / len(day_5) * 100:.1f}%")
    print(f"  Std Dev: {day_5.std():.4f}%")
    
    print(f"\nAll Days:")
    print(f"  Mean Return: {all_days.mean():.4f}%")
    print(f"  Mean Absolute Return: {spy['AbsReturn'].mean():.4f}%")
    print(f"  Win Rate: {(all_days > 0).sum() / len(all_days) * 100:.1f}%")
    
    # T-test
    t_stat, p_value = stats.ttest_ind(day_5, all_days)
    print(f"\nT-Test (Day 5 vs All Days): p-value = {p_value:.4f}")
    
    # ==========================================================================
    # ANALYSIS 2: Volatility by Day of Month
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: Average Absolute Return by Day of Month")
    print("="*70)
    
    print(f"\n{'Day':>5} {'Count':>8} {'Mean %':>10} {'Abs Mean %':>12} {'Win Rate':>10}")
    print("-"*50)
    
    for day in range(1, 29):
        day_data = spy[spy['DayOfMonth'] == day]['Return']
        abs_data = spy[spy['DayOfMonth'] == day]['AbsReturn']
        if len(day_data) > 0:
            win_rate = (day_data > 0).sum() / len(day_data) * 100
            marker = " <--" if day == 5 else ""
            print(f"{day:>5} {len(day_data):>8} {day_data.mean():>10.4f} {abs_data.mean():>12.4f} {win_rate:>9.1f}%{marker}")
    
    # ==========================================================================
    # ANALYSIS 3: Notable Day 5 Events
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 3: Major Day 5 Events (|Return| > 2%)")
    print("="*70)
    
    major_day_5 = spy[(spy['DayOfMonth'] == 5) & (spy['AbsReturn'] > 2)]
    major_day_5_sorted = major_day_5.sort_values('Return')
    
    print(f"\nTop Day 5 Crashes:")
    for idx, row in major_day_5_sorted.head(10).iterrows():
        ret = float(row['Return'])
        print(f"  {idx.strftime('%Y-%m-%d')}: {ret:+.2f}%")
    
    print(f"\nTop Day 5 Rallies:")
    for idx, row in major_day_5_sorted.tail(10).iloc[::-1].iterrows():
        ret = float(row['Return'])
        print(f"  {idx.strftime('%Y-%m-%d')}: {ret:+.2f}%")

    
    # ==========================================================================
    # ANALYSIS 4: Day 5 Window (Day 4-6) Analysis
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 4: Day 5 Window (Days 4-6)")
    print("="*70)
    
    day_window = spy[spy['DayOfMonth'].isin([4, 5, 6])]['Return']
    day_5_only = spy[spy['DayOfMonth'] == 5]['Return']
    
    print(f"\nDay 4-6 Window:")
    print(f"  Mean Return: {day_window.mean():.4f}%")
    print(f"  Win Rate: {(day_window > 0).sum() / len(day_window) * 100:.1f}%")
    
    # Compare volatility
    vol_day_5 = day_5.std()
    vol_all = all_days.std()
    print(f"\nVolatility Comparison:")
    print(f"  Day 5 Std: {vol_day_5:.4f}%")
    print(f"  All Days Std: {vol_all:.4f}%")
    print(f"  Ratio: {vol_day_5/vol_all:.2f}x")
    
    # ==========================================================================
    # ANALYSIS 5: Day 5 After Oversold (Your Trading Thesis)
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 5: Day 5 After Big Drop (Reversal Opportunity)")
    print("="*70)
    
    # If previous day (Day 4) was a crash, what happens on Day 5?
    spy['Prev_Return'] = spy['Return'].shift(1)
    
    # Day 5 where previous day was down > 1%
    day_5_after_drop = spy[(spy['DayOfMonth'] == 5) & (spy['Prev_Return'] < -1)]
    day_5_after_big_drop = spy[(spy['DayOfMonth'] == 5) & (spy['Prev_Return'] < -2)]
    
    print(f"\nDay 5 after Day 4 dropped >1%:")
    print(f"  Count: {len(day_5_after_drop)}")
    if len(day_5_after_drop) > 0:
        print(f"  Mean Return: {day_5_after_drop['Return'].mean():.4f}%")
        print(f"  Win Rate: {(day_5_after_drop['Return'] > 0).sum() / len(day_5_after_drop) * 100:.1f}%")
    
    print(f"\nDay 5 after Day 4 dropped >2%:")
    print(f"  Count: {len(day_5_after_big_drop)}")
    if len(day_5_after_big_drop) > 0:
        print(f"  Mean Return: {day_5_after_big_drop['Return'].mean():.4f}%")
        print(f"  Win Rate: {(day_5_after_big_drop['Return'] > 0).sum() / len(day_5_after_big_drop) * 100:.1f}%")
    
    # ==========================================================================
    # ANALYSIS 6: Day 5 by Month (Seasonal Pattern?)
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 6: Day 5 Returns by Month")
    print("="*70)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print(f"\n{'Month':>6} {'Count':>8} {'Mean %':>10} {'Abs Mean %':>12} {'Win Rate':>10}")
    print("-"*55)
    
    for month in range(1, 13):
        month_data = spy[(spy['DayOfMonth'] == 5) & (spy['Month'] == month)]['Return']
        abs_month = spy[(spy['DayOfMonth'] == 5) & (spy['Month'] == month)]['AbsReturn']
        if len(month_data) > 0:
            win_rate = (month_data > 0).sum() / len(month_data) * 100
            print(f"{month_names[month-1]:>6} {len(month_data):>8} {month_data.mean():>10.4f} {abs_month.mean():>12.4f} {win_rate:>9.1f}%")
    
    # ==========================================================================
    # CONCLUSION
    # ==========================================================================
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    edge = day_5.mean() - all_days.mean()
    abs_edge = spy[spy['DayOfMonth'] == 5]['AbsReturn'].mean() - spy['AbsReturn'].mean()
    
    print(f"\n• Day 5 Return Edge: {edge:+.4f}% per day")
    print(f"• Day 5 Absolute Move Edge: {abs_edge:+.4f}% (higher volatility?)")
    print(f"• Statistical Significance: p={p_value:.4f}")
    
    if abs_edge > 0.1:
        print("\n✅ Day 5 has HIGHER VOLATILITY - more extreme moves")
        print("   This could explain your observation of big wins on Day 5!")
    
    return spy

if __name__ == "__main__":
    data = analyze_day_5()
