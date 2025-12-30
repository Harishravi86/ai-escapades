#!/usr/bin/env python3
"""
================================================================================
RESEARCH: First Monday of Month Effect
================================================================================

Analyzing if the first Monday of each month has a statistically significant edge.
This is similar to "Turn of Month" effect but specifically targeting Mondays.

Key Questions:
1. Does the first Monday of the month outperform other Mondays?
2. Does it outperform all days?
3. Is there a specific "window" around first Monday that works best?
4. Can this be used as a v8.0 feature?

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

def analyze_first_monday():
    print("="*70)
    print("RESEARCH: First Monday of Month Effect")
    print("="*70)
    
    # Download SPY data
    print("\nLoading SPY data (2000-2025)...")
    spy = yf.download("SPY", start="2000-01-01", end="2025-12-31", progress=False)
    
    close = safe_series(spy['Close'])
    spy['Return'] = close.pct_change() * 100
    spy['DayOfWeek'] = spy.index.dayofweek  # 0=Monday, 4=Friday
    spy['DayOfMonth'] = spy.index.day
    spy['Month'] = spy.index.month
    spy['Year'] = spy.index.year
    
    # Create flags
    spy['Is_Monday'] = (spy['DayOfWeek'] == 0).astype(int)
    
    # First Monday of Month = Monday where day <= 7
    spy['Is_First_Monday'] = ((spy['DayOfWeek'] == 0) & (spy['DayOfMonth'] <= 7)).astype(int)
    
    # First trading day of month
    spy['Is_First_Day'] = (spy.groupby(['Year', 'Month']).cumcount() == 0).astype(int)
    
    # Second Monday of Month
    spy['Is_Second_Monday'] = ((spy['DayOfWeek'] == 0) & (spy['DayOfMonth'] > 7) & (spy['DayOfMonth'] <= 14)).astype(int)
    
    spy = spy.dropna()

    
    print(f"Total trading days: {len(spy)}")
    print(f"Date range: {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")
    
    # ==========================================================================
    # ANALYSIS 1: Compare Mean Returns
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: Mean Daily Returns by Category")
    print("="*70)
    
    all_days = spy['Return']
    all_mondays = spy[spy['Is_Monday'] == 1]['Return']
    first_mondays = spy[spy['Is_First_Monday'] == 1]['Return']
    second_mondays = spy[spy['Is_Second_Monday'] == 1]['Return']
    first_days = spy[spy['Is_First_Day'] == 1]['Return']
    
    categories = {
        'All Days': all_days,
        'All Mondays': all_mondays,
        'First Monday of Month': first_mondays,
        'Second Monday of Month': second_mondays,
        'First Trading Day': first_days,
    }
    
    print(f"\n{'Category':<30} {'Count':>8} {'Mean %':>10} {'Median %':>10} {'Std %':>10} {'Win Rate':>10}")
    print("-"*80)
    
    for name, data in categories.items():
        mean = data.mean()
        median = data.median()
        std = data.std()
        win_rate = (data > 0).sum() / len(data) * 100
        print(f"{name:<30} {len(data):>8} {mean:>10.4f} {median:>10.4f} {std:>10.3f} {win_rate:>9.1f}%")
    
    # ==========================================================================
    # ANALYSIS 2: Statistical Tests
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: T-Tests (First Monday vs Baseline)")
    print("="*70)
    
    # First Monday vs All Days
    t_stat, p_value = stats.ttest_ind(first_mondays, all_days)
    print(f"\nFirst Monday vs All Days:")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")
    
    # First Monday vs All Mondays
    t_stat2, p_value2 = stats.ttest_ind(first_mondays, all_mondays)
    print(f"\nFirst Monday vs All Mondays:")
    print(f"  T-statistic: {t_stat2:.4f}")
    print(f"  P-value: {p_value2:.4f}")
    print(f"  Significant (p<0.05): {'YES' if p_value2 < 0.05 else 'NO'}")
    
    # First Monday vs Second Monday
    t_stat3, p_value3 = stats.ttest_ind(first_mondays, second_mondays)
    print(f"\nFirst Monday vs Second Monday:")
    print(f"  T-statistic: {t_stat3:.4f}")
    print(f"  P-value: {p_value3:.4f}")
    print(f"  Significant (p<0.05): {'YES' if p_value3 < 0.05 else 'NO'}")
    
    # ==========================================================================
    # ANALYSIS 3: Annualized Returns (What's the edge?)
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 3: Estimated Annualized Edge")
    print("="*70)
    
    avg_days = 252
    first_mondays_per_year = 12
    
    for name, data in categories.items():
        # Annualize based on frequency
        if 'Monday' in name:
            # ~52 Mondays per year for "All Mondays"
            # ~12 first Mondays per year for "First Monday"
            if name == 'All Mondays':
                freq = 52
            elif 'First' in name or 'Second' in name:
                freq = 12
            else:
                freq = 12
        else:
            freq = avg_days
            
        mean_return = data.mean()
        annualized = mean_return * freq
        print(f"{name:<30}: {annualized:>8.2f}% annualized (freq={freq})")
    
    # ==========================================================================
    # ANALYSIS 4: Day of Week Effect (Is Monday special?)
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 4: Day of Week Effect")
    print("="*70)
    
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    print(f"\n{'Day':<15} {'Count':>8} {'Mean %':>10} {'Win Rate':>10}")
    print("-"*50)
    
    for dow in range(5):
        data = spy[spy['DayOfWeek'] == dow]['Return']
        mean = data.mean()
        win_rate = (data > 0).sum() / len(data) * 100
        print(f"{dow_names[dow]:<15} {len(data):>8} {mean:>10.4f} {win_rate:>9.1f}%")
    
    # ==========================================================================
    # ANALYSIS 5: First Week of Month Effect
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 5: First Week of Month (Days 1-5) vs Rest")
    print("="*70)
    
    first_week = spy[spy['DayOfMonth'] <= 5]['Return']
    rest_of_month = spy[spy['DayOfMonth'] > 5]['Return']
    
    print(f"\nFirst Week (Days 1-5):")
    print(f"  Count: {len(first_week)}")
    print(f"  Mean Return: {first_week.mean():.4f}%")
    print(f"  Win Rate: {(first_week > 0).sum() / len(first_week) * 100:.1f}%")
    
    print(f"\nRest of Month (Days 6+):")
    print(f"  Count: {len(rest_of_month)}")
    print(f"  Mean Return: {rest_of_month.mean():.4f}%")
    print(f"  Win Rate: {(rest_of_month > 0).sum() / len(rest_of_month) * 100:.1f}%")
    
    edge = first_week.mean() - rest_of_month.mean()
    print(f"\n**EDGE (First Week - Rest): {edge:.4f}% per day**")
    
    t_stat4, p_value4 = stats.ttest_ind(first_week, rest_of_month)
    print(f"P-value: {p_value4:.4f} ({'SIGNIFICANT' if p_value4 < 0.05 else 'NOT SIGNIFICANT'})")
    
    # ==========================================================================
    # ANALYSIS 6: Monthly First Monday by Decade
    # ==========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 6: First Monday Effect by Decade")
    print("="*70)
    
    decades = [
        ('2000s', 2000, 2009),
        ('2010s', 2010, 2019),
        ('2020s', 2020, 2025),
    ]
    
    print(f"\n{'Decade':<10} {'FM Count':>10} {'FM Mean':>10} {'All Days':>10} {'Edge':>10}")
    print("-"*55)
    
    for name, start, end in decades:
        decade_data = spy[(spy['Year'] >= start) & (spy['Year'] <= end)]
        fm = decade_data[decade_data['Is_First_Monday'] == 1]['Return']
        all_d = decade_data['Return']
        
        edge = fm.mean() - all_d.mean()
        print(f"{name:<10} {len(fm):>10} {fm.mean():>10.4f} {all_d.mean():>10.4f} {edge:>+10.4f}")
    
    # ==========================================================================
    # CONCLUSION
    # ==========================================================================
    print("\n" + "="*70)
    print("CONCLUSION: v8.0 Feature Recommendation")
    print("="*70)
    
    fm_edge = first_mondays.mean() - all_days.mean()
    fw_edge = first_week.mean() - rest_of_month.mean()
    
    print(f"\n• First Monday Edge: {fm_edge:+.4f}% per day")
    print(f"• First Week Edge: {fw_edge:+.4f}% per day")
    
    if p_value < 0.10:
        print("\n✅ RECOMMENDATION: Add 'is_first_monday' feature to v8.0")
        print("   The effect has marginal statistical significance.")
    else:
        print("\n⚠️ CAUTION: First Monday effect NOT statistically significant")
        print("   Consider using broader 'first_week' feature instead.")
    
    return spy

if __name__ == "__main__":
    data = analyze_first_monday()
