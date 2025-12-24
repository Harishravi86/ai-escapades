"""
Moon-Uranus Opposition Hypothesis Test
======================================
Tests whether Moon-Uranus oppositions have predictive power for SPY bottoms.

Run locally where yfinance works:
    python moon_uranus_backtest.py
"""

import ephem
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

def calculate_oppositions(start_year=2015, end_year=2025):
    """Find all Moon-Uranus oppositions (180°) with hourly precision"""
    print(f"Calculating Moon-Uranus Oppositions ({start_year}-{end_year})...")
    
    start_dt = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, 12, 31)
    
    obs = ephem.Observer()
    moon = ephem.Moon()
    uranus = ephem.Uranus()
    
    opposition_dates = []
    found_event = False
    
    total_hours = int((end_dt - start_dt).total_seconds() / 3600)
    
    for i in range(total_hours):
        t = start_dt + timedelta(hours=i)
        obs.date = t
        moon.compute(obs)
        uranus.compute(obs)
        
        m_lon = math.degrees(ephem.Ecliptic(moon).lon)
        u_lon = math.degrees(ephem.Ecliptic(uranus).lon)
        
        sep = abs(m_lon - u_lon)
        if sep > 180: sep = 360 - sep
        
        if abs(sep - 180) < 1.0:
            if not found_event:
                opposition_dates.append(t.date())
                found_event = True
        elif found_event and abs(sep - 180) > 5.0:
            found_event = False
            
    return sorted(list(set(opposition_dates)))


def run_hypothesis_test():
    # Get all opposition dates
    opp_dates = calculate_oppositions(2015, 2025)
    print(f"Found {len(opp_dates)} opposition events\n")
    
    # Get SPY data
    print("Fetching SPY data...")
    df = yf.download("SPY", start="2015-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Forward returns
    df['Fwd_1d_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Fwd_3d_Return'] = df['Close'].shift(-3) / df['Close'] - 1
    df['Fwd_5d_Return'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # RSI(2)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_2'] = 100 - (100 / (1 + rs))
    
    # Below 20-day MA
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Below_MA'] = df['Close'] < df['MA20']
    
    # Match opposition dates to trading days
    signal_results = []
    
    for d in opp_dates:
        d_ts = pd.Timestamp(d)
        
        for offset in [0, 1, -1]:
            check_date = d_ts + timedelta(days=offset)
            if check_date in df.index:
                row = df.loc[check_date]
                signal_results.append({
                    'date': check_date,
                    'close': float(row['Close']),
                    'rsi2': float(row['RSI_2']),
                    'below_ma': bool(row['Below_MA']),
                    'fwd_1d': float(row['Fwd_1d_Return']) if pd.notna(row['Fwd_1d_Return']) else np.nan,
                    'fwd_3d': float(row['Fwd_3d_Return']) if pd.notna(row['Fwd_3d_Return']) else np.nan,
                    'fwd_5d': float(row['Fwd_5d_Return']) if pd.notna(row['Fwd_5d_Return']) else np.nan,
                })
                break
    
    results_df = pd.DataFrame(signal_results).dropna()
    
    # ================================================================
    # RESULTS
    # ================================================================
    print("=" * 70)
    print("MOON-URANUS OPPOSITION BACKTEST (2015-2025)")
    print("=" * 70)
    print(f"\nTotal Opposition Events: {len(results_df)}")
    
    # Base rates
    base_1d = df['Fwd_1d_Return'].mean()
    base_3d = df['Fwd_3d_Return'].mean()
    base_5d = df['Fwd_5d_Return'].mean()
    base_win_3d = (df['Fwd_3d_Return'] > 0).mean()
    
    # Signal rates
    sig_1d = results_df['fwd_1d'].mean()
    sig_3d = results_df['fwd_3d'].mean()
    sig_5d = results_df['fwd_5d'].mean()
    sig_win_3d = (results_df['fwd_3d'] > 0).mean()
    
    print(f"\n{'Metric':<25} | {'Base Rate':>12} | {'Signal Rate':>12} | {'Edge':>10}")
    print("-" * 65)
    print(f"{'Avg 1-Day Return':<25} | {base_1d:>11.3%} | {sig_1d:>11.3%} | {sig_1d - base_1d:>+9.3%}")
    print(f"{'Avg 3-Day Return':<25} | {base_3d:>11.3%} | {sig_3d:>11.3%} | {sig_3d - base_3d:>+9.3%}")
    print(f"{'Avg 5-Day Return':<25} | {base_5d:>11.3%} | {sig_5d:>11.3%} | {sig_5d - base_5d:>+9.3%}")
    print(f"{'Win Rate (3d > 0)':<25} | {base_win_3d:>11.1%} | {sig_win_3d:>11.1%} | {sig_win_3d - base_win_3d:>+9.1%}")
    
    # ================================================================
    # CONDITIONAL: Opposition + Below MA (Your "Antigravity" Setup)
    # ================================================================
    print("\n" + "=" * 70)
    print("CONDITIONAL: Opposition + Below 20-MA (Antigravity Setup)")
    print("=" * 70)
    
    filtered = results_df[results_df['below_ma'] == True]
    if len(filtered) >= 10:
        f_3d = filtered['fwd_3d'].mean()
        f_5d = filtered['fwd_5d'].mean()
        f_win = (filtered['fwd_3d'] > 0).mean()
        
        print(f"Sample Size: {len(filtered)} events")
        print(f"Avg 3-Day Return: {f_3d:.3%} (vs base {base_3d:.3%}) | Edge: {f_3d - base_3d:+.3%}")
        print(f"Avg 5-Day Return: {f_5d:.3%} (vs base {base_5d:.3%}) | Edge: {f_5d - base_5d:+.3%}")
        print(f"Win Rate: {f_win:.1%} (vs base {base_win_3d:.1%})")
    else:
        print(f"Only {len(filtered)} events - insufficient sample")
    
    # ================================================================
    # CONDITIONAL: Opposition + RSI(2) < 30
    # ================================================================
    print("\n" + "=" * 70)
    print("CONDITIONAL: Opposition + RSI(2) < 30 (Oversold)")
    print("=" * 70)
    
    filtered_rsi = results_df[results_df['rsi2'] < 30]
    if len(filtered_rsi) >= 10:
        fr_3d = filtered_rsi['fwd_3d'].mean()
        fr_5d = filtered_rsi['fwd_5d'].mean()
        fr_win = (filtered_rsi['fwd_3d'] > 0).mean()
        
        print(f"Sample Size: {len(filtered_rsi)} events")
        print(f"Avg 3-Day Return: {fr_3d:.3%} | Edge: {fr_3d - base_3d:+.3%}")
        print(f"Avg 5-Day Return: {fr_5d:.3%} | Edge: {fr_5d - base_5d:+.3%}")
        print(f"Win Rate: {fr_win:.1%}")
    else:
        print(f"Only {len(filtered_rsi)} events - insufficient sample")
    
    # ================================================================
    # STATISTICAL SIGNIFICANCE (Simple t-test)
    # ================================================================
    from scipy import stats
    
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    
    # Compare signal days vs random sample of same size
    random_sample = df['Fwd_3d_Return'].dropna().sample(n=len(results_df), random_state=42)
    t_stat, p_value = stats.ttest_ind(results_df['fwd_3d'], random_sample)
    
    print(f"T-Statistic: {t_stat:.3f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("RESULT: Statistically significant at 95% confidence")
    elif p_value < 0.10:
        print("RESULT: Marginally significant (90% confidence)")
    else:
        print("RESULT: NOT statistically significant - could be random noise")
    
    # ================================================================
    # RECENT EVENTS (2024-2025)
    # ================================================================
    print("\n" + "=" * 70)
    print("RECENT OPPOSITION EVENTS (2024-2025)")
    print("=" * 70)
    
    recent = results_df[results_df['date'] >= '2024-01-01']
    for _, row in recent.iterrows():
        marker = "✓✓" if row['fwd_3d'] > 0.02 else ("✓" if row['fwd_3d'] > 0 else "✗")
        below = "↓MA" if row['below_ma'] else "   "
        print(f"{row['date'].strftime('%Y-%m-%d')} | RSI2: {row['rsi2']:5.1f} | {below} | 3d: {row['fwd_3d']:+6.2%} | 5d: {row['fwd_5d']:+6.2%} | {marker}")
    
    return results_df


if __name__ == "__main__":
    results = run_hypothesis_test()