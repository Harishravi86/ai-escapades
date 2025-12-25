
import pandas as pd
import numpy as np
import yfinance as yf
import ephem
from datetime import datetime

# Ephemeris
planets = {
    'Sun': ephem.Sun(), 'Moon': ephem.Moon(), 'Mercury': ephem.Mercury(),
    'Venus': ephem.Venus(), 'Mars': ephem.Mars(), 'Jupiter': ephem.Jupiter(),
    'Saturn': ephem.Saturn(), 'Uranus': ephem.Uranus(), 'Neptune': ephem.Neptune(),
    'Pluto': ephem.Pluto()
}

def get_planet_positions(date_obj):
    obs = ephem.Observer()
    obs.date = date_obj.strftime('%Y/%m/%d')
    longitudes = []
    for name, planet in planets.items():
        planet.compute(obs)
        longitudes.append(np.degrees(planet.hlon) % 360)
    return sorted(longitudes)

def calculate_spread(longitudes):
    max_gap = 0
    n = len(longitudes)
    for i in range(n):
        gap = longitudes[(i + 1) % n] - longitudes[i]
        if gap < 0: gap += 360
        max_gap = max(max_gap, gap)
    return 360 - max_gap

def analyze():
    print("Fetching SPY (1993-2025)...")
    spy = yf.download('SPY', start='1993-01-01', end='2025-01-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    
    df = pd.DataFrame(index=spy.index)
    df['Close'] = spy['Close']
    df['Returns'] = df['Close'].pct_change()
    
    # METRICS
    # 1. Realized Volatility (Next 20 days)
    df['Future_Vol_20d'] = df['Returns'].rolling(20).std().shift(-20) * np.sqrt(252) * 100 # Annualized %
    
    # 2. Big Move Probability (Next 5 days)
    # Did we have a > 1% day in the next 5 days?
    df['Big_Move_Next_5d'] = df['Returns'].abs().rolling(5).max().shift(-5) > 0.015 # 1.5% Threshold
    
    # CALCULATE SPREADS
    print("Calculating planetary spreads...")
    spreads = []
    dates = df.index.to_pydatetime()
    for d in dates:
        lons = get_planet_positions(d)
        spreads.append(calculate_spread(lons))
    df['Spread'] = spreads
    df.dropna(inplace=True)
    
    # BINNING ANALYSIS
    # Bin size = 20 degrees
    bins = np.arange(0, 380, 20)
    df['Bin'] = pd.cut(df['Spread'], bins)
    
    print("\n--- VOLATILITY CURVE (Spread vs Risk) ---\n")
    print(f"{'Spread Range':<15} | {'Days':<6} | {'Avg Volularity':<15} | {'Big Move %':<12}")
    print("-" * 60)
    
    grouped = df.groupby('Bin', observed=True)
    
    risk_curve = []
    
    for name, group in grouped:
        count = len(group)
        if count < 50: continue # Skip noise
        
        avg_vol = group['Future_Vol_20d'].mean()
        prob_big = group['Big_Move_Next_5d'].mean() * 100
        
        print(f"{str(name):<15} | {count:<6} | {avg_vol:.2f}%          | {prob_big:.1f}%")
        
        risk_curve.append({
            'bin_mid': name.mid,
            'vol': avg_vol,
            'prob': prob_big
        })
        
    # TAIL RISK CHECK
    # Top 5% Tightest Days
    tight_threshold = df['Spread'].quantile(0.05)
    tight_data = df[df['Spread'] < tight_threshold]
    baseline_vol = df['Future_Vol_20d'].mean()
    tight_vol = tight_data['Future_Vol_20d'].mean()
    
    print("\n--- TAIL RISK SUMMARY ---")
    print(f"Tightest 5% Threshold: < {tight_threshold:.1f}°")
    print(f"Baseline Volatility:   {baseline_vol:.2f}%")
    print(f"Tight Cluster Vol:     {tight_vol:.2f}%")
    print(f"Risk Multiplier:       x{tight_vol/baseline_vol:.2f}")

    # Check for "Safe Zone"
    # Find the bin with LOWEST volatility
    safe_bin = min(risk_curve, key=lambda x: x['vol'])
    print(f"\nSAFEST ZONE: Spreads around {safe_bin['bin_mid']}° have lowest vol ({safe_bin['vol']:.2f}%)")

if __name__ == "__main__":
    analyze()
