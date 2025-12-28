
import pandas as pd
import numpy as np
import yfinance as yf
import ephem
from datetime import datetime, timedelta

def get_eclipses(start_year=2000, end_year=2025):
    """Find all solar and lunar eclipses using New/Full moon + Latitude check."""
    solar_eclipses = []
    lunar_eclipses = []
    
    # Solar (New Moon + Lat < 1.5)
    d = ephem.Date(f'{start_year}-01-01')
    end_d = ephem.Date(f'{end_year}-12-31')
    
    while d < end_d:
        try:
            next_nm = ephem.next_new_moon(d)
            if next_nm > end_d:
                break
            
            # Check Latitude via Ecliptic wrapper
            m = ephem.Moon()
            m.compute(next_nm)
            ecl = ephem.Ecliptic(m)
            lat = abs(ecl.lat * 180 / 3.14159) # Convert to degrees
            
            # Solar eclipse limit approx 1.6 deg (standard limit)
            if lat < 1.6:
                solar_eclipses.append(next_nm.datetime().date())
                
            d = ephem.Date(next_nm + 1)
        except Exception as e:
            print(f"Error finding solar eclipse: {e}")
            break
            
    # Lunar (Full Moon + Lat < 1.5)
    d = ephem.Date(f'{start_year}-01-01')
    while d < end_d:
        try:
            next_fm = ephem.next_full_moon(d)
            if next_fm > end_d:
                break
                
            # Check Latitude via Ecliptic wrapper
            m = ephem.Moon()
            m.compute(next_fm)
            ecl = ephem.Ecliptic(m)
            lat = abs(ecl.lat * 180 / 3.14159)
            
            # Lunar eclipse limit approx 1.6 deg
            if lat < 1.6:
                lunar_eclipses.append(next_fm.datetime().date())
                
            d = ephem.Date(next_fm + 1)
        except Exception as e:
            print(f"Error finding lunar eclipse: {e}")
            break
            
    return sorted(list(set(solar_eclipses))), sorted(list(set(lunar_eclipses)))

def get_market_data():
    """Download SPY and VIX data."""
    print("Downloading Market Data...")
    spy = yf.download("SPY", start="2000-01-01", end="2025-12-31", progress=False)
    vix = yf.download("^VIX", start="2000-01-01", end="2025-12-31", progress=False)
    
    # Clean MultiIndex
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    df = pd.DataFrame(index=spy.index)
    df['Close'] = spy['Close']
    df['Return'] = df['Close'].pct_change()
    df['Abs_Return'] = df['Return'].abs()
    
    # Align VIX
    df['VIX'] = vix['Close']
    df['VIX_Spike'] = (df['VIX'] > 30).astype(int)
    
    df.dropna(inplace=True)
    return df

def analyze_impact(df, solar_dates, lunar_dates):
    """Tag dates and calculate stats."""
    
    # Convert dates to datetime for comparison
    solar_dates = pd.to_datetime(solar_dates)
    lunar_dates = pd.to_datetime(lunar_dates)
    
    # Helper to check window
    def in_window(date, event_dates, days=3):
        # Find closest event
        # This is strictly a vectorized check or apply
        # For simplicity in this script loop is fine or apply
        # Let's use a simpler method: broaden events to set
        pass # implemented below efficiently
    
    # Create mask arrays
    solar_mask_3 = np.zeros(len(df), dtype=bool)
    solar_mask_7 = np.zeros(len(df), dtype=bool)
    lunar_mask_3 = np.zeros(len(df), dtype=bool)
    lunar_mask_7 = np.zeros(len(df), dtype=bool)
    
    dates = df.index
    
    for eclipse_date in solar_dates:
        # +/- 3 days
        mask3 = (dates >= (eclipse_date - timedelta(days=3))) & (dates <= (eclipse_date + timedelta(days=3)))
        solar_mask_3 |= mask3
        # +/- 7 days
        mask7 = (dates >= (eclipse_date - timedelta(days=7))) & (dates <= (eclipse_date + timedelta(days=7)))
        solar_mask_7 |= mask7
        
    for eclipse_date in lunar_dates:
        # +/- 3 days
        mask3 = (dates >= (eclipse_date - timedelta(days=3))) & (dates <= (eclipse_date + timedelta(days=3)))
        lunar_mask_3 |= mask3
        # +/- 7 days
        mask7 = (dates >= (eclipse_date - timedelta(days=7))) & (dates <= (eclipse_date + timedelta(days=7)))
        lunar_mask_7 |= mask7
        
    df['Solar_7d'] = solar_mask_7
    df['Lunar_7d'] = lunar_mask_7
    df['Any_Eclipse_7d'] = solar_mask_7 | lunar_mask_7
    
    df['Solar_3d'] = solar_mask_3
    df['Lunar_3d'] = lunar_mask_3
    df['Any_Eclipse_3d'] = solar_mask_3 | lunar_mask_3
    
    # print stats
    print("\n" + "="*50)
    print(f"ECLIPSE IMPACT ANALYSIS (2000-2025)")
    print("="*50)
    print(f"Total Trading Days: {len(df)}")
    print(f"Solar Eclipses: {len(solar_dates)}")
    print(f"Lunar Eclipses: {len(lunar_dates)}")
    print("-" * 50)
    
    base_vix = df['VIX'].mean()
    base_ret = df['Abs_Return'].mean() * 100
    base_spike = df['VIX_Spike'].mean() * 100
    
    print(f"BASELINE (No Filter):")
    print(f"  Avg VIX:        {base_vix:.2f}")
    print(f"  Avg Abs Return: {base_ret:.3f}%")
    print(f"  VIX Spike Prob: {base_spike:.1f}%")
    
    groups = [
        ('Solar +/- 7d', df[df['Solar_7d']]),
        ('Lunar +/- 7d', df[df['Lunar_7d']]),
        ('Any Eclipse +/- 7d', df[df['Any_Eclipse_7d']]),
        ('Solar +/- 3d', df[df['Solar_3d']]),
        ('Lunar +/- 3d', df[df['Lunar_3d']]),
        ('Any Eclipse +/- 3d', df[df['Any_Eclipse_3d']]),
        ('NO Eclipse (Control)', df[~df['Any_Eclipse_7d']])
    ]
    
    for name, subset in groups:
        print("-" * 50)
        vix = subset['VIX'].mean()
        ret = subset['Abs_Return'].mean() * 100
        spike = subset['VIX_Spike'].mean() * 100
        
        vix_diff = (vix - base_vix) / base_vix * 100
        ret_diff = (ret - base_ret) / base_ret * 100
        
        print(f"REGIME: {name} ({len(subset)} days)")
        print(f"  Avg VIX:        {vix:.2f} ({vix_diff:+.1f}%)")
        print(f"  Avg Abs Return: {ret:.3f}% ({ret_diff:+.1f}%)")
        print(f"  VIX Spike Prob: {spike:.1f}%")

    # Check for Directionality Bias (Bullish vs Bearish)
    print("-" * 50)
    print("DIRECTIONAL BIAS (Win Rate)")
    base_wr = (df['Return'] > 0).mean() * 100
    print(f"  Baseline Win Rate: {base_wr:.1f}%")
    
    for name, subset in groups:
        if name == 'NO Eclipse (Control)': continue
        wr = (subset['Return'] > 0).mean() * 100
        print(f"  {name}: {wr:.1f}% (Delta: {wr-base_wr:+.1f}%)")

if __name__ == "__main__":
    solar, lunar = get_eclipses()
    df = get_market_data()
    analyze_impact(df, solar, lunar)
