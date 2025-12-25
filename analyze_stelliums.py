
import pandas as pd
import numpy as np
import ephem
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURATION ---
OUTPUT_FILE = 'stellium_analysis_full.csv'

# Ephemeris setup
planets = {
    'Sun': ephem.Sun(),
    'Moon': ephem.Moon(),
    'Mercury': ephem.Mercury(),
    'Venus': ephem.Venus(),
    'Mars': ephem.Mars(),
    'Jupiter': ephem.Jupiter(),
    'Saturn': ephem.Saturn(),
    'Uranus': ephem.Uranus(),
    'Neptune': ephem.Neptune(),
    'Pluto': ephem.Pluto()
}

def get_planet_positions(date_obj): # Changed to take date object
    obs = ephem.Observer()
    obs.date = date_obj.strftime('%Y/%m/%d')
    longitudes = []
    for name, planet in planets.items():
        planet.compute(obs)
        lon = np.degrees(planet.hlon) % 360 
        longitudes.append(lon)
    return sorted(longitudes)

def calculate_spread(longitudes):
    max_gap = 0
    n = len(longitudes)
    for i in range(n):
        current = longitudes[i]
        next_val = longitudes[(i + 1) % n]
        gap = next_val - current
        if gap < 0: gap += 360
        if gap > max_gap: max_gap = gap
    return 360 - max_gap

def process_data():
    print("Fetching full history from yfinance (1990-2025)...")
    try:
        spy = yf.download('SPY', start='1993-01-01', end='2025-01-01', progress=False)
        vix = yf.download('^VIX', start='1993-01-01', end='2025-01-01', progress=False)
        
        # yfinance multi-index fix
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
        
        df = pd.DataFrame(index=spy.index)
        df['SPY'] = spy['Close']
        df['VIX'] = vix['Close'] # VIX might have different index, will align on join
        df = df.dropna()

        # Metrics
        df['Next_20D_Return'] = df['SPY'].pct_change(20).shift(-20)
        
        print("Calculating planetary usage for entire history...")
        
        results = []
        dates = df.index.to_pydatetime() # Faster iteration
        
        total = len(dates)
        for i, date_obj in enumerate(dates):
            if i % 1000 == 0:
                print(f"Processing {i}/{total} ({date_obj.date()})...")
                
            lons = get_planet_positions(date_obj)
            spread = calculate_spread(lons)
            
            results.append({
                'Date': date_obj,
                'Spread': spread
            })
            
        astro_df = pd.DataFrame(results)
        astro_df.set_index('Date', inplace=True)
        
        final_df = df.join(astro_df)
        
        # --- ANALYSIS ---
        print("\n--- ANALYSIS RESULTS (FULL HISTORY) ---\n")
        
        # Percentiles
        spread_10pct = final_df['Spread'].quantile(0.10)
        spread_90pct = final_df['Spread'].quantile(0.90)
        print(f"Tightest 10% Spread (Cluster): < {spread_10pct:.1f}°")
        print(f"Widest 10% Spread (Dispersed): > {spread_90pct:.1f}°")

        final_df['Is_Tight'] = final_df['Spread'] < spread_10pct
        final_df['Is_Wide'] = final_df['Spread'] > spread_90pct

        # 1. Correlations
        corr_vix = final_df['Spread'].corr(final_df['VIX'])
        print(f"Correlation (Spread vs VIX): {corr_vix:.4f} (Negative = Tighter is Volatile)")
        
        # 2. VIX Impact
        vix_all = final_df['VIX'].mean()
        vix_tight = final_df[final_df['Is_Tight']]['VIX'].mean()
        vix_wide = final_df[final_df['Is_Wide']]['VIX'].mean()
        
        print(f"\nAverage VIX (All Time): {vix_all:.2f}")
        print(f"Average VIX (Tightest 10%): {vix_tight:.2f} (Delta: {((vix_tight/vix_all)-1)*100:.1f}%)")
        print(f"Average VIX (Widest 10%):  {vix_wide:.2f} (Delta: {((vix_wide/vix_all)-1)*100:.1f}%)")

        # 3. Market Drop Probability
        # Did SPY crash in the next 20 days?
        final_df['Crash_Next_20D'] = final_df['Next_20D_Return'] < -0.05 # 5% drop
        prob_all = final_df['Crash_Next_20D'].mean()
        prob_tight = final_df[final_df['Is_Tight']]['Crash_Next_20D'].mean()
        
        print(f"\nProbability of >5% Drop in next 20 Days:")
        print(f"  Baseline: {prob_all*100:.1f}%")
        print(f"  During Tight Clusters: {prob_tight*100:.1f}%")

        # 4. Key Dates
        print("\n--- KEY DATES CHECK ---\n")
        check_dates = ['2000-05-05', '2008-09-15', '2020-01-12', '2020-03-20']
        
        for d_str in check_dates:
            try:
                target = pd.to_datetime(d_str)
                idx = final_df.index.get_indexer([target], method='nearest')[0]
                row = final_df.iloc[idx]
                dist_days = abs((row.name - target).days)
                
                print(f"Date: {d_str} (Found: {row.name.date()}, diff {dist_days}d)")
                print(f"  Planetary Spread: {row['Spread']:.1f}°")
                if row['Is_Tight']: print("  ** TIGHT CLUSTER DAY **")
                print(f"  VIX: {row['VIX']:.2f}")
                print(f"  Next 20D Return: {row['Next_20D_Return']*100:.1f}%")
                print("-" * 30)
            except: pass

        final_df.to_csv(OUTPUT_FILE)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_data()
