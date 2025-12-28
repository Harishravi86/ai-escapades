
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timedelta
from celestial_engine import CelestialEngine
from scipy import stats

def get_data():
    """Download SPY data."""
    print("Downloading SPY data...")
    df = yf.download("SPY", start="2000-01-01", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['Return'] = df['Close'].pct_change()
    df['NextReturn'] = df['Return'].shift(-1)
    df.dropna(inplace=True)
    return df

def get_dignity_score(sign_index):
    """
    +1 = Libra (6)
    -1 = Aries (0)
     0 = Others
    """
    if sign_index == 6: return 1
    if sign_index == 0: return -1
    return 0

def add_sign_features(df):
    """Add Both Tropical and Sidereal Signs."""
    print("Calculating Saturn Signs (Tropical & Sidereal)...")
    engine = CelestialEngine()
    
    tropical_signs = []
    sidereal_signs = []
    
    dates = df.index
    
    # Approx Lahiri Ayanamsa for 2000-2024
    # 2000: ~23.85
    # 2024: ~24.18
    # Using 24.0 as a robust approximation
    AYANAMSA = 24.0 
    
    for i, d in enumerate(dates):
        # 1. Get raw longitude (Tropical)
        # Re-using the logic from engine for speed
        if isinstance(d, pd.Timestamp):
            d_obj = d.to_pydatetime()
        else:
            d_obj = d
            
        lons = engine.get_planet_positions(d_obj) 
        # Wait, get_planet_positions returns all sorted lons. 
        # We need specific Saturn lon.
        # Let's use the internal calculation to be fast
        
        obs = engine.planets['Saturn']
        # Actually need to set date on Observer
        # This is getting messy, let's just instantiate a new Observer/Body manually for speed logic
        # OR just call engine.get_planet_sign and adjust manually
        
        # Tropical sign index (0-11)
        trop_idx = engine.get_planet_sign(d_obj, 'Saturn')
        tropical_signs.append(trop_idx)
        
        # Sidereal calculation
        # We need degrees first. 
        # engine.get_planet_sign does int(lon // 30)
        # We need lon. 
        # Let's peek into engine internals or just recalc here.
        
        import ephem
        obs_e = ephem.Observer()
        obs_e.date = d_obj.strftime('%Y/%m/%d')
        sat = ephem.Saturn()
        sat.compute(obs_e)
        lon_trop_deg = np.degrees(ephem.Ecliptic(sat).lon)
        
        lon_sid_deg = (lon_trop_deg - AYANAMSA) % 360
        sid_idx = int(lon_sid_deg // 30)
        sidereal_signs.append(sid_idx)
        
    df['Saturn_Sign_Tropical'] = tropical_signs
    df['Saturn_Sign_Sidereal'] = sidereal_signs
    
    # Calculate Dignities
    df['Dignity_Tropical'] = df['Saturn_Sign_Tropical'].apply(get_dignity_score)
    df['Dignity_Sidereal'] = df['Saturn_Sign_Sidereal'].apply(get_dignity_score)
    
    return df

def compare_performance(df):
    """Compare pure signal strength."""
    print("\n" + "="*50)
    print("SIDEREAL vs TROPICAL SHOWDOWN")
    print("="*50)
    
    print(f"Total Days: {len(df)}")
    
    # Tropical Stats
    trop_exalted = df[df['Dignity_Tropical'] == 1]
    trop_debilit = df[df['Dignity_Tropical'] == -1]
    avg_t_ex = trop_exalted['NextReturn'].mean() * 100
    avg_t_db = trop_debilit['NextReturn'].mean() * 100
    
    print(f"\nTROPICAL (Western):")
    print(f"Exalted (Libra) Count: {len(trop_exalted)} | Avg Return: {avg_t_ex:+.4f}%")
    print(f"Debilit (Aries) Count: {len(trop_debilit)} | Avg Return: {avg_t_db:+.4f}%")
    print(f"Spread (Edge): {avg_t_ex - avg_t_db:+.4f}%")
    
    # Sidereal Stats
    sid_exalted = df[df['Dignity_Sidereal'] == 1]
    sid_debilit = df[df['Dignity_Sidereal'] == -1]
    avg_s_ex = sid_exalted['NextReturn'].mean() * 100
    avg_s_db = sid_debilit['NextReturn'].mean() * 100
    
    print(f"\nSIDEREAL (Vedic -24°):")
    print(f"Exalted (Libra) Count: {len(sid_exalted)} | Avg Return: {avg_s_ex:+.4f}%")
    print(f"Debilit (Aries) Count: {len(sid_debilit)} | Avg Return: {avg_s_db:+.4f}%")
    print(f"Spread (Edge): {avg_s_ex - avg_s_db:+.4f}%")
    
    # Winner?
    trop_edge = avg_t_ex - avg_t_db
    sid_edge = avg_s_ex - avg_s_db
    
    print("\n" + "-"*30)
    if sid_edge > trop_edge:
        print(f"WINNER: SIDEREAL (+{sid_edge - trop_edge:.4f}% better spread)")
    else:
        print(f"WINNER: TROPICAL (+{trop_edge - sid_edge:.4f}% better spread)")
    print("-"*30)

    # ML Importance Check
    print("\nChecking ML Preference...")
    features = ['Dignity_Tropical', 'Dignity_Sidereal']
    target = (df['NextReturn'] > 0).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=2, random_state=42)
    model.fit(df[features], target)
    
    print("Feature Importance:")
    print(pd.DataFrame({'Feature': features, 'Imp': model.feature_importances_}))

if __name__ == "__main__":
    df = get_data()
    df = add_sign_features(df)
    compare_performance(df)
