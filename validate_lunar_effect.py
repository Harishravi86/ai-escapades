
import pandas as pd
import numpy as np
import ephem
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SYMBOLS = ['SPY', 'IWM', 'QQQ']
WINDOW_DAYS_BEFORE = 2
WINDOW_DAYS_AFTER = 2
PLACEBO_TRIALS = 1000

# ==============================================================================
# 1. DATA LOADING & PROCESSING
# ==============================================================================
def load_data(symbol):
    fname = f"{symbol}_data.csv"
    base_dir = r"c:\Users\jhana\trading_research\spy-trading"
    path = f"{base_dir}\\{fname}"
    print(f"Loading data for {symbol} from {path}...")
    
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None

    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    # Calculate Returns
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    return df

# ==============================================================================
# 2. CELESTIAL CALCULATOR (Isolated Lunar Phase)
# ==============================================================================
def calculate_moon_phases(dates):
    print("Calculating Moon Phases (this takes a moment)...")
    phases = []
    
    obs = ephem.Observer()
    sun = ephem.Sun()
    moon = ephem.Moon()
    
    # Cache ephem objects to avoid re-instantiation if possible, 
    # but the loop overhead is main cost.
    
    for d in dates:
        # Mid-day calculation
        dt = d + timedelta(hours=12) 
        obs.date = dt
        sun.compute(obs)
        moon.compute(obs)
        
        # Ecliptic Longitude
        s_lon = ephem.Ecliptic(sun).lon
        m_lon = ephem.Ecliptic(moon).lon
        
        # Phase Angle (0 to 2pi)
        sep = m_lon - s_lon
        while sep < 0: sep += 2*np.pi
        while sep >= 2*np.pi: sep -= 2*np.pi
        
        # Convert to degrees (0 = New, 90 = 1st Q, 180 = Full, 270 = 3rd Q)
        deg = sep * 180.0 / np.pi
        phases.append(deg)
        
    return np.array(phases)

# ==============================================================================
# 3. STRATEGY LOGIC
# ==============================================================================
def get_lunar_window_mask(dates, phases, window_pre, window_post):
    """
    Returns a boolean mask for trading days within [NewMoon - pre, NewMoon + post].
    New Moon is defined as the day where phase resets from ~360 to ~0.
    """
    n = len(dates)
    mask = np.zeros(n, dtype=bool)
    
    # Identify New Moon Days: simpler robust method
    # 0 degrees is New Moon.
    # We look for phase < 15 or phase > 345 (approx 1 day window) effectively
    # finding the minimum.
    
    # Rigorous: Find local minima of phase (or wrap-around points)
    # Wrap around: Phase goes from 350 -> 10.
    
    new_moon_indices = []
    
    for i in range(1, n):
        # Check for wrap-around (New Moon moment)
        if phases[i-1] > 300 and phases[i] < 60:
            # The crossing happened between yesterday and today.
            # Which day is closer to 0?
            dist_prev = 360 - phases[i-1]
            dist_curr = phases[i]
            
            if dist_curr < dist_prev:
                new_moon_indices.append(i)
            else:
                new_moon_indices.append(i-1)
                
    # Mark windows
    for idx_nm in new_moon_indices:
        start = max(0, idx_nm - window_pre)
        end = min(n, idx_nm + window_post + 1) # +1 for slice inclusive
        mask[start:end] = True
        
    return mask

# ==============================================================================
# 4. STATISTICAL VALIDATION
# ==============================================================================
def run_validation():
    print("="*60)
    print(f"MULTI-ASSET LUNAR VALIDATION (Option A)")
    print("="*60)

    for symbol in SYMBOLS:
        # Load
        df = load_data(symbol)
        if df is None: continue
        
        dates = df.index
        
        # Moon
        phases = calculate_moon_phases(dates)
        
        # Generate Strategy Mask
        mask_strategy = get_lunar_window_mask(dates, phases, WINDOW_DAYS_BEFORE, WINDOW_DAYS_AFTER)
        df['In_Window'] = mask_strategy
        
        # --- PERFORMANCE METRICS ---
        
        # 1. Total Returns
        df['Strategy_Ret'] = np.where(df['In_Window'], df['Return'], 0.0)
        
        # Cumulative Growth
        df['Cum_BnH'] = (1 + df['Return']).cumprod()
        df['Cum_Strategy'] = (1 + df['Strategy_Ret']).cumprod()
        
        total_ret_bnh = df['Cum_BnH'].iloc[-1] - 1
        total_ret_strat = df['Cum_Strategy'].iloc[-1] - 1
        
        # 2. Time in Market
        days_market = len(df)
        days_strat = df['In_Window'].sum()
        exposure_pct = days_strat / days_market
        
        # 3. Hypothesis Test (T-Test)
        rets_lunar = df[df['In_Window']]['Return']
        rets_other = df[~df['In_Window']]['Return']
        
        t_stat, p_val = stats.ttest_ind(rets_lunar, rets_other, equal_var=False)
        
        # 4. Placebo Test (Randomized Control)
        print(f"\n[{symbol}] Running {PLACEBO_TRIALS} Placebo Trials...")
        placebo_returns = []
        
        # We randomize the *starting points* of the new moons, preserving the frequency
        num_moons = len([x for x in range(1, len(phases)) if phases[x-1] > 300 and phases[x] < 60])
        
        for _ in range(PLACEBO_TRIALS):
            # Generate random indices for "Fake New Moons"
            fake_indices = np.random.choice(len(df), size=num_moons, replace=False)
            fake_mask = np.zeros(len(df), dtype=bool)
            
            for idx in fake_indices:
                start = max(0, idx - WINDOW_DAYS_BEFORE)
                end = min(len(df), idx + WINDOW_DAYS_AFTER + 1)
                fake_mask[start:end] = True
                
            # Calc return
            fake_ret_series = np.where(fake_mask, df['Return'], 0.0)
            fake_total_ret = (1 + fake_ret_series).prod() - 1
            placebo_returns.append(fake_total_ret)
            
        placebo_returns = np.array(placebo_returns)
        
        # Placebo Stats
        p_val_monte_carlo = (placebo_returns >= total_ret_strat).mean()
        
        # ==========================================================================
        # REPORTING
        # ==========================================================================
        print("-" * 60)
        print(f"REPORT: {symbol}")
        print("-" * 60)
        print(f"Data Range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        
        print(f"Buy & Hold Return:      {total_ret_bnh:.2%}")
        print(f"Lunar Strategy Return:  {total_ret_strat:.2%}")
        print(f"Market Exposure:        {exposure_pct:.1%}")
        
        print("HYPOTHESIS TEST (T-Test)")
        print(f"Avg Daily Ret (Lunar):  {rets_lunar.mean():.4%}")
        print(f"Avg Daily Ret (Other):  {rets_other.mean():.4%}")
        print(f"T-Statistic:            {t_stat:.4f}")
        print(f"P-Value (Two-Sided):    {p_val:.4f}")
        
        print(f"PLACEBO TEST ({PLACEBO_TRIALS} Trials)")
        print(f"Placebo Mean Return:    {placebo_returns.mean():.2%}")
        print(f"Confirming Significance:{(1-p_val_monte_carlo)*100:.1f}% > 95%?")
        print(f"Monte Carlo P-Value:    {p_val_monte_carlo:.4f}")
        
        sig_mc = "VALIDATED" if p_val_monte_carlo < 0.05 else "FAILED"
        print(f"Result for {symbol}: {sig_mc}")
        print("-" * 60)

if __name__ == "__main__":
    run_validation()
