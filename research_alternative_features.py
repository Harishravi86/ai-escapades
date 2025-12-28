
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

def load_data():
    print("Loading Market Data...")
    tickers = ['SPY', '^VIX', '^VIX3M']
    data = yf.download(tickers, start='2000-01-01', progress=False)
    
    # Handle MultiIndex columns if present (yfinance update)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Try to simplify if "Price" level exists
            if 'Close' in data.columns.levels[0]:
                df = data['Close']
            elif 'Price' in data.columns.names:
                df = data.xs('Close', level='Price', axis=1)
            else:
                # Fallback: just take Close
                df = data['Close']
        except:
             df = data['Close']
    else:
        df = data['Close']
        
    return df

def analyze_vix_term_structure(df):
    print("\n==================================================")
    print("ANALYSIS 1: VIX TERM STRUCTURE (^VIX / ^VIX3M)")
    print("==================================================")
    
    # Filter for valid VIX3M data (starts ~2006)
    vix_df = df[['^VIX', '^VIX3M', 'SPY']].dropna().copy()
    
    # Calculate Ratio
    # Ratio > 1.0 = Backwardation (Spot > 3M) = Panic/Fear
    # Ratio < 1.0 = Contango (Spot < 3M) = Normal/Complacency
    vix_df['Ratio'] = vix_df['^VIX'] / vix_df['^VIX3M']
    
    # Calculate Forward Returns for SPY
    vix_df['Ret_1d'] = vix_df['SPY'].pct_change().shift(-1) # Return for TOMORROW
    vix_df['Ret_5d'] = vix_df['SPY'].pct_change(5).shift(-5)
    vix_df['Ret_20d'] = vix_df['SPY'].pct_change(20).shift(-20)
    
    # Classify Regime
    # Using 1.0 as the strict theoretical boundary
    vix_df['Regime'] = np.where(vix_df['Ratio'] > 1.0, 'Backwardation (Fear)', 'Contango (Calm)')
    
    # Group stats
    stats = vix_df.groupby('Regime')[['Ret_1d', 'Ret_5d', 'Ret_20d']].mean() * 100
    counts = vix_df.groupby('Regime')['Ret_1d'].count()
    
    print(f"Data Date Range: {vix_df.index.min().date()} to {vix_df.index.max().date()}")
    print("\nMean Forward Returns (%):")
    print(stats)
    print("\nObservations:")
    print(counts)
    
    # Win Rate (Positive Returns)
    wr_1d = vix_df.groupby('Regime')['Ret_1d'].apply(lambda x: (x > 0).mean()) * 100
    print("\nWin Rate 1-Day Forward (%):")
    print(wr_1d)
    print("==================================================\n")
    
    return vix_df

def analyze_fomc(df):
    print("\n==================================================")
    print("ANALYSIS 2: FOMC MEETING EFFECT")
    print("==================================================")
    
    try:
        fomc = pd.read_csv('fomc_dates.csv')
        fomc_dates = pd.to_datetime(fomc['Date']).dt.date
        fomc_set = set(fomc_dates)
        print(f"Loaded {len(fomc_set)} FOMC meeting dates.")
    except Exception as e:
        print(f"Error loading FOMC dates: {e}")
        return

    spy = df[['SPY']].copy()
    spy['Return'] = spy['SPY'].pct_change()
    
    # Flags
    # Day 0 = FOMC Date
    # Day -1 = Pre-FOMC Drift
    spy['Date'] = spy.index.date
    spy['Is_FOMC'] = spy['Date'].isin(fomc_set)
    
    # Find Pre-FOMC days (shift -1 because Is_FOMC is on Day 0, we want to flag the day BEFORE)
    # Actually, easiest way is to reindex or use boolean logic
    # Day -1: The trading day BEFORE an FOMC day.
    # We can perform a join.
    
    # Create a mapping of Date -> Type
    # Default 'Normal'
    spy['Day_Type'] = 'Normal'
    
    # Label FOMC Days
    spy.loc[spy['Is_FOMC'], 'Day_Type'] = 'FOMC Day (Event)'
    
    # Label Pre-FOMC Days
    # Valid Pre-FOMC day is one where the NEXT trading day is FOMC
    # Shift Is_FOMC backwards by 1 (Day T is Pre if T+1 is FOMC) -> shift(-1) looks forward
    is_pre_fomc = spy['Is_FOMC'].shift(-1).fillna(False)
    spy.loc[is_pre_fomc, 'Day_Type'] = 'Pre-FOMC (Drift)'
    
    # Group stats
    stats = spy.groupby('Day_Type')['Return'].describe()[['count', 'mean', '50%']] # 50% is median
    stats['mean'] = stats['mean'] * 100
    stats['50%'] = stats['50%'] * 100 # Median
    
    # Win Rate
    win_rate = spy.groupby('Day_Type')['Return'].apply(lambda x: (x > 0).mean() * 100)
    stats['Win Rate %'] = win_rate
    
    print("\nDaily Returns by Day Type (%):")
    print(stats.sort_values('mean', ascending=False))


def analyze_seasonality(df):
    print("\n==================================================")
    print("ANALYSIS 3: SEASONALITY (Halloween & Turn-of-Month)")
    print("==================================================")
    
    spy = df[['SPY']].copy()
    spy['Return'] = spy['SPY'].pct_change()
    spy['Month'] = spy.index.month
    spy['Day'] = spy.index.day
    
    # 1. Halloween Effect (Nov-Apr vs May-Oct)
    # Crypto/Tech often calls this "Sell in May"
    spy['Period'] = np.where(spy['Month'].isin([11,12,1,2,3,4]), 'Halloween (Nov-Apr)', 'Summer (May-Oct)')
    
    print("\n--- Halloween Effect ---")
    h_stats = spy.groupby('Period')['Return'].agg(['count', 'mean'])
    h_stats['mean'] = h_stats['mean'] * 100
    h_stats['Annualized %'] = h_stats['mean'] * 252
    print(h_stats)
    
    # 2. Turn of Month
    # Last 3 days + First 2 days
    # We need to identify business month ends.
    # Simple logic: Is day > 25 (potential end) or < 4 (start)?
    # Better: Use pandas calculated business days.
    
    # Determine "Days from Month End"
    # This is tricky without a custom calendar, but we can approximate.
    # Day > 26 or Day < 4 covers most effects.
    # Let's use the user's definition: "Last 3 + First 2"
    # Logic: If Day <= 2, match. If Day >= last_day - 3?
    
    # Vectorized check for Month End
    # Group by Year-Month, find max day.
    
    spy['YearMonth'] = spy.index.to_period('M')
    last_days = spy.groupby('YearMonth')['Day'].transform('max')
    
    # Logic:
    # First 2 days: Day <= 2
    # Last 3 days: Day >= (last_days - 2)
    
    spy['Is_TOM'] = (spy['Day'] <= 2) | (spy['Day'] >= (last_days - 2))
    spy['TOM_Label'] = np.where(spy['Is_TOM'], 'Turn of Month', 'Rest of Month')
    
    print("\n--- Turn of Month (Last 3 + First 2 Days) ---")
    t_stats = spy.groupby('TOM_Label')['Return'].agg(['count', 'mean'])
    t_stats['mean'] = t_stats['mean'] * 100
    t_stats['Win Rate %'] = spy.groupby('TOM_Label')['Return'].apply(lambda x: (x > 0).mean() * 100)
    print(t_stats)
    
    # 3. September Effect
    print("\n--- September Effect ---")
    spy['Is_Sept'] = np.where(spy['Month'] == 9, 'September', 'Other Months')
    s_stats = spy.groupby('Is_Sept')['Return'].agg(['count', 'mean'])
    s_stats['mean'] = s_stats['mean'] * 100
    print(s_stats)


if __name__ == "__main__":
    df = load_data()
    
    # Run Analyses
    analyze_vix_term_structure(df)
    analyze_fomc(df)
    analyze_seasonality(df)
