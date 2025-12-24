
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def run_strategy():
    # Parameters
    ticker = "SPY"
    length = 20
    std_dev = 2.0
    entry_threshold = -0.06
    # exit_threshold = 1.0  # Not used for entry signal
    # extreme_overbought = 1.2 # Not used for entry signal
    daily_return_threshold = -0.0088

    print(f"Fetching data for {ticker}...")
    # Fetch data (enough history for moving averages)
    # Using 'max' to get full history or a significant period like 20 years
    df = yf.download(ticker, period="max", interval="1d", progress=False)

    if df.empty:
        print("Error: No data fetched.")
        return

    # Ensure 'Close' is 1D array if MultiIndex (common issue with new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[1] else df
        # Or sometimes it's level 0 if only one ticker. 
        # Let's handle standard single ticker download which usually returns 'Close', 'Open' etc directly
        # But recently yf might return MultiIndex even for single ticker.
        # simpler check:
        if 'Close' not in df.columns and ticker in df.columns:
             # Just in case columns are just Ticker names? No, usually it's Price Type -> Ticker
             pass 
             
    # Clean up column names just in case
    # If standard download for single ticker, columns are 'Open', 'High', 'Low', 'Close', ...
    # If yf returns MultiIndex with ticker: ('Close', 'SPY')
    if isinstance(df.columns, pd.MultiIndex):
        try:
             df = df.xs(ticker, axis=1, level=1)
        except:
             # Fallback: maybe level 0?
             if ticker in df.columns.levels[0]:
                 df = df[ticker]
    
    # Calculate Bollinger Bands
    # Middle Band = 20-day SMA
    df['SMA'] = df['Close'].rolling(window=length).mean()
    # STD
    df['STD'] = df['Close'].rolling(window=length).std()
    
    # Upper/Lower
    df['Upper'] = df['SMA'] + (std_dev * df['STD'])
    df['Lower'] = df['SMA'] - (std_dev * df['STD'])
    
    # Percent B: (Close - Lower) / (Upper - Lower)
    df['PercentB'] = (df['Close'] - df['Lower']) / (df['Upper'] - df['Lower'])
    
    # Daily Return: (Close - PrevClose) / PrevClose
    # Pine's close[1] is the previous bar's close.
    df['DailyReturn'] = df['Close'].pct_change()
    
    # Previous Percent B
    df['PercentB_Prev'] = df['PercentB'].shift(1)
    
    # Entry Condition Logic
    # entry_condition = percent_b < entry_threshold and percent_b[1] >= entry_threshold and daily_return < daily_return_threshold
    
    # Create a boolean series
    buy_signals = (
        (df['PercentB'] < entry_threshold) & 
        (df['PercentB_Prev'] >= entry_threshold) & 
        (df['DailyReturn'] < daily_return_threshold)
    )
    
    # Filter dates
    buy_dates = df.index[buy_signals]
    
    print(f"\n Strategy Analysis for {ticker} (TV %B Strategy)")
    print(f" Parameters: Length={length}, Std={std_dev}, Entry<%B={entry_threshold}, Return<%={daily_return_threshold}")
    print("-" * 60)
    print(f"Total Trading Days: {len(df)}")
    print(f"Total Buy Signals: {len(buy_dates)}")
    print("-" * 60)
    print("Buy Signal Dates:")
    
    with open("buy_dates.txt", "w") as f:
        for date in buy_dates:
            date_str = date.strftime("%Y-%m-%d")
            print(date_str)
            f.write(date_str + "\n")
    
    print(f"\n[Info] Buy dates saved to buy_dates.txt")
        
    print("-" * 60)
    
    # Check for recent alert
    # If the last bar (or recent bars) triggered entry
    if not buy_dates.empty:
        last_signal = buy_dates[-1]
        last_date_in_data = df.index[-1]
        
        # Check if the last signal was "today" or very recent (last row of data)
        # Note: In backtesting, we see all historical dates.
        # For an "Alert", we care if the *latest* available data point triggered it.
        
        if last_signal == last_date_in_data:
            print(f"!!! ALERT: BUY SIGNAL TRIGGERED ON {last_signal.strftime('%Y-%m-%d')} !!!")
            # This is where we would send the actual alert
            print("Sending alert...") 
        else:
            print(f"No signal on the last data date ({last_date_in_data.strftime('%Y-%m-%d')}). Last signal was: {last_signal.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    run_strategy()
