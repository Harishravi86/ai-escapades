
import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime

def run_comparison():
    # 1. Fetch Data
    ticker = "SPY"
    print(f"Fetching data for {ticker}...")
    # Get enough data to cover 2010 to present
    df = yf.download(ticker, start="2009-01-01", interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        # Handle multiindex (Price, Ticker) -> just get Price
        try:
            df = df.xs(ticker, axis=1, level=1)
        except:
            if 'Close' not in df.columns:
                 # Attempt to flatten or just use what we have if it looks like single layer
                 pass

    # Ensure we have just the columns we need and dropna
    df = df.dropna()
    
    # 2. Re-calculate Strategy Signals
    length = 20
    std_dev = 2.0
    entry_threshold = -0.06
    daily_return_threshold = -0.0088

    # Indicators
    df['SMA'] = df['Close'].rolling(window=length).mean()
    df['STD'] = df['Close'].rolling(window=length).std()
    df['Upper'] = df['SMA'] + (std_dev * df['STD'])
    df['Lower'] = df['SMA'] - (std_dev * df['STD'])
    df['PercentB'] = (df['Close'] - df['Lower']) / (df['Upper'] - df['Lower'])
    df['DailyReturn'] = df['Close'].pct_change()
    df['PercentB_Prev'] = df['PercentB'].shift(1)
    
    # Filter for Data >= 2010-01-01 for the actual trades
    df_sim = df.loc["2010-01-01":].copy()
    
    # Strategy Signals on this subset 
    # (Note: rolling calculations from 2009 data ensure 2010 start is valid)
    strategy_mask = (
        (df_sim['PercentB'] < entry_threshold) & 
        (df_sim['PercentB_Prev'] >= entry_threshold) & 
        (df_sim['DailyReturn'] < daily_return_threshold)
    )
    
    strategy_dates = df_sim.index[strategy_mask]
    num_trades = len(strategy_dates)
    
    print(f"Strategy Trades since 2010: {num_trades}")
    
    if num_trades == 0:
        print("No trades found since 2010.")
        return

    # 3. Calculate Strategy Performance
    investment_per_trade = 5000
    final_price = df_sim['Close'].iloc[-1]
    
    def calculate_portfolio_value(dates, entry_investment, final_price_val):
        # Vectorized calculation
        # Prices at entry
        entry_prices = df_sim.loc[dates]['Close']
        shares = entry_investment / entry_prices
        total_shares = shares.sum()
        final_value = total_shares * final_price_val
        return final_value, total_shares

    strategy_final_value, strategy_shares = calculate_portfolio_value(strategy_dates, investment_per_trade, final_price)
    total_invested = num_trades * investment_per_trade
    strategy_return_pct = ((strategy_final_value - total_invested) / total_invested) * 100

    print("-" * 50)
    print(f"Strategy Results (Buy & Hold from Signals):")
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"Final Value:    ${strategy_final_value:,.2f}")
    print(f"Net Profit:     ${strategy_final_value - total_invested:,.2f}")
    print(f"Return:         {strategy_return_pct:.2f}%")
    print("-" * 50)

    # 4. Monte Carlo Simulation (Random Entries)
    # Pool of valid dates is all dates in df_sim
    valid_dates = df_sim.index.tolist()
    num_simulations = 1000
    random_values = []
    
    print(f"Running {num_simulations} random simulations...")
    
    for _ in range(num_simulations):
        # Pick 'num_trades' random dates
        # Use random.sample to pick without replacement (modelling unique entry days)
        # or replacement? usually "random entry" implies just picking dates. 
        # sample (no replacement) makes sense as you can't buy twice on same day in this model usually (unless doubled up)
        # Let's assume unique days for parity.
        random_dates = random.sample(valid_dates, num_trades)
        r_val, _ = calculate_portfolio_value(random_dates, investment_per_trade, final_price)
        random_values.append(r_val)
        
    random_values = np.array(random_values)
    avg_random_value = np.mean(random_values)
    median_random_value = np.median(random_values)
    min_random = np.min(random_values)
    max_random = np.max(random_values)
    
    # 5. Comparison
    outperformance = strategy_final_value - avg_random_value
    outperformance_pct = (outperformance / avg_random_value) * 100
    
    # Calculate Percentile
    # What % of random runs did strategy beat?
    beats = np.sum(strategy_final_value > random_values)
    win_rate = (beats / num_simulations) * 100

    print(f"\nRandom Simulation Results (Benchmark):")
    print(f"Average Final Value: ${avg_random_value:,.2f}")
    print(f"Median Final Value:  ${median_random_value:,.2f}")
    print(f"Max Random Value:    ${max_random:,.2f}")
    print(f"Min Random Value:    ${min_random:,.2f}")
    print("-" * 50)
    print(f"COMPARISON:")
    print(f"Strategy Outperformance ($): ${outperformance:,.2f}")
    print(f"Strategy Outperformance (%): {outperformance_pct:.2f}% better than average random timing")
    print(f"Strategy Win Rate (vs Random): {win_rate:.1f}%")
    print("-" * 50)

if __name__ == "__main__":
    run_comparison()
