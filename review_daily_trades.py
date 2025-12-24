import pandas as pd
import sys
import os

# Add current directory to path to allow import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bulletproof_strategy_v7_2_ml import BulletproofStrategyV72, load_data_yf

def review_trades():
    print("=" * 70)
    print("REVIEWING DAILY STRATEGY TRADES (SINCE JAN 1, 2010)")
    print("=" * 70)

    try:
        print("\nLoading data...")
        spy = load_data_yf("SPY")
        qqq = load_data_yf("QQQ")
        vix = load_data_yf("^VIX")
        
        # Intersect indices
        idx = spy.index.intersection(qqq.index).intersection(vix.index)
        spy = spy.loc[idx]
        qqq = qqq.loc[idx]
        vix = vix.loc[idx]
        
        # Filter for 2010 onwards
        start_date = "2010-01-01"
        spy = spy[spy.index >= start_date]
        qqq = qqq[qqq.index >= start_date]
        vix = vix[vix.index >= start_date]
        
        print(f"Loaded {len(spy)} trading days from {start_date}")
        
        print("\n" + "=" * 70)
        print("BACKTEST: Trading QQQ")
        print("=" * 70)
        
        # Initialize and run strategy
        strategy = BulletproofStrategyV72(spy, qqq, vix)
        
        # Train models first (using the filtered data - or should we train on all? 
        # Ideally train on all, but for this review, let's stick to the requested period to avoid lookahead bias if we were simulating strictly)
        # Actually, the strategy trains internally if not trained. 
        # Let's just run backtest.
        
        strategy.backtest(verbose=True) # Verbose=True to see trades
        
        strategy.print_results()
        
        # Post-process trade log for user requirements
        df_trades = pd.DataFrame(strategy.trade_log)
        
        # 1. Track $100,000 running capital
        initial_capital = 100000.0
        df_trades['equity_start'] = initial_capital
        
        current_equity = initial_capital
        equity_col = []
        
        for ret in df_trades['return']:
            # Apply return to current equity
            # Note: Strategy logic handles compounding, here we just replicate it for display
            # Simple compounding: New Equity = Old Equity * (1 + Return)
            current_equity = current_equity * (1 + ret)
            equity_col.append(current_equity)
            
        df_trades['running_equity'] = [f"${x:,.2f}" for x in equity_col]
        
        # 2. Format Return as % (String for display)
        # User requested "% xx.yy%" format. Adding explicit sign for clarity.
        df_trades['return_pct'] = df_trades['return'].apply(lambda x: f"{x*100:+.2f}%")
        
        # 3. Calculate Duration
        # Ensure dates are datetime objects
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        df_trades['duration_days'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
        
        # Reorder columns
        cols = ['entry_date', 'exit_date', 'duration_days', 'return_pct', 'running_equity', 'reason', 'partial_taken', 'return']
        df_trades = df_trades[cols]
        
        # Save formatted log
        log_file = "daily_trades_2010_formatted_v2.csv"
        df_trades.to_csv(log_file, index=False)
        print(f"\nFormatted trade log (with Duration & $100k tracking) saved to {log_file}")
        
        # Print first few rows to verify
        print("\nFirst 5 Trades:")
        print(df_trades[['entry_date', 'exit_date', 'duration_days', 'return_pct', 'running_equity']].head().to_string())
        print("\nLast 5 Trades:")
        print(df_trades[['entry_date', 'exit_date', 'duration_days', 'return_pct', 'running_equity']].tail().to_string())

    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    review_trades()
