"""
================================================================================
ML HOURLY STRATEGY (3 STD EDITION) 🤖
================================================================================

Trains a Random Forest model specifically on HOURLY data to "buy the dip".
Features include Bollinger Bands (2.0 and 3.0 STD) as requested.

Usage:
    python ml_hourly_strategy.py SPY NVDA

================================================================================
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import argparse
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


def safe_series(col) -> pd.Series:
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


class HourlyFeatureEngine:
    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = safe_series(df['Close'])
        features = pd.DataFrame(index=df.index)
        
        # 1. Bollinger Bands (2.0 STD) - Standard
        bb2 = ta.bbands(close, length=20, std=2.0)
        if bb2 is not None:
            # Handle different column naming conventions
            try:
                lower2 = bb2.iloc[:, 0]
                upper2 = bb2.iloc[:, 2]
                pctb2 = (close - lower2) / (upper2 - lower2)
            except:
                pctb2 = pd.Series(0, index=df.index)
            
            features['BB_2.0_pctb'] = pctb2
            features['BB_2.0_lower_hit'] = (pctb2 < 0).astype(int)
        
        # 2. Bollinger Bands (3.0 STD) - EXTREME DIP (User Request)
        bb3 = ta.bbands(close, length=20, std=3.0)
        if bb3 is not None:
            try:
                lower3 = bb3.iloc[:, 0]
                upper3 = bb3.iloc[:, 2]
                pctb3 = (close - lower3) / (upper3 - lower3)
            except:
                pctb3 = pd.Series(0, index=df.index)
            
            features['BB_3.0_pctb'] = pctb3
            features['BB_3.0_lower_hit'] = (pctb3 < 0).astype(int) # The "Crash" signal
            features['BB_3.0_deep_crash'] = (pctb3 < -0.1).astype(int)
            
        # 3. RSI
        features['RSI_14'] = ta.rsi(close, length=14)
        features['RSI_2'] = ta.rsi(close, length=2)
        
        # 4. Hourly Returns
        features['HOURLY_RET'] = close.pct_change(1)
        features['HOURLY_PANIC'] = (features['HOURLY_RET'] < -0.01).astype(int) # -1% in an hour
        
        # 5. Time Features (Crucial for Intraday)
        # Convert index to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
            
        features['HOUR'] = df.index.hour
        features['IS_MORNING'] = (features['HOUR'] < 11).astype(int)
        features['IS_CLOSING'] = (features['HOUR'] > 14).astype(int)
        
        return features.fillna(0)


def train_and_backtest(ticker: str):
    print(f"\n{'='*70}")
    print(f"ML HOURLY TRAINING: {ticker}")
    print(f"{'='*70}")
    
    # 1. Load Data
    print("Loading data (730 days, 1h)...")
    try:
        df = yf.download(ticker, period="730d", interval="1h", progress=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    if df.empty:
        print("No data found.")
        return
        
    print(f"Loaded {len(df)} bars.")
    
    # 2. Feature Engineering
    print("Calculating features (BB 2.0, BB 3.0, RSI)...")
    X = HourlyFeatureEngine.calculate(df)
    close = safe_series(df['Close'])
    
    # 3. Labeling (The "Truth")
    # We want to find dips that bounced.
    # Target: Return > 1.5% in next 12 hours AND Min Drawdown > -1.0% (Stop Loss)
    LOOK_AHEAD = 12
    STOP_LOSS = -0.01
    TARGET_PROFIT = 0.015
    
    y = pd.Series(0, index=df.index)
    
    for i in range(len(df) - LOOK_AHEAD):
        entry_price = float(close.iloc[i])
        future_window = close.iloc[i+1 : i+1+LOOK_AHEAD]
        
        max_price = future_window.max()
        min_price = future_window.min()
        
        max_ret = (max_price - entry_price) / entry_price
        min_ret = (min_price - entry_price) / entry_price
        
        if max_ret > TARGET_PROFIT and min_ret > STOP_LOSS:
            y.iloc[i] = 1
            
    print(f"Positive Labels (Profitable Dips): {y.sum()} ({y.mean():.1%})")
    
    # 4. Train/Test Split (Time-based)
    split_idx = int(len(df) * 0.6) # Train on first 60%, Test on last 40%
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    close_test = close.iloc[split_idx:]
    
    # 5. Train Model
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 5 Features:")
    print(importances.head(5))
    
    # 6. Backtest on Test Set
    print("\nBacktesting on Test Set (Unseen Data)...")
    probs = model.predict_proba(X_test)[:, 1]
    
    # Threshold Optimization (Simple loop)
    best_thresh = 0.5
    best_ret = -999
    
    # Simulation
    cash = 100000
    shares = 0
    entry_price = 0
    trade_log = []
    
    # Use a lower threshold to ensure trades
    THRESHOLD = 0.55
    print(f"Max Probability in Test Set: {probs.max():.4f}")
    
    for i in range(len(X_test)):
        date = X_test.index[i]
        price = float(close_test.iloc[i])
        prob = probs[i]
        
        # Entry
        if shares == 0:
            if prob > THRESHOLD:
                shares = cash / price
                cash = 0
                entry_price = price
                # print(f"BUY  @ {price:.2f} (Prob: {prob:.2f})")
        
        # Exit (Simple logic for backtest: Hold 12 hours or Stop Loss)
        # Actually, let's use the model's target logic: Exit if +1.5% or -1%
        elif shares > 0:
            ret = (price - entry_price) / entry_price
            
            if ret > TARGET_PROFIT or ret < STOP_LOSS:
                cash = shares * price
                shares = 0
                trade_log.append(ret)
                # print(f"SELL @ {price:.2f} (Ret: {ret:.2%})")
                
    # Final Value
    if shares > 0:
        cash = shares * close_test.iloc[-1]
    
    total_return = (cash - 100000) / 100000
    bh_return = (close_test.iloc[-1] - close_test.iloc[0]) / close_test.iloc[0]
    
    print(f"\nRESULTS: {ticker} (Test Set)")
    print(f"Strategy Return: {total_return*100:.2f}%")
    print(f"Buy & Hold:      {bh_return*100:.2f}%")
    print(f"Trades:          {len(trade_log)}")
    print(f"Win Rate:        {len([r for r in trade_log if r > 0]) / len(trade_log) if trade_log else 0:.1%}")
    
    with open('ml_hourly_results.txt', 'a') as f:
        f.write(f"TICKER: {ticker}\nRETURN: {total_return*100:.2f}%\nB&H: {bh_return*100:.2f}%\nTRADES: {len(trade_log)}\nTOP_FEAT: {importances.index[0]}\n---\n")

    # PLOTTING
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(close_test.index, close_test.values, label='Price', alpha=0.5)
        
        # Plot Buys
        buy_dates = []
        buy_prices = []
        for i in range(len(X_test)):
            if probs[i] > THRESHOLD:
                buy_dates.append(X_test.index[i])
                buy_prices.append(close_test.iloc[i])
        
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='ML Buy Signal')
        plt.title(f"SPY Hourly ML Trades (Return: {total_return*100:.1f}%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('spy_ml_trades.png')
        print("Chart saved to spy_ml_trades.png")
    except ImportError:
        print("matplotlib not installed, skipping plot.")


if __name__ == "__main__":
    # Hardcoded for SPY only as requested
    train_and_backtest('SPY')
