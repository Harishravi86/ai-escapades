"""
================================================================================
HOURLY STRATEGY BACKTESTER (v7.2 Logic)
================================================================================

Tests the v7.2 strategy on HOURLY (1h) data.
Note: yfinance limits 1h data to the last 730 days (~2 years).

Usage:
    python backtest_hourly.py SPY NVDA

================================================================================
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import argparse
from datetime import datetime
from typing import Dict

import warnings
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


class TechnicalEngine:
    """Adapted for Hourly Data"""
    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = safe_series(df['Close'])
        high = safe_series(df['High'])
        low = safe_series(df['Low'])
        volume = safe_series(df['Volume'])
        features = pd.DataFrame(index=df.index)
        
        # RSI
        for length in [2, 5, 14, 21, 50]:
            rsi = ta.rsi(close, length=length)
            features[f'RSI_{length}'] = rsi
            features[f'RSI_{length}_oversold'] = (rsi < 30).astype(int)
            features[f'RSI_{length}_extreme'] = (rsi < 20).astype(int)
            features[f'RSI_{length}_overbought'] = (rsi > 70).astype(int)
        
        # Bollinger Bands
        for length in [20, 50]:
            bb = ta.bbands(close, length=length, std=2.0)
            if bb is not None:
                col_pctb = f'BBP_{length}_2.0'
                if col_pctb not in bb.columns:
                    lower_col = [c for c in bb.columns if 'BBL' in c][0]
                    upper_col = [c for c in bb.columns if 'BBU' in c][0]
                    lower = bb[lower_col]
                    upper = bb[upper_col]
                    pctb = (close - lower) / (upper - lower)
                else:
                    pctb = bb[col_pctb]
                
                features[f'BB_{length}_pctb'] = pctb
                features[f'BB_{length}_oversold'] = (pctb < 0).astype(int)
                features[f'BB_{length}_sharktooth'] = (pctb < -0.1).astype(int)
                features[f'BB_{length}_overbought'] = (pctb > 1).astype(int)
                features[f'BB_{length}_sharktooth_bear'] = (pctb > 1.1).astype(int)
                
                features[f'BB_{length}_crossunder'] = ((pctb < -0.06) & (pctb.shift(1) >= -0.06)).astype(int)
                features[f'BB_{length}_crossover'] = ((pctb > 1.0) & (pctb.shift(1) <= 1.0)).astype(int)
        
        # MACD
        macd = ta.macd(close)
        if macd is not None:
            features['MACD_line'] = macd.iloc[:, 0]
            features['MACD_hist'] = macd.iloc[:, 1]
            features['MACD_oversold'] = (features['MACD_line'] < -2.0).astype(int)
            features['MACD_overbought'] = (features['MACD_line'] > 2.0).astype(int)
        
        # Stochastic
        stoch = ta.stoch(high, low, close)
        if stoch is not None:
            k = stoch.iloc[:, 0]
            d = stoch.iloc[:, 1]
            features['STOCH_k'] = k
            features['STOCH_oversold'] = (k < 20).astype(int)
            features['STOCH_sharktooth'] = ((k < 20) & (k > d)).astype(int)
            features['STOCH_overbought'] = (k > 80).astype(int)
            features['STOCH_sharktooth_bear'] = ((k > 80) & (k < d)).astype(int)
        
        # Williams %R
        for length in [14, 28]:
            willr = ta.willr(high, low, close, length=length)
            features[f'WILLR_{length}'] = willr
            features[f'WILLR_{length}_oversold'] = (willr < -80).astype(int)
            features[f'WILLR_{length}_overbought'] = (willr > -20).astype(int)
        
        # CCI
        cci = ta.cci(high, low, close, length=20)
        cci = cci.clip(-500, 500) if cci is not None else pd.Series(0, index=df.index)
        features['CCI_20'] = cci
        features['CCI_oversold'] = (cci < -100).astype(int)
        features['CCI_overbought'] = (cci > 100).astype(int)
        
        # MFI
        mfi = ta.mfi(high, low, close, volume, length=14)
        features['MFI_14'] = mfi
        features['MFI_oversold'] = (mfi < 20).astype(int)
        features['MFI_overbought'] = (mfi > 80).astype(int)
        
        # Price action
        features['RET_1p'] = close.pct_change(1) * 100
        
        # PERIOD RETURN FEATURES (Adapted from Daily)
        # Note: -0.88% in an hour is a significant move, similar to -0.88% in a day?
        # Actually, hourly volatility is lower than daily. 
        # But let's keep the threshold or maybe scale it down?
        # User said "same strategy", so we keep the thresholds.
        period_return = close.pct_change(1)
        features['PERIOD_RETURN_PANIC'] = (period_return < -0.0088).astype(int)
        features['PERIOD_RETURN_CRASH'] = (period_return < -0.02).astype(int)
        features['PERIOD_RETURN_SURGE'] = (period_return > 0.02).astype(int)
        
        # Composites
        oversold_cols = [c for c in features.columns if 'oversold' in c.lower()]
        sharktooth_bull_cols = [c for c in features.columns if 'sharktooth' in c.lower() and 'bear' not in c.lower()]
        features['OVERSOLD_COUNT'] = features[oversold_cols].sum(axis=1)
        features['BULL_SHARKTOOTH_COUNT'] = features[sharktooth_bull_cols].sum(axis=1)
        
        overbought_cols = [c for c in features.columns if 'overbought' in c.lower()]
        sharktooth_bear_cols = [c for c in features.columns if 'sharktooth_bear' in c.lower()]
        features['OVERBOUGHT_COUNT'] = features[overbought_cols].sum(axis=1)
        features['BEAR_SHARKTOOTH_COUNT'] = features[sharktooth_bear_cols].sum(axis=1)
        
        return features


class SimpleDetector:
    def __init__(self, mode='bull'):
        self.mode = mode
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        features = TechnicalEngine.calculate(df)
        
        if self.mode == 'bull':
            oversold_count = features['OVERSOLD_COUNT']
            sharktooth_count = features['BULL_SHARKTOOTH_COUNT']
            panic = features.get('PERIOD_RETURN_PANIC', 0)
            prob = (oversold_count / 12 * 0.4 + sharktooth_count / 4 * 0.4 + panic * 0.2)
            prob = prob.clip(0, 1)
        else:
            overbought_count = features['OVERBOUGHT_COUNT']
            bear_shark_count = features['BEAR_SHARKTOOTH_COUNT']
            surge = features.get('PERIOD_RETURN_SURGE', 0)
            prob = (overbought_count / 12 * 0.4 + bear_shark_count / 4 * 0.4 + surge * 0.2)
            prob = prob.clip(0, 1)
        
        return pd.Series(prob, index=df.index, name=f'{self.mode}_prob')


def backtest(ticker: str, initial_capital: float = 100000, hold_days: int = 0, fixed_target: float = 0.0):
    print(f"\n{'='*70}")
    print(f"HOURLY BACKTEST: {ticker}")
    print(f"Hold: {hold_days} days | Fixed Target: ${fixed_target}")
    print(f"{'='*70}")
    
    # Load data (Max 730 days for hourly)
    print(f"Loading {ticker} hourly data (last 730 days)...")
    try:
        stock = yf.download(ticker, period="730d", interval="1h", progress=False)
    except Exception as e:
        print(f"Error downloading: {e}")
        return
        
    if stock.empty:
        print("ERROR: Could not load data")
        return
    
    print(f"Data loaded: {len(stock)} hours")
    print(f"Range: {stock.index[0]} to {stock.index[-1]}")
    
    # Calculate signals
    print("Calculating signals...")
    features = TechnicalEngine.calculate(stock)
    bull_probs = SimpleDetector(mode='bull').predict(stock)
    bear_probs = SimpleDetector(mode='bear').predict(stock)
    
    close = safe_series(stock['Close'])
    
    # Parameters (Same as Daily)
    params = {
        'bull_threshold': 0.45,
        'high_conviction_prob': 0.70,
        'high_conviction_count': 4,
        'medium_conviction_prob': 0.50,
        'medium_conviction_count': 3,
        'bear_threshold': 0.60,
        'bear_sharktooth_count': 3,
        'base_stop_loss': 0.12,
        'trailing_stop': 0.08,
        'profit_take_threshold': 0.20,
        'profit_take_pct': 0.25,
        'profit_take_bear_min': 0.30,
    }
    
    # Backtest
    cash = initial_capital
    shares = 0.0
    entry_price = 0.0
    max_price = 0.0
    entry_date = None
    partial_exit_taken = False
    
    trade_log = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    cooldown = 0
    bear_prob_history = []
    
    # Approx trading hours per day
    HOURS_PER_DAY = 7
    min_hold_hours = hold_days * HOURS_PER_DAY
    hours_held = 0
    
    for date in stock.index:
        price = float(close.loc[date])
        bull_prob = float(bull_probs.loc[date])
        bear_prob = float(bear_probs.loc[date])
        bull_count = float(features.loc[date, 'BULL_SHARKTOOTH_COUNT'])
        bear_count = float(features.loc[date, 'BEAR_SHARKTOOTH_COUNT'])
        
        bear_prob_history.append(bear_prob)
        if len(bear_prob_history) > 5:
            bear_prob_history.pop(0)
        bear_prob_avg = np.mean(bear_prob_history)
        
        if cooldown > 0:
            cooldown -= 1
        
        # ENTRY
        if shares == 0 and cooldown == 0:
            bull_signal = bull_prob > params['bull_threshold']
            sharktooth_signal = bull_count >= params['medium_conviction_count']
            
            if bull_signal or sharktooth_signal:
                size = 0.0
                if bull_prob > params['high_conviction_prob'] or bull_count >= params['high_conviction_count']:
                    size = 1.0
                elif bull_prob > params['medium_conviction_prob'] or bull_count >= params['medium_conviction_count']:
                    size = 0.5
                
                if size > 0:
                    invest = cash * size
                    shares = invest / price
                    cash -= invest
                    entry_price = price
                    max_price = price
                    entry_date = date
                    partial_exit_taken = False
                    hours_held = 0
        
        # EXIT
        elif shares > 0:
            hours_held += 1
            max_price = max(max_price, price)
            unrealized = (price - entry_price) / entry_price
            dd_from_high = (max_price - price) / max_price
            
            exit_signal = False
            
            # 1. Fixed Dollar Target (User Request)
            if fixed_target > 0 and (price - entry_price) >= fixed_target:
                exit_signal = True
            
            # 2. Forced Hold Logic (Only if fixed target not hit)
            elif hours_held < min_hold_hours:
                pass
                
            # 3. Standard Logic
            else:
                if not partial_exit_taken and unrealized > params['profit_take_threshold']:
                    if bear_prob > bear_prob_avg and bear_prob > params['profit_take_bear_min']:
                        sell_shares = shares * params['profit_take_pct']
                        cash += sell_shares * price
                        shares -= sell_shares
                        partial_exit_taken = True

                if bear_prob > params['bear_threshold']: exit_signal = True
                elif bear_count >= params['bear_sharktooth_count']: exit_signal = True
                elif unrealized < -params['base_stop_loss']: exit_signal = True
                elif dd_from_high > params['trailing_stop']: exit_signal = True
            
            if exit_signal:
                cash += shares * price
                ret = (price - entry_price) / entry_price
                trade_log.append({'return': ret})
                shares = 0
                cooldown = 3 # 3 hours cooldown
        
        equity = cash + shares * price
        if equity > peak_equity: peak_equity = equity
        max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
    
    # Close final position
    if shares > 0:
        cash += shares * close.iloc[-1]
        ret = (close.iloc[-1] - entry_price) / entry_price
        trade_log.append({'return': ret})
    
    # Stats
    total_return = (cash - initial_capital) / initial_capital
    years = len(stock) / (252 * 7) # Approx 7 trading hours per day
    cagr = (cash / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    buy_hold_return = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    
    print(f"\nRESULTS: {ticker} (Hourly)")
    print(f"Strategy Return:    {total_return*100:,.2f}%")
    print(f"Buy & Hold Return:  {buy_hold_return*100:,.2f}%")
    print(f"Outperformance:     {(total_return - buy_hold_return)*100:+,.2f}%")
    print(f"Max Drawdown:       {max_drawdown*100:.2f}%")
    print(f"Total Trades:       {len(trade_log)}")
    print(f"{'='*70}")
    
    with open('hourly_results.txt', 'a') as f:
        f.write(f"TICKER: {ticker}\nHOLD: {hold_days} days | TARGET: ${fixed_target}\nRETURN: {total_return*100:.2f}%\nB&H: {buy_hold_return*100:.2f}%\nDRAWDOWN: {max_drawdown*100:.2f}%\nTRADES: {len(trade_log)}\n---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tickers', nargs='+')
    parser.add_argument('--hold-days', type=int, default=0, help='Minimum holding period in days')
    parser.add_argument('--fixed-target', type=float, default=0.0, help='Fixed dollar profit target')
    args = parser.parse_args()
    
    for ticker in args.tickers:
        backtest(ticker, hold_days=args.hold_days, fixed_target=args.fixed_target)
