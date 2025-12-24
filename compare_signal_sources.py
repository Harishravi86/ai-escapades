"""
================================================================================
SIGNAL SOURCE COMPARISON: SPY vs STOCK'S OWN SIGNALS
================================================================================

This script compares two approaches:
1. Using SPY signals to trade individual stocks (market-wide panic detection)
2. Using each stock's own signals (stock-specific panic detection)

Run: python compare_signal_sources.py NVDA AAPL MSFT

================================================================================
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import argparse
from datetime import datetime
from typing import Dict, Optional, List

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
    """Same as v7.2 - calculates all technical features"""
    
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
            features[f'RSI_{length}_extreme_high'] = (rsi > 80).astype(int)
        
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
                
                features[f'BB_{length}_crossunder'] = (
                    (pctb < -0.06) & (pctb.shift(1) >= -0.06)
                ).astype(int)
                features[f'BB_{length}_crossover'] = (
                    (pctb > 1.0) & (pctb.shift(1) <= 1.0)
                ).astype(int)
        
        # MACD
        macd = ta.macd(close)
        if macd is not None:
            features['MACD_line'] = macd.iloc[:, 0]
            features['MACD_hist'] = macd.iloc[:, 1]
            features['MACD_signal'] = macd.iloc[:, 2]
            features['MACD_oversold'] = (features['MACD_line'] < -2.0).astype(int)
            features['MACD_overbought'] = (features['MACD_line'] > 2.0).astype(int)
        
        # Stochastic
        stoch = ta.stoch(high, low, close)
        if stoch is not None:
            k = stoch.iloc[:, 0]
            d = stoch.iloc[:, 1]
            features['STOCH_k'] = k
            features['STOCH_d'] = d
            features['STOCH_oversold'] = (k < 20).astype(int)
            features['STOCH_sharktooth'] = ((k < 20) & (k > d)).astype(int)
            features['STOCH_overbought'] = (k > 80).astype(int)
            features['STOCH_sharktooth_bear'] = ((k > 80) & (k < d)).astype(int)
        
        # Williams %R
        for length in [14, 28]:
            willr = ta.willr(high, low, close, length=length)
            features[f'WILLR_{length}'] = willr
            features[f'WILLR_{length}_oversold'] = (willr < -80).astype(int)
            features[f'WILLR_{length}_extreme'] = (willr < -90).astype(int)
            features[f'WILLR_{length}_overbought'] = (willr > -20).astype(int)
        
        # CCI
        cci = ta.cci(high, low, close, length=20)
        cci = cci.clip(-500, 500) if cci is not None else pd.Series(0, index=df.index)
        features['CCI_20'] = cci
        features['CCI_oversold'] = (cci < -100).astype(int)
        features['CCI_extreme'] = (cci < -200).astype(int)
        features['CCI_overbought'] = (cci > 100).astype(int)
        
        # MFI
        mfi = ta.mfi(high, low, close, volume, length=14)
        features['MFI_14'] = mfi
        features['MFI_oversold'] = (mfi < 20).astype(int)
        features['MFI_overbought'] = (mfi > 80).astype(int)
        
        # Price action
        features['RET_1d'] = close.pct_change(1) * 100
        features['RET_5d'] = close.pct_change(5) * 100
        
        for period in [10, 20, 50]:
            rolling_max = close.rolling(period).max()
            features[f'DD_{period}d'] = (close - rolling_max) / rolling_max * 100
            rolling_min = close.rolling(period).min()
            features[f'RALLY_{period}d'] = (close - rolling_min) / rolling_min * 100
        
        # Daily return features
        daily_return = close.pct_change(1)
        features['DAILY_RETURN_PANIC'] = (daily_return < -0.0088).astype(int)
        features['DAILY_RETURN_CRASH'] = (daily_return < -0.02).astype(int)
        features['DAILY_RETURN_SURGE'] = (daily_return > 0.02).astype(int)
        
        # Pine entry signal
        features['PINE_ENTRY_SIGNAL'] = (
            features.get('BB_20_crossunder', 0) & features['DAILY_RETURN_PANIC']
        ).astype(int)
        
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
            panic = features.get('DAILY_RETURN_PANIC', 0)
            
            prob = (oversold_count / 12 * 0.4 + 
                   sharktooth_count / 4 * 0.4 +
                   panic * 0.2)
            prob = prob.clip(0, 1)
            
        else:
            overbought_count = features['OVERBOUGHT_COUNT']
            bear_shark_count = features['BEAR_SHARKTOOTH_COUNT']
            surge = features.get('DAILY_RETURN_SURGE', 0)
            
            prob = (overbought_count / 12 * 0.4 +
                   bear_shark_count / 4 * 0.4 +
                   surge * 0.2)
            prob = prob.clip(0, 1)
        
        return pd.Series(prob, index=df.index, name=f'{self.mode}_prob')


def backtest(
    ticker: str,
    signal_source: str,  # 'SPY' or 'SELF'
    start_date: str = '2000-01-01',
    initial_capital: float = 100000,
) -> Dict:
    """
    Backtest with specified signal source.
    """
    
    # Load stock data
    stock = yf.download(ticker, start=start_date, progress=False)
    if stock.empty:
        return None
    
    # Load signal source
    if signal_source == 'SPY':
        signal_data = yf.download("SPY", start=start_date, progress=False)
    else:
        signal_data = stock.copy()
    
    # Align dates
    common_idx = stock.index.intersection(signal_data.index)
    stock = stock.loc[common_idx]
    signal_data = signal_data.loc[common_idx]
    
    # Calculate signals
    features = TechnicalEngine.calculate(signal_data)
    bull_probs = SimpleDetector(mode='bull').predict(signal_data)
    bear_probs = SimpleDetector(mode='bear').predict(signal_data)
    close = safe_series(stock['Close'])
    
    # Parameters
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
    
    for date in signal_data.index:
        if date not in close.index:
            continue
        
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
                
                if bull_prob > params['high_conviction_prob'] or \
                   bull_count >= params['high_conviction_count']:
                    size = 1.0
                elif bull_prob > params['medium_conviction_prob'] or \
                     bull_count >= params['medium_conviction_count']:
                    size = 0.5
                
                if size > 0:
                    invest = cash * size
                    shares = invest / price
                    cash -= invest
                    entry_price = price
                    max_price = price
                    entry_date = date
                    partial_exit_taken = False
        
        # EXIT
        elif shares > 0:
            max_price = max(max_price, price)
            unrealized = (price - entry_price) / entry_price
            dd_from_high = (max_price - price) / max_price
            
            if not partial_exit_taken and unrealized > params['profit_take_threshold']:
                if bear_prob > bear_prob_avg and bear_prob > params['profit_take_bear_min']:
                    sell_shares = shares * params['profit_take_pct']
                    cash += sell_shares * price
                    shares -= sell_shares
                    partial_exit_taken = True

            exit_signal = False
            
            if bear_prob > params['bear_threshold']:
                exit_signal = True
            elif bear_count >= params['bear_sharktooth_count']:
                exit_signal = True
            elif unrealized < -params['base_stop_loss']:
                exit_signal = True
            elif dd_from_high > params['trailing_stop']:
                exit_signal = True
            
            if exit_signal:
                cash += shares * price
                ret = (price - entry_price) / entry_price
                trade_log.append({'return': ret})
                shares = 0
                cooldown = 3
        
        equity = cash + shares * price
        if equity > peak_equity:
            peak_equity = equity
        max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
    
    # Close final position
    if shares > 0:
        cash += shares * close.iloc[-1]
        ret = (close.iloc[-1] - entry_price) / entry_price
        trade_log.append({'return': ret})
    
    # Stats
    total_return = (cash - initial_capital) / initial_capital
    years = len(stock) / 252
    cagr = (cash / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    returns = [t['return'] for t in trade_log]
    win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
    
    buy_hold_return = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    
    return {
        'ticker': ticker,
        'signal_source': signal_source,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trade_log),
        'buy_hold_return': buy_hold_return,
        'years': years,
    }


def compare_stock(ticker: str, start_date: str = '2000-01-01'):
    """Compare SPY signals vs Stock's own signals"""
    
    print(f"\n{'='*70}")
    print(f"COMPARING SIGNAL SOURCES: {ticker}")
    print(f"{'='*70}")
    
    # Test with SPY signals
    print(f"\nTesting with SPY signals...")
    spy_result = backtest(ticker, 'SPY', start_date)
    
    # Test with stock's own signals
    print(f"Testing with {ticker}'s own signals...")
    self_result = backtest(ticker, 'SELF', start_date)
    
    if not spy_result or not self_result:
        print(f"ERROR: Could not load data for {ticker}")
        return None
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"RESULTS COMPARISON: {ticker}")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'SPY Signals':>15} {f'{ticker} Signals':>15} {'Winner':>12}")
    print("-" * 70)
    
    metrics = [
        ('Total Return', 'total_return', '%', 100),
        ('CAGR', 'cagr', '%', 100),
        ('Max Drawdown', 'max_drawdown', '%', 100),
        ('Win Rate', 'win_rate', '%', 100),
        ('Total Trades', 'total_trades', '', 1),
    ]
    
    for name, key, suffix, mult in metrics:
        spy_val = spy_result[key] * mult
        self_val = self_result[key] * mult
        
        # Determine winner (lower is better for drawdown)
        if key == 'max_drawdown':
            winner = "SPY" if spy_val < self_val else ticker if self_val < spy_val else "TIE"
        else:
            winner = "SPY" if spy_val > self_val else ticker if self_val > spy_val else "TIE"
        
        if suffix == '%':
            print(f"{name:<20} {spy_val:>14.1f}% {self_val:>14.1f}% {winner:>12}")
        else:
            print(f"{name:<20} {spy_val:>15.0f} {self_val:>15.0f} {winner:>12}")
    
    print("-" * 70)
    print(f"{'Buy & Hold'::<20} {spy_result['buy_hold_return']*100:>14.1f}%")
    print(f"{'Years'::<20} {spy_result['years']:>15.1f}")
    print("=" * 70)
    
    # Determine overall winner
    spy_score = 0
    self_score = 0
    
    if spy_result['total_return'] > self_result['total_return']:
        spy_score += 2
    else:
        self_score += 2
    
    if spy_result['max_drawdown'] < self_result['max_drawdown']:
        spy_score += 1
    else:
        self_score += 1
    
    if spy_result['win_rate'] > self_result['win_rate']:
        spy_score += 1
    else:
        self_score += 1
        
    overall_winner = "SPY Signals" if spy_score > self_score else f"{ticker}'s Own Signals"
    print(f"\n*** OVERALL WINNER: {overall_winner}")
    print(f"RAW_RESULTS: {ticker} SPY={spy_result['total_return']:.4f} SELF={self_result['total_return']:.4f}")
    with open('raw_comparison.txt', 'a') as f:
        f.write(f"{ticker} SPY={spy_result['total_return']:.4f} SELF={self_result['total_return']:.4f}\n")
    
    return {
        'ticker': ticker,
        'spy_result': spy_result,
        'self_result': self_result,
        'winner': overall_winner,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare SPY signals vs Stock signals')
    parser.add_argument('tickers', nargs='+', help='Stock ticker(s) to compare')
    parser.add_argument('--start', default='2000-01-01', help='Start date')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SIGNAL SOURCE COMPARISON")
    print("SPY Signals vs Stock's Own Signals")
    print("=" * 70)
    
    results = []
    for ticker in args.tickers:
        result = compare_stock(ticker.upper(), args.start)
        if result:
            results.append(result)
    
    # Summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY - ALL STOCKS")
        print("=" * 70)
        print(f"{'Ticker':<8} {'SPY Return':>15} {'Self Return':>15} {'Winner':>15}")
        print("-" * 70)
        
        spy_wins = 0
        self_wins = 0
        
        for r in results:
            spy_ret = r['spy_result']['total_return'] * 100
            self_ret = r['self_result']['total_return'] * 100
            winner = "SPY" if spy_ret > self_ret else r['ticker']
            
            if winner == "SPY":
                spy_wins += 1
            else:
                self_wins += 1
            
            print(f"{r['ticker']:<8} {spy_ret:>14.1f}% {self_ret:>14.1f}% {winner:>15}")
        
        print("-" * 70)
        print(f"\nSPY Signals won: {spy_wins}/{len(results)} stocks")
        print(f"Self Signals won: {self_wins}/{len(results)} stocks")
        print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python compare_signal_sources.py NVDA AAPL MSFT")
        print("\nThis will compare using SPY signals vs each stock's own signals")
    else:
        main()
