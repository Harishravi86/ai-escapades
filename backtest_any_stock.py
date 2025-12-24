"""
================================================================================
BULLETPROOF STRATEGY v7.2 - UNIVERSAL STOCK BACKTESTER
================================================================================

Test the v7.2 ML Integrated strategy on ANY stock.

Usage:
    python backtest_any_stock.py AAPL
    python backtest_any_stock.py NVDA TSLA MSFT GOOGL
    python backtest_any_stock.py AAPL --start 2010-01-01

IMPORTANT CAVEATS:
- Model was trained on SPY (index) patterns
- Individual stocks have higher volatility and gaps
- Results may not generalize well to all stocks
- Best candidates: Large-cap tech stocks that correlate with QQQ/SPY

================================================================================
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import argparse
from datetime import datetime
from typing import Dict, Optional

import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


def safe_series(col) -> pd.Series:
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


# =============================================================================
# TECHNICAL ENGINE v7.2 (Same as main strategy)
# =============================================================================

class TechnicalEngine:
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
                
                # Pine Script crossover features
                features[f'BB_{length}_crossunder'] = (
                    (pctb < -0.06) & (pctb.shift(1) >= -0.06)
                ).astype(int)
                features[f'BB_{length}_crossover'] = (
                    (pctb > 1.0) & (pctb.shift(1) <= 1.0)
                ).astype(int)
                features[f'BB_{length}_extreme_crossover'] = (
                    (pctb > 1.2) & (pctb.shift(1) <= 1.2)
                ).astype(int)
        
        # MACD
        macd = ta.macd(close)
        if macd is not None:
            features['MACD_line'] = macd.iloc[:, 0]
            features['MACD_hist'] = macd.iloc[:, 1]
            features['MACD_signal'] = macd.iloc[:, 2]
            features['MACD_oversold'] = (features['MACD_line'] < -2.0).astype(int)
            features['MACD_turnup'] = ((features['MACD_hist'] > features['MACD_hist'].shift(1)) & 
                                     (features['MACD_hist'] < 0)).astype(int)
            features['MACD_overbought'] = (features['MACD_line'] > 2.0).astype(int)
            features['MACD_turndown'] = ((features['MACD_hist'] < features['MACD_hist'].shift(1)) & 
                                       (features['MACD_hist'] > 0)).astype(int)
        
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
        
        # Daily return features (Pine Script)
        daily_return = close.pct_change(1)
        features['DAILY_RETURN_PANIC'] = (daily_return < -0.0088).astype(int)
        features['DAILY_RETURN_CRASH'] = (daily_return < -0.02).astype(int)
        features['DAILY_RETURN_EXTREME'] = (daily_return < -0.03).astype(int)
        features['DAILY_RETURN_SURGE'] = (daily_return > 0.02).astype(int)
        
        # Pine entry signal
        features['PINE_ENTRY_SIGNAL'] = (
            features.get('BB_20_crossunder', 0) & features['DAILY_RETURN_PANIC']
        ).astype(int)
        
        features['MULTI_CROSSUNDER'] = (
            features.get('BB_20_crossunder', 0) + features.get('BB_50_crossunder', 0)
        ).clip(0, 2)
        
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


# =============================================================================
# SIMPLIFIED DETECTOR (Rule-based, no ML training needed)
# =============================================================================

class SimpleDetector:
    """
    Rule-based detector that mimics ML behavior using composite scores.
    This avoids the need to retrain models for each stock.
    """
    
    def __init__(self, mode='bull'):
        self.mode = mode
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        features = TechnicalEngine.calculate(df)
        
        if self.mode == 'bull':
            # Bull probability based on oversold signals
            oversold_count = features['OVERSOLD_COUNT']
            sharktooth_count = features['BULL_SHARKTOOTH_COUNT']
            panic = features.get('DAILY_RETURN_PANIC', 0)
            
            # Normalize to 0-1 probability
            # Max oversold count is ~12, max sharktooth is ~4
            prob = (oversold_count / 12 * 0.4 + 
                   sharktooth_count / 4 * 0.4 +
                   panic * 0.2)
            prob = prob.clip(0, 1)
            
        else:  # bear
            overbought_count = features['OVERBOUGHT_COUNT']
            bear_shark_count = features['BEAR_SHARKTOOTH_COUNT']
            surge = features.get('DAILY_RETURN_SURGE', 0)
            
            prob = (overbought_count / 12 * 0.4 +
                   bear_shark_count / 4 * 0.4 +
                   surge * 0.2)
            prob = prob.clip(0, 1)
        
        return pd.Series(prob, index=df.index, name=f'{self.mode}_prob')


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_stock(
    ticker: str,
    start_date: str = '2000-01-01',
    initial_capital: float = 100000,
    use_spy_signals: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Backtest the v7.2 strategy on any stock.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date for backtest
        initial_capital: Starting capital
        use_spy_signals: If True, use SPY for signal generation (recommended)
        verbose: Print trade details
    
    Returns:
        Dictionary with backtest results
    """
    
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {ticker}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading {ticker} data...")
    stock = yf.download(ticker, start=start_date, progress=False)
    if stock.empty:
        print(f"ERROR: Could not load data for {ticker}")
        return None
    
    # Load SPY for signals (recommended - model was trained on SPY)
    if use_spy_signals:
        print("Loading SPY for signal generation...")
        spy = yf.download("SPY", start=start_date, progress=False)
        signal_data = spy
    else:
        signal_data = stock
    
    # Align dates
    common_idx = stock.index.intersection(signal_data.index)
    
    stock = stock.loc[common_idx]
    signal_data = signal_data.loc[common_idx]
    
    print(f"Data loaded: {len(stock)} trading days")
    print(f"Date range: {stock.index[0].strftime('%Y-%m-%d')} to {stock.index[-1].strftime('%Y-%m-%d')}")
    
    # Calculate features and signals
    print("Calculating signals...")
    features = TechnicalEngine.calculate(signal_data)
    
    bull_detector = SimpleDetector(mode='bull')
    bear_detector = SimpleDetector(mode='bear')
    
    bull_probs = bull_detector.predict(signal_data)
    bear_probs = bear_detector.predict(signal_data)
    
    close = safe_series(stock['Close'])
    
    # Backtest parameters (v7.2)
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
    
    # Run backtest
    cash = initial_capital
    shares = 0.0
    entry_price = 0.0
    max_price = 0.0
    entry_date = None
    partial_exit_taken = False
    
    trade_log = []
    equity_curve = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    cooldown = 0
    
    bear_prob_history = []
    
    print("Running backtest...")
    
    for i, date in enumerate(signal_data.index):
        if date not in close.index:
            continue
        
        price = float(close.loc[date])
        
        # Signals
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
                conviction = "SKIP"
                
                if bull_prob > params['high_conviction_prob'] or \
                   bull_count >= params['high_conviction_count']:
                    size = 1.0
                    conviction = "HIGH"
                elif bull_prob > params['medium_conviction_prob'] or \
                     bull_count >= params['medium_conviction_count']:
                    size = 0.5
                    conviction = "MEDIUM"
                
                if size > 0:
                    invest = cash * size
                    shares = invest / price
                    cash -= invest
                    entry_price = price
                    max_price = price
                    entry_date = date
                    partial_exit_taken = False
                    
                    if verbose and date.year >= 2020:
                        print(f"  [{date.strftime('%Y-%m-%d')}] BUY @ ${price:.2f} "
                              f"(Bull: {bull_prob:.0%}, Count: {bull_count:.0f}, {conviction})")
        
        # EXIT
        elif shares > 0:
            max_price = max(max_price, price)
            unrealized = (price - entry_price) / entry_price
            dd_from_high = (max_price - price) / max_price
            
            # Profit Taking
            if not partial_exit_taken and unrealized > params['profit_take_threshold']:
                if bear_prob > bear_prob_avg and bear_prob > params['profit_take_bear_min']:
                    sell_shares = shares * params['profit_take_pct']
                    cash += sell_shares * price
                    shares -= sell_shares
                    partial_exit_taken = True

            exit_signal = False
            reason = ""
            
            if bear_prob > params['bear_threshold']:
                exit_signal = True
                reason = "BEAR_TWIN"
            elif bear_count >= params['bear_sharktooth_count']:
                exit_signal = True
                reason = "BEAR_SHARK"
            elif unrealized < -params['base_stop_loss']:
                exit_signal = True
                reason = "STOP_LOSS"
            elif dd_from_high > params['trailing_stop']:
                exit_signal = True
                reason = "TRAILING"
            
            if exit_signal:
                cash += shares * price
                ret = (price - entry_price) / entry_price
                trade_log.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'return': ret,
                    'reason': reason,
                })
                if verbose and date.year >= 2020:
                    print(f"  [{date.strftime('%Y-%m-%d')}] SELL @ ${price:.2f} ({reason}, {ret:+.1%})")
                shares = 0
                cooldown = 3
        
        equity = cash + shares * price
        equity_curve.append({'date': date, 'equity': equity})
        if equity > peak_equity: 
            peak_equity = equity
        max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
    
    # Close final position
    if shares > 0:
        cash += shares * close.iloc[-1]
        ret = (close.iloc[-1] - entry_price) / entry_price
        trade_log.append({
            'entry_date': entry_date,
            'exit_date': stock.index[-1],
            'return': ret,
            'reason': 'END',
        })
    
    # Calculate stats
    total_return = (cash - initial_capital) / initial_capital
    years = len(stock) / 252
    cagr = (cash / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    returns = [t['return'] for t in trade_log]
    win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
    avg_return = np.mean(returns) if returns else 0
    
    # Buy and hold comparison
    buy_hold_return = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    buy_hold_cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    results = {
        'ticker': ticker,
        'total_return': total_return,
        'final_equity': cash,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_trades': len(trade_log),
        'buy_hold_return': buy_hold_return,
        'buy_hold_cagr': buy_hold_cagr,
        'years': years,
    }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {ticker}")
    print(f"{'='*70}")
    print(f"Strategy Return:    {total_return*100:,.2f}%")
    print(f"Buy & Hold Return:  {buy_hold_return*100:,.2f}%")
    print(f"Outperformance:     {(total_return - buy_hold_return)*100:+,.2f}%")
    print(f"-" * 70)
    print(f"Final Equity:       ${cash:,.2f}")
    print(f"CAGR:               {cagr*100:.2f}% (vs B&H: {buy_hold_cagr*100:.2f}%)")
    print(f"Max Drawdown:       {max_drawdown*100:.2f}%")
    print(f"Win Rate:           {win_rate*100:.1f}%")
    
    source_str = "SPY" if use_spy_signals else "SELF"
    with open('btc_final_summary.txt', 'a') as f:
        f.write(f"TICKER: {ticker}\nSOURCE: {source_str}\nRETURN: {total_return*100:.2f}%\nDRAWDOWN: {max_drawdown*100:.2f}%\nB&H: {buy_hold_return*100:.2f}%\n---\n")
    print(f"Avg Return/Trade:   {avg_return*100:.2f}%")
    print(f"Total Trades:       {len(trade_log)}")
    print(f"Years:              {years:.1f}")
    print(f"{'='*70}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Backtest v7.2 strategy on any stock')
    parser.add_argument('tickers', nargs='+', help='Stock ticker(s) to backtest')
    parser.add_argument('--start', default='2000-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--stock-signals', action='store_true', 
                       help='Use stock for signals (default: use SPY)')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BULLETPROOF STRATEGY v7.2 - UNIVERSAL BACKTESTER")
    print("=" * 70)
    
    if not args.stock_signals:
        print("\nNote: Using SPY for signal generation (recommended)")
        print("      Add --stock-signals to use each stock's own signals")
    
    results = []
    for ticker in args.tickers:
        result = backtest_stock(
            ticker=ticker.upper(),
            start_date=args.start,
            initial_capital=args.capital,
            use_spy_signals=not args.stock_signals,
            verbose=not args.quiet
        )
        if result:
            results.append(result)
    
    # Summary if multiple stocks
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY - ALL STOCKS")
        print("=" * 70)
        print(f"{'Ticker':<8} {'Return':>12} {'B&H':>12} {'Alpha':>10} {'Win%':>8} {'Trades':>8}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: x['total_return'], reverse=True):
            alpha = r['total_return'] - r['buy_hold_return']
            print(f"{r['ticker']:<8} {r['total_return']*100:>11.1f}% {r['buy_hold_return']*100:>11.1f}% "
                  f"{alpha*100:>+9.1f}% {r['win_rate']*100:>7.1f}% {r['total_trades']:>8}")
        print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - show example
        print("Usage: python backtest_any_stock.py AAPL NVDA TSLA MSFT")
        print("\nRunning example with AAPL...")
        backtest_stock("AAPL", verbose=True)
    else:
        main()
