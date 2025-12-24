"""
================================================================================
INTELLIGENT WALL STREET TRADER (v1.0) 🤖
================================================================================

A Live Paper Trading Bot using the Bulletproof v7.2 Strategy.
Features:
- Real-time Portfolio Management ($100k Paper Money)
- Daily Signal Logic with Intraday Monitoring
- "Narrative Engine" for LLM-ready explanations

Usage:
    python intelligent_trader.py
================================================================================
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. STRATEGY LOGIC (v7.2)
# =============================================================================

def safe_series(col) -> pd.Series:
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col

class TechnicalEngine:
    """Calculates v7.2 Features"""
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
        
        # Bollinger Bands
        for length in [20, 50]:
            bb = ta.bbands(close, length=length, std=2.0)
            if bb is not None:
                col_pctb = f'BBP_{length}_2.0'
                if col_pctb not in bb.columns:
                    # Fallback if names differ
                    try:
                        lower = bb.iloc[:, 0]
                        upper = bb.iloc[:, 2]
                        pctb = (close - lower) / (upper - lower)
                    except:
                        pctb = pd.Series(0, index=df.index)
                else:
                    pctb = bb[col_pctb]
                
                features[f'BB_{length}_pctb'] = pctb
                features[f'BB_{length}_oversold'] = (pctb < 0).astype(int)
                features[f'BB_{length}_sharktooth'] = (pctb < -0.1).astype(int)
                features[f'BB_{length}_overbought'] = (pctb > 1).astype(int)
                features[f'BB_{length}_sharktooth_bear'] = (pctb > 1.1).astype(int)
        
        # Daily Return Features (Crucial for v7.2)
        daily_return = close.pct_change(1)
        features['DAILY_RETURN_PANIC'] = (daily_return < -0.0088).astype(int)
        features['DAILY_RETURN_SURGE'] = (daily_return > 0.02).astype(int)
        
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
    """Rule-based Signal Detector (Mimics ML Model)"""
    def __init__(self, mode='bull'):
        self.mode = mode
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        features = TechnicalEngine.calculate(df)
        
        if self.mode == 'bull':
            oversold_count = features['OVERSOLD_COUNT']
            sharktooth_count = features['BULL_SHARKTOOTH_COUNT']
            panic = features.get('DAILY_RETURN_PANIC', 0)
            prob = (oversold_count / 12 * 0.4 + sharktooth_count / 4 * 0.4 + panic * 0.2)
        else:
            overbought_count = features['OVERBOUGHT_COUNT']
            bear_shark_count = features['BEAR_SHARKTOOTH_COUNT']
            surge = features.get('DAILY_RETURN_SURGE', 0)
            prob = (overbought_count / 12 * 0.4 + bear_shark_count / 4 * 0.4 + surge * 0.2)
            
        return pd.Series(prob.clip(0, 1), index=df.index, name=f'{self.mode}_prob')

# =============================================================================
# 2. PAPER BROKER (Accounting)
# =============================================================================

class PaperBroker:
    def __init__(self, initial_capital: float = 100000.0):
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {} # {ticker: {'shares': 10, 'avg_price': 150.0}}
        self.trade_log: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        equity = self.cash
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos['avg_price']) # Fallback to cost if no price
            equity += pos['shares'] * price
        return equity
    
    def buy(self, ticker: str, shares: float, price: float, reason: str = ""):
        cost = shares * price
        if cost > self.cash:
            print(f"❌ [BROKER] Insufficient funds to buy {shares} {ticker} @ {price}")
            return False
        
        self.cash -= cost
        
        if ticker in self.positions:
            # Average up/down
            current_shares = self.positions[ticker]['shares']
            current_avg = self.positions[ticker]['avg_price']
            new_shares = current_shares + shares
            new_avg = ((current_shares * current_avg) + cost) / new_shares
            self.positions[ticker] = {'shares': new_shares, 'avg_price': new_avg}
        else:
            self.positions[ticker] = {'shares': shares, 'avg_price': price}
            
        self.trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'reason': reason
        })
        print(f"✅ [BROKER] BOUGHT {shares:.2f} {ticker} @ ${price:.2f} ({reason})")
        return True
    
    def sell(self, ticker: str, shares: float, price: float, reason: str = ""):
        if ticker not in self.positions or self.positions[ticker]['shares'] < shares:
            print(f"❌ [BROKER] Cannot sell {shares} {ticker} (Not enough shares)")
            return False
        
        self.cash += shares * price
        self.positions[ticker]['shares'] -= shares
        
        # Remove if empty
        if self.positions[ticker]['shares'] < 0.0001:
            del self.positions[ticker]
            
        self.trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'SELL',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'reason': reason
        })
        print(f"✅ [BROKER] SOLD {shares:.2f} {ticker} @ ${price:.2f} ({reason})")
        return True

# =============================================================================
# 3. TRADER AGENT (The Brain)
# =============================================================================

class TraderAgent:
    def __init__(self, broker: PaperBroker, tickers: List[str]):
        self.broker = broker
        self.tickers = tickers
        self.params = {
            'bull_threshold': 0.45,
            'bear_threshold': 0.60,
            'stop_loss': 0.12,
            'trailing_stop': 0.08,
            'profit_target': 0.20
        }
        self.market_state = {} # Stores latest data/signals
        
    def fetch_data(self):
        """Fetches latest Daily data for all tickers"""
        prices = {}
        for ticker in self.tickers:
            try:
                # Get last 100 days to ensure enough for indicators
                df = yf.download(ticker, period="100d", interval="1d", progress=False)
                if not df.empty:
                    self.market_state[ticker] = df
                    prices[ticker] = float(safe_series(df['Close']).iloc[-1])
            except Exception as e:
                print(f"⚠️ Error fetching {ticker}: {e}")
        return prices

    def analyze_market(self):
        """Runs v7.2 Strategy on latest data"""
        decisions = []
        
        for ticker, df in self.market_state.items():
            # Calculate Signals
            bull_prob = SimpleDetector(mode='bull').predict(df).iloc[-1]
            bear_prob = SimpleDetector(mode='bear').predict(df).iloc[-1]
            current_price = float(safe_series(df['Close']).iloc[-1])
            
            # 1. Check for EXITS (if we own it)
            if ticker in self.broker.positions:
                pos = self.broker.positions[ticker]
                entry_price = pos['avg_price']
                unrealized_pct = (current_price - entry_price) / entry_price
                
                # Logic: Stop Loss or Bear Signal
                if unrealized_pct < -self.params['stop_loss']:
                    decisions.append({
                        'action': 'SELL', 'ticker': ticker, 'price': current_price,
                        'reason': f"STOP LOSS hit ({unrealized_pct:.1%})"
                    })
                elif bear_prob > self.params['bear_threshold']:
                    decisions.append({
                        'action': 'SELL', 'ticker': ticker, 'price': current_price,
                        'reason': f"BEAR SIGNAL (Prob: {bear_prob:.0%})"
                    })
                elif unrealized_pct > self.params['profit_target']:
                     decisions.append({
                        'action': 'SELL', 'ticker': ticker, 'price': current_price,
                        'reason': f"PROFIT TARGET ({unrealized_pct:.1%})"
                    })
            
            # 2. Check for ENTRIES (if we have cash)
            else:
                if bull_prob > self.params['bull_threshold']:
                    decisions.append({
                        'action': 'BUY', 'ticker': ticker, 'price': current_price,
                        'reason': f"BULL SIGNAL (Prob: {bull_prob:.0%})"
                    })
        
        return decisions

    def execute_decisions(self, decisions):
        for d in decisions:
            if d['action'] == 'BUY':
                # Position Sizing: 20% of current equity per trade
                equity = self.broker.get_equity({t: float(safe_series(self.market_state[t]['Close']).iloc[-1]) for t in self.tickers if t in self.market_state})
                target_size = equity * 0.20
                shares = target_size / d['price']
                self.broker.buy(d['ticker'], shares, d['price'], d['reason'])
                
            elif d['action'] == 'SELL':
                shares = self.broker.positions[d['ticker']]['shares']
                self.broker.sell(d['ticker'], shares, d['price'], d['reason'])

    def explain_decision(self, ticker: str) -> str:
        """Narrative Engine: Explains the bot's view on a ticker in English"""
        if ticker not in self.market_state:
            return f"I don't have data for {ticker} yet."
            
        df = self.market_state[ticker]
        bull_prob = SimpleDetector(mode='bull').predict(df).iloc[-1]
        bear_prob = SimpleDetector(mode='bear').predict(df).iloc[-1]
        price = float(safe_series(df['Close']).iloc[-1])
        
        rsi = ta.rsi(safe_series(df['Close']), length=14).iloc[-1]
        
        narrative = f"**Analysis for {ticker} (${price:.2f}):**\n"
        narrative += f"- **Bull Probability:** {bull_prob:.1%} "
        if bull_prob > 0.45: narrative += "(HIGH 🟢)\n"
        else: narrative += "(LOW 🔴)\n"
        
        narrative += f"- **Bear Probability:** {bear_prob:.1%} "
        if bear_prob > 0.60: narrative += "(DANGER ⚠️)\n"
        else: narrative += "(SAFE 🛡️)\n"
        
        narrative += f"- **RSI (14):** {rsi:.1f}\n"
        
        if ticker in self.broker.positions:
            pos = self.broker.positions[ticker]
            pnl = (price - pos['avg_price']) / pos['avg_price']
            narrative += f"- **Position:** I own {pos['shares']:.2f} shares. P&L: {pnl:+.2%}.\n"
            if pnl < -0.10: narrative += "  - WARNING: Approaching Stop Loss!\n"
        else:
            narrative += "- **Position:** I am currently watching from the sidelines.\n"
            
        return narrative

# =============================================================================
# MAIN RUNNER
# =============================================================================

if __name__ == "__main__":
    print("Initializing Intelligent Trader...")
    broker = PaperBroker()
    agent = TraderAgent(broker, tickers=['SPY', 'NVDA', 'AAPL', 'MSFT', 'TSLA'])
    
    print("Fetching initial market data...")
    prices = agent.fetch_data()
    
    print("\n--- MARKET SNAPSHOT ---")
    for ticker, price in prices.items():
        print(f"{ticker}: ${price:.2f}")
        
    print("\n--- AGENT ANALYSIS ---")
    decisions = agent.analyze_market()
    if not decisions:
        print("No trades triggered. Market is quiet.")
    else:
        print(f"Found {len(decisions)} potential trades.")
        agent.execute_decisions(decisions)
        
    print("\n--- NARRATIVE REPORT ---")
    print(agent.explain_decision('NVDA'))
    
    print("\nBot is ready for Dashboard integration.")
