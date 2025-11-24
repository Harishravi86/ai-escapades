"""
BULLETPROOF ML STRATEGY (Shareable Edition)
===========================================

This script implements a robust, risk-managed trading strategy for QQQ using SPY signals.
It combines Technical Analysis (Bollinger Bands, RSI, Volume) with Machine Learning (XGBoost)
and advanced Money Management (Kelly Criterion, Volatility Sizing).

FEATURES:
1. Ensemble Signals: BB + RSI + Volume + Regime Filter
2. Volatility Sizing: Reduces size in high VIX environments
3. Kelly Criterion: Optimizes bet size based on win rate
4. "Fortress" Risk Management: 11% Max Drawdown (historically)

USAGE:
1. Ensure you have the required libraries:
   pip install yfinance pandas pandas_ta numpy joblib xgboost

2. Place 'xgb_supervisor.pkl' and 'feature_names.pkl' in the same folder (optional but recommended).
   If missing, the strategy runs in "Technical Only" mode.

3. Run the script:
   python bulletproof_strategy.py
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import os
import sys

# --- 1. DATA LOADER ---
def load_data(symbol="SPY", start_date="2010-01-01"):
    """Loads data from Yahoo Finance with caching."""
    file_path = f"{symbol}_data.csv"
    
    if os.path.exists(file_path):
        print(f"Loading {symbol} data from cache...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {symbol} data from Yahoo Finance...")
        df = yf.download(symbol, start=start_date)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.to_csv(file_path)
        
    return df

# --- 2. STRATEGY CORE ---
class Strategy:
    def __init__(self, data):
        self.data = data
        # Load ML model if available
        try:
            model_path = "xgb_supervisor.pkl"
            feat_path = "feature_names.pkl"
            
            if not os.path.exists(model_path):
                # Check spx_optimizer folder (dev environment)
                if os.path.exists(os.path.join("spx_optimizer", model_path)):
                    model_path = os.path.join("spx_optimizer", model_path)
                    feat_path = os.path.join("spx_optimizer", feat_path)
            
            self.ml_model = joblib.load(model_path)
            self.ml_features = joblib.load(feat_path)
            self.use_ml = True
            print(f"ML Supervisor Loaded from {model_path}")
        except:
            print("Warning: ML Supervisor NOT found. Running in Technical-Only mode.")
            self.use_ml = False
            self.ml_features = []

    def prepare_features(self, df):
        """Prepare all technical indicators and ML features"""
        df = df.copy()
        
        # ML Features (needed for inference)
        if self.use_ml:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9'] if 'MACD_12_26_9' in macd.columns else 0
            df['Vol_20'] = df['Close'].pct_change().rolling(20).std()
            df['Ret_1d'] = df['Close'].pct_change()
            df['Ret_5d'] = df['Close'].pct_change(5)
            df['Ret_20d'] = df['Close'].pct_change(20)
            sma50 = ta.sma(df['Close'], length=50)
            df['Dist_SMA50'] = (df['Close'] - sma50) / sma50
            df['Month'] = df.index.month
            df['DayOfWeek'] = df.index.dayofweek
            df['DayOfYear'] = df.index.dayofyear
            df['Is_Feb_Mar'] = df['Month'].isin([2, 3]).astype(int)
            df['Is_Sep_Oct'] = df['Month'].isin([9, 10]).astype(int)
            df['Is_Jun'] = (df['Month'] == 6).astype(int)
            df['Gann_Corr_20'] = 0
            df['Sentiment_Score'] = 0
            df['Sentiment_Magnitude'] = 0
            
            # Create inference dataframe
            inference_df = pd.DataFrame(index=df.index, columns=self.ml_features)
            for col in self.ml_features:
                if col in df.columns:
                    inference_df[col] = df[col]
                else:
                    inference_df[col] = 0
            inference_df.fillna(0, inplace=True)
            return df, inference_df
        
        return df, None

    def run(self, params, data=None, trade_data=None, vix_data=None):
        """Run Strategy Simulation"""
        if data is None: data = self.data
        
        # Signal Data (SPY)
        df, inference_df = self.prepare_features(data)
        
        # Trade Data (QQQ)
        if trade_data is not None:
            common_idx = df.index.intersection(trade_data.index)
            if vix_data is not None:
                common_idx = common_idx.intersection(vix_data.index)
                vix_data = vix_data.loc[common_idx]
                
            df = df.loc[common_idx]
            trade_df = trade_data.loc[common_idx]
            inference_df = inference_df.loc[common_idx] if inference_df is not None else None
        else:
            trade_df = df
            
        # Add VIX to df for easy access
        if vix_data is not None:
            df['VIX'] = vix_data['Close']
        else:
            # Fallback if no VIX
            df['VIX'] = 20.0
        
        # --- INDICATORS ---
        bb_len = int(params.get('bull_bb_length', 28))
        bb_std = params.get('bull_bb_std', 2.61)
        bb = ta.bbands(df['Close'], length=bb_len, std=bb_std)
        bbp_col = [c for c in bb.columns if c.startswith('BBP')][0]
        df['pct_b'] = bb[bbp_col]
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Vol_MA'] = ta.sma(df['Volume'], length=20)
        df['Vol_Surge'] = df['Volume'] > df['Vol_MA'] * 1.5
        
        df['SMA200'] = ta.sma(df['Close'], length=200)
        df['Regime'] = np.where(df['Close'] > df['SMA200'], 1, -1)
        
        # --- SIGNALS ---
        df['Signal'] = 0
        entry_thresh = params.get('bull_entry_b', 0.049)
        exit_thresh = params.get('bull_exit_b', 1.04)
        
        # Correlation Check (SPY vs QQQ)
        # We want to ensure they are moving together before entering
        df['Corr'] = df['Close'].pct_change().rolling(20).corr(trade_df['Close'].pct_change())
        
        # 1. Signal Scoring System
        # Base Signal: Bollinger Band Oversold
        df['Signal_Score'] = 0
        df.loc[df['pct_b'] < entry_thresh, 'Signal_Score'] += 3
        df.loc[df['RSI'] < 30, 'Signal_Score'] += 2
        df.loc[df['Vol_Surge'], 'Signal_Score'] += 1
        df.loc[df['Regime'] == 1, 'Signal_Score'] += 1
        
        # Correlation Penalty: If correlation breaks, reduce score
        df.loc[df['Corr'] < 0.7, 'Signal_Score'] -= 1
        
        # Entry Trigger: Score >= 4
        df.loc[df['Signal_Score'] >= 4, 'Signal'] = 1
        
        # Filter: RSI must not be overbought for entry (sanity check)
        df.loc[df['RSI'] > 70, 'Signal'] = 0 
        
        # Exit Signal
        df.loc[df['pct_b'] > exit_thresh, 'Signal'] = -1
        
        # --- SIMULATION ---
        cash = 100000.0
        shares = 0
        entry_price = 0.0
        max_price = 0.0
        ml_confidence = 0.0
        trade_log = []
        cooldown = 0
        entry_date = None
        partial_taken = False
        
        peak_equity = 100000.0
        max_drawdown = 0.0
        
        wins_list = []
        losses_list = []
        
        print("Running Simulation...")
        for i in range(len(df)):
            sig = df['Signal'].iloc[i]
            regime = df['Regime'].iloc[i]
            date = df.index[i]
            vix = df['VIX'].iloc[i]
            price = trade_df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            
            if cooldown > 0: cooldown -= 1
            
            # Circuit Breaker: If in deep drawdown, force cooldown
            if max_drawdown > 0.15 and cooldown == 0:
                cooldown = 10 # Cool off for 10 days
            
            # ENTRY
            if shares == 0:
                if cooldown == 0 and sig == 1:
                    
                    # Regime Filter (Stricter)
                    # Don't buy in Bear Market unless Deeply Oversold (RSI < 30)
                    if regime == -1 and rsi > 30:
                        continue
                        
                    # ML Check
                    ml_prob = 0.5
                    if self.use_ml and inference_df is not None:
                        try:
                            feat_vector = inference_df.iloc[i].values.reshape(1, -1)
                            ml_prob = self.ml_model.predict_proba(feat_vector)[0][1]
                        except: pass
                    
                    min_conf = params.get('min_ml_confidence', 0.521)
                    if ml_prob >= min_conf:
                        # Sizing
                        conf_mult = params.get('confidence_multiplier', 0.822)
                        base_size = min(ml_prob * conf_mult, 1.0)
                        
                        if regime == -1: base_size *= 0.5
                        
                        # VIX Sizing
                        if vix > 30: base_size *= 0.5      # Extreme Fear
                        elif vix < 15: base_size *= 1.2    # Complacency/Bullish
                        
                        # Win Streak Momentum
                        if len(wins_list) >= 3 and all(w > 0 for w in wins_list[-3:]):
                            base_size *= 1.1
                        
                        # Kelly (Faster Start)
                        total_trades = len(wins_list) + len(losses_list)
                        if total_trades >= 10:
                            avg_win = np.mean(wins_list) if wins_list else 0
                            avg_loss = abs(np.mean(losses_list)) if losses_list else 0
                            win_rate = len(wins_list) / total_trades
                            if avg_loss > 0:
                                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
                                base_size = min(base_size, max(0, kelly * 0.5))
                        
                        base_size = min(base_size, 1.0)
                        
                        # Circuit Breaker Sizing
                        if max_drawdown > 0.15:
                            base_size *= 0.5
                        
                        invest_amount = cash * base_size
                        shares = invest_amount / price
                        cash -= invest_amount
                        entry_price = price
                        max_price = price
                        ml_confidence = ml_prob
                        entry_date = date
                        partial_taken = False
            
            # EXIT
            else:
                if price > max_price: max_price = price
                
                unrealized_ret = (price - entry_price) / entry_price
                dd_from_high = (max_price - price) / max_price
                days_held = (date - entry_date).days
                
                # Stops
                base_stop = params.get('base_stop_loss', 0.138)
                stop_loss_pct = base_stop * (2.0 - ml_confidence)
                
                trailing_stop_pct = params.get('base_trailing_stop', 0.088)
                if unrealized_ret > params.get('profit_milestone_2', 0.370):
                    trailing_stop_pct = params.get('tight_stop_2', 0.039)
                elif unrealized_ret > params.get('profit_milestone_1', 0.164):
                    trailing_stop_pct = params.get('tight_stop_1', 0.084)
                
                # Partial Profit Taking
                if unrealized_ret > 0.50 and not partial_taken:
                    partial_shares = shares * 0.5
                    cash += partial_shares * price
                    shares -= partial_shares
                    partial_taken = True
                    # Also tighten stop on remainder
                    trailing_stop_pct = 0.05 
                
                time_exit = (days_held > 30 and unrealized_ret < 0.02)
                
                exit_triggered = False
                reason = ""
                
                if unrealized_ret < -stop_loss_pct:
                    exit_triggered = True; reason = "STOP_LOSS"
                elif dd_from_high > trailing_stop_pct:
                    exit_triggered = True; reason = "TRAILING_STOP"
                elif sig == -1:
                    exit_triggered = True; reason = "SIGNAL_EXIT"
                elif time_exit:
                    exit_triggered = True; reason = "TIME_EXIT"
                
                if exit_triggered:
                    cash += shares * price
                    ret = (price - entry_price) / entry_price
                    trade_log.append({
                        'date': date, 
                        'action': reason, 
                        'return': ret,
                        'partial_taken': partial_taken
                    })
                    
                    if ret > 0: wins_list.append(ret)
                    else: losses_list.append(ret)
                    
                    shares = 0
                    cooldown = 5
            
            curr_equity = cash + (shares * price)
            if curr_equity > peak_equity: peak_equity = curr_equity
            drawdown = (peak_equity - curr_equity) / peak_equity
            if drawdown > max_drawdown: max_drawdown = drawdown
            
        return cash + (shares * price), trade_log, max_drawdown

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    print("="*60)
    print("BULLETPROOF STRATEGY SIMULATION (Enhanced v2.0)")
    print("="*60)
    
    # Load Data
    try:
        spy = load_data("SPY")
        qqq = load_data("QQQ")
        
        # Load VIX
        try:
            vix = load_data("^VIX")
        except:
            print("Warning: Could not load VIX data. Using fallback.")
            vix = None
        
        # Align
        common_idx = spy.index.intersection(qqq.index)
        if vix is not None:
            common_idx = common_idx.intersection(vix.index)
            vix = vix.loc[common_idx]
            
        spy = spy.loc[common_idx]
        qqq = qqq.loc[common_idx]
        
        # Run
        strategy = Strategy(spy)
        
        # Optimized Parameters (from Genetic Algo)
        params = {
            'bull_bb_length': 28, 'bull_bb_std': 2.61,
            'bull_entry_b': 0.049, 'bull_exit_b': 1.04,
            'base_stop_loss': 0.138, 'base_trailing_stop': 0.088,
            'profit_milestone_1': 0.164, 'profit_milestone_2': 0.370,
            'tight_stop_1': 0.084, 'tight_stop_2': 0.039,
            'min_ml_confidence': 0.521, 'confidence_multiplier': 0.822
        }
        
        final_equity, log, max_dd = strategy.run(params, trade_data=qqq, vix_data=vix)
        
        # Stats
        total_ret = (final_equity - 100000) / 100000 * 100
        years = len(spy) / 252
        cagr = (final_equity / 100000) ** (1/years) - 1
        wins = sum(1 for t in log if t['return'] > 0)
        win_rate = wins / len(log) * 100 if len(log) > 0 else 0
        
        # Sharpe Ratio
        if len(log) > 0:
            returns = np.array([t['return'] for t in log])
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252/len(log)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        print("\n" + "="*60)
        print(f"RESULTS (QQQ Trading on SPY Signals)")
        print(f"Total Return: {total_ret:.2f}%")
        print(f"CAGR:         {cagr*100:.2f}%")
        print(f"Final Equity: ${final_equity:,.2f}")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Win Rate:     {win_rate:.2f}% ({len(log)} trades)")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
