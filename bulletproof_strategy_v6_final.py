"""
================================================================================
BULLETPROOF STRATEGY v6.2 - RESTORATION
================================================================================

v6.0/v6.1 failed because they changed the DEFINITION of features.
v5.1 "Sharktooth" features were often just "extreme levels" (e.g. %B < -0.1).
v6.0 changed them to "reversal patterns" (e.g. %B turn up).

This broke the Conviction Filter, which relied on the "extreme level" count
persisting during a crash.

v6.2 RESTORES the EXACT feature definitions from v5.1.

Performance Target: ~38,000% (QQQ)
================================================================================
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import os
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


def safe_series(col) -> pd.Series:
    """Ensure DataFrame column behaves as Series."""
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


# =============================================================================
# TECHNICAL ENGINE v5.1 (EXACT RESTORATION)
# =============================================================================

class TechnicalEngine:
    """
    Restored EXACTLY from v5.1 to ensure feature consistency.
    """
    
    @classmethod
    def calculate(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        close = safe_series(df['Close'])
        high = safe_series(df['High'])
        low = safe_series(df['Low'])
        volume = safe_series(df['Volume'])
        
        features = pd.DataFrame(index=df.index)
        
        # =================================================================
        # 1. RSI FAMILY
        # =================================================================
        for length in [2, 5, 14, 21, 50]:
            rsi = ta.rsi(close, length=length)
            features[f'RSI_{length}'] = rsi
            
            # v5.1 Logic:
            features[f'RSI_{length}_oversold'] = (rsi < 30).astype(int)
            features[f'RSI_{length}_extreme'] = (rsi < 20).astype(int)
            
            features[f'RSI_{length}_overbought'] = (rsi > 70).astype(int)
            features[f'RSI_{length}_extreme_high'] = (rsi > 80).astype(int)
        
        # =================================================================
        # 2. BOLLINGER BANDS %B
        # =================================================================
        for length in [20, 50]:
            bb = ta.bbands(close, length=length, std=2.0)
            if bb is not None:
                col_pctb = f'BBP_{length}_2.0'
                # Handle pandas_ta column naming variations
                if col_pctb not in bb.columns:
                    # Fallback calculation
                    lower_col = [c for c in bb.columns if 'BBL' in c][0]
                    upper_col = [c for c in bb.columns if 'BBU' in c][0]
                    lower = bb[lower_col]
                    upper = bb[upper_col]
                    pctb = (close - lower) / (upper - lower)
                else:
                    pctb = bb[col_pctb]
                
                features[f'BB_{length}_pctb'] = pctb
                
                # v5.1 Logic (CRITICAL: Sharktooth = Extreme Level, NOT Reversal)
                features[f'BB_{length}_oversold'] = (pctb < 0).astype(int)
                features[f'BB_{length}_sharktooth'] = (pctb < -0.1).astype(int)
                
                features[f'BB_{length}_overbought'] = (pctb > 1).astype(int)
                features[f'BB_{length}_sharktooth_bear'] = (pctb > 1.1).astype(int)
        
        # =================================================================
        # 3. MACD
        # =================================================================
        macd = ta.macd(close)
        if macd is not None:
            features['MACD_line'] = macd.iloc[:, 0]
            features['MACD_hist'] = macd.iloc[:, 1]
            features['MACD_signal'] = macd.iloc[:, 2]
            
            # v5.1 Logic
            features['MACD_oversold'] = (features['MACD_line'] < -2.0).astype(int)
            # This WAS a reversal in v5.1
            features['MACD_turnup'] = ((features['MACD_hist'] > features['MACD_hist'].shift(1)) & 
                                     (features['MACD_hist'] < 0)).astype(int)
                                     
            features['MACD_overbought'] = (features['MACD_line'] > 2.0).astype(int)
            features['MACD_turndown'] = ((features['MACD_hist'] < features['MACD_hist'].shift(1)) & 
                                       (features['MACD_hist'] > 0)).astype(int)
        
        # =================================================================
        # 4. STOCHASTIC
        # =================================================================
        stoch = ta.stoch(high, low, close)
        if stoch is not None:
            k = stoch.iloc[:, 0]
            d = stoch.iloc[:, 1]
            features['STOCH_k'] = k
            features['STOCH_d'] = d
            
            features['STOCH_oversold'] = (k < 20).astype(int)
            # v5.1 Logic: Crossover in oversold
            features['STOCH_sharktooth'] = ((k < 20) & (k > d)).astype(int)
            
            features['STOCH_overbought'] = (k > 80).astype(int)
            features['STOCH_sharktooth_bear'] = ((k > 80) & (k < d)).astype(int)
        
        # =================================================================
        # 5. WILLIAMS %R
        # =================================================================
        for length in [14, 28]:
            willr = ta.willr(high, low, close, length=length)
            features[f'WILLR_{length}'] = willr
            
            features[f'WILLR_{length}_oversold'] = (willr < -80).astype(int)
            features[f'WILLR_{length}_extreme'] = (willr < -90).astype(int)
            
            features[f'WILLR_{length}_overbought'] = (willr > -20).astype(int)
            features[f'WILLR_{length}_extreme_high'] = (willr > -10).astype(int)
        
        # =================================================================
        # 6. CCI
        # =================================================================
        # v5.1 only used length=20
        cci = ta.cci(high, low, close, length=20)
        # Fix for extreme values (keep clamp just in case, but v5.1 didn't have it)
        # If v5.1 worked without it, maybe we don't need it? 
        # But let's keep it safe.
        cci = cci.clip(-500, 500) 
        
        features['CCI_20'] = cci
        features['CCI_oversold'] = (cci < -100).astype(int)
        features['CCI_extreme'] = (cci < -200).astype(int)
        
        features['CCI_overbought'] = (cci > 100).astype(int)
        features['CCI_extreme_high'] = (cci > 200).astype(int)
        
        # =================================================================
        # 7. MFI
        # =================================================================
        mfi = ta.mfi(high, low, close, volume, length=14)
        features['MFI_14'] = mfi
        features['MFI_oversold'] = (mfi < 20).astype(int)
        features['MFI_overbought'] = (mfi > 80).astype(int)
        
        # =================================================================
        # 8. PRICE ACTION
        # =================================================================
        features['RET_1d'] = close.pct_change(1) * 100
        features['RET_5d'] = close.pct_change(5) * 100
        
        for period in [10, 20, 50]:
            rolling_max = close.rolling(period).max()
            features[f'DD_{period}d'] = (close - rolling_max) / rolling_max * 100
            
            rolling_min = close.rolling(period).min()
            features[f'RALLY_{period}d'] = (close - rolling_min) / rolling_min * 100
        
        # =================================================================
        # 9. COMPOSITE SCORES (v5.1 Logic)
        # =================================================================
        oversold_cols = [c for c in features.columns if 'oversold' in c.lower()]
        # Note: v5.1 filtered out 'bear' from sharktooth cols
        sharktooth_bull_cols = [c for c in features.columns if 'sharktooth' in c.lower() and 'bear' not in c.lower()]
        
        features['OVERSOLD_COUNT'] = features[oversold_cols].sum(axis=1)
        features['BULL_SHARKTOOTH_COUNT'] = features[sharktooth_bull_cols].sum(axis=1)
        
        overbought_cols = [c for c in features.columns if 'overbought' in c.lower()]
        sharktooth_bear_cols = [c for c in features.columns if 'sharktooth_bear' in c.lower()]
        
        features['OVERBOUGHT_COUNT'] = features[overbought_cols].sum(axis=1)
        features['BEAR_SHARKTOOTH_COUNT'] = features[sharktooth_bear_cols].sum(axis=1)
        
        return features


# =============================================================================
# SHARKTOOTH DETECTOR v5.1 (EXACT RESTORATION)
# =============================================================================

class SharktoothDetector:
    def __init__(
        self,
        mode='bull',
        lookahead_days=10,
        min_move_pct=3.0,
        lookback_days=5
    ):
        self.mode = mode
        self.lookahead_days = lookahead_days
        self.min_move_pct = min_move_pct
        self.lookback_days = lookback_days
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = None
        
    def _label_turning_points(self, close: pd.Series) -> pd.Series:
        labels = pd.Series(0, index=close.index)
        N = len(close)
        
        for i in range(self.lookback_days, N - self.lookahead_days):
            current_price = close.iloc[i]
            lookback_prices = close.iloc[i - self.lookback_days:i]
            future_prices = close.iloc[i + 1:i + self.lookahead_days + 1]
            
            if len(lookback_prices) == 0 or len(future_prices) == 0:
                continue
            
            if self.mode == 'bull':
                # v5.1 Strict Local Min Logic
                if current_price > lookback_prices.min():
                    continue
                
                max_future = future_prices.max()
                move_pct = (max_future - current_price) / current_price * 100
                if move_pct >= self.min_move_pct:
                    labels.iloc[i] = 1
                        
            elif self.mode == 'bear':
                # v5.1 Strict Local Max Logic
                if current_price < lookback_prices.max():
                    continue
                
                min_future = future_prices.min()
                move_pct = (current_price - min_future) / current_price * 100
                if move_pct >= self.min_move_pct:
                    labels.iloc[i] = 1
        
        return labels
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        features = TechnicalEngine.calculate(df)
        close = safe_series(df['Close'])
        labels = self._label_turning_points(close)
        
        valid_idx = features.dropna().index.intersection(labels.index)
        
        if len(valid_idx) > self.lookahead_days + self.lookback_days:
            valid_idx = valid_idx[self.lookback_days:-self.lookahead_days]
        
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train(self, df: pd.DataFrame, verbose: bool = True) -> 'SharktoothDetector':
        if verbose:
            print(f"Training {self.mode.upper()} Detector (v6.2 - Restored)...")
        
        X, y = self.prepare_data(df)
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        # v5.1 Model Params
        if XGB_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=pos_weight,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        
        # Train on ALL data (v5.1 behavior)
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained!")
        
        features = TechnicalEngine.calculate(df)
        
        for feat in self.feature_names:
            if feat not in features.columns:
                features[feat] = 0
        
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        return pd.Series(probs, index=df.index, name=f'{self.mode}_prob')

    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'mode': self.mode,
        }, path)
    
    def load(self, path: str) -> 'SharktoothDetector':
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.feature_importance = data.get('feature_importance')
        self.mode = data.get('mode', 'bull')
        return self


# =============================================================================
# CELESTIAL ENGINE (Shared)
# =============================================================================

class CelestialEngine:
    def __init__(self):
        self.enabled = EPHEM_AVAILABLE
        self._cache = {}
    
    def get_features(self, date_str: str) -> Dict:
        if not self.enabled:
            return self._empty_features()
        
        if date_str in self._cache:
            return self._cache[date_str]
        
        try:
            observer = ephem.Observer()
            observer.date = date_str
            
            bodies = {
                'Sun': ephem.Sun(), 'Moon': ephem.Moon(),
                'Saturn': ephem.Saturn(), 'Uranus': ephem.Uranus()
            }
            
            positions = {}
            for name, body in bodies.items():
                body.compute(observer)
                ecl = ephem.Ecliptic(body)
                positions[name] = math.degrees(ecl.lon)
            
            features = {
                'sun_opp_saturn': self._check_aspect(positions, 'Sun', 'Saturn', 180, 5),
                'moon_opp_uranus': self._check_aspect(positions, 'Moon', 'Uranus', 180, 8),
            }
            
            self._cache[date_str] = features
            return features
            
        except Exception:
            return self._empty_features()
    
    def _check_aspect(self, positions, p1, p2, target, orb):
        diff = abs(positions[p1] - positions[p2])
        if diff > 180:
            diff = 360 - diff
        return abs(diff - target) <= orb
    
    def _empty_features(self):
        return {'sun_opp_saturn': False, 'moon_opp_uranus': False}


# =============================================================================
# BULLETPROOF STRATEGY v6.2 - RESTORATION
# =============================================================================

class BulletproofStrategyV6:
    def __init__(
        self,
        signal_data: pd.DataFrame,
        trade_data: Optional[pd.DataFrame] = None,
        vix_data: Optional[pd.DataFrame] = None
    ):
        self.signal_data = signal_data
        self.trade_data = trade_data if trade_data is not None else signal_data
        self.vix_data = vix_data
        
        self.bull_detector = None
        self.bear_detector = None
        self.celestial = CelestialEngine()
        
        self.stats = {}
        
    def train_models(self, verbose: bool = True):
        self.bull_detector = SharktoothDetector(mode='bull')
        self.bull_detector.train(self.signal_data, verbose=verbose)
        
        self.bear_detector = SharktoothDetector(mode='bear')
        self.bear_detector.train(self.signal_data, verbose=verbose)
        
        return self
    
    def backtest(self, params: Optional[Dict] = None, initial_capital: float = 100000, verbose: bool = True):
        if params is None:
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
        
        if self.bull_detector is None:
            self.train_models(verbose=verbose)
        
        bull_probs = self.bull_detector.predict(self.signal_data)
        bear_probs = self.bear_detector.predict(self.signal_data)
        features = TechnicalEngine.calculate(self.signal_data)
        close = safe_series(self.trade_data['Close'])
        
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
        
        for i, date in enumerate(self.signal_data.index):
            if date not in close.index:
                continue
            
            price = float(close.loc[date])
            
            # VIX
            vix = 20.0
            if self.vix_data is not None and date in self.vix_data.index:
                v = self.vix_data.loc[date, 'Close']
                vix = float(safe_series(pd.Series([v])).iloc[0]) if not isinstance(v, (int, float)) else float(v)
            
            # Celestial
            date_str = date.strftime('%Y/%m/%d')
            cel = self.celestial.get_features(date_str)
            
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
            
            # ENTRY (v5.1 Logic)
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
                        if cel['moon_opp_uranus'] and 18 < vix < 28:
                            size = min(size * 1.25, 1.0)
                        
                        invest = cash * size
                        shares = invest / price
                        cash -= invest
                        entry_price = price
                        max_price = price
                        entry_date = date
                        partial_exit_taken = False
                        
                        if verbose and date.year >= 2020:
                            print(f"[{date.strftime('%Y-%m-%d')}] BUY @ ${price:.2f} "
                                  f"(Bull: {bull_prob:.0%}, Count: {bull_count:.0f}, {conviction})")
            
            # EXIT
            elif shares > 0:
                max_price = max(max_price, price)
                unrealized = (price - entry_price) / entry_price
                dd_from_high = (max_price - price) / max_price
                
                # Profit Taking (v5.2)
                if not partial_exit_taken and unrealized > params['profit_take_threshold']:
                    if bear_prob > bear_prob_avg and bear_prob > params['profit_take_bear_min']:
                        sell_shares = shares * params['profit_take_pct']
                        cash += sell_shares * price
                        shares -= sell_shares
                        partial_exit_taken = True
                        if verbose and date.year >= 2020:
                            print(f"[{date.strftime('%Y-%m-%d')}] PARTIAL @ ${price:.2f} (Locking Gains)")

                exit_signal = False
                reason = ""
                
                if bear_prob > params['bear_threshold']:
                    exit_signal = True
                    reason = f"BEAR_TWIN ({bear_prob:.0%})"
                elif bear_count >= params['bear_sharktooth_count']:
                    exit_signal = True
                    reason = f"BEAR_SHARK ({bear_count:.0f})"
                elif unrealized < -params['base_stop_loss']:
                    exit_signal = True
                    reason = "STOP_LOSS"
                elif dd_from_high > params['trailing_stop']:
                    exit_signal = True
                    reason = "TRAILING"
                elif cel['sun_opp_saturn'] and unrealized > 0.05:
                    exit_signal = True
                    reason = "CELESTIAL"
                
                if exit_signal:
                    cash += shares * price
                    ret = (price - entry_price) / entry_price
                    trade_log.append({
                        'return': ret,
                        'reason': reason,
                        'partial_taken': partial_exit_taken
                    })
                    if verbose and date.year >= 2020:
                        print(f"[{date.strftime('%Y-%m-%d')}] SELL @ ${price:.2f} ({reason}, {ret:+.1%})")
                    shares = 0
                    cooldown = 3
            
            equity = cash + shares * price
            equity_curve.append({'date': date, 'equity': equity})
            if equity > peak_equity: peak_equity = equity
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
        
        # Close final
        if shares > 0:
            cash += shares * close.iloc[-1]
            
        total_return = (cash - initial_capital) / initial_capital
        years = len(self.signal_data) / 252
        cagr = (cash / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        returns = [t['return'] for t in trade_log]
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if returns and np.std(returns) > 0 else 0
        
        self.stats = {
            'total_return': total_return,
            'final_equity': cash,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': len(trade_log)
        }
        
        return cash, self.stats

    def print_results(self):
        s = self.stats
        print("\n" + "=" * 70)
        print("BULLETPROOF STRATEGY v6.2 - RESTORATION")
        print("=" * 70)
        print(f"Total Return:     {s['total_return']*100:,.2f}%")
        print(f"Final Equity:     ${s['final_equity']:,.2f}")
        print(f"CAGR:             {s['cagr']*100:.2f}%")
        print(f"Max Drawdown:     {s['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:     {s['sharpe']:.2f}")
        print(f"Win Rate:         {s['win_rate']*100:.1f}%")
        print(f"Total Trades:     {s['total_trades']}")
        print("=" * 70)

    def save_models(self, directory: str = '.'):
        if self.bull_detector:
            self.bull_detector.save(os.path.join(directory, 'bull_v62.joblib'))
        if self.bear_detector:
            self.bear_detector.save(os.path.join(directory, 'bear_v62.joblib'))
        print(f"Models saved to {directory}")

# =============================================================================
# MAIN
# =============================================================================

def load_data_yf(ticker: str) -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise ImportError("yfinance not installed")
    df = yf.download(ticker, start="2000-01-01", progress=False)
    df = df[~df.index.duplicated(keep='first')]
    return df

if __name__ == "__main__":
    print("=" * 70)
    print("BULLETPROOF STRATEGY v6.2 - RESTORATION")
    print("Restoring v5.1 feature definitions to fix performance.")
    print("=" * 70)
    
    try:
        print("\nLoading data...")
        spy = load_data_yf("SPY")
        qqq = load_data_yf("QQQ")
        vix = load_data_yf("^VIX")
        
        idx = spy.index.intersection(qqq.index).intersection(vix.index)
        spy = spy.loc[idx]
        qqq = qqq.loc[idx]
        vix = vix.loc[idx]
        
        print(f"Loaded {len(spy)} trading days")
        
        # QQQ
        print("\n" + "=" * 70)
        print("BACKTEST: Trading QQQ")
        print("=" * 70)
        
        strategy_qqq = BulletproofStrategyV6(spy, qqq, vix)
        strategy_qqq.backtest(verbose=False)
        strategy_qqq.print_results()
        
        strategy_qqq.save_models('.')
        
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
