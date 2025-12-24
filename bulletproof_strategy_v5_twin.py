import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import os
import math
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# Optional: XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Optional: Ephem for celestial data
try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False


# =============================================================
# TECHNICAL ENGINE (Shared)
# =============================================================

class TechnicalEngine:
    """
    Generates massive feature set for ML models.
    Calculates ~150 features across multiple timeframes.
    """
    
    @staticmethod
    def calculate_features(df):
        df = df.copy()
        
        # Ensure 'close' is a Series
        if 'Close' in df.columns:
            close = df['Close']
        else:
            close = df['Adj Close']
            
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        open_ = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
        
        features = pd.DataFrame(index=df.index)
        
        # -----------------------------------------------------------
        # 1. RSI FAMILY (Momentum)
        # -----------------------------------------------------------
        for length in [2, 5, 14, 21, 50]:
            rsi = ta.rsi(close, length=length)
            features[f'RSI_{length}'] = rsi
            
            # Sharktooth: RSI < 30 (Oversold)
            features[f'RSI_{length}_oversold'] = (rsi < 30).astype(int)
            features[f'RSI_{length}_extreme'] = (rsi < 20).astype(int)
            
            # Bearish Sharktooth: RSI > 70 (Overbought)
            features[f'RSI_{length}_overbought'] = (rsi > 70).astype(int)
            features[f'RSI_{length}_extreme_high'] = (rsi > 80).astype(int)

        # -----------------------------------------------------------
        # 2. BOLLINGER BANDS (%B)
        # -----------------------------------------------------------
        for length in [20, 50]:
            bb = ta.bbands(close, length=length, std=2.0)
            # pctB = (price - lower) / (upper - lower)
            col_pctb = f'BBP_{length}_2.0'
            if col_pctb in bb.columns:
                features[f'BB_{length}_pctb'] = bb[col_pctb]
                
                # Sharktooth: %B < 0 (Price below lower band)
                features[f'BB_{length}_oversold'] = (bb[col_pctb] < 0).astype(int)
                features[f'BB_{length}_sharktooth'] = (bb[col_pctb] < -0.1).astype(int)
                
                # Bearish Sharktooth: %B > 1 (Price above upper band)
                features[f'BB_{length}_overbought'] = (bb[col_pctb] > 1).astype(int)
                features[f'BB_{length}_sharktooth_bear'] = (bb[col_pctb] > 1.1).astype(int)

        # -----------------------------------------------------------
        # 3. MACD (Trend/Momentum)
        # -----------------------------------------------------------
        macd = ta.macd(close)
        # MACD_12_26_9, MACDh_12_26_9 (hist), MACDs_12_26_9 (signal)
        if macd is not None:
            features['MACD_line'] = macd.iloc[:, 0]
            features['MACD_hist'] = macd.iloc[:, 1]
            features['MACD_signal'] = macd.iloc[:, 2]
            
            # Sharktooth: Histogram turning up from deep negative
            features['MACD_oversold'] = (features['MACD_line'] < -2.0).astype(int)
            features['MACD_turnup'] = ((features['MACD_hist'] > features['MACD_hist'].shift(1)) & 
                                     (features['MACD_hist'] < 0)).astype(int)
                                     
            # Bearish Sharktooth: Histogram turning down from deep positive
            features['MACD_overbought'] = (features['MACD_line'] > 2.0).astype(int)
            features['MACD_turndown'] = ((features['MACD_hist'] < features['MACD_hist'].shift(1)) & 
                                       (features['MACD_hist'] > 0)).astype(int)

        # -----------------------------------------------------------
        # 4. STOCHASTIC OSCILLATOR
        # -----------------------------------------------------------
        stoch = ta.stoch(high, low, close)
        if stoch is not None:
            k = stoch.iloc[:, 0] # %K
            d = stoch.iloc[:, 1] # %D
            features['STOCH_k'] = k
            features['STOCH_d'] = d
            
            features['STOCH_oversold'] = (k < 20).astype(int)
            features['STOCH_sharktooth'] = ((k < 20) & (k > d)).astype(int) # Bullish cross in oversold
            
            features['STOCH_overbought'] = (k > 80).astype(int)
            features['STOCH_sharktooth_bear'] = ((k > 80) & (k < d)).astype(int) # Bearish cross in overbought

        # -----------------------------------------------------------
        # 5. WILLIAMS %R
        # -----------------------------------------------------------
        for length in [14, 28]:
            willr = ta.willr(high, low, close, length=length)
            features[f'WILLR_{length}'] = willr
            
            features[f'WILLR_{length}_oversold'] = (willr < -80).astype(int)
            features[f'WILLR_{length}_extreme'] = (willr < -90).astype(int)
            
            features[f'WILLR_{length}_overbought'] = (willr > -20).astype(int)
            features[f'WILLR_{length}_extreme_high'] = (willr > -10).astype(int)

        # -----------------------------------------------------------
        # 6. CCI (Commodity Channel Index)
        # -----------------------------------------------------------
        cci = ta.cci(high, low, close, length=20)
        features['CCI_20'] = cci
        features['CCI_oversold'] = (cci < -100).astype(int)
        features['CCI_extreme'] = (cci < -200).astype(int)
        
        features['CCI_overbought'] = (cci > 100).astype(int)
        features['CCI_extreme_high'] = (cci > 200).astype(int)

        # -----------------------------------------------------------
        # 7. MFI (Money Flow Index)
        # -----------------------------------------------------------
        mfi = ta.mfi(high, low, close, volume, length=14)
        features['MFI_14'] = mfi
        features['MFI_oversold'] = (mfi < 20).astype(int)
        features['MFI_overbought'] = (mfi > 80).astype(int)

        # -----------------------------------------------------------
        # 8. PRICE ACTION / RETURNS
        # -----------------------------------------------------------
        features['RET_1d'] = close.pct_change(1) * 100
        features['RET_5d'] = close.pct_change(5) * 100
        
        # Drawdown from recent high (for Bottoms)
        for period in [10, 20, 50]:
            rolling_max = close.rolling(period).max()
            features[f'DD_{period}d'] = (close - rolling_max) / rolling_max * 100
            
        # Rally from recent low (for Tops)
        for period in [10, 20, 50]:
            rolling_min = close.rolling(period).min()
            features[f'RALLY_{period}d'] = (close - rolling_min) / rolling_min * 100
        
        # -----------------------------------------------------------
        # 9. COMPOSITE SCORES
        # -----------------------------------------------------------
        # Bullish Scores
        oversold_cols = [c for c in features.columns if 'oversold' in c.lower()]
        sharktooth_bull_cols = [c for c in features.columns if 'sharktooth' in c.lower() and 'bear' not in c.lower()]
        
        features['OVERSOLD_COUNT'] = features[oversold_cols].sum(axis=1)
        features['BULL_SHARKTOOTH_COUNT'] = features[sharktooth_bull_cols].sum(axis=1)
        
        # Bearish Scores
        overbought_cols = [c for c in features.columns if 'overbought' in c.lower()]
        sharktooth_bear_cols = [c for c in features.columns if 'sharktooth_bear' in c.lower()]
        
        features['OVERBOUGHT_COUNT'] = features[overbought_cols].sum(axis=1)
        features['BEAR_SHARKTOOTH_COUNT'] = features[sharktooth_bear_cols].sum(axis=1)
        
        return features


# =============================================================
# SHARKTOOTH DETECTOR (Generic Twin)
# =============================================================

class SharktoothDetector:
    """
    ML-based pattern detector for turning points.
    Can be configured as a Bottom Detector (Bull) or Top Detector (Bear).
    """
    
    def __init__(
        self,
        mode='bull',  # 'bull' or 'bear'
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
        
    def _label_turning_points(self, df, features):
        """Label turning points based on future price action."""
        if 'Close' in df.columns:
            close = df['Close']
        else:
            close = df['Adj Close']
            
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        labels = pd.Series(0, index=df.index)
        
        for i in range(self.lookback_days, len(df) - self.lookahead_days):
            current_price = close.iloc[i]
            lookback_prices = close.iloc[max(0, i - self.lookback_days):i]
            future_prices = close.iloc[i + 1:i + self.lookahead_days + 1]
            
            if len(lookback_prices) == 0 or len(future_prices) == 0:
                continue

            if self.mode == 'bull':
                # BOTTOM DETECTION
                # 1. Must be a local minimum (optional but helps precision)
                if current_price > lookback_prices.min():
                    continue
                
                # 2. Must recover by min_move_pct
                max_future = future_prices.max()
                move_pct = (max_future - current_price) / current_price * 100
                
                if move_pct >= self.min_move_pct:
                    labels.iloc[i] = 1
                    
            elif self.mode == 'bear':
                # TOP DETECTION
                # 1. Must be a local maximum
                if current_price < lookback_prices.max():
                    continue
                
                # 2. Must decline by min_move_pct
                min_future = future_prices.min()
                move_pct = (current_price - min_future) / current_price * 100
                
                if move_pct >= self.min_move_pct:
                    labels.iloc[i] = 1
        
        return labels
    
    def prepare_data(self, df):
        """Prepare features and labels."""
        features = TechnicalEngine.calculate_features(df)
        labels = self._label_turning_points(df, features)
        
        # Remove non-feature columns
        drop_cols = ['close']
        feature_cols = [c for c in features.columns if c not in drop_cols]
        
        # Get valid index (no NaN)
        valid_idx = features[feature_cols].dropna().index
        valid_idx = valid_idx.intersection(labels.dropna().index)
        
        # Remove lookahead bias period
        if len(valid_idx) > self.lookahead_days + self.lookback_days:
            valid_idx = valid_idx[self.lookback_days:-self.lookahead_days]
        
        X = features.loc[valid_idx, feature_cols]
        y = labels.loc[valid_idx]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, df, verbose=True):
        """Train the model."""
        if verbose:
            print(f"Training {self.mode.upper()} Sharktooth Detector...")
        
        X, y = self.prepare_data(df)
        
        if verbose:
            print(f"  Samples: {len(X)}")
            print(f"  Targets found: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        
        # Class imbalance weight
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Model selection
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
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        
        self.model.fit(X_scaled, y)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_') and verbose:
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            
            print(f"\n  Top 5 {self.mode.upper()} Features:")
            for feat, imp in self.feature_importance.head(5).items():
                print(f"    {feat}: {imp:.4f}")
        
        return self
    
    def predict(self, df):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        features = TechnicalEngine.calculate_features(df)
        
        # Ensure all features exist
        for feat in self.feature_names:
            if feat not in features.columns:
                features[feat] = 0
        
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        return pd.Series(probs, index=df.index, name=f'{self.mode}_prob')


# =============================================================
# CELESTIAL ENGINE (Shared)
# =============================================================

class CelestialEngine:
    """Celestial timing signals."""
    
    def __init__(self):
        self.enabled = EPHEM_AVAILABLE
        self._cache = {}
    
    def get_features(self, date_str):
        if not self.enabled:
            return self._empty_features()
        
        if date_str in self._cache:
            return self._cache[date_str]
        
        try:
            observer = ephem.Observer()
            observer.date = date_str
            
            bodies = {
                'Sun': ephem.Sun(), 'Moon': ephem.Moon(),
                'Mars': ephem.Mars(), 'Jupiter': ephem.Jupiter(),
                'Saturn': ephem.Saturn(), 'Uranus': ephem.Uranus()
            }
            
            positions = {}
            for name, body in bodies.items():
                body.compute(observer)
                ecl = ephem.Ecliptic(body)
                positions[name] = math.degrees(ecl.lon)
            
            # Lunar phase
            diff = positions['Moon'] - positions['Sun']
            if diff < 0:
                diff += 360
            lunar_phase = diff / 360.0
            
            features = {
                'lunar_phase': lunar_phase,
                'is_new_moon': lunar_phase < 0.25 or lunar_phase > 0.75,
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
        return {
            'lunar_phase': 0.5, 'is_new_moon': False,
            'sun_opp_saturn': False, 'moon_opp_uranus': False
        }


# =============================================================
# TWIN STRATEGY v5
# =============================================================

class BulletproofStrategyV5:
    """
    Bull/Bear Twin Strategy.
    Uses two ML models: one for bottoms (Bull), one for tops (Bear).
    """
    
    def __init__(self, signal_data, trade_data=None, vix_data=None):
        self.signal_data = signal_data
        self.trade_data = trade_data if trade_data is not None else signal_data
        self.vix_data = vix_data
        
        self.bull_detector = None
        self.bear_detector = None
        self.celestial = CelestialEngine()
        
        self.stats = {}
    
    def train_models(self, verbose=True):
        """Train both Bull and Bear models."""
        if verbose:
            print("Training Twin Models...")
            print("=" * 60)
        
        # Bull Model (Bottoms)
        self.bull_detector = SharktoothDetector(
            mode='bull',
            lookahead_days=10,
            min_move_pct=3.0,
            lookback_days=5
        )
        self.bull_detector.train(self.signal_data, verbose=verbose)
        
        # Bear Model (Tops)
        self.bear_detector = SharktoothDetector(
            mode='bear',
            lookahead_days=10,
            min_move_pct=3.0,
            lookback_days=5
        )
        self.bear_detector.train(self.signal_data, verbose=verbose)
        
        return self
    
    def backtest(self, params=None, initial_capital=100000, verbose=True):
        if params is None:
            params = {
                'bull_threshold': 0.45,        # Entry threshold
                'bear_threshold': 0.60,        # Exit threshold (Bearish signal)
                'base_stop_loss': 0.12,
                'trailing_stop': 0.08,
            }
        
        if self.bull_detector is None or self.bear_detector is None:
            self.train_models(verbose=verbose)
        
        # Generate Signals
        bull_probs = self.bull_detector.predict(self.signal_data)
        bear_probs = self.bear_detector.predict(self.signal_data)
        features = TechnicalEngine.calculate_features(self.signal_data)
        
        # Trading Simulation
        cash = initial_capital
        shares = 0
        entry_price = 0
        max_price = 0
        
        trade_log = []
        equity_curve = []
        peak_equity = initial_capital
        max_drawdown = 0
        cooldown = 0
        
        if 'Close' in self.trade_data.columns:
            close = self.trade_data['Close']
        else:
            close = self.trade_data['Adj Close']
            
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        # Iterate
        for i, date in enumerate(self.signal_data.index):
            if date not in close.index:
                continue
                
            price = float(close.loc[date])
            
            # VIX
            vix = 20.0
            if self.vix_data is not None and date in self.vix_data.index:
                v = self.vix_data.loc[date, 'Close']
                vix = float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
            
            # Celestial
            date_str = date.strftime('%Y/%m/%d')
            cel = self.celestial.get_features(date_str)
            
            if cooldown > 0:
                cooldown -= 1
                
            # Signals
            bull_prob = float(bull_probs.loc[date])
            bear_prob = float(bear_probs.loc[date])
            
            # ENTRY (Bullish)
            if shares == 0 and cooldown == 0:
                # Bull Signal OR (Sharktooth Count High AND VIX High)
                bull_signal = bull_prob > params['bull_threshold']
                sharktooth_bull = features.loc[date, 'BULL_SHARKTOOTH_COUNT'] >= 3
                
                if bull_signal or sharktooth_bull:
                    # Conviction Filter (v5.1)
                    # High Conviction: >70% Prob OR >=4 Indicators -> Full Size
                    # Medium Conviction: >50% Prob OR >=3 Indicators -> Half Size
                    # Low Conviction: Skip
                    
                    sharktooth_count = features.loc[date, 'BULL_SHARKTOOTH_COUNT']
                    size = 0.0
                    
                    if bull_prob > 0.70 or sharktooth_count >= 4:
                        size = 1.0
                    elif bull_prob > 0.50 or sharktooth_count >= 3:
                        size = 0.5
                        
                    if size > 0:
                        if cel['moon_opp_uranus']: size = min(size * 1.25, 1.0) # Quiet Bottom Bonus
                        
                        invest = cash * size
                        shares = invest / price
                        cash -= invest
                        
                        entry_price = price
                        max_price = price
                        entry_date = date
                        partial_exit_taken = False
                    
                    if verbose and date.year >= 2010 and size > 0:
                        print(f"[{date.strftime('%Y-%m-%d')}] BUY @ {price:.2f} (Bull: {bull_prob:.2f})")

            # EXIT (Bearish / Risk)
            elif shares > 0:
                max_price = max(max_price, price)
                unrealized = (price - entry_price) / entry_price
                dd_from_high = (max_price - price) / max_price
                
                # Profit Taking (v5.2)
                # If >20% gain AND Bear Prob rising (but < threshold), take 25% off
                if not partial_exit_taken and unrealized > 0.20:
                    bear_avg_5d = bear_probs.iloc[i-5:i].mean() if i > 5 else 0
                    if bear_prob > bear_avg_5d and bear_prob > 0.30: # Rising risk
                        sell_shares = shares * 0.25
                        cash += sell_shares * price
                        shares -= sell_shares
                        partial_exit_taken = True
                        if verbose and date.year >= 2010:
                            print(f"[{date.strftime('%Y-%m-%d')}] PARTIAL SELL @ {price:.2f} (Locking Gains, Ret: {unrealized*100:.1f}%)")

                # Bear Signal Exit
                bear_signal = bear_prob > params['bear_threshold']
                sharktooth_bear = features.loc[date, 'BEAR_SHARKTOOTH_COUNT'] >= 3
                
                exit_signal = False
                reason = ""
                
                if bear_signal:
                    exit_signal = True
                    reason = f"BEAR_TWIN ({bear_prob:.2f})"
                elif sharktooth_bear:
                    exit_signal = True
                    reason = "BEAR_SHARKTOOTH"
                elif unrealized < -params['base_stop_loss']:
                    exit_signal = True
                    reason = "STOP_LOSS"
                elif dd_from_high > params['trailing_stop']:
                    exit_signal = True
                    reason = "TRAILING"
                elif cel['sun_opp_saturn'] and unrealized > 0.05:
                    exit_signal = True
                    reason = "CELESTIAL_PEAK"
                    
                if exit_signal:
                    cash += shares * price
                    ret = (price - entry_price) / entry_price
                    trade_log.append({'return': ret, 'reason': reason})
                    shares = 0
                    cooldown = 3
                    
                    if verbose and date.year >= 2010:
                        print(f"[{date.strftime('%Y-%m-%d')}] SELL @ {price:.2f} ({reason}, Ret: {ret*100:.1f}%)")

            # Equity
            equity = cash + shares * price
            equity_curve.append({'date': date, 'equity': equity})
            if equity > peak_equity: peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, dd)

        # Stats
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
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'trades': len(trade_log)
        }
        
        return cash, self.stats

    def print_results(self):
        s = self.stats
        print("\n" + "=" * 60)
        print("BULL/BEAR TWIN STRATEGY v5.0 - RESULTS")
        print("=" * 60)
        print(f"Total Return:    {s['total_return']*100:.2f}%")
        print(f"CAGR:            {s['cagr']*100:.2f}%")
        print(f"Max Drawdown:    {s['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:    {s['sharpe']:.2f}")
        print(f"Win Rate:        {s['win_rate']*100:.1f}%")
        print(f"Total Trades:    {s['trades']}")
        print("=" * 60)

    def get_current_signal(self):
        if self.bull_detector is None:
            self.train_models(verbose=False)
            
        bull_prob = float(self.bull_detector.predict(self.signal_data).iloc[-1])
        bear_prob = float(self.bear_detector.predict(self.signal_data).iloc[-1])
        
        return {
            'date': self.signal_data.index[-1],
            'bull_prob': bull_prob,
            'bear_prob': bear_prob,
            'signal': 'BUY' if bull_prob > 0.5 else ('SELL' if bear_prob > 0.6 else 'HOLD')
        }

# =============================================================
# MAIN
# =============================================================

def load_data_yf(ticker):
    df = yf.download(ticker, start="2000-01-01", progress=False)
    df = df[~df.index.duplicated(keep='first')]
    return df

if __name__ == "__main__":
    print("Initializing Bull/Bear Twin Strategy v5.0...")
    
    try:
        spy = load_data_yf("SPY")
        qqq = load_data_yf("QQQ")
        vix = load_data_yf("^VIX")
        
        # Align
        idx = spy.index.intersection(qqq.index).intersection(vix.index)
        spy = spy.loc[idx]
        qqq = qqq.loc[idx]
        vix = vix.loc[idx]
        
        # Run 1: Trade QQQ
        print("\n--- TRADING QQQ ---")
        strategy_qqq = BulletproofStrategyV5(spy, qqq, vix)
        strategy_qqq.backtest(verbose=False) # Turn off verbose to reduce noise
        strategy_qqq.print_results()
        
        # Run 2: Trade SPY
        print("\n--- TRADING SPY ---")
        strategy_spy = BulletproofStrategyV5(spy, spy, vix)
        strategy_spy.backtest(verbose=False)
        strategy_spy.print_results()
        
        sig = strategy_qqq.get_current_signal()
        print(f"\nCURRENT SIGNAL ({sig['date'].strftime('%Y-%m-%d')})")
        print(f"Bull Probability: {sig['bull_prob']*100:.1f}%")
        print(f"Bear Probability: {sig['bear_prob']*100:.1f}%")
        print(f"Action: {sig['signal']}")
        
        # Save results (QQQ)
        with open('v5_results.txt', 'w') as f:
            f.write(f"QQQ Total Return: {strategy_qqq.stats['total_return']*100:.2f}%\n")
            f.write(f"SPY Total Return: {strategy_spy.stats['total_return']*100:.2f}%\n")
            f.write(f"Bull Prob: {sig['bull_prob']*100:.1f}%\n")
            f.write(f"Bear Prob: {sig['bear_prob']*100:.1f}%\n")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
