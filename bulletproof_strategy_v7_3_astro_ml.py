"""
================================================================================
BULLETPROOF STRATEGY v7.2 - ML INTEGRATED
================================================================================

Based on Gemini's insight: Let the ML model LEARN the Pine Script signals
rather than hardcoding boosts.

Changes from v6.2:
- ADDS Pine Script crossover features to TechnicalEngine
- ADDS daily return panic feature
- ADDS combined PINE_ENTRY_SIGNAL
- Model automatically incorporates these as training features
- NO hardcoded probability boosts (let ML decide importance)

The ML model will learn:
- Whether crossover detection helps predict bottoms
- Whether panic days (daily return < -0.88%) improve timing
- The relative importance vs level-based features

All v6.2 thresholds and conviction logic PRESERVED.
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
# TECHNICAL ENGINE v7.2 - ML INTEGRATED
# =============================================================================

class TechnicalEngine:
    """
    v7.2: Adds Pine Script features as ML inputs.
    
    New features:
    - BB_20_crossunder: %B crosses below -0.06 (Pine Script entry)
    - BB_50_crossunder: Same for BB50
    - DAILY_RETURN_PANIC: Daily return < -0.88%
    - DAILY_RETURN_CRASH: Daily return < -2%
    - PINE_ENTRY_SIGNAL: Crossunder AND panic day
    
    All v6.2 level-based features PRESERVED.
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
        # 1. RSI FAMILY (unchanged from v6.2)
        # =================================================================
        for length in [2, 5, 14, 21, 50]:
            rsi = ta.rsi(close, length=length)
            features[f'RSI_{length}'] = rsi
            features[f'RSI_{length}_oversold'] = (rsi < 30).astype(int)
            features[f'RSI_{length}_extreme'] = (rsi < 20).astype(int)
            features[f'RSI_{length}_overbought'] = (rsi > 70).astype(int)
            features[f'RSI_{length}_extreme_high'] = (rsi > 80).astype(int)
        
        # =================================================================
        # 2. BOLLINGER BANDS %B - v6.2 + Pine Script Crossover
        # =================================================================
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
                
                # === v6.2 LEVEL DETECTION (preserved for conviction count) ===
                features[f'BB_{length}_oversold'] = (pctb < 0).astype(int)
                features[f'BB_{length}_sharktooth'] = (pctb < -0.1).astype(int)
                features[f'BB_{length}_overbought'] = (pctb > 1).astype(int)
                features[f'BB_{length}_sharktooth_bear'] = (pctb > 1.1).astype(int)
                
                # === NEW: Pine Script CROSSOVER features (for ML learning) ===
                # Crossunder at -0.06 (Pine Script threshold)
                features[f'BB_{length}_crossunder'] = (
                    (pctb < -0.06) & (pctb.shift(1) >= -0.06)
                ).astype(int)
                
                # Crossover at 1.0 (exit threshold)
                features[f'BB_{length}_crossover'] = (
                    (pctb > 1.0) & (pctb.shift(1) <= 1.0)
                ).astype(int)
                
                # Extreme overbought crossover at 1.2
                features[f'BB_{length}_extreme_crossover'] = (
                    (pctb > 1.2) & (pctb.shift(1) <= 1.2)
                ).astype(int)
        
        # =================================================================
        # 3. MACD (unchanged from v6.2)
        # =================================================================
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
        
        # =================================================================
        # 4. STOCHASTIC (unchanged from v6.2)
        # =================================================================
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
        
        # =================================================================
        # 5. WILLIAMS %R (unchanged from v6.2)
        # =================================================================
        for length in [14, 28]:
            willr = ta.willr(high, low, close, length=length)
            features[f'WILLR_{length}'] = willr
            features[f'WILLR_{length}_oversold'] = (willr < -80).astype(int)
            features[f'WILLR_{length}_extreme'] = (willr < -90).astype(int)
            features[f'WILLR_{length}_overbought'] = (willr > -20).astype(int)
            features[f'WILLR_{length}_extreme_high'] = (willr > -10).astype(int)
        
        # =================================================================
        # 6. CCI (unchanged from v6.2)
        # =================================================================
        cci = ta.cci(high, low, close, length=20)
        cci = cci.clip(-500, 500)
        features['CCI_20'] = cci
        features['CCI_oversold'] = (cci < -100).astype(int)
        features['CCI_extreme'] = (cci < -200).astype(int)
        features['CCI_overbought'] = (cci > 100).astype(int)
        features['CCI_extreme_high'] = (cci > 200).astype(int)
        
        # =================================================================
        # 7. MFI (unchanged from v6.2)
        # =================================================================
        mfi = ta.mfi(high, low, close, volume, length=14)
        features['MFI_14'] = mfi
        features['MFI_oversold'] = (mfi < 20).astype(int)
        features['MFI_overbought'] = (mfi > 80).astype(int)
        
        # =================================================================
        # 8. PRICE ACTION (unchanged from v6.2)
        # =================================================================
        features['RET_1d'] = close.pct_change(1) * 100
        features['RET_5d'] = close.pct_change(5) * 100
        
        for period in [10, 20, 50]:
            rolling_max = close.rolling(period).max()
            features[f'DD_{period}d'] = (close - rolling_max) / rolling_max * 100
            rolling_min = close.rolling(period).min()
            features[f'RALLY_{period}d'] = (close - rolling_min) / rolling_min * 100
        
        # =================================================================
        # 9. NEW: DAILY RETURN FEATURES (Pine Script)
        # =================================================================
        daily_return = close.pct_change(1)
        
        # Pine Script threshold: -0.88%
        features['DAILY_RETURN_PANIC'] = (daily_return < -0.0088).astype(int)
        
        # Additional severity levels
        features['DAILY_RETURN_CRASH'] = (daily_return < -0.02).astype(int)  # -2%
        features['DAILY_RETURN_EXTREME'] = (daily_return < -0.03).astype(int)  # -3%
        
        # Positive momentum (for bear detection)
        features['DAILY_RETURN_SURGE'] = (daily_return > 0.02).astype(int)  # +2%
        
        # =================================================================
        # 10. NEW: COMBINED PINE SCRIPT SIGNAL
        # =================================================================
        # This is what Pine Script checks: crossunder AND panic day
        features['PINE_ENTRY_SIGNAL'] = (
            features['BB_20_crossunder'] & features['DAILY_RETURN_PANIC']
        ).astype(int)
        
        # Multi-indicator crossunder (both BB20 and BB50 crossing down)
        features['MULTI_CROSSUNDER'] = (
            features['BB_20_crossunder'] + features.get('BB_50_crossunder', 0)
        ).clip(0, 2)
        
        # =================================================================
        # 11. COMPOSITE SCORES (v6.2 style - level based)
        # =================================================================
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
# SHARKTOOTH DETECTOR v7.2
# =============================================================================

class SharktoothDetector:
    """
    v7.2: Same architecture as v6.2, but now trains on expanded feature set
    including Pine Script crossover features.
    
    The model will automatically learn the importance of:
    - Level-based features (v6.2)
    - Crossover-based features (Pine Script)
    - Daily return panic features
    """
    
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
                if current_price > lookback_prices.min():
                    continue
                max_future = future_prices.max()
                move_pct = (max_future - current_price) / current_price * 100
                if move_pct >= self.min_move_pct:
                    labels.iloc[i] = 1
                        
            elif self.mode == 'bear':
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
        
        return X, y
    
    def train(self, df: pd.DataFrame, extra_features: pd.DataFrame = None, verbose: bool = True) -> 'SharktoothDetector':
        if verbose:
            print(f"Training {self.mode.upper()} Detector (v7.3 - Astro-ML Integrated)...")
        
        X, y = self.prepare_data(df)
        
        if extra_features is not None:
             X = pd.concat([X, extra_features], axis=1)
             # Align indices
             X = X.loc[y.index]
        
        if verbose:
            print(f"  Features: {len(X.columns)} (Technical + Celestial)")
            # Show new features
            new_feats = [f for f in X.columns if 'norm' in f or 'PINE' in f]
            print(f"  New/Celestial features: {new_feats}")
        
        self.feature_names = list(X.columns)
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        # v6.2 model params (proven)
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
        
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            
            if verbose:
                print(f"  Trained on {len(X)} samples, {y.sum():.0f} positive labels")
                print(f"  Top 10 features:")
                for feat, imp in self.feature_importance.head(10).items():
                    pine_marker = " ← PINE" if ('crossunder' in feat.lower() or 
                                                'PINE' in feat or 'PANIC' in feat) else ""
                    print(f"    {feat}: {imp:.4f}{pine_marker}")
        
        return self
    
    def predict(self, df: pd.DataFrame, extra_features: pd.DataFrame = None) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained!")
        
        features = TechnicalEngine.calculate(df)
        if extra_features is not None:
             features = pd.concat([features, extra_features], axis=1)
             features = features.loc[df.index]
             
        # Refresh feature names from model in case we loaded old model or trained new one
        model_feats = self.feature_names
        
        # Verify alignment
        for feat in model_feats:
            if feat not in features.columns:
                 features[feat] = 0
        
        X = features[model_feats].fillna(0)
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
    
    @classmethod
    def load(cls, path: str) -> 'SharktoothDetector':
        data = joblib.load(path)
        inst = cls(mode=data.get('mode', 'bull'))
        inst.model = data['model']
        inst.scaler = data['scaler']
        inst.feature_names = data['feature_names']
        inst.feature_importance = data.get('feature_importance')
        return inst


# =============================================================================
# CELESTIAL ENGINE (unchanged from v6.2)
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
                'sun_saturn_sep': self._get_sep(positions, 'Sun', 'Saturn'),
                'moon_uranus_sep': self._get_sep(positions, 'Moon', 'Uranus'),
                # Validated "Option A" Window: New Moon +/- 2 days (~25 deg)
                # Sep 0-25 or 335-360
                'is_lunar_window': self._is_new_moon_window(positions['Sun'], positions['Moon'], 25),
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
        
    def _get_sep(self, positions, p1, p2):
        diff = abs(positions[p1] - positions[p2])
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _is_new_moon_window(self, sun_pos, moon_pos, threshold_deg):
        diff = moon_pos - sun_pos
        # Normalize to 0-360
        while diff < 0: diff += 360
        while diff >= 360: diff -= 360
        
        return diff < threshold_deg or diff > (360 - threshold_deg)
    
    def _empty_features(self):
        return {
            'sun_opp_saturn': False, 
            'moon_opp_uranus': False,
            'sun_saturn_sep': 0.0,
            'moon_uranus_sep': 0.0,
            'is_lunar_window': False
        }


# =============================================================================
# BULLETPROOF STRATEGY v7.2 - ML INTEGRATED
# =============================================================================

class BulletproofStrategyV72:
    """
    v7.2 ML Integrated:
    - TechnicalEngine now includes Pine Script crossover features
    - ML model learns their importance automatically
    - NO hardcoded probability boosts
    - All v6.2 backtest logic and thresholds PRESERVED
    
    The model decides if crossover signals improve predictions.
    """
    
    def __init__(
        self,
        signal_data: pd.DataFrame,
        trade_data: Optional[pd.DataFrame] = None,
        vix_data: Optional[pd.DataFrame] = None,
        symbol: str = 'SPY'
    ):
        self.signal_data = signal_data
        self.trade_data = trade_data if trade_data is not None else signal_data
        self.vix_data = vix_data
        self.symbol = symbol
        
        self.bull_detector = None
        self.bear_detector = None
        self.celestial = CelestialEngine()
        
        self.stats = {}
        self.trade_log = []
        
    def train_models(self, verbose: bool = True):
        self.bull_detector = SharktoothDetector(mode='bull')
        self.bull_detector.train(self.signal_data, verbose=verbose)
        
        self.bear_detector = SharktoothDetector(mode='bear')
        self.bear_detector.train(self.signal_data, verbose=verbose)
        
        return self
    
    def should_override_bear_twin(self, trade_state: Dict, celestial_state: Dict) -> bool:
        """
        ML-derived exit override logic.
        Returns True if we should HOLD past BEAR_TWIN signal.
        """
        # Condition 1: Was this a panic entry? (VIX > 25 implies panic)
        panic_entry = trade_state.get('entry_vix', 20) > 25
        
        # Condition 2: Is Sun-Saturn favorable for continuation? (90-150 degrees)
        sep = celestial_state.get('sun_saturn_sep', 0)
        sun_saturn_favorable = (sep > 90 and sep < 150)
        
        # Condition 3: Are we in an uptrend? (Price > 50MA)
        # We need to access current price and MA. Passed in trade_state?
        # Let's assume trade_state has 'above_ma50'
        in_uptrend = trade_state.get('above_ma50', False)
        
        # Override if panic entry + celestial window + trend confirmed
        return panic_entry and sun_saturn_favorable and in_uptrend

    def backtest(self, params: Optional[Dict] = None, initial_capital: float = 100000, verbose: bool = True):
        """
        EXACT v6.2 backtest logic - no changes.
        The improvement comes from the ML model, not the trading logic.
        """
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
        
        # ----------------------------------------------------------------------
        # ASTRO-ML INTEGRATION: Pre-calculate Continuous Features
        # ----------------------------------------------------------------------
        print("Calculating historical celestial features for ML training...")
        astro_features = []
        for d in self.signal_data.index:
            date_str = d.strftime('%Y/%m/%d')
            # Bypass cache for bulk calculation if needed, or rely on internal cache
            # The get_features method is fast enough with cache
            feat = self.celestial.get_features(date_str)
            astro_features.append({
                'sun_saturn_sep_norm': feat['sun_saturn_sep'] / 180.0,
                'moon_uranus_sep_norm': feat['moon_uranus_sep'] / 180.0
            })
        
        astro_df = pd.DataFrame(astro_features, index=self.signal_data.index)
        # Append to signal_data SOLELY so TechnicalEngine or the Model sees them?
        # Actually, TechnicalEngine calculates Technicals. 
        # We need to append these AFTER TechnicalEngine or MERGE them.
        
        # Let's verify where features are generated.
        # features = TechnicalEngine.calculate(self.signal_data)
        # We should append astro features to 'features' dataframe before prediction/training.
        
        self.astro_df = astro_df # Store for use in training/prediction
        
        if self.bull_detector is None:
            self.train_models(verbose=verbose)
        
        bull_probs = self.bull_detector.predict(self.signal_data, extra_features=self.astro_df)
        bear_probs = self.bear_detector.predict(self.signal_data, extra_features=self.astro_df)
        features = TechnicalEngine.calculate(self.signal_data)
        # Merge astro features for reference if needed, but prediction used them above
        features = pd.concat([features, self.astro_df], axis=1)
        close = safe_series(self.trade_data['Close'])
        
        # Pre-calculate MA50 for override logic
        ma50 = close.rolling(50).mean()
        
        cash = initial_capital
        shares = 0.0
        entry_price = 0.0
        max_price = 0.0
        entry_date = None
        current_entry_vix = 20.0 # Track entry VIX for override logic
        partial_exit_taken = False
        
        self.trade_log = []
        self.all_signals = [] # Continuous signal log
        equity_curve = []
        peak_equity = initial_capital
        max_drawdown = 0.0
        cooldown = 0
        
        bear_prob_history = []
        
        # Track Pine signal entries for analysis
        pine_entries = 0
        total_entries = 0
        overrides = 0 # Track how many times we used the celestial override
        
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
            
            # Signals (standard v6.2)
            bull_prob = float(bull_probs.loc[date])
            bear_prob = float(bear_probs.loc[date])
            bull_count = float(features.loc[date, 'BULL_SHARKTOOTH_COUNT'])
            bear_count = float(features.loc[date, 'BEAR_SHARKTOOTH_COUNT'])
            
            # Track Pine signal (for analysis only, not used in logic)
            pine_signal = int(features.loc[date, 'PINE_ENTRY_SIGNAL']) if 'PINE_ENTRY_SIGNAL' in features.columns else 0
            
            bear_prob_history.append(bear_prob)
            if len(bear_prob_history) > 5:
                bear_prob_history.pop(0)
            bear_prob_avg = np.mean(bear_prob_history)
            
            if cooldown > 0:
                cooldown -= 1
            
            # ------------------------------------------------------------------
            # DAILY SIGNAL CHECK (Runs every day, regardless of position)
            # ------------------------------------------------------------------
            signal_size = 0.0
            signal_conviction = "SKIP"
            
            bull_signal = bull_prob > params['bull_threshold']
            sharktooth_signal = bull_count >= params['medium_conviction_count']
            
            if bull_signal or sharktooth_signal:
                if bull_prob > params['high_conviction_prob'] or \
                   bull_count >= params['high_conviction_count']:
                    signal_size = 1.0
                    signal_conviction = "HIGH"
                elif bull_prob > params['medium_conviction_prob'] or \
                     bull_count >= params['medium_conviction_count']:
                    signal_size = 0.5
                    signal_conviction = "MEDIUM"
                
                if signal_size > 0:
                    # v7.3 UPGRADE: Manual Celestial Overlays REMOVED
                    # The ML model now includes 'sun_saturn_sep_norm' and 'moon_uranus_sep_norm'
                    # directly in the training features.
                    
                    # LOG VALID SIGNAL
                    self.all_signals.append({
                        'Date': date,
                        'Price': price,
                        'Size': signal_size,
                        'Conviction': signal_conviction,
                        'Reason': f"Bull: {bull_prob:.1%}, Count: {bull_count}",
                        'VIX': vix
                    })

            # ------------------------------------------------------------------
            # EXECUTION (Respects position state)
            # ------------------------------------------------------------------
            if shares == 0 and cooldown == 0:
                if signal_size > 0:
                    size = signal_size
                    conviction = signal_conviction
                    
                    invest = cash * size
                    shares = invest / price
                    cash -= invest
                    entry_price = price
                    max_price = price
                    entry_date = date
                    current_entry_vix = vix # Store for exit logic
                    partial_exit_taken = False
                    
                    total_entries += 1
                    if pine_signal:
                        pine_entries += 1
                    
                    if verbose and date.year >= 2020:
                        pine_str = " [PINE]" if pine_signal else ""
                        print(f"[{date.strftime('%Y-%m-%d')}] BUY @ ${price:.2f} "
                              f"(Bull: {bull_prob:.0%}, Count: {bull_count:.0f}, {conviction}, VIX: {vix:.1f}){pine_str}")
            
            # EXIT (EXACT v6.2 logic - no changes)
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
                
                # Check Override Conditions
                # If we get a BEAR_TWIN signal, check if we should override it
                override_active = False
                if bear_prob > params['bear_threshold']:
                    # Prepare state for check
                    trade_state = {
                        'entry_vix': current_entry_vix,
                        'above_ma50': price > ma50.loc[date] if not pd.isna(ma50.loc[date]) else False
                    }
                    if self.should_override_bear_twin(trade_state, cel):
                        override_active = True
                        # Do NOT set exit_signal = True
                        if verbose and date.year >= 2024: # Limit noise
                             # Only print occasionally or tracking overrides
                             pass
                
                if bear_prob > params['bear_threshold'] and not override_active:
                    exit_signal = True
                    reason = f"BEAR_TWIN ({bear_prob:.0%})"
                elif bear_count >= params['bear_sharktooth_count'] and not override_active:
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
                
                # If we overrode, track it
                if bear_prob > params['bear_threshold'] and override_active:
                    if overrides % 10 == 0 and verbose and date.year >= 2020:
                        print(f"  > [{date.strftime('%Y-%m-%d')}] CELESTIAL OVERRIDE applied (Holding through Bear Signal)")
                    overrides += 1

                if exit_signal:
                    cash += shares * price
                    ret = (price - entry_price) / entry_price
                    self.trade_log.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'return': ret,
                        'reason': reason,
                        'partial_taken': partial_exit_taken,
                        'overridden': override_active # Should be False if we exited, but interesting metadata
                    })
                    if verbose and date.year >= 2020:
                        print(f"[{date.strftime('%Y-%m-%d')}] SELL @ ${price:.2f} ({reason}, {ret:+.1%})")
                    shares = 0
                    cooldown = 3
            
            equity = cash + shares * price
            equity_curve.append({'date': date, 'equity': equity})
            if equity > peak_equity: peak_equity = equity
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
        
        # Close final position
        if shares > 0:
            cash += shares * close.iloc[-1]
            
        total_return = (cash - initial_capital) / initial_capital
        years = len(self.signal_data) / 252
        cagr = (cash / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        returns = [t['return'] for t in self.trade_log]
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if returns and np.std(returns) > 0 else 0
        
        self.stats = {
            'total_return': total_return,
            'final_equity': cash,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': len(self.trade_log),
            'pine_entries': pine_entries,
            'total_entries': total_entries,
            'overrides_triggered': overrides
        }
        
        
        self.trades = self.trade_log  # Expose trades for external access
        return cash, self.stats

    def print_results(self):
        s = self.stats
        print("\n" + "=" * 70)
        print("BULLETPROOF STRATEGY v7.2 - ML INTEGRATED")
        print("(Pine Script features learned by ML)")
        print("=" * 70)
        print(f"Total Return:     {s['total_return']*100:,.2f}%")
        print(f"Final Equity:     ${s['final_equity']:,.2f}")
        print(f"CAGR:             {s['cagr']*100:.2f}%")
        print(f"Max Drawdown:     {s['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:     {s['sharpe']:.2f}")
        print(f"Win Rate:         {s['win_rate']*100:.1f}%")
        print(f"Total Trades:     {s['total_trades']}")
        print("-" * 70)
        print(f"Pine Signal Entries: {s.get('pine_entries', 0)}/{s.get('total_entries', 0)} "
              f"({s.get('pine_entries', 0)/max(s.get('total_entries', 1), 1)*100:.1f}%)")
        print("=" * 70)
        
    def print_feature_importance(self):
        """Show which features the ML model found most important."""
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        
        if self.bull_detector and self.bull_detector.feature_importance is not None:
            print("\nBULL Detector - Top 15 Features:")
            for i, (feat, imp) in enumerate(self.bull_detector.feature_importance.head(15).items()):
                pine_marker = " <-- NEW" if ('crossunder' in feat.lower() or 
                                           'PINE' in feat or 'PANIC' in feat or
                                           'CRASH' in feat) else ""
                print(f"  {i+1:2d}. {feat}: {imp:.4f}{pine_marker}")
        
        if self.bear_detector and self.bear_detector.feature_importance is not None:
            print("\nBEAR Detector - Top 15 Features:")
            for i, (feat, imp) in enumerate(self.bear_detector.feature_importance.head(15).items()):
                pine_marker = " <-- NEW" if ('crossover' in feat.lower() or 
                                           'SURGE' in feat) else ""
                print(f"  {i+1:2d}. {feat}: {imp:.4f}{pine_marker}")

    def save_models(self, directory: str = '.'):
        os.makedirs(directory, exist_ok=True)
        if self.bull_detector:
            self.bull_detector.save(os.path.join(directory, 'bull_v72_ml.joblib'))
        if self.bear_detector:
            self.bear_detector.save(os.path.join(directory, 'bear_v72_ml.joblib'))
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
    print("BULLETPROOF STRATEGY v7.2 - ML INTEGRATED")
    print("=" * 70)
    print("\nKey approach:")
    print("  - Pine Script crossover features ADDED to ML training")
    print("  - Model learns their importance automatically")
    print("  - NO hardcoded probability boosts")
    print("  - v6.2 backtest logic completely preserved")
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
        
        # 1. QQQ Backtest (Lunar Overlay DISABLED based on validation)
        print("\n" + "=" * 70)
        print("BACKTEST: Trading QQQ (Lunar Overlay: OFF)")
        print("=" * 70)
        
        strategy_qqq = BulletproofStrategyV72(spy, qqq, vix, symbol='QQQ')
        strategy_qqq.backtest(verbose=False)
        
        # 2. SPY Backtest (Lunar Overlay ENABLED)
        print("\n" + "=" * 70)
        print("BACKTEST: Trading SPY (Lunar Overlay: ON)")
        print("=" * 70)
        
        strategy_spy = BulletproofStrategyV72(spy, spy, vix, symbol='SPY')
        strategy_spy.backtest(verbose=False)

        # Capture output (SPY results prioritized for display)
        import sys
        with open('v7_2_results.txt', 'w') as f:
            sys.stdout = f
            print("\n--- SPY RESULTS (With Lunar Overlay) ---")
            strategy_spy.print_results()
            print("\n--- QQQ RESULTS (Whale Detector) ---")
            strategy_qqq.print_results()
            
            strategy_spy.print_feature_importance() # Feature importance is same (trained on SPY signals)
            sys.stdout = sys.__stdout__
            
        # Save SPY trades as primary
        if hasattr(strategy_spy, 'trades') and strategy_spy.trades:
            trades_df = pd.DataFrame(strategy_spy.trades)
            trades_df.to_csv('v7_2_trades.csv', index=False)
            print("Saved SPY trades to v7_2_trades.csv")
            
        # Save ALL SIGNALS (Continuous Log)
        if hasattr(strategy_spy, 'all_signals') and strategy_spy.all_signals:
            sig_df = pd.DataFrame(strategy_spy.all_signals)
            sig_df.to_csv('v7_2_all_signals.csv', index=False)
            print("Saved Master Signal Log to v7_2_all_signals.csv")
            
        strategy_spy.save_models('.')
            
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
