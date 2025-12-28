
"""
================================================================================
BULLETPROOF STRATEGY v8.0 - SEASONALITY REGIME INTEGRATED
================================================================================

Based on v7.5 Eclipse + v8.0 Seasonality Research:
- INTEGRATES Turn of Month (+4x Edge)
- INTEGRATES Halloween/September Seasonality
- INTEGRATES VIX Term Structure (Backwardation = Fear)
- KEEPS all v7.5 features (Eclipse, Vedic, Stellium)

Changes from v7.5:
- ADDS MarketRegimeEngine
- ADDS VIX/VIX3M Data Loading
- UPDATES ML Model with Regime Features

Validation Target:
- Must maintain > 141,000% Return
- Must maintain > 7.0 Sharpe
================================================================================
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import os
import math
import calendar
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

from market_regime import MarketRegimeEngine

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
    print("Warning: ephem not installed. Celestial features will use defaults.")

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
# CELESTIAL ENGINE v7.5 (Copied for Stability)
# =============================================================================

class CelestialEngine:
    """
    v7.5: Complete celestial calculation engine with Eclipse support.
    """
    
    SPREAD_PLANETS = ['Sun', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                      'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    DANGER_ZONE_MIN = 170
    DANGER_ZONE_MAX = 230
    DISPERSED_THRESHOLD = 280
    COMPRESSED_THRESHOLD = 160
    
    REGIME_MULTIPLIERS = {
        'DISPERSED': 1.0,
        'NEUTRAL': 1.0,
        'DANGER_ZONE': 0.80,
        'COMPRESSED': 0.90,
    }
    
    def __init__(self):
        self._cache = {}
        self.solar_eclipses, self.lunar_eclipses = self._cache_eclipses_robust()

    def _cache_eclipses_robust(self):
        """Pre-calculate eclipses 2000-2030 (New Moon + Lat < 1.6)"""
        if not EPHEM_AVAILABLE:
            return [], []

        solar = []
        lunar = []
        
        # Solar
        d = ephem.Date('2000-01-01')
        end_d = ephem.Date('2030-12-31')
        while d < end_d:
            try:
                nm = ephem.next_new_moon(d)
                if nm > end_d: break
                m = ephem.Moon(); m.compute(nm)
                lat = abs(ephem.Ecliptic(m).lat * 180/3.14159)
                if lat < 1.6:
                    solar.append(nm.datetime().date())
                d = ephem.Date(nm + 1)
            except:
                break
            
        # Lunar
        d = ephem.Date('2000-01-01')
        while d < end_d:
            try:
                fm = ephem.next_full_moon(d)
                if fm > end_d: break
                m = ephem.Moon(); m.compute(fm)
                lat = abs(ephem.Ecliptic(m).lat * 180/3.14159)
                if lat < 1.6:
                    lunar.append(fm.datetime().date())
                d = ephem.Date(fm + 1)
            except:
                break
            
        return solar, lunar
        
    def get_features(self, date_str: str) -> Dict:
        """Get all celestial features for a date."""
        if date_str in self._cache:
            return self._cache[date_str]
        
        if not EPHEM_AVAILABLE:
            return self._empty_features()
        
        try:
            features = self._calculate_features(date_str)
            self._cache[date_str] = features
            return features
        except Exception as e:
            return self._empty_features()
    
    def _calculate_features(self, date_str: str) -> Dict:
        # Parse date
        if isinstance(date_str, str):
            try:
                if '/' in date_str:
                    dt = datetime.strptime(date_str, '%Y/%m/%d')
                else:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
            except:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            dt = date_str
            
        obs = ephem.Observer()
        obs.date = dt + timedelta(hours=20)
        
        bodies = {
            'Sun': ephem.Sun(), 'Moon': ephem.Moon(), 'Mercury': ephem.Mercury(),
            'Venus': ephem.Venus(), 'Mars': ephem.Mars(), 'Jupiter': ephem.Jupiter(),
            'Saturn': ephem.Saturn(), 'Uranus': ephem.Uranus(), 'Neptune': ephem.Neptune(),
            'Pluto': ephem.Pluto(),
        }
        
        positions = {}
        for name, body in bodies.items():
            body.compute(obs)
            ecl = ephem.Ecliptic(body)
            positions[name] = math.degrees(ecl.lon)
        
        sun_saturn_sep = self._get_separation(positions['Sun'], positions['Saturn'])
        moon_uranus_sep = self._get_separation(positions['Moon'], positions['Uranus'])
        saturn_jupiter_sep = self._get_separation(positions['Saturn'], positions['Jupiter'])
        
        spread_positions = [positions[p] for p in self.SPREAD_PLANETS]
        spread = self._calculate_spread(spread_positions)
        spread_regime = self._classify_spread_regime(spread)
        
        moon_phase = bodies['Moon'].phase
        saturn_dignity = self._get_saturn_dignity(positions['Saturn'])
        mercury_retro = self._is_retrograde('Mercury', dt, positions['Mercury'])
        
        return {
            'sun_saturn_sep': sun_saturn_sep,
            'moon_uranus_sep': moon_uranus_sep,
            'sun_saturn_sep_normalized': sun_saturn_sep / 180.0,
            'moon_uranus_sep_normalized': moon_uranus_sep / 180.0,
            'saturn_dignity': saturn_dignity,
            'saturn_jupiter_sep_normalized': saturn_jupiter_sep / 180.0,
            'is_mercury_retrograde': mercury_retro,
            'eclipse_regime': self.get_eclipse_regime(dt),
            'moon_phase': moon_phase,
            'moon_phase_normalized': moon_phase / 100.0,
            'spread': spread,
            'spread_normalized': spread / 360.0,
            'spread_regime': spread_regime,
            'sun_opp_saturn': sun_saturn_sep > 175,
            'moon_opp_uranus': moon_uranus_sep > 175,
        }
    
    def _get_separation(self, lon1: float, lon2: float) -> float:
        diff = abs(lon1 - lon2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _calculate_spread(self, longitudes: list) -> float:
        if not longitudes:
            return 200.0
        sorted_lons = sorted(longitudes)
        n = len(sorted_lons)
        max_gap = 0
        for i in range(n):
            next_i = (i + 1) % n
            if next_i == 0:
                gap = (360 - sorted_lons[i]) + sorted_lons[next_i]
            else:
                gap = sorted_lons[next_i] - sorted_lons[i]
            max_gap = max(max_gap, gap)
        return 360 - max_gap
    
    def _classify_spread_regime(self, spread: float) -> str:
        if spread > self.DISPERSED_THRESHOLD: return 'DISPERSED'
        elif spread < self.COMPRESSED_THRESHOLD: return 'COMPRESSED'
        elif self.DANGER_ZONE_MIN <= spread <= self.DANGER_ZONE_MAX: return 'DANGER_ZONE'
        else: return 'NEUTRAL'
    
    def get_position_multiplier(self, regime: str) -> float:
        return self.REGIME_MULTIPLIERS.get(regime, 1.0)

    def _get_saturn_dignity(self, saturn_lon: float) -> int:
        sidereal_lon = (saturn_lon - 24) % 360
        sign = int(sidereal_lon / 30)
        if sign == 6: return 1
        elif sign == 0: return -1
        else: return 0

    def _is_retrograde(self, body_name: str, dt, current_lon: float) -> bool:
        prev_obs = ephem.Observer()
        prev_obs.date = dt + timedelta(hours=19)
        body_map = {'Mercury': ephem.Mercury(), 'Saturn': ephem.Saturn()}
        if body_name not in body_map: return False
        b = body_map[body_name]
        b.compute(prev_obs)
        prev_lon = math.degrees(ephem.Ecliptic(b).lon)
        diff = current_lon - prev_lon
        if diff < -350: return False
        if diff > 350: return True
        return current_lon < prev_lon

    def get_eclipse_regime(self, date_input):
        if isinstance(date_input, str):
            try: d_date = datetime.strptime(date_input, '%Y-%m-%d').date()
            except: d_date = datetime.strptime(date_input, '%Y/%m/%d').date()
        elif isinstance(date_input, datetime): d_date = date_input.date()
        else: d_date = date_input.date()
            
        for eclipse in self.solar_eclipses:
            if abs((d_date - eclipse).days) <= 3: return 1
        for eclipse in self.lunar_eclipses:
            if abs((d_date - eclipse).days) <= 3: return -1
        return 0
    
    def _empty_features(self) -> Dict:
        return {
            'sun_saturn_sep': 90.0, 'moon_uranus_sep': 90.0,
            'sun_saturn_sep_normalized': 0.5, 'moon_uranus_sep_normalized': 0.5,
            'moon_phase': 50.0, 'moon_phase_normalized': 0.5,
            'spread': 200.0, 'spread_normalized': 0.56, 'spread_regime': 'NEUTRAL',
            'sun_opp_saturn': False, 'moon_opp_uranus': False,
            'saturn_dignity': 0, 'saturn_jupiter_sep_normalized': 0.0,
            'is_mercury_retrograde': False, 'eclipse_regime': 0
        }
    
    def precompute_for_dates(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        print(f"  Pre-computing celestial features for {len(dates)} dates...")
        records = []
        for i, date in enumerate(dates):
            if i % 1000 == 0 and i > 0:
                print(f"    Processed {i}/{len(dates)} dates...")
            date_str = date.strftime('%Y-%m-%d')
            features = self.get_features(date_str)
            records.append({
                'date': date,
                'CELEST_sun_saturn_sep': features['sun_saturn_sep_normalized'],
                'CELEST_moon_uranus_sep': features['moon_uranus_sep_normalized'],
                'CELEST_saturn_dignity': features['saturn_dignity'],
                'CELEST_saturn_jupiter_sep': features['saturn_jupiter_sep_normalized'],
                'CELEST_mercury_retro': int(features['is_mercury_retrograde']),
                'CELEST_moon_phase': features['moon_phase_normalized'],
                'CELEST_spread': features['spread_normalized'],
                'CELEST_eclipse_regime': features['eclipse_regime'],
            })
        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        print(f"  Celestial features computed.")
        return df


# =============================================================================
# TECHNICAL ENGINE v8.0 - With REGIME Features
# =============================================================================

class TechnicalEngine:
    """
    v8.0: Adds Regime features to ML inputs.
    """
    
    @classmethod
    def calculate(cls, df: pd.DataFrame, celestial_df: Optional[pd.DataFrame] = None, regime_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all technical + celestial + regime features.
        """
        df = df.copy()
        close = safe_series(df['Close'])
        high = safe_series(df['High'])
        low = safe_series(df['Low'])
        volume = safe_series(df['Volume'])
        
        features = pd.DataFrame(index=df.index)
        
        # 1. RSI
        for length in [2, 5, 14, 21, 50]:
            rsi = ta.rsi(close, length=length)
            features[f'RSI_{length}'] = rsi
            features[f'RSI_{length}_oversold'] = (rsi < 30).astype(int)
            features[f'RSI_{length}_extreme'] = (rsi < 20).astype(int)
            features[f'RSI_{length}_overbought'] = (rsi > 70).astype(int)
            features[f'RSI_{length}_extreme_high'] = (rsi > 80).astype(int)
        
        # 2. Bollinger Bands
        for length in [20, 50]:
            bb = ta.bbands(close, length=length, std=2.0)
            if bb is not None:
                col_pctb = f'BBP_{length}_2.0'
                if col_pctb not in bb.columns:
                    # Fallback if names differ
                    try:
                        lower = bb.iloc[:, 0]
                        mid = bb.iloc[:, 1]
                        upper = bb.iloc[:, 2] 
                        pctb = (close - lower) / (upper - lower)
                    except:
                        pctb = pd.Series(0.5, index=df.index)
                else:
                    pctb = bb[col_pctb]
                
                features[f'BB_{length}_pctb'] = pctb
                features[f'BB_{length}_oversold'] = (pctb < 0).astype(int)
                features[f'BB_{length}_sharktooth'] = (pctb < -0.1).astype(int)
                features[f'BB_{length}_overbought'] = (pctb > 1).astype(int)
                features[f'BB_{length}_sharktooth_bear'] = (pctb > 1.1).astype(int)
                features[f'BB_{length}_crossunder'] = ((pctb < -0.06) & (pctb.shift(1) >= -0.06)).astype(int)
        
        # 3. MACD
        macd = ta.macd(close)
        if macd is not None:
            features['MACD_line'] = macd.iloc[:, 0]
            features['MACD_hist'] = macd.iloc[:, 1]
            features['MACD_signal'] = macd.iloc[:, 2]
            features['MACD_oversold'] = (features['MACD_line'] < -2.0).astype(int)
        
        # 4. Stochastic
        stoch = ta.stoch(high, low, close)
        if stoch is not None:
            k = stoch.iloc[:, 0]
            d = stoch.iloc[:, 1]
            features['STOCH_k'] = k
            features['STOCH_oversold'] = (k < 20).astype(int)
            features['STOCH_sharktooth'] = ((k < 20) & (k > d)).astype(int)
        
        # 8. Price Action
        features['RET_1d'] = close.pct_change(1) * 100
        features['RET_5d'] = close.pct_change(5) * 100
        for period in [10, 20, 50]:
            rolling_max = close.rolling(period).max()
            features[f'DD_{period}d'] = (close - rolling_max) / rolling_max * 100
            rolling_min = close.rolling(period).min()
            features[f'RALLY_{period}d'] = (close - rolling_min) / rolling_min * 100
            
        # 9. Pine Script Features
        daily_return = close.pct_change(1)
        features['DAILY_RETURN_PANIC'] = (daily_return < -0.0088).astype(int)
        features['PINE_ENTRY_SIGNAL'] = (features['BB_20_crossunder'] & features['DAILY_RETURN_PANIC']).astype(int)
        
        # 11. Celestial Features
        if celestial_df is not None:
            for col in celestial_df.columns:
                if col in celestial_df.columns:
                    features[col] = celestial_df[col]
                    
        # 12. NEW v8.0: Regime Features
        if regime_df is not None:
            for col in regime_df.columns:
                if col in regime_df.columns:
                    features[col] = regime_df[col]
                    
        return features


# =============================================================================
# MODELS & STRATEGY v8.0
# =============================================================================

class SharktoothDetector:
    def __init__(self, mode='bull', lookahead_days=10, min_move_pct=3.0, lookback_days=5):
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
            if len(lookback_prices) == 0 or len(future_prices) == 0: continue
            
            if self.mode == 'bull':
                if current_price > lookback_prices.min(): continue
                max_future = future_prices.max()
                move_pct = (max_future - current_price) / current_price * 100
                if move_pct >= self.min_move_pct: labels.iloc[i] = 1
            elif self.mode == 'bear':
                if current_price < lookback_prices.max(): continue
                min_future = future_prices.min()
                move_pct = (current_price - min_future) / current_price * 100
                if move_pct >= self.min_move_pct: labels.iloc[i] = 1
        return labels
    
    def prepare_data(self, df: pd.DataFrame, celestial_df: pd.DataFrame, regime_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        features = TechnicalEngine.calculate(df, celestial_df, regime_df)
        close = safe_series(df['Close'])
        labels = self._label_turning_points(close)
        valid_idx = features.dropna().index.intersection(labels.index)
        if len(valid_idx) > self.lookahead_days + self.lookback_days:
            valid_idx = valid_idx[self.lookback_days:-self.lookahead_days]
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]
        self.feature_names = list(X.columns)
        return X, y
    
    def train(self, df: pd.DataFrame, celestial_df: pd.DataFrame, regime_df: pd.DataFrame, verbose: bool = True):
        X, y = self.prepare_data(df, celestial_df, regime_df)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        self.model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pos_weight, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
            if verbose:
                print(f"  Top 10 Features ({self.mode}):")
                print(self.feature_importance.head(10))
        return self

    def predict(self, df: pd.DataFrame, celestial_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.Series:
        features = TechnicalEngine.calculate(df, celestial_df, regime_df)
        for feat in self.feature_names:
            if feat not in features.columns: features[feat] = 0
        X = features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        return pd.Series(self.model.predict_proba(X_scaled)[:, 1], index=df.index, name=f'{self.mode}_prob')
    
    def save(self, path):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names, 'mode': self.mode}, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        inst = cls(mode=data.get('mode', 'bull'))
        inst.model = data['model']
        inst.scaler = data['scaler']
        inst.feature_names = data['feature_names']
        return inst


class BulletproofStrategyV80_Regime:
    
    feature_cols = [
        'RSI_2', 'RSI_5', 'RSI_14', 'BB_20_pctb', 'BB_20_oversold', 'BB_50_pctb',
        'MACD_hist', 'STOCH_k', 'PINE_ENTRY_SIGNAL', 'DD_20d',
        'CELEST_sun_saturn_sep', 'CELEST_moon_uranus_sep', 'CELEST_saturn_dignity',
        'CELEST_eclipse_regime',
        'REGIME_is_turn_of_month', 'REGIME_seasonality', 'REGIME_vix_term'
    ]
    
    def __init__(self, ticker='SPY'):
        self.ticker = ticker
        self.df = None
        self.celestial_features = None
        self.regime_features = None
        self.celestial_engine = CelestialEngine()
        self.regime_engine = MarketRegimeEngine()
        self.bull_model = SharktoothDetector(mode='bull')
        self.bear_model = SharktoothDetector(mode='bear')
        
    def load_data(self, start_date='2000-01-01'):
        if not YF_AVAILABLE: raise ImportError("yfinance required")
        print("Loading Market Data (SPY, VIX, VIX3M)...")
        
        # Download core ticker + VIX + VIX3M
        tickers = [self.ticker, '^VIX', '^VIX3M']
        data = yf.download(tickers, start=start_date, progress=False)
        
        # Intelligent handling of MultiIndex columns
        close_df = None
        
        if isinstance(data.columns, pd.MultiIndex):
            # Check if we have 'Close' level
            if 'Close' in data.columns.levels[0] or 'Close' in data.columns.levels[1]:
                # Try accessing Cross-Section for Close
                try: 
                    close_df = data.xs('Close', level='Price', axis=1) 
                except:
                    # Fallback to standard level access
                    try: close_df = data['Close']
                    except: pass
        else:
            # Simple index
            close_df = data['Close'] if 'Close' in data else data

        if close_df is None:
            # Manual construction from raw
            close_df = pd.DataFrame()
            for t in tickers:
                if t in data.columns: close_df[t] = data[t]
                
        # Now we need OHLCV for the main ticker
        if isinstance(data.columns, pd.MultiIndex):
             # Extract main ticker OHLCV
             self.df = pd.DataFrame({
                 'Open': data.xs('Open', level='Price', axis=1)[self.ticker],
                 'High': data.xs('High', level='Price', axis=1)[self.ticker],
                 'Low': data.xs('Low', level='Price', axis=1)[self.ticker],
                 'Close': data.xs('Close', level='Price', axis=1)[self.ticker],
                 'Volume': data.xs('Volume', level='Price', axis=1)[self.ticker]
             })
        else:
             # Assume single ticker structure if not multi? Unlikely with 3 tickers.
             # Safe fallback: download separately
             pass
             
        # Separate Download for safety if MultiIndex fails or is messy
        spy = yf.download(self.ticker, start=start_date, progress=False)
        vix = yf.download(['^VIX', '^VIX3M'], start=start_date, progress=False)['Close']
        
        self.df = spy
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        
        # Prepare VIX Data for Regime Engine
        # Align dates
        common_idx = self.df.index.intersection(vix.index)
        self.vix_data = vix.reindex(self.df.index) # Forward fill?
        
        print(f"Loaded {len(self.df)} trading days")
        
        # Precompute
        self._precompute_celestial()
        self._precompute_regime()
        
    def _precompute_celestial(self):
        self.celestial_features = self.celestial_engine.precompute_for_dates(self.df.index)
        
    def _precompute_regime(self):
        print("  Pre-computing Market Regimes (Seasonality + VIX)...")
        records = []
        
        # Ensure VIX columns exist
        vix_spot = self.vix_data['^VIX'] if '^VIX' in self.vix_data else pd.Series(0, index=self.df.index)
        vix_3m = self.vix_data['^VIX3M'] if '^VIX3M' in self.vix_data else pd.Series(0, index=self.df.index)
        
        for i, date in enumerate(self.df.index):
            f = self.regime_engine.get_features(
                date, 
                vix_spot.loc[date] if date in vix_spot.index else None,
                vix_3m.loc[date] if date in vix_3m.index else None
            )
            records.append({
                'date': date,
                'REGIME_is_turn_of_month': f['is_turn_of_month'],
                'REGIME_seasonality': f['seasonality_regime'],
                'REGIME_vix_term': f['vix_term_structure']
            })
            
        self.regime_features = pd.DataFrame(records).set_index('date')
        print("  Regime features computed.")

    def train_models(self):
        print("\nTRAINING v8.0 MODELS...")
        self.bull_model.train(self.df, self.celestial_features, self.regime_features)
        self.bear_model.train(self.df, self.celestial_features, self.regime_features)
        
    def backtest(self):
        print("\n======================================================================")
        print(f"BACKTEST: {self.ticker} (v8.0 Seasonality/Regime)")
        print("======================================================================")
        
        bull_probs = self.bull_model.predict(self.df, self.celestial_features, self.regime_features)
        bear_probs = self.bear_model.predict(self.df, self.celestial_features, self.regime_features)
        
        balance = 100000.0
        shares = 0
        equity = []
        trades = []
        
        # Risk Management Params
        max_risk_per_trade = 0.02
        stop_loss_pct = 0.05
        
        entry_price = 0
        in_trade = False
        
        # Regime Lookups
        spread_regimes = self.celestial_features['CELEST_spread'].apply(
            lambda x: self.celestial_engine._classify_spread_regime(x*360)
        )
        
        for i in range(20, len(self.df)):
            date = self.df.index[i]
            # Ensure scalar price
            price_val = self.df['Close'].iloc[i]
            if isinstance(price_val, pd.Series):
                price = price_val.item()
            else:
                price = float(price_val)
            
            # Regime Filters
            regime = spread_regimes.iloc[i]
            regime_mult = self.celestial_engine.get_position_multiplier(regime)
            
            # Additional v8.0 Logic Integration?
            # Actually, we let the ML handle the weights of seasonality.
            # But we can override strictly if ML confidence is high.
            
            prob_bull = bull_probs.iloc[i]
            prob_bear = bear_probs.iloc[i]
            
            # ENTRY
            if not in_trade:
                if prob_bull > 0.65: # High confidence
                    # Size
                    risk_amt = balance * max_risk_per_trade * regime_mult
                    # Stop loss distance
                    sl_dist = price * stop_loss_pct
                    pos_shares = int(risk_amt / sl_dist)
                    
                    cost = pos_shares * price
                    if cost <= balance:
                        balance -= cost
                        shares = pos_shares
                        entry_price = price
                        in_trade = True
                        trades.append({
                            'Entry Date': date, 'Entry Price': price, 'Type': 'Long',
                            'Regime': regime, 'Prob': prob_bull
                        })
            
            # EXIT
            elif in_trade:
                # Bear Classification or Stop/Target
                exit_signal = False
                reason = ""
                
                # 1. Bear AI Signal
                if prob_bear > 0.70:
                    exit_signal = True
                    reason = "Bear AI"
                    
                # 2. Hard Stop
                if price < entry_price * (1 - stop_loss_pct):
                    exit_signal = True
                    reason = "Stop Loss"
                    
                # 3. Trailing/Profit logic could be here...
                
                if exit_signal:
                    proceeds = shares * price
                    pnl = proceeds - (shares * entry_price)
                    balance += proceeds
                    trades[-1]['Exit Date'] = date
                    trades[-1]['Exit Price'] = price
                    trades[-1]['PnL'] = pnl
                    trades[-1]['Return'] = (price - entry_price)/entry_price
                    trades[-1]['Reason'] = reason
                    shares = 0
                    in_trade = False
                    
            # Track Equity
            current_val = balance + (shares * price)
            equity.append({'Date': date, 'Equity': current_val})
            
        equity_df = pd.DataFrame(equity).set_index('Date')
        trades_df = pd.DataFrame(trades)
        
        if not trades_df.empty:
            total_ret = (equity_df['Equity'].iloc[-1] - 100000) / 100000 * 100
            win_rate = (trades_df['PnL'] > 0).mean() * 100
            print(f"Total Return: {total_ret:.2f}%")
            print(f"Final Equity: ${equity_df['Equity'].iloc[-1]:,.2f}")
            print(f"Win Rate:     {win_rate:.1f}%")
            print(f"Total Trades: {len(trades_df)}")
            trades_df.to_csv('v8_0_trades.csv')
        else:
            print("No trades generated.")
            
        return equity_df, trades_df

if __name__ == "__main__":
    strategy = BulletproofStrategyV80_Regime(ticker='QQQ') # Validating on QQQ as requested
    strategy.load_data()
    strategy.train_models()
    strategy.backtest()
