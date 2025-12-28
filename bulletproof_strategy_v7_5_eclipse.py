"""
================================================================================
BULLETPROOF STRATEGY v7.5 - ECLIPSE ML INTEGRATED
================================================================================

Based on v7.4 Vedic + v7.5 Eclipse Research:
- INTEGRATES Solar/Lunar Eclipse Regimes (+5.5% Edge)
- KEEPS all v7.4 features (Saturn Dignity, Stellium)

Changes from v7.4:
- ADDS eclipse_regime (+1/0/-1) as ML feature
- Uses robust manual eclipse detection (New Moon + Lat < 1.6)

Key validation from Research:
- Solar Eclipses (+/- 3d) show 60% Bullish Win Rate
- Lunar Eclipses (+/- 3d) are slightly Bearish/Neutral
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
# CELESTIAL ENGINE v7.3 - Integrated
# =============================================================================

class CelestialEngine:
    """
    v7.4: Complete celestial calculation engine.
    
    Features for ML:
    - sun_saturn_sep_normalized (0-1): NEAT weight +1.84
    - moon_uranus_sep_normalized (0-1): NEAT weight -0.83
    - spread_normalized (0-1): Planetary dispersion
    - saturn_dignity (-1/0/+1): Vedic regime filter
    - saturn_jupiter_sep_normalized (0-1): Great Conjunction
    
    Risk Management:
    - Spread regime classification (Danger Zone detection)
    - Position sizing multipliers
    """
    
    # Planets for spread calculation (excluding Moon - too fast)
    SPREAD_PLANETS = ['Sun', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                      'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    # Spread regime thresholds (validated on 1993-2025 data)
    DANGER_ZONE_MIN = 170
    DANGER_ZONE_MAX = 230
    DISPERSED_THRESHOLD = 280
    COMPRESSED_THRESHOLD = 160
    
    # Position sizing multipliers
    REGIME_MULTIPLIERS = {
        'DISPERSED': 1.0,      # Full size - planets spread, low aspect tension
        'NEUTRAL': 1.0,        # Normal operations
        'DANGER_ZONE': 0.80,   # 20% reduction - peak volatility regime
        'COMPRESSED': 0.90,    # 10% reduction - calm before storm warning
    }
    
    def __init__(self):
        self._cache = {}
        # Pre-calculate eclipses
        self.solar_eclipses, self.lunar_eclipses = self._cache_eclipses_robust()

    def _cache_eclipses_robust(self):
        """Pre-calculate eclipses 2000-2030 (New Moon + Lat < 1.6)"""
        # print("    DEBUG: Starting eclipse pre-calc...")
        if not EPHEM_AVAILABLE:
            # print("    DEBUG: Ephem NOT available!")
            return [], []

        solar = []
        lunar = []
        
        # Solar
        d = ephem.Date('2000-01-01')
        end_d = ephem.Date('2030-12-31')
        count_checks = 0
        while d < end_d:
            try:
                nm = ephem.next_new_moon(d)
                if nm > end_d: break
                m = ephem.Moon(); m.compute(nm)
                lat = abs(ephem.Ecliptic(m).lat * 180/3.14159)
                if lat < 1.6:
                    solar.append(nm.datetime().date())
                d = ephem.Date(nm + 1)
                count_checks += 1
            except Exception as e:
                # print(f"    DEBUG error solar: {e}")
                break
        # print(f"    DEBUG: Found {len(solar)} solar eclipses")
            
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
            except Exception as e:
                # print(f"    DEBUG error lunar: {e}")
                break
        # print(f"    DEBUG: Found {len(lunar)} lunar eclipses")
            
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
        """Core calculation logic."""
        # Parse date
        if isinstance(date_str, str):
            try:
                if '/' in date_str:
                    dt = datetime.strptime(date_str, '%Y/%m/%d')
                else:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
            except:
                dt = datetime.strptime(date_str, '%Y-%m-%d') # Fallback
        else:
            dt = date_str
            
        # Set observer time (20:00 UTC - market close)
        obs = ephem.Observer()
        obs.date = dt + timedelta(hours=20)
        
        # Calculate planetary positions
        bodies = {
            'Sun': ephem.Sun(),
            'Moon': ephem.Moon(),
            'Mercury': ephem.Mercury(),
            'Venus': ephem.Venus(),
            'Mars': ephem.Mars(),
            'Jupiter': ephem.Jupiter(),
            'Saturn': ephem.Saturn(),
            'Uranus': ephem.Uranus(),
            'Neptune': ephem.Neptune(),
            'Pluto': ephem.Pluto(),
        }
        
        positions = {}
        for name, body in bodies.items():
            body.compute(obs)
            ecl = ephem.Ecliptic(body)
            positions[name] = math.degrees(ecl.lon)
        
        # Key separations (NEAT-validated)
        sun_saturn_sep = self._get_separation(positions['Sun'], positions['Saturn'])
        moon_uranus_sep = self._get_separation(positions['Moon'], positions['Uranus'])
        saturn_jupiter_sep = self._get_separation(positions['Saturn'], positions['Jupiter'])
        
        # Planetary spread (excluding Moon)
        spread_positions = [positions[p] for p in self.SPREAD_PLANETS]
        spread = self._calculate_spread(spread_positions)
        spread_regime = self._classify_spread_regime(spread)
        
        # Moon phase
        moon_phase = bodies['Moon'].phase
        
        # Vedic / Retrograde calculations
        saturn_dignity = self._get_saturn_dignity(positions['Saturn'])
        mercury_retro = self._is_retrograde('Mercury', dt, positions['Mercury'])
        
        return {
            # Raw separations (degrees)
            'sun_saturn_sep': sun_saturn_sep,
            'moon_uranus_sep': moon_uranus_sep,
            
            # Normalized for ML (0-1) - KEY FEATURES FOR v7.3
            'sun_saturn_sep_normalized': sun_saturn_sep / 180.0,
            'moon_uranus_sep_normalized': moon_uranus_sep / 180.0,
            
            # v7.4 Vedic Features
            'saturn_dignity': saturn_dignity,
            'saturn_jupiter_sep_normalized': saturn_jupiter_sep / 180.0,
            'is_mercury_retrograde': mercury_retro,
            'eclipse_regime': self.get_eclipse_regime(dt),
            
            # Moon phase
            'moon_phase': moon_phase,
            'moon_phase_normalized': moon_phase / 100.0,
            
            # Spread regime (for position sizing)
            'spread': spread,
            'spread_normalized': spread / 360.0,
            'spread_regime': spread_regime,
            
            # Binary flags (kept for backwards compatibility / exit override)
            'sun_opp_saturn': sun_saturn_sep > 175,
            'moon_opp_uranus': moon_uranus_sep > 175,
        }
    
    def _get_separation(self, lon1: float, lon2: float) -> float:
        """Calculate angular separation between two longitudes."""
        diff = abs(lon1 - lon2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _calculate_spread(self, longitudes: list) -> float:
        """Calculate minimum arc containing all planets."""
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
        """Classify spread into risk regime."""
        if spread > self.DISPERSED_THRESHOLD:
            return 'DISPERSED'
        elif spread < self.COMPRESSED_THRESHOLD:
            return 'COMPRESSED'
        elif self.DANGER_ZONE_MIN <= spread <= self.DANGER_ZONE_MAX:
            return 'DANGER_ZONE'
        else:
            return 'NEUTRAL'
    
    def get_position_multiplier(self, regime: str) -> float:
        """Get position sizing multiplier for regime."""
        return self.REGIME_MULTIPLIERS.get(regime, 1.0)

    
    def _get_saturn_dignity(self, saturn_lon: float) -> int:
        """
        Calculate Sidereal Saturn Dignity.
        Approximate Sidereal = Tropical - 24 degrees (Lahiri Ayanamsa)
        """
        sidereal_lon = (saturn_lon - 24) % 360
        sign = int(sidereal_lon / 30)
        
        if sign == 6: # Libra (Exalted)
            return 1
        elif sign == 0: # Aries (Debilitated)
            return -1
        else:
            return 0

    def _is_retrograde(self, body_name: str, dt, current_lon: float) -> bool:
        """Check if planet is retrograde (longitude decreasing)."""
        # Compute position 1 hour ago
        prev_obs = ephem.Observer()
        prev_obs.date = dt + timedelta(hours=19) # 20:00 - 1h
        
        body_map = {
            'Mercury': ephem.Mercury(),
            'Saturn': ephem.Saturn()
        }
        
        if body_name not in body_map:
            return False
            
        b = body_map[body_name]
        b.compute(prev_obs)
        prev_lon = math.degrees(ephem.Ecliptic(b).lon)
        
        # Check wrap around 360/0
        diff = current_lon - prev_lon
        if diff < -350: # Wrapped 0 -> 360 (forward)
            return False
        if diff > 350: # Wrapped 360 -> 0 (retrograde)
            return True
            
        return current_lon < prev_lon

    def get_eclipse_regime(self, date_input):
        """
        Returns Eclipse Regime:
         1: Within 3 days of Solar Eclipse (Bullish)
        -1: Within 3 days of Lunar Eclipse (Bearish)
         0: None
        """
        if isinstance(date_input, str):
            try:
                d_date = datetime.strptime(date_input, '%Y-%m-%d').date()
            except:
                d_date = datetime.strptime(date_input, '%Y/%m/%d').date()
        elif isinstance(date_input, datetime):
            d_date = date_input.date()
        else:
            d_date = date_input.date() # pd.Timestamp
            
        # Check Solar
        for eclipse in self.solar_eclipses:
            if abs((d_date - eclipse).days) <= 3:
                return 1
                
        # Check Lunar
        for eclipse in self.lunar_eclipses:
            if abs((d_date - eclipse).days) <= 3:
                return -1
                
        return 0
    
    def _empty_features(self) -> Dict:
        """Return default features when ephem unavailable."""
        return {
            'sun_saturn_sep': 90.0,
            'moon_uranus_sep': 90.0,
            'sun_saturn_sep_normalized': 0.5,
            'moon_uranus_sep_normalized': 0.5,
            'moon_phase': 50.0,
            'moon_phase_normalized': 0.5,
            'spread': 200.0,
            'spread_normalized': 0.56,
            'spread_regime': 'NEUTRAL',
            'sun_opp_saturn': False,
            'moon_opp_uranus': False,
            'saturn_dignity': 0,
            'saturn_jupiter_sep_normalized': 0.0,
            'is_mercury_retrograde': False,
            'eclipse_regime': 0
        }
    
    def precompute_for_dates(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Batch compute celestial features for entire dataset.
        Call this BEFORE training to add features to signal_data.
        """
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
                'CELEST_saturn_dignity': features['saturn_dignity'], # v7.4
                'CELEST_saturn_jupiter_sep': features['saturn_jupiter_sep_normalized'], # v7.4
                'CELEST_mercury_retro': int(features['is_mercury_retrograde']), # v7.3
                'CELEST_moon_phase': features['moon_phase_normalized'],
                'CELEST_spread': features['spread_normalized'],
                'CELEST_eclipse_regime': features['eclipse_regime'], # v7.5
            })
        
        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        print(f"  Celestial features computed.")
        return df


# =============================================================================
# TECHNICAL ENGINE v7.3 - WITH CELESTIAL FEATURES
# =============================================================================

class TechnicalEngine:
    """
    v7.4: Adds Vedic features to ML inputs.
    
    New ML features:
    - CELEST_sun_saturn_sep: Normalized 0-1 (NEAT weight: +1.84)
    - CELEST_moon_uranus_sep: Normalized 0-1 (NEAT weight: -0.83)
    - CELEST_saturn_dignity: -1/0/+1 (Vedic)
    - CELEST_saturn_jupiter_sep: 0-1 (Great Conjunction)
    - CELEST_mercury_retro: 0/1 (Risk Factor)
    - CELEST_moon_phase: Normalized 0-1
    - CELEST_spread: Planetary spread normalized 0-1
    - CELEST_eclipse_regime: -1/0/+1 (Eclipse Window)
    
    Removed from logic (now learned by ML):
    - Hardcoded moon_opp_uranus boost
    - Hardcoded is_lunar_window boost
    """
    
    @classmethod
    def calculate(cls, df: pd.DataFrame, celestial_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all technical + celestial features.
        
        Args:
            df: OHLCV DataFrame
            celestial_df: Pre-computed celestial features (from CelestialEngine.precompute_for_dates)
        """
        df = df.copy()
        
        close = safe_series(df['Close'])
        high = safe_series(df['High'])
        low = safe_series(df['Low'])
        volume = safe_series(df['Volume'])
        
        features = pd.DataFrame(index=df.index)
        
        # =================================================================
        # 1. RSI FAMILY (unchanged)
        # =================================================================
        for length in [2, 5, 14, 21, 50]:
            rsi = ta.rsi(close, length=length)
            features[f'RSI_{length}'] = rsi
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
                
                # Pine Script crossover features (v7.2)
                features[f'BB_{length}_crossunder'] = (
                    (pctb < -0.06) & (pctb.shift(1) >= -0.06)
                ).astype(int)
                features[f'BB_{length}_crossover'] = (
                    (pctb > 1.0) & (pctb.shift(1) <= 1.0)
                ).astype(int)
        
        # =================================================================
        # 3. MACD
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
        # 4. STOCHASTIC
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
        cci = ta.cci(high, low, close, length=20)
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
        # 9. DAILY RETURN FEATURES (Pine Script)
        # =================================================================
        daily_return = close.pct_change(1)
        features['DAILY_RETURN_PANIC'] = (daily_return < -0.0088).astype(int)
        features['DAILY_RETURN_CRASH'] = (daily_return < -0.02).astype(int)
        features['DAILY_RETURN_EXTREME'] = (daily_return < -0.03).astype(int)
        features['DAILY_RETURN_SURGE'] = (daily_return > 0.02).astype(int)
        
        # Combined Pine Script signal
        features['PINE_ENTRY_SIGNAL'] = (
            features['BB_20_crossunder'] & features['DAILY_RETURN_PANIC']
        ).astype(int)
        
        # =================================================================
        # 10. COMPOSITE SCORES
        # =================================================================
        oversold_cols = [c for c in features.columns if 'oversold' in c.lower()]
        sharktooth_bull_cols = [c for c in features.columns if 'sharktooth' in c.lower() and 'bear' not in c.lower()]
        
        features['OVERSOLD_COUNT'] = features[oversold_cols].sum(axis=1)
        features['BULL_SHARKTOOTH_COUNT'] = features[sharktooth_bull_cols].sum(axis=1)
        
        overbought_cols = [c for c in features.columns if 'overbought' in c.lower()]
        sharktooth_bear_cols = [c for c in features.columns if 'sharktooth_bear' in c.lower()]
        
        features['OVERBOUGHT_COUNT'] = features[overbought_cols].sum(axis=1)
        features['BEAR_SHARKTOOTH_COUNT'] = features[sharktooth_bear_cols].sum(axis=1)
        
        # =================================================================
        # 11. NEW v7.3: CELESTIAL FEATURES (from pre-computed DataFrame)
        # =================================================================
        if celestial_df is not None:
            # Join celestial features
            for col in celestial_df.columns:
                if col in celestial_df.columns:
                    features[col] = celestial_df[col]
        
        return features


# =============================================================================
# SHARKTOOTH DETECTOR v7.3
# =============================================================================

class SharktoothDetector:
    """
    v7.3: Now trains on celestial features as continuous inputs.
    
    Expected behavior:
    - CELEST_sun_saturn_sep should get positive importance (hold signal)
    - CELEST_moon_uranus_sep should get importance (entry near 1.0 = opposition)
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
    
    def prepare_data(self, df: pd.DataFrame, celestial_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features including celestial data."""
        features = TechnicalEngine.calculate(df, celestial_df)
        close = safe_series(df['Close'])
        labels = self._label_turning_points(close)
        
        valid_idx = features.dropna().index.intersection(labels.index)
        
        if len(valid_idx) > self.lookahead_days + self.lookback_days:
            valid_idx = valid_idx[self.lookback_days:-self.lookahead_days]
        
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train(self, df: pd.DataFrame, celestial_df: pd.DataFrame, verbose: bool = True) -> 'SharktoothDetector':
        if verbose:
            print(f"Training {self.mode.upper()} Detector (v7.3 - Astro-ML)...")
        
        X, y = self.prepare_data(df, celestial_df)
        
        if verbose:
            print(f"  Total features: {len(self.feature_names)}")
            celest_features = [f for f in self.feature_names if 'CELEST' in f]
            print(f"  Celestial features: {celest_features}")
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
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
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            
            if verbose:
                print(f"  Trained on {len(X)} samples, {y.sum():.0f} positive labels")
                print(f"  Top 15 features:")
                for i, (feat, imp) in enumerate(self.feature_importance.head(15).items()):
                    marker = ""
                    if 'CELEST' in feat:
                        marker = " ← CELESTIAL"
                    elif 'PINE' in feat or 'PANIC' in feat:
                        marker = " ← PINE"
                    print(f"    {i+1:2d}. {feat}: {imp:.4f}{marker}")
        
        return self
    
    def predict(self, df: pd.DataFrame, celestial_df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained!")
        
        features = TechnicalEngine.calculate(df, celestial_df)
        
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
# BULLETPROOF STRATEGY v7.5 - ECLIPSE
# =============================================================================

class BulletproofStrategyV75_Eclipse:
    """
    v7.5 Eclipse Integrated:
    
    Key Changes from v7.4:
    1. Integrates Solar/Lunar Eclipse Regimes (+5.5% Edge)
    2. Keeps all v7.4 Vedic logic
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
        self.celestial_df = None  # Pre-computed celestial features
        
        self.stats = {}
        self.trade_log = []
        
    def _precompute_celestial(self):
        """Pre-compute celestial features for all dates."""
        if self.celestial_df is None:
            self.celestial_df = self.celestial.precompute_for_dates(self.signal_data.index)
        return self.celestial_df
        
    def train_models(self, verbose: bool = True):
        """Train both detectors with celestial features."""
        # Pre-compute celestial data first
        celestial_df = self._precompute_celestial()
        
        self.bull_detector = SharktoothDetector(mode='bull')
        self.bull_detector.train(self.signal_data, celestial_df, verbose=verbose)
        
        self.bear_detector = SharktoothDetector(mode='bear')
        self.bear_detector.train(self.signal_data, celestial_df, verbose=verbose)
        
        return self
    
    def should_override_bear_exit(
        self, 
        entry_vix: float,
        current_sun_saturn_sep: float,
        above_ma50: bool
    ) -> bool:
        """
        Validated exit override rule (+1.5% alpha over 20 days).
        Returns True if we should HOLD past BEAR_TWIN signal.
        """
        panic_entry = entry_vix > 25
        sun_saturn_favorable = 90 < current_sun_saturn_sep < 150
        in_uptrend = above_ma50
        
        return panic_entry and sun_saturn_favorable and in_uptrend

    def backtest(self, params: Optional[Dict] = None, initial_capital: float = 100000, verbose: bool = True):
        """
        v7.5 backtest with Eclipse + Vedic features.
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
        
        # Ensure celestial data and models are ready
        celestial_df = self._precompute_celestial()
        
        if self.bull_detector is None:
            self.train_models(verbose=verbose)
        
        # Generate predictions
        bull_probs = self.bull_detector.predict(self.signal_data, celestial_df)
        bear_probs = self.bear_detector.predict(self.signal_data, celestial_df)
        features = TechnicalEngine.calculate(self.signal_data, celestial_df)
        close = safe_series(self.trade_data['Close'])
        
        # Pre-calculate MA50 for exit override
        ma50 = close.rolling(50).mean()
        
        # Backtest state
        cash = initial_capital
        shares = 0.0
        entry_price = 0.0
        max_price = 0.0
        entry_date = None
        entry_vix = 20.0
        partial_exit_taken = False
        
        self.trade_log = []
        self.all_signals = []
        equity_curve = []
        peak_equity = initial_capital
        max_drawdown = 0.0
        cooldown = 0
        
        bear_prob_history = []
        
        # Stats tracking
        total_entries = 0
        danger_zone_reductions = 0
        exit_overrides = 0
        
        for i, date in enumerate(self.signal_data.index):
            if date not in close.index:
                continue
            
            price = float(close.loc[date])
            
            # VIX
            vix = 20.0
            if self.vix_data is not None and date in self.vix_data.index:
                v = self.vix_data.loc[date, 'Close']
                vix = float(safe_series(pd.Series([v])).iloc[0]) if not isinstance(v, (int, float)) else float(v)
            
            # Celestial features for this date
            date_str = date.strftime('%Y-%m-%d')
            cel = self.celestial.get_features(date_str)
            
            # ML signals (now include celestial knowledge)
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
            
            # ------------------------------------------------------------------
            # ENTRY SIGNAL (ML now handles celestial weighting)
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
                
                # =============================================================
                # v7.3: SPREAD REGIME POSITION SIZING
                # =============================================================
                if signal_size > 0:
                    spread_regime = cel.get('spread_regime', 'NEUTRAL')
                    regime_mult = self.celestial.get_position_multiplier(spread_regime)
                    
                    if regime_mult < 1.0:
                        signal_size *= regime_mult
                        danger_zone_reductions += 1
                        if verbose and date.year >= 2020:
                            print(f"  > [{date.strftime('%Y-%m-%d')}] DANGER ZONE: "
                                  f"Size reduced to {signal_size:.0%} (Spread: {cel.get('spread', 0):.0f}°)")
                    
                    # Log signal
                    self.all_signals.append({
                        'Date': date,
                        'Price': price,
                        'Size': signal_size,
                        'Conviction': signal_conviction,
                        'Bull_Prob': bull_prob,
                        'Bull_Count': bull_count,
                        'Spread_Regime': spread_regime,
                        'VIX': vix
                    })
            
            # ------------------------------------------------------------------
            # EXECUTION
            # ------------------------------------------------------------------
            if shares == 0 and cooldown == 0 and signal_size > 0:
                invest = cash * signal_size
                shares = invest / price
                cash -= invest
                entry_price = price
                max_price = price
                entry_date = date
                entry_vix = vix
                entry_size = signal_size # Store for logging!
                partial_exit_taken = False
                total_entries += 1
                
                if verbose and date.year >= 2020:
                    print(f"[{date.strftime('%Y-%m-%d')}] BUY @ ${price:.2f} "
                          f"(Prob: {bull_prob:.0%}, Count: {bull_count:.0f}, "
                          f"{signal_conviction}, VIX: {vix:.1f})")
            
            # ------------------------------------------------------------------
            # EXIT LOGIC
            # ------------------------------------------------------------------
            elif shares > 0:
                max_price = max(max_price, price)
                unrealized = (price - entry_price) / entry_price
                dd_from_high = (max_price - price) / max_price
                
                # Profit taking
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
                override_applied = False
                
                # Check exit override conditions
                if bear_prob > params['bear_threshold']:
                    above_ma = price > ma50.loc[date] if not pd.isna(ma50.loc[date]) else False
                    
                    if self.should_override_bear_exit(entry_vix, cel['sun_saturn_sep'], above_ma):
                        override_applied = True
                        exit_overrides += 1
                    else:
                        exit_signal = True
                        reason = f"BEAR_TWIN ({bear_prob:.0%})"
                
                if not exit_signal and bear_count >= params['bear_sharktooth_count']:
                    exit_signal = True
                    reason = f"BEAR_SHARK ({bear_count:.0f})"
                
                if not exit_signal and unrealized < -params['base_stop_loss']:
                    exit_signal = True
                    reason = "STOP_LOSS"
                
                if not exit_signal and dd_from_high > params['trailing_stop']:
                    exit_signal = True
                    reason = "TRAILING"
                
                # Sun opposition exit (kept as rule - ML doesn't learn exits well)
                if not exit_signal and cel['sun_opp_saturn'] and unrealized > 0.05:
                    exit_signal = True
                    reason = "SUN_SATURN_OPP"
                
                if exit_signal:
                    cash += shares * price
                    trade_return = (price - entry_price) / entry_price
                    
                    self.trade_log.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'size': entry_size, # Use stored size!
                        'return': trade_return,
                        'reason': reason,
                        'partial_taken': partial_exit_taken,
                        'override_applied': override_applied
                    })
                    
                    shares = 0
                    cooldown = 2
                    
                    if verbose and date.year >= 2020:
                        print(f"[{date.strftime('%Y-%m-%d')}] SELL @ ${price:.2f} ({reason}) "
                              f"Return: {trade_return:.2%}")
            
            # Track equity
            equity = cash + shares * price
            equity_curve.append({'date': date, 'equity': equity})
            if equity > peak_equity:
                peak_equity = equity
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
        
        # Final cleanup
        if shares > 0:
            cash += shares * close.iloc[-1]
        
        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital
        
        years = len(self.signal_data) / 252
        cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        returns = [t['return'] for t in self.trade_log]
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if returns and np.std(returns) > 0 else 0
        
        self.stats = {
            'total_return': total_return,
            'final_equity': final_value,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': len(self.trade_log),
            'total_entries': total_entries,
            'danger_zone_reductions': danger_zone_reductions,
            'exit_overrides': exit_overrides,
        }
        
        self.trades = self.trade_log
        return final_value, self.stats

    def print_results(self):
        s = self.stats
        print("\n" + "=" * 70)
        print("BULLETPROOF STRATEGY v7.5 - ECLIPSE")
        print("(Eclipse + Vedic + NEAT)")
        print("=" * 70)
        print(f"Total Return:     {s['total_return']*100:,.2f}%")
        print(f"Final Equity:     ${s['final_equity']:,.2f}")
        print(f"CAGR:             {s['cagr']*100:.2f}%")
        print(f"Max Drawdown:     {s['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:     {s['sharpe']:.2f}")
        print(f"Win Rate:         {s['win_rate']*100:.1f}%")
        print(f"Total Trades:     {s['total_trades']}")
        print("-" * 70)
        print(f"Danger Zone Reductions: {s.get('danger_zone_reductions', 0)}")
        print(f"Exit Overrides Used:    {s.get('exit_overrides', 0)}")
        print("=" * 70)
        
    def print_feature_importance(self):
        """Show feature importance with focus on celestial features."""
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS (v7.5)")
        print("=" * 70)
        
        if self.bull_detector and self.bull_detector.feature_importance is not None:
            print("\nBULL Detector - Top 20 Features:")
            for i, (feat, imp) in enumerate(self.bull_detector.feature_importance.head(20).items()):
                marker = ""
                if 'CELEST' in feat:
                    marker = " << CELESTIAL"
                elif 'PINE' in feat or 'PANIC' in feat:
                    marker = " <- PINE"
                print(f"    {i+1:2d}. {feat}: {imp:.4f}{marker}")
            
            # Specifically report celestial importance
            celest_imp = self.bull_detector.feature_importance[[
                f for f in self.bull_detector.feature_importance.index if 'CELEST' in f
            ]]
            if len(celest_imp) > 0:
                print(f"\n  CELESTIAL TOTAL IMPORTANCE: {celest_imp.sum():.4f} ({celest_imp.sum()*100:.2f}%)")

    def save_models(self, directory: str = '.'):
        os.makedirs(directory, exist_ok=True)
        if self.bull_detector:
            self.bull_detector.save(os.path.join(directory, 'bull_v75_eclipse.joblib'))
        if self.bear_detector:
            self.bear_detector.save(os.path.join(directory, 'bear_v75_eclipse.joblib'))
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
    print("BULLETPROOF STRATEGY v7.5 - ECLIPSE ML INTEGRATED")
    print("=" * 70)
    print("\nKey changes from v7.4:")
    print("  [+] Integrates solar_eclipse_regime (Bullish Bias)")
    print("  [+] Integrates lunar_eclipse_regime (Profit Taking)")
    print("  [+] Keeps all v7.4 Vedic logic")
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
        
        # SPY Backtest
        print("\n" + "=" * 70)
        print("BACKTEST: SPY")
        print("=" * 70)
        
        strategy_spy = BulletproofStrategyV75_Eclipse(spy, spy, vix, symbol='SPY')
        final_equity, stats = strategy_spy.backtest(verbose=False)
        strategy_spy.print_results()
        strategy_spy.print_feature_importance()
        
        # QQQ Backtest
        print("\n" + "=" * 70)
        print("BACKTEST: QQQ")
        print("=" * 70)
        
        strategy_qqq = BulletproofStrategyV75_Eclipse(spy, qqq, vix, symbol='QQQ')
        strategy_qqq.celestial_df = strategy_spy.celestial_df  # Reuse celestial data
        strategy_qqq.train_models(verbose=False)
        strategy_qqq.backtest(verbose=False)
        strategy_qqq.print_results()
        
        # Save
        strategy_spy.save_models('.')
        
        # Save trades
        if strategy_spy.trades:
            trades_df = pd.DataFrame(strategy_spy.trades)
            trades_df.to_csv('v7_5_trades_spy.csv', index=False)
            print("\nSaved trades to v7_5_trades_spy.csv")
            
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
