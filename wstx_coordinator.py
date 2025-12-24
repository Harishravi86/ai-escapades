# WST-X: Wall Street Trader eXpert
# Integrated with V6.2 restored feature definitions (Bulletproof v6.2)
# - Provides 7 engines: Trend, Volatility, Reversal, ML, Macro, Event, PositionSizing
# - Includes a conversational HTTP layer (FastAPI) + simple CLI
# - Drop-in compatible with your V6.2 TechnicalEngine & SharktoothDetector definitions

"""
Run:
    pip install -r requirements.txt
    uvicorn wstx_coordinator:app --reload --port 8000

Endpoints:
    GET /signal?ticker=QQQ
    POST /ask {"ticker":"QQQ","question":"Should I buy?","cash":100000}

This file assumes you have the v6.2 modules available in the same folder or integrated below.
"""

import os
import math
import time
import joblib
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from sklearn.preprocessing import StandardScaler

# Lightweight web layer
from fastapi import FastAPI, Query
from pydantic import BaseModel

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

# ... (Configuration and Utils remain same)

# ---------------------------
# Celestial Engine (Restored from v6.2)
# ---------------------------
class CelestialEngine:
    def __init__(self):
        self.enabled = EPHEM_AVAILABLE
        self._cache = {}
    
    def get_features(self, date_str: str) -> Dict[str, bool]:
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

# ... (TechnicalEngineV62 remains same)

class SharktoothDetectorV62:
    """Restored detector matching v6.2 behavior. Trains on TechnicalEngineV62 features and labels."""
    def __init__(self, mode='bull', lookahead_days=10, min_move_pct=3.0, lookback_days=5):
        self.mode = mode
        self.lookahead_days = lookahead_days
        self.min_move_pct = min_move_pct
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _label_turning_points(self, close: pd.Series) -> pd.Series:
        labels = pd.Series(0, index=close.index)
        N = len(close)
        for i in range(self.lookback_days, N - self.lookahead_days):
            current_price = close.iloc[i]
            lookback_prices = close.iloc[max(0, i - self.lookback_days):i]
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
            else:
                if current_price < lookback_prices.max():
                    continue
                min_future = future_prices.min()
                move_pct = (current_price - min_future) / current_price * 100
                if move_pct >= self.min_move_pct:
                    labels.iloc[i] = 1
        return labels

    def prepare_data(self, df: pd.DataFrame):
        feats = TechnicalEngineV62.calculate(df)
        close = safe_series(df['Close'])
        labels = self._label_turning_points(close)
        valid_idx = feats.dropna().index.intersection(labels.dropna().index)
        # trim lookback/lookahead edges
        if len(valid_idx) > self.lookahead_days + self.lookback_days:
            valid_idx = valid_idx[self.lookback_days:-self.lookahead_days]
        X = feats.loc[valid_idx].fillna(0)
        y = labels.loc[valid_idx]
        return X, y

    def train(self, df: pd.DataFrame, verbose: bool = False):
        X, y = self.prepare_data(df)
        if X.shape[0] == 0:
            raise RuntimeError('Not enough data to train SharktoothDetector')
        self.feature_names = X.columns.tolist()
        Xs = self.scaler.fit_transform(X)
        
        if XGB_AVAILABLE:
            # UNIFIED v7.0: Added min_child_weight=3 for regularization (Opus recommendation)
            self.model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                min_child_weight=3, # <--- Added regularization
                scale_pos_weight=(len(y)-y.sum())/max(y.sum(),1),
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                use_label_encoder=False, eval_metric='logloss'
            )
        else:
            self.model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
        
        self.model.fit(Xs, y)
        if verbose and hasattr(self.model, 'feature_importances_'):
            fi = pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
            print(f"Trained {self.mode} detector. Top features:\n{fi.head(8)}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError('Model not trained')
        feats = TechnicalEngineV62.calculate(df)
        for f in self.feature_names:
            if f not in feats.columns:
                feats[f] = 0
        X = feats[self.feature_names].fillna(0)
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)[:, 1]
        return pd.Series(probs, index=df.index)

    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'mode': self.mode
        }, path)

    @classmethod
    def load(cls, path: str) -> 'SharktoothDetectorV62':
        data = joblib.load(path)
        inst = cls(mode=data.get('mode', 'bull'))
        inst.model = data['model']
        inst.scaler = data['scaler']
        inst.feature_names = data['feature_names']
        return inst

# ---------------------------
# Engines
# ---------------------------

class TrendEngine:
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        close = safe_series(df['Close'])
        s5 = np.log(close).diff(5).rolling(5).mean()
        s21 = np.log(close).diff(21).rolling(5).mean()
        bias = 'FLAT'
        if s5.iloc[-1] > 0 and s21.iloc[-1] > 0:
            bias='BULL'
        elif s5.iloc[-1] < 0 and s21.iloc[-1] < 0:
            bias='BEAR'
        return {'bias': bias, 's5': float(s5.iloc[-1]), 's21': float(s21.iloc[-1])}

class VolatilityEngine:
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        close = safe_series(df['Close'])
        rv20 = close.pct_change().rolling(20).std() * math.sqrt(252)
        rv5 = close.pct_change().rolling(5).std() * math.sqrt(252)
        vol_state='NORMAL'
        if rv20.iloc[-1] > 0.6: vol_state='ELEVATED'
        if rv20.iloc[-1] > 1.0: vol_state='EXTREME'
        return {'rv20': float(rv20.iloc[-1]), 'rv5': float(rv5.iloc[-1]), 'state': vol_state}

class ReversalEngine:
    @staticmethod
    def score(df: pd.DataFrame) -> Dict[str, Any]:
        feats = TechnicalEngineV62.calculate(df)
        last = feats.iloc[-1]
        signals = {}
        signals['oversold_count'] = int(last.get('OVERSOLD_COUNT', 0))
        signals['bull_shark_count'] = int(last.get('BULL_SHARKTOOTH_COUNT', 0))
        signals['pctb20'] = float(last.get('BB_20_pctb', np.nan) if 'BB_20_pctb' in feats.columns else last.get('BB_20_pctb', np.nan))
        return signals

class MacroEngine:
    @staticmethod
    def score(vix_df: pd.DataFrame=None) -> Dict[str, Any]:
        if vix_df is None:
            return {'vix': None, 'macro': 'NEUTRAL'}
        v = safe_series(vix_df['Close']).iloc[-1]
        if v>30:
            return {'vix': float(v), 'macro':'RISK_OFF'}
        if v>20:
            return {'vix': float(v), 'macro':'CAUTION'}
        return {'vix': float(v), 'macro':'RISK_ON'}

class EventEngine:
    @staticmethod
    def check_events(date: pd.Timestamp, ticker: str) -> Dict[str, Any]:
        # Minimal implementation: check if next day is earnings using yfinance
        try:
            cal = yf.Ticker(ticker).calendar
            # calendar rows vary; this is a safe attempt
            return {'has_earnings': False}
        except Exception:
            return {'has_earnings': False}

class PositionSizingEngine:
    @staticmethod
    def size(confidence: float, vol_state: str, base_cash: float=100000) -> Dict[str, Any]:
        # confidence in [0,1]
        if confidence >= 0.8:
            size_pct = 1.0
        elif confidence >= 0.6:
            size_pct = 0.5
        elif confidence >= 0.4:
            size_pct = 0.25
        else:
            size_pct = 0.0
        if vol_state == 'ELEVATED':
            size_pct *= 0.66
        if vol_state == 'EXTREME':
            size_pct *= 0.33
        return {'size_pct': size_pct, 'notional': base_cash * size_pct}

class WSTXCoordinator:
    def __init__(self):
        self.bull_detector = SharktoothDetectorV62(mode='bull')
        self.bear_detector = SharktoothDetectorV62(mode='bear')
        self.celestial = CelestialEngine() # Added Celestial
        self.models_trained_on = {}

    # ... (train_if_needed remains same)

    def signal(self, ticker: str, lookback_days: int = 800) -> Dict[str, Any]:
        df = load_yf(ticker, start=(pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d'))
        # ... (data loading checks)

        feats = TechnicalEngineV62.calculate(df)
        self.train_if_needed(df)

        bull_probs = self.bull_detector.predict(df)
        bear_probs = self.bear_detector.predict(df)
        bull_prob = float(bull_probs.iloc[-1])
        bear_prob = float(bear_probs.iloc[-1])

        trend = TrendEngine.score(df)
        vol = VolatilityEngine.score(df)
        rev = ReversalEngine.score(df)
        macro = MacroEngine.score(vix_df=None)
        events = EventEngine.check_events(df.index[-1], ticker)
        
        # Celestial
        date_str = str(df.index[-1].date())
        cel_feats = self.celestial.get_features(date_str)

        # ... (Composite calculation remains same)
        
        # Override with Conviction Filter (v6.2 + Celestial)
        bull_count = rev.get('bull_shark_count', 0)
        
        HIGH_CONVICTION_PROB = 0.70
        HIGH_CONVICTION_COUNT = 4
        MEDIUM_CONVICTION_PROB = 0.50
        MEDIUM_CONVICTION_COUNT = 3
        
        size_pct = 0.0
        action = 'HOLD'
        reason = 'Neutral'
        
        if bull_prob > HIGH_CONVICTION_PROB or bull_count >= HIGH_CONVICTION_COUNT:
            size_pct = 1.0
            conf = 0.95
            action = 'BUY'
            reason = 'HIGH CONVICTION (Sharktooth/ML)'
        elif bull_prob > MEDIUM_CONVICTION_PROB or bull_count >= MEDIUM_CONVICTION_COUNT:
            size_pct = 0.5
            conf = max(conf, 0.75)
            action = 'BUY'
            reason = 'MEDIUM CONVICTION'
        
        # Celestial Modifiers
        if size_pct > 0:
            if cel_feats['moon_opp_uranus']:
                size_pct = min(size_pct * 1.25, 1.0)
                reason += ' + CELESTIAL BOOST (Moon-Uranus)'
        
        # Celestial Exit Warning
        if cel_feats['sun_opp_saturn']:
            action = 'SELL' if size_pct == 0 else 'REDUCE'
            reason = 'CELESTIAL WARNING (Sun-Saturn)'
            size_pct = 0.0 # Force exit on Sun-Saturn? v6.2 logic says "Reduce" or "Exit"?
            # v6.2 backtest logic: if cel['sun_opp_saturn']: exit_signal = True
            # So yes, force exit.

        sizing = {'size_pct': size_pct, 'notional': 100000 * size_pct}

        return {
            'ticker': ticker,
            'date': date_str,
            'bull_prob': bull_prob,
            'bear_prob': bear_prob,
            'confidence': conf,
            'trend': trend,
            'volatility': vol,
            'reversal': rev,
            'macro': macro,
            'celestial': cel_feats, # Added to output
            'events': events,
            'size': sizing,
            'action': action,
            'reason': reason
        }

# ---------------------------
# FastAPI conversational layer
# ---------------------------

app = FastAPI(title='WST-X Trade Assistant')
coordinator = WSTXCoordinator()

class AskRequest(BaseModel):
    ticker: str
    question: str
    cash: Optional[float] = 100000

@app.get('/signal')
def get_signal(ticker: str = Query(..., description='Ticker symbol (e.g. QQQ)')):
    try:
        sig = coordinator.signal(ticker)
        return sig
    except Exception as e:
        return {'error': str(e)}

@app.post('/ask')
def ask(req: AskRequest):
    # Minimal conversational wrapper: use the signal + question heuristics
    sig = coordinator.signal(req.ticker)
    # Very simple natural-language mapping for now
    q = req.question.lower()
    reply = {}
    if 'buy' in q or 'long' in q:
        reply['recommendation'] = sig['action']
        reply['reason'] = 'Combined model outputs (ML + Trend + Reversal + Macro)'
        reply['sizing'] = sig['size']
    elif 'sell' in q or 'short' in q:
        reply['recommendation'] = 'SELL' if sig['confidence']<0.35 else 'HOLD'
        reply['reason'] = 'Confidence low for long, trend/vol considerations'
    elif 'hold' in q or 'wait' in q or 'position' in q:
        reply['recommendation'] = 'HOLD' if 0.35 < sig['confidence'] < 0.6 else sig['action']
        reply['reason'] = 'Neutral confidence or active signal'
    else:
        reply['recommendation'] = sig['action']
        reply['reason'] = 'Default composite decision'

    reply['signal'] = sig
    return reply

# ---------------------------
# CLI utility
# ---------------------------
if __name__ == '__main__':
    print('WST-X coordinator starting (CLI mode).')
    coordinator.train_if_needed(load_yf('SPY'))
    while True:
        try:
            ticker = input('Ticker (or q to quit): ').strip().upper()
            if ticker.lower() in ('q','quit','exit'):
                break
            sig = coordinator.signal(ticker)
            print('\n=== SIGNAL ===')
            for k,v in sig.items():
                print(f"{k}: {v}")
            print('============\n')
        except Exception as e:
            print('Error:', e)
            continue
