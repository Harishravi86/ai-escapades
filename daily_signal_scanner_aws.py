#!/usr/bin/env python3
"""
================================================================================
BULLETPROOF v7.5 - AWS DAILY SIGNAL SCANNER (ECLIPSE + VEDIC + ML)
================================================================================

Runs daily at 3:00 PM CST via cron and sends email with trading signals.
Based on "Champion" Strategy v7.5.

Setup:
    1. Install dependencies:
       pip install yfinance pandas pandas_ta ephem xgboost joblib scikit-learn

    2. Configure email & model paths in CONFIG section below.

    3. Add to crontab (3 PM CST = 4 PM EST = 21:00 UTC):
       crontab -e
       0 21 * * 1-5 /usr/bin/python3 /home/ubuntu/daily_signal_scanner_aws.py >> /var/log/bulletproof.log 2>&1

================================================================================
"""

import argparse
import os
import sys
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, date
import math
import traceback

import numpy as np
import pandas as pd
import joblib

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

try:
    import pandas_ta as ta
except ImportError:
    print("ERROR: pandas_ta not installed. Run: pip install pandas_ta")
    sys.exit(1)

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False
    print("WARNING: ephem not installed. Celestial features disabled.")

try:
    import xgboost as xgb
except ImportError:
    print("WARNING: xgboost not installed. models will fail to load.")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # ------------------------------------------------------------------
    # EMAIL SETTINGS
    # ------------------------------------------------------------------
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'smtp_username': 'harishravi86@gmail.com', 
    'smtp_password': os.environ.get('GMAIL_APP_PASSWORD', 'YOUR_APP_PASSWORD_HERE'),
    
    'email_from': 'bulletproof@yourdomain.com',
    'email_to': ['harishravi86@gmail.com'],

    # ------------------------------------------------------------------
    # MODEL PATHS (v7.5)
    # ------------------------------------------------------------------
    'bull_model_path': os.environ.get('BULL_MODEL_PATH', 'bull_v75_eclipse.joblib'),
    'bear_model_path': os.environ.get('BEAR_MODEL_PATH', 'bear_v75_eclipse.joblib'),
    
    # ------------------------------------------------------------------
    # STRATEGY PARAMETERS
    # ------------------------------------------------------------------
    'bull_threshold': 0.65,      # High confidence for entry
    'bear_threshold': 0.70,      # High confidence for exit
    
    # Spread regime thresholds
    'danger_zone_min': 170,
    'danger_zone_max': 230,
    'dispersed_threshold': 280,
    'compressed_threshold': 160,
}

# =============================================================================
# CELESTIAL ENGINE (v7.5)
# =============================================================================

class CelestialEngine:
    SPREAD_PLANETS = ['Sun', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                      'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    def __init__(self):
        self.solar_eclipses, self.lunar_eclipses = self._cache_eclipses_robust()

    def _cache_eclipses_robust(self):
        if not EPHEM_AVAILABLE: return [], []
        solar, lunar = [], []
        start_yr = datetime.now().year - 1
        end_yr = datetime.now().year + 2
        
        d = ephem.Date(f'{start_yr}-01-01')
        end_d = ephem.Date(f'{end_yr}-12-31')
        
        while d < end_d:
            try:
                nm = ephem.next_new_moon(d)
                if nm > end_d: break
                m = ephem.Moon(); m.compute(nm)
                if abs(ephem.Ecliptic(m).lat * 180/math.pi) < 1.6: solar.append(nm.datetime().date())
                d = ephem.Date(nm + 1)
            except: break
            
        d = ephem.Date(f'{start_yr}-01-01')
        while d < end_d:
            try:
                fm = ephem.next_full_moon(d)
                if fm > end_d: break
                m = ephem.Moon(); m.compute(fm)
                if abs(ephem.Ecliptic(m).lat * 180/math.pi) < 1.6: lunar.append(fm.datetime().date())
                d = ephem.Date(fm + 1)
            except: break
            
        return solar, lunar

    def get_features(self, date_obj) -> dict:
        if not EPHEM_AVAILABLE: return self._empty_features()
        try:
            obs = ephem.Observer()
            obs.date = date_obj + timedelta(hours=20) # Market close approx
            
            bodies = {n: getattr(ephem, n)() for n in ['Sun','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto']}
            pos = {}
            for n, b in bodies.items():
                b.compute(obs)
                pos[n] = math.degrees(ephem.Ecliptic(b).lon)
                
            sun_sat = self._get_sep(pos['Sun'], pos['Saturn'])
            moon_ura = self._get_sep(pos['Moon'], pos['Uranus'])
            sat_jup = self._get_sep(pos['Saturn'], pos['Jupiter'])
            
            spread_vals = [pos[p] for p in self.SPREAD_PLANETS]
            spread = self._calculate_spread(spread_vals)
            
            sat_dignity, sat_sign = self._get_saturn_dignity(pos['Saturn'])
            
            return {
                'CELEST_sun_saturn_sep': sun_sat / 180.0,
                'CELEST_moon_uranus_sep': moon_ura / 180.0,
                'CELEST_saturn_dignity': sat_dignity,
                'CELEST_eclipse_regime': self.get_eclipse_regime(date_obj),
                '#raw_spread': spread,
                '#spread_regime': self._classify_spread_regime(spread),
                '#raw_sun_sat': sun_sat,
                '#raw_moon_ura': moon_ura,
                '#saturn_sign': sat_sign,
                '#saturn_jupiter_sep': sat_jup
            }
        except Exception as e:
            print(f"Celestial Error: {e}")
            return self._empty_features()

    def _get_sep(self, l1, l2):
        d = abs(l1 - l2)
        return 360 - d if d > 180 else d

    def _calculate_spread(self, lons):
        lons = sorted(lons)
        n = len(lons)
        max_gap = 0
        for i in range(n):
            gap = (lons[(i+1)%n] - lons[i]) if (i+1)<n else (360 - lons[i] + lons[0])
            max_gap = max(max_gap, gap)
        return 360 - max_gap

    def _classify_spread_regime(self, spread):
        if spread > CONFIG['dispersed_threshold']: return 'DISPERSED'
        if spread < CONFIG['compressed_threshold']: return 'COMPRESSED'
        if CONFIG['danger_zone_min'] <= spread <= CONFIG['danger_zone_max']: return 'DANGER_ZONE'
        return 'NEUTRAL'

    def _get_saturn_dignity(self, trop_lon):
        sid = (trop_lon - 24.0) % 360
        signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
        sign_idx = int(sid / 30)
        name = signs[sign_idx]
        if sign_idx == 0: return -1, name # Aries
        if sign_idx == 6: return 1, name  # Libra
        return 0, name

    def get_eclipse_regime(self, dt):
        d = dt.date()
        for e in self.solar_eclipses:
            if abs((d - e).days) <= 3: return 1
        for e in self.lunar_eclipses:
            if abs((d - e).days) <= 3: return -1
        return 0

    def _empty_features(self):
        return {
            'CELEST_sun_saturn_sep': 0.5, 'CELEST_moon_uranus_sep': 0.5,
            'CELEST_saturn_dignity': 0, 'CELEST_eclipse_regime': 0,
            '#raw_spread': 200, '#spread_regime': 'NEUTRAL',
            '#raw_sun_sat': 90, '#raw_moon_ura': 90, '#saturn_sign': 'Unknown', '#saturn_jupiter_sep': 0
        }

# =============================================================================
# TECHNICAL ENGINE (v7.5 Inference)
# =============================================================================

def safe_series(c): return c.iloc[:,0] if isinstance(c, pd.DataFrame) else c

def calculate_features(df, celestial_data):
    # Ensure sufficient history
    if len(df) < 60: return None
    
    close = safe_series(df['Close'])
    high = safe_series(df['High'])
    low = safe_series(df['Low'])
    volume = safe_series(df['Volume'])
    
    feats = pd.DataFrame(index=df.index)
    
    # 1. RSI
    for length in [2, 5, 14, 21, 50]:
        rsi = ta.rsi(close, length=length)
        feats[f'RSI_{length}'] = rsi
        feats[f'RSI_{length}_oversold'] = (rsi < 30).astype(int)
        feats[f'RSI_{length}_extreme'] = (rsi < 20).astype(int)
        feats[f'RSI_{length}_overbought'] = (rsi > 70).astype(int)
        feats[f'RSI_{length}_extreme_high'] = (rsi > 80).astype(int)
    
    # 2. Bollinger Bands
    for length in [20, 50]:
        bb = ta.bbands(close, length=length, std=2.0)
        if bb is None: continue
        
        # Dynamic Column Finding
        cols = list(bb.columns)
        lower_col = next((c for c in cols if c.startswith(f'BBL_{length}')), None)
        mid_col = next((c for c in cols if c.startswith(f'BBM_{length}')), None) 
        upper_col = next((c for c in cols if c.startswith(f'BBU_{length}')), None)
        
        if lower_col and upper_col:
            lower = bb[lower_col]
            upper = bb[upper_col]
            # Handle potential division by zero
            denom = upper - lower
            denom = denom.replace(0, 0.0001) 
            pctb = (close - lower) / denom
            
            feats[f'BB_{length}_pctb'] = pctb
            feats[f'BB_{length}_oversold'] = (pctb < 0).astype(int)
            feats[f'BB_{length}_sharktooth'] = (pctb < -0.1).astype(int)
            feats[f'BB_{length}_overbought'] = (pctb > 1).astype(int)
            feats[f'BB_{length}_sharktooth_bear'] = (pctb > 1.1).astype(int)
            
            # Crossunder/Crossover need shift
            # Crossunder: Currently < -0.06 but Previous >= -0.06
            prev_pctb = pctb.shift(1)
            feats[f'BB_{length}_crossunder'] = ((pctb < -0.06) & (prev_pctb >= -0.06)).astype(int)
            feats[f'BB_{length}_crossover'] = ((pctb > 1.06) & (prev_pctb <= 1.06)).astype(int)

    # 3. MACD
    macd = ta.macd(close)
    if macd is not None:
        # MACD returns diff names usually: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        # Map them carefully
        cols = macd.columns
        line_col = next((c for c in cols if c.startswith('MACD_')), None)
        hist_col = next((c for c in cols if c.startswith('MACDh_')), None)
        sig_col = next((c for c in cols if c.startswith('MACDs_')), None)
        
        if line_col and hist_col and sig_col:
            feats['MACD_line'] = macd[line_col]
            feats['MACD_hist'] = macd[hist_col]
            feats['MACD_signal'] = macd[sig_col]
            feats['MACD_oversold'] = (feats['MACD_line'] < -2.0).astype(int)
    
    # 4. Stochastic
    stoch = ta.stoch(high, low, close)
    if stoch is not None:
        # Returns STOCHk_14_3_3, STOCHd_14_3_3
        cols = stoch.columns
        k_col = next((c for c in cols if c.startswith('STOCHk')), None)
        d_col = next((c for c in cols if c.startswith('STOCHd')), None)
        
        if k_col and d_col:
            k = stoch[k_col]
            d = stoch[d_col]
            feats['STOCH_k'] = k
            feats['STOCH_oversold'] = (k < 20).astype(int)
            feats['STOCH_sharktooth'] = ((k < 20) & (k > d)).astype(int)
    
    # 8. Price Action
    feats['RET_1d'] = close.pct_change(1) * 100
    feats['RET_5d'] = close.pct_change(5) * 100
    for period in [10, 20, 50]:
        rolling_max = close.rolling(period).max()
        feats[f'DD_{period}d'] = (close - rolling_max) / rolling_max * 100
        rolling_min = close.rolling(period).min()
        feats[f'RALLY_{period}d'] = (close - rolling_min) / rolling_min * 100
        
    # 9. Pine Script Features
    daily_return = close.pct_change(1)
    feats['DAILY_RETURN_PANIC'] = (daily_return < -0.0088).astype(int)
    if f'BB_20_crossunder' in feats.columns:
        feats['PINE_ENTRY_SIGNAL'] = (feats['BB_20_crossunder'] & feats['DAILY_RETURN_PANIC']).astype(int)
    else:
        feats['PINE_ENTRY_SIGNAL'] = 0
    
    # Merge Celestial (Last Row Logic handled in main)
    # But for calculation we should probably broadcast if needed, 
    # but here we just need the columns to exist if model expects them.
    # The caller manages merging the single-day celestial features.
    
    # 11. Celestial Features Integration
    # We iterate over the dict and assign to the whole column (constant value for the day, which matches inference context)
    for k, v in celestial_data.items():
        if not k.startswith('#'):
            feats[k] = v
            
    return feats.iloc[[-1]] # Return last row only

# =============================================================================
# MAIN LOGIC
# =============================================================================

def send_email(subject, html, text):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = CONFIG['email_from']
        msg['To'] = ', '.join(CONFIG['email_to'])
        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))
        
        ctx = ssl.create_default_context()
        with smtplib.SMTP(CONFIG['smtp_server'], CONFIG['smtp_port']) as server:
            server.starttls(context=ctx)
            server.login(CONFIG['smtp_username'], CONFIG['smtp_password'])
            server.sendmail(CONFIG['email_from'], CONFIG['email_to'], msg.as_string())
        print("Email sent!")
    except Exception as e:
        print(f"Email Failed: {e}")

def format_report(date_obj, price, change, vix, celestial, probs, signal):
    # Determine emoji/color
    s_color = '#eab308' # yellow
    if signal['type'] == 'BUY': s_color = '#22c55e'
    if signal['type'] == 'EXIT': s_color = '#ef4444'
    
    # Alerts
    alerts = []
    if celestial['#raw_moon_ura'] > 175: alerts.append('⚡ MOON-URANUS OPPOSITION (Entry Window)')
    if celestial['#raw_sun_sat'] > 175: alerts.append('⚡ SUN-SATURN OPPOSITION (Exit Signal)')
    if celestial['CELEST_eclipse_regime'] == 1: alerts.append('🌑 SOLAR ECLIPSE WINDOW (Accumulation Zone)')
    if celestial['CELEST_saturn_dignity'] == -1: alerts.append('⚠️ SATURN DEBILITATED (Reduced Sizing)')
    
    alert_html = ""
    for a in alerts: alert_html += f"<div style='padding:10px; background:#334155; margin-bottom:5px; border-left:4px solid #fcd34d;'>{a}</div>"

    html = f"""
    <html><body style='font-family:sans-serif; background:#0f172a; color:#e2e8f0; padding:20px;'>
    <div style='max-width:600px; margin:0 auto;'>
        <div style='text-align:center; padding:20px; background:#1e293b; border-radius:10px;'>
            <h2 style='margin:0;'>🛡️ Bulletproof v7.5</h2>
            <p style='color:#94a3b8;'>{date_obj.strftime('%Y-%m-%d')}</p>
        </div>
        
        <div style='text-align:center; padding:30px; margin:20px 0; background:#1e293b; border-radius:10px; border-left:5px solid {s_color};'>
            <h1 style='color:{s_color}; margin:0;'>{signal['type']}</h1>
            <p>Conviction: {signal['conviction']}</p>
            <p>Size: {signal['size']*100:.0f}%</p>
        </div>
        
        <div style='background:#1e293b; padding:15px; border-radius:10px; margin-bottom:15px;'>
            <h3 style='color:#94a3b8; border-bottom:1px solid #334155;'>📊 Market</h3>
            <p>SPY: <b>${price:.2f}</b> ({change:+.2f}%)</p>
            <p>VIX: <b>{vix:.2f}</b></p>
            <p>Bull Prob: <b>{probs['bull']:.1%}</b></p>
            <p>Bear Prob: <b>{probs['bear']:.1%}</b></p>
        </div>
        
        <div style='background:#1e293b; padding:15px; border-radius:10px;'>
            <h3 style='color:#94a3b8; border-bottom:1px solid #334155;'>🌙 Celestial</h3>
            <p>Eclipse Regime: <b>{celestial['CELEST_eclipse_regime']}</b></p>
            <p>Saturn Dignity: <b>{celestial['#saturn_sign']}</b> ({celestial['CELEST_saturn_dignity']})</p>
            <p>Spread Regime: <b>{celestial['#spread_regime']}</b> ({celestial['#raw_spread']:.1f}°)</p>
            <p>Moon-Uranus: {celestial['#raw_moon_ura']:.1f}°</p>
        </div>
        
        <br>
        {alert_html}
        
    </div></body></html>
    """
    return html

def main():
    print("Staritng v7.5 Scanner...")
    now = datetime.now()
    
    # 1. Get Market Data
    print("Fetching Data...")
    spy = yf.download("SPY", start=(now - timedelta(days=200)).strftime('%Y-%m-%d'), progress=False)
    vix_df = yf.download("^VIX", start=(now - timedelta(days=10)).strftime('%Y-%m-%d'), progress=False)
    
    if spy.empty:
        print("Empty Data"); return
        
    close = safe_series(spy['Close'])
    latest_price = float(close.iloc[-1])
    daily_change = float(close.pct_change().iloc[-1]) * 100
    vix = float(safe_series(vix_df['Close']).iloc[-1])
    
    print(f"SPY: ${latest_price:.2f}, VIX: {vix:.2f}")
    
    # 2. Celestial Features
    ce = CelestialEngine()
    celestial = ce.get_features(now)
    print(f"Celestial: Eclipse={celestial['CELEST_eclipse_regime']}, Saturn={celestial['#saturn_sign']}")
    
    # 3. Technical Features (ML Ready)
    # Calculate all potential features
    features = calculate_features(spy, celestial)
    
    # 4. Load Models & Predict
    probs = {'bull': 0.0, 'bear': 0.0}
    
    try:
        bull_data = joblib.load(CONFIG['bull_model_path'])
        bear_data = joblib.load(CONFIG['bear_model_path'])
        
        # Robust Inference using saved feature names
        if isinstance(bull_data, dict) and 'feature_names' in bull_data:
            fnames = bull_data['feature_names']
            
            # Ensure all needed columns exist (fill 0 if missing)
            for f in fnames:
                if f not in features.columns: features[f] = 0.0
                
            X = features[fnames].fillna(0)
            
            # Bull Prediction
            if 'scaler' in bull_data:
                X_scaled = pd.DataFrame(bull_data['scaler'].transform(X), columns=fnames)
                probs['bull'] = bull_data['model'].predict_proba(X_scaled)[:, 1][0]
            else:
                probs['bull'] = bull_data['model'].predict_proba(X)[:, 1][0]
                
            # Bear Prediction
            # Assuming same features for bear model (usually true for v7.5)
            if 'feature_names' in bear_data:
                fnames_bear = bear_data['feature_names']
                for f in fnames_bear:
                   if f not in features.columns: features[f] = 0.0
                X_bear = features[fnames_bear].fillna(0)
                
                if 'scaler' in bear_data:
                     X_bear_scaled = pd.DataFrame(bear_data['scaler'].transform(X_bear), columns=fnames_bear)
                     probs['bear'] = bear_data['model'].predict_proba(X_bear_scaled)[:, 1][0]
                else:
                     probs['bear'] = bear_data['model'].predict_proba(X_bear)[:, 1][0]
            
            print(f"Probabilities: Bull={probs['bull']:.3f}, Bear={probs['bear']:.3f}")
            
        else:
             print("Error: Model file format not recognized (missing feature_names)")
             # Fallback 0.0
             
    except Exception as e:
        print(f"Model Inference Failed: {e}")
        traceback.print_exc()
        pass 
        
    # 5. Determine Signal
    signal = {'type': 'HOLD', 'conviction': 'NONE', 'size': 0.0}
    
    # Multipliers
    regime_mult = 1.0
    if celestial['#spread_regime'] == 'DANGER_ZONE': regime_mult = 0.8
    if celestial['CELEST_saturn_dignity'] == -1: regime_mult *= 0.9
    
    # Entry
    if probs['bull'] > CONFIG['bull_threshold']:
        signal['type'] = 'BUY'
        signal['conviction'] = 'HIGH' if probs['bull'] > 0.8 else 'MEDIUM'
        signal['size'] = 1.0 * regime_mult
        
    # Exit
    if probs['bear'] > CONFIG['bear_threshold']:
        signal['type'] = 'EXIT'
        signal['conviction'] = 'AI_BEAR'
        signal['size'] = 0.0
        
    if celestial['#raw_sun_sat'] > 175:
        signal['type'] = 'EXIT'
        signal['conviction'] = 'SUN_SATURN_OPP'
        signal['size'] = 0.0
        
    print(f"Final Signal: {signal}")
    
    # 6. Send Email
    subject = f"[{signal['type']}] Bulletproof v7.5: {signal['type']} - SPY ${latest_price:.2f}"
    html = format_report(now, latest_price, daily_change, vix, celestial, probs, signal)
    send_email(subject, html, "Please enable HTML to view report.")

if __name__ == "__main__":
    main()
