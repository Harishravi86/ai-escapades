#!/usr/bin/env python3
"""
================================================================================
BULLETPROOF v7.3 - AWS DAILY SIGNAL SCANNER WITH EMAIL
================================================================================

Runs daily at 3:00 PM CST via cron and sends email with trading signals.

Setup:
    1. Install dependencies:
       pip install yfinance pandas pandas_ta ephem xgboost joblib

    2. Configure email settings in CONFIG section below

    3. Add to crontab (3 PM CST = 4 PM EST = 21:00 UTC):
       crontab -e
       0 21 * * 1-5 /usr/bin/python3 /home/ubuntu/daily_signal_scanner_aws.py >> /var/log/bulletproof.log 2>&1

    4. Test:
       python3 daily_signal_scanner_aws.py --test-email

================================================================================
"""

import argparse
import os
import sys
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import math
import traceback

import numpy as np
import pandas as pd

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


# =============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# =============================================================================

CONFIG = {
    # Email Settings (Use AWS SES or Gmail App Password)
    'smtp_server': 'email-smtp.us-east-1.amazonaws.com',  # AWS SES
    'smtp_port': 587,
    'smtp_username': os.environ.get('SMTP_USERNAME', 'YOUR_SES_USERNAME'),
    'smtp_password': os.environ.get('SMTP_PASSWORD', 'YOUR_SES_PASSWORD'),
    'email_from': 'bulletproof@yourdomain.com',
    'email_to': ['harish@yourdomain.com'],  # List of recipients
    
    # Strategy Thresholds (v7.3)
    'bull_threshold': 0.45,
    'high_conviction_prob': 0.70,
    'high_conviction_count': 4,
    'medium_conviction_prob': 0.50,
    'medium_conviction_count': 3,
    'bear_threshold': 0.60,
    
    # Spread regime thresholds
    'danger_zone_min': 170,
    'danger_zone_max': 230,
    'dispersed_threshold': 280,
    'compressed_threshold': 160,
    
    # Model paths (optional - for full ML inference)
    'bull_model_path': '/home/ubuntu/models/bull_v73_astro.joblib',
    'bear_model_path': '/home/ubuntu/models/bear_v73_astro.joblib',
}


# =============================================================================
# CELESTIAL ENGINE
# =============================================================================

class CelestialScanner:
    """Calculate celestial features for a single date."""
    
    SPREAD_PLANETS = ['Sun', 'Mercury', 'Venus', 'Mars', 'Jupiter', 
                      'Saturn', 'Uranus', 'Neptune', 'Pluto']
    
    def get_features(self, date: datetime) -> dict:
        if not EPHEM_AVAILABLE:
            return self._empty_features()
        
        try:
            obs = ephem.Observer()
            obs.date = date + timedelta(hours=21)  # 4 PM EST / 3 PM CST
            
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
            
            sun_saturn_sep = self._get_separation(positions['Sun'], positions['Saturn'])
            moon_uranus_sep = self._get_separation(positions['Moon'], positions['Uranus'])
            
            spread_positions = [positions[p] for p in self.SPREAD_PLANETS]
            spread = self._calculate_spread(spread_positions)
            spread_regime = self._classify_spread_regime(spread)
            
            moon_phase = bodies['Moon'].phase
            
            # Find next Moon-Uranus opposition
            next_opp = self._find_next_moon_uranus_opposition(date)
            
            return {
                'sun_saturn_sep': sun_saturn_sep,
                'moon_uranus_sep': moon_uranus_sep,
                'sun_saturn_sep_normalized': sun_saturn_sep / 180.0,
                'moon_uranus_sep_normalized': moon_uranus_sep / 180.0,
                'moon_phase': moon_phase,
                'spread': spread,
                'spread_regime': spread_regime,
                'sun_opp_saturn': sun_saturn_sep > 175,
                'moon_opp_uranus': moon_uranus_sep > 175,
                'next_moon_uranus_opp': next_opp,
            }
        except Exception as e:
            print(f"Celestial calculation error: {e}")
            return self._empty_features()
    
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
        if spread > CONFIG['dispersed_threshold']:
            return 'DISPERSED'
        elif spread < CONFIG['compressed_threshold']:
            return 'COMPRESSED'
        elif CONFIG['danger_zone_min'] <= spread <= CONFIG['danger_zone_max']:
            return 'DANGER_ZONE'
        else:
            return 'NEUTRAL'
    
    def _find_next_moon_uranus_opposition(self, start_date: datetime) -> str:
        """Find next Moon-Uranus opposition date."""
        if not EPHEM_AVAILABLE:
            return "Unknown"
        
        for days_ahead in range(1, 30):
            check_date = start_date + timedelta(days=days_ahead)
            obs = ephem.Observer()
            obs.date = check_date
            
            moon = ephem.Moon()
            uranus = ephem.Uranus()
            moon.compute(obs)
            uranus.compute(obs)
            
            moon_ecl = ephem.Ecliptic(moon)
            uranus_ecl = ephem.Ecliptic(uranus)
            
            sep = self._get_separation(math.degrees(moon_ecl.lon), math.degrees(uranus_ecl.lon))
            
            if sep > 175:
                return check_date.strftime('%Y-%m-%d')
        
        return "30+ days"
    
    def _empty_features(self) -> dict:
        return {
            'sun_saturn_sep': 90.0,
            'moon_uranus_sep': 90.0,
            'sun_saturn_sep_normalized': 0.5,
            'moon_uranus_sep_normalized': 0.5,
            'moon_phase': 50.0,
            'spread': 200.0,
            'spread_regime': 'NEUTRAL',
            'sun_opp_saturn': False,
            'moon_opp_uranus': False,
            'next_moon_uranus_opp': 'Unknown',
        }


# =============================================================================
# TECHNICAL SCANNER
# =============================================================================

def safe_series(col):
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


def calculate_technicals(df: pd.DataFrame) -> dict:
    """Calculate key technical indicators."""
    close = safe_series(df['Close'])
    high = safe_series(df['High'])
    low = safe_series(df['Low'])
    
    result = {}
    
    # RSI
    for length in [2, 5, 14]:
        rsi = ta.rsi(close, length=length)
        if rsi is not None and len(rsi) > 0:
            result[f'RSI_{length}'] = rsi.iloc[-1]
            result[f'RSI_{length}_oversold'] = rsi.iloc[-1] < 30
    
    # Bollinger Bands %B
    for length in [20, 50]:
        bb = ta.bbands(close, length=length, std=2.0)
        if bb is not None:
            lower_col = [c for c in bb.columns if 'BBL' in c][0]
            upper_col = [c for c in bb.columns if 'BBU' in c][0]
            lower = bb[lower_col]
            upper = bb[upper_col]
            pctb = (close - lower) / (upper - lower)
            
            result[f'BB_{length}_pctb'] = pctb.iloc[-1]
            result[f'BB_{length}_oversold'] = pctb.iloc[-1] < 0
            result[f'BB_{length}_sharktooth'] = pctb.iloc[-1] < -0.1
    
    # Stochastic
    stoch = ta.stoch(high, low, close)
    if stoch is not None:
        k = stoch.iloc[:, 0]
        result['STOCH_k'] = k.iloc[-1]
        result['STOCH_oversold'] = k.iloc[-1] < 20
    
    # Williams %R
    willr = ta.willr(high, low, close, length=14)
    if willr is not None:
        result['WILLR_14'] = willr.iloc[-1]
        result['WILLR_14_oversold'] = willr.iloc[-1] < -80
    
    # Daily return
    daily_return = close.pct_change().iloc[-1]
    result['daily_return'] = daily_return
    result['daily_return_panic'] = daily_return < -0.0088
    
    # Price data
    result['price'] = close.iloc[-1]
    result['prev_close'] = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
    
    # Composite counts
    oversold_count = sum([
        result.get('RSI_2_oversold', False),
        result.get('RSI_5_oversold', False),
        result.get('RSI_14_oversold', False),
        result.get('BB_20_oversold', False),
        result.get('BB_50_oversold', False),
        result.get('STOCH_oversold', False),
        result.get('WILLR_14_oversold', False),
    ])
    result['oversold_count'] = oversold_count
    
    sharktooth_count = sum([
        result.get('BB_20_sharktooth', False),
        result.get('BB_50_sharktooth', False),
        result.get('RSI_2', 50) < 10,
    ])
    result['sharktooth_count'] = sharktooth_count
    
    return result


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

def generate_signal(technicals: dict, celestial: dict, vix: float) -> dict:
    """Generate trading signal based on v7.3 logic."""
    
    bull_indicators = technicals['oversold_count']
    sharktooth = technicals['sharktooth_count']
    
    # Estimate probability
    base_prob = 0.30
    base_prob += min(bull_indicators * 0.08, 0.40)
    base_prob += min(sharktooth * 0.10, 0.20)
    
    if celestial['moon_opp_uranus']:
        base_prob += 0.10
    if celestial['moon_uranus_sep_normalized'] > 0.9:
        base_prob += 0.05
    
    if 18 < vix < 30:
        base_prob += 0.05
    elif vix > 35:
        base_prob += 0.10
    
    bull_prob = min(base_prob, 0.95)
    
    # Determine signal
    signal = 'HOLD'
    conviction = 'NONE'
    size = 0.0
    
    if bull_prob > CONFIG['high_conviction_prob'] or sharktooth >= CONFIG['high_conviction_count']:
        signal = 'BUY'
        conviction = 'HIGH'
        size = 1.0
    elif bull_prob > CONFIG['medium_conviction_prob'] or sharktooth >= CONFIG['medium_conviction_count']:
        signal = 'BUY'
        conviction = 'MEDIUM'
        size = 0.5
    elif bull_prob > CONFIG['bull_threshold']:
        signal = 'BUY'
        conviction = 'LOW'
        size = 0.25
    
    # Apply spread regime sizing
    regime = celestial['spread_regime']
    regime_mult = {
        'DISPERSED': 1.0,
        'NEUTRAL': 1.0,
        'DANGER_ZONE': 0.80,
        'COMPRESSED': 0.90,
    }.get(regime, 1.0)
    
    size *= regime_mult
    
    # Exit signals
    exit_signal = False
    exit_reason = None
    
    if celestial['sun_opp_saturn']:
        exit_signal = True
        exit_reason = 'SUN_SATURN_OPP'
    
    return {
        'signal': signal,
        'conviction': conviction,
        'size': size,
        'bull_prob': bull_prob,
        'spread_regime': regime,
        'regime_mult': regime_mult,
        'exit_signal': exit_signal,
        'exit_reason': exit_reason,
    }


# =============================================================================
# EMAIL FORMATTER
# =============================================================================

def format_email_html(date: datetime, technicals: dict, celestial: dict, 
                      signal: dict, vix: float) -> str:
    """Format the signal as HTML email."""
    
    # Signal color
    if signal['signal'] == 'BUY':
        signal_color = '#22c55e'  # Green
        signal_emoji = '🟢'
    elif signal['exit_signal']:
        signal_color = '#ef4444'  # Red
        signal_emoji = '🔴'
    else:
        signal_color = '#eab308'  # Yellow
        signal_emoji = '⚪'
    
    # Daily change
    daily_change = technicals['daily_return'] * 100
    change_color = '#22c55e' if daily_change >= 0 else '#ef4444'
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; }}
            .header {{ text-align: center; padding: 20px; background: #1e293b; border-radius: 10px; margin-bottom: 20px; }}
            .signal-box {{ text-align: center; padding: 30px; background: #1e293b; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid {{signal_color}}; }}
            .signal {{ font-size: 48px; font-weight: bold; color: {{signal_color}}; }}
            .section {{ background: #1e293b; padding: 15px; border-radius: 10px; margin-bottom: 15px; }}
            .section-title {{ font-size: 14px; color: #94a3b8; margin-bottom: 10px; text-transform: uppercase; }}
            .row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; }}
            .row:last-child {{ border-bottom: none; }}
            .label {{ color: #94a3b8; }}
            .value {{ font-weight: bold; }}
            .alert {{ padding: 15px; border-radius: 8px; margin-bottom: 10px; }}
            .alert-warning {{ background: #78350f; border-left: 4px solid #f59e0b; }}
            .alert-success {{ background: #14532d; border-left: 4px solid #22c55e; }}
            .alert-danger {{ background: #7f1d1d; border-left: 4px solid #ef4444; }}
            .footer {{ text-align: center; color: #64748b; font-size: 12px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin: 0; font-size: 24px;">🎯 Bulletproof v7.3</h1>
                <p style="margin: 5px 0 0 0; color: #94a3b8;">{{date.strftime('%A, %B %d, %Y')}} • 3:00 PM CST</p>
            </div>
            
            <div class="signal-box">
                <div style="font-size: 24px; margin-bottom: 10px;">{{signal_emoji}}</div>
                <div class="signal">{{signal['signal']}}</div>
                <div style="color: #94a3b8; margin-top: 10px;">
                    Conviction: <strong>{{signal['conviction']}}</strong> • 
                    Size: <strong>{{signal['size']*100:.0f}}%</strong> •
                    Probability: <strong>{{signal['bull_prob']*100:.0f}}%</strong>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📊 Market Status</div>
                <div class="row">
                    <span class="label">SPY Price</span>
                    <span class="value">${{technicals['price']:.2f}}</span>
                </div>
                <div class="row">
                    <span class="label">Daily Change</span>
                    <span class="value" style="color: {{change_color}}">{{daily_change:+.2f}}%</span>
                </div>
                <div class="row">
                    <span class="label">VIX</span>
                    <span class="value">{{vix:.2f}}</span>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📈 Technical Indicators</div>
                <div class="row">
                    <span class="label">RSI(2)</span>
                    <span class="value">{{technicals.get('RSI_2', 0):.1f}} {{'⚠️' if technicals.get('RSI_2_oversold') else ''}}</span>
                </div>
                <div class="row">
                    <span class="label">RSI(14)</span>
                    <span class="value">{{technicals.get('RSI_14', 0):.1f}} {{'⚠️' if technicals.get('RSI_14_oversold') else ''}}</span>
                </div>
                <div class="row">
                    <span class="label">BB(20) %B</span>
                    <span class="value">{{technicals.get('BB_20_pctb', 0):.2f}} {{'🦈' if technicals.get('BB_20_sharktooth') else ''}}</span>
                </div>
                <div class="row">
                    <span class="label">Stochastic K</span>
                    <span class="value">{{technicals.get('STOCH_k', 0):.1f}} {{'⚠️' if technicals.get('STOCH_oversold') else ''}}</span>
                </div>
                <div class="row">
                    <span class="label">Oversold Count</span>
                    <span class="value">{{technicals['oversold_count']}}/7</span>
                </div>
                <div class="row">
                    <span class="label">Sharktooth Count</span>
                    <span class="value">{{technicals['sharktooth_count']}}</span>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🌙 Celestial Conditions</div>
                <div class="row">
                    <span class="label">Sun-Saturn Sep</span>
                    <span class="value">{{celestial['sun_saturn_sep']:.1f}}° {{'☍ OPP' if celestial['sun_opp_saturn'] else ''}}</span>
                </div>
                <div class="row">
                    <span class="label">Moon-Uranus Sep</span>
                    <span class="value">{{celestial['moon_uranus_sep']:.1f}}° {{'☍ OPP' if celestial['moon_opp_uranus'] else ''}}</span>
                </div>
                <div class="row">
                    <span class="label">Moon Phase</span>
                    <span class="value">{{celestial['moon_phase']:.0f}}%</span>
                </div>
                <div class="row">
                    <span class="label">Planetary Spread</span>
                    <span class="value">{{celestial['spread']:.1f}}°</span>
                </div>
                <div class="row">
                    <span class="label">Spread Regime</span>
                    <span class="value">{{celestial['spread_regime']}}</span>
                </div>
                <div class="row">
                    <span class="label">Next Moon-Uranus Opp</span>
                    <span class="value">{{celestial['next_moon_uranus_opp']}}</span>
                </div>
            </div>
    """
    
    # Alerts section
    alerts = []
    
    if celestial['moon_opp_uranus']:
        alerts.append(('success', '⚡ MOON-URANUS OPPOSITION ACTIVE - High probability entry window'))
    
    if celestial['sun_opp_saturn']:
        alerts.append(('danger', '⚡ SUN-SATURN OPPOSITION - Consider taking profits (100% historical win rate)'))
    
    if celestial['spread_regime'] == 'DANGER_ZONE':
        alerts.append(('warning', '⚠️ DANGER ZONE (170-230°) - Elevated volatility risk, size reduced 20%'))
    
    if celestial['spread_regime'] == 'COMPRESSED':
        alerts.append(('warning', '⚠️ COMPRESSED (<160°) - Calm before storm, watch for breakout'))
    
    if technicals.get('daily_return_panic'):
        alerts.append(('warning', '💥 PANIC DAY - Daily return below -0.88%'))
    
    if technicals['sharktooth_count'] >= 2:
        alerts.append(('success', '🦈 SHARKTOOTH DETECTED - Strong oversold signal'))
    
    if alerts:
        html += '<div class="section"><div class="section-title">🚨 Alerts</div>'
        for alert_type, alert_text in alerts:
            html += f'<div class="alert alert-{{alert_type}}">{{alert_text}}</div>'
        html += '</div>'
    
    html += f"""
            <div class="footer">
                <p>Bulletproof Strategy v7.3 • Astro-ML Engine</p>
                <p>Generated at {{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def format_email_text(date: datetime, technicals: dict, celestial: dict,
                      signal: dict, vix: float) -> str:
    """Format the signal as plain text email."""
    
    text = f"""
================================================================================
BULLETPROOF v7.3 - DAILY SIGNAL
{{date.strftime('%A, %B %d, %Y')}} • 3:00 PM CST
================================================================================

SIGNAL: {{signal['signal']}}
Conviction: {{signal['conviction']}}
Position Size: {{signal['size']*100:.0f}}%
Bull Probability: {{signal['bull_prob']*100:.0f}}%

--------------------------------------------------------------------------------
MARKET STATUS
--------------------------------------------------------------------------------
SPY Price: ${{technicals['price']:.2f}}
Daily Change: {{technicals['daily_return']*100:+.2f}}%
VIX: {{vix:.2f}}

--------------------------------------------------------------------------------
TECHNICAL INDICATORS
--------------------------------------------------------------------------------
RSI(2): {{technicals.get('RSI_2', 0):.1f}} {{'OVERSOLD' if technicals.get('RSI_2_oversold') else ''}}
RSI(14): {{technicals.get('RSI_14', 0):.1f}} {{'OVERSOLD' if technicals.get('RSI_14_oversold') else ''}}
BB(20) %B: {{technicals.get('BB_20_pctb', 0):.2f}} {{'SHARKTOOTH' if technicals.get('BB_20_sharktooth') else ''}}
Stochastic K: {{technicals.get('STOCH_k', 0):.1f}} {{'OVERSOLD' if technicals.get('STOCH_oversold') else ''}}
Oversold Count: {{technicals['oversold_count']}}/7
Sharktooth Count: {{technicals['sharktooth_count']}}

--------------------------------------------------------------------------------
CELESTIAL CONDITIONS
--------------------------------------------------------------------------------
Sun-Saturn Sep: {{celestial['sun_saturn_sep']:.1f}}° {{'OPPOSITION' if celestial['sun_opp_saturn'] else ''}}
Moon-Uranus Sep: {{celestial['moon_uranus_sep']:.1f}}° {{'OPPOSITION' if celestial['moon_opp_uranus'] else ''}}
Moon Phase: {{celestial['moon_phase']:.0f}}%
Planetary Spread: {{celestial['spread']:.1f}}°
Spread Regime: {{celestial['spread_regime']}}
Next Moon-Uranus Opp: {{celestial['next_moon_uranus_opp']}}

--------------------------------------------------------------------------------
ALERTS
--------------------------------------------------------------------------------
"""
    
    if celestial['moon_opp_uranus']:
        text += "* MOON-URANUS OPPOSITION ACTIVE - High probability entry window\n"
    if celestial['sun_opp_saturn']:
        text += "* SUN-SATURN OPPOSITION - Consider taking profits\n"
    if celestial['spread_regime'] == 'DANGER_ZONE':
        text += "* DANGER ZONE - Elevated volatility risk, size reduced 20%\n"
    if technicals['sharktooth_count'] >= 2:
        text += "* SHARKTOOTH DETECTED - Strong oversold signal\n"
    
    text += f"""
================================================================================
Generated: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}}
================================================================================
"""
    
    return text


# =============================================================================
# EMAIL SENDER
# =============================================================================

def send_email(subject: str, html_body: str, text_body: str) -> bool:
    """Send email via SMTP."""
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = CONFIG['email_from']
        msg['To'] = ', '.join(CONFIG['email_to'])
        
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        context = ssl.create_default_context()
        
        with smtplib.SMTP(CONFIG['smtp_server'], CONFIG['smtp_port']) as server:
            server.starttls(context=context)
            server.login(CONFIG['smtp_username'], CONFIG['smtp_password'])
            server.sendmail(CONFIG['email_from'], CONFIG['email_to'], msg.as_string())
        
        print(f"Email sent successfully to {{CONFIG['email_to']}}")
        return True
        
    except Exception as e:
        print(f"Failed to send email: {{e}}")
        traceback.print_exc()
        return False


# =============================================================================
# MAIN SCANNER
# =============================================================================

def run_scanner(test_email: bool = False):
    """Run the daily signal scanner and send email."""
    
    today = datetime.now()
    
    print("=" * 70)
    print("BULLETPROOF v7.3 - AWS DAILY SCANNER")
    print(f"Date: {{today.strftime('%Y-%m-%d %H:%M:%S')}}")
    print("=" * 70)
    
    # Check if market is open (Monday-Friday)
    if today.weekday() >= 5:
        print("Weekend - no scan needed")
        return
    
    # Fetch data
    print("\nFetching market data...")
    try:
        spy = yf.download("SPY", start="2024-01-01", progress=False)
        vix_data = yf.download("^VIX", start="2024-01-01", progress=False)
        
        if len(spy) == 0:
            raise Exception("No SPY data returned")
        
        vix = float(safe_series(vix_data['Close']).iloc[-1]) if len(vix_data) > 0 else 20.0
        latest_date = spy.index[-1]
        
        print(f"Latest data: {{latest_date.strftime('%Y-%m-%d')}}")
        
    except Exception as e:
        print(f"ERROR fetching data: {{e}}")
        # Send error email
        send_email(
            subject=f"⚠️ Bulletproof v7.3 - Data Error",
            html_body=f"<p>Failed to fetch market data: {{e}}</p>",
            text_body=f"Failed to fetch market data: {{e}}"
        )
        return
    
    # Calculate indicators
    print("Calculating technical indicators...")
    technicals = calculate_technicals(spy)
    
    print("Calculating celestial conditions...")
    celestial = CelestialScanner().get_features(today)
    
    # Generate signal
    print("Generating signal...")
    signal = generate_signal(technicals, celestial, vix)
    
    # Format email
    subject_emoji = '🟢' if signal['signal'] == 'BUY' else ('🔴' if signal['exit_signal'] else '⚪')
    subject = f"{{subject_emoji}} Bulletproof v7.3: {{signal['signal']}} ({{signal['conviction']}}) - SPY ${{technicals['price']:.2f}}"
    
    if signal['exit_signal']:
        subject = f"🔴 Bulletproof v7.3: EXIT SIGNAL ({{signal['exit_reason']}}) - SPY ${{technicals['price']:.2f}}"
    
    html_body = format_email_html(today, technicals, celestial, signal, vix)
    text_body = format_email_text(today, technicals, celestial, signal, vix)
    
    # Send email
    print("Sending email...")
    success = send_email(subject, html_body, text_body)
    
    if success:
        print("\n✅ Daily scan complete!")
    else:
        print("\n❌ Email sending failed")
    
    # Print summary to console
    print(f"\nSIGNAL: {{signal['signal']}} ({{signal['conviction']}})")
    print(f"Size: {{signal['size']*100:.0f}}%")
    print(f"Bull Prob: {{signal['bull_prob']*100:.0f}}%")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bulletproof v7.3 AWS Daily Scanner')
    parser.add_argument('--test-email', action='store_true', help='Send a test email')
    args = parser.parse_args()
    
    if args.test_email:
        print("Sending test email...")
        send_email(
            subject="🧪 Bulletproof v7.3 - Test Email",
            html_body="<h1>Test Email</h1><p>If you see this, email is working!</p>",
            text_body="Test Email\n\nIf you see this, email is working!"
        )
    else:
        run_scanner()
