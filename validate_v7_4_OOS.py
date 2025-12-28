
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import ephem
import xgboost as xgb
from datetime import datetime, timedelta
import math
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CELESTIAL ENGINE (v7.4 - Vedic)
# =============================================================================
class CelestialEngine:
    def __init__(self):
        pass

    def get_features(self, date):
        """Calculate celestial features for a given date."""
        try:
            obs = ephem.Observer()
            obs.date = date
            
            bodies = {
                'Sun': ephem.Sun(),
                'Saturn': ephem.Saturn(),
                'Moon': ephem.Moon(),
                'Uranus': ephem.Uranus(),
                'Jupiter': ephem.Jupiter(),
                'Mercury': ephem.Mercury(),
                'Venus': ephem.Venus(),
                'Mars': ephem.Mars(),
                'Neptune': ephem.Neptune(),
                'Pluto': ephem.Pluto(),
            }
            
            for body in bodies.values():
                body.compute(obs)
            
            # Helper for separation
            def get_sep(b1, b2):
                l1 = math.degrees(ephem.Ecliptic(b1).lon)
                l2 = math.degrees(ephem.Ecliptic(b2).lon)
                diff = abs(l1 - l2)
                if diff > 180: diff = 360 - diff
                return diff
            
            # Helper for Sidereal Dignity (Saturn)
            def get_saturn_dignity(saturn_body):
                trop_lon = math.degrees(ephem.Ecliptic(saturn_body).lon)
                # Lahiri Ayanamsa approx: Tropical - 24.0
                sidereal_lon = (trop_lon - 24.0) % 360
                
                # Aries (Debilitated): 0-30
                if 0 <= sidereal_lon < 30:
                    return -1
                # Libra (Exalted): 180-210
                elif 180 <= sidereal_lon < 210:
                    return 1
                else:
                    return 0
            
            # Helper for Retrograde (Mercury)
            def is_retrograde(body):
                # Simple velocity check: lon now vs lon yesterday
                # For high precision we use full ephem logic but this is sufficient for daily resolution
                return 0 # Placeholder if velocity calc is complex, but let's do it right
                # Actually ephem doesn't give momentary velocity easily without diff
                # We will handle retrograde via pre-calc or simple boolean if possible
                # For this OOS script, let's stick to the Dignity core feature
                pass

            # Features
            features = {}
            features['CELEST_sun_saturn_sep'] = get_sep(bodies['Sun'], bodies['Saturn']) / 180.0
            features['CELEST_moon_uranus_sep'] = get_sep(bodies['Moon'], bodies['Uranus']) / 180.0
            features['CELEST_saturn_jupiter_sep'] = get_sep(bodies['Saturn'], bodies['Jupiter']) / 180.0
            features['CELEST_moon_phase'] = bodies['Moon'].phase / 100.0
            features['CELEST_saturn_dignity'] = get_saturn_dignity(bodies['Saturn'])
             
            # Planetary Spread
            lons = [math.degrees(ephem.Ecliptic(b).lon) for b in bodies.values() if b != bodies['Moon']]
            lons.sort()
            max_gap = 0
            for i in range(len(lons)):
                gap = (lons[(i+1)%len(lons)] - lons[i]) % 360
                max_gap = max(max_gap, gap)
            features['CELEST_spread'] = (360 - max_gap) / 360.0
            
            return features

        except Exception as e:
            return {}

# =============================================================================
# DATA PREP
# =============================================================================
def prepare_data():
    print("Downloading SPY Data (2000-2025)...")
    spy = yf.download("SPY", start="2000-01-01", end="2025-12-28", progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # Technicals
    print("Calculating Technicals...")
    spy['RSI2'] = ta.rsi(spy['Close'], length=2)
    spy['RSI14'] = ta.rsi(spy['Close'], length=14)
    spy['SMA200'] = ta.sma(spy['Close'], length=200)
    
    # BB %B
    bb = ta.bbands(spy['Close'], length=20, std=2)
    # Dynamically find columns
    lower_col = [c for c in bb.columns if 'BBL' in c][0]
    upper_col = [c for c in bb.columns if 'BBU' in c][0]
    spy['BB_pctb'] = (spy['Close'] - bb[lower_col]) / (bb[upper_col] - bb[lower_col])
    
    # Target: Next Day Return
    spy['Target'] = spy['Close'].pct_change().shift(-1)
    
    # Celestial
    print("Calculating Celestial Features (Slow)...")
    engine = CelestialEngine()
    
    celestial_data = []
    dates = spy.index
    for d in dates:
        feats = engine.get_features(d)
        celestial_data.append(feats)
    
    celestial_df = pd.DataFrame(celestial_data, index=spy.index)
    spy = pd.concat([spy, celestial_df], axis=1)
    
    spy.dropna(inplace=True)
    return spy

# =============================================================================
# VALIDATION
# =============================================================================
def run_validation():
    df = prepare_data()
    
    # Feature List (Match v7.4)
    features = [
        'RSI2', 'RSI14', 'BB_pctb',
        'CELEST_sun_saturn_sep',
        'CELEST_moon_uranus_sep',
        'CELEST_saturn_jupiter_sep',
        'CELEST_moon_phase',
        'CELEST_saturn_dignity',
        'CELEST_spread'
    ]
    
    # SPLIT
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    
    train_df = df[:train_end]
    test_df = df[test_start:]
    
    with open('oos_results_final.txt', 'w', encoding='utf-8') as f:
        f.write(f"--- DATA SPLIT ---\n")
        f.write(f"Train: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} days)\n")
        f.write(f"Test:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)\n\n")
        
        # TRAIN
        print("Training XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(train_df[features], train_df['Target'])
        
        # PREDICT
        target_col = 'Target' # Already defined
        
        train_df['Pred'] = model.predict(train_df[features])
        test_df['Pred'] = model.predict(test_df[features])
        
        # STRATEGY
        test_df['Strategy_Ret'] = np.where(test_df['Pred'] > 0.0005, test_df[target_col], 0)
        
        oos_ret = test_df['Strategy_Ret'].sum()
        oos_bh = test_df[target_col].sum()
        oos_sharpe = (test_df['Strategy_Ret'].mean() / test_df['Strategy_Ret'].std()) * np.sqrt(252)

        # DRAWDOWN
        test_df['Strategy_Cum'] = (1 + test_df['Strategy_Ret']).cumprod()
        test_df['BH_Cum'] = (1 + test_df[target_col]).cumprod()
        
        test_df['Strategy_DD'] = test_df['Strategy_Cum'] / test_df['Strategy_Cum'].cummax() - 1
        test_df['BH_DD'] = test_df['BH_Cum'] / test_df['BH_Cum'].cummax() - 1
        
        strat_max_dd = test_df['Strategy_DD'].min()
        bh_max_dd = test_df['BH_DD'].min()
        
        f.write(f"--- OOS RESULTS (2023-2025) ---\n")
        f.write(f"Strategy Total Return: {oos_ret*100:.2f}%\n")
        f.write(f"Buy & Hold Return:     {oos_bh*100:.2f}%\n")
        f.write(f"OOS Sharpe Ratio:      {oos_sharpe:.4f}\n")
        f.write(f"Strategy Max DD:       {strat_max_dd*100:.2f}%\n")
        f.write(f"Buy & Hold Max DD:     {bh_max_dd*100:.2f}%\n\n")
        
        # FEATURE IMPORTANCE
        f.write(f"--- FEATURE IMPORTANCE (Trained on 2000-2022) ---\n")
        fi = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        f.write(fi.to_string())
        f.write("\n\n")
        
        # SATURN CHECK
        saturn_imp = fi[fi['Feature'] == 'CELEST_saturn_dignity']['Importance'].values[0]
        f.write(f"Saturn Dignity Importance: {saturn_imp:.4f}\n")
        if saturn_imp > 0.005:
            f.write("VALIDATED: Saturn Dignity contributed.\n")
        else:
            f.write("WARNING: Saturn Dignity was ignored.\n")
            
    print("Done. Results saved to oos_results_final.txt")

if __name__ == "__main__":
    run_validation()
