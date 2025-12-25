
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import ephem
from datetime import datetime

# --- CONFIGURATION ---
TARGET_LABEL = 'Crash_Risk' # Predict if market drops > X% in next 20 days
DROP_THRESHOLD = -0.05      # 5% drop

# Ephemeris setup
planets = {
    'Sun': ephem.Sun(), 'Moon': ephem.Moon(), 'Mercury': ephem.Mercury(),
    'Venus': ephem.Venus(), 'Mars': ephem.Mars(), 'Jupiter': ephem.Jupiter(),
    'Saturn': ephem.Saturn(), 'Uranus': ephem.Uranus(), 'Neptune': ephem.Neptune(),
    'Pluto': ephem.Pluto()
}

def get_planet_positions(date_obj):
    obs = ephem.Observer()
    obs.date = date_obj.strftime('%Y/%m/%d')
    longitudes = []
    for name, planet in planets.items():
        planet.compute(obs)
        longitudes.append(np.degrees(planet.hlon) % 360)
    return sorted(longitudes)

def calculate_spread(longitudes):
    max_gap = 0
    n = len(longitudes)
    for i in range(n):
        gap = longitudes[(i + 1) % n] - longitudes[i]
        if gap < 0: gap += 360
        max_gap = max(max_gap, gap)
    return 360 - max_gap

def prepare_data():
    print("Fetching market data (1993-2024)...")
    spy = yf.download('SPY', start='1993-01-01', end='2025-01-01', progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    
    df = pd.DataFrame(index=spy.index)
    df['Close'] = spy['Close']
    
    # 1. Technical Baseline Features (The "Control" Group)
    # We use basic momentum/volatility to see if Astrology adds *new* info
    df['Returns_5d'] = df['Close'].pct_change(5)
    df['Returns_20d'] = df['Close'].pct_change(20)
    df['Vol_20d'] = df['Close'].rolling(20).std() / df['Close']
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / 
                              df['Close'].diff().apply(lambda x: abs(min(x, 0))).rolling(14).mean()))

    # 2. Astro Feature (The "Test" Variable)
    print("Calculating Planetary Spreads...")
    spreads = []
    dates = df.index.to_pydatetime()
    for d in dates:
        lons = get_planet_positions(d)
        spreads.append(calculate_spread(lons))
    df['Planetary_Spread'] = spreads
    
    # 3. Target (Crash prediction)
    # Did market drop > 5% in next 20 days?
    df['Future_Return'] = df['Close'].pct_change(20).shift(-20)
    df['Target'] = (df['Future_Return'] < DROP_THRESHOLD).astype(int)
    
    df.dropna(inplace=True)
    return df

def train_and_evaluate():
    df = prepare_data()
    
    features_baseline = ['Returns_5d', 'Returns_20d', 'Vol_20d', 'RSI']
    features_astro = features_baseline + ['Planetary_Spread']
    
    X = df[features_astro]
    y = df['Target']
    
    # Time Series Split Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    scores_baseline = []
    scores_astro = []
    
    print("\n--- TRAINING XGBOOST MODELS ---")
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 1. Train Baseline
        model_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model_base.fit(X_train[features_baseline], y_train)
        probs_base = model_base.predict_proba(X_test[features_baseline])[:, 1]
        auc_base = roc_auc_score(y_test, probs_base)
        scores_baseline.append(auc_base)
        
        # 2. Train Astro
        model_astro = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model_astro.fit(X_train[features_astro], y_train)
        probs_astro = model_astro.predict_proba(X_test[features_astro])[:, 1]
        auc_astro = roc_auc_score(y_test, probs_astro)
        scores_astro.append(auc_astro)
        
        print(f"Fold {fold}: Baseline AUC: {auc_base:.4f} | Astro AUC: {auc_astro:.4f} | Diff: {auc_astro-auc_base:+.4f}")
        fold += 1
        
    avg_base = np.mean(scores_baseline)
    avg_astro = np.mean(scores_astro)
    
    print("\n--- RESULTS ---")
    print(f"Average Baseline AUC: {avg_base:.4f}")
    print(f"Average Astro AUC:    {avg_astro:.4f}")
    print(f"Improvement:          {(avg_astro - avg_base)*100:.2f}%")
    
    # Feature Importance of final model
    print("\n--- FEATURE IMPORTANCE (Final Fold) ---")
    imp = pd.Series(model_astro.feature_importances_, index=features_astro).sort_values(ascending=False)
    print(imp)

if __name__ == "__main__":
    train_and_evaluate()
