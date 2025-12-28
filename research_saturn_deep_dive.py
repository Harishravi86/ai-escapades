
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import xgboost as xgb
from datetime import datetime, timedelta
from celestial_engine import CelestialEngine
from scipy import stats

def get_data():
    """Download SPY data."""
    print("Downloading SPY data...")
    df = yf.download("SPY", start="2000-01-01", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['Return'] = df['Close'].pct_change()
    df['NextReturn'] = df['Return'].shift(-1)
    df.dropna(inplace=True)
    return df

def add_saturn_features(df):
    """Add Saturn Research Features."""
    print("Calculating Saturn Features (this may take a minute)...")
    engine = CelestialEngine()
    
    saturn_retro = []
    saturn_mars = []
    saturn_jupiter = []
    saturn_sign = []
    
    dates = df.index
    
    for i, d in enumerate(dates):
        if i % 1000 == 0: print(f"Processing row {i}/{len(df)}...")
        
        # We can use the get_features dict or individual methods
        # Using methods directly is faster if we only need specific ones
        # But get_features caches, so let's stick to methods for clarity here
        
        saturn_retro.append(engine.get_saturn_retrograde(d))
        saturn_mars.append(engine.get_dual_separation(d, 'Saturn', 'Mars'))
        saturn_jupiter.append(engine.get_dual_separation(d, 'Saturn', 'Jupiter'))
        saturn_sign.append(engine.get_planet_sign(d, 'Saturn'))
        
    df['Saturn_Retrograde'] = saturn_retro
    df['Saturn_Mars_Sep'] = saturn_mars
    df['Saturn_Jupiter_Sep'] = saturn_jupiter
    df['Saturn_Sign'] = saturn_sign
    
    return df

def check_aries_libra(df):
    """Deep Dive: Saturn Sign Impact."""
    print("\n" + "="*50)
    print("STUDY 1: SATURN SIGNS (Aries vs Libra)")
    print("="*50)
    
    aries = df[df['Saturn_Sign'] == 0] # 0 = Aries
    libra = df[df['Saturn_Sign'] == 6] # 6 = Libra
    
    print(f"Saturn in Aries (Debilitated) Days: {len(aries)}")
    print(f"Saturn in Libra (Exalted) Days:     {len(libra)}")
    
    avg_aries = aries['NextReturn'].mean() * 100
    avg_libra = libra['NextReturn'].mean() * 100
    
    print(f"Avg Return (Aries): {avg_aries:+.4f}%")
    print(f"Avg Return (Libra): {avg_libra:+.4f}%")
    
    t_stat, p_val = stats.ttest_ind(aries['NextReturn'], libra['NextReturn'], equal_var=False)
    print(f"T-Test Difference: P={p_val:.4f}")

def check_ml_importance(df):
    """XGBoost Feature Importance."""
    print("\n" + "="*50)
    print("STUDY 2: ML FEATURE IMPORTANCE")
    print("="*50)
    
    # 1. Technicals Baseline
    df['RSI'] = ta.rsi(df['Close'], length=14)
    bb = ta.bbands(df['Close'], length=20)
    df['BB_PCTB'] = (df['Close'] - bb[bb.columns[0]]) / (bb[bb.columns[2]] - bb[bb.columns[0]])
    df['Returns_5d'] = df['Close'].pct_change(5)
    
    df.dropna(inplace=True)
    
    feature_cols = [
        'RSI', 'BB_PCTB', 'Returns_5d',
        'Saturn_Retrograde',
        'Saturn_Mars_Sep',
        'Saturn_Jupiter_Sep',
        'Saturn_Sign'
    ]
    
    target = (df['NextReturn'] > 0).astype(int)
    split = int(len(df) * 0.8)
    
    X_train = df[feature_cols].iloc[:split]
    y_train = target.iloc[:split]
    X_test = df[feature_cols].iloc[split:]
    y_test = target.iloc[split:]
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    
    importance = model.feature_importances_
    results = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values('Importance', ascending=False)
    
    print(results)
    
    # Simple Backtest of "Enhanced" Model
    preds = model.predict(X_test)
    daily_ret = preds * df['NextReturn'].iloc[split:]
    cum_ret = (1 + daily_ret).cumprod().iloc[-1] - 1
    
    # Baseline Model (No Saturn)
    base_cols = ['RSI', 'BB_PCTB', 'Returns_5d']
    model_b = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
    model_b.fit(X_train[base_cols], y_train)
    preds_b = model_b.predict(X_test[base_cols])
    daily_ret_b = preds_b * df['NextReturn'].iloc[split:]
    cum_ret_b = (1 + daily_ret_b).cumprod().iloc[-1] - 1
    
    print(f"\nBACKTEST RESULT (Total Return):")
    print(f"Baseline: {cum_ret_b*100:.1f}%")
    print(f"Saturn Enhanced: {cum_ret*100:.1f}%")
    print(f"Diff: {(cum_ret - cum_ret_b)*100:+.1f}%")

if __name__ == "__main__":
    df = get_data()
    df = add_saturn_features(df)
    check_aries_libra(df)
    check_ml_importance(df)
