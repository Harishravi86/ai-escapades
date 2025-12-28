
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
    
    # Handle multi-index columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate returns
    df['Return'] = df['Close'].pct_change()
    df['NextReturn'] = df['Return'].shift(-1)
    df.dropna(inplace=True)
    return df

def add_retrograde_feature(df):
    """Add is_mercury_retrograde to DataFrame."""
    print("Calculating Mercury Retrograde status (this may take a moment)...")
    engine = CelestialEngine()
    
    # Use list comprehension for speed, or just apply
    # Since ephem calculations are CPU bound, applying row by row
    retro_status = []
    dates = df.index
    
    for i, d in enumerate(dates):
        # engine.get_mercury_retrograde expects datetime/string/timestamp
        is_retro = engine.get_mercury_retrograde(d)
        if i < 5:
            print(f"Date: {d}, Retro: {is_retro}")
        retro_status.append(is_retro)
        
    df['Mercury_Retrograde'] = retro_status
    print(f"Retrograde Counts:\n{df['Mercury_Retrograde'].value_counts()}")
    return df

def study_1_statistical_impact(df):
    """Compare Market Returns: Retrograde vs Direct."""
    print("\n" + "="*50)
    print("STUDY 1: STATISTICAL IMPACT (Standalone)")
    print("="*50)
    
    retro = df[df['Mercury_Retrograde'] == True]
    direct = df[df['Mercury_Retrograde'] == False]
    
    n_total = len(df)
    n_retro = len(retro)
    perc_retro = (n_retro / n_total) * 100
    
    print(f"Total Days: {n_total}")
    print(f"Retrograde Days: {n_retro} ({perc_retro:.1f}%)")
    
    # Metrics
    avg_retro = retro['NextReturn'].mean() * 100
    avg_direct = direct['NextReturn'].mean() * 100
    
    vol_retro = retro['NextReturn'].std() * 100
    vol_direct = direct['NextReturn'].std() * 100
    
    win_retro = (len(retro[retro['NextReturn'] > 0]) / len(retro)) * 100
    win_direct = (len(direct[direct['NextReturn'] > 0]) / len(direct)) * 100
    
    print(f"\nMETRIC          | RETROGRADE | DIRECT   | DIFF")
    print(f"-"*50)
    print(f"Avg Daily Return| {avg_retro:+.4f}%  | {avg_direct:+.4f}% | {avg_retro-avg_direct:+.4f}%")
    print(f"Volatility (Std)| {vol_retro:.4f}%   | {vol_direct:.4f}%  | {vol_retro-vol_direct:+.4f}%")
    print(f"Win Rate        | {win_retro:.1f}%     | {win_direct:.1f}%    | {win_retro-win_direct:+.1f}%")
    
    # T-Test
    t_stat, p_val = stats.ttest_ind(retro['NextReturn'], direct['NextReturn'], equal_var=False)
    print(f"\nT-Test (Difference in Means):")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(NOT SIGNIFICANT)'}")

    return p_val < 0.05

def study_2_ml_importance(df):
    """Train XGBoost to see if Retrograde adds value."""
    print("\n" + "="*50)
    print("STUDY 2: ML FEATURE IMPORTANCE (Inside v7.3 Context)")
    print("="*50)
    
    # 1. Add Technical Features (Simplified v7.3 set)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['BB_PCTB'] = ta.bbands(df['Close'], length=20).iloc[:, 0] # Just grab one for simplicity test, usually %B is computed manually
    # Actually let's use ta.bbands properly to get %B
    bb = ta.bbands(df['Close'], length=20)
    df['BBL'] = bb[bb.columns[0]]
    df['BBU'] = bb[bb.columns[2]]
    df['BB_PCTB'] = (df['Close'] - df['BBL']) / (df['BBU'] - df['BBL'])
    
    df['Returns_5d'] = df['Close'].pct_change(5)
    
    df.dropna(inplace=True)
    
    # 2. Prepare Data
    features = [
        'RSI', 'BB_PCTB', 'Returns_5d', # Core Technicals
        'Mercury_Retrograde'            # The Candidate
    ]
    
    target = (df['NextReturn'] > 0).astype(int) # Predicting Direction (Bull/Bear)
    
    # Train/Test Split (Time Series)
    split = int(len(df) * 0.8)
    X_train = df[features].iloc[:split]
    y_train = target.iloc[:split]
    X_test = df[features].iloc[split:]
    y_test = target.iloc[split:]
    
    # 3. Train Model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model Accuracy (Train): {train_score:.1%}")
    print(f"Model Accuracy (Test):  {test_score:.1%}")
    
    # 5. Feature Importance
    importance = model.feature_importances_
    results = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(results)
    
    retro_imp = results[results['Feature'] == 'Mercury_Retrograde']['Importance'].values[0]
    print(f"\nMercury Retrograde Importance: {retro_imp:.1%}")
    
    if retro_imp > 0.01:
        print(">> CONCLUSION: The model FOUND value in this feature.")
    else:
        print(">> CONCLUSION: The model IGNORED this feature (Noise).")

def study_3_backtest_impact(df):
    """Compare Equity Curves: Baseline vs With-Retrograde."""
    print("\n" + "="*50)
    print("STUDY 3: BACKTEST PERFORMANCE (PnL Impact)")
    print("="*50)
    
    # Setup Data
    # Baseline Features
    feats_base = ['RSI', 'BB_PCTB', 'Returns_5d']
    # Enhanced Features
    feats_astro = feats_base + ['Mercury_Retrograde']
    
    target = (df['NextReturn'] > 0).astype(int)
    split = int(len(df) * 0.8)
    
    # Train Baseline
    X_train_b = df[feats_base].iloc[:split]
    y_train = target.iloc[:split]
    X_test_b = df[feats_base].iloc[split:]
    y_test = target.iloc[split:]
    test_returns = df['NextReturn'].iloc[split:]
    
    model_b = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
    model_b.fit(X_train_b, y_train)
    preds_b = model_b.predict(X_test_b)
    
    # Train Astro
    X_train_a = df[feats_astro].iloc[:split]
    X_test_a = df[feats_astro].iloc[split:]
    
    model_a = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
    model_a.fit(X_train_a, y_train)
    preds_a = model_a.predict(X_test_a)
    
    # Calculate Returns
    # Strategy: If Pred=1, Buy and hold for 1 day. If 0, Cash (0 return).
    daily_ret_b = preds_b * test_returns
    daily_ret_a = preds_a * test_returns
    
    # Cumulative Return
    cum_ret_b = (1 + daily_ret_b).cumprod().iloc[-1] - 1
    cum_ret_a = (1 + daily_ret_a).cumprod().iloc[-1] - 1
    
    # Sharpe (Assume risk free = 0 for simplicity in comparison)
    sharpe_b = (daily_ret_b.mean() / daily_ret_b.std()) * np.sqrt(252) if daily_ret_b.std() > 0 else 0
    sharpe_a = (daily_ret_a.mean() / daily_ret_a.std()) * np.sqrt(252) if daily_ret_a.std() > 0 else 0
    
    print(f"METRIC          | BASELINE   | +MERCURY   | DIFF")
    print(f"-"*50)
    print(f"Total Return    | {cum_ret_b*100:6.1f}%    | {cum_ret_a*100:6.1f}%    | {(cum_ret_a-cum_ret_b)*100:+.1f}%")
    print(f"Sharpe Ratio    | {sharpe_b:6.2f}     | {sharpe_a:6.2f}     | {sharpe_a-sharpe_b:+.2f}")
    
    if cum_ret_a > cum_ret_b:
        print("\n>> RESULT: Adding Mercury Retrograde IMPROVED performance.")
    else:
        print("\n>> RESULT: Adding Mercury Retrograde HURT performance (Overfitting?).")

if __name__ == "__main__":
    # 1. Load Data
    df = get_data()
    
    # 2. Add Feature
    df = add_retrograde_feature(df)
    
    # 3. Run Studies
    try:
        study_1_statistical_impact(df)
        study_2_ml_importance(df)
        study_3_backtest_impact(df)
    except Exception as e:
        print(f"An error occurred: {e}")
