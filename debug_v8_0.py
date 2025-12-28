
import pandas as pd
import pandas_ta as ta
import numpy as np
from bulletproof_strategy_v8_0_regime import BulletproofStrategyV80_Regime, TechnicalEngine

def debug():
    print("Initializing Strategy...")
    s = BulletproofStrategyV80_Regime(ticker='QQQ')
    
    print("Loading Data...")
    s.load_data()
    
    print(f"Data Loaded: {len(s.df)} rows")
    
    # Check Features
    print("Calculating Features...")
    features = TechnicalEngine.calculate(s.df, s.celestial_features, s.regime_features)
    print(features.iloc[-1])
    
    # 1. Test Bull Model
    print("\n--- Testing Bull Model Training ---")
    try:
        s.bull_model.train(s.df, s.celestial_features, s.regime_features)
        print("Bull Model Trained OK.")
    except Exception as e:
        print(f"Bull Model Failed: {e}")

    # 2. Test Bear Model
    print("\n--- Testing Bear Model Training ---")
    try:
        detector = s.bear_model
        close = s.df['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        labels = detector._label_turning_points(close)
        print(f"  Bear Labels: {labels.sum()}")
        
        s.bear_model.train(s.df, s.celestial_features, s.regime_features)
        print("Bear Model Trained OK.")
    except Exception as e:
        print(f"Bear Model Failed: {e}")
        import traceback
        traceback.print_exc()
        
    # 3. Test Backtest
    print("\n--- Testing Backtest Loop ---")
    try:
        s.backtest()
        print("Backtest Finished OK.")
    except Exception as e:
        print(f"Backtest Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
