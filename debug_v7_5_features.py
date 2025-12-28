
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bulletproof_strategy_v7_5_eclipse import CelestialEngine, TechnicalEngine

def test_features():
    print("Initializing CelestialEngine...")
    eng = CelestialEngine()
    
    # Test specific dates
    dates = [
        '2024-04-08', # Solar Eclipse
        '2024-03-25', # Lunar Eclipse
        '2024-01-01'  # Nothing
    ]
    
    print("\nChecking Individual Dates:")
    for d in dates:
        feat = eng.get_features(d)
        print(f"{d}: Eclipse Regime = {feat.get('eclipse_regime', 'MISSING')}")

    # Test Batch Precompute
    print("\nChecking Batch Precompute:")
    # Create 60 days of data around eclipse
    start_date = datetime(2024, 3, 1)
    date_list = [start_date + timedelta(days=x) for x in range(60)]
    dt_index = pd.DatetimeIndex(date_list)
    
    celestial_df = eng.precompute_for_dates(dt_index)
    
    # Check if we have regime hits in the DF
    hits = celestial_df[celestial_df['CELEST_eclipse_regime'] != 0]
    print(f"\nFound {len(hits)} days with non-zero eclipse regime in batch.")
    if len(hits) > 0:
        print(hits[['CELEST_eclipse_regime']])
    
    # Test Technical Merge
    print("\nChecking Technical Merge:")
    # Create dummy price data for 60 days
    price_data = pd.DataFrame({
        'Close': np.random.normal(100, 5, 60),
        'High': np.random.normal(105, 5, 60),
        'Low': np.random.normal(95, 5, 60),
        'Volume': np.random.randint(1000, 5000, 60)
    }, index=dt_index)
    
    # Ensure some data cleaning like in strategy
    # Calculate requires pandas series behavior
    
    try:
        merged = TechnicalEngine.calculate(price_data, celestial_df)
        if 'CELEST_eclipse_regime' in merged.columns:
            print("Success! CELEST_eclipse_regime found in merged features.")
            print(merged['CELEST_eclipse_regime'].value_counts())
        else:
            print("FAILURE! CELEST_eclipse_regime NOT found in merged features.")
    except Exception as e:
        print(f"Merge crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_features()
