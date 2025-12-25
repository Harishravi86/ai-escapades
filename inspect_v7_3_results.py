
import pandas as pd
import joblib
import os

def inspect_results():
    print("--- v7.3 ANALYTICS INSPECTION ---")
    
    # 1. Inspect Feature Importance (SPY Model)
    model_path = 'bull_v73_astro.joblib'
    if os.path.exists(model_path):
        print(f"\n[ML Model] Loading {model_path}...")
        data = joblib.load(model_path)
        imp = data.get('feature_importance')
        
        if imp is not None:
            print(f"Top 10 Features:")
            print(imp.head(10))
            
            print("\nCelestial Feature Ranks:")
            celest = [f for f in imp.index if 'CELEST' in f]
            for f in celest:
                rank = list(imp.index).index(f) + 1
                score = imp[f]
                print(f"  #{rank}: {f:<25} (Imp: {score:.4f})")
                
            total_celest = imp[celest].sum()
            print(f"  > Total Celestial Importance: {total_celest:.2%}")
        else:
            print("  No feature importance found in model file.")
    else:
        print(f"  Model file {model_path} not found.")

    # 2. Inspect Trades (Danger Zone & Overrides)
    trade_path = 'v7_3_trades_spy.csv'
    if os.path.exists(trade_path):
        print(f"\n[Trade Log] Loading {trade_path}...")
        trades = pd.read_csv(trade_path)
        
        # Check column names
        cols = [c.lower() for c in trades.columns]
        trades.columns = cols
        
        print(f"Total Trades: {len(trades)}")
        
        # Danger Zone Reductions (Size < 1.0)
        # Note: In v7.3 code, Size is recorded.
        if 'size' in trades.columns:
            reductions = trades[trades['size'] < 0.99]
            print(f"Danger Zone Reductions: {len(reductions)}")
            if len(reductions) > 0:
                print("  Sample:")
                print(reductions[['entry_date', 'size']].head(3))
        else:
            print("  'size' column missing.")

        # Exit Overrides
        if 'override_applied' in trades.columns:
            overrides = trades[trades['override_applied'] == True]
            print(f"Exit Overrides Applied: {len(overrides)}")
            if len(overrides) > 0:
                print("  Sample:")
                print(overrides[['exit_date', 'entry_price']].head(3))
        else:
            print("  'override_applied' column missing.")
            
    else:
        print(f"  Trade file {trade_path} not found.")

if __name__ == "__main__":
    inspect_results()
