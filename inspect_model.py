import joblib
import pandas as pd
import os
import sys

def analyze_model(model_path, model_name):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print(f"\nANALYZING MODEL: {model_name}")
    print("-" * 50)
    
    try:
        data = joblib.load(model_path)
        # Check if feature importance is stored
        if 'feature_importance' in data and data['feature_importance'] is not None:
            imp = data['feature_importance']
            print("Top 20 Features:")
            for i, (feat, val) in enumerate(imp.head(20).items()):
                marker = " [ASTRO]" if 'norm' in feat or 'moon' in feat or 'sun' in feat else ""
                print(f"{i+1:2d}. {feat:<30} {val:.5f}{marker}")
        else:
            print("No feature importance found in model file.")
            
    except Exception as e:
        print(f"Error loading model: {e}")

def analyze_trades(csv_path):
    if not os.path.exists(csv_path):
        print(f"\nTrade file not found: {csv_path}")
        return

    print(f"\nANALYZING TRADES: {csv_path}")
    print("-" * 50)
    try:
        df = pd.read_csv(csv_path)
        print(f"Total Trades: {len(df)}")
        
        if 'return' in df.columns:
            total_ret = ((df['return'] + 1).prod()) - 1
            print(f"Total Return (Compounded): {total_ret*100:.2f}%")
            
            # Simple winrate
            wins = df[df['return'] > 0]
            win_rate = len(wins) / len(df)
            print(f"Win Rate: {win_rate*100:.1f}%")
            
            # Avg return
            avg_ret = df['return'].mean()
            print(f"Avg Return per Trade: {avg_ret*100:.2f}%")
            
    except Exception as e:
        print(f"Error analyzing trades: {e}")

if __name__ == "__main__":
    # Check current directory for models
    current_dir = "."
    analyze_model(os.path.join(current_dir, "bull_v72_ml.joblib"), "BULL DETECTOR (v7.3)")
    analyze_model(os.path.join(current_dir, "bear_v72_ml.joblib"), "BEAR DETECTOR (v7.3)")
    
    analyze_trades(os.path.join(current_dir, "v7_2_trades.csv"))
