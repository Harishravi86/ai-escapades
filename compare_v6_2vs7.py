"""
================================================================================
A/B COMPARISON: v6.2 vs v7.2
================================================================================
Proper out-of-sample testing to determine if v7.2 is actually better.

Key methodology:
1. Train both on SAME period (2000-2019)
2. Test on HOLDOUT (2020-2025)  
3. Compare metrics that matter: Sharpe, MaxDD, not just CAGR

Run: python compare_v62_v72.py
================================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import both strategy versions
# Assuming both files are in the same directory
try:
    from bulletproof_strategy_v6_final import BulletproofStrategyV6, TechnicalEngine as TE62
    from bulletproof_strategy_v7_2_ml import BulletproofStrategyV72, TechnicalEngine as TE72
except ImportError as e:
    print(f"Error importing strategies: {e}")
    # Fallback/Debug note if files are named differently
    print("Please ensure bulletproof_strategy_v6_final.py and bulletproof_strategy_v7_2_ml.py are in the same directory.")
    exit(1)


def load_data():
    """Load SPY, QQQ, VIX data"""
    print("Loading market data...")
    spy = yf.download("SPY", start="2000-01-01", progress=False)
    qqq = yf.download("QQQ", start="2000-01-01", progress=False)
    vix = yf.download("^VIX", start="2000-01-01", progress=False)
    
    # Align indices
    idx = spy.index.intersection(qqq.index).intersection(vix.index)
    spy = spy.loc[idx]
    qqq = qqq.loc[idx]
    vix = vix.loc[idx]
    
    print(f"Loaded {len(spy)} trading days")
    return spy, qqq, vix


def split_data(spy, qqq, vix, train_end='2019-12-31'):
    """Split into train/test periods"""
    train_end_dt = pd.Timestamp(train_end)
    
    spy_train = spy[spy.index <= train_end_dt]
    spy_test = spy[spy.index > train_end_dt]
    
    qqq_train = qqq[qqq.index <= train_end_dt]
    qqq_test = qqq[qqq.index > train_end_dt]
    
    vix_train = vix[vix.index <= train_end_dt]
    vix_test = vix[vix.index > train_end_dt]
    
    print(f"\nTrain period: {spy_train.index[0].date()} to {spy_train.index[-1].date()} ({len(spy_train)} days)")
    print(f"Test period:  {spy_test.index[0].date()} to {spy_test.index[-1].date()} ({len(spy_test)} days)")
    
    return (spy_train, qqq_train, vix_train), (spy_test, qqq_test, vix_test)


def run_strategy_v62(train_data, test_data, verbose=False):
    """Run v6.2 strategy with train/test split"""
    spy_train, qqq_train, vix_train = train_data
    spy_test, qqq_test, vix_test = test_data
    
    # Initialize with TRAIN data (so train_models uses the correct data)
    strategy = BulletproofStrategyV6(spy_train, qqq_train, vix_train)
    
    # Train on training data
    strategy.train_models(verbose=verbose)
    
    # Backtest on test data
    strategy.signal_data = spy_test
    strategy.trade_data = qqq_test
    strategy.vix_data = vix_test
    
    cash, stats = strategy.backtest(verbose=verbose)
    return stats


def run_strategy_v72(train_data, test_data, verbose=False):
    """Run v7.2 strategy with train/test split"""
    spy_train, qqq_train, vix_train = train_data
    spy_test, qqq_test, vix_test = test_data
    
    strategy = BulletproofStrategyV72(spy_train, qqq_train, vix_train)
    
    # Train on training data
    strategy.train_models(verbose=verbose)
    
    # Get feature importance
    bull_importance = strategy.bull_detector.feature_importance
    
    # Backtest on test data
    strategy.signal_data = spy_test
    strategy.trade_data = qqq_test
    strategy.vix_data = vix_test
    
    cash, stats = strategy.backtest(verbose=verbose)
    
    return stats, bull_importance


def compare_results(v62_stats, v72_stats, v72_importance):
    """Print comparison table and save to markdown"""
    output = []
    
    def log(msg=""):
        print(msg)
        output.append(msg)
        
    log("\n" + "=" * 70)
    log("A/B COMPARISON: v6.2 vs v7.2 (OUT-OF-SAMPLE TEST)")
    log("=" * 70)
    
    metrics = [
        ('Total Return', 'total_return', '%', 100),
        ('CAGR', 'cagr', '%', 100),
        ('Max Drawdown', 'max_drawdown', '%', 100),
        ('Sharpe Ratio', 'sharpe', '', 1),
        ('Win Rate', 'win_rate', '%', 100),
        ('Total Trades', 'total_trades', '', 1),
    ]
    
    log(f"\n{'Metric':<20} | {'v6.2':>15} | {'v7.2':>15} | {'Diff':>12} | {'Winner':>8}")
    log("-" * 75)
    
    v62_score = 0
    v72_score = 0
    
    for name, key, suffix, mult in metrics:
        v62_val = v62_stats.get(key, 0) * mult
        v72_val = v72_stats.get(key, 0) * mult
        
        if key == 'max_drawdown':
            # Lower is better for drawdown
            delta = v62_val - v72_val
            winner = "v7.2" if v72_val < v62_val else "v6.2"
            if v72_val < v62_val:
                v72_score += 1
            else:
                v62_score += 1
        else:
            # Higher is better for other metrics
            delta = v72_val - v62_val
            winner = "v7.2" if v72_val > v62_val else "v6.2"
            if v72_val > v62_val:
                v72_score += 1
            else:
                v62_score += 1
        
        delta_str = f"{delta:+.2f}{suffix}"
        log(f"{name:<20} | {v62_val:>14.2f}{suffix} | {v72_val:>14.2f}{suffix} | {delta_str:>12} | {winner:>8}")
    
    log("-" * 75)
    log(f"\nOVERALL WINNER: {'v7.2' if v72_score > v62_score else 'v6.2'} ({max(v62_score, v72_score)}/{len(metrics)} metrics)")
    
    # Risk-adjusted comparison
    log("\n" + "=" * 70)
    log("RISK-ADJUSTED ANALYSIS")
    log("=" * 70)
    
    v62_calmar = v62_stats['cagr'] / v62_stats['max_drawdown'] if v62_stats['max_drawdown'] > 0 else 0
    v72_calmar = v72_stats['cagr'] / v72_stats['max_drawdown'] if v72_stats['max_drawdown'] > 0 else 0
    
    log(f"Calmar Ratio (CAGR/MaxDD):")
    log(f"  v6.2: {v62_calmar:.2f}")
    log(f"  v7.2: {v72_calmar:.2f}")
    log(f"  Winner: {'v7.2' if v72_calmar > v62_calmar else 'v6.2'}")
    
    # Feature importance analysis
    log("\n" + "=" * 70)
    log("v7.2 FEATURE IMPORTANCE (Did Pine features help?)")
    log("=" * 70)
    
    if v72_importance is not None:
        pine_features = ['BB_20_crossunder', 'BB_50_crossunder', 'BB_20_crossover', 
                        'BB_50_crossover', 'DAILY_RETURN_PANIC', 'DAILY_RETURN_CRASH',
                        'PINE_ENTRY_SIGNAL', 'MULTI_CROSSUNDER', 'DAILY_RETURN_EXTREME',
                        'DAILY_RETURN_SURGE']
        
        total_importance = v72_importance.sum()
        pine_importance = sum(v72_importance.get(f, 0) for f in pine_features)
        pine_pct = pine_importance / total_importance * 100 if total_importance > 0 else 0
        
        log(f"\nPine Script features contribution: {pine_pct:.1f}% of total importance")
        
        log("\nPine feature rankings:")
        for feat in pine_features:
            if feat in v72_importance.index:
                rank = list(v72_importance.index).index(feat) + 1
                imp = v72_importance[feat]
                log(f"  #{rank:2d}: {feat} ({imp:.4f})")
        
        if pine_pct < 5:
            log("\nWARNING: Pine features contribute <5% importance.")
            log("   The ML model is not finding them useful.")
            log("   v7.2 may be adding noise, not signal.")
        elif pine_pct < 15:
            log("\nNOTE: Pine features contribute moderately.")
            log("   They're adding some value but aren't dominant.")
        else:
            log("\nSUCCESS: Pine features are significant!")
            log("   The crossover detection is helping the model.")
    
    # Final verdict
    log("\n" + "=" * 70)
    log("VERDICT")
    log("=" * 70)
    
    sharpe_delta = v72_stats['sharpe'] - v62_stats['sharpe']
    dd_delta = v62_stats['max_drawdown'] - v72_stats['max_drawdown']
    
    if sharpe_delta > 0.1 and dd_delta > 0:
        log("v7.2 is BETTER: Higher Sharpe AND lower drawdown")
        log("  -> Upgrade to v7.2")
    elif sharpe_delta > 0.1:
        log("v7.2 has better Sharpe but similar/worse drawdown")
        log("  -> Consider v7.2 if you prioritize risk-adjusted returns")
    elif dd_delta > 0.02:  # 2% lower drawdown
        log("v7.2 has lower drawdown but similar Sharpe")
        log("  -> Consider v7.2 if you prioritize capital preservation")
    elif abs(sharpe_delta) < 0.05 and abs(dd_delta) < 0.01:
        log("No significant difference between v6.2 and v7.2")
        log("  -> Stick with v6.2 (simpler is better)")
    else:
        log("v6.2 appears BETTER on out-of-sample data")
        log("  -> v7.2 may be overfitting to training data")
        log("  -> Stick with v6.2")

    # Save to file
    with open("comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    print("\nReport saved to comparison_report.md")


def main():
    print("=" * 70)
    print("BULLETPROOF STRATEGY: v6.2 vs v7.2 COMPARISON")
    print("Out-of-Sample Testing")
    print("=" * 70)
    
    # Load data
    spy, qqq, vix = load_data()
    
    # Split into train/test
    train_data, test_data = split_data(spy, qqq, vix, train_end='2019-12-31')
    
    # Run v6.2
    print("\n" + "-" * 70)
    print("Running v6.2...")
    print("-" * 70)
    v62_stats = run_strategy_v62(train_data, test_data, verbose=False)
    print(f"v6.2 Test Return: {v62_stats['total_return']*100:.2f}%")
    
    # Run v7.2
    print("\n" + "-" * 70)
    print("Running v7.2...")
    print("-" * 70)
    v72_stats, v72_importance = run_strategy_v72(train_data, test_data, verbose=False)
    print(f"v7.2 Test Return: {v72_stats['total_return']*100:.2f}%")
    
    # Compare
    compare_results(v62_stats, v72_stats, v72_importance)


if __name__ == "__main__":
    main()