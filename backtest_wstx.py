import pandas as pd
import numpy as np
from wstx_coordinator import (
    WSTXCoordinator, TechnicalEngineV62, TrendEngine, 
    VolatilityEngine, ReversalEngine, MacroEngine, PositionSizingEngine,
    CelestialEngine, load_yf, safe_series
)

def backtest_wstx():
    print("Loading data...")
    spy = load_yf("SPY")
    qqq = load_yf("QQQ")
    vix = load_yf("^VIX")
    
    # Align
    idx = spy.index.intersection(qqq.index).intersection(vix.index)
    spy = spy.loc[idx]
    qqq = qqq.loc[idx]
    vix = vix.loc[idx]
    
    print(f"Data loaded: {len(spy)} days")
    
    coord = WSTXCoordinator()
    cel_engine = CelestialEngine() # Initialize Celestial Engine
    print("Training models...")
    coord.train_if_needed(spy)
    
    # Pre-calculate features/probs for speed
    print("Generating signals...")
    bull_probs = coord.bull_detector.predict(spy)
    bear_probs = coord.bear_detector.predict(spy)
    feats = TechnicalEngineV62.calculate(spy)
    
    # Backtest Loop
    cash = 100000.0
    shares = 0.0
    cooldown = 0
    equity_curve = []
    
    close_spy = safe_series(spy['Close'])
    close_qqq = safe_series(qqq['Close'])
    close_vix = safe_series(vix['Close'])
    
    # Reversal
    bull_shark_count = feats['BULL_SHARKTOOTH_COUNT']
    
    print("Running backtest loop...")
    for i in range(50, len(spy)):
        date = spy.index[i]
        date_str = str(date.date())
        price = float(close_qqq.iloc[i])
        
        # Celestial Features
        cel = cel_engine.get_features(date_str)
        
        # v6.2 Logic (Replicated from WST-X Coordinator)
        b_shark = bull_shark_count.iloc[i]
        bull_prob = float(bull_probs.iloc[i])
        bear_prob = float(bear_probs.iloc[i])
        bear_shark = feats['BEAR_SHARKTOOTH_COUNT'].iloc[i]
        
        # v6.2 Thresholds
        HIGH_CONVICTION_PROB = 0.70
        HIGH_CONVICTION_COUNT = 4
        MEDIUM_CONVICTION_PROB = 0.50
        MEDIUM_CONVICTION_COUNT = 3
        
        BEAR_THRESHOLD = 0.60
        BEAR_SHARK_COUNT = 3
        TRAILING_STOP = 0.08
        BASE_STOP = 0.12
        PROFIT_TAKE_THRESH = 0.20
        PROFIT_TAKE_PCT = 0.25
        PROFIT_TAKE_BEAR_MIN = 0.30
        
        # Cooldown Logic (Crucial for v6.2 parity)
        if cooldown > 0:
            cooldown -= 1
            equity = cash + shares * price
            equity_curve.append(equity)
            continue

        # Entry Logic
        if shares == 0:
            size_pct = 0.0
            if bull_prob > HIGH_CONVICTION_PROB or b_shark >= HIGH_CONVICTION_COUNT:
                size_pct = 1.0
            elif bull_prob > MEDIUM_CONVICTION_PROB or b_shark >= MEDIUM_CONVICTION_COUNT:
                size_pct = 0.5
            
            # Celestial Boost (Moon-Uranus)
            if size_pct > 0 and cel['moon_opp_uranus']:
                size_pct = min(size_pct * 1.25, 1.0)
            
            if size_pct > 0:
                invest = cash * size_pct
                shares = invest / price
                cash -= invest
                entry_price = price
                max_price = price
                partial_exit_taken = False
        
        # Exit/Manage Logic
        elif shares > 0:
            max_price = max(max_price, price)
            unrealized = (price - entry_price) / entry_price
            dd_from_high = (max_price - price) / max_price
            
            # Profit Taking
            if not partial_exit_taken and unrealized > PROFIT_TAKE_THRESH:
                # Strict v6.2: Bear prob > 5-day average
                bear_avg_5d = bear_probs.iloc[i-5:i].mean() if i > 5 else 0
                if bear_prob > bear_avg_5d and bear_prob > PROFIT_TAKE_BEAR_MIN:
                    sell_shares = shares * PROFIT_TAKE_PCT
                    cash += sell_shares * price
                    shares -= sell_shares
                    partial_exit_taken = True
            
            # Exits
            exit_signal = False
            if bear_prob > BEAR_THRESHOLD: exit_signal = True
            elif bear_shark >= BEAR_SHARK_COUNT: exit_signal = True
            elif unrealized < -BASE_STOP: exit_signal = True
            elif dd_from_high > TRAILING_STOP: exit_signal = True
            elif cel['sun_opp_saturn']: exit_signal = True # Celestial Exit
            
            if exit_signal:
                cash += shares * price
                shares = 0
                cooldown = 3 # v6.2 Cooldown
                
        equity = cash + shares * price
        equity_curve.append(equity)
        
    final_equity = equity_curve[-1]
    ret = (final_equity - 100000) / 100000
    print(f"\n=== WST-X BACKTEST RESULTS ===")
    print(f"Total Return: {ret*100:,.2f}%")
    print(f"Final Equity: ${final_equity:,.2f}")

if __name__ == "__main__":
    backtest_wstx()
