# analyze_neat_winner.py (v7 - Modern GA)
import os
import pickle
import warnings
import json
from datetime import datetime, timedelta

import neat
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================================
# Config (Must match evolve_spy_trader.py)
# ==========================================================
INITIAL_CAPITAL = 100_000
RISK_PCT = 0.02               # 2% per trade
MAX_TRADE_PCT = 0.10          # 10% of equity
SLIPPAGE = 0.0005             # 0.05%
COMMISSION = 1.0
MAX_HOLD = 50
COOLDOWN = 3
MIN_DATA = 200

# ==========================================================
# Data Loader — %B ONLY
# ==========================================================
def load_data(days=720):
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    try:
        df = yf.download("SPY", start=start, end=end, interval="1h", progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None
        
    if df is None or df.empty:
        return None, None

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Clean
    close = df["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    
    high = df["High"]
    if isinstance(high, pd.DataFrame): high = high.iloc[:, 0]
    
    low = df["Low"]
    if isinstance(low, pd.DataFrame): low = low.iloc[:, 0]
    
    volume = df["Volume"]
    if isinstance(volume, pd.DataFrame): volume = volume.iloc[:, 0]

    # ---- %B Explicit ----
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    pct_b = (close - sma20) / (2 * std20)

    # ---- Additional Features ----
    rsi = ta.rsi(close, length=14) / 100.0
    macd = ta.macd(close)
    if macd is not None:
        macd_hist = macd.iloc[:, 1]
    else:
        macd_hist = close * 0.0
        
    macd_norm = macd_hist / (macd_hist.rolling(100).std() + 1e-9)
    macd_norm = macd_norm.clip(-3, 3) / 3.0

    atr = ta.atr(high, low, close, length=14).fillna(method="bfill")

    # Normalized inputs
    df2 = pd.DataFrame({
        "Close": close,
        "PCT_B": pct_b.clip(-3, 3) / 3.0,
        "RSI": rsi,
        "MACD": macd_norm,
        "ATR": atr,
        "RET": close.pct_change().fillna(0) * 100.0,
        "MOM": close.pct_change(5).fillna(0).clip(-5, 5) / 5.0,
        "HOUR": (df.index.hour - 9.5) / 6.5,
        "DOW": df.index.dayofweek / 4.0,
    })

    df2 = df2.dropna()
    if len(df2) < MIN_DATA:
        return None, None

    # Use ALL data for analysis to see full history including Oct 13
    return df2


# ==========================================================
# Gap‑Aware Trade Simulator (Analysis Mode)
# ==========================================================
def simulate(net, df, verbose=False):
    if df is None or len(df) < MIN_DATA:
        return {"fitness": -1.0}, [], 0.0

    cols = ["PCT_B", "RSI", "MACD", "RET", "MOM", "HOUR", "DOW"]
    X = df[cols].values
    close = df["Close"].values
    atr = df["ATR"].values
    dates = df.index

    equity = INITIAL_CAPITAL
    cash = equity
    shares = 0.0

    entry = 0.0
    stop = 0.0
    target = 0.0
    bars = 0
    cooldown = 0

    peak = equity
    maxdd = 0.0

    trades = []
    trade_details = []

    for i in range(len(X)):
        price = close[i]
        date = dates[i]

        # Floating PNL update
        cur_equity = cash + shares * price
        peak = max(peak, cur_equity)
        if peak > 0:
            maxdd = max(maxdd, (peak - cur_equity) / peak)

        if cooldown > 0:
            cooldown -= 1

        # =============================================
        # EXIT LOGIC (gap aware)
        # =============================================
        if shares != 0:
            bars += 1

            # Long
            if shares > 0:
                gap_hit = price <= stop
                target_hit = price >= target

                if gap_hit or target_hit or bars >= MAX_HOLD:
                    slip = price * SLIPPAGE
                    exec_price = price - slip
                    ret = (exec_price - entry) / entry
                    cash += shares * exec_price - COMMISSION
                    trades.append(ret)
                    
                    reason = "STOP" if gap_hit else ("TARGET" if target_hit else "TIME")
                    trade_details.append({
                        "entry_date": str(entry_date),
                        "exit_date": str(date),
                        "entry_price": float(entry),
                        "exit_price": float(exec_price),
                        "return": float(ret),
                        "reason": reason
                    })
                    
                    if verbose:
                        print(f"EXIT {reason}: {date} Price={price:.2f} Ret={ret*100:.2f}% Eq={cash:.2f}")

                    shares = 0.0
                    bars = 0
                    cooldown = COOLDOWN
                continue

        # =============================================
        # ENTRY LOGIC
        # =============================================
        if cooldown == 0:
            out = net.activate(X[i])
            buy = out[0]
            sell = out[1]

            if buy > 0.5 and buy > sell:
                # Position sizing: 2% risk or 10% cap
                atr_val = atr[i]
                risk_dollar = equity * RISK_PCT
                stop_dist = max(atr_val * 1.5, 0.01)
                qty_risk = risk_dollar / stop_dist

                alloc_cap = equity * MAX_TRADE_PCT
                qty_alloc = alloc_cap / price

                qty = max(1.0, min(qty_risk, qty_alloc))

                cost = qty * (price + price * SLIPPAGE) + COMMISSION
                if cost <= cash:
                    exec_price = price + price * SLIPPAGE
                    cash -= cost
                    shares = qty
                    entry = exec_price
                    entry_date = date
                    stop = entry - stop_dist
                    target = entry + atr_val * 3.0
                    bars = 0
                    
                    if verbose:
                        print(f"BUY: {date} Price={price:.2f} Qty={qty:.2f} Stop={stop:.2f} Target={target:.2f}")

    # =============================================
    # Final liquidation
    # =============================================
    if shares > 0:
        final_price = close[-1]
        exec_price = final_price - final_price * SLIPPAGE
        cash += shares * exec_price - COMMISSION
        ret = (exec_price - entry) / entry
        trades.append(ret)
        trade_details.append({
            "entry_date": str(entry_date),
            "exit_date": str(dates[-1]),
            "entry_price": float(entry),
            "exit_price": float(exec_price),
            "return": float(ret),
            "reason": "FINAL"
        })
        shares = 0.0

    final_eq = cash
    total_profit = final_eq - INITIAL_CAPITAL

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    winrate = len(wins) / len(trades) if trades else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else 999.0

    metrics = {
        "fitness": 0.0, # Not needed for analysis
        "profit": float(total_profit),
        "final_equity": float(final_eq),
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "maxdd": float(maxdd),
        "trades": int(len(trades)),
    }

    if verbose:
        print("\n========================================")
        print("NEAT TRADER PERFORMANCE (v7)")
        print("========================================")
        print(f"Total Trades:    {metrics['trades']}")
        print(f"Win Rate:        {metrics['winrate']*100:.1f}%")
        print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
        print(f"Total Return:    {metrics['profit']/INITIAL_CAPITAL*100:.2f}%")
        print(f"Max Drawdown:    {metrics['maxdd']*100:.2f}%")
        print(f"Final Equity:    ${metrics['final_equity']:.2f}")
        print("========================================")

    # Save results
    results = {
        "metrics": metrics,
        "trades": trade_details
    }
    with open("neat_results_v7.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Results saved to 'neat_results_v7.json'")

    return metrics, trades, total_profit


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.cfg")
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load best genome
    try:
        with open("best_ga_spy.pkl", "rb") as f:
            winner = pickle.load(f)
            
        print("Loaded best genome.")
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        
        # Load data
        df = load_data(days=720)
        
        # Run simulation
        simulate(net, df, verbose=True)
        
    except Exception as e:
        print(f"Error loading genome or running analysis: {e}")
