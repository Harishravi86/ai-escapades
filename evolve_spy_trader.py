# GA Strategy Rewritten (Clean, Modern, %B‑Only Version)
# -------------------------------------------------------
# Key Changes:
# - Removed Bollinger Bands logic entirely; **%B only**
# - Simplified feature set to match your actual strategy components
# - Added proper gap‑aware stop logic
# - Option‑or‑stock compatible sizing (2% capital per trade)
# - Improved reward shaping for high return + stable drawdown
# - Eliminated unused/unstable indicators
# - Fully modular, easier for further evolution

import os
import pickle
import warnings
from datetime import datetime, timedelta

import neat
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================================
# Config
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
    # Ensure inputs are Series
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

    split = int(len(df2) * 0.7)
    return df2.iloc[:split], df2.iloc[split:]


TRAIN, TEST = load_data()

# ==========================================================
# Gap‑Aware Trade Simulator
# ==========================================================
def simulate(net, df, verbose=False):
    if df is None or len(df) < MIN_DATA:
        return {"fitness": -1.0}, [], 0.0

    cols = ["PCT_B", "RSI", "MACD", "RET", "MOM", "HOUR", "DOW"]
    X = df[cols].values
    close = df["Close"].values
    atr = df["ATR"].values

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
    rng = np.random.default_rng()

    for i in range(len(X)):
        price = close[i]

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
                stop_dist = max(atr_val * 1.5, 0.01) # Prevent div by zero
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
                    stop = entry - stop_dist
                    target = entry + atr_val * 3.0
                    bars = 0

    # =============================================
    # Final liquidation
    # =============================================
    if shares > 0:
        final_price = close[-1]
        exec_price = final_price - final_price * SLIPPAGE
        cash += shares * exec_price - COMMISSION
        trades.append((exec_price - entry) / entry)
        shares = 0.0

    final_eq = cash
    total_profit = final_eq - INITIAL_CAPITAL

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    winrate = len(wins) / len(trades) if trades else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else 999.0

    # ==========================================================
    # FITNESS — High Return, Low Drawdown
    # ==========================================================
    if len(trades) < 5:
        fitness = -1.0
    else:
        calmar = total_profit / (INITIAL_CAPITAL * max(maxdd, 0.01))
        fitness = float(np.tanh(calmar * 0.5))
        if winrate > 0.6:
            fitness += 0.05
        if maxdd > 0.3:
            fitness -= 0.3

    metrics = {
        "fitness": fitness,
        "profit": total_profit,
        "final_equity": final_eq,
        "winrate": winrate,
        "profit_factor": profit_factor,
        "maxdd": maxdd,
        "trades": len(trades),
    }

    if verbose:
        print(metrics)

    return metrics, trades, total_profit

# ==========================================================
# NEAT Evaluation
# ==========================================================
def eval_genomes(genomes, config):
    if TRAIN is None:
        for gid, genome in genomes:
            genome.fitness = -1.0
        return

    for gid, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        metr, _, _ = simulate(net, TRAIN)
        genome.fitness = metr["fitness"]


def run(cfg_path, generations=100):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_path,
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    print(f"Running GA — {generations} generations...")
    winner = pop.run(eval_genomes, generations)

    with open("best_ga_spy.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Testing best genome...")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    simulate(net, TEST, verbose=True)

    print("Saved best genome (best_ga_spy.pkl)")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    cfg = os.path.join(local_dir, "neat_config.cfg")
    run(cfg, generations=15)
