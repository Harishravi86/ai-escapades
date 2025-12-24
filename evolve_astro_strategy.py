
import pandas as pd
import numpy as np
import os
import math
import pickle
import warnings
from datetime import datetime, timedelta
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# NEAT import
try:
    import neat
except ImportError:
    print("Error: neat-python not installed. Please run: pip install neat-python")
    exit(1)

# Celestial
try:
    import ephem
    HAS_EPHEM = True
except ImportError:
    HAS_EPHEM = False
    print("Warning: ephem not installed. Celestial features will be zeros.")

# ==============================================================================
# NEAT CONFIGURATION (Embedded)
# ==============================================================================
NEAT_CONFIG_CONTENT = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000.0
pop_size              = 500
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
# Node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu
single_structural_mutation = False
structural_mutation_surer = true

# Node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
bias_init_type          = gaussian

# Genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# Connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.2

# Connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0

# Feed_forward network
feed_forward            = True
initial_connection      = full

# Node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# Network parameters
num_hidden              = 0
num_inputs              = 9
num_outputs             = 1

# Node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
response_init_type      = gaussian

# Connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
weight_init_type        = gaussian

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 0

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""

# ==============================================================================
# CELESTIAL ENGINE
# ==============================================================================
class CelestialCalculator:
    def __init__(self):
        if not HAS_EPHEM: return
        self.obs = ephem.Observer()
    
    def get_features(self, date_str):
        if not HAS_EPHEM:
            return {'sun_saturn_sep': 0, 'moon_uranus_sep': 0, 'moon_phase': 0}
            
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(hours=20)
            self.obs.date = dt
            
            sun = ephem.Sun()
            saturn = ephem.Saturn()
            moon = ephem.Moon()
            uranus = ephem.Uranus()
            
            for body in [sun, saturn, moon, uranus]:
                body.compute(self.obs)
            
            s_lon = math.degrees(ephem.Ecliptic(sun).lon)
            sat_lon = math.degrees(ephem.Ecliptic(saturn).lon)
            sun_sat = abs(s_lon - sat_lon)
            if sun_sat > 180: sun_sat = 360 - sun_sat
            
            m_lon = math.degrees(ephem.Ecliptic(moon).lon)
            u_lon = math.degrees(ephem.Ecliptic(uranus).lon)
            moon_ura = abs(m_lon - u_lon)
            if moon_ura > 180: moon_ura = 360 - moon_ura
            
            return {
                'sun_saturn_sep': sun_sat,
                'moon_uranus_sep': moon_ura,
                'moon_phase': moon.phase
            }
        except:
             return {'sun_saturn_sep': 0, 'moon_uranus_sep': 0, 'moon_phase': 0}

# ==============================================================================
# WORLD STATE BUILDER
# ==============================================================================
class WorldState:
    def __init__(self):
        self.data = None
        self.celestial = CelestialCalculator()
        
    def load_data(self):
        print("Loading market data...")
        try:
            # Load QQQ and VIX
            base_dir = r"c:\Users\jhana\trading_research\spy-trading"
            qqq_path = os.path.join(base_dir, "QQQ_data.csv")
            vix_path = os.path.join(base_dir, "VIX_data.csv")
            
            if not os.path.exists(qqq_path):
                print(f"Error: {qqq_path} not found.")
                return False

            price_df = pd.read_csv(qqq_path, index_col=0)
            price_df.index = pd.to_datetime(price_df.index, utc=True).tz_localize(None)
            price_df.columns = [c.capitalize() for c in price_df.columns]
            
            # Ensure unique index
            price_df = price_df[~price_df.index.duplicated(keep='first')]
            
            vix_df = pd.read_csv(vix_path, index_col=0)
            vix_df.index = pd.to_datetime(vix_df.index, utc=True).tz_localize(None)
            vix_df = vix_df[~vix_df.index.duplicated(keep='first')]

            # Align data
            df = price_df.copy()
            df['VIX'] = vix_df['Close'].reindex(df.index).fillna(20.0)
            
            # Feature Engineering 
            close = df['Close']
            
            # RSI 14
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # BB %B
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = ma20 + 2*std20
            lower = ma20 - 2*std20
            df['BB_PctB'] = (close - lower) / (upper - lower)
            
            # Dist MA50
            ma50 = close.rolling(50).mean()
            df['Dist_MA50'] = (close - ma50) / close
            
            # Daily Return 
            df['Return'] = close.pct_change()
            
            # Forward fill NaNs
            df = df.fillna(method='ffill').dropna()
            
            # Celestial Features 
            print("Calculating celestial features (vectorized loop)...")
            sun_sat = []
            moon_ura = []
            moon_ph = []
            
            dates = df.index
            for d in dates:
                feat = self.celestial.get_features(d.strftime('%Y-%m-%d'))
                sun_sat.append(feat['sun_saturn_sep'])
                moon_ura.append(feat['moon_uranus_sep'])
                moon_ph.append(feat['moon_phase'])
                
            df['Sun_Saturn_Sep'] = sun_sat
            df['Moon_Uranus_Sep'] = moon_ura
            df['Moon_Phase'] = moon_ph
            
            # Normalize Inputs for Neural Net
            df['Input_RSI'] = df['RSI_14'] / 100.0
            df['Input_BB'] = df['BB_PctB']
            df['Input_MA_Dist'] = df['Dist_MA50'] * 5.0
            df['Input_VIX'] = df['VIX'] / 50.0
            df['Input_Sun_Sat'] = df['Sun_Saturn_Sep'] / 180.0
            df['Input_Moon_Ura'] = df['Moon_Uranus_Sep'] / 180.0
            df['Input_Moon_Ph'] = df['Moon_Phase'] / 100.0
            
            self.data = df
            print(f"World State Build Complete: {len(df)} days.")
            return True
            
        except Exception as e:
            print(f"Data load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_arrays(self):
        input_cols = ['Input_RSI', 'Input_BB', 'Input_MA_Dist', 'Input_VIX', 
                      'Input_Sun_Sat', 'Input_Moon_Ura', 'Input_Moon_Ph']
        
        inputs = self.data[input_cols].values
        returns = self.data['Return'].values
        prices = self.data['Close'].values # Needed for correct PnL
        dates = self.data.index
        return inputs, returns, prices, dates

# ==============================================================================
# EVOLUTION LOGIC
# ==============================================================================

GLOBAL_INPUTS = None
GLOBAL_RETURNS = None
GLOBAL_PRICES = None
TRAIN_END_IDX = 0  # To be set based on 2020 split

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    cash = 10000.0
    shares = 0.0
    entry_price = 0.0
    equity_curve = []
    trades = 0
    
    # Train/Test Split: Use Training Data ONLY (2000-2019)
    # TRAIN_END_IDX is set in run_evolution based on date < 2020-01-01
    inputs = GLOBAL_INPUTS[:TRAIN_END_IDX]
    returns = GLOBAL_RETURNS[:TRAIN_END_IDX]
    prices = GLOBAL_PRICES[:TRAIN_END_IDX]
    
    days_held = 0
    unrealized_feat = 0.0
    
    # Fast Loop
    for i in range(len(returns)):
        price = prices[i]
        ret = returns[i]
        market_feats = inputs[i]
        
        # Update Position State
        if shares > 0:
            shares *= (1 + ret)
            days_held += 1
            # Correct PnL Calculation
            pct_gain = (price - entry_price) / entry_price if entry_price > 0 else 0
            unrealized_feat = np.clip(pct_gain * 5.0, -1.0, 1.0)
        else:
            days_held = 0
            unrealized_feat = 0.0
        
        # Build Input Vector (9 inputs)
        held_feat = min(days_held / 20.0, 1.0)
        net_input = np.append(market_feats, [held_feat, unrealized_feat])
        
        # Activate Brain
        signal = net.activate(net_input)[0]
        
        # Execute
        if signal > 0.1 and shares == 0:
            # BUY
            shares = cash
            cash = 0
            entry_price = price
            trades += 1
            days_held = 0
            unrealized_feat = 0.0
            
        elif signal < -0.1 and shares > 0:
            # SELL
            cash = shares
            shares = 0
            days_held = 0
            unrealized_feat = 0.0
            
        equity_curve.append(cash + shares)
    
    # Metrics
    final_equity = cash + shares
    total_return = (final_equity - 10000) / 10000
    
    # Drawdown Penalty
    peak = 10000.0
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)
        
    # Stats
    eq_arr = np.array(equity_curve)
    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    daily_rets = np.nan_to_num(daily_rets) # Handle /0
    
    std = np.std(daily_rets)
    sharpe = (np.mean(daily_rets) / std * np.sqrt(252)) if std > 1e-9 else 0
    
    # Fitness Calculation
    if trades < 5:
        return -100.0 # Penalty for inactivity
        
    if total_return < 0:
        return total_return * 100
        
    # Primary Reward: Return * Sharpe
    fitness = total_return * 100 * (1 + max(0, sharpe))
    
    # Drawdown Penalty: Heavy hit if > 20% (Stricter)
    if max_dd > 0.20:
        fitness *= (1 - max_dd * 2.0) # Quadratic-like penalty (0.2 -> 0.6 multiplier, 0.5 -> 0.0)
        
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def validate_winner(genome, config):
    print("\n" + "="*50)
    print("VALIDATION ON HOLDOUT DATA (Post-2018)")
    print("="*50)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # VALIDATION on 2020-2025 Data
    inputs = GLOBAL_INPUTS[TRAIN_END_IDX:]
    returns = GLOBAL_RETURNS[TRAIN_END_IDX:]
    prices = GLOBAL_PRICES[TRAIN_END_IDX:]
    
    cash = 10000.0
    shares = 0.0
    entry_price = 0.0
    equity_curve = []
    trades = 0
    
    days_held = 0
    unrealized_feat = 0.0
    
    for i in range(len(returns)):
        price = prices[i]
        ret = returns[i]
        market_feats = inputs[i]
        
        if shares > 0:
            shares *= (1 + ret)
            days_held += 1
            pct_gain = (price - entry_price) / entry_price if entry_price > 0 else 0
            unrealized_feat = np.clip(pct_gain * 5.0, -1.0, 1.0)
        else:
            days_held = 0
            unrealized_feat = 0.0
            
        held_feat = min(days_held / 20.0, 1.0)
        net_input = np.append(market_feats, [held_feat, unrealized_feat])
        signal = net.activate(net_input)[0]
        
        if signal > 0.1 and shares == 0:
            shares = cash; cash = 0; entry_price = price; trades += 1
        elif signal < -0.1 and shares > 0:
            cash = shares; shares = 0
            
        equity_curve.append(cash + shares)
        
    final = cash + shares
    ret = (final - 10000)/10000
    print(f"Validation Return: {ret:.1%}")
    print(f"Validation Trades: {trades}")
    print(f"Final Equity: ${final:.2f}")

def run_evolution():
    # 1. Setup Data
    world = WorldState()
    if not world.load_data(): return
        
    global GLOBAL_INPUTS, GLOBAL_RETURNS, GLOBAL_PRICES, TRAIN_END_IDX
    GLOBAL_INPUTS, GLOBAL_RETURNS, GLOBAL_PRICES, dates = world.get_arrays()
    
    # STRICT SPLIT: Train < 2020-01-01
    split_date = pd.Timestamp('2020-01-01')
    TRAIN_END_IDX = np.searchsorted(dates, split_date)
    print(f"Data Split: Train (2000-2019) Ends at Index {TRAIN_END_IDX} (Date: {dates[TRAIN_END_IDX]})")
    
    # 2. Setup Config
    config_path = 'neat_config_temp.txt'
    with open(config_path, 'w') as f:
        f.write(NEAT_CONFIG_CONTENT)
        
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # 3. Evolve (100 Generations)
    print("\nStarting Evolution (Training on 2000-2019)...")
    winner = p.run(eval_genomes, 100)
    
    print(f'\nBest genome:\n{winner}')
    
    # 4. Save & Validate
    with open('best_astro_trader.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    validate_winner(winner, config)
    
    if os.path.exists(config_path):
        os.remove(config_path)

if __name__ == '__main__':
    run_evolution()
