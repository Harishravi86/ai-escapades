#!/usr/bin/env python3
"""
================================================================================
NEAT EVOLUTIONARY STRATEGY v2.0 - CORRECTED
================================================================================

Fixes from v1:
1. Train/Test Split: 2000-2019 train, 2020-2025 holdout
2. 100 Generations: Proper topology evolution time
3. Drawdown Penalty: fitness *= (1 - max_dd) if DD > 20%
4. Fixed Unrealized PnL: Proper entry price tracking
5. Price Array: Enable accurate PnL calculation

Goal: Discover non-linear celestial interactions that XGBoost can't represent.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import math
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import tempfile

warnings.filterwarnings('ignore')

# NEAT import
try:
    import neat
    HAS_NEAT = True
except ImportError:
    HAS_NEAT = False
    print("Error: neat-python not installed. Run: pip install neat-python")

# Celestial
try:
    import ephem
    HAS_EPHEM = True
except ImportError:
    HAS_EPHEM = False
    print("Warning: ephem not installed. Celestial features will be zeros.")

# Data loading
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ==============================================================================
# NEAT CONFIGURATION
# ==============================================================================
NEAT_CONFIG_CONTENT = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 50000.0
pop_size              = 300
reset_on_extinction   = True
no_fitness_termination = False

[DefaultGenome]
# Node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu

# Structural mutation
single_structural_mutation = False
structural_mutation_surer = default

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

# Genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# Connection add/remove rates
conn_add_prob           = 0.4
conn_delete_prob        = 0.2

# Connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

# Feed-forward network
feed_forward            = True
initial_connection      = full_nodirect

# Node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.1

# Network parameters
num_hidden              = 0
num_inputs              = 11
num_outputs             = 1

# Node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# Connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.2
min_species_size   = 2
"""


# ==============================================================================
# CELESTIAL ENGINE
# ==============================================================================
class CelestialCalculator:
    """Calculate celestial features for each trading day."""
    
    def __init__(self):
        if not HAS_EPHEM:
            return
        self.obs = ephem.Observer()
        self._cache = {}
    
    def get_features(self, date_str: str) -> Dict:
        """Get normalized celestial features for a date."""
        if not HAS_EPHEM:
            return self._empty()
        
        if date_str in self._cache:
            return self._cache[date_str]
        
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(hours=20)
            self.obs.date = dt
            
            sun = ephem.Sun()
            saturn = ephem.Saturn()
            moon = ephem.Moon()
            uranus = ephem.Uranus()
            
            for body in [sun, saturn, moon, uranus]:
                body.compute(self.obs)
            
            # Get ecliptic longitudes
            s_lon = math.degrees(ephem.Ecliptic(sun).lon)
            sat_lon = math.degrees(ephem.Ecliptic(saturn).lon)
            m_lon = math.degrees(ephem.Ecliptic(moon).lon)
            u_lon = math.degrees(ephem.Ecliptic(uranus).lon)
            
            # Calculate separations
            sun_sat = abs(s_lon - sat_lon)
            if sun_sat > 180: sun_sat = 360 - sun_sat
            
            moon_ura = abs(m_lon - u_lon)
            if moon_ura > 180: moon_ura = 360 - moon_ura
            
            # Moon phase (0-100)
            moon_phase = moon.phase
            
            # Binary aspects (within orb)
            sun_sat_opp = 1.0 if abs(sun_sat - 180) < 5 else 0.0
            moon_ura_opp = 1.0 if abs(moon_ura - 180) < 8 else 0.0
            
            # New Moon window (+/- 25 deg)
            sun_moon_diff = m_lon - s_lon
            while sun_moon_diff < 0: sun_moon_diff += 360
            while sun_moon_diff >= 360: sun_moon_diff -= 360
            new_moon_window = 1.0 if (sun_moon_diff < 25 or sun_moon_diff > 335) else 0.0
            
            features = {
                'sun_saturn_sep': sun_sat / 180.0,  # Normalized 0-1
                'moon_uranus_sep': moon_ura / 180.0,
                'moon_phase': moon_phase / 100.0,
                'sun_sat_opp': sun_sat_opp,
                'moon_ura_opp': moon_ura_opp,
                'new_moon_window': new_moon_window,
            }
            
            self._cache[date_str] = features
            return features
            
        except Exception:
            return self._empty()
    
    def _empty(self) -> Dict:
        return {
            'sun_saturn_sep': 0.5,
            'moon_uranus_sep': 0.5,
            'moon_phase': 0.5,
            'sun_sat_opp': 0.0,
            'moon_ura_opp': 0.0,
            'new_moon_window': 0.0,
        }


# ==============================================================================
# WORLD STATE BUILDER
# ==============================================================================
class WorldState:
    """
    Pre-compute all features for fast simulation.
    
    Input Features (11 total):
    0. RSI_14 normalized (0-1)
    1. BB %B (can be <0 or >1)
    2. Distance from MA50 (scaled)
    3. VIX normalized
    4. Sun-Saturn separation (0-1)
    5. Moon-Uranus separation (0-1)
    6. Moon phase (0-1)
    7. Sun-Saturn opposition active (0/1)
    8. Moon-Uranus opposition active (0/1)
    9. Days held (normalized, 0-1, max 20)
    10. Unrealized PnL (scaled, clipped -1 to 1)
    """
    
    def __init__(self):
        self.data = None
        self.celestial = CelestialCalculator()
        
        # Arrays for fast simulation
        self.inputs = None  # Static market features
        self.returns = None
        self.prices = None
        self.dates = None
        
        # Train/Test indices
        self.train_end_idx = None
        self.test_start_idx = None
        
    def load_data(self, csv_path: str = None) -> bool:
        """Load and prepare all data."""
        print("Loading market data...")
        
        try:
            if csv_path and os.path.exists(csv_path):
                # Load from CSV
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            elif HAS_YF:
                # Download from yfinance
                print("Downloading QQQ data...")
                df = yf.download('QQQ', start='2000-01-01', progress=False)
                
                print("Downloading VIX data...")
                vix = yf.download('^VIX', start='2000-01-01', progress=False)
                df['VIX'] = vix['Close'].reindex(df.index).fillna(20.0)
            else:
                print("Error: No data source available")
                return False
            
            # Ensure datetime index with strict conversion
            # coerce=True forces invalid dates to NaT (then dropped)
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
            df = df[df.index.notna()]
            df.index = df.index.tz_localize(None)
            
            # Add VIX if missing
            if 'VIX' not in df.columns:
                df['VIX'] = 20.0
            
            # Feature Engineering
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # RSI 14
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands %B
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            df['BB_PctB'] = (close - lower) / (upper - lower)
            
            # Distance from MA50
            ma50 = close.rolling(50).mean()
            df['Dist_MA50'] = (close - ma50) / close
            
            # Daily Return
            df['Return'] = close.pct_change()
            
            # Forward fill and drop NaN
            df = df.ffill().dropna()
            
            # Celestial Features
            print("Calculating celestial features...")
            sun_sat_sep = []
            moon_ura_sep = []
            moon_ph = []
            sun_sat_opp = []
            moon_ura_opp = []
            new_moon_win = []
            
            for d in df.index:
                # Handle both Timestamp and string
                d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
                feat = self.celestial.get_features(d_str)
                sun_sat_sep.append(feat['sun_saturn_sep'])
                moon_ura_sep.append(feat['moon_uranus_sep'])
                moon_ph.append(feat['moon_phase'])
                sun_sat_opp.append(feat['sun_sat_opp'])
                moon_ura_opp.append(feat['moon_ura_opp'])
                new_moon_win.append(feat['new_moon_window'])
            
            df['Sun_Saturn_Sep'] = sun_sat_sep
            df['Moon_Uranus_Sep'] = moon_ura_sep
            df['Moon_Phase'] = moon_ph
            df['Sun_Sat_Opp'] = sun_sat_opp
            df['Moon_Ura_Opp'] = moon_ura_opp
            df['New_Moon_Win'] = new_moon_win
            
            # Normalize inputs for neural network
            df['Input_RSI'] = df['RSI_14'] / 100.0
            df['Input_BB'] = df['BB_PctB'].clip(-1, 2)  # Clip extreme values
            df['Input_MA_Dist'] = (df['Dist_MA50'] * 10).clip(-1, 1)
            df['Input_VIX'] = (df['VIX'] / 40.0).clip(0, 2)
            
            self.data = df
            
            # Build arrays
            self._build_arrays()
            
            print(f"World State Complete: {len(df)} days")
            print(f"  Train period: {self.dates[0].strftime('%Y-%m-%d')} to {self.dates[self.train_end_idx].strftime('%Y-%m-%d')}")
            print(f"  Test period: {self.dates[self.test_start_idx].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            print(f"Data load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_arrays(self):
        """Build numpy arrays for fast simulation."""
        df = self.data
        
        # Static market features (9 features)
        input_cols = [
            'Input_RSI',       # 0
            'Input_BB',        # 1
            'Input_MA_Dist',   # 2
            'Input_VIX',       # 3
            'Sun_Saturn_Sep',  # 4
            'Moon_Uranus_Sep', # 5
            'Moon_Phase',      # 6
            'Sun_Sat_Opp',     # 7
            'Moon_Ura_Opp',    # 8
            # 9, 10 are dynamic (days_held, unrealized_pnl)
        ]
        
        self.inputs = df[input_cols].values.astype(np.float32)
        self.returns = df['Return'].values.astype(np.float32)
        self.prices = df['Close'].values.astype(np.float32)
        self.dates = df.index.to_numpy()
        
        # Find train/test split (2020-01-01)
        split_date = pd.Timestamp('2020-01-01')
        for i, d in enumerate(self.dates):
            if pd.Timestamp(d) >= split_date:
                self.train_end_idx = i - 1
                self.test_start_idx = i
                break
        
        if self.train_end_idx is None:
            # All data before 2020
            self.train_end_idx = len(self.dates) - 1
            self.test_start_idx = len(self.dates)
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get training data (2000-2019)."""
        end = self.train_end_idx + 1
        return self.inputs[:end], self.returns[:end], self.prices[:end]
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get test data (2020-2025)."""
        start = self.test_start_idx
        return self.inputs[start:], self.returns[start:], self.prices[start:]


# ==============================================================================
# SIMULATION ENGINE
# ==============================================================================
def simulate_genome(
    net,
    inputs: np.ndarray,
    returns: np.ndarray,
    prices: np.ndarray,
    initial_capital: float = 10000.0
) -> Dict:
    """
    Run trading simulation for a NEAT network.
    
    Returns dict with:
        - final_equity
        - total_return
        - sharpe
        - max_drawdown
        - trades
        - equity_curve
    """
    cash = initial_capital
    shares = 0.0
    entry_price = 0.0
    days_held = 0
    
    equity_curve = []
    trades = 0
    wins = 0
    trade_returns = []
    
    n_days = len(returns)
    
    for i in range(n_days):
        price = prices[i]
        ret = returns[i]
        market_feats = inputs[i]  # 9 static features
        
        # Update position value
        if shares > 0:
            shares *= (1 + ret)
            days_held += 1
        
        # Calculate dynamic features
        held_feat = min(days_held / 20.0, 1.0)
        
        if shares > 0 and entry_price > 0:
            unrealized_pct = (price - entry_price) / entry_price
            unrealized_feat = np.clip(unrealized_pct * 5, -1, 1)
        else:
            unrealized_feat = 0.0
        
        # Build full input vector (11 features)
        net_input = np.concatenate([market_feats, [held_feat, unrealized_feat]])
        
        # Get network output
        output = net.activate(net_input)
        signal = output[0]  # Tanh: -1 to 1
        
        # Current equity
        current_equity = cash + shares
        
        # Trading logic
        if signal > 0.2:  # BUY signal
            if shares == 0:
                # Entry
                entry_price = price
                shares = cash
                cash = 0
                trades += 1
                days_held = 0
                
        elif signal < -0.2:  # SELL signal
            if shares > 0:
                # Exit
                exit_value = shares
                # Track trade return
                actual_ret = (price - entry_price) / entry_price if entry_price > 0 else 0
                trade_returns.append(actual_ret)
                if actual_ret > 0:
                    wins += 1
                
                cash = shares
                shares = 0
                days_held = 0
                entry_price = 0
        
        equity_curve.append(cash + shares)
    
    # Final equity
    final_equity = cash + shares
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Calculate Sharpe
    eq_arr = np.array(equity_curve)
    if len(eq_arr) > 1:
        daily_rets = np.diff(eq_arr) / (eq_arr[:-1] + 1e-9)
        daily_rets = np.nan_to_num(daily_rets, nan=0, posinf=0, neginf=0)
        std = np.std(daily_rets)
        sharpe = (np.mean(daily_rets) / (std + 1e-9)) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Calculate Max Drawdown
    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    # Win rate
    win_rate = wins / trades if trades > 0 else 0
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades,
        'win_rate': win_rate,
        'equity_curve': equity_curve,
        'trade_returns': trade_returns,
    }


# ==============================================================================
# FITNESS FUNCTION
# ==============================================================================

# Global state for parallel evaluation
GLOBAL_TRAIN_INPUTS = None
GLOBAL_TRAIN_RETURNS = None
GLOBAL_TRAIN_PRICES = None


def eval_genome(genome, config) -> float:
    """
    Evaluate a single genome on TRAINING data only.
    
    Fitness = Return × (1 + Sharpe) × DrawdownPenalty
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    results = simulate_genome(
        net,
        GLOBAL_TRAIN_INPUTS,
        GLOBAL_TRAIN_RETURNS,
        GLOBAL_TRAIN_PRICES,
        initial_capital=10000.0
    )
    
    total_return = results['total_return']
    sharpe = results['sharpe']
    max_dd = results['max_drawdown']
    trades = results['trades']
    
    # Penalty for doing nothing
    if trades < 10:
        return -100.0
    
    # Penalty for losing money
    if total_return < 0:
        return total_return * 100
    
    # Base fitness: return percentage
    fitness = total_return * 100
    
    # Sharpe bonus (only positive Sharpe helps)
    if sharpe > 0:
        fitness *= (1 + sharpe * 0.5)
    
    # Drawdown penalty (harsh above 20%)
    if max_dd > 0.20:
        fitness *= (1 - max_dd)
    
    # Extra penalty for extreme drawdowns
    if max_dd > 0.40:
        fitness *= 0.5
    
    return fitness


def eval_genomes(genomes, config):
    """Evaluate all genomes in population."""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


# ==============================================================================
# EVOLUTION RUNNER
# ==============================================================================
class NEATEvolver:
    """Run NEAT evolution with proper train/test validation."""
    
    def __init__(self, world: WorldState):
        self.world = world
        self.winner = None
        self.stats = None
        
    def run(self, generations: int = 100, checkpoint_freq: int = 25) -> neat.DefaultGenome:
        """Run evolution for specified generations."""
        global GLOBAL_TRAIN_INPUTS, GLOBAL_TRAIN_RETURNS, GLOBAL_TRAIN_PRICES
        
        # Set global training data
        GLOBAL_TRAIN_INPUTS, GLOBAL_TRAIN_RETURNS, GLOBAL_TRAIN_PRICES = self.world.get_train_data()
        
        print(f"\nTraining data: {len(GLOBAL_TRAIN_INPUTS)} days")
        print(f"Running evolution for {generations} generations...")
        print("=" * 70)
        
        # Write config to temp file
        config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
        with open(config_path, 'w') as f:
            f.write(NEAT_CONFIG_CONTENT)
        
        # Load config
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Create population
        pop = neat.Population(config)
        
        # Add reporters
        pop.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        
        # Add checkpointer
        pop.add_reporter(neat.Checkpointer(
            checkpoint_freq,
            filename_prefix='neat-v2-checkpoint-'
        ))
        
        # Run evolution
        self.winner = pop.run(eval_genomes, generations)
        
        # Cleanup
        os.unlink(config_path)
        
        print("\n" + "=" * 70)
        print("EVOLUTION COMPLETE")
        print("=" * 70)
        
        return self.winner
    
    def validate_winner(self) -> Dict:
        """Validate winner on holdout test set (2020-2025)."""
        if self.winner is None:
            print("No winner to validate!")
            return {}
        
        print("\n" + "=" * 70)
        print("HOLDOUT VALIDATION (2020-2025)")
        print("=" * 70)
        
        # Load config for network creation
        config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
        with open(config_path, 'w') as f:
            f.write(NEAT_CONFIG_CONTENT)
        
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        os.unlink(config_path)
        
        # Create network from winner
        net = neat.nn.FeedForwardNetwork.create(self.winner, config)
        
        # Get test data
        test_inputs, test_returns, test_prices = self.world.get_test_data()
        
        print(f"Test data: {len(test_inputs)} days")
        
        # Run simulation
        results = simulate_genome(
            net,
            test_inputs,
            test_returns,
            test_prices,
            initial_capital=10000.0
        )
        
        # Also run on training data for comparison
        train_inputs, train_returns, train_prices = self.world.get_train_data()
        train_results = simulate_genome(
            net,
            train_inputs,
            train_returns,
            train_prices,
            initial_capital=10000.0
        )
        
        print("\n--- TRAINING PERIOD (2000-2019) ---")
        print(f"  Total Return: {train_results['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {train_results['sharpe']:.2f}")
        print(f"  Max Drawdown: {train_results['max_drawdown']*100:.2f}%")
        print(f"  Trades:       {train_results['trades']}")
        print(f"  Win Rate:     {train_results['win_rate']*100:.1f}%")
        
        print("\n--- TEST PERIOD (2020-2025) ---")
        print(f"  Total Return: {results['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"  Trades:       {results['trades']}")
        print(f"  Win Rate:     {results['win_rate']*100:.1f}%")
        
        # Overfit detection
        train_sharpe = train_results['sharpe']
        test_sharpe = results['sharpe']
        sharpe_decay = (train_sharpe - test_sharpe) / train_sharpe if train_sharpe > 0 else 0
        
        print(f"\n--- OVERFIT ANALYSIS ---")
        print(f"  Sharpe Decay: {sharpe_decay*100:.1f}%")
        if sharpe_decay > 0.5:
            print("  ⚠️  WARNING: >50% Sharpe decay suggests overfitting")
        elif sharpe_decay > 0.25:
            print("  ⚠️  CAUTION: 25-50% Sharpe decay, moderate overfit risk")
        else:
            print("  ✓  GOOD: <25% Sharpe decay, model generalizes well")
        
        return {
            'train': train_results,
            'test': results,
            'sharpe_decay': sharpe_decay,
        }
    
    def save_winner(self, path: str = 'best_neat_trader.pkl'):
        """Save the winning genome."""
        if self.winner is None:
            print("No winner to save!")
            return
        
        with open(path, 'wb') as f:
            pickle.dump(self.winner, f)
        print(f"Winner saved to {path}")
    
    def analyze_network(self):
        """Analyze the evolved network structure."""
        if self.winner is None:
            print("No winner to analyze!")
            return
        
        print("\n" + "=" * 70)
        print("NETWORK ANALYSIS")
        print("=" * 70)
        
        # Count nodes and connections
        nodes = list(self.winner.nodes.keys())
        connections = [(k, v) for k, v in self.winner.connections.items() if v.enabled]
        
        input_nodes = [n for n in nodes if n < 0]
        output_nodes = [n for n in nodes if n == 0]  # Output is node 0
        hidden_nodes = [n for n in nodes if n > 0 and n not in output_nodes]
        
        print(f"Nodes: {len(nodes)} total")
        print(f"  Input: {len(input_nodes)}")
        print(f"  Hidden: {len(hidden_nodes)}")
        print(f"  Output: {len(output_nodes)}")
        print(f"Connections: {len(connections)} enabled")
        
        # Input feature names
        input_names = [
            'RSI_14', 'BB_PctB', 'MA_Dist', 'VIX',
            'Sun_Saturn_Sep', 'Moon_Uranus_Sep', 'Moon_Phase',
            'Sun_Sat_Opp', 'Moon_Ura_Opp',
            'Days_Held', 'Unrealized_PnL'
        ]
        
        # Analyze input weights
        print("\nInput Weights to Output:")
        input_weights = {}
        for (in_node, out_node), conn in self.winner.connections.items():
            if conn.enabled and in_node < 0:
                # Input node indices are negative: -1, -2, ..., -11
                idx = -in_node - 1
                if idx < len(input_names):
                    name = input_names[idx]
                    input_weights[name] = conn.weight
        
        # Sort by absolute weight
        sorted_weights = sorted(input_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, weight in sorted_weights:
            bar = "█" * int(min(abs(weight) * 5, 30))
            sign = "+" if weight > 0 else "-"
            print(f"  {name:20s} {sign}{abs(weight):6.3f} {bar}")
        
        # Celestial feature importance
        celestial_features = ['Sun_Saturn_Sep', 'Moon_Uranus_Sep', 'Moon_Phase',
                              'Sun_Sat_Opp', 'Moon_Ura_Opp']
        celestial_total = sum(abs(input_weights.get(f, 0)) for f in celestial_features)
        total_weight = sum(abs(w) for w in input_weights.values())
        
        print(f"\nCelestial Feature Contribution: {celestial_total/(total_weight+1e-9)*100:.1f}%")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    """Main entry point."""
    print("=" * 70)
    print("NEAT EVOLUTIONARY STRATEGY v2.0")
    print("=" * 70)
    print("\nImprovements:")
    print("  ✓ Train/Test Split (2000-2019 / 2020-2025)")
    print("  ✓ Drawdown Penalty (harsh above 20%)")
    print("  ✓ Fixed Unrealized PnL tracking")
    print("  ✓ Proper price array for accurate P&L")
    print("=" * 70)
    
    if not HAS_NEAT:
        print("\nError: neat-python required. Install with: pip install neat-python")
        return
    
    # Build world state
    world = WorldState()
    
    # Try to load data
    # First check for local CSV
    csv_paths = [
        'QQQ_data.csv',
        'c:/Users/jhana/trading_research/spy-trading/QQQ_data.csv',
        '/home/claude/QQQ_data.csv',
    ]
    
    loaded = False
    for path in csv_paths:
        if os.path.exists(path):
            loaded = world.load_data(path)
            break
    
    if not loaded:
        # Try yfinance
        loaded = world.load_data()
    
    if not loaded:
        print("Failed to load data!")
        return
    
    # Create evolver
    evolver = NEATEvolver(world)
    
    # Run evolution
    winner = evolver.run(generations=100, checkpoint_freq=25)
    
    # Validate on holdout
    validation = evolver.validate_winner()
    
    # Analyze network
    evolver.analyze_network()
    
    # Save winner
    evolver.save_winner('best_neat_trader_v2.pkl')
    
    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review holdout performance vs training")
    print("  2. If Sharpe decay < 25%, network discovered real patterns")
    print("  3. Analyze which celestial features have high weights")
    print("  4. Backport discoveries to v7.2 as features")


if __name__ == '__main__':
    main()
