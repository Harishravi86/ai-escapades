
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# ML imports
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# For celestial calculations
try:
    import ephem
    HAS_EPHEM = True
except ImportError:
    HAS_EPHEM = False
    print("Warning: ephem not installed. Celestial features will be disabled.")

class CelestialCalculator:
    """Calculate celestial aspects for exit decisions."""
    
    def __init__(self):
        if not HAS_EPHEM:
            raise ImportError("ephem required for celestial calculations")
        self.obs = ephem.Observer()
    
    def get_moon_uranus_separation(self, date_str: str) -> float:
        """Get Moon-Uranus ecliptic longitude separation in degrees."""
        # Add time 20:00 UTC (market close-ish) to avoid midnight edge cases
        dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(hours=20)
        self.obs.date = dt
        
        moon = ephem.Moon()
        uranus = ephem.Uranus()
        moon.compute(self.obs)
        uranus.compute(self.obs)
        
        m_lon = math.degrees(ephem.Ecliptic(moon).lon)
        u_lon = math.degrees(ephem.Ecliptic(uranus).lon)
        
        sep = abs(m_lon - u_lon)
        if sep > 180:
            sep = 360 - sep
        return sep
    
    def get_sun_saturn_separation(self, date_str: str) -> float:
        """Get Sun-Saturn ecliptic longitude separation in degrees."""
        dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(hours=20)
        self.obs.date = dt
        
        sun = ephem.Sun()
        saturn = ephem.Saturn()
        sun.compute(self.obs)
        saturn.compute(self.obs)
        
        s_lon = math.degrees(ephem.Ecliptic(sun).lon)
        sat_lon = math.degrees(ephem.Ecliptic(saturn).lon)
        
        sep = abs(s_lon - sat_lon)
        if sep > 180:
            sep = 360 - sep
        return sep
    
    def get_moon_phase(self, date_str: str) -> float:
        """Get moon phase as percentage (0=new, 100=full)."""
        dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(hours=20)
        self.obs.date = dt
        moon = ephem.Moon()
        moon.compute(self.obs)
        return moon.phase
    
    def is_moon_uranus_opposition(self, date_str: str, orb: float = 5.0) -> bool:
        """Check if Moon-Uranus opposition is active (within orb degrees of 180)."""
        sep = self.get_moon_uranus_separation(date_str)
        return abs(sep - 180) <= orb
    
    def is_sun_saturn_opposition(self, date_str: str, orb: float = 5.0) -> bool:
        """Check if Sun-Saturn opposition is active."""
        sep = self.get_sun_saturn_separation(date_str)
        return abs(sep - 180) <= orb
    
    def days_to_next_sun_saturn_opposition(self, date_str: str, max_days: int = 180) -> int:
        """Calculate days until next Sun-Saturn opposition."""
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        # Simple/Naive check forward
        for i in range(1, max_days + 1):
            check_date = (dt + timedelta(days=i)).strftime('%Y-%m-%d')
            # Use wider orb for "approaching" detection if needed, but keeping tight for exact
            if self.is_sun_saturn_opposition(check_date, orb=2.0):
                return i
        return max_days


class ExitOptimizer:
    """Main class for optimizing exit decisions."""
    
    def __init__(self, trade_log_path: str = None, trade_log_df: pd.DataFrame = None):
        """
        Initialize with either a path to CSV or a DataFrame.
        """
        if trade_log_df is not None:
            self.trades = trade_log_df.copy()
        elif trade_log_path:
            self.trades = pd.read_csv(trade_log_path)
        else:
            raise ValueError("Must provide either trade_log_path or trade_log_df")
        
        # Parse dates
        self.trades['entry_date'] = pd.to_datetime(self.trades['entry_date'])
        self.trades['exit_date'] = pd.to_datetime(self.trades['exit_date'])
        
        # Initialize celestial calculator
        self.celestial = CelestialCalculator() if HAS_EPHEM else None
        
        # Price data cache
        self.prices = None
        self.vix = None
    
    def load_price_data(self, symbol: str = 'QQQ', start: str = '1999-01-01'):
        """Load price data from local CSVs."""
        print(f"Loading {symbol} price data from local CSV...")
        try:
            # Assuming standard structure '../QQQ_data.csv' relative to script in ai-escapades
            # Or absolute path based on known location
            base_dir = r"c:\Users\jhana\trading_research\spy-trading"
            qqq_path = os.path.join(base_dir, "QQQ_data.csv")
            vix_path = os.path.join(base_dir, "VIX_data.csv")
            
            if not os.path.exists(qqq_path):
                 print(f"Error: {qqq_path} not found.")
                 return

            self.prices = pd.read_csv(qqq_path, index_col=0)
            self.prices.index = pd.to_datetime(self.prices.index, utc=True).tz_localize(None)
            
            self.vix = pd.read_csv(vix_path, index_col=0)
            self.vix.index = pd.to_datetime(self.vix.index, utc=True).tz_localize(None)
            
            # Map columns
            self.prices.columns = [c.capitalize() for c in self.prices.columns]
            self.vix.columns = [c.capitalize() for c in self.vix.columns]
            
            self._add_technical_indicators()
            print(f"Loaded {len(self.prices)} price bars, {len(self.vix)} VIX bars")
            
        except Exception as e:
            print(f"Failed to load local data: {e}")
            import traceback
            traceback.print_exc()
            
    def _add_technical_indicators(self):
        """Add technical indicators to price data."""
        df = self.prices
        
        # Moving averages
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # RSI(2) for oversold detection
        gain2 = (delta.where(delta > 0, 0)).rolling(2).mean()
        loss2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs2 = gain2 / loss2
        df['RSI_2'] = 100 - (100 / (1 + rs2))
        
        # Bollinger Bands
        df['BB_mid'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
        df['BB_pctb'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Daily returns
        df['daily_return'] = df['Close'].pct_change()
        
        # Momentum
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        df['volatility_20d'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
        
        # Position relative to MAs
        df['above_MA20'] = (df['Close'] > df['MA_20']).astype(int)
        df['above_MA50'] = (df['Close'] > df['MA_50']).astype(int)
        
        self.prices = df
    
    def get_price_data_for_date(self, date: datetime) -> dict:
        """Get price and technical data for a specific date."""
        if self.prices is None:
            return {}
        
        # Find the closest trading day
        try:
            # Use 'pad' to get previous close if date is non-trading
            idx = self.prices.index.get_indexer([date], method='pad')[0]
            if idx < 0:
                return {}
            row = self.prices.iloc[idx]
            return row.to_dict()
        except:
            return {}
    
    def get_vix_for_date(self, date: datetime) -> float:
        """Get VIX level for a specific date."""
        if self.vix is None:
            return np.nan
        
        try:
            idx = self.vix.index.get_indexer([date], method='pad')[0]
            if idx < 0:
                return np.nan
            return self.vix.iloc[idx]['Close']
        except:
            return np.nan
    
    def calculate_forward_return(self, exit_date: datetime, horizon: int = 20) -> float:
        """Calculate forward return from exit date over horizon trading days."""
        if self.prices is None:
            return np.nan
        
        try:
            idx = self.prices.index.get_indexer([exit_date], method='pad')[0]
            if idx < 0: return np.nan
            
            # Robust boundary check
            end_idx = min(idx + horizon, len(self.prices) - 1)
            
            exit_price = self.prices.iloc[idx]['Close']
            future_price = self.prices.iloc[end_idx]['Close']
            return (future_price - exit_price) / exit_price
        except:
            return np.nan
    
    def calculate_max_forward_return(self, exit_date: datetime, horizon: int = 20) -> float:
        """Calculate maximum forward return achievable over horizon."""
        if self.prices is None:
            return np.nan
        
        try:
            idx = self.prices.index.get_indexer([exit_date], method='pad')[0]
            if idx < 0: return np.nan
            
            end_idx = min(idx + horizon + 1, len(self.prices))
            
            exit_price = self.prices.iloc[idx]['Close']
            future_prices = self.prices.iloc[idx:end_idx]['Close']
            max_price = future_prices.max()
            return (max_price - exit_price) / exit_price
        except:
            return np.nan
    
    def extract_conviction_from_reason(self, reason: str) -> float:
        """Extract conviction percentage from exit reason string."""
        # e.g., "BEAR_TWIN (94%)" -> 0.94
        if isinstance(reason, str) and 'BEAR_TWIN' in reason and '(' in reason:
            try:
                pct = reason.split('(')[1].split('%')[0]
                return float(pct) / 100
            except:
                pass
        return np.nan
    
    def generate_exit_decisions(self) -> pd.DataFrame:
        """
        Generate one row per BEAR_TWIN trade at the exit decision point.
        """
        decisions = []
        
        for idx, trade in self.trades.iterrows():
            reason = trade['reason']
            
            # Only analyze BEAR_TWIN exits - these are the decision points
            if not isinstance(reason, str) or not reason.startswith('BEAR_TWIN'):
                continue
            
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            trade_return = trade['return']
            
            # Get price/technical data at exit
            exit_data = self.get_price_data_for_date(exit_date)
            if not exit_data:
                continue
            
            # Calculate forward returns (what if we'd held?)
            forward_5d = self.calculate_forward_return(exit_date, 5)
            forward_10d = self.calculate_forward_return(exit_date, 10)
            forward_20d = self.calculate_forward_return(exit_date, 20)
            max_forward_20d = self.calculate_max_forward_return(exit_date, 20)
            
            if pd.isna(forward_20d):
                # If we can't calculate forward return (e.g. end of data), skip
                continue
            
            # Extract features
            features = {
                # Trade state
                'trade_return': trade_return,
                'days_held': (exit_date - entry_date).days,
                'exit_conviction': self.extract_conviction_from_reason(reason),
                
                # Market state at exit
                'rsi_14': exit_data.get('RSI_14', np.nan),
                'rsi_2': exit_data.get('RSI_2', np.nan),
                'bb_pctb': exit_data.get('BB_pctb', np.nan),
                'above_ma20': exit_data.get('above_MA20', np.nan),
                'above_ma50': exit_data.get('above_MA50', np.nan),
                'daily_return': exit_data.get('daily_return', np.nan),
                'momentum_5d': exit_data.get('momentum_5d', np.nan),
                'momentum_20d': exit_data.get('momentum_20d', np.nan),
                'volatility_20d': exit_data.get('volatility_20d', np.nan),
                
                # VIX state
                'vix_level': self.get_vix_for_date(exit_date),
                
                # Entry conditions (was this a panic entry?)
                'entry_rsi_14': self.get_price_data_for_date(entry_date).get('RSI_14', np.nan),
                'entry_rsi_2': self.get_price_data_for_date(entry_date).get('RSI_2', np.nan),
                'entry_vix': self.get_vix_for_date(entry_date),
            }
            
            # Celestial features (if available)
            if self.celestial:
                exit_date_str = exit_date.strftime('%Y-%m-%d')
                entry_date_str = entry_date.strftime('%Y-%m-%d')
                
                features.update({
                    # Exit date celestial
                    'moon_uranus_sep': self.celestial.get_moon_uranus_separation(exit_date_str),
                    'sun_saturn_sep': self.celestial.get_sun_saturn_separation(exit_date_str),
                    'moon_phase': self.celestial.get_moon_phase(exit_date_str),
                    'moon_uranus_opp_active': int(self.celestial.is_moon_uranus_opposition(exit_date_str)),
                    'sun_saturn_opp_active': int(self.celestial.is_sun_saturn_opposition(exit_date_str)),
                    'days_to_sun_saturn_opp': self.celestial.days_to_next_sun_saturn_opposition(exit_date_str),
                    
                    # Entry date celestial
                    'entry_moon_uranus_opp': int(self.celestial.is_moon_uranus_opposition(entry_date_str)),
                })
            
            # Target: Should we have held?
            # Holding is better if forward return exceeds current return + hurdle
            # Or simpler: forward_return_20d > 2%.
            # NOTE: trade_return is already locked. New return is vs exit price.
            # So forward_20d > 0.02 means price went up 2% more after exit.
            hurdle = 0.02
            
            features['forward_5d'] = forward_5d
            features['forward_10d'] = forward_10d
            features['forward_20d'] = forward_20d
            features['max_forward_20d'] = max_forward_20d
            
            # Target: 1 if holding would have been significantly better
            features['target_should_hold'] = int(forward_20d > hurdle)
            
            features['target_missed_upside'] = int(max_forward_20d > 0.05) # Missed a 5% move?
            
            # Metadata
            features['entry_date'] = entry_date
            features['exit_date'] = exit_date
            features['exit_reason'] = reason
            
            decisions.append(features)
        
        return pd.DataFrame(decisions)
    
    def train_exit_model(self, decisions_df: pd.DataFrame, 
                         target_col: str = 'target_should_hold',
                         test_year_start: int = 2020):
        """
        Train XGBoost classifier to predict when to hold past BEAR_TWIN.
        """
        exclude_cols = ['entry_date', 'exit_date', 'exit_reason', 
                        'target_should_hold', 'target_missed_upside',
                        'forward_5d', 'forward_10d', 'forward_20d', 
                        'max_forward_20d', 'hold_advantage_20d']
        
        feature_cols = [c for c in decisions_df.columns if c not in exclude_cols]
        
        # Split by time
        train_mask = decisions_df['exit_date'].dt.year < test_year_start
        test_mask = decisions_df['exit_date'].dt.year >= test_year_start
        
        X_train = decisions_df.loc[train_mask, feature_cols].copy()
        y_train = decisions_df.loc[train_mask, target_col].copy()
        X_test = decisions_df.loc[test_mask, feature_cols].copy()
        y_test = decisions_df.loc[test_mask, target_col].copy()
        
        print(f"\nTraining set: {len(X_train)} samples ({train_mask.sum()} BEAR_TWIN exits)")
        print(f"Test set: {len(X_test)} samples ({test_mask.sum()} BEAR_TWIN exits)")
        print(f"Train positive rate: {y_train.mean():.1%}")
        print(f"Test positive rate: {y_test.mean():.1%}")
        
        if y_train.sum() == 0 or len(np.unique(y_train)) < 2:
            print("Error: Training set has single class. Cannot train.")
            return {}

        scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=['TAKE_EXIT', 'HOLD']))
        
        auc = 0
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC: {auc:.3f}")
        
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE (Top 15)")
        print("="*60)
        for i, row in importance.head(15).iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"{row['feature']:30s} {row['importance']:.4f} {bar}")
            
        celestial_features = ['moon_uranus_sep', 'sun_saturn_sep', 'moon_phase',
                              'moon_uranus_opp_active', 'sun_saturn_opp_active',
                              'days_to_sun_saturn_opp', 'entry_moon_uranus_opp']
        celestial_importance = importance[importance['feature'].isin(celestial_features)]
        
        print("\n" + "="*60)
        print("CELESTIAL FEATURE IMPORTANCE")
        print("="*60)
        if len(celestial_importance) > 0:
            total_celestial = celestial_importance['importance'].sum()
            print(f"Total celestial contribution: {total_celestial:.1%}")
            for _, row in celestial_importance.iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")
        
        results = {
            'model': model,
            'importance': importance,
            'auc': auc,
            'test_predictions': pd.DataFrame({
                'exit_date': decisions_df.loc[test_mask, 'exit_date'],
                'trade_return': decisions_df.loc[test_mask, 'trade_return'],
                'forward_20d': decisions_df.loc[test_mask, 'forward_20d'],
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
        }
        
        return results
    
    def analyze_learned_rules(self, results: dict, decisions_df: pd.DataFrame):
        if not results: return
        
        print("\n" + "="*60)
        print("LEARNED RULES ANALYSIS")
        print("="*60)
        
        importance = results['importance']
        print("\nModel's top 5 decision drivers:")
        for i, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        test_df = results['test_predictions']
        hardcoded_hold = (test_df['trade_return'] > 0.05).astype(int)
        model_hold = test_df['y_pred']
        
        agreement = (hardcoded_hold == model_hold).mean()
        print(f"\nAgreement with 'unrealized > 5%' rule: {agreement:.1%}")
        
        disagree_mask = hardcoded_hold != model_hold
        if disagree_mask.any():
            print(f"\nDisagreements: {disagree_mask.sum()} trades")
            
            # Model says HOLD, Rule says EXIT
            model_hold_extra = (model_hold == 1) & (hardcoded_hold == 0)
            if model_hold_extra.any():
                avg = test_df[model_hold_extra]['forward_20d'].mean()
                print(f"  Model HOLDS when Rule EXITS ({model_hold_extra.sum()}): Avg Fwd 20d: {avg:.1%}")
            
            # Model says EXIT, Rule says HOLD
            model_exit_extra = (model_hold == 0) & (hardcoded_hold == 1)
            if model_exit_extra.any():
                avg = test_df[model_exit_extra]['forward_20d'].mean()
                print(f"  Model EXITS when Rule HOLDS ({model_exit_extra.sum()}): Avg Fwd 20d: {avg:.1%}")

def main():
    trades_path = 'v7_2_trades.md'
    if not os.path.exists(trades_path):
         print(f"Trade file {trades_path} not found.")
         return

    print("="*60)
    print("EXIT STRATEGY OPTIMIZER (REAL DATA)")
    print("="*60)
    
    optimizer = ExitOptimizer(trade_log_path=trades_path)
    
    # Load Real Price Data
    print("\n" + "-"*60)
    optimizer.load_price_data('QQQ')
    
    print("\n" + "-"*60)
    print("Generating exit decision features...")
    decisions = optimizer.generate_exit_decisions()
    print(f"Generated {len(decisions)} BEAR_TWIN exit decision points")
    
    # Train
    print("\n" + "-"*60)
    print("Training exit decision model...")
    results = optimizer.train_exit_model(decisions, test_year_start=2020)
    
    # Analyze
    optimizer.analyze_learned_rules(results, decisions)

if __name__ == '__main__':
    main()
