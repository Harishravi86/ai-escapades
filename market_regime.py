
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, date
import calendar

class MarketRegimeEngine:
    """
    v8.0: Seasonality & Volatility Regime Logic
    
    Identified Statistical Edges (2000-2025):
    1. Turn of Month (TOM): +4x Baseline (Last 3 days + First 2 days)
    2. Halloween Effect: +2x Baseline (Nov-Apr)
    3. September Effect: Negative Bias
    4. VIX Term Structure: Backwardation = Fear (> 1.0)
    """
    
    def __init__(self):
        pass

    def get_features(self, date_input, vix_spot=None, vix_3m=None) -> Dict:
        """Calculate seasonality and regime features."""
        dt = self._parse_date(date_input)
        
        return {
            'is_turn_of_month': int(self.is_turn_of_month(dt)),
            'seasonality_regime': self.get_seasonality_regime(dt), # 1=Winter, -1=Sept, 0=Summer
            'vix_term_structure': self.get_vix_regime(vix_spot, vix_3m)
        }

    def _parse_date(self, date_input) -> date:
        if isinstance(date_input, str):
            try:
                return datetime.strptime(date_input, '%Y-%m-%d').date()
            except:
                return datetime.strptime(date_input, '%Y/%m/%d').date()
        elif isinstance(date_input, datetime):
            return date_input.date()
        else: # pd.Timestamp or date
            if hasattr(date_input, 'date'):
                return date_input.date()
            return date_input

    def is_turn_of_month(self, d: date) -> bool:
        """
        Buying window: Last 3 trading days of month + First 2 trading days of next month.
        Since we don't have a trading calendar here easily, we approximate with calendar days:
        - Day <= 2
        - Day >= DaysInMonth - 3
        """
        day = d.day
        try:
             # calendar.monthrange returns (first_weekday, number_of_days)
            _, days_in_month = calendar.monthrange(d.year, d.month)
            
            if day <= 2:
                return True
            if day >= days_in_month - 3: 
                return True
            return False
            
        except Exception:
            return False

    def get_seasonality_regime(self, d: date) -> int:
        """
        1: Halloween / Winter (Nov-Apr) - Bullish Bias
        -1: September - Bearish Bias
        0: Summer (May-Oct, excluding Sept) - Weak/Neutral
        """
        m = d.month
        
        if m in [11, 12, 1, 2, 3, 4]:
            return 1 # Bullish Season
        elif m == 9:
            return -1 # September Effect
        else:
            return 0 # Neutral/Summer

    def get_vix_regime(self, spot, term3m) -> int:
        """
        VIX Term Structure: Spot / 3M
        > 1.05: Backwardation (Fear) -> 1
        < 0.90: Contango (Calm) -> -1
        Else: Normal -> 0
        """
        if spot is None or term3m is None or pd.isna(spot) or pd.isna(term3m) or term3m == 0:
            return 0
            
        ratio = spot / term3m
        
        if ratio > 1.05:
            return 1 # Fear / Backwardation
        elif ratio < 0.90:
            return -1 # Complacency
        else:
            return 0
