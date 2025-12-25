
import ephem
import numpy as np
import math
from datetime import datetime

class CelestialEngine:
    def __init__(self):
        self.enabled = True
        self.planets = {
            'Sun': ephem.Sun(),
            'Moon': ephem.Moon(),
            'Mercury': ephem.Mercury(),
            'Venus': ephem.Venus(),
            'Mars': ephem.Mars(),
            'Jupiter': ephem.Jupiter(),
            'Saturn': ephem.Saturn(),
            'Uranus': ephem.Uranus(),
            'Neptune': ephem.Neptune(),
            'Pluto': ephem.Pluto()
        }
        self._cache = {}

    # --- NEW STELLIUM RISK LOGIC ---

    def get_planet_positions(self, date_input):
        """
        Returns a sorted list of longitudes for a given date (for spread calc).
        """
        if isinstance(date_input, str):
            d = datetime.strptime(date_input, '%Y-%m-%d')
        elif isinstance(date_input, datetime):
            d = date_input
        else: # pd.Timestamp
            d = date_input.to_pydatetime()
            
        obs = ephem.Observer()
        obs.date = d.strftime('%Y/%m/%d')
        
        longitudes = []
        for name, planet in self.planets.items():
            planet.compute(obs)
            lons = np.degrees(planet.hlon) % 360
            longitudes.append(lons)
            
        return sorted(longitudes)

    def calculate_spread(self, longitudes):
        max_gap = 0
        n = len(longitudes)
        for i in range(n):
            current = longitudes[i]
            next_val = longitudes[(i + 1) % n]
            gap = next_val - current
            if gap < 0: gap += 360
            if gap > max_gap: max_gap = gap
        return 360 - max_gap

    def get_spread_regime(self, date_input):
        """
        Determines Risk Regime:
        - DANGER (170-230): Peak Volatility (~17.4%). reduce size.
        - DISPERSED (>280): Safe Zone (~13.6% Vol). full size.
        - COMPRESSED (<160): Suppression Anomaly. warning only.
        """
        lons = self.get_planet_positions(date_input)
        spread = self.calculate_spread(lons)
        
        if spread > 280:
            return 'DISPERSED', spread
        elif 170 <= spread <= 230:
            return 'DANGER_ZONE', spread
        elif spread < 160:
            return 'COMPRESSED', spread
        else:
            return 'NEUTRAL', spread

    def get_position_sizer(self, date_input):
        regime, spread = self.get_spread_regime(date_input)
        if regime == 'DANGER_ZONE': return 0.80
        return 1.0

    # --- LEGACY FEATURES FOR STRATEGY COMPATIBILITY ---

    def get_features(self, date_str):
        """
        Returns dictionary of all celestial features for the strategy.
        Now includes 'position_sizer' and 'spread_regime'.
        """
        if not self.enabled:
            return self._empty_features()
        
        # Normalize date_str key (sometimes it's datetime)
        if not isinstance(date_str, str):
            d_key = date_str.strftime('%Y-%m-%d')
        else:
            d_key = date_str

        if d_key in self._cache:
            return self._cache[d_key]
        
        try:
            # 1. Compute Legacy Aspects (Sun-Saturn, Moon-Uranus)
            # Use same observer logic
            obs = ephem.Observer()
            obs.date = d_key.replace('-', '/') # ephem likes yyyy/mm/dd
            
            positions = {}
            # Minimal set needed for legacy aspects
            for name in ['Sun', 'Moon', 'Saturn', 'Uranus']:
                body = self.planets[name]
                body.compute(obs)
                ecl = ephem.Ecliptic(body)
                positions[name] = math.degrees(ecl.lon)
            
            features = {
                'sun_opp_saturn': self._check_aspect(positions, 'Sun', 'Saturn', 180, 5),
                'moon_opp_uranus': self._check_aspect(positions, 'Moon', 'Uranus', 180, 8),
                'sun_saturn_sep': self._get_sep(positions, 'Sun', 'Saturn'),
                'moon_uranus_sep': self._get_sep(positions, 'Moon', 'Uranus'),
                'is_lunar_window': self._is_new_moon_window(positions['Sun'], positions['Moon'], 25),
            }
            
            # 2. Add New Stellium features
            regime, spread = self.get_spread_regime(d_key)
            features['spread_val'] = spread
            features['spread_regime'] = regime
            features['position_sizer'] = self.get_position_sizer(d_key)
            
            self._cache[d_key] = features
            return features
            
        except Exception as e:
            # print(f"Celestial Error: {e}") 
            return self._empty_features()

    def _check_aspect(self, positions, p1, p2, target, orb):
        diff = abs(positions[p1] - positions[p2])
        if diff > 180: diff = 360 - diff
        return abs(diff - target) <= orb

    def _get_sep(self, positions, p1, p2):
        diff = abs(positions[p1] - positions[p2])
        if diff > 180: diff = 360 - diff
        return diff

    def _is_new_moon_window(self, sun_pos, moon_pos, threshold_deg):
        diff = moon_pos - sun_pos
        while diff < 0: diff += 360
        while diff >= 360: diff -= 360
        return diff < threshold_deg or diff > (360 - threshold_deg)

    def _empty_features(self):
        return {
            'sun_opp_saturn': False, 
            'moon_opp_uranus': False,
            'sun_saturn_sep': 0.0,
            'moon_uranus_sep': 0.0,
            'is_lunar_window': False,
            'position_sizer': 1.0,
            'spread_regime': 'NEUTRAL'
        }

if __name__ == "__main__":
    # Test Integration
    engine = CelestialEngine()
    print("Testing Legacy + New Logic Integration:")
    
    dates = ['2008-09-15', '2020-03-20']
    for d in dates:
        f = engine.get_features(d)
        print(f"Date: {d}")
        print(f"  Regime: {f['spread_regime']} ({f['spread_val']:.1f}°)")
        print(f"  Sizer:  {f['position_sizer']}")
        print(f"  Legacy Aspect (Sun-Saturn): {f['sun_saturn_sep']:.1f}°")
