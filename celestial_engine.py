
import ephem
import numpy as np
import math
from datetime import datetime, timedelta

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
        # Pre-calculate eclipses
        self.solar_eclipses, self.lunar_eclipses = self._cache_eclipses_robust()

    def _cache_eclipses_robust(self):
        """Pre-calculate eclipses 2000-2030 (New Moon + Lat < 1.6)"""
        solar = []
        lunar = []
        
        # Solar
        d = ephem.Date('2000-01-01')
        end_d = ephem.Date('2030-12-31')
        while d < end_d:
            try:
                nm = ephem.next_new_moon(d)
                if nm > end_d: break
                m = ephem.Moon(); m.compute(nm)
                if abs(ephem.Ecliptic(m).lat * 180/3.14159) < 1.6:
                    solar.append(nm.datetime().date())
                d = ephem.Date(nm + 1)
            except: break
            
        # Lunar
        d = ephem.Date('2000-01-01')
        while d < end_d:
            try:
                fm = ephem.next_full_moon(d)
                if fm > end_d: break
                m = ephem.Moon(); m.compute(fm)
                if abs(ephem.Ecliptic(m).lat * 180/3.14159) < 1.6:
                    lunar.append(fm.datetime().date())
                d = ephem.Date(fm + 1)
            except: break
            
        return solar, lunar

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

    def get_mercury_retrograde(self, date_input):
        """
        Check if Mercury is in retrograde (apparent backward motion).
        Logic: Longitude today < Longitude yesterday
        """
        if isinstance(date_input, str):
            d = datetime.strptime(date_input, '%Y-%m-%d')
        elif isinstance(date_input, datetime):
            d = date_input
        else:
            d = date_input.to_pydatetime()

        # Get position for today
        obs = ephem.Observer()
        obs.date = d.strftime('%Y/%m/%d')
        mercury = self.planets['Mercury']
        mercury.compute(obs)
        lon_today = np.degrees(ephem.Ecliptic(mercury).lon) % 360

        # Get position for yesterday
        # Fix datetime timedelta issue - just subtract 1 day directly from datetime object
        # Re-doing date logic safely:
        prev_date = d - timedelta(days=1)
        obs.date = prev_date.strftime('%Y/%m/%d')
        mercury.compute(obs)
        lon_yesterday = np.degrees(ephem.Ecliptic(mercury).lon) % 360

        # Retrograde if longitude decreases
        # Handle wrap-around (360 -> 0)
        diff = lon_today - lon_yesterday
        
        # If diff is negative (e.g. 100 -> 99), it's retrograde.
        # Exception: Crossing 0 (e.g. 1 -> 359 = +358, not retrograde. 359 -> 1 = -358, not retrograde)
        # However, Mercury moves ~1-2 degrees per day.
        # If diff is huge positive (e.g. 1 -> 359, diff +358), that's actually -2 movement (Retrograde wrapping)
        # If diff is huge negative (e.g. 359 -> 1, diff -358), that's actually +2 movement (Direct wrapping)
        
        if diff < -300: # Crossed 0 forward (359 -> 1)
            return False 
        if diff > 300: # Crossed 0 backward (1 -> 359)
            return True
            
        return diff < 0

    def get_saturn_retrograde(self, date_input):
        """
        Check if Saturn is in retrograde.
        Logic: Geocentric Ecliptic Longitude today < yesterday
        """
        if isinstance(date_input, str):
            d = datetime.strptime(date_input, '%Y-%m-%d')
        elif isinstance(date_input, datetime):
            d = date_input
        else:
            d = date_input.to_pydatetime()

        obs = ephem.Observer()
        obs.date = d.strftime('%Y/%m/%d')
        saturn = self.planets['Saturn']
        saturn.compute(obs)
        lon_today = np.degrees(ephem.Ecliptic(saturn).lon) % 360

        prev_date = d - timedelta(days=1)
        obs.date = prev_date.strftime('%Y/%m/%d')
        saturn.compute(obs)
        lon_yesterday = np.degrees(ephem.Ecliptic(saturn).lon) % 360

        diff = lon_today - lon_yesterday
        
        if diff < -300: return False 
        if diff > 300: return True
            
        return diff < 0

    def get_dual_separation(self, date_input, p1_name, p2_name):
        """Returns separation between two planets (0-180 degrees)."""
        if isinstance(date_input, str):
            d = datetime.strptime(date_input, '%Y-%m-%d')
        elif isinstance(date_input, datetime):
            d = date_input
        else:
            d = date_input.to_pydatetime()
            
        obs = ephem.Observer()
        obs.date = d.strftime('%Y/%m/%d')
        
        p1 = self.planets[p1_name]
        p2 = self.planets[p2_name]
        p1.compute(obs)
        p2.compute(obs)
        
        l1 = np.degrees(ephem.Ecliptic(p1).lon)
        l2 = np.degrees(ephem.Ecliptic(p2).lon)
        
        return self._calc_sep(l1, l2)

    def get_planet_sign(self, date_input, planet_name):
        """Returns Zodiac Sign Index (0=Aries, 11=Pisces)."""
        if isinstance(date_input, str):
            d = datetime.strptime(date_input, '%Y-%m-%d')
        elif isinstance(date_input, datetime):
            d = date_input
        else:
            d = date_input.to_pydatetime()
            
        obs = ephem.Observer()
        obs.date = d.strftime('%Y/%m/%d')
        p = self.planets[planet_name]
        p.compute(obs)
        lon = np.degrees(ephem.Ecliptic(p).lon)
        
        return int(lon // 30)

    def get_saturn_dignity(self, date_input):
        """
        Vedic Dignity Score (Sidereal Zodiac).
        +1 = Exalted (Libra)
        -1 = Debilitated (Aries)
         0 = Neutral
        """
        if isinstance(date_input, str):
            d = datetime.strptime(date_input, '%Y-%m-%d')
        elif isinstance(date_input, datetime):
            d = date_input
        else:
            d = date_input.to_pydatetime()
            
        obs = ephem.Observer()
        obs.date = d.strftime('%Y/%m/%d')
        p = self.planets['Saturn']
        p.compute(obs)
        
        trop_lon = np.degrees(ephem.Ecliptic(p).lon)
        sid_lon = (trop_lon - 24.0) % 360 # Lahiri approx correction
        
        # Sidereal Libra: 180-210
        if 180 <= sid_lon < 210: return 1
        # Sidereal Aries: 0-30
        if 0 <= sid_lon < 30: return -1
        
        return 0

    def _calc_sep(self, l1, l2):
        diff = abs(l1 - l2)
        if diff > 180: diff = 360 - diff
        return diff

    def get_eclipse_regime(self, date_input):
        """
        Returns Eclipse Regime:
         1: Within 3 days of Solar Eclipse (Bullish)
        -1: Within 3 days of Lunar Eclipse (Bearish)
         0: None
        """
        if isinstance(date_input, str):
            d_date = datetime.strptime(date_input, '%Y-%m-%d').date()
        elif isinstance(date_input, datetime):
            d_date = date_input.date()
        else:
            d_date = date_input.date() # pd.Timestamp
            
        # Check Solar
        for eclipse in self.solar_eclipses:
            if abs((d_date - eclipse).days) <= 3:
                return 1
                
        # Check Lunar
        for eclipse in self.lunar_eclipses:
            if abs((d_date - eclipse).days) <= 3:
                return -1
                
        return 0

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
            features['is_mercury_retrograde'] = self.get_mercury_retrograde(d_key)
            
            # 3. Add Research Features (Saturn)
            features['saturn_retrograde'] = self.get_saturn_retrograde(d_key)
            features['saturn_mars_sep'] = self.get_dual_separation(d_key, 'Saturn', 'Mars')
            features['saturn_jupiter_sep'] = self.get_dual_separation(d_key, 'Saturn', 'Jupiter')
            features['saturn_jupiter_sep'] = self.get_dual_separation(d_key, 'Saturn', 'Jupiter')
            features['saturn_dignity'] = self.get_saturn_dignity(d_key)
            features['eclipse_regime'] = self.get_eclipse_regime(d_key)
            
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
            'spread_regime': 'NEUTRAL',
            'is_mercury_retrograde': False,
            'saturn_retrograde': False,
            'saturn_mars_sep': 0.0,
            'saturn_jupiter_sep': 0.0,
            'saturn_mars_sep': 0.0,
            'saturn_jupiter_sep': 0.0,
            'saturn_dignity': 0, # Neutral default
            'eclipse_regime': 0
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
