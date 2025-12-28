
import ephem
import math

known_eclipse = '2024-04-08'
print(f"Checking Moon Latitude on {known_eclipse} (Total Solar Eclipse)...")

m = ephem.Moon()
s = ephem.Sun()
date = ephem.Date(known_eclipse)
m.compute(date)
s.compute(date)

# Ecliptic Lat
lat_rad = m.ecl_lat
lat_deg = math.degrees(lat_rad)

# Separation
sep = ephem.separation(m, s)
sep_deg = math.degrees(sep)

print(f"Moon Ecliptic Lat (rad): {lat_rad}")
print(f"Moon Ecliptic Lat (deg): {lat_deg}")
print(f"Sun-Moon Separation (deg): {sep_deg}")

# Try different attribute
try:
    print(f"Moon.lat (Declination?): {m.lat}")
except:
    pass

# Try next_new_moon from that date
nm = ephem.next_new_moon(date)
print(f"Next New Moon from date: {nm} ({nm.datetime()})")
m.compute(nm)
print(f"Moon Lat at New Moon: {math.degrees(m.ecl_lat)}")
