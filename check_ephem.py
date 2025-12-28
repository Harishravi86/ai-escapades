
import ephem
print(f"Version: {ephem.__version__}")
print("Attributes:")
for x in dir(ephem):
    if 'ecl' in x.lower() or 'next' in x.lower():
        print(x)
