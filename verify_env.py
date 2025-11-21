import sys
import os

print("Checking environment...")
try:
    import dotenv
    print("✅ python-dotenv is installed.")
except ImportError:
    print("❌ python-dotenv is NOT installed.")
    sys.exit(1)

print("\nChecking daily_bot import...")
try:
    sys.path.append(os.path.join(os.getcwd(), 'spx_optimizer'))
    import daily_bot
    print("✅ daily_bot imported successfully.")
except Exception as e:
    print(f"❌ Failed to import daily_bot: {e}")
    sys.exit(1)

print("\nVerification successful!")
