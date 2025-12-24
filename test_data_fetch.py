import yfinance as yf
from datetime import datetime, timedelta

def test_download():
    days = 730
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    print(f"Attempting download: {start} to {end}")
    
    try:
        df = yf.download("SPY", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1h", progress=False)
        print(f"Download result type: {type(df)}")
        if df is not None:
            print(f"Shape: {df.shape}")
            print(df.head())
        else:
            print("df is None")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_download()
