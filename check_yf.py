import yfinance as yf
import pandas as pd

print("Pandas version:", pd.__version__)
print("YFinance version:", yf.__version__)

df = yf.download("SPY", period="5d", interval="1h", progress=False)
print("\nColumns:", df.columns)
print("Type of df['Close']:", type(df['Close']))
print("df['Close'] shape:", df['Close'].shape)
print("df['Close'] values sample:", df['Close'].values[:5])

close = df["Close"]
if isinstance(close, pd.DataFrame):
    print("Close is DataFrame. Selecting first column...")
    close = close.iloc[:, 0]

print("Final Close Type:", type(close))
print("Final Close Shape:", close.shape)
print("Final Close Values[0] type:", type(close.values[0]))
