
import pandas as pd
import numpy as np

def analyze_impact():
    # Load trade log
    try:
        trades = pd.read_csv("v7_2_trades.csv")
    except FileNotFoundError:
        print("Error: v7_2_trades.csv not found.")
        return

    # Normalize columns to lowercase for safety
    trades.columns = [c.lower() for c in trades.columns]
    
    # Check for 'size' column
    if 'size' not in trades.columns:
        print("Error: 'size' column missing in trade log. Available:", trades.columns)
        return

    print("--- STELLIUM RISK IMPACT ANALYSIS ---")
    print(f"Total Trades: {len(trades)}")

    # Filter for "Risk Reduced" trades (Size approx 0.8)
    # Floating point tolerance
    reduced = trades[ (trades['size'] > 0.75) & (trades['size'] < 0.85) ].copy()
    
    print(f"Trades in DANGER ZONE (Size 0.8x): {len(reduced)}")
    
    if len(reduced) == 0:
        print("No risk-reduced trades found. Check dates or logic.")
        return

    # Calculate PnL Impact
    # PnL = (Exit - Entry) * Shares
    # We want to see the difference between Realized PnL and "What If Full Size"
    
    # Reconstruct investment amount roughly if not present
    # Usually strategy starts with 100k. 
    # Let's focus on % Return difference to be scale-invariant
    
    reduced['Return_Pct'] = (reduced['Exit Price'] - reduced['Entry Price']) / reduced['Entry Price']
    
    # Avg Return of these trades
    avg_return = reduced['Return_Pct'].mean()
    print(f"Avg Return of Reduced Trades: {avg_return:.2%}")
    
    # Calculate PnL Saved/Lost on a $10,000 basis per trade
    base_bet = 10000 
    
    reduced['Realized_PnL'] = base_bet * 0.8 * reduced['Return_Pct']
    reduced['Hypothetical_PnL'] = base_bet * 1.0 * reduced['Return_Pct']
    reduced['Saved'] = reduced['Realized_PnL'] - reduced['Hypothetical_PnL']
    
    total_saved = reduced['Saved'].sum()
    
    print("\n--- PERFORMANCE IN DANGER ZONE ---\n")
    print(reduced[['Entry Date', 'Entry Price', 'Exit Price', 'Return_Pct', 'Saved']].head(10).to_string())
    
    print("\n-------------------------------------------")
    if total_saved > 0:
        print(f"SUCCESS: Risk Module SAVED an estimated ${total_saved:.2f} (per $10k base bet)")
        print("Interpretation: The module correctly reduced size during losing trades.")
    else:
        print(f"COST: Risk Module COST an estimated ${abs(total_saved):.2f} (per $10k base bet)")
        print("Interpretation: The module reduced size during winning trades (Opportunity Cost).")
        
    # Check specific Lehman Date
    lehman = reduced[reduced['Entry Date'].str.contains('2008-09') | reduced['Entry Date'].str.contains('2008-10')]
    if not lehman.empty:
        print("\n--- LEHMAN/2008 CRISIS TRADES ---")
        print(lehman[['Entry Date', 'Return_Pct', 'Saved']].to_string())

if __name__ == "__main__":
    analyze_impact()
