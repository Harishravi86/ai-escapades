import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from intelligent_trader import PaperBroker, TraderAgent
import time

# Page Config
st.set_page_config(page_title="Wall Street Trader AI", layout="wide", page_icon="🤖")

# Title
st.title("🤖 Intelligent Wall Street Trader")
st.markdown("### Powered by Bulletproof Strategy v7.2")

# Initialize Session State (Persist Broker/Agent across re-runs)
if 'broker' not in st.session_state:
    st.session_state.broker = PaperBroker(initial_capital=100000.0)
    st.session_state.agent = TraderAgent(st.session_state.broker, tickers=['SPY', 'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL'])
    st.session_state.last_update = None

broker = st.session_state.broker
agent = st.session_state.agent

# Sidebar controls
with st.sidebar:
    st.header("Control Panel")
    if st.button("🔄 Refresh Market Data", type="primary"):
        with st.spinner("Fetching latest data..."):
            agent.fetch_data()
            decisions = agent.analyze_market()
            agent.execute_decisions(decisions)
            st.session_state.last_update = time.strftime("%H:%M:%S")
        st.success("Market Data Updated!")
    
    st.markdown("---")
    st.metric("Last Update", st.session_state.last_update if st.session_state.last_update else "Never")
    
    st.markdown("### Settings")
    st.checkbox("Auto-Trading Enabled", value=True, disabled=True)
    st.checkbox("Paper Trading Mode", value=True, disabled=True)

# Main Dashboard
col1, col2, col3 = st.columns(3)

# Portfolio Metrics
current_prices = {t: float(agent.market_state[t]['Close'].iloc[-1]) for t in agent.tickers if t in agent.market_state}
equity = broker.get_equity(current_prices)
pnl = (equity - 100000) / 100000

with col1:
    st.metric("Total Equity", f"${equity:,.2f}", f"{pnl:+.2%}")
with col2:
    st.metric("Cash Balance", f"${broker.cash:,.2f}")
with col3:
    st.metric("Active Positions", len(broker.positions))

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Market Scanner", "💼 Portfolio", "🧠 LLM Consultant"])

with tab1:
    st.subheader("Live Market Analysis (v7.2)")
    
    if not agent.market_state:
        st.info("Click 'Refresh Market Data' to start scanning.")
    else:
        # Create a nice dataframe for the scanner
        scan_data = []
        for ticker in agent.tickers:
            if ticker in agent.market_state:
                df = agent.market_state[ticker]
                price = float(df['Close'].iloc[-1])
                bull_prob = agent.explain_decision(ticker).split("Bull Probability:** ")[1].split(" ")[0] # Hacky parsing for display
                bear_prob = agent.explain_decision(ticker).split("Bear Probability:** ")[1].split(" ")[0]
                
                # Clean up percentages
                bull_val = float(bull_prob.strip('%')) / 100
                bear_val = float(bear_prob.strip('%')) / 100
                
                status = "NEUTRAL"
                if bull_val > 0.45: status = "🟢 BUY ZONE"
                if bear_val > 0.60: status = "🔴 DANGER"
                
                scan_data.append({
                    "Ticker": ticker,
                    "Price": f"${price:.2f}",
                    "Status": status,
                    "Bull Prob": bull_prob,
                    "Bear Prob": bear_prob
                })
        
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True)

with tab2:
    st.subheader("Active Positions")
    if not broker.positions:
        st.write("No active positions. Cash is King! 👑")
    else:
        pos_data = []
        for ticker, pos in broker.positions.items():
            current_price = current_prices.get(ticker, pos['avg_price'])
            market_val = pos['shares'] * current_price
            unrealized_pnl = (current_price - pos['avg_price']) / pos['avg_price']
            
            pos_data.append({
                "Ticker": ticker,
                "Shares": f"{pos['shares']:.2f}",
                "Avg Price": f"${pos['avg_price']:.2f}",
                "Current Price": f"${current_price:.2f}",
                "Market Value": f"${market_val:,.2f}",
                "Unrealized P&L": f"{unrealized_pnl:+.2%}"
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

with tab3:
    st.subheader("🧠 AI Consultant")
    
    selected_ticker = st.selectbox("Select Asset to Analyze", agent.tickers)
    
    if st.button("Generate Narrative Report"):
        if selected_ticker in agent.market_state:
            narrative = agent.explain_decision(selected_ticker)
            st.markdown(narrative)
            
            st.markdown("---")
            st.markdown("**📋 Prompt for External LLM:**")
            st.code(f"""
I am trading {selected_ticker}. 
Current Price: ${current_prices.get(selected_ticker, 0):.2f}.
My Strategy (Mean Reversion) says:
{narrative}

Based on this technical data, what are the key risks I should be aware of right now? 
Please analyze the macro context for {selected_ticker} as well.
            """, language="text")
        else:
            st.warning("Please refresh market data first.")
