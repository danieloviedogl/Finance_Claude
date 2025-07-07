import streamlit as st

# --- Contenido Específico de la Página de Inicio ---
st.header("Stock Analysis Pro", anchor="introduction-header")

# En introduction.py, reemplaza el st.markdown existente con esto:

st.markdown("""
    **Welcome to the Stock Analysis & Portfolio Optimization Dashboard!**

    This application is designed to empower you with data-driven insights for making more informed investment decisions. Whether you want to dissect a single stock or build and test a diversified portfolio, you'll find the tools you need here.

    ---

    ### **What can you do? :)**

    *   **📈 Stock Analysis Page:**
        *   Dive deep into an individual stock by entering its ticker symbol.
        *   Get a comprehensive suite of metrics:
            *   Technical indicators and signals.
            *   Fundamental data and valuation ratios.
            *   Risk analysis relative to a benchmark (like `SPY`).
            *   Time series forecasting (ARIMA/GARCH).
            *   Monte Carlo simulations for future price projections.
            *   Analyst sentiment summary.
        *   Receive a data-driven **investment recommendation** (Strong Buy, Buy, Hold, Reduce, Sell) based on a weighted score across all analyses.

    *   **📊 Portfolio Analysis Page:**
        *   Manage your custom list of stocks (your watchlist or current holdings).
        *   Calculate the **optimal portfolio allocation** based on Modern Portfolio Theory (MPT) to maximize the risk-adjusted return (Sharpe Ratio).
        *   Perform **backtesting** on the optimized portfolio to see how the strategy would have performed historically against a benchmark.

    ---

    ### **Getting Started**

    1.  **(Optional) Configure Global Settings:** Adjust the *Risk-Free Rate* and *Benchmark Ticker* below if needed. These settings influence calculations like the Sharpe Ratio and risk metrics.
    2.  **Navigate using the sidebar:**
        *   Select **📈 Stock Analysis** to analyze a single ticker. Use its sidebar to enter the ticker and run the analysis.
        *   Select **📊 Portfolio Analysis** to manage your list, optimize, and backtest. Use its sidebar to modify the ticker list and run the optimization/backtesting.

    ---

    *Disclaimer: This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own thorough research before making any investment decisions.*

    **Select an option from the menu on the `left sidebar` to begin!**
    """)

st.divider()

# In introduction.py

# --- Global Configuration Section ---
st.subheader("Global Configuration")
st.markdown("""
    Set the core assumptions used for calculations throughout the analysis tools. These settings apply globally.

    *   **Risk-Free Rate:** Represents the theoretical return of an investment with zero risk (often proxied by government T-bills). It's crucial for calculating risk-adjusted returns.
    *   **Benchmark Ticker:** The index or ETF you want to compare performance against (e.g., the overall market or a specific sector). Used to calculate Beta, Alpha, and for backtesting comparisons.

    Defaults are provided, but adjust them based on your assumptions or current market conditions if needed.
    """)

# Risk-Free Rate Input with more context
st.session_state.risk_free_rate = st.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=0.0, max_value=20.0,
    value=st.session_state.get('risk_free_rate', 0.04) * 100, # Use .get for safety
    step=0.1,
    format="%.1f",
    key="config_risk_free_rate_input_intro", # Unique key for this page
    help="Enter the expected return of a 'risk-free' asset, like a short-term government bond."
) / 100
st.caption("This rate is used for calculating metrics like the Sharpe, Sortino, and Treynor Ratios, as well as for estimating Alpha via the CAPM model.")

# Benchmark Ticker Input with more context
st.session_state.benchmark = st.text_input(
    "Benchmark Ticker",
    value=st.session_state.get('benchmark', 'SPY'), # Use .get for safety
    key="config_benchmark_input_intro", # Unique key for this page
    help="Enter the ticker symbol for market comparison (e.g., SPY, VOO, QQQ, VT, ^IXIC)."
).upper()
st.caption("Select a relevant market index or ETF (like SPY for S&P 500, QQQ for Nasdaq 100, or VT for Total World) to compare against. Used for Beta, Alpha, and Backtesting.")

st.divider() # Add a visual separator

# Portfolio Context Section
st.subheader("Portfolio Context")
st.markdown(f"""
    The analysis tools (like Stock Analysis recommendations and Portfolio Optimization)
    will use the following list of **{len(st.session_state.get('portfolio_tickers', []))} tickers** as the base portfolio for calculations:
    """)
# Display the list clearly
st.info(f"{', '.join(st.session_state.get('portfolio_tickers', []))}")
st.caption("You can manage this list on the '📊 Portfolio Analysis' page.")

# --- Footer ---
# (Sin cambios, gestionado por CSS global)