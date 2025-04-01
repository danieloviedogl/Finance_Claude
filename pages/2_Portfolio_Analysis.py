# pages/2_Portfolio_Analysis.py
# Note: File name still includes number as per user's last clarification.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from Analizer import AdvancedStockAnalyzer, DEFAULT_PORTFOLIO_TICKERS # Import your class and defaults
import json
import os # Added import for path checking if needed later

# --- Specific page configuration ---
# st.set_page_config() no longer needed here

# --- Title and Header ---
st.title("Portfolio Analysis and Optimization") # Changed title

# --- CSS (Optional) ---
st.markdown("""
    <style>
    /* Specific styles if needed */
    .metric-container {
        background-color: #F8FAFC;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    .metric-title {
        font-weight: bold;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization for this page ---
if 'portfolio_optimization_result' not in st.session_state:
    st.session_state.portfolio_optimization_result = None
if 'portfolio_optimization_error' not in st.session_state:
    st.session_state.portfolio_optimization_error = None
if 'portfolio_backtest_result' not in st.session_state:
    st.session_state.portfolio_backtest_result = None
if 'portfolio_backtest_error' not in st.session_state:
    st.session_state.portfolio_backtest_error = None
if 'running_optimization' not in st.session_state:
    st.session_state.running_optimization = False
if 'running_backtest' not in st.session_state:
    st.session_state.running_backtest = False
if 'portfolio_analyzer_instance' not in st.session_state:
    st.session_state.portfolio_analyzer_instance = None # To save instance after optimization

# --- Helpers and Callbacks ---
def validate_ticker(ticker):
    """Validate basic ticker format""" # Changed docstring
    # Allow tickers with '^' like benchmark
    return ticker and isinstance(ticker, str) and 1 <= len(ticker) <= 10 and (ticker.isalpha() or ticker.isalnum() or '^' in ticker)

def add_ticker_callback():
    """Add ticker to portfolio""" # Changed docstring
    custom_ticker = st.session_state.get('custom_ticker_input_portfolio', "").upper()
    if validate_ticker(custom_ticker) and custom_ticker not in st.session_state.portfolio_tickers:
        st.session_state.portfolio_tickers.append(custom_ticker)
        st.session_state.portfolio_tickers = sorted(list(set(st.session_state.portfolio_tickers)))
        st.session_state.custom_ticker_input_portfolio = "" # Clear input
    elif custom_ticker in st.session_state.portfolio_tickers:
         # Changed toast message
         st.toast(f"Ticker '{custom_ticker}' is already in the portfolio.", icon="⚠️")
    else:
         # Changed toast message
        st.toast(f"Invalid ticker '{custom_ticker}'.", icon="❌")


def remove_ticker_callback(ticker_to_remove):
    """Remove ticker from portfolio""" # Changed docstring
    if ticker_to_remove in st.session_state.portfolio_tickers:
        st.session_state.portfolio_tickers.remove(ticker_to_remove)
    # No need to refresh explicitly, Streamlit will do it

def clear_portfolio_callback():
    """Reset portfolio to default""" # Changed docstring
    st.session_state.portfolio_tickers = DEFAULT_PORTFOLIO_TICKERS.copy() # Use default from Analyzer.py

def run_optimization_callback():
    """Prepare to run optimization""" # Changed docstring
    st.session_state.portfolio_optimization_result = None
    st.session_state.portfolio_optimization_error = None
    st.session_state.portfolio_backtest_result = None # Reset backtest too
    st.session_state.portfolio_backtest_error = None
    st.session_state.portfolio_analyzer_instance = None # Reset instance
    if len(st.session_state.portfolio_tickers) >= 2:
        st.session_state.running_optimization = True
    else:
         # Changed error message
        st.session_state.portfolio_optimization_error = "At least 2 tickers are needed for optimization."
        st.session_state.running_optimization = False

def run_backtest_callback():
    """Prepare to run backtest""" # Changed docstring
    st.session_state.portfolio_backtest_result = None
    st.session_state.portfolio_backtest_error = None
    # Check if there are optimization results to use
    if st.session_state.portfolio_optimization_result and 'optimal_weights' in st.session_state.portfolio_optimization_result:
         # Check if the instance is saved
         if st.session_state.portfolio_analyzer_instance:
             st.session_state.running_backtest = True
         else:
              # Changed error message
              st.session_state.portfolio_backtest_error = "Internal error: Analyzer instance not found after optimization."
              st.session_state.running_backtest = False
    else:
         # Changed error message
        st.session_state.portfolio_backtest_error = "You must run optimization first to get the weights."
        st.session_state.running_backtest = False

# --- Sidebar (Controls Specific to this Page) ---
with st.sidebar:
    # Changed header
    st.header("Portfolio Management")

    # Changed expander label and caption
    with st.expander("Tickers in Portfolio", expanded=True):
        st.caption("Tickers used for optimization and backtesting.")

        # Display current tickers with remove button
        if not st.session_state.portfolio_tickers:
             # Changed message
            st.write("Portfolio empty.")
        else:
            cols = st.columns(3)
            for i, t in enumerate(st.session_state.portfolio_tickers):
                cols[i % 3].button(
                    f"➖ {t}",
                    key=f"remove_{t}_portfolio",
                    on_click=remove_ticker_callback,
                    args=(t,),
                     # Changed help text
                    help=f"Remove {t} from portfolio"
                )

        # Add new ticker
        # Changed label and placeholder
        st.text_input(
            "Add Ticker:",
            key="custom_ticker_input_portfolio",
            placeholder="E.g.: NVDA, ^GSPC" # Changed E.g.
        )
        # Changed button text
        st.button("➕ Add", on_click=add_ticker_callback, use_container_width=True)

        # Portfolio action buttons
        col1, col2 = st.columns(2)
        # Changed button text
        col1.button("🗑️ Clear All", on_click=clear_portfolio_callback, use_container_width=True)
        #col2.button("↺ Reset Default", on_click=clear_portfolio_callback, use_container_width=True) # Redundant?

    # Changed button text and caption
    st.button(
        "📊 Optimize Portfolio",
        on_click=run_optimization_callback,
        type="primary",
        use_container_width=True
    )

    st.info("""
    **Tips:**
    *   Use a diverse set of tickers for better optimization results.
    *   The **Optimize Portfolio** button finds weights maximizing the Sharpe Ratio (risk-adjusted return).
    *   Backtesting shows how this optimized portfolio *might* have performed historically. Past performance is not indicative of future results.
    """)

    st.caption("""
        Remember to adjust the Risk-Free Rate and Benchmark on the 'Introduction' page if needed, as they affect optimization and backtesting calculations.
        """)


# --- Main Optimization Logic ---
if st.session_state.running_optimization:
    portfolio_list = st.session_state.portfolio_tickers
    risk_free_rate = st.session_state.get('risk_free_rate', 0.04)
    benchmark = st.session_state.get('benchmark', 'SPY')

    # Changed sub-header
    st.markdown(f'<div class="sub-header">Optimizing Portfolio with {len(portfolio_list)} Tickers...</div>', unsafe_allow_html=True)
    # Keep 'Tickers:' label
    st.write(f"Tickers: {', '.join(portfolio_list)}")

    try:
         # Changed spinner message
        with st.spinner("Calculating optimal weights..."):
            # Create an instance. The main 'ticker' is not very relevant here,
            # but __init__ needs it. Use the first one from the list.
            temp_analyzer = AdvancedStockAnalyzer(
                ticker=portfolio_list[0], # 'dummy' ticker for initialization
                portfolio_tickers=portfolio_list,
                risk_free_rate=risk_free_rate,
                benchmark=benchmark
            )
            # It's important to call fetch_data if calculate_optimal_portfolio needs it
            # (Although yfinance downloads within the function, having benchmark_data can be useful)
            temp_analyzer.fetch_data() # To ensure benchmark is loaded if necessary
            opt_result = temp_analyzer.calculate_optimal_portfolio()

            if opt_result and 'error' not in opt_result:
                st.session_state.portfolio_optimization_result = opt_result
                st.session_state.portfolio_analyzer_instance = temp_analyzer # Save instance for backtest
                # Changed success message
                st.success("Optimization completed!")
            elif opt_result:
                 # Changed error message
                st.session_state.portfolio_optimization_error = f"Optimization failed: {opt_result.get('error', 'Unknown reason')}"
            else:
                 # Changed error message
                st.session_state.portfolio_optimization_error = "Optimization returned no results."

    except Exception as e:
         # Changed error message
        st.session_state.portfolio_optimization_error = f"Unexpected error during optimization: {e}"
        st.error(st.session_state.portfolio_optimization_error)
        import traceback
        st.exception(traceback.format_exc())

    finally:
        st.session_state.running_optimization = False # Deactivate flag


# --- Display Optimization Errors ---
if st.session_state.portfolio_optimization_error:
     # Changed error message
    st.error(f"Optimization Error: {st.session_state.portfolio_optimization_error}")

# --- Display Optimization Results ---
if st.session_state.portfolio_optimization_result:
    opt_res = st.session_state.portfolio_optimization_result
    # Changed sub-header
    st.markdown('<div class="sub-header">Optimization Results (Max Sharpe Ratio)</div>', unsafe_allow_html=True)

    col_chart, col_metrics = st.columns([1, 1])

    with col_chart:
         # Changed subheader
        st.subheader("Optimal Composition")
        weights_dict = opt_res.get('optimal_weights', {})
        # Filter small weights
        plot_weights = {ticker: weight for ticker, weight in weights_dict.items() if weight > 0.001}

        if plot_weights:
            pie_labels = list(plot_weights.keys())
            pie_values = list(plot_weights.values())

            fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=.4,
                                             textinfo='label+percent', pull=[0.02]*len(pie_labels))])
             # Changed plot title
            fig_pie.update_layout(title_text="Optimal Portfolio Weights", height=350, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
             # Changed info message
            st.info("No significant optimal weights found to plot.")

    with col_metrics:
         # Changed subheader
        st.subheader("Expected Portfolio Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        ret = opt_res.get('optimal_return')
        vol = opt_res.get('optimal_volatility')
        sharpe = opt_res.get('optimal_sharpe')
         # Changed metric labels
        m_col1.metric("Expected Ann. Return", f"{ret*100:.1f}%" if isinstance(ret, float) else "N/A")
        m_col2.metric("Expected Ann. Volatility", f"{vol*100:.1f}%" if isinstance(vol, float) else "N/A")
        m_col3.metric("Expected Sharpe Ratio", f"{sharpe:.2f}" if isinstance(sharpe, float) else "N/A")

        # Display weights in table
        # Changed column header
        st.dataframe(
             pd.DataFrame.from_dict(weights_dict, orient='index', columns=['Optimal Weight'])
             .applymap(lambda x: f"{x*100:.2f}%") # Format as %
             .sort_values(by='Optimal Weight', ascending=False),
             use_container_width=True
         )

    # --- Backtesting Section (Appears after optimization) ---
    st.divider()
    # Changed subheader
    st.subheader("Optimized Portfolio Backtesting")

    # Changed label and help text
    years_to_backtest = st.number_input(
        "Select Years to Backtest:",
        min_value=1, max_value=15, value=5, step=1, key="years_backtest_portfolio",
        help="Number of years back from today to test the strategy."
    )
    # Changed button text
    st.button("📈 Run Backtest", on_click=run_backtest_callback, use_container_width=True)

    # --- Backtesting Logic ---
    if st.session_state.running_backtest:
         # Changed sub-header
        st.markdown(f'<div class="sub-header">Running {years_to_backtest}-year Backtest...</div>', unsafe_allow_html=True)
        try:
             # Changed spinner message
            with st.spinner("Performing historical simulation..."):
                # Use the saved instance which already has optimization results
                analyzer_instance = st.session_state.portfolio_analyzer_instance
                if analyzer_instance:
                    backtest_data = analyzer_instance.backtest_portfolio(years_back=int(years_to_backtest)) # Ensure int

                    if backtest_data and 'error' not in backtest_data:
                        st.session_state.portfolio_backtest_result = backtest_data
                         # Changed success message
                        st.success(f"{years_to_backtest}-year backtest completed!")
                    elif backtest_data:
                         # Changed error message
                        st.session_state.portfolio_backtest_error = f"Backtest failed: {backtest_data.get('error', 'Unknown reason')}"
                    else:
                         # Changed error message
                        st.session_state.portfolio_backtest_error = "Backtest returned no results."
                else:
                     # Changed error message
                     st.session_state.portfolio_backtest_error = "Error: Analyzer instance needed for backtest not found."

        except Exception as e:
             # Changed error message
            st.session_state.portfolio_backtest_error = f"Unexpected error during backtest: {e}"
            st.error(st.session_state.portfolio_backtest_error)
            import traceback
            st.exception(traceback.format_exc())
        finally:
            st.session_state.running_backtest = False # Deactivate flag

    # --- Display Backtest Error ---
    if st.session_state.portfolio_backtest_error:
         # Changed warning message
        st.warning(f"Backtest Error: {st.session_state.portfolio_backtest_error}")

    # --- Display Backtest Results ---
    if st.session_state.portfolio_backtest_result:
        bt_res = st.session_state.portfolio_backtest_result
        # Changed markdown title
        st.markdown(f"#### Backtest Results ({bt_res.get('years_backtested', 'N/A')} Years: {bt_res.get('start_date','?')} to {bt_res.get('end_date','?')})")

        # Main backtest metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
         # Changed title
        st.markdown('<div class="metric-title">Historical Portfolio Performance</div>', unsafe_allow_html=True)
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        ann_ret = bt_res.get('annualized_return_percent')
        ann_vol = bt_res.get('annualized_volatility_percent')
        sharpe_bt = bt_res.get('sharpe_ratio')
        mdd = bt_res.get('max_drawdown_percent')
         # Changed metric labels
        b_col1.metric("Annualized Return", f"{ann_ret:.1f}%" if isinstance(ann_ret, float) else 'N/A')
        b_col2.metric("Annualized Volatility", f"{ann_vol:.1f}%" if isinstance(ann_vol, float) else 'N/A')
        b_col3.metric("Sharpe Ratio (Backtest)", f"{sharpe_bt:.2f}" if isinstance(sharpe_bt, float) else 'N/A')
        b_col4.metric("Max Drawdown", f"{mdd:.1f}%" if isinstance(mdd, float) else 'N/A')
        st.markdown('</div>', unsafe_allow_html=True)

        # Benchmark Comparison
        bench_comp = bt_res.get('benchmark_comparison', {})
        benchmark_ticker = bench_comp.get('benchmark', st.session_state.get('benchmark','SPY'))
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
         # Changed title
        st.markdown(f'<div class="metric-title">Benchmark Comparison ({benchmark_ticker})</div>', unsafe_allow_html=True)
        bc_col1, bc_col2, bc_col3 = st.columns(3)
        b_ret = bench_comp.get('benchmark_annualized_return_percent')
        b_vol = bench_comp.get('benchmark_annualized_volatility_percent')
        outperf = bench_comp.get('outperformance_percent')
         # Changed metric labels
        bc_col1.metric(f"{benchmark_ticker} Ann. Return", f"{b_ret:.1f}%" if isinstance(b_ret, float) else 'N/A')
        bc_col2.metric(f"{benchmark_ticker} Ann. Volatility", f"{b_vol:.1f}%" if isinstance(b_vol, float) else 'N/A')
        delta_val = f"{outperf:.1f}%" if isinstance(outperf, float) else None
        bc_col3.metric("Outperformance vs Benchmark", f"{outperf:.1f}%" if isinstance(outperf, float) else 'N/A', delta=delta_val)
        st.markdown('</div>', unsafe_allow_html=True)

        # Cumulative Returns Plot
        cum_ret_port_dict = bt_res.get('cumulative_returns_portfolio', {})
        cum_ret_bench_dict = bt_res.get('cumulative_returns_benchmark', {})

        if cum_ret_port_dict:
            try:
                # Convert dicts to Series and ensure datetime index
                port_series = pd.Series(cum_ret_port_dict)
                port_series.index = pd.to_datetime(port_series.index)
                port_series = port_series.sort_index()

                # Consistent conversion to % (assuming input is 1+return)
                if not port_series.empty:
                    if abs(port_series.iloc[0] - 1) < 0.1 or port_series.iloc[0] == 0: # Heuristic
                        plot_df = pd.DataFrame({'Portfolio': (port_series - 1) * 100})
                    else:
                        plot_df = pd.DataFrame({'Portfolio': port_series * 100})
                else:
                    plot_df = pd.DataFrame({'Portfolio': port_series})


                if cum_ret_bench_dict:
                    try:
                        # --- START CORRECTION BLOCK (Keep logic) ---
                        if len(cum_ret_bench_dict) == 1 and isinstance(list(cum_ret_bench_dict.values())[0], dict):
                            actual_bench_data_dict = list(cum_ret_bench_dict.values())[0]
                            # st.write(f"Debug: Nested structure detected for benchmark '{list(cum_ret_bench_dict.keys())[0]}'. Extracting inner data.")
                        elif isinstance(cum_ret_bench_dict, dict):
                             actual_bench_data_dict = {k: v for k, v in cum_ret_bench_dict.items() if isinstance(k, str)}
                             if len(actual_bench_data_dict) != len(cum_ret_bench_dict):
                                  st.warning("Filtered out non-string keys from benchmark cumulative returns data.")
                        else:
                             # Changed warning
                            st.warning("Unexpected format for benchmark cumulative data.")
                            actual_bench_data_dict = {}

                        if actual_bench_data_dict:
                            bench_series = pd.Series(actual_bench_data_dict)
                            bench_series.index = pd.to_datetime(bench_series.index)
                            bench_series = bench_series.sort_index()

                            # Consistent conversion to %
                            if not bench_series.empty:
                                if abs(bench_series.iloc[0] - 1) < 0.1 or bench_series.iloc[0] == 0:
                                    plot_df[f'Benchmark ({benchmark_ticker})'] = (bench_series - 1) * 100
                                else:
                                     plot_df[f'Benchmark ({benchmark_ticker})'] = bench_series * 100
                        # --- END CORRECTION BLOCK ---

                    except Exception as e:
                         # Changed warning message
                        st.warning(f"Could not process benchmark ({benchmark_ticker}) cumulative returns: {e}")


                if not plot_df.empty:
                     if 'Portfolio' not in plot_df.columns:
                          plot_df['Portfolio'] = np.nan
                     if f'Benchmark ({benchmark_ticker})' not in plot_df.columns:
                          plot_df[f'Benchmark ({benchmark_ticker})'] = np.nan

                     plot_df = plot_df.ffill().dropna(how='all')

                     if not plot_df.empty:
                          st.markdown("---") # Visual separator
                          # Changed subheader
                          st.subheader("Cumulative Return Evolution")
                           # Changed plot title and labels
                          fig_cum_ret = px.line(plot_df, title="Cumulative Return (%) - Portfolio vs Benchmark",
                                                labels={'index': 'Date', 'value': 'Cumulative Return (%)', 'variable': 'Series'})
                          fig_cum_ret.update_layout(yaxis_ticksuffix='%', height=400, legend_title_text='')
                          fig_cum_ret.update_traces(connectgaps=True) # Connect gaps if any
                          st.plotly_chart(fig_cum_ret, use_container_width=True)
                     else:
                           # Changed warning
                          st.warning("Not enough valid cumulative return data to plot after preprocessing.")
                else:
                      # Changed warning
                     st.warning("Could not generate cumulative return plot due to missing portfolio data.")


            except Exception as e:
                 # Changed warning
                 st.warning(f"Could not plot cumulative returns: {e}")
                 st.exception(e)
        else:
              # Changed info message
             st.info("Portfolio cumulative return data unavailable for plotting.")


# --- Initial Message if no optimization ---
if not st.session_state.portfolio_optimization_result and not st.session_state.portfolio_optimization_error and not st.session_state.running_optimization:
     # Changed info message
    st.info("💼 Manage your **ticker list** in the sidebar and click `📊 Optimize Portfolio` to start.")

# --- Footer (Optional) ---
# st.markdown('<div class="footer">Portfolio Analysis | © 2024</div>', unsafe_allow_html=True)