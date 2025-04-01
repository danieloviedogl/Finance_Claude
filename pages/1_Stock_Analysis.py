# pages/1_Stock_Analysis.py
# Note: File name still includes number as per user's last clarification.
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from Analizer import AdvancedStockAnalyzer # Import your class
import json

# --- Specific page configuration ---
# st.set_page_config() is no longer needed here, done in app.py

# --- Title and Header ---
st.title("Stock Analysis Pro") # Changed title

# --- CSS (Optional, if styles are needed only for this page) ---
st.markdown("""
    <style>
    /* Specific styles for this page if necessary */
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border-left: 5px solid #cccccc; /* Default border */
    }
    .recommendation-box:hover {
        transform: scale(1.01);
    }
    .metric-container {
        background-color: #F8FAFC;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
        transition: box-shadow 0.3s ease;
    }
    .metric-container:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-weight: bold;
        color: #334155; /* slate-700 */
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization for this page ---
if 'stock_analysis_result' not in st.session_state:
    st.session_state.stock_analysis_result = None
if 'stock_analysis_error' not in st.session_state:
    st.session_state.stock_analysis_error = None
if 'running_stock_analysis' not in st.session_state:
    st.session_state.running_stock_analysis = False # Flag to know if started

# --- Helper to validate ticker ---
def validate_ticker(ticker):
    # Assuming US stock tickers for simplicity, adjust if needed
    return ticker and isinstance(ticker, str) and 1 <= len(ticker) <= 6 and ticker.isalpha()

# --- Callback for the analysis button ---
def run_analysis_callback():
    """Resets results and activates the analysis flag"""
    st.session_state.stock_analysis_result = None
    st.session_state.stock_analysis_error = None
    # Update the current ticker from the input before running
    st.session_state.current_ticker = st.session_state.get('ticker_input_stock', st.session_state.current_ticker).upper()
    if validate_ticker(st.session_state.current_ticker):
        st.session_state.running_stock_analysis = True # Indicate that analysis should run
    else:
        # Translate error message
        st.session_state.stock_analysis_error = "Please enter a valid ticker symbol (e.g., AAPL)."
        st.session_state.running_stock_analysis = False

# --- Sidebar (Controls Specific to this Page) ---
with st.sidebar:
    st.header("Individual Analysis") # Changed header
    ticker_input = st.text_input(
        "Enter Ticker to Analyze", # Changed label
        value=st.session_state.current_ticker,
        key="ticker_input_stock",
        help="E.g.: AAPL, MSFT, GOOGL" # Changed help text
    )
    st.button(
        "🚀 Analyze Stock", # Changed button text
        on_click=run_analysis_callback,
        type="primary",
        use_container_width=True
    )
    # Information about the analysis
    st.info("""
        This dashboard visualizes the results of a deep stock analysis. 
        The analysis includes:
        
        - Technical Analysis
        - Risk Metrics
        - Time Series Forecasting
        - Fundamental Analysis
        - Monte Carlo Simulation
        - Portfolio Optimization
        - Sentiment Analysis
        """)
    # Disclaimer
    st.caption(
        """
        Disclaimer: This analysis is for informational purposes only 
        and should not be considered financial advice. Always conduct 
        your own research before making investment decisions.
        """
    )


# --- Main Analysis Logic ---
# Execute only if the flag is active and no prior validation error
if st.session_state.running_stock_analysis and not st.session_state.stock_analysis_error:
    current_ticker = st.session_state.current_ticker
    portfolio_list = st.session_state.get('portfolio_tickers', [])
    risk_free_rate = st.session_state.get('risk_free_rate', 0.04)
    benchmark = st.session_state.get('benchmark', 'SPY')

    # Translate sub-header
    st.markdown(f'<div class="sub-header">Analysis for Ticker: {current_ticker}</div>', unsafe_allow_html=True)

    try:
        # Translate spinner message
        with st.spinner(f"Analyzing {current_ticker}... This may take a moment."):
            # Ensure the current ticker is in the portfolio list for analysis
            if current_ticker not in portfolio_list:
                 portfolio_list_for_analysis = sorted(list(set([current_ticker] + portfolio_list)))
                 # Translate warning message
                 st.warning(f"{current_ticker} was not in the portfolio list; it has been temporarily added for contextual analysis.")
            else:
                 portfolio_list_for_analysis = portfolio_list

            analyzer = AdvancedStockAnalyzer(
                ticker=current_ticker,
                portfolio_tickers=portfolio_list_for_analysis,
                risk_free_rate=risk_free_rate,
                benchmark=benchmark
            )
            result = analyzer.run_full_analysis() # run_full_analysis includes optimization

            if result and 'error' not in result:
                st.session_state.stock_analysis_result = result
                # We don't store 'analyzer_instance' globally to avoid serialization issues
                # and keep pages more independent. Recreate if needed.
                # Translate success message
                st.success(f"Analysis for {current_ticker} completed!")
            elif result:
                 # Translate error message
                 st.session_state.stock_analysis_error = f"Analysis failed for {current_ticker}: {result.get('error', 'Unknown reason')}"
            else:
                 # Translate error message
                st.session_state.stock_analysis_error = f"Analysis for {current_ticker} returned no results."

    except Exception as e:
         # Translate error message
        st.session_state.stock_analysis_error = f"An unexpected error occurred during the analysis of {current_ticker}: {e}"
        st.error(st.session_state.stock_analysis_error)
        import traceback
        # Shows traceback in the app for debugging (no translation needed here)
        st.exception(traceback.format_exc())

    finally:
         st.session_state.running_stock_analysis = False # Reset flag after attempt

# --- Display Errors (If any occurred) ---
if st.session_state.stock_analysis_error:
    # Translate error message
    st.error(f"Analysis Error: {st.session_state.stock_analysis_error}")

# --- Display Analysis Results (If they exist) ---
if st.session_state.stock_analysis_result:
    result = st.session_state.stock_analysis_result
    current_ticker = result.get('ticker', 'N/A')
    
    # --- Recommendation Card ---
    rec_color = result.get('recommendation_color', '#cccccc') # Default color
    rec_text = result.get('recommendation_text', 'Recommendation unavailable.') # Translate default text if desired, but the actual text comes from Analyzer
    rec_conf = result.get('confidence', 'N/A')
    rec_score = result.get('total_score_normalized', 'N/A')
    curr_price_val = result.get('current_price', None)
    curr_price_str = f"${curr_price_val:.2f}" if isinstance(curr_price_val, (int, float)) else 'N/A'
    anl_date = result.get('analysis_date', 'N/A')
    comp_name = result.get('company_name', current_ticker)

    # Translate labels within the card
    st.markdown(
        f"""
        <div class="recommendation-box" style="border-left: 5px solid {rec_color}; background-color: {rec_color}20;">
            <h2 style="color: {rec_color}; margin-top: 0;">{result.get('recommendation', 'N/A')} ({rec_conf} Confidence)</h2>
            <h4>{current_ticker} - {comp_name}</h4>
            <p style="font-size: 1.1rem;">Score: {rec_score}/100 | Current Price: {curr_price_str} | Analysis Date: {anl_date}</p>
            <p>{rec_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Detailed Metrics ---
    col_score, col_details = st.columns([1, 2]) # Adjust ratio if needed

    with col_score:
        st.subheader("Investment Score") # Changed header
        # Gauge for overall score
        if isinstance(rec_score, (int, float)):
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=rec_score,
                domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Overall Score (0-100)"}, # Changed title
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': rec_color},
                    'steps': [
                        {'range': [0, 35], 'color': "#FECACA"}, {'range': [35, 50], 'color': "#FED7AA"},
                        {'range': [50, 65], 'color': "#FEF08A"}, {'range': [65, 80], 'color': "#BBF7D0"},
                        {'range': [80, 100], 'color': "#86EFAC"} ],
                    'threshold': {'line': {'color': "gray", 'width': 4}, 'thickness': 0.75, 'value': rec_score}
                }))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.metric("Overall Score", "N/A") # Changed label

        # Bar chart for category scores
        st.subheader("Score Breakdown") # Changed header
        cat_scores = result.get('category_scores', {})
        max_scores = result.get('max_category_scores', {})
        if cat_scores and max_scores:
             # Translate df column names for display if needed (here it's automatic from dict keys)
            scores_data = [{"Category": k.replace('_', ' ').title(), "Score": v, "Max": max_scores.get(k, 0)}
                           for k, v in cat_scores.items() if max_scores.get(k, 0) > 0] # Exclude if max is 0
            scores_df = pd.DataFrame(scores_data)

            if not scores_df.empty:
                # Translate axis labels and title
                fig_bar = px.bar(scores_df, x="Score", y="Category", orientation='h',
                                color="Score", color_continuous_scale=px.colors.sequential.Blues,
                                text="Score", range_x=[0, scores_df["Max"].max() + 2])
                # Add max score lines (subtle)
                for i, row in scores_df.iterrows():
                    fig_bar.add_shape(type="line", x0=row["Max"], y0=i-0.4, x1=row["Max"], y1=i+0.4,
                                        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot"))
                    fig_bar.add_annotation(x=row["Max"], y=i, text=f"Max: {row['Max']}", showarrow=False, xshift=25, font=dict(size=9, color="grey"))

                # Translate layout titles
                fig_bar.update_layout(title="Score per Category", xaxis_title="Score", yaxis_title="",
                                      coloraxis_showscale=False, height=350, margin=dict(l=10, r=10, t=40, b=10))
                fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                 # Translate info message
                 st.info("Category score breakdown unavailable.")
        else:
             # Translate info message
             st.info("Category score breakdown unavailable.")

    with col_details:
        st.subheader("Detailed Key Metrics") # Changed header
        # Tabs for different metric categories
        key_metrics = result.get('key_metrics', {})
        # Check if keys exist before creating tabs
        available_tabs = []
        # Translate tab names
        tab_map = {
            "Technical & Risk": ('technical', 'risk_metrics'),
            "Fundamental & T-Series": ('fundamental', 'time_series'),
            "Monte Carlo & Sentiment": ('monte_carlo', 'sentiment'),
            #"Portfolio": ('portfolio',) # Portfolio context is better viewed on its own page
        }
        display_names = []
        metric_keys_in_tab = {}

        for name, keys in tab_map.items():
            if any(k in key_metrics for k in keys):
                display_names.append(name)
                metric_keys_in_tab[name] = keys

        if not display_names:
             # Translate warning
            st.warning("No detailed metrics available to display.")
        else:
            tabs = st.tabs(display_names)
            tab_dict = dict(zip(display_names, tabs)) # Access by name

            # --- Tab: Technical & Risk ---
            if "Technical & Risk" in tab_dict:
                with tab_dict["Technical & Risk"]:
                    tech = key_metrics.get('technical', {})
                    risk = key_metrics.get('risk_metrics', {})

                    if tech:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                         # Translate title
                        st.markdown('<div class="metric-title">Technical Indicators</div>', unsafe_allow_html=True)
                        m_col1, m_col2, m_col3 = st.columns(3)
                        # Translate metric labels
                        m_col1.metric("RSI", f"{tech.get('rsi', 'N/A'):.2f}" if isinstance(tech.get('rsi'), float) else 'N/A')
                        m_col2.metric("Trend Signal (MA)", tech.get('trend', 'N/A'))
                        m_col3.metric("MACD Signal", tech.get('macd_signal', 'N/A'))
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                         # Translate caption
                         st.caption("Technical data unavailable.")

                    if risk:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                         # Translate title
                        st.markdown('<div class="metric-title">Risk Metrics</div>', unsafe_allow_html=True)
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        # Translate metric labels
                        m_col1.metric("Sharpe Ratio", f"{risk.get('sharpe_ratio', 'N/A'):.2f}" if isinstance(risk.get('sharpe_ratio'), float) else 'N/A')
                        m_col2.metric("Max Drawdown", f"{risk.get('max_drawdown_percent', 'N/A'):.1f}%" if isinstance(risk.get('max_drawdown_percent'), float) else 'N/A')
                        m_col3.metric("Beta", f"{risk.get('beta', 'N/A'):.2f}" if isinstance(risk.get('beta'), float) else 'N/A')
                        m_col4.metric("Annual Alpha", f"{risk.get('alpha_annual_percent', 'N/A'):.1f}%" if isinstance(risk.get('alpha_annual_percent'), float) else 'N/A')
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                         # Translate caption
                         st.caption("Risk metrics unavailable.")

            # --- Tab: Fundamental & T-Series ---
            if "Fundamental & T-Series" in tab_dict:
                with tab_dict["Fundamental & T-Series"]:
                    fund = key_metrics.get('fundamental', {})
                    ts = key_metrics.get('time_series', {})

                    if fund:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                         # Translate title
                        st.markdown('<div class="metric-title">Fundamental Metrics</div>', unsafe_allow_html=True)
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        # Translate metric labels
                        m_col1.metric("P/E Ratio", f"{fund.get('pe_ratio', 'N/A'):.1f}" if isinstance(fund.get('pe_ratio'), float) else 'N/A')
                        m_col2.metric("Analyst Upside", f"{fund.get('price_target_upside_percent', 'N/A'):.1f}%" if isinstance(fund.get('price_target_upside_percent'), float) else 'N/A')
                        m_col3.metric("Profit Margin", f"{fund.get('profit_margins_percent', 'N/A'):.1f}%" if isinstance(fund.get('profit_margins_percent'), float) else 'N/A')
                        m_col4.metric("Analyst Rating", fund.get('recommendation', 'N/A'))
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                         # Translate caption
                         st.caption("Fundamental metrics unavailable.")

                    if ts:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                         # Translate title
                        st.markdown('<div class="metric-title">Time Series Forecast</div>', unsafe_allow_html=True)
                        m_col1, m_col2 = st.columns(2)
                        stat = ts.get('stationary')
                         # Translate metric labels and values
                        m_col1.metric("Returns Stationary?", "Yes" if stat else "No" if stat is False else "N/A")
                        ret_30d = ts.get('expected_return_30d_percent')
                        m_col2.metric("Expected 30d Return", f"{ret_30d:.1f}%" if isinstance(ret_30d, float) else 'N/A')
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                         # Translate caption
                         st.caption("Time series forecast unavailable.")

            # --- Tab: Monte Carlo & Sentiment ---
            if "Monte Carlo & Sentiment" in tab_dict:
                 with tab_dict["Monte Carlo & Sentiment"]:
                    mc = key_metrics.get('monte_carlo', {})
                    sent = key_metrics.get('sentiment', {})

                    if mc:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                         # Translate title
                        st.markdown('<div class="metric-title">Monte Carlo Simulation (1 Year)</div>', unsafe_allow_html=True)
                        m_col1, m_col2 = st.columns(2)
                        mc_ret = mc.get('expected_return_percent')
                        mc_prob = mc.get('prob_profit_percent')
                         # Translate metric labels
                        m_col1.metric("Expected Return", f"{mc_ret:.1f}%" if isinstance(mc_ret, float) else 'N/A')
                        m_col2.metric("Probability of Profit", f"{mc_prob:.1f}%" if isinstance(mc_prob, float) else 'N/A')
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                         # Translate caption
                         st.caption("Monte Carlo simulation unavailable.")

                    if sent:
                         st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                         # Translate title
                         st.markdown('<div class="metric-title">Analyst Sentiment</div>', unsafe_allow_html=True)
                         m_col1, m_col2 = st.columns(2)
                         sent_score = sent.get('sentiment_score_0_1')
                         # Translate metric labels
                         m_col1.metric("Overall Sentiment", sent.get('sentiment', 'N/A'))
                         m_col2.metric("Sentiment Score (0-1)", f"{sent_score:.2f}" if isinstance(sent_score, float) else 'N/A')
                         st.markdown('</div>', unsafe_allow_html=True)
                    else:
                         # Translate caption
                        st.caption("Sentiment analysis unavailable.")

# --- Initial Message if no results ---
if not st.session_state.stock_analysis_result and not st.session_state.stock_analysis_error and not st.session_state.running_stock_analysis:
     # Translate info message
    st.info("Enter a ticker in the sidebar and click 'Analyze Stock' to view the results.")

# --- Footer (Optional, if not globally set in app.py) ---
# st.markdown('<div class="footer">Stock Analysis | © 2024</div>', unsafe_allow_html=True)