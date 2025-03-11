import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from Analizer import AdvancedStockAnalyzer


# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 0.8rem;
    }
    .portfolio-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# Lista predefinida de empresas populares para incluir en el portafolio
# Esta lista podría expandirse o cargarse desde un archivo externo
DEFAULT_COMPANIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "JPM",
    "BAC", "WFC", "GS", "V", "MA", "DIS", "NFLX", "PG", "KO", "PEP", "WMT", "HD",
    "CSCO", "VZ", "T", "PFE", "MRK", "JNJ", "UNH", "CVX", "XOM", "BA", "CAT",
    "MMM", "IBM", "ORCL", "CRM", "ADBE", "PYPL", "BABA", "TSM", "ASML", "NKE"
]

# Inicializar estado de sesión para el portafolio
if 'portfolio_companies' not in st.session_state:
    st.session_state.portfolio_companies = []

if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# Sidebar for ticker input and analysis options
with st.sidebar:
    st.title("Stock Analysis Controls")
    
    # Input for ticker symbol
    ticker = st.text_input("Enter Ticker Symbol", value="CLSK")
    
    # Portfolio options section
    st.subheader("Portfolio Options")
    
    # Add company selector
    st.markdown("##### Add companies to your portfolio")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_company = st.selectbox(
            "Select companies to include:",
            options=sorted([company for company in DEFAULT_COMPANIES if company not in st.session_state.portfolio_companies]),
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Add", key="add_company"):
            if selected_company and selected_company not in st.session_state.portfolio_companies:
                st.session_state.portfolio_companies.append(selected_company)
    
    # Display current portfolio
    st.markdown("##### Your Portfolio")
    
    if not st.session_state.portfolio_companies:
        st.info("No companies added yet. Add companies to include in portfolio optimization.")
    else:
        portfolio_container = st.container()
        
        with portfolio_container:
            for idx, company in enumerate(st.session_state.portfolio_companies):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{company}**")
                with col2:
                    if st.button("✖", key=f"remove_{idx}", help="Remove this company"):
                        st.session_state.portfolio_companies.remove(company)
                        #st.experimental_rerun()
        
        # Option to clear portfolio
        if st.button("Clear Portfolio"):
            st.session_state.portfolio_companies = []
            #st.experimental_rerun()
    
    # Manually enter a ticker that's not in the predefined list
    st.markdown("##### Add custom ticker")
    custom_ticker = st.text_input("Enter custom ticker symbol:", key="custom_ticker")
    
    if st.button("Add Custom Ticker"):
        if custom_ticker and custom_ticker not in st.session_state.portfolio_companies:
            st.session_state.portfolio_companies.append(custom_ticker.upper())
            st.experimental_rerun()
    
    # Run analysis button
    if st.button("Run Analysis", type="primary"):
        st.session_state.is_analyzing = True

    st.divider()
    
    # Display information about the analysis
    st.subheader("About This Analysis")
    st.info(
        """
        This dashboard visualizes the results of the AdvancedStockAnalyzer. 
        The analysis includes:
        
        - Technical Analysis
        - Risk Metrics
        - Time Series Forecasting
        - Fundamental Analysis
        - Monte Carlo Simulation
        - Portfolio Optimization
        - Sentiment Analysis
        """
    )
    
    # Disclaimer
    st.caption(
        """
        Disclaimer: This analysis is for informational purposes only 
        and should not be considered financial advice. Always conduct 
        your own research before making investment decisions.
        """
    )

# Main dashboard content
st.markdown('<div class="main-header">Stock Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Investment Recommendation System</div>', unsafe_allow_html=True)

# Ejecutar análisis si se ha presionado el botón o mostrar indicador de carga
if st.session_state.is_analyzing:
    with st.spinner(f"Analyzing {ticker}... This may take a moment."):
        # Pasar el portafolio al constructor
        analyzer = AdvancedStockAnalyzer(ticker, portfolio_tickers=st.session_state.portfolio_companies)
        result = analyzer.run_full_analysis()
        st.session_state.analysis_result = result
        st.session_state.is_analyzing = False
        
        # Mostrar mensaje de éxito
        st.success(f"Analysis for {ticker} completed successfully!")

# Verificar si hay resultados disponibles
if 'analysis_result' in st.session_state:
    # Get result from session state
    result = st.session_state.analysis_result

    # Display main recommendation card
    recommendation_color = result['recommendation_color']
    st.markdown(
        f"""
        <div class="recommendation-box" style="background-color: {recommendation_color}20; border-left: 5px solid {recommendation_color};">
            <h2 style="color: {recommendation_color}; margin-top: 0;">{result['recommendation']} ({result['confidence']} Confidence)</h2>
            <h3>{result['ticker']} - {result['company_name']}</h3>
            <p style="font-size: 1.1rem;">Current Price: ${result['current_price']:.2f} | Analysis Date: {result['analysis_date']}</p>
            <p style="font-size: 1.1rem;">{result['recommendation_text']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create three columns for the score gauge, score breakdown, and key metrics
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Investment Score")
        
        # Create gauge chart for overall score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['total_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Total Score (0-100)"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                'bar': {'color': recommendation_color},
                'steps': [
                    {'range': [0, 35], 'color': "#FECACA"},  # Red - Sell
                    {'range': [35, 50], 'color': "#FED7AA"},  # Orange - Reduce
                    {'range': [50, 65], 'color': "#FEF08A"},  # Yellow - Hold
                    {'range': [65, 80], 'color': "#BBF7D0"},  # Light Green - Buy
                    {'range': [80, 100], 'color': "#86EFAC"}  # Green - Strong Buy
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': result['total_score']
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create horizontal bar chart for category scores
        scores_df = pd.DataFrame([
            {"Category": k, "Score": v, "Max Score": [20, 20, 15, 20, 15, 10, 10][i]}
            for i, (k, v) in enumerate(result['category_scores'].items())
        ])
        
        fig = px.bar(
            scores_df,
            x="Score",
            y="Category",
            color="Score",
            color_continuous_scale=px.colors.sequential.Blues,
            text="Score",
            orientation='h',
            height=400,
        )
        
        # Add reference line for max possible score in each category
        for i, row in scores_df.iterrows():
            fig.add_shape(
                type="line",
                x0=0,
                y0=i - 0.4,
                x1=row["Max Score"],
                y1=i - 0.4,
                line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dot"),
            )
        
        fig.update_layout(
            title="Score Breakdown by Category",
            xaxis_title="Score",
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Analysis Metrics")
        
        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(["Technical & Risk", "Fundamental & Time Series", "Monte Carlo & Portfolio", "Sentiment"])
        
        with tab1:
            # Changed to a single column layout to avoid nesting
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Technical Analysis</div>', unsafe_allow_html=True)
            
            # Create RSI gauge
            rsi = result['key_metrics']['technical']['rsi']
            rsi_color = "#22C55E" if 30 <= rsi <= 70 else "#DC2626"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "RSI"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': rsi_color},
                    'steps': [
                        {'range': [0, 30], 'color': "#FECACA"},
                        {'range': [30, 70], 'color': "#BBF7D0"},
                        {'range': [70, 100], 'color': "#FECACA"}
                    ]
                }
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics vertically instead of in nested columns
            st.metric("Trend", result['key_metrics']['technical']['trend'])
            st.metric("MACD Signal", result['key_metrics']['technical']['macd_signal'])
            st.markdown('</div>', unsafe_allow_html=True)

            # Risk metrics section
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Risk Metrics</div>', unsafe_allow_html=True)
            
            # Display metrics vertically
            st.metric("Sharpe Ratio", f"{result['key_metrics']['risk_metrics']['sharpe_ratio']:.2f}")
            st.metric("Value at Risk (95%)", f"{result['key_metrics']['risk_metrics']['var_95']:.2f}%")
            st.metric("Max Drawdown", f"{result['key_metrics']['risk_metrics']['max_drawdown']:.2f}%")
            
            # Visualize drawdown
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=result['key_metrics']['risk_metrics']['max_drawdown'],
                title={"text": "Maximum Drawdown"},
                delta={'reference': 0, 'relative': False, 'valueformat': '.1f%'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            
            fig.update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Fundamental Analysis</div>', unsafe_allow_html=True)
            
            col_a, col_b = st.columns([1,2])
            with col_a:
                st.metric("P/E Ratio", result['key_metrics']['fundamental']['pe_ratio'])
                st.metric("Profit Margins", f"{result['key_metrics']['fundamental']['profit_margins']:.2%}")
            with col_b:
                upside = result['key_metrics']['fundamental']['price_target_upside']
                st.metric("Price Target Upside", f"{upside:.2f}%", delta=f"{upside:.1f}%")
            
            # Add a visual for upside potential
            current_price = result['current_price']
            target_price = current_price * (1 + upside/100)
            
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=target_price,
                title={"text": "Price Target"},
                delta={'reference': current_price, 'relative': False, 'valueformat': '.2f'},
                number={'prefix': "$"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            
            fig.update_layout(height=150, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Time series metrics
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Time Series Forecast</div>', unsafe_allow_html=True)
            
            col_a, col_b = st.columns([1,2])
            with col_a:
                expected_return = result['key_metrics']['time_series']['expected_return_30d']
                st.metric("Expected 30-Day Return", f"{expected_return:.2f}%", delta=f"{expected_return:.1f}%")
            with col_b:
                st.metric("Returns Stationary", "Yes" if result['key_metrics']['time_series']['stationary'] else "No")
            
            # Add a simple forecast visualization
            days = list(range(30))
            forecasted_values = [current_price * (1 + (expected_return/100) * (i/30)) for i in range(31)]
            
            fig = px.line(
                x=days, 
                y=forecasted_values[1:],
                labels={"x": "Days", "y": "Forecasted Price"},
                title="30-Day Price Forecast"
            )
            
            fig.add_hline(
                y=current_price, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Current Price", 
                annotation_position="bottom right"
            )
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            # Monte Carlo metrics
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Monte Carlo Simulation</div>', unsafe_allow_html=True)
            
            expected_return = result['key_metrics']['monte_carlo']['expected_return']
            std_final_price = result['key_metrics']['monte_carlo']['std_final_price']
            
            col_a, col_b = st.columns([1,2])
            with col_a:
                st.metric("Expected Return", f"{expected_return:.2f}%", delta=f"{expected_return:.1f}%")
            with col_b:
                st.metric("Price Std Dev", f"${std_final_price:.2f}")
            
            # Create a mock Monte Carlo simulation visualization
            import random
            
            # Generate 50 random paths for illustration
            num_paths = 50
            horizon = 252  # Trading days in a year
            paths = []
            
            for _ in range(num_paths):
                path = [current_price]
                for day in range(horizon):
                    # Simple random walk with drift
                    daily_return = np.random.normal(expected_return/252, std_final_price/np.sqrt(252)/current_price)
                    new_price = path[-1] * (1 + daily_return/100)
                    path.append(new_price)
                paths.append(path)
            
            # Convert to DataFrame for plotting
            days = list(range(horizon + 1))
            
            # Create the chart
            fig = go.Figure()
            
            # Add all simulation paths with low opacity
            for i, path in enumerate(paths):
                if i < 5:  # Only show a few for better visuals
                    fig.add_trace(go.Scatter(
                        x=days, 
                        y=path,
                        mode='lines',
                        line=dict(color='rgba(0,100,255,0.1)'),
                        showlegend=False
                    ))
            
            # Add a line for the current price
            fig.add_trace(go.Scatter(
                x=[0, horizon],
                y=[current_price, current_price],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Current Price'
            ))
            
            # Add a line for the expected final price
            expected_final_price = current_price * (1 + expected_return/100)
            fig.add_trace(go.Scatter(
                x=[0, horizon],
                y=[current_price, expected_final_price],
                mode='lines',
                line=dict(color='green'),
                name='Expected Path'
            ))
            
            fig.update_layout(
                title="Monte Carlo Price Simulation",
                xaxis_title="Trading Days",
                yaxis_title="Price ($)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=200,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Portfolio metrics
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Portfolio Optimization</div>', unsafe_allow_html=True)
            
            ticker_weight = result['key_metrics']['portfolio']['current_ticker_weight']
            
            # Display the recommended weight
            st.metric("Recommended Weight", f"{ticker_weight:.2f}%")
            
            # Mostrar empresas en el portafolio
            st.markdown("##### Portfolio Composition")

            # Asignar pesos reales 'optimal_weights': {'CLSK': np.float64(0.45888073593956846), 'AAPL': np.float64(4.7704895589362195e-18), 'MSFT': np.float64(0.04597262914416003), 'GOOGL': np.float64(0.07052332117112003), 'AMZN': np.float64(2.6020852139652106e-18), 'BRK-B': np.float64(0.42462331374515144)}
            ticker_weights = list(result['key_metrics']['portfolio']['optimal_weights'].values())
            tickers = list(result['key_metrics']['portfolio']['optimal_weights'].keys())
            # Crear la tabla de composición del portafolio
            if len(st.session_state.portfolio_companies) > 0:
                portfolio_df = pd.DataFrame({
                    "Ticker": tickers,
                    "Weight (%)": ticker_weights
                })
                st.dataframe(portfolio_df, use_container_width=True)

            
            # Create a pie chart for portfolio allocation
            labels = tickers if len(st.session_state.portfolio_companies) > 0 else [ticker, 'Cash/Other']
            values = ticker_weights if len(st.session_state.portfolio_companies) > 0 else [ticker_weight, 100-ticker_weight]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                textinfo='label+percent',
                marker_colors=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#6366F1'][:len(labels)]
            )])
            
            fig.update_layout(
                title="Optimal Portfolio Allocation",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Sentiment Analysis</div>', unsafe_allow_html=True)
            
            sentiment = result['key_metrics']['sentiment']['sentiment']
            sentiment_score = result['key_metrics']['sentiment']['sentiment_score']
            
            # Overall sentiment gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentiment Score"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#3B82F6"},
                    'steps': [
                        {'range': [0, 33], 'color': "#FECACA"},
                        {'range': [33, 66], 'color': "#FEF08A"},
                        {'range': [66, 100], 'color': "#BBF7D0"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_score * 100
                    }
                }
            ))
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Display sentiment information
            st.metric("Overall Sentiment", sentiment)
            
            # Create a simple visualization of sentiment
            fig = go.Figure()
            
            # Add a sentiment meter
            fig.add_trace(go.Indicator(
                mode="number+delta+gauge",
                value=sentiment_score * 100,
                delta={'reference': 50, 'position': "top"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Market Sentiment"},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [0, 100]},
                    'threshold': {
                        'line': {'color': "red", 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    },
                    'steps': [
                        {'range': [0, 25], 'color': "#EF4444"},
                        {'range': [25, 50], 'color': "#F59E0B"},
                        {'range': [50, 75], 'color': "#10B981"},
                        {'range': [75, 100], 'color': "#047857"}
                    ],
                    'bar': {'color': "#3B82F6"}
                }
            ))
            
            fig.update_layout(height=150, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment breakdown explanation
            if sentiment_score > 0.75:
                st.success("Extremely positive market sentiment with strong bullish indicators from news and social media.")
            elif sentiment_score > 0.6:
                st.info("Positive market sentiment with favorable mentions in financial news and investor discussion forums.")
            elif sentiment_score > 0.4:
                st.warning("Neutral market sentiment with mixed signals from various sources.")
            else:
                st.error("Negative market sentiment with concerning signals from news and social media analysis.")
        
            st.markdown('</div>', unsafe_allow_html=True)
else:
    # Mostrar mensaje inicial cuando no hay resultados
    st.info("Enter a ticker symbol and click 'Run Analysis' to get started.")
    
    # Explicación de cómo usar la selección de portafolio
    st.markdown("""
    ### How to use Portfolio Selection
    
    1. Enter a main ticker symbol in the sidebar
    2. Add additional companies to include in portfolio optimization
    3. Click "Run Analysis" to see results
    
    The analysis will calculate the optimal allocation across your selected securities.
    """)

# Footer
st.markdown('<div class="footer">Analysis powered by Advanced Stock Analyzer | © 2025 Stock Analytics</div>', unsafe_allow_html=True)