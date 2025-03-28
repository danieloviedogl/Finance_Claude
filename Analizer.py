import yfinance as yf
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class AdvancedStockAnalyzer:
    """
    A comprehensive stock analysis tool that combines technical, fundamental,
    statistical, and machine learning approaches to provide investment recommendations.
    """
    
    def __init__(self, ticker, start_date='2018-01-01', benchmark='SPY', portfolio_tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B']):
        """Initialize with stock ticker and analysis parameters"""
        self.ticker = ticker
        self.start_date = start_date
        self.benchmark = benchmark
        self.portfolio_tickers = portfolio_tickers
        self.data = None
        self.benchmark_data = None
        self.results = {}
    
    def fetch_data(self):
        """Fetch historical stock data and benchmark data"""
        print(f"Fetching data for {self.ticker}...")
        
        # Adding parameter group_by='ticker' to ensure proper structure
        self.data = yf.download(self.ticker, start=self.start_date)
        self.benchmark_data = yf.download(self.benchmark, start=self.start_date)
        
        # Get fundamental data
        self.stock_info = yf.Ticker(self.ticker).info
        self.company_name = self.stock_info.get('longName', self.ticker)
        
        # Calculate daily returns
        self.data['Returns'] = self.data['Close'].pct_change()
        self.benchmark_data['Returns'] = self.benchmark_data['Close'].pct_change()
        
        # Create individual Series first to ensure 1D structure
        price_series = self.data['Close'].squeeze()  # squeeze converts DataFrame with 1 column to Series
        returns_series = self.data['Returns'].squeeze()
        benchmark_returns_series = self.benchmark_data['Returns'].squeeze()
        
        # Merge with benchmark for comparison
        self.merged_data = pd.DataFrame({
            f'{self.ticker}_price': price_series,
            f'{self.ticker}_returns': returns_series,
            f'{self.benchmark}_returns': benchmark_returns_series
        }).dropna()
        
        # Fetch financial statements if available
        try:
            self.quarterly_financials = yf.Ticker(self.ticker).quarterly_financials
            self.balance_sheet = yf.Ticker(self.ticker).balance_sheet
            self.cash_flow = yf.Ticker(self.ticker).cash_flow
        except:
            print("Full financial statements not available, continuing with price data only")
        
        return self.data
    
    def perform_technical_analysis(self):
        """Calculate technical indicators"""
        data = self.data.copy()
        
        # Moving averages
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands
        data['20d_rolling_std'] = data['Close'].rolling(window=20).std()
        data['upper_band'] = data['SMA_50'] + (data['20d_rolling_std'] * 2)
        data['lower_band'] = data['SMA_50'] - (data['20d_rolling_std'] * 2)
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - \
                        data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # OBV (On-Balance Volume)
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        
        # Momentum
        data['Momentum'] = data['Close'] / data['Close'].shift(10)
        
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Technical signals
        data['MA_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1)
        data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
        data['MACD_Signal_Line'] = np.where(data['MACD'] > data['MACD_Signal'], 1, -1)
        
        # Combined signal (simple equal weighting)
        data['Tech_Signal'] = data[['MA_Signal', 'RSI_Signal', 'MACD_Signal_Line']].mean(axis=1)
        
        self.tech_data = data.copy()
        self.results['technical'] = {
            'current_price': data['Close'].iloc[-1],
            'current_rsi': data['RSI'].iloc[-1],
            'ma_signal': data['MA_Signal'].iloc[-1],
            'macd_signal': data['MACD_Signal_Line'].iloc[-1],
            'combined_signal': data['Tech_Signal'].iloc[-1]
        }
        
        return self.tech_data
    
    def calculate_risk_metrics(self):
        """Calculate risk metrics and ratios"""
        # Crear un DataFrame combinado para hacer dropna en ambos a la vez
        combined = pd.DataFrame({
            'returns': self.data['Returns'],
            'benchmark_returns': self.benchmark_data['Returns']
        })
        
        # Eliminar filas donde cualquiera de las dos series tenga NaN
        combined = combined.dropna()
        
        # Extraer las series limpias
        returns = combined['returns']
        benchmark_returns = combined['benchmark_returns']
        
        # Beta calculation
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance
        
        # Alpha calculation (CAPM)
        risk_free_rate = 0.045  # Approximate current rate
        expected_return = risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate)
        alpha = returns.mean() * 252 - expected_return
        
        # Sharpe Ratio
        sharpe = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))
        
        # Sortino Ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() * 252 - risk_free_rate) / (downside_returns.std() * np.sqrt(252))
        
        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Information Ratio
        active_return = returns - benchmark_returns
        information_ratio = (active_return.mean() * 252) / (active_return.std() * np.sqrt(252))
        
        # Calmar Ratio
        calmar = (returns.mean() * 252) / abs(max_drawdown)
        
        # Treynor Ratio
        treynor = (returns.mean() * 252 - risk_free_rate) / beta if beta != 0 else np.nan
        
        self.results['risk_metrics'] = {
            'beta': beta,
            'alpha': alpha,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar,
            'treynor_ratio': treynor,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'annualized_return': returns.mean() * 252
        }
        
        return self.results['risk_metrics']
    
    def perform_time_series_analysis(self):
        """Perform time series analysis and forecasting"""
        # Use log returns for better statistical properties
        log_returns = np.log(1 + self.data['Returns']).dropna()
        
        # Stationarity test (Augmented Dickey-Fuller)
        adf_result = adfuller(log_returns)
        
        # ARIMA model for returns forecasting
        try:
            # Determine best order based on AIC
            best_aic = float('inf')
            best_order = None
            for p in range(5):
                for d in range(2):
                    for q in range(5):
                        try:
                            model = ARIMA(log_returns, order=(p, d, q))
                            model_fit = model.fit()
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Fit the best model
            best_model = ARIMA(log_returns, order=best_order)
            best_model_fit = best_model.fit()
            
            # Forecast next 30 days
            forecast = best_model_fit.forecast(steps=30)
            
            # Convert log returns forecast to price forecast
            last_price = self.data['Close'].iloc[-1]
            price_forecast = [last_price]
            for ret in forecast:
                price_forecast.append(price_forecast[-1] * np.exp(ret))
            
            self.results['time_series'] = {
                'is_stationary': bool(adf_result[1] < 0.05),  # Convertir a bool explícitamente
                'adf_pvalue': float(adf_result[1]),  # Convertir a float
                'best_arima_order': best_order,
                'forecast_30d': price_forecast[1:],
                'expected_price_30d': float(price_forecast[-1]),  # Convertir a float
                'expected_return_30d': float((price_forecast[-1] / last_price - 1) * 100)  # Convertir a float
            }
            
            # GARCH model for volatility forecasting
            try:
                garch_model = arch_model(log_returns * 100, vol='GARCH', p=1, q=1)
                garch_result = garch_model.fit(disp='off')
                garch_forecast = garch_result.forecast(horizon=30)
                forecast_variance = garch_forecast.variance.iloc[-1].mean()
                
                self.results['time_series']['volatility_forecast_30d'] = np.sqrt(forecast_variance)
            except:
                print("GARCH modeling failed, skipping volatility forecast")
                
        except Exception as e:
            print(f"Time series analysis error: {e}")
            self.results['time_series'] = {
                'is_stationary': adf_result[1] < 0.05,
                'adf_pvalue': adf_result[1],
                'forecast_error': str(e)
            }
        
        return self.results['time_series']
    
    def calculate_fundamental_metrics(self):
        """Calculate and evaluate fundamental metrics"""
        try:
            # Basic valuation metrics
            self.results['fundamental'] = {
                'pe_ratio': self.stock_info.get('trailingPE', np.nan),
                'forward_pe': self.stock_info.get('forwardPE', np.nan),
                'peg_ratio': self.stock_info.get('pegRatio', np.nan),
                'price_to_book': self.stock_info.get('priceToBook', np.nan),
                'enterprise_value': self.stock_info.get('enterpriseValue', np.nan),
                'enterprise_to_revenue': self.stock_info.get('enterpriseToRevenue', np.nan),
                'enterprise_to_ebitda': self.stock_info.get('enterpriseToEbitda', np.nan),
                'profit_margins': self.stock_info.get('profitMargins', np.nan),
                'dividend_yield': self.stock_info.get('dividendYield', 0) * 100 if self.stock_info.get('dividendYield') else 0,
                'debt_to_equity': self.stock_info.get('debtToEquity', np.nan),
                'return_on_equity': self.stock_info.get('returnOnEquity', np.nan),
                'current_ratio': self.stock_info.get('currentRatio', np.nan),
                'quick_ratio': self.stock_info.get('quickRatio', np.nan),
                'target_median_price': self.stock_info.get('targetMedianPrice', np.nan),
                'analyst_recommendation': self.stock_info.get('recommendationMean', np.nan)
            }
            
            # Calculate potential upside/downside based on analyst targets
            if pd.notna(self.results['fundamental']['target_median_price']):
                current_price = self.data['Close'].iloc[-1]
                self.results['fundamental']['price_target_upside'] = float((
                    self.results['fundamental']['target_median_price'] / current_price - 1
                ) * 100)
            
        except Exception as e:
            print(f"Fundamental analysis error: {e}")
            self.results['fundamental'] = {'error': str(e)}
        
        return self.results['fundamental']
    
    def perform_monte_carlo_simulation(self, simulations=1000, days=252):
        """Perform Monte Carlo simulation to estimate price range"""
        returns = self.data['Returns'].dropna()
        
        # Parameters for geometric Brownian motion
        mu = returns.mean()
        sigma = returns.std()
        
        # Simulate future price paths
        last_price = float(self.data['Close'].iloc[-1])  # Convert to float to avoid Series issues
        simulation_df = pd.DataFrame()
        
        for i in range(simulations):
            # Daily returns are random normal with our mu and sigma
            daily_returns = np.random.normal(mu, sigma, days) + 1
            
            # Cumulative product of returns starting at last price
            price_series = last_price * np.cumprod(daily_returns)  # Use numpy's cumprod instead
            simulation_df[i] = price_series
        
        # Calculate statistics from simulations
        self.results['monte_carlo'] = {
            'simulations': simulations,
            'days': days,
            'mean_final_price': float(simulation_df.iloc[-1].mean()),
            'median_final_price': float(simulation_df.iloc[-1].median()),
            'min_final_price': float(simulation_df.iloc[-1].min()),
            'max_final_price': float(simulation_df.iloc[-1].max()),
            'std_final_price': float(simulation_df.iloc[-1].std()),
            'current_price': last_price,
            'expected_return': float((simulation_df.iloc[-1].mean() / last_price - 1) * 100),
            'percentiles': {
                '5%': np.percentile(simulation_df.iloc[-1], 5),
                '25%': np.percentile(simulation_df.iloc[-1], 25),
                '50%': np.percentile(simulation_df.iloc[-1], 50),
                '75%': np.percentile(simulation_df.iloc[-1], 75),
                '95%': np.percentile(simulation_df.iloc[-1], 95)
            }
        }
        
        self.simulation_results = simulation_df
        return self.results['monte_carlo']
    
    def calculate_optimal_portfolio(self):
        """Calculate optimal portfolio using Modern Portfolio Theory"""
        # Add the current ticker to the list if not already included
        if self.ticker not in self.portfolio_tickers:
            tickers = [self.ticker] + self.portfolio_tickers
        else:
            tickers = self.portfolio_tickers.copy()
        
        # Download historical data for all tickers
        portfolio_data = yf.download(tickers, start=self.start_date)['Close']
        
        # Calculate returns
        returns = portfolio_data.pct_change().dropna()
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Function to calculate portfolio performance
        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns, std
        
        # Function to minimize negative Sharpe Ratio
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.045):
            p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
            return -(p_returns - risk_free_rate) / p_std
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(len(tickers)))
        
        # Initial guess (equal weighting)
        initial_guess = np.array([1/len(tickers)] * len(tickers))
        
        # Optimize portfolio
        optimal_result = minimize(neg_sharpe_ratio, initial_guess, 
                                  args=(mean_returns, cov_matrix),
                                  method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Extract results
        optimal_weights = optimal_result['x']
        optimal_returns, optimal_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        optimal_sharpe = (optimal_returns - 0.045) / optimal_std
        
        # Calculate efficient frontier
        target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, 50)
        efficient_portfolios = []
        
        for target in target_returns:
            # Function to minimize portfolio variance
            def portfolio_variance(weights, mean_returns, cov_matrix, target_return):
                returns = np.sum(mean_returns * weights)
                variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                penalty = 100 * abs(returns - target_return)
                return variance + penalty
            
            # New constraint ensuring target return
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                         {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target})
            
            result = minimize(portfolio_variance, initial_guess,
                            args=(mean_returns, cov_matrix, target),
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result['success']:
                efficient_portfolios.append({
                    'return': np.sum(mean_returns * result['x']),
                    'volatility': np.sqrt(np.dot(result['x'].T, np.dot(cov_matrix, result['x']))),
                    'weights': result['x']
                })
        
        self.results['portfolio'] = {
            'tickers': tickers,
            'optimal_weights': dict(zip(tickers, optimal_weights)),
            'optimal_return': optimal_returns,
            'optimal_volatility': optimal_std,
            'optimal_sharpe': optimal_sharpe,
            'current_ticker_weight': dict(zip(tickers, optimal_weights))[self.ticker],
            'efficient_frontier': efficient_portfolios,
            'portfolio_recommendation': "Include" if dict(zip(tickers, optimal_weights))[self.ticker] > 0.05 else "Exclude"
        }
        
        return self.results['portfolio']
    
    def sentiment_analysis(self):
        """Perform basic sentiment analysis using recent news and ratings"""
        try:
            # Get news
            ticker_obj = yf.Ticker(self.ticker)
            news = ticker_obj.news
            
            # Just get analyst recommendations
            recommendations = ticker_obj.recommendations
            
            if recommendations is not None:
                # Simple analysis of recommendations
                rec_counts = recommendations.iloc[0]
                positive_recs = rec_counts['buy'] + rec_counts['strongBuy']
                negative_recs = rec_counts['sell'] + rec_counts['strongSell']
                neutral_recs = rec_counts['hold']
                
                total_recs = positive_recs + negative_recs + neutral_recs
                sentiment_score = (positive_recs - negative_recs) / total_recs if total_recs > 0 else 0
                
                sentiment_result = {
                    'total_recommendations': total_recs,
                    'positive_recommendations': positive_recs,
                    'negative_recommendations': negative_recs,
                    'neutral_recommendations': neutral_recs,
                    'sentiment_score': sentiment_score,
                    'sentiment': 'Positive' if sentiment_score > 0.2 else ('Negative' if sentiment_score < -0.2 else 'Neutral')
                }
            else:
                sentiment_result = {
                    'sentiment': 'Neutral',
                    'sentiment_score': 0,
                    'note': 'No recent analyst recommendations available'
                }
            
            self.results['sentiment'] = sentiment_result
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            self.results['sentiment'] = {
                'sentiment': 'Neutral',
                'sentiment_score': 0,
                'error': str(e)
            }
        
        return self.results['sentiment']
    
    def backtest_portfolio(self, years_back=3):
        """
        Backtest the recommended portfolio performance for the specified number of years given that the object has ran the run_full_analysis function.
        
        Args:
            years_back (int): Number of years to backtrack for portfolio performance analysis
        
        Returns:
            dict: Portfolio backtest results including annualized return, volatility, and comparisons
        """
        # Ensure we have the portfolio tickers
        if not hasattr(self, 'results') or 'portfolio' not in self.results:
            raise ValueError("Run calculate_optimal_portfolio() first to get portfolio composition")
        
        # Get portfolio composition
        portfolio_tickers = list(self.results['portfolio']['optimal_weights'].keys())
        portfolio_weights = list(self.results['portfolio']['optimal_weights'].values())
        
        # Define start date for backtest
        backtest_start_date = pd.to_datetime(self.start_date) - pd.DateOffset(years=years_back)
        
        # Download historical prices
        historical_prices = yf.download(portfolio_tickers, start=backtest_start_date, end=self.start_date)['Close']
        
        # Calculate returns
        returns = historical_prices.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(portfolio_weights)
        
        # Annualize returns
        annualized_return = (1 + portfolio_returns).prod() ** (1/years_back) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)  # Assuming 252 trading days
        
        # Sharpe Ratio (using risk-free rate of 4.5%)
        risk_free_rate = 0.045
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        # Benchmark comparison (using SPY as benchmark)
        benchmark_returns = yf.download('SPY', start=backtest_start_date, end=self.start_date)['Close'].pct_change().dropna()
        benchmark_annualized_return = (1 + benchmark_returns).prod() ** (1/years_back) - 1
        benchmark_annualized_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Prepare backtest results
        backtest_results = {
            'portfolio_tickers': portfolio_tickers,
            'portfolio_weights': dict(zip(portfolio_tickers, portfolio_weights)),
            'annualized_return': float(annualized_return * 100),  # Percentage
            'annualized_volatility': float(annualized_volatility * 100),  # Percentage
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown * 100),  # Percentage
            'benchmark_comparison': {
                'benchmark': 'SPY',
                'benchmark_annualized_return': float(benchmark_annualized_return * 100),
                'benchmark_annualized_volatility': float(benchmark_annualized_volatility * 100),
                'outperformance': float((annualized_return - benchmark_annualized_return) * 100)
            }
        }
        
        # Store results for future reference
        if 'portfolio_backtest' not in self.results:
            self.results['portfolio_backtest'] = {}
        self.results['portfolio_backtest'][f'{years_back}_years'] = backtest_results
        
        return backtest_results
    
    def run_full_analysis(self):
        """Run all analyses and generate comprehensive results"""
        self.fetch_data()
        self.perform_technical_analysis()
        self.calculate_risk_metrics()
        self.perform_time_series_analysis()
        self.calculate_fundamental_metrics()
        self.perform_monte_carlo_simulation()
        self.calculate_optimal_portfolio()
        self.sentiment_analysis()
        
        return self.generate_investment_recommendation()
    
    def generate_investment_recommendation(self):
        """Generate final investment recommendation based on all analyses"""
        # Current price and basic information
        current_price = float(self.data['Close'].iloc[-1])
        
        # Scoring system (0-100)
        scores = {}
        
        # Technical analysis score (0-20)
        tech_score = 0
        if self.results['technical']['ma_signal'] > 0:
            tech_score += 7
        if 30 <= self.results['technical']['current_rsi'] <= 70:
            tech_score += 5  # Neutral RSI is good
        elif self.results['technical']['current_rsi'] < 30:
            tech_score += 7  # Oversold could mean buying opportunity
        if self.results['technical']['macd_signal'] > 0:
            tech_score += 6
        scores['technical'] = tech_score
        
        # Risk metrics score (0-20)
        risk_score = 0
        if self.results['risk_metrics']['sharpe_ratio'] > 1:
            risk_score += 5
        elif self.results['risk_metrics']['sharpe_ratio'] > 0.5:
            risk_score += 3
        
        if self.results['risk_metrics']['sortino_ratio'] > 1:
            risk_score += 5
        elif self.results['risk_metrics']['sortino_ratio'] > 0.5:
            risk_score += 3
        
        if self.results['risk_metrics']['max_drawdown'] > -0.2:
            risk_score += 5
        elif self.results['risk_metrics']['max_drawdown'] > -0.3:
            risk_score += 3
        
        if self.results['risk_metrics']['alpha'] > 0:
            risk_score += 5
        scores['risk'] = risk_score
        
        # Time series forecast score (0-15)
        time_series_score = 0
        if 'expected_return_30d' in self.results['time_series'] and self.results['time_series']['expected_return_30d'] > 5:
            time_series_score += 10
        elif 'expected_return_30d' in self.results['time_series'] and self.results['time_series']['expected_return_30d'] > 2:
            time_series_score += 5
        
        if 'is_stationary' in self.results['time_series'] and self.results['time_series']['is_stationary']:
            time_series_score += 5  # Stationary returns are more predictable
        scores['time_series'] = time_series_score
        
        # Fundamental analysis score (0-20)
        fundamental_score = 0
        if 'pe_ratio' in self.results['fundamental'] and pd.notna(self.results['fundamental']['pe_ratio']):
            sector_avg_pe = 20  # Simplified, would ideally use sector-specific average
            pe_ratio = self.results['fundamental']['pe_ratio']
            if pe_ratio == 'Infinity':
                fundamental_score += 0
            elif self.results['fundamental']['pe_ratio'] < sector_avg_pe:
                fundamental_score += 5
            else:
                fundamental_score += 2
        
        if 'price_target_upside' in self.results['fundamental'] and pd.notna(self.results['fundamental']['price_target_upside']):
            if self.results['fundamental']['price_target_upside'] > 15:
                fundamental_score += 8
            elif self.results['fundamental']['price_target_upside'] > 5:
                fundamental_score += 4
        
        if 'profit_margins' in self.results['fundamental'] and pd.notna(self.results['fundamental']['profit_margins']):
            if self.results['fundamental']['profit_margins'] > 0.15:
                fundamental_score += 7
            elif self.results['fundamental']['profit_margins'] > 0.08:
                fundamental_score += 4
        scores['fundamental'] = fundamental_score
        
        # Monte Carlo simulation score (0-15)
        mc_score = 0
        expected_return = self.results['monte_carlo']['expected_return']
        if expected_return > 15:
            mc_score += 10
        elif expected_return > 8:
            mc_score += 7
        elif expected_return > 0:
            mc_score += 3
        
        risk_reward = float(expected_return / (self.results['monte_carlo']['std_final_price'] / current_price * 100))
        if risk_reward > 1:
            mc_score += 5
        elif risk_reward > 0.5:
            mc_score += 3
        scores['monte_carlo'] = mc_score
        
        # Portfolio optimization score (0-10)
        portfolio_score = 0
        if self.results['portfolio']['current_ticker_weight'] > 0.15:
            portfolio_score += 10
        elif self.results['portfolio']['current_ticker_weight'] > 0.10:
            portfolio_score += 7
        elif self.results['portfolio']['current_ticker_weight'] > 0.05:
            portfolio_score += 4
        scores['portfolio'] = portfolio_score
        
        # Sentiment analysis score (0-10)
        sentiment_score = 0
        if self.results['sentiment']['sentiment'] == 'Positive':
            sentiment_score += 8
        elif self.results['sentiment']['sentiment'] == 'Neutral':
            sentiment_score += 5
        else:
            sentiment_score += 2
        
        if 'sentiment_score' in self.results['sentiment']:
            sentiment_score += int(self.results['sentiment']['sentiment_score'] * 2)
        scores['sentiment'] = sentiment_score if sentiment_score <= 10 else 10
        
        # Total score
        total_score = sum(scores.values())
        
        # Generate recommendation based on total score
        if total_score >= 80:
            recommendation = "Strong Buy"
            confidence = "High"
            color = "#15803D"  # green
            recommendation_text = f"Strong Buy recommendation for {self.company_name} with high confidence. The stock shows excellent technical indicators, strong fundamentals, and favorable risk-reward metrics. Our analysis suggests significant potential for capital appreciation in both short and long-term horizons."
        elif total_score >= 65:
            recommendation = "Buy"
            confidence = "Moderate"
            color = "#65A30D"  # light green
            recommendation_text = f"Buy recommendation for {self.company_name} with moderate confidence. Our analysis indicates positive momentum, reasonable valuation, and favorable growth prospects. The stock appears to offer an attractive opportunity with manageable downside risk."
        elif total_score >= 50:
            recommendation = "Hold"
            confidence = "Moderate"
            color = "#CA8A04"  # yellow
            recommendation_text = f"Hold recommendation for {self.company_name}. While the stock shows some positive indicators, there are also potential concerns or limitations. Current investors should maintain positions, but new investment should be approached with caution pending further developments."
        elif total_score >= 35:
            recommendation = "Reduce"
            confidence = "Moderate"
            color = "#EA580C"  # orange
            recommendation_text = f"Reduce recommendation for {self.company_name}. Our analysis indicates potential downside risks outweighing upside potential. Investors might consider gradually reducing position size while looking for more favorable entry points in the future."
        else:
            recommendation = "Sell"
            confidence = "High"
            color = "#DC2626"  # red
            recommendation_text = f"Sell recommendation for {self.company_name} with high confidence. Multiple indicators suggest significant downside risk with limited upside potential. The risk-reward profile is unfavorable based on both technical and fundamental analysis."
        
        # Compile final recommendation with key metrics
        final_recommendation = {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'recommendation': recommendation,
            'recommendation_text': recommendation_text,
            'recommendation_color': color,
            'confidence': confidence,
            'total_score': total_score,
            'category_scores': scores,
            'current_price': current_price,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'key_metrics': {
                'technical': {
                    'rsi': round(self.results['technical']['current_rsi'], 2),
                    'trend': 'Bullish' if self.results['technical']['ma_signal'] > 0 else 'Bearish',
                    'macd_signal': 'Bullish' if self.results['technical']['macd_signal'] > 0 else 'Bearish'
                },
                'risk_metrics': {
                    'sharpe_ratio': round(self.results['risk_metrics']['sharpe_ratio'], 2),
                    'max_drawdown': round(self.results['risk_metrics']['max_drawdown'] * 100, 2),
                    'var_95': round(self.results['risk_metrics']['var_95'] * 100, 2)
                },
                'time_series': {
                    'stationary': self.results['time_series']['is_stationary'],
                    'expected_return_30d': round(self.results['time_series']['expected_return_30d'], 2)
                },
                'fundamental': {
                    'pe_ratio': self.results['fundamental'].get('pe_ratio', np.nan),
                    'price_target_upside': round(self.results['fundamental'].get('price_target_upside', np.nan), 2),
                    'profit_margins': round(self.results['fundamental'].get('profit_margins', np.nan), 4)
                },
                'monte_carlo': {
                    'expected_return': round(self.results['monte_carlo']['expected_return'], 2),
                    'std_final_price': round(self.results['monte_carlo']['std_final_price'], 2)
                },
                'portfolio': {
                    'current_ticker_weight': round(self.results['portfolio']['current_ticker_weight'] * 100, 2),
                    'optimal_weights': self.results['portfolio']['optimal_weights']
                },
                'sentiment': {
                    'sentiment': self.results['sentiment']['sentiment'],
                    'sentiment_score': round(self.results['sentiment']['sentiment_score'], 2)
                }
            }
        }
        print('Se generó la recomendación final/n')
        print(final_recommendation)
        return final_recommendation

if __name__ == "__main__":
    # Example usage
    analyzer = AdvancedStockAnalyzer('AAPL', portfolio_tickers= ['ADBE','CSCO','IBM','KO','AMD','NVDA','TSLA','MSFT','GOOGL','AMZN'])
    result = analyzer.run_full_analysis()
    backtest_result = analyzer.backtest_portfolio()
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print(json.dumps(backtest_result, indent=4, ensure_ascii=False))