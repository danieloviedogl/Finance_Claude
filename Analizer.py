# Analyzer.py (Improved Version)
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
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Default list, can be overridden
DEFAULT_PORTFOLIO_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B']

class AdvancedStockAnalyzer:
    """
    A comprehensive stock analysis tool that combines technical, fundamental,
    statistical, and machine learning approaches to provide investment recommendations.
    """
    # start_date = str = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')
    def __init__(self, ticker: str, start_date: str = '2025-01-01',
                 benchmark: str = 'SPY', risk_free_rate: float = 0.045,
                 portfolio_tickers: Optional[List[str]] = None):
        """
        Initialize with stock ticker and analysis parameters.

        Args:
            ticker (str): The main stock ticker symbol to analyze.
            start_date (str): Start date for historical data (YYYY-MM-DD).
            benchmark (str): Ticker symbol for the benchmark index (e.g., 'SPY').
            risk_free_rate (float): Annual risk-free rate for calculations (e.g., 0.045 for 4.5%).
            portfolio_tickers (Optional[List[str]]): List of tickers for portfolio optimization.
                                                    Defaults to a predefined list if None.
        """
        self.ticker: str = ticker.upper()
        self.start_date: str = start_date
        self.benchmark: str = benchmark.upper()
        self.risk_free_rate: float = risk_free_rate
        # Use default if None/empty, otherwise use provided list, ensuring the main ticker is included
        _initial_portfolio = portfolio_tickers if portfolio_tickers else DEFAULT_PORTFOLIO_TICKERS
        self.portfolio_tickers: List[str] = sorted(list(set([self.ticker] + [pt.upper() for pt in _initial_portfolio])))

        self.data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.stock_info: Dict[str, Any] = {}
        self.company_name: str = self.ticker
        self.tech_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        self.simulation_results: Optional[pd.DataFrame] = None
        self.quarterly_financials: Optional[pd.DataFrame] = None
        self.balance_sheet: Optional[pd.DataFrame] = None
        self.cash_flow: Optional[pd.DataFrame] = None

        # Use a dictionary to store all results cleanly
        self.results: Dict[str, Any] = {}

    def _to_serializable(self, value: Any) -> Any:
        """Converts numpy types to standard Python types for JSON compatibility."""
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        elif isinstance(value, (np.float64, np.float32, np.float16)):
            # Handle potential NaN or Inf values before conversion
            if pd.isna(value):
                return None # Or np.nan if downstream code handles it
            elif np.isinf(value):
                return None # Or represent as string 'Infinity'/-'Infinity' if needed
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return self._to_serializable(value.tolist()) # Recursive for arrays inside
        elif isinstance(value, (pd.Timestamp, pd.Period)):
             return value.isoformat() # Convert timestamps to ISO format strings
        elif isinstance(value, list):
            return [self._to_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        # Handle pandas Series or DataFrame if needed, e.g., convert to list/dict
        # elif isinstance(value, pd.Series):
        #     return self._to_serializable(value.to_list())
        # elif isinstance(value, pd.DataFrame):
        #     return self._to_serializable(value.to_dict(orient='records'))
        return value # Return unchanged if not a numpy/pandas type needing conversion

    # In Analyzer.py

    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical stock data, benchmark data, and company info."""
        print(f"Fetching data for {self.ticker} and benchmark {self.benchmark}...")
        try:
            # Download data
            # Ensure group_by='column' (or default) which usually returns simple index for single ticker
            self.data = yf.download(self.ticker, start=self.start_date, progress=False, group_by='column', auto_adjust=False)
            self.benchmark_data = yf.download(self.benchmark, start=self.start_date, progress=False, group_by='column', auto_adjust=False)

            # --- Basic Data Validation ---
            if self.data is None or self.data.empty:
                print(f"Error: No data downloaded for ticker {self.ticker}")
                raise ValueError(f"No data for ticker {self.ticker}")
            if self.benchmark_data is None or self.benchmark_data.empty:
                print(f"Warning: No data downloaded for benchmark {self.benchmark}. Risk metrics requiring benchmark will be limited.")
                self.benchmark_data = pd.DataFrame(index=self.data.index, columns=['Adj Close', 'Returns'])

            # --- *** NEW: Flatten Column MultiIndex if Present *** ---
            if isinstance(self.data.columns, pd.MultiIndex):
                print("Detected MultiIndex columns for main ticker, flattening...")
                # Assumes the ticker is the second level (level=1)
                # Get the actual ticker name from the MultiIndex if needed, though self.ticker should match
                ticker_level_name = self.data.columns.names[1] if len(self.data.columns.names) > 1 else None
                if ticker_level_name:
                     self.data.columns = self.data.columns.droplevel(ticker_level_name)
                else: # Fallback if names aren't set, assume level 1 is ticker
                     self.data.columns = self.data.columns.droplevel(1)
                print(f"Flattened columns: {self.data.columns.tolist()}")

            # Also flatten benchmark data if necessary (less likely but good practice)
            if isinstance(self.benchmark_data.columns, pd.MultiIndex):
                 print("Detected MultiIndex columns for benchmark, flattening...")
                 ticker_level_name = self.benchmark_data.columns.names[1] if len(self.benchmark_data.columns.names) > 1 else None
                 if ticker_level_name:
                      self.benchmark_data.columns = self.benchmark_data.columns.droplevel(ticker_level_name)
                 else:
                      self.benchmark_data.columns = self.benchmark_data.columns.droplevel(1)


            # --- Get Company Info ---
            ticker_obj = yf.Ticker(self.ticker)
            self.stock_info = ticker_obj.info
            self.company_name = self.stock_info.get('longName', self.ticker)

            # --- Calculate Returns (Use Adjusted Close) ---
            # Ensure 'Adj Close' exists after potential flattening
            if 'Adj Close' not in self.data.columns:
                 print("Error: 'Adj Close' column missing after potential flattening. Check download structure.")
                 # Attempt fallback to 'Close' if 'Adj Close' is truly missing
                 price_col = 'Close' if 'Close' in self.data.columns else None
                 if not price_col: raise ValueError("Neither 'Adj Close' nor 'Close' found in data.")
                 print(f"Warning: Using '{price_col}' instead of 'Adj Close' for returns.")
            else:
                 price_col = 'Adj Close'

            self.data['Returns'] = self.data[price_col].pct_change()

            if not self.benchmark_data.empty:
                if 'Adj Close' in self.benchmark_data.columns:
                     self.benchmark_data['Returns'] = self.benchmark_data['Adj Close'].pct_change()
                elif 'Close' in self.benchmark_data.columns:
                     print(f"Warning: Using 'Close' for benchmark returns.")
                     self.benchmark_data['Returns'] = self.benchmark_data['Close'].pct_change()
                else:
                     self.benchmark_data['Returns'] = np.nan
            else:
                self.benchmark_data['Returns'] = np.nan


            # --- Create Merged Data using JOIN ---
            print("Preparing merged data...")
            # Use the determined price_col
            ticker_df = self.data[[price_col, 'Returns']].copy()
            ticker_df.columns = [f'{self.ticker}_price', f'{self.ticker}_returns']

            benchmark_df = self.benchmark_data[['Returns']].copy()
            benchmark_df.columns = [f'{self.benchmark}_returns']

            self.merged_data = ticker_df.join(benchmark_df, how='left')
            self.merged_data = self.merged_data.dropna(subset=[f'{self.ticker}_returns'])

            print(f"Merged data created with shape: {self.merged_data.shape}")
            if self.merged_data.empty:
                 print("Warning: Merged data is empty after join/dropna.")


            # --- Fetch Financial Statements ---
            # (Keep existing code here)
            try:
                self.quarterly_financials = ticker_obj.quarterly_financials
                self.balance_sheet = ticker_obj.balance_sheet
                self.cash_flow = ticker_obj.cash_flow
                print("Financial statements fetched.")
            except Exception as e:
                print(f"Warning: Could not fetch full financial statements for {self.ticker}: {e}")
                self.quarterly_financials = None
                self.balance_sheet = None
                self.cash_flow = None


            print("Data fetching complete.")
            return self.data

        except Exception as e:
            print(f"An error occurred during data fetching: {e}")
            import traceback
            traceback.print_exc()
            self.data = None
            self.benchmark_data = None
            self.merged_data = None
            self.stock_info = {}
            return None
        
    def perform_technical_analysis(self) -> Optional[Dict[str, Any]]:
        """Calculate technical indicators and generate simple signals."""
        if self.data is None or self.data.empty:
            print("Error: No data available for technical analysis.")
            return None
        if len(self.data) < 200:
             print("Warning: Insufficient data for 200-day SMA. Some indicators might be NaN.")

        print("Performing technical analysis...")
        data = self.data.copy() # Work on a copy

        # Moving averages
        data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Adj Close'].rolling(window=200).mean()
        data['EMA_20'] = data['Adj Close'].ewm(span=20, adjust=False).mean()

        # Bollinger Bands (using SMA 20 as center)
        try:
            # Calculate SMA 20 first
            sma_20 = data['Adj Close'].rolling(window=20).mean()

            # Calculate rolling std dev separately first
            rolling_std_20 = data['Adj Close'].rolling(window=20).std()

            # Now calculate the bands using the calculated Series
            data['upper_band'] = sma_20 + (rolling_std_20 * 2)
            data['lower_band'] = sma_20 - (rolling_std_20 * 2)

            # Optionally store the std dev if needed elsewhere *after* band calculation
            data['20d_rolling_std'] = rolling_std_20
            print("Bollinger Bands calculated successfully.") # Confirmation

        except Exception as e:
             print(f"Error calculating Bollinger Bands: {e}")
             # Assign NaN if calculation fails to avoid downstream errors
             data['upper_band'] = np.nan
             data['lower_band'] = np.nan
             data['20d_rolling_std'] = np.nan
             # Add traceback for detail
             import traceback
             traceback.print_exc()

        # RSI (Relative Strength Index)
        delta = data['Adj Close'].diff()
        gain = delta.where(delta > 0, 0.0).fillna(0.0)
        loss = -delta.where(delta < 0, 0.0).fillna(0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = data['RSI'].fillna(50) # Fill initial NaNs with neutral 50

        # MACD (Moving Average Convergence Divergence)
        ema_12 = data['Adj Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Adj Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # OBV (On-Balance Volume) - ensure Volume exists and is numeric
        if 'Volume' in data.columns and pd.api.types.is_numeric_dtype(data['Volume']):
            data['OBV'] = (np.sign(data['Adj Close'].diff().fillna(0)) * data['Volume']).cumsum()
        else:
             data['OBV'] = np.nan
             print("Warning: Volume data missing or non-numeric, OBV not calculated.")

        # Momentum (10-day)
        data['Momentum'] = data['Adj Close'] / data['Adj Close'].shift(10) - 1 # As percentage change

        # Average True Range (ATR) - requires High, Low, Close
        # Check if required columns exist
        required_cols = ['High', 'Low', 'Adj Close']
        if all(col in data.columns for col in required_cols):
            try:
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Adj Close'].shift())
                low_close = np.abs(data['Low'] - data['Adj Close'].shift())

                # Concatenate into a DataFrame
                ranges = pd.concat([high_low, high_close, low_close], axis=1)

                # Calculate max, np.nanmax returns a numpy array
                true_range_np = np.nanmax(ranges, axis=1)

                # Convert the numpy array back to a pandas Series WITH THE CORRECT INDEX
                true_range_series = pd.Series(true_range_np, index=data.index, name='TrueRange')

                # Now perform the rolling calculation on the Series
                data['ATR'] = true_range_series.rolling(14, min_periods=1).mean()
                print("ATR calculated successfully.") # Confirmation

            except Exception as e:
                print(f"Error calculating ATR: {e}")
                data['ATR'] = np.nan # Assign NaN on failure
                # Add traceback for detail
                import traceback
                traceback.print_exc()
        else:
            data['ATR'] = np.nan # Assign NaN if columns are missing
            print(f"Warning: Required columns ({', '.join(required_cols)}) missing for ATR calculation.")

        # Technical signals (use last available non-NaN value)
        data['MA_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, np.where(data['SMA_50'] < data['SMA_200'], -1, 0))
        data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
        data['MACD_Signal_Line'] = np.where(data['MACD'] > data['MACD_Signal'], 1, np.where(data['MACD'] < data['MACD_Signal'], -1, 0))

        # Combined signal (simple average, ignoring NaN signals)
        data['Tech_Signal'] = data[['MA_Signal', 'RSI_Signal', 'MACD_Signal_Line']].mean(axis=1, skipna=True)

        print('Data columns:')
        print(data.columns)
        print(f'Data type: {type(data)}')
        self.tech_data = data.dropna(subset=['Adj Close']) # Ensure we don't keep rows with no price

        # Store latest results, converting types
        self.results['technical'] = self._to_serializable({
            'current_price': data['Adj Close'].iloc[-1] if not data.empty else None,
            'current_rsi': data['RSI'].iloc[-1] if not data.empty and 'RSI' in data.columns else None,
            'ma_signal': data['MA_Signal'].iloc[-1] if not data.empty and 'MA_Signal' in data.columns and pd.notna(data['MA_Signal'].iloc[-1]) else 0,
            'macd_signal': data['MACD_Signal_Line'].iloc[-1] if not data.empty and 'MACD_Signal_Line' in data.columns and pd.notna(data['MACD_Signal_Line'].iloc[-1]) else 0,
            'combined_signal': data['Tech_Signal'].iloc[-1] if not data.empty and 'Tech_Signal' in data.columns else None
        })

        print("Technical analysis complete.")
        return self.results['technical']

    def calculate_risk_metrics(self) -> Optional[Dict[str, Any]]:
        """Calculate risk metrics and ratios relative to the benchmark."""
        if self.merged_data is None or self.merged_data.empty or len(self.merged_data) < 2:
            print("Error: Insufficient merged data for risk metric calculation.")
            return None
        if f'{self.benchmark}_returns' not in self.merged_data.columns:
             print("Warning: Benchmark returns not available. Some risk metrics (Beta, Alpha, Info Ratio) cannot be calculated.")


        print("Calculating risk metrics...")
        returns = self.merged_data[f'{self.ticker}_returns']
        benchmark_returns = self.merged_data.get(f'{self.benchmark}_returns', None) # Use .get

        # Metrics requiring benchmark
        beta = alpha = information_ratio = treynor = np.nan
        if benchmark_returns is not None and not benchmark_returns.isnull().all():
            # Ensure returns and benchmark_returns are aligned and have enough non-NaN pairs
            valid_idx = returns.notna() & benchmark_returns.notna()
            if valid_idx.sum() >= 2: # Need at least 2 points for covariance
                returns_valid = returns[valid_idx]
                benchmark_returns_valid = benchmark_returns[valid_idx]

                # Beta calculation
                # Using statsmodels OLS for potentially more robust calculation incl. intercept (alpha)
                X = sm.add_constant(benchmark_returns_valid) # Add intercept column
                y = returns_valid
                try:
                    model = sm.OLS(y, X).fit()
                    alpha_daily, beta = model.params # Intercept is daily alpha, slope is beta
                    # Annualize alpha: (1 + daily_alpha)^252 - 1 OR approx daily_alpha * 252
                    alpha = alpha_daily * 252 # Simple annualization
                    # alpha = (1 + alpha_daily)**252 - 1 # Compound annualization

                    # Information Ratio
                    active_return = returns_valid - benchmark_returns_valid
                    if active_return.std() > 1e-9: # Avoid division by zero
                         information_ratio = (active_return.mean() * 252) / (active_return.std() * np.sqrt(252))

                    # Treynor Ratio
                    annualized_return = returns.mean() * 252
                    if beta != 0:
                        treynor = (annualized_return - self.risk_free_rate) / beta

                except Exception as e:
                    print(f"Warning: OLS regression for Beta/Alpha failed: {e}. Using covariance method for Beta.")
                    # Fallback to covariance method if OLS fails
                    covariance = np.cov(returns_valid, benchmark_returns_valid)[0, 1]
                    benchmark_variance = np.var(benchmark_returns_valid)
                    if benchmark_variance > 1e-9:
                        beta = covariance / benchmark_variance
                        # Simple CAPM alpha if OLS failed
                        expected_return = self.risk_free_rate + beta * (benchmark_returns_valid.mean() * 252 - self.risk_free_rate)
                        alpha = returns_valid.mean() * 252 - expected_return
                    else:
                        beta = np.nan
                        alpha = np.nan
            else:
                print("Warning: Not enough overlapping data points to calculate Beta/Alpha.")


        # Metrics not requiring benchmark (use original full 'returns' series after initial dropna)
        returns = returns.dropna()
        if len(returns) < 2:
            print("Warning: Not enough return data points for other risk metrics.")
            self.results['risk_metrics'] = {}
            return {}

        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        sharpe = np.nan
        if annualized_volatility > 1e-9:
             sharpe = (annualized_return - self.risk_free_rate) / annualized_volatility

        # Sortino Ratio
        downside_returns = returns[returns < returns.mean()] # Or use 0 as threshold: returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = np.nan
        if downside_std > 1e-9:
             sortino = (annualized_return - self.risk_free_rate) / downside_std

        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min() if not drawdown.empty else np.nan

        # Value at Risk (VaR) - 95th percentile of losses (5th percentile of returns)
        var_95 = np.percentile(returns, 5) if not returns.empty else np.nan

        # Conditional VaR (CVaR) - Expected shortfall beyond VaR
        cvar_95 = returns[returns <= var_95].mean() if not returns.empty and pd.notna(var_95) else np.nan

        # Calmar Ratio
        calmar = np.nan
        if max_drawdown is not None and abs(max_drawdown) > 1e-9:
            calmar = annualized_return / abs(max_drawdown)


        self.results['risk_metrics'] = self._to_serializable({
            'beta': beta,
            'alpha': alpha, # Annualized excess return (Jensen's Alpha)
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'var_95': var_95, # Daily VaR
            'cvar_95': cvar_95, # Daily CVaR
            'information_ratio': information_ratio,
            'calmar_ratio': calmar,
            'treynor_ratio': treynor,
            'annualized_volatility': annualized_volatility,
            'annualized_return': annualized_return
        })

        print("Risk metrics calculation complete.")
        return self.results['risk_metrics']


    def perform_time_series_analysis(self) -> Optional[Dict[str, Any]]:
        """Perform time series analysis (stationarity, ARIMA, GARCH) and forecasting."""
        if self.data is None or 'Returns' not in self.data.columns or self.data['Returns'].isnull().all():
             print("Error: No return data available for time series analysis.")
             return None

        log_returns = np.log(1 + self.data['Returns']).dropna()

        if len(log_returns) < 30: # Need sufficient data for ARIMA/GARCH
            print("Warning: Insufficient data for reliable time series analysis.")
            return None

        print("Performing time series analysis...")

        # Stationarity test (Augmented Dickey-Fuller)
        try:
            adf_result = adfuller(log_returns)
            is_stationary = adf_result[1] < 0.05
            adf_pvalue = adf_result[1]
        except Exception as e:
             print(f"ADF test failed: {e}")
             is_stationary = None
             adf_pvalue = None

        arima_results = {'error': 'ARIMA modeling skipped'}
        garch_results = {'error': 'GARCH modeling skipped'}

        # ARIMA model for returns forecasting
        try:
            # Simple order selection (e.g., AIC based loop or fixed order like (1,0,1) or (1,1,1))
            # Using a fixed, common order for simplicity and speed. Adjust as needed.
            order_to_try = (1, 0, 1) # Common for returns
            model = ARIMA(log_returns, order=order_to_try)
            model_fit = model.fit()

            # Forecast next 30 days
            forecast_log_returns = model_fit.forecast(steps=30)

            # Convert log returns forecast to price forecast
            last_price = self.data['Adj Close'].iloc[-1]
            price_forecast = [last_price]
            # Ensure forecast_log_returns is iterable
            if isinstance(forecast_log_returns, (pd.Series, np.ndarray)):
                for log_ret in forecast_log_returns:
                    price_forecast.append(price_forecast[-1] * np.exp(log_ret))
            else: # Handle scalar case if steps=1
                 price_forecast.append(price_forecast[-1] * np.exp(forecast_log_returns))


            arima_results = {
                'best_arima_order': order_to_try, # Store the order used
                'forecast_30d_prices': price_forecast[1:], # List of prices
                'expected_price_30d': price_forecast[-1],
                'expected_return_30d': (price_forecast[-1] / last_price - 1) * 100 if last_price else 0
            }

        except Exception as e:
            print(f"ARIMA modeling failed: {e}")
            arima_results = {'error': str(e)}


        # GARCH model for volatility forecasting
        try:
            # Scale returns for better GARCH fitting (common practice)
            scaled_log_returns = log_returns * 100
            garch_model = arch_model(scaled_log_returns, vol='Garch', p=1, o=0, q=1, dist='Normal') # Common GARCH(1,1)
            garch_result = garch_model.fit(disp='off', show_warning=False)

            # Forecast next 30 days variance
            garch_forecast = garch_result.forecast(horizon=30, reindex=False) # Use reindex=False
            # Get the forecast variance for the last day (h.30)
            # Access variance using column names like 'h.1', 'h.2', ... 'h.30'
            forecast_variance_scaled = garch_forecast.variance.iloc[0, -1] # Last column, first row

            # Unscale the volatility forecast (std dev)
            forecast_volatility_daily = np.sqrt(forecast_variance_scaled) / 100
            # Annualize (optional)
            # forecast_volatility_annualized = forecast_volatility_daily * np.sqrt(252)

            garch_results = {
                 'volatility_forecast_30d_daily': forecast_volatility_daily,
                 # 'volatility_forecast_annualized': forecast_volatility_annualized
            }

        except Exception as e:
            print(f"GARCH modeling failed: {e}")
            garch_results = {'error': str(e)}


        self.results['time_series'] = self._to_serializable({
            'is_stationary': is_stationary,
            'adf_pvalue': adf_pvalue,
            **arima_results, # Unpack ARIMA results
            **garch_results # Unpack GARCH results
        })

        print("Time series analysis complete.")
        return self.results['time_series']


    def calculate_fundamental_metrics(self) -> Dict[str, Any]:
        """Calculate and evaluate fundamental metrics from stock info."""
        if not self.stock_info:
            print("Warning: Stock info not available for fundamental analysis.")
            self.results['fundamental'] = {'error': 'Stock info unavailable'}
            return self.results['fundamental']

        print("Calculating fundamental metrics...")

        # Helper function to safely get numeric values
        def safe_get_numeric(key: str, default: Any = np.nan) -> Optional[float]:
            val = self.stock_info.get(key, default)
            if isinstance(val, (int, float)) and pd.notna(val) and not np.isinf(val):
                return float(val)
            return default if default is np.nan else float(default) # Return None if default is nan


        fund_metrics = {
            'pe_ratio': safe_get_numeric('trailingPE'),
            'forward_pe': safe_get_numeric('forwardPE'),
            'peg_ratio': safe_get_numeric('pegRatio'),
            'price_to_book': safe_get_numeric('priceToBook'),
            'price_to_sales': safe_get_numeric('priceToSalesTrailing12Months'),
            'enterprise_value': safe_get_numeric('enterpriseValue'),
            'enterprise_to_revenue': safe_get_numeric('enterpriseToRevenue'),
            'enterprise_to_ebitda': safe_get_numeric('enterpriseToEbitda'),
            'profit_margins': safe_get_numeric('profitMargins'),
            'operating_margins': safe_get_numeric('operatingMargins'),
            'dividend_yield': safe_get_numeric('dividendYield', 0) * 100, # Convert to percentage
            'payout_ratio': safe_get_numeric('payoutRatio'),
            'debt_to_equity': safe_get_numeric('debtToEquity'),
            'return_on_equity': safe_get_numeric('returnOnEquity'),
            'return_on_assets': safe_get_numeric('returnOnAssets'),
            'current_ratio': safe_get_numeric('currentRatio'),
            'quick_ratio': safe_get_numeric('quickRatio'),
            'target_median_price': safe_get_numeric('targetMedianPrice'),
            'analyst_recommendation_mean': safe_get_numeric('recommendationMean'), # Lower is better (1=Strong Buy, 5=Sell)
            'analyst_recommendation_key': self.stock_info.get('recommendationKey', 'N/A'), # e.g., 'buy', 'hold'
            'number_of_analyst_opinions': safe_get_numeric('numberOfAnalystOpinions', 0)
        }

        # Calculate potential upside/downside based on analyst targets
        current_price = self.data['Adj Close'].iloc[-1] if self.data is not None and not self.data.empty else None
        target_price = fund_metrics['target_median_price']

        if current_price is not None and current_price > 0 and target_price is not None and pd.notna(target_price):
            fund_metrics['price_target_upside_percent'] = (target_price / current_price - 1) * 100
        else:
            fund_metrics['price_target_upside_percent'] = np.nan # Use NaN for missing data

        self.results['fundamental'] = self._to_serializable(fund_metrics)
        print("Fundamental metrics calculation complete.")
        return self.results['fundamental']


    def perform_monte_carlo_simulation(self, simulations: int = 1000, days: int = 252) -> Optional[Dict[str, Any]]:
        """Perform Monte Carlo simulation using Geometric Brownian Motion."""
        if self.data is None or 'Returns' not in self.data.columns or self.data['Returns'].isnull().all():
             print("Error: No return data available for Monte Carlo simulation.")
             return None

        returns = self.data['Returns'].dropna()
        if len(returns) < 2:
             print("Warning: Insufficient return data for Monte Carlo simulation.")
             return None

        print("Performing Monte Carlo simulation...")

        # Parameters for geometric Brownian motion
        log_returns = np.log(1 + returns)
        mu = log_returns.mean() # Use log returns mean
        sigma = log_returns.std() # Use log returns std dev
        last_price = self.data['Adj Close'].iloc[-1]

        if not isinstance(last_price, (int, float)) or pd.isna(last_price):
             print("Error: Valid last price not found for Monte Carlo simulation.")
             return None

        simulation_results = np.zeros((days + 1, simulations))
        simulation_results[0] = last_price

        # Simulate future price paths using vectorized operations
        for t in range(1, days + 1):
            # Generate random shocks (Z) from standard normal distribution
            Z = np.random.standard_normal(simulations)
            # Apply GBM formula: S(t) = S(t-1) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            # Assuming dt = 1 (daily simulation)
            simulation_results[t] = simulation_results[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) + sigma * Z
            )

        # Store full results if needed (can be large)
        # self.simulation_results = pd.DataFrame(simulation_results)

        # Calculate statistics from final simulated prices
        final_prices = simulation_results[-1]

        self.results['monte_carlo'] = self._to_serializable({
            'simulations': simulations,
            'days': days,
            'mean_final_price': np.mean(final_prices),
            'median_final_price': np.median(final_prices),
            'min_final_price': np.min(final_prices),
            'max_final_price': np.max(final_prices),
            'std_final_price': np.std(final_prices),
            'current_price': last_price,
            'expected_return_percent': (np.mean(final_prices) / last_price - 1) * 100 if last_price else 0,
            'prob_profit': np.mean(final_prices > last_price) * 100 if last_price else 0,
            'percentiles': {
                f'{p}%': np.percentile(final_prices, p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
        })

        print("Monte Carlo simulation complete.")
        return self.results['monte_carlo']


    def calculate_optimal_portfolio(self) -> Optional[Dict[str, Any]]:
        """Calculate optimal portfolio weights (max Sharpe ratio) using MPT."""
        if not self.portfolio_tickers:
             print("Warning: No portfolio tickers defined for optimization.")
             return None

        print(f"Optimizing portfolio for tickers: {self.portfolio_tickers}")

        try:
            # Download historical data for all tickers
            portfolio_data = yf.download(self.portfolio_tickers, start=self.start_date, progress=False, auto_adjust=False)['Adj Close']

            if portfolio_data.empty or portfolio_data.isnull().all().all():
                print("Error: Could not download valid data for portfolio optimization.")
                return None

            # Handle cases where some tickers might fail download (results in all NaN columns)
            portfolio_data = portfolio_data.dropna(axis=1, how='all')
            valid_tickers = portfolio_data.columns.tolist()
            if not valid_tickers or len(valid_tickers) < 2:
                 print("Warning: Less than 2 assets with valid data, cannot perform optimization.")
                 # Return weights for single valid asset if only one exists
                 if len(valid_tickers) == 1:
                      single_ticker = valid_tickers[0]
                      self.results['portfolio'] = self._to_serializable({
                           'tickers': valid_tickers,
                           'optimal_weights': {single_ticker: 1.0},
                           'optimal_return': self.results.get('risk_metrics', {}).get('annualized_return', np.nan),
                           'optimal_volatility': self.results.get('risk_metrics', {}).get('annualized_volatility', np.nan),
                           'optimal_sharpe': self.results.get('risk_metrics', {}).get('sharpe_ratio', np.nan),
                           'current_ticker_weight': 1.0 if single_ticker == self.ticker else 0.0,
                           'portfolio_recommendation': "Include" if single_ticker == self.ticker else "Exclude"
                      })
                      return self.results['portfolio']
                 return None


            # Calculate returns
            returns = portfolio_data.pct_change().dropna()
            if returns.empty or len(returns) < 2:
                print("Error: Not enough return data for covariance calculation.")
                return None

            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            num_assets = len(valid_tickers)

            # Function to calculate portfolio performance
            def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
                ret = np.sum(mean_returns * weights)
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return ret, std

            # Function to minimize negative Sharpe Ratio (maximize Sharpe)
            def neg_sharpe_ratio(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
                p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
                if p_std < 1e-9: # Avoid division by zero
                    return 0 # Or a large positive number to avoid selection
                return -(p_returns - risk_free_rate) / p_std

            # Constraints: weights sum to 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            # Bounds: weights between 0 and 1 (no short selling)
            bounds = tuple((0, 1) for asset in range(num_assets))
            # Initial guess: equal weighting
            initial_guess = np.array([1/num_assets] * num_assets)

            # Optimize portfolio
            optimal_result = minimize(neg_sharpe_ratio, initial_guess,
                                      args=(mean_returns, cov_matrix, self.risk_free_rate),
                                      method='SLSQP', bounds=bounds, constraints=constraints)

            if not optimal_result.success:
                print(f"Warning: Portfolio optimization failed to converge. Message: {optimal_result.message}")
                 # Provide equal weights as fallback or handle error differently
                optimal_weights = initial_guess
            else:
                optimal_weights = optimal_result.x

            # Clean up tiny weights (set very small values to 0)
            optimal_weights[optimal_weights < 1e-4] = 0
            optimal_weights /= np.sum(optimal_weights) # Re-normalize

            # Calculate performance of optimal portfolio
            optimal_returns, optimal_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
            optimal_sharpe = (optimal_returns - self.risk_free_rate) / optimal_std if optimal_std > 1e-9 else np.nan

            # Prepare results dictionary
            optimal_weights_dict = dict(zip(valid_tickers, optimal_weights))

            self.results['portfolio'] = self._to_serializable({
                'tickers': valid_tickers,
                'optimal_weights': optimal_weights_dict,
                'optimal_return': optimal_returns,
                'optimal_volatility': optimal_std,
                'optimal_sharpe': optimal_sharpe,
                'current_ticker_weight': optimal_weights_dict.get(self.ticker, 0.0), # Use .get for safety
                # 'efficient_frontier': efficient_portfolios, # Removed for simplicity, add back if needed
                'portfolio_recommendation': "Include" if optimal_weights_dict.get(self.ticker, 0.0) > 0.01 else "Exclude" # Threshold 1%
            })

            print("Portfolio optimization complete.")
            return self.results['portfolio']

        except Exception as e:
            print(f"An error occurred during portfolio optimization: {e}")
            self.results['portfolio'] = {'error': str(e)}
            return self.results['portfolio']


    def sentiment_analysis(self) -> Dict[str, Any]:
        """Perform basic sentiment analysis using analyst recommendations."""
        print("Performing sentiment analysis (based on analyst ratings)...")
        try:
            ticker_obj = yf.Ticker(self.ticker)
            # Use recommendations_summary which is often more reliable
            recommendations = ticker_obj.recommendations_summary

            sentiment_result = {
                'sentiment': 'Neutral',
                'sentiment_score': 0.5, # Default neutral score (0=Sell, 1=Buy)
                'note': 'No analyst recommendations available or parsable.'
            }

            if recommendations is not None and not recommendations.empty:
                 # Try to extract counts (structure might vary slightly)
                 # Common columns: 'strongBuy', 'buy', 'hold', 'sell', 'strongSell'
                 rec_counts = recommendations.iloc[-1].to_dict() # Get latest summary row as dict

                 strong_buy = int(rec_counts.get('strongBuy', 0))
                 buy = int(rec_counts.get('buy', 0))
                 hold = int(rec_counts.get('hold', 0))
                 sell = int(rec_counts.get('sell', 0))
                 strong_sell = int(rec_counts.get('strongSell', 0))

                 total_recs = strong_buy + buy + hold + sell + strong_sell

                 if total_recs > 0:
                     # Weighted score: StrongBuy=1, Buy=0.75, Hold=0.5, Sell=0.25, StrongSell=0
                     weighted_sum = (strong_buy * 1.0 + buy * 0.75 + hold * 0.5 + sell * 0.25 + strong_sell * 0.0)
                     sentiment_score = weighted_sum / total_recs

                     # Determine sentiment category
                     if sentiment_score >= 0.8: sentiment = 'Very Positive'
                     elif sentiment_score >= 0.6: sentiment = 'Positive'
                     elif sentiment_score >= 0.4: sentiment = 'Neutral'
                     elif sentiment_score >= 0.2: sentiment = 'Negative'
                     else: sentiment = 'Very Negative'

                     sentiment_result = {
                         'total_recommendations': total_recs,
                         'strong_buy': strong_buy,
                         'buy': buy,
                         'hold': hold,
                         'sell': sell,
                         'strong_sell': strong_sell,
                         'sentiment_score': sentiment_score, # Scale 0-1
                         'sentiment': sentiment,
                         'note': f"{total_recs} analyst ratings summarized."
                     }

            self.results['sentiment'] = self._to_serializable(sentiment_result)

        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            self.results['sentiment'] = self._to_serializable({
                'sentiment': 'Neutral',
                'sentiment_score': 0.5,
                'error': str(e)
            })

        print("Sentiment analysis complete.")
        return self.results['sentiment']


    # In Analyzer.py

    # En Analyzer.py

    def backtest_portfolio(self, years_back: int = 3) -> Optional[Dict[str, Any]]:
        """
        Backtest the recommended optimal portfolio's performance.

        Args:
            years_back (int): Number of years to backtest.

        Returns:
            dict: Backtest results including performance metrics and cumulative returns.
                  Returns error dictionary if prerequisites fail or errors occur.
        """
        # --- Prerequisite Checks ---
        if 'portfolio' not in self.results or 'optimal_weights' not in self.results['portfolio']:
            msg = "Optimal portfolio weights not calculated. Run calculate_optimal_portfolio() first."
            print(f"Error: {msg}")
            return {'error': msg}

        portfolio_weights_dict = self.results['portfolio']['optimal_weights']
        if not portfolio_weights_dict:
             msg = "Optimal weights dictionary is empty. Skipping backtest."
             print(f"Warning: {msg}")
             return {'error': msg}

        print(f"Starting backtest for the last {years_back} years...")

        portfolio_tickers = list(portfolio_weights_dict.keys())
        portfolio_weights = np.array([portfolio_weights_dict[ticker] for ticker in portfolio_tickers])

        # Define backtest period
        end_date = pd.to_datetime(self.start_date)
        backtest_start_date = end_date - pd.DateOffset(years=years_back)
        print(self.start_date, end_date)

        # --- Initialize results to NaN/empty ---
        annualized_return = np.nan
        annualized_volatility = np.nan
        sharpe_ratio = np.nan
        max_drawdown = np.nan
        actual_years = np.nan
        final_cumulative_return = np.nan
        cumulative_returns = pd.Series(dtype=float)
        benchmark_annualized_return = np.nan
        benchmark_annualized_volatility = np.nan
        outperformance = np.nan
        benchmark_cumulative_returns_dict = {}

        try:
            # --- Download Historical Data ---
            historical_prices_dl = yf.download(portfolio_tickers, start=backtest_start_date, end=end_date, progress=False, auto_adjust=False)
            if 'Adj Close' not in historical_prices_dl.columns:
                 msg = "'Adj Close' column not found in downloaded portfolio data."
                 print(f"Error: {msg}")
                 raise ValueError(msg)
            historical_prices = historical_prices_dl['Adj Close']


            # --- Data Validation (Explicit Checks) ---
            if historical_prices.empty:
                msg = "Could not download valid data for backtesting period (DataFrame is empty)."
                print(f"Error: {msg}")
                raise ValueError(msg)
            if historical_prices.isnull().all().all():
                 msg = "Downloaded portfolio data contains only NaN values."
                 print(f"Error: {msg}")
                 raise ValueError(msg)

            # Drop columns that are entirely NaN
            historical_prices = historical_prices.dropna(axis=1, how='all')
            valid_tickers = historical_prices.columns.tolist()

            if not valid_tickers:
                 msg = "No valid historical price data available for selected tickers after dropping NaN columns."
                 print(f"Error: {msg}")
                 raise ValueError(msg)


            # --- Adjust Weights if Necessary ---
            if set(valid_tickers) != set(portfolio_tickers):
                print("Warning: Some tickers lacked data for the full backtest period. Adjusting weights.")
                filtered_weights_dict = {t: portfolio_weights_dict[t] for t in valid_tickers if t in portfolio_weights_dict}
                if not filtered_weights_dict:
                     msg = "No overlap between optimal weights and available backtest data."
                     print(f"Error: {msg}")
                     raise ValueError(msg)

                portfolio_weights = np.array(list(filtered_weights_dict.values()))
                weight_sum = np.sum(portfolio_weights)
                if weight_sum > 1e-6:
                     portfolio_weights /= weight_sum
                else:
                     print("Warning: Sum of filtered weights is near zero. Using equal weight fallback.")
                     portfolio_weights = np.ones(len(valid_tickers)) / len(valid_tickers)

                portfolio_tickers = list(filtered_weights_dict.keys())
                historical_prices = historical_prices[portfolio_tickers]


            # --- Calculate Returns ---
            returns = historical_prices.pct_change().dropna()
            if returns.empty:
                 msg = "No valid returns calculated (returns DataFrame is empty)."
                 print(f"Error: {msg}")
                 raise ValueError(msg)

            # --- Calculate Portfolio Returns ---
            if list(returns.columns) != portfolio_tickers:
                 print("Warning: Aligning returns columns with portfolio tickers for dot product.")
                 returns = returns.reindex(columns=portfolio_tickers).dropna(axis=1, how='all')
                 if returns.empty:
                      msg = "Returns empty after reindexing columns."
                      print(f"Error: {msg}")
                      raise ValueError(msg)

                 portfolio_tickers = returns.columns.tolist()
                 filtered_weights_dict = {t: portfolio_weights_dict[t] for t in portfolio_tickers if t in portfolio_weights_dict}
                 portfolio_weights = np.array(list(filtered_weights_dict.values()))
                 weight_sum = np.sum(portfolio_weights)
                 if weight_sum > 1e-6: portfolio_weights /= weight_sum

            portfolio_returns = returns.dot(portfolio_weights)
            if portfolio_returns.empty or portfolio_returns.isnull().all():
                 msg = "Portfolio returns calculation resulted in empty or all-NaN Series."
                 print(f"Error: {msg}")
                 raise ValueError(msg)

            # --- Calculate Performance Metrics ---
            actual_years = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days / 365.25
            if actual_years < 0.1:
                 print("Warning: Backtest period too short for reliable annualization.")
                 annualized_return = portfolio_returns.mean() * 252
            else:
                 cumulative_return_total = (1 + portfolio_returns).prod()
                 # Ensure cumulative_return_total is scalar
                 if isinstance(cumulative_return_total, (pd.Series, np.ndarray)): cumulative_return_total = cumulative_return_total.item()

                 if pd.notna(cumulative_return_total) and cumulative_return_total > 0:
                      annualized_return = cumulative_return_total ** (1 / actual_years) - 1
                 else:
                      print("Warning: Negative/zero cumulative return for portfolio, using arithmetic mean.")
                      annualized_return = portfolio_returns.mean() * 252

            annualized_volatility = portfolio_returns.std() * np.sqrt(252)
            # Ensure annualized_volatility is scalar
            if isinstance(annualized_volatility, (pd.Series, np.ndarray)): annualized_volatility = annualized_volatility.item()

            if pd.notna(annualized_volatility) and annualized_volatility > 1e-9:
                if pd.notna(annualized_return): # Check return is valid too
                    sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility

            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            if not cumulative_returns.empty:
                 running_max = cumulative_returns.cummax()
                 drawdown = (cumulative_returns / running_max) - 1
                 max_drawdown = drawdown.min()
                 # Ensure max_drawdown is scalar
                 if isinstance(max_drawdown, (pd.Series, np.ndarray)): max_drawdown = max_drawdown.item()
                 # Get final cumulative return scalar
                 final_cumulative_return = cumulative_returns.iloc[-1]
                 if isinstance(final_cumulative_return, (pd.Series, np.ndarray)): final_cumulative_return = final_cumulative_return.item()


            # --- Benchmark Comparison ---
            try:
                 print(f"Processing benchmark {self.benchmark}...")
                 benchmark_data_download = yf.download(self.benchmark, start=backtest_start_date, end=end_date, progress=False, auto_adjust=False)
                 if not benchmark_data_download.empty and 'Adj Close' in benchmark_data_download.columns:
                      benchmark_data = benchmark_data_download['Adj Close']
                 else:
                      benchmark_data = pd.Series(dtype=float)

                 if not benchmark_data.empty:
                      benchmark_returns = benchmark_data.pct_change().dropna()
                      if not benchmark_returns.empty:
                           bench_actual_years = (benchmark_returns.index[-1] - benchmark_returns.index[0]).days / 365.25
                           if bench_actual_years > 0.1:
                                bench_cum_ret_maybe_series = (1 + benchmark_returns).prod()
                                bench_cum_ret = np.nan # Initialize scalar
                                if isinstance(bench_cum_ret_maybe_series, (pd.Series, np.ndarray)):
                                    if bench_cum_ret_maybe_series.size == 1: bench_cum_ret = bench_cum_ret_maybe_series.item()
                                elif isinstance(bench_cum_ret_maybe_series, (int, float)):
                                     bench_cum_ret = bench_cum_ret_maybe_series

                                if pd.notna(bench_cum_ret) and bench_cum_ret > 0:
                                     benchmark_annualized_return = bench_cum_ret ** (1 / bench_actual_years) - 1
                                else:
                                     print("Warning: Using arithmetic mean for benchmark annual return.")
                                     benchmark_annualized_return = benchmark_returns.mean() * 252
                           else:
                                benchmark_annualized_return = benchmark_returns.mean() * 252

                           benchmark_annualized_volatility = benchmark_returns.std() * np.sqrt(252)
                           # --- >>> Ensure benchmark metrics are scalar <<< ---
                           if isinstance(benchmark_annualized_return, (pd.Series, np.ndarray)): benchmark_annualized_return = benchmark_annualized_return.item()
                           if isinstance(benchmark_annualized_volatility, (pd.Series, np.ndarray)): benchmark_annualized_volatility = benchmark_annualized_volatility.item()


                           # Calculate benchmark cumulative returns (for plotting)
                           benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
                           if not benchmark_cumulative_returns.empty:
                                # --- >>> CORRECTED RENAME <<< ---
                                benchmark_cumulative_returns.name = 'Benchmark' # Set name attribute
                                benchmark_cumulative_returns_dict = benchmark_cumulative_returns.to_dict()
                                # --- >>> END CORRECTION <<< ---

                           print(f"Benchmark {self.benchmark} processed.")
                      else: print(f"Warning: No benchmark returns calculated for {self.benchmark}.")
                 else: print(f"Warning: Could not download or process benchmark data ({self.benchmark}).")

            except Exception as e:
                 print(f"Warning: Error processing benchmark data for backtest: {e}")
                 import traceback
                 traceback.print_exc()
                 # Ensure metrics remain NaN on error
                 benchmark_annualized_return = np.nan
                 benchmark_annualized_volatility = np.nan
                 benchmark_cumulative_returns_dict = {}

            # Calculate Outperformance AFTER benchmark processing is done
            if pd.notna(annualized_return) and pd.notna(benchmark_annualized_return):
                 outperformance = annualized_return - benchmark_annualized_return
                 # Ensure outperformance is scalar
                 if isinstance(outperformance, (pd.Series, np.ndarray)): outperformance = outperformance.item()
            else:
                 outperformance = np.nan


        except Exception as e:
            # Catch errors from main portfolio processing
            print(f"An error occurred during portfolio backtesting: {e}")
            import traceback
            traceback.print_exc()
            # Return error dictionary immediately if main processing fails
            return {'error': str(e)}


        # --- Prepare Final Backtest Results Dictionary ---
        # All variables used here should now be scalars or serializable dicts
        backtest_results = {
            'years_backtested': round(actual_years, 2) if pd.notna(actual_years) else np.nan,
            'start_date': backtest_start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'portfolio_tickers': portfolio_tickers,
            'portfolio_weights': dict(zip(portfolio_tickers, portfolio_weights)),
            'annualized_return_percent': annualized_return * 100 if pd.notna(annualized_return) else np.nan,
            'annualized_volatility_percent': annualized_volatility * 100 if pd.notna(annualized_volatility) else np.nan,
            'sharpe_ratio': sharpe_ratio if pd.notna(sharpe_ratio) else np.nan,
            'max_drawdown_percent': max_drawdown * 100 if pd.notna(max_drawdown) else np.nan,
            'final_cumulative_return_percent': (final_cumulative_return - 1) * 100 if pd.notna(final_cumulative_return) else np.nan,
            'benchmark_comparison': {
                'benchmark': self.benchmark,
                'benchmark_annualized_return_percent': benchmark_annualized_return * 100 if pd.notna(benchmark_annualized_return) else np.nan,
                'benchmark_annualized_volatility_percent': benchmark_annualized_volatility * 100 if pd.notna(benchmark_annualized_volatility) else np.nan,
                'outperformance_percent': outperformance * 100 if pd.notna(outperformance) else np.nan
            },
            'cumulative_returns_portfolio': cumulative_returns.to_dict() if not cumulative_returns.empty else {},
            'cumulative_returns_benchmark': benchmark_cumulative_returns_dict,
        }

        # Store results, ensuring final serializability
        if 'portfolio_backtest' not in self.results:
            self.results['portfolio_backtest'] = {}
        self.results['portfolio_backtest'][f'{years_back}_years'] = self._to_serializable(backtest_results)

        print("Backtesting complete.")
        return self.results['portfolio_backtest'][f'{years_back}_years']
    
    def run_full_analysis(self) -> Optional[Dict[str, Any]]:
        """Run all analysis steps sequentially."""
        print(f"\n--- Starting Full Analysis for {self.ticker} ---")
        if self.fetch_data() is None:
            print("Analysis aborted due to data fetching issues.")
            return None

        self.perform_technical_analysis()
        self.calculate_risk_metrics()
        self.perform_time_series_analysis()
        self.calculate_fundamental_metrics()
        self.perform_monte_carlo_simulation()
        self.sentiment_analysis()

        print("--- Full Analysis Completed ---")
        # Generate the recommendation based on the collected results
        return self.generate_investment_recommendation()


    def generate_investment_recommendation(self) -> Dict[str, Any]:
        """Generate final investment recommendation based on aggregated analysis scores."""
        print("Generating final investment recommendation...")

        if not self.results:
             print("Error: No analysis results found to generate recommendation.")
             return {"error": "Analysis results missing."}

        # Helper to safely get nested results and handle potential missing keys/NaNs
        def get_metric(category: str, metric: str, default: Any = np.nan) -> Any:
             val = self.results.get(category, {}).get(metric, default)
             # Return default if value is None or NaN, unless default itself is NaN
             return default if pd.isna(val) else val


        # --- Scoring System (adjust weights and thresholds as needed) ---
        scores = {}
        max_scores = {'technical': 20, 'risk': 20, 'time_series': 15, 'fundamental': 20, 'monte_carlo': 15, 'portfolio': 10, 'sentiment': 10}

        # 1. Technical Score (Max 20)
        tech_score = 0
        ma_signal = get_metric('technical', 'ma_signal', 0)
        rsi = get_metric('technical', 'current_rsi', 50) # Default neutral RSI
        macd_signal = get_metric('technical', 'macd_signal', 0)
        if ma_signal > 0: tech_score += 7 # Bullish trend
        if 30 <= rsi <= 70: tech_score += 5 # Neutral momentum (often stable)
        elif rsi < 30: tech_score += 7      # Oversold (potential reversal)
        # else: tech_score += 0             # Overbought (potential pullback)
        if macd_signal > 0: tech_score += 6 # Bullish momentum crossover
        scores['technical'] = min(tech_score, max_scores['technical'])

        # 2. Risk Score (Max 20) - Higher is better (good risk-adjusted returns, low drawdown)
        risk_score = 0
        sharpe = get_metric('risk_metrics', 'sharpe_ratio', 0)
        sortino = get_metric('risk_metrics', 'sortino_ratio', 0)
        max_drawdown = get_metric('risk_metrics', 'max_drawdown', -1.0) # Default high drawdown
        alpha = get_metric('risk_metrics', 'alpha', 0)
        if sharpe > 1.0: risk_score += 5
        elif sharpe > 0.5: risk_score += 3
        if sortino > 1.5: risk_score += 5 # Higher threshold for Sortino
        elif sortino > 0.75: risk_score += 3
        if max_drawdown > -0.20: risk_score += 5 # Max loss < 20%
        elif max_drawdown > -0.35: risk_score += 3 # Max loss < 35%
        if alpha > 0.01: risk_score += 5 # Positive alpha > 1%
        scores['risk'] = min(risk_score, max_scores['risk'])

        # 3. Time Series Score (Max 15)
        ts_score = 0
        expected_return_30d = get_metric('time_series', 'expected_return_30d', 0)
        is_stationary = get_metric('time_series', 'is_stationary', False) # Default non-stationary
        if expected_return_30d > 3: ts_score += 7 # Forecast > 3% in 30d
        elif expected_return_30d > 1: ts_score += 4 # Forecast > 1% in 30d
        if is_stationary: ts_score += 5 # More predictable if stationary
        # Add GARCH vol forecast? Lower vol could be positive.
        vol_forecast = get_metric('time_series', 'volatility_forecast_30d_daily', 0.02) # Default 2% daily vol
        if vol_forecast < 0.015: ts_score += 3 # Lower expected vol is good
        scores['time_series'] = min(ts_score, max_scores['time_series'])

        # 4. Fundamental Score (Max 20)
        fund_score = 0
        pe = get_metric('fundamental', 'pe_ratio', 100) # Default high PE
        target_upside = get_metric('fundamental', 'price_target_upside_percent', -100) # Default downside
        profit_margin = get_metric('fundamental', 'profit_margins', -1) # Default negative margin
        roe = get_metric('fundamental', 'return_on_equity', -1)
        # Compare PE to a generic market average (e.g., 20) - refine with sector data if possible
        if 0 < pe < 20: fund_score += 5 # Undervalued based on PE
        elif 20 <= pe < 35: fund_score += 3 # Reasonably valued
        if target_upside > 20: fund_score += 7 # Analyst upside > 20%
        elif target_upside > 5: fund_score += 4 # Analyst upside > 5%
        if profit_margin > 0.15: fund_score += 5 # High profit margin > 15%
        elif profit_margin > 0.05: fund_score += 3 # Decent profit margin > 5%
        if roe > 0.15: fund_score += 3 # Good ROE > 15%
        scores['fundamental'] = min(fund_score, max_scores['fundamental'])

        # 5. Monte Carlo Score (Max 15)
        mc_score = 0
        mc_exp_return = get_metric('monte_carlo', 'expected_return_percent', 0)
        mc_prob_profit = get_metric('monte_carlo', 'prob_profit', 0)
        mc_std_dev_pct = (get_metric('monte_carlo', 'std_final_price', 0) /
                          get_metric('monte_carlo', 'current_price', 1)) * 100 if get_metric('monte_carlo', 'current_price') else 0

        if mc_exp_return > 15: mc_score += 7 # High expected return > 15%
        elif mc_exp_return > 5: mc_score += 4 # Modest expected return > 5%
        if mc_prob_profit > 60: mc_score += 5 # Probability of profit > 60%
        elif mc_prob_profit > 50: mc_score += 3
        if mc_std_dev_pct < 30 and mc_std_dev_pct > 0 : mc_score += 3 # Lower relative std dev < 30% is good
        scores['monte_carlo'] = min(mc_score, max_scores['monte_carlo'])

        # 6. Portfolio Score (Max 10)
        # Se elimina la metrica porque pasa a otro lado
        """
        port_score = 0
        port_weight = get_metric('portfolio', 'current_ticker_weight', 0)
        if port_weight > 0.15: port_score += 10 # High allocation > 15%
        elif port_weight > 0.05: port_score += 6 # Moderate allocation > 5%
        elif port_weight > 0.01: port_score += 3 # Small allocation > 1%
        scores['portfolio'] = min(port_score, max_scores['portfolio'])"
        """
        # 7. Sentiment Score (Max 10)
        sent_score = 0
        sentiment_val = get_metric('sentiment', 'sentiment_score', 0.5) # 0-1 scale
        sentiment_cat = get_metric('sentiment', 'sentiment', 'Neutral')
        # Scale sentiment_val (0-1) to score (0-10)
        sent_score = sentiment_val * 10
        scores['sentiment'] = min(round(sent_score), max_scores['sentiment']) # Round to nearest int


        # --- Total Score and Recommendation ---
        total_score = sum(scores.values()) /0.9 # Divide sobre 0.9 porque se elimino Portafolio Optimization
        max_possible_score = sum(max_scores.values())
        # Normalize score to 0-100 for easier interpretation if max_possible_score != 100
        normalized_score = round((total_score / max_possible_score) * 100) if max_possible_score else 0


        # Define recommendation levels based on normalized score
        if normalized_score >= 80:
            recommendation = "Strong Buy"; confidence = "High"; color = "#15803D" # Dark Green
        elif normalized_score >= 65:
            recommendation = "Buy"; confidence = "Moderate"; color = "#65A30D" # Light Green
        elif normalized_score >= 50:
            recommendation = "Hold"; confidence = "Moderate"; color = "#CA8A04" # Yellow/Gold
        elif normalized_score >= 35:
            recommendation = "Reduce"; confidence = "Moderate"; color = "#EA580C" # Orange
        else:
            recommendation = "Sell"; confidence = "High"; color = "#DC2626" # Red

        recommendation_text = f"Overall score of {normalized_score}/100 suggests a '{recommendation}' stance for {self.company_name} ({self.ticker}). "
        # Add more dynamic text based on top/bottom scores if desired

        # Compile final recommendation dictionary
        final_recommendation = {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'recommendation': recommendation,
            'recommendation_text': recommendation_text, # Keep it concise here
            'recommendation_color': color,
            'confidence': confidence,
            'total_score_normalized': normalized_score, # Use normalized score
            'category_scores': self._to_serializable(scores), # Raw scores per category
            'max_category_scores': max_scores, # Max possible score per category
            'current_price': get_metric('technical', 'current_price', None),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'key_metrics': { # Include a selection of key metrics for quick view
                'technical': {
                    'rsi': round(rsi, 2) if pd.notna(rsi) else None,
                    'trend': 'Bullish' if ma_signal > 0 else ('Bearish' if ma_signal < 0 else 'Neutral'),
                    'macd_signal': 'Bullish' if macd_signal > 0 else ('Bearish' if macd_signal < 0 else 'Neutral')
                },
                'risk_metrics': {
                    'sharpe_ratio': round(sharpe, 2) if pd.notna(sharpe) else None,
                    'max_drawdown_percent': round(max_drawdown * 100, 2) if pd.notna(max_drawdown) else None,
                    'beta': round(get_metric('risk_metrics', 'beta'), 2) if pd.notna(get_metric('risk_metrics', 'beta')) else None,
                    'alpha_annual_percent': round(alpha * 100, 2) if pd.notna(alpha) else None
                },
                'time_series': {
                    'stationary': is_stationary,
                    'expected_return_30d_percent': round(expected_return_30d, 2) if pd.notna(expected_return_30d) else None
                },
                'fundamental': {
                    'pe_ratio': round(pe, 2) if pd.notna(pe) else None,
                    'price_target_upside_percent': round(target_upside, 2) if pd.notna(target_upside) else None,
                    'profit_margins_percent': round(profit_margin * 100, 2) if pd.notna(profit_margin) else None,
                    'recommendation': get_metric('fundamental', 'analyst_recommendation_key', 'N/A')
                },
                'monte_carlo': {
                    'expected_return_percent': round(mc_exp_return, 2) if pd.notna(mc_exp_return) else None,
                    'prob_profit_percent': round(mc_prob_profit, 2) if pd.notna(mc_prob_profit) else None
                },
                'sentiment': {
                    'sentiment': sentiment_cat,
                    'sentiment_score_0_1': round(sentiment_val, 2) if pd.notna(sentiment_val) else None
                },
            },
            # Optionally include the full self.results if needed by frontend, but can be large
            # 'full_analysis_results': self.results
        }

        print("Recommendation generated.")
        # print(json.dumps(final_recommendation, indent=4)) # DEBUG
        return self._to_serializable(final_recommendation) # Ensure final output is serializable

# --- Example Usage ---
if __name__ == "__main__":
    # Example with a specific portfolio list
    tickers_max_rendimiento = [
        # -- Núcleo Tecnológico (Gigantes Innovadores) --
        'MSFT',  # Microsoft: Cloud (Azure), IA, Software, Gaming. Diversificado dentro de la tecnología.
        'NVDA',  # NVIDIA: Líder indiscutible en GPUs para IA, Data Centers, Gaming. Crecimiento explosivo.
        'GOOGL', # Alphabet (Google): Dominancia en Búsqueda/Publicidad, Cloud (GCP), IA (DeepMind), Waymo.
        'AMZN',  # Amazon: Líder en E-commerce y Cloud (AWS). Expansión a Publicidad, Salud.

        # -- Disrupción y Alto Crecimiento Específico --
        'TSLA',  # Tesla: Líder en Vehículos Eléctricos, Energía, Potencial en Autonomía y Robótica (Alto Riesgo/Recompensa).
        'LLY',   # Eli Lilly: Farmacéutica líder con medicamentos de alto crecimiento (Diabetes/Obesidad - GLP-1), Oncología.
        'ASML',  # ASML Holding N.V.: Monopolio virtual en máquinas de litografía EUV, cruciales para semiconductores avanzados.
        'AMD',   # Advanced Micro Devices: Fuerte competidor de Intel y NVIDIA en CPUs y GPUs. Crecimiento en Data Centers.

        # -- Plataformas y Software de Crecimiento --
        'CRM',   # Salesforce: Líder en CRM basado en la nube, ecosistema en expansión.
        'SHOP',  # Shopify: Plataforma líder para e-commerce, empoderando a pequeños y grandes negocios.

        # -- Potencial en Salud Innovadora --
        'ISRG',  # Intuitive Surgical: Líder en cirugía robótica (Da Vinci). Innovación constante.

        # -- Financiero con Componente Tecnológico --
        'V',     # Visa: Red de pagos global dominante, beneficiándose del cambio a pagos digitales. Fuerte moat.
    ]
    analyzer = AdvancedStockAnalyzer(
        ticker=tickers_max_rendimiento[0],
        portfolio_tickers=tickers_max_rendimiento,
        risk_free_rate=0.04
    )

    # Run the full analysis
    final_result = analyzer.run_full_analysis()

    if final_result and 'error' not in final_result:
        print("\n--- Final Recommendation ---")
        print(json.dumps(final_result, indent=4, ensure_ascii=False))

        # Run a 5-year backtest if analysis was successful
        print("\n--- Running Backtest ---")
        backtest_result_5y = analyzer.backtest_portfolio(years_back=5)

        if backtest_result_5y and 'error' not in backtest_result_5y:
            print("\n--- 5-Year Backtest Result ---")
            # Exclude the long cumulative series for cleaner printout
            print_backtest = {k: v for k, v in backtest_result_5y.items() if not k.startswith('cumulative_')}
            print(json.dumps(print_backtest, indent=4, ensure_ascii=False))
        elif backtest_result_5y:
             print(f"\n--- Backtest Error --- \n{backtest_result_5y['error']}")

    else:
        print("\n--- Analysis Failed ---")
        if final_result:
             print(json.dumps(final_result, indent=4))