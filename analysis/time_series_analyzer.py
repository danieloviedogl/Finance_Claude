import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from typing import Dict, Any, Optional
from .utils import to_serializable

def perform_time_series_analysis(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Perform time series analysis (stationarity, ARIMA, GARCH) and forecasting."""
    if data is None or 'Returns' not in data.columns or data['Returns'].isnull().all():
        print("Error: No return data available for time series analysis.")
        return None

    log_returns = np.log(1 + data['Returns']).dropna()

    if len(log_returns) < 30:
        print("Warning: Insufficient data for reliable time series analysis.")
        return None

    print("Performing time series analysis...")

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

    try:
        order_to_try = (1, 0, 1)
        model = ARIMA(log_returns, order=order_to_try)
        model_fit = model.fit()
        forecast_log_returns = model_fit.forecast(steps=30)
        last_price = data['Adj Close'].iloc[-1]
        price_forecast = [last_price]
        if isinstance(forecast_log_returns, (pd.Series, np.ndarray)):
            for log_ret in forecast_log_returns:
                price_forecast.append(price_forecast[-1] * np.exp(log_ret))
        else:
            price_forecast.append(price_forecast[-1] * np.exp(forecast_log_returns))

        arima_results = {
            'best_arima_order': order_to_try,
            'forecast_30d_prices': price_forecast[1:],
            'expected_price_30d': price_forecast[-1],
            'expected_return_30d': (price_forecast[-1] / last_price - 1) * 100 if last_price else 0
        }
    except Exception as e:
        print(f"ARIMA modeling failed: {e}")
        arima_results = {'error': str(e)}

    try:
        scaled_log_returns = log_returns * 100
        garch_model = arch_model(scaled_log_returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
        garch_result = garch_model.fit(disp='off', show_warning=False)
        garch_forecast = garch_result.forecast(horizon=30, reindex=False)
        forecast_variance_scaled = garch_forecast.variance.iloc[0, -1]
        forecast_volatility_daily = np.sqrt(forecast_variance_scaled) / 100
        garch_results = {
            'volatility_forecast_30d_daily': forecast_volatility_daily,
        }
    except Exception as e:
        print(f"GARCH modeling failed: {e}")
        garch_results = {'error': str(e)}

    results = to_serializable({
        'is_stationary': is_stationary,
        'adf_pvalue': adf_pvalue,
        **arima_results,
        **garch_results
    })

    print("Time series analysis complete.")
    return results
