import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .utils import to_serializable

def perform_monte_carlo_simulation(data: pd.DataFrame, simulations: int = 1000, days: int = 252) -> Optional[Dict[str, Any]]:
    """Perform Monte Carlo simulation using Geometric Brownian Motion."""
    if data is None or 'Returns' not in data.columns or data['Returns'].isnull().all():
        print("Error: No return data available for Monte Carlo simulation.")
        return None

    returns = data['Returns'].dropna()
    if len(returns) < 2:
        print("Warning: Insufficient return data for Monte Carlo simulation.")
        return None

    print("Performing Monte Carlo simulation...")

    log_returns = np.log(1 + returns)
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = data['Adj Close'].iloc[-1]

    if not isinstance(last_price, (int, float)) or pd.isna(last_price):
        print("Error: Valid last price not found for Monte Carlo simulation.")
        return None

    simulation_results = np.zeros((days + 1, simulations))
    simulation_results[0] = last_price

    for t in range(1, days + 1):
        Z = np.random.standard_normal(simulations)
        simulation_results[t] = simulation_results[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) + sigma * Z
        )

    final_prices = simulation_results[-1]

    results = to_serializable({
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
    return results
