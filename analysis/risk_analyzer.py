import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, Optional
from .utils import to_serializable

def calculate_risk_metrics(merged_data: pd.DataFrame, ticker: str, benchmark: str, risk_free_rate: float) -> Optional[Dict[str, Any]]:
    """Calculate risk metrics and ratios relative to the benchmark."""
    if merged_data is None or merged_data.empty or len(merged_data) < 2:
        print("Error: Insufficient merged data for risk metric calculation.")
        return None
    if f'{benchmark}_returns' not in merged_data.columns:
        print("Warning: Benchmark returns not available. Some risk metrics (Beta, Alpha, Info Ratio) cannot be calculated.")

    print("Calculating risk metrics...")
    returns = merged_data[f'{ticker}_returns']
    benchmark_returns = merged_data.get(f'{benchmark}_returns', None)

    beta = alpha = information_ratio = treynor = np.nan
    if benchmark_returns is not None and not benchmark_returns.isnull().all():
        valid_idx = returns.notna() & benchmark_returns.notna()
        if valid_idx.sum() >= 2:
            returns_valid = returns[valid_idx]
            benchmark_returns_valid = benchmark_returns[valid_idx]

            X = sm.add_constant(benchmark_returns_valid)
            y = returns_valid
            try:
                model = sm.OLS(y, X).fit()
                alpha_daily, beta = model.params
                alpha = alpha_daily * 252

                active_return = returns_valid - benchmark_returns_valid
                if active_return.std() > 1e-9:
                    information_ratio = (active_return.mean() * 252) / (active_return.std() * np.sqrt(252))

                annualized_return = returns.mean() * 252
                if beta != 0:
                    treynor = (annualized_return - risk_free_rate) / beta

            except Exception as e:
                print(f"Warning: OLS regression for Beta/Alpha failed: {e}. Using covariance method for Beta.")
                covariance = np.cov(returns_valid, benchmark_returns_valid)[0, 1]
                benchmark_variance = np.var(benchmark_returns_valid)
                if benchmark_variance > 1e-9:
                    beta = covariance / benchmark_variance
                    expected_return = risk_free_rate + beta * (benchmark_returns_valid.mean() * 252 - risk_free_rate)
                    alpha = returns_valid.mean() * 252 - expected_return
                else:
                    beta = np.nan
                    alpha = np.nan
        else:
            print("Warning: Not enough overlapping data points to calculate Beta/Alpha.")

    returns = returns.dropna()
    if len(returns) < 2:
        print("Warning: Not enough return data points for other risk metrics.")
        return {}

    annualized_return = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)

    sharpe = np.nan
    if annualized_volatility > 1e-9:
        sharpe = (annualized_return - risk_free_rate) / annualized_volatility

    downside_returns = returns[returns < returns.mean()]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = np.nan
    if downside_std > 1e-9:
        sortino = (annualized_return - risk_free_rate) / downside_std

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan

    var_95 = np.percentile(returns, 5) if not returns.empty else np.nan
    cvar_95 = returns[returns <= var_95].mean() if not returns.empty and pd.notna(var_95) else np.nan

    calmar = np.nan
    if max_drawdown is not None and abs(max_drawdown) > 1e-9:
        calmar = annualized_return / abs(max_drawdown)

    results = to_serializable({
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
        'annualized_volatility': annualized_volatility,
        'annualized_return': annualized_return
    })

    print("Risk metrics calculation complete.")
    return results
