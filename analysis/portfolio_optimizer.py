import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Optional, List, Tuple
from .utils import to_serializable
from curl_cffi import requests
session = requests.Session(impersonate="edge")

def calculate_optimal_portfolio(portfolio_tickers: List[str], start_date: str, risk_free_rate: float, ticker: str, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calculate optimal portfolio weights (max Sharpe ratio) using MPT."""
    if not portfolio_tickers:
        print("Warning: No portfolio tickers defined for optimization.")
        return None

    print(f"Optimizing portfolio for tickers: {portfolio_tickers}")

    try:
        portfolio_data = yf.download(portfolio_tickers, start=start_date, progress=False, auto_adjust=False, session=session)['Adj Close']

        if portfolio_data.empty or portfolio_data.isnull().all().all():
            print("Error: Could not download valid data for portfolio optimization.")
            return None

        portfolio_data = portfolio_data.dropna(axis=1, how='all')
        valid_tickers = portfolio_data.columns.tolist()
        if not valid_tickers or len(valid_tickers) < 2:
            print("Warning: Less than 2 assets with valid data, cannot perform optimization.")
            if len(valid_tickers) == 1:
                single_ticker = valid_tickers[0]
                return to_serializable({
                    'tickers': valid_tickers,
                    'optimal_weights': {single_ticker: 1.0},
                    'optimal_return': results.get('risk_metrics', {}).get('annualized_return', np.nan),
                    'optimal_volatility': results.get('risk_metrics', {}).get('annualized_volatility', np.nan),
                    'optimal_sharpe': results.get('risk_metrics', {}).get('sharpe_ratio', np.nan),
                    'current_ticker_weight': 1.0 if single_ticker == ticker else 0.0,
                    'portfolio_recommendation': "Include" if single_ticker == ticker else "Exclude"
                })
            return None

        returns = portfolio_data.pct_change().dropna()
        if returns.empty or len(returns) < 2:
            print("Error: Not enough return data for covariance calculation.")
            return None

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(valid_tickers)

        def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
            ret = np.sum(mean_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return ret, std

        def neg_sharpe_ratio(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
            p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
            if p_std < 1e-9:
                return 0
            return -(p_returns - risk_free_rate) / p_std

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)

        optimal_result = minimize(neg_sharpe_ratio, initial_guess,
                                  args=(mean_returns, cov_matrix, risk_free_rate),
                                  method='SLSQP', bounds=bounds, constraints=constraints)

        if not optimal_result.success:
            print(f"Warning: Portfolio optimization failed to converge. Message: {optimal_result.message}")
            optimal_weights = initial_guess
        else:
            optimal_weights = optimal_result.x

        optimal_weights[optimal_weights < 1e-4] = 0
        optimal_weights /= np.sum(optimal_weights)

        optimal_returns, optimal_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
        optimal_sharpe = (optimal_returns - risk_free_rate) / optimal_std if optimal_std > 1e-9 else np.nan

        optimal_weights_dict = dict(zip(valid_tickers, optimal_weights))

        portfolio_results = to_serializable({
            'tickers': valid_tickers,
            'optimal_weights': optimal_weights_dict,
            'optimal_return': optimal_returns,
            'optimal_volatility': optimal_std,
            'optimal_sharpe': optimal_sharpe,
            'current_ticker_weight': optimal_weights_dict.get(ticker, 0.0),
            'portfolio_recommendation': "Include" if optimal_weights_dict.get(ticker, 0.0) > 0.01 else "Exclude"
        })

        print("Portfolio optimization complete.")
        return portfolio_results

    except Exception as e:
        print(f"An error occurred during portfolio optimization: {e}")
        return {'error': str(e)}

def backtest_portfolio(results: Dict[str, Any], start_date: str, benchmark: str, risk_free_rate: float, years_back: int = 3) -> Optional[Dict[str, Any]]:
    """Backtest the recommended optimal portfolio's performance."""
    if 'portfolio' not in results or 'optimal_weights' not in results['portfolio']:
        msg = "Optimal portfolio weights not calculated. Run calculate_optimal_portfolio() first."
        print(f"Error: {msg}")
        return {'error': msg}

    portfolio_weights_dict = results['portfolio']['optimal_weights']
    if not portfolio_weights_dict:
        msg = "Optimal weights dictionary is empty. Skipping backtest."
        print(f"Warning: {msg}")
        return {'error': msg}

    print(f"Starting backtest for the last {years_back} years...")

    portfolio_tickers = list(portfolio_weights_dict.keys())
    portfolio_weights = np.array([portfolio_weights_dict[ticker] for ticker in portfolio_tickers])

    end_date = pd.to_datetime(start_date)
    backtest_start_date = end_date - pd.DateOffset(years=years_back)
    print(start_date, end_date)

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
        historical_prices_dl = yf.download(portfolio_tickers, start=backtest_start_date, end=end_date, progress=False, auto_adjust=False, session=session)
        if 'Adj Close' not in historical_prices_dl.columns:
            msg = "'Adj Close' column not found in downloaded portfolio data."
            print(f"Error: {msg}")
            raise ValueError(msg)
        historical_prices = historical_prices_dl['Adj Close']

        if historical_prices.empty:
            msg = "Could not download valid data for backtesting period (DataFrame is empty)."
            print(f"Error: {msg}")
            raise ValueError(msg)
        if historical_prices.isnull().all().all():
            msg = "Downloaded portfolio data contains only NaN values."
            print(f"Error: {msg}")
            raise ValueError(msg)

        historical_prices = historical_prices.dropna(axis=1, how='all')
        valid_tickers = historical_prices.columns.tolist()

        if not valid_tickers:
            msg = "No valid historical price data available for selected tickers after dropping NaN columns."
            print(f"Error: {msg}")
            raise ValueError(msg)

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

        returns = historical_prices.pct_change().dropna()
        if returns.empty:
            msg = "No valid returns calculated (returns DataFrame is empty)."
            print(f"Error: {msg}")
            raise ValueError(msg)

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

        actual_years = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days / 365.25
        if actual_years < 0.1:
            print("Warning: Backtest period too short for reliable annualization.")
            annualized_return = portfolio_returns.mean() * 252
        else:
            cumulative_return_total = (1 + portfolio_returns).prod()
            if isinstance(cumulative_return_total, (pd.Series, np.ndarray)): cumulative_return_total = cumulative_return_total.item()

            if pd.notna(cumulative_return_total) and cumulative_return_total > 0:
                annualized_return = cumulative_return_total ** (1 / actual_years) - 1
            else:
                print("Warning: Negative/zero cumulative return for portfolio, using arithmetic mean.")
                annualized_return = portfolio_returns.mean() * 252

        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        if isinstance(annualized_volatility, (pd.Series, np.ndarray)): annualized_volatility = annualized_volatility.item()

        if pd.notna(annualized_volatility) and annualized_volatility > 1e-9:
            if pd.notna(annualized_return):
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        cumulative_returns = (1 + portfolio_returns).cumprod()
        if not cumulative_returns.empty:
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            max_drawdown = drawdown.min()
            if isinstance(max_drawdown, (pd.Series, np.ndarray)): max_drawdown = max_drawdown.item()
            final_cumulative_return = cumulative_returns.iloc[-1]
            if isinstance(final_cumulative_return, (pd.Series, np.ndarray)): final_cumulative_return = final_cumulative_return.item()

        try:
            print(f"Processing benchmark {benchmark}...")
            benchmark_data_download = yf.download(benchmark, start=backtest_start_date, end=end_date, progress=False, auto_adjust=False, session=session)
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
                        bench_cum_ret = np.nan
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
                    if isinstance(benchmark_annualized_return, (pd.Series, np.ndarray)): benchmark_annualized_return = benchmark_annualized_return.item()
                    if isinstance(benchmark_annualized_volatility, (pd.Series, np.ndarray)): benchmark_annualized_volatility = benchmark_annualized_volatility.item()

                    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
                    if not benchmark_cumulative_returns.empty:
                        benchmark_cumulative_returns.name = 'Benchmark'
                        benchmark_cumulative_returns_dict = benchmark_cumulative_returns.to_dict()

                    print(f"Benchmark {benchmark} processed.")
                else: print(f"Warning: No benchmark returns calculated for {benchmark}.")
            else: print(f"Warning: Could not download or process benchmark data ({benchmark}).")

        except Exception as e:
            print(f"Warning: Error processing benchmark data for backtest: {e}")
            import traceback
            traceback.print_exc()
            benchmark_annualized_return = np.nan
            benchmark_annualized_volatility = np.nan
            benchmark_cumulative_returns_dict = {}

        if pd.notna(annualized_return) and pd.notna(benchmark_annualized_return):
            outperformance = annualized_return - benchmark_annualized_return
            if isinstance(outperformance, (pd.Series, np.ndarray)): outperformance = outperformance.item()
        else:
            outperformance = np.nan

    except Exception as e:
        print(f"An error occurred during portfolio backtesting: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

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
            'benchmark': benchmark,
            'benchmark_annualized_return_percent': benchmark_annualized_return * 100 if pd.notna(benchmark_annualized_return) else np.nan,
            'benchmark_annualized_volatility_percent': benchmark_annualized_volatility * 100 if pd.notna(benchmark_annualized_volatility) else np.nan,
            'outperformance_percent': outperformance * 100 if pd.notna(outperformance) else np.nan
        },
        'cumulative_returns_portfolio': cumulative_returns.to_dict() if not cumulative_returns.empty else {},
        'cumulative_returns_benchmark': benchmark_cumulative_returns_dict,
    }

    print("Backtesting complete.")
    return to_serializable(backtest_results)
