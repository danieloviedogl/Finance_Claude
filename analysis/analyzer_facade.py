from typing import List, Dict, Any, Optional
import pandas as pd
from . import data_provider
from . import technical_analyzer
from . import risk_analyzer
from . import time_series_analyzer
from . import fundamental_analyzer
from . import monte_carlo
from . import portfolio_optimizer
from . import sentiment_analyzer
from . import recommendation_engine

DEFAULT_PORTFOLIO_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B']

class StockAnalyzerFacade:
    def __init__(self, ticker: str, start_date: str = '2025-01-01',
                 benchmark: str = 'SPY', risk_free_rate: float = 0.045,
                 portfolio_tickers: Optional[List[str]] = None):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.benchmark = benchmark.upper()
        self.risk_free_rate = risk_free_rate
        _initial_portfolio = portfolio_tickers if portfolio_tickers else DEFAULT_PORTFOLIO_TICKERS
        self.portfolio_tickers = sorted(list(set([self.ticker] + [pt.upper() for pt in _initial_portfolio])))
        self.results: Dict[str, Any] = {}
        self.data_cache: Dict[str, Any] = {}

    def run_full_analysis(self) -> Optional[Dict[str, Any]]:
        """Run all analysis steps sequentially."""
        print(f"\n--- Starting Full Analysis for {self.ticker} ---")
        
        # 1. Fetch Data
        self.data_cache = data_provider.fetch_data(self.ticker, self.start_date, self.benchmark)
        if not self.data_cache:
            print("Analysis aborted due to data fetching issues.")
            return None

        # 2. Technical Analysis
        tech_results = technical_analyzer.perform_technical_analysis(self.data_cache['data'])
        if tech_results:
            self.results['technical'] = tech_results['analysis']
            self.data_cache['technical_data'] = tech_results['technical_data']

        # 3. Risk Metrics
        self.results['risk_metrics'] = risk_analyzer.calculate_risk_metrics(
            self.data_cache['merged_data'], self.ticker, self.benchmark, self.risk_free_rate
        )

        # 4. Time Series Analysis
        self.results['time_series'] = time_series_analyzer.perform_time_series_analysis(self.data_cache['data'])

        # 5. Fundamental Analysis
        self.results['fundamental'] = fundamental_analyzer.calculate_fundamental_metrics(
            self.data_cache['stock_info'], self.data_cache['data']
        )

        # 6. Monte Carlo Simulation
        self.results['monte_carlo'] = monte_carlo.perform_monte_carlo_simulation(self.data_cache['data'])

        # 7. Sentiment Analysis
        self.results['sentiment'] = sentiment_analyzer.sentiment_analysis(self.ticker)

        # 8. Generate Recommendation
        final_recommendation = recommendation_engine.generate_investment_recommendation(
            self.results, self.ticker, self.data_cache['company_name']
        )
        
        print("--- Full Analysis Completed ---")
        return final_recommendation

    def run_portfolio_analysis(self) -> Optional[Dict[str, Any]]:
        """Run portfolio optimization and backtesting."""
        print(f"\n--- Starting Portfolio Analysis for {self.ticker} ---")
        
        # Run data fetching if not already done
        if not self.data_cache:
            self.data_cache = data_provider.fetch_data(self.ticker, self.start_date, self.benchmark)
            if not self.data_cache:
                print("Portfolio analysis aborted due to data fetching issues.")
                return None
        
        # Run risk metrics if not already done
        if 'risk_metrics' not in self.results:
            self.results['risk_metrics'] = risk_analyzer.calculate_risk_metrics(
                self.data_cache['merged_data'], self.ticker, self.benchmark, self.risk_free_rate
            )

        # 1. Portfolio Optimization
        self.results['portfolio'] = portfolio_optimizer.calculate_optimal_portfolio(
            self.portfolio_tickers, self.start_date, self.risk_free_rate, self.ticker, self.results
        )

        # 2. Backtesting
        if self.results.get('portfolio') and 'error' not in self.results['portfolio']:
            backtest_result = portfolio_optimizer.backtest_portfolio(
                self.results, self.start_date, self.benchmark, self.risk_free_rate, years_back=5
            )
            if backtest_result:
                if 'portfolio_backtest' not in self.results:
                    self.results['portfolio_backtest'] = {}
                self.results['portfolio_backtest']['5_years'] = backtest_result
        
        print("--- Portfolio Analysis Completed ---")
        return self.results
