import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from curl_cffi import requests
session = requests.Session(impersonate="edge")

def fetch_data(ticker: str, start_date: str, benchmark: str) -> Optional[Dict[str, Any]]:
    """Fetch historical stock data, benchmark data, and company info."""
    print(f"Fetching data for {ticker} and benchmark {benchmark}...")
    try:
        data = yf.download(ticker, start=start_date, progress=False, group_by='column', auto_adjust=False, session=session)
        benchmark_data = yf.download(benchmark, start=start_date, progress=False, group_by='column', auto_adjust=False, session=session)

        if data is None or data.empty:
            print(f"Error: No data downloaded for ticker {ticker}")
            raise ValueError(f"No data for ticker {ticker}")
        if benchmark_data is None or benchmark_data.empty:
            print(f"Warning: No data downloaded for benchmark {benchmark}. Risk metrics requiring benchmark will be limited.")
            benchmark_data = pd.DataFrame(index=data.index, columns=['Adj Close', 'Returns'])

        if isinstance(data.columns, pd.MultiIndex):
            print("Detected MultiIndex columns for main ticker, flattening...")
            ticker_level_name = data.columns.names[1] if len(data.columns.names) > 1 else None
            if ticker_level_name:
                data.columns = data.columns.droplevel(ticker_level_name)
            else:
                data.columns = data.columns.droplevel(1)
            print(f"Flattened columns: {data.columns.tolist()}")

        if isinstance(benchmark_data.columns, pd.MultiIndex):
            print("Detected MultiIndex columns for benchmark, flattening...")
            ticker_level_name = benchmark_data.columns.names[1] if len(benchmark_data.columns.names) > 1 else None
            if ticker_level_name:
                benchmark_data.columns = benchmark_data.columns.droplevel(ticker_level_name)
            else:
                benchmark_data.columns = benchmark_data.columns.droplevel(1)

        ticker_obj = yf.Ticker(ticker, session=session)
        stock_info = ticker_obj.info
        company_name = stock_info.get('longName', ticker)

        if 'Adj Close' not in data.columns:
            print("Error: 'Adj Close' column missing after potential flattening. Check download structure.")
            price_col = 'Close' if 'Close' in data.columns else None
            if not price_col: raise ValueError("Neither 'Adj Close' nor 'Close' found in data.")
            print(f"Warning: Using '{price_col}' instead of 'Adj Close' for returns.")
        else:
            price_col = 'Adj Close'

        data['Returns'] = data[price_col].pct_change()

        if not benchmark_data.empty:
            if 'Adj Close' in benchmark_data.columns:
                benchmark_data['Returns'] = benchmark_data['Adj Close'].pct_change()
            elif 'Close' in benchmark_data.columns:
                print(f"Warning: Using 'Close' for benchmark returns.")
                benchmark_data['Returns'] = benchmark_data['Close'].pct_change()
            else:
                benchmark_data['Returns'] = np.nan
        else:
            benchmark_data['Returns'] = np.nan

        print("Preparing merged data...")
        ticker_df = data[[price_col, 'Returns']].copy()
        ticker_df.columns = [f'{ticker}_price', f'{ticker}_returns']

        benchmark_df = benchmark_data[['Returns']].copy()
        benchmark_df.columns = [f'{benchmark}_returns']

        merged_data = ticker_df.join(benchmark_df, how='left')
        merged_data = merged_data.dropna(subset=[f'{ticker}_returns'])

        print(f"Merged data created with shape: {merged_data.shape}")
        if merged_data.empty:
            print("Warning: Merged data is empty after join/dropna.")

        try:
            quarterly_financials = ticker_obj.quarterly_financials
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cash_flow
            print("Financial statements fetched.")
        except Exception as e:
            print(f"Warning: Could not fetch full financial statements for {ticker}: {e}")
            quarterly_financials = None
            balance_sheet = None
            cash_flow = None

        print("Data fetching complete.")
        return {
            "data": data,
            "benchmark_data": benchmark_data,
            "stock_info": stock_info,
            "company_name": company_name,
            "merged_data": merged_data,
            "quarterly_financials": quarterly_financials,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow
        }

    except Exception as e:
        print(f"An error occurred during data fetching: {e}")
        import traceback
        traceback.print_exc()
        return None
