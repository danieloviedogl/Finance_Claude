import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .utils import to_serializable

def calculate_fundamental_metrics(stock_info: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate and evaluate fundamental metrics from stock info."""
    if not stock_info:
        print("Warning: Stock info not available for fundamental analysis.")
        return {'error': 'Stock info unavailable'}

    print("Calculating fundamental metrics...")

    def safe_get_numeric(key: str, default: Any = np.nan) -> Optional[float]:
        val = stock_info.get(key, default)
        if isinstance(val, (int, float)) and pd.notna(val) and not np.isinf(val):
            return float(val)
        return default if default is np.nan else float(default)

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
        'dividend_yield': safe_get_numeric('dividendYield', 0) * 100,
        'payout_ratio': safe_get_numeric('payoutRatio'),
        'debt_to_equity': safe_get_numeric('debtToEquity'),
        'return_on_equity': safe_get_numeric('returnOnEquity'),
        'return_on_assets': safe_get_numeric('returnOnAssets'),
        'current_ratio': safe_get_numeric('currentRatio'),
        'quick_ratio': safe_get_numeric('quickRatio'),
        'target_median_price': safe_get_numeric('targetMedianPrice'),
        'analyst_recommendation_mean': safe_get_numeric('recommendationMean'),
        'analyst_recommendation_key': stock_info.get('recommendationKey', 'N/A'),
        'number_of_analyst_opinions': safe_get_numeric('numberOfAnalystOpinions', 0)
    }

    current_price = data['Adj Close'].iloc[-1] if data is not None and not data.empty else None
    target_price = fund_metrics['target_median_price']

    if current_price is not None and current_price > 0 and target_price is not None and pd.notna(target_price):
        fund_metrics['price_target_upside_percent'] = (target_price / current_price - 1) * 100
    else:
        fund_metrics['price_target_upside_percent'] = np.nan

    print("Fundamental metrics calculation complete.")
    return to_serializable(fund_metrics)
