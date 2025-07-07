import pandas as pd
from typing import Dict, Any
from .utils import to_serializable

def generate_investment_recommendation(results: Dict[str, Any], ticker: str, company_name: str) -> Dict[str, Any]:
    """Generate final investment recommendation based on aggregated analysis scores."""
    print("Generating final investment recommendation...")

    if not results:
        print("Error: No analysis results found to generate recommendation.")
        return {"error": "Analysis results missing."}

    def get_metric(category: str, metric: str, default: Any = pd.NA) -> Any:
        val = results.get(category, {}).get(metric, default)
        return default if pd.isna(val) else val

    scores = {}
    max_scores = {'technical': 20, 'risk': 20, 'time_series': 15, 'fundamental': 20, 'monte_carlo': 15, 'portfolio': 10, 'sentiment': 10}

    # 1. Technical Score
    tech_score = 0
    ma_signal = get_metric('technical', 'ma_signal', 0)
    rsi = get_metric('technical', 'current_rsi', 50)
    macd_signal = get_metric('technical', 'macd_signal', 0)
    if ma_signal > 0: tech_score += 7
    if 30 <= rsi <= 70: tech_score += 5
    elif rsi < 30: tech_score += 7
    if macd_signal > 0: tech_score += 6
    scores['technical'] = min(tech_score, max_scores['technical'])

    # 2. Risk Score
    risk_score = 0
    sharpe = get_metric('risk_metrics', 'sharpe_ratio', 0)
    sortino = get_metric('risk_metrics', 'sortino_ratio', 0)
    max_drawdown = get_metric('risk_metrics', 'max_drawdown', -1.0)
    alpha = get_metric('risk_metrics', 'alpha', 0)
    if sharpe > 1.0: risk_score += 5
    elif sharpe > 0.5: risk_score += 3
    if sortino > 1.5: risk_score += 5
    elif sortino > 0.75: risk_score += 3
    if max_drawdown > -0.20: risk_score += 5
    elif max_drawdown > -0.35: risk_score += 3
    if alpha > 0.01: risk_score += 5
    scores['risk'] = min(risk_score, max_scores['risk'])

    # 3. Time Series Score
    ts_score = 0
    expected_return_30d = get_metric('time_series', 'expected_return_30d', 0)
    is_stationary = get_metric('time_series', 'is_stationary', False)
    if expected_return_30d > 3: ts_score += 7
    elif expected_return_30d > 1: ts_score += 4
    if is_stationary: ts_score += 5
    vol_forecast = get_metric('time_series', 'volatility_forecast_30d_daily', 0.02)
    if vol_forecast < 0.015: ts_score += 3
    scores['time_series'] = min(ts_score, max_scores['time_series'])

    # 4. Fundamental Score
    fund_score = 0
    pe = get_metric('fundamental', 'pe_ratio', 100)
    target_upside = get_metric('fundamental', 'price_target_upside_percent', -100)
    profit_margin = get_metric('fundamental', 'profit_margins', -1)
    roe = get_metric('fundamental', 'return_on_equity', -1)
    if 0 < pe < 20: fund_score += 5
    elif 20 <= pe < 35: fund_score += 3
    if target_upside > 20: fund_score += 7
    elif target_upside > 5: fund_score += 4
    if profit_margin > 0.15: fund_score += 5
    elif profit_margin > 0.05: fund_score += 3
    if roe > 0.15: fund_score += 3
    scores['fundamental'] = min(fund_score, max_scores['fundamental'])

    # 5. Monte Carlo Score
    mc_score = 0
    mc_exp_return = get_metric('monte_carlo', 'expected_return_percent', 0)
    mc_prob_profit = get_metric('monte_carlo', 'prob_profit', 0)
    mc_std_dev_pct = (get_metric('monte_carlo', 'std_final_price', 0) /
                      get_metric('monte_carlo', 'current_price', 1)) * 100 if get_metric('monte_carlo', 'current_price') else 0
    if mc_exp_return > 15: mc_score += 7
    elif mc_exp_return > 5: mc_score += 4
    if mc_prob_profit > 60: mc_score += 5
    elif mc_prob_profit > 50: mc_score += 3
    if mc_std_dev_pct < 30 and mc_std_dev_pct > 0 : mc_score += 3
    scores['monte_carlo'] = min(mc_score, max_scores['monte_carlo'])

    # 6. Sentiment Score
    sent_score = 0
    sentiment_val = get_metric('sentiment', 'sentiment_score', 0.5)
    sent_score = sentiment_val * 10
    scores['sentiment'] = min(round(sent_score), max_scores['sentiment'])

    total_score = sum(scores.values()) / 0.9
    max_possible_score = sum(max_scores.values())
    normalized_score = round((total_score / max_possible_score) * 100) if max_possible_score else 0

    if normalized_score >= 80:
        recommendation = "Strong Buy"; confidence = "High"; color = "#15803D"
    elif normalized_score >= 65:
        recommendation = "Buy"; confidence = "Moderate"; color = "#65A30D"
    elif normalized_score >= 50:
        recommendation = "Hold"; confidence = "Moderate"; color = "#CA8A04"
    elif normalized_score >= 35:
        recommendation = "Reduce"; confidence = "Moderate"; color = "#EA580C"
    else:
        recommendation = "Sell"; confidence = "High"; color = "#DC2626"

    recommendation_text = f"Overall score of {normalized_score}/100 suggests a '{recommendation}' stance for {company_name} ({ticker}). "

    final_recommendation = {
        'ticker': ticker,
        'company_name': company_name,
        'recommendation': recommendation,
        'recommendation_text': recommendation_text,
        'recommendation_color': color,
        'confidence': confidence,
        'total_score_normalized': normalized_score,
        'category_scores': to_serializable(scores),
        'max_category_scores': max_scores,
        'current_price': get_metric('technical', 'current_price', None),
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'key_metrics': {
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
                'sentiment': get_metric('sentiment', 'sentiment', 'Neutral'),
                'sentiment_score_0_1': round(sentiment_val, 2) if pd.notna(sentiment_val) else None
            },
        },
    }

    print("Recommendation generated.")
    return to_serializable(final_recommendation)
