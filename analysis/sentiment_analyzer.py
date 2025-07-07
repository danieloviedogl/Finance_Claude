import yfinance as yf
from typing import Dict, Any
from .utils import to_serializable
from curl_cffi import requests
session = requests.Session(impersonate="edge")

def sentiment_analysis(ticker: str) -> Dict[str, Any]:
    """Perform basic sentiment analysis using analyst recommendations."""
    print("Performing sentiment analysis (based on analyst ratings)...")
    try:
        ticker_obj = yf.Ticker(ticker, session=session)
        recommendations = ticker_obj.recommendations_summary

        sentiment_result = {
            'sentiment': 'Neutral',
            'sentiment_score': 0.5,
            'note': 'No analyst recommendations available or parsable.'
        }

        if recommendations is not None and not recommendations.empty:
            rec_counts = recommendations.iloc[-1].to_dict()

            strong_buy = int(rec_counts.get('strongBuy', 0))
            buy = int(rec_counts.get('buy', 0))
            hold = int(rec_counts.get('hold', 0))
            sell = int(rec_counts.get('sell', 0))
            strong_sell = int(rec_counts.get('strongSell', 0))

            total_recs = strong_buy + buy + hold + sell + strong_sell

            if total_recs > 0:
                weighted_sum = (strong_buy * 1.0 + buy * 0.75 + hold * 0.5 + sell * 0.25 + strong_sell * 0.0)
                sentiment_score = weighted_sum / total_recs

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
                    'sentiment_score': sentiment_score,
                    'sentiment': sentiment,
                    'note': f"{total_recs} analyst ratings summarized."
                }

        print("Sentiment analysis complete.")
        return to_serializable(sentiment_result)

    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return to_serializable({
            'sentiment': 'Neutral',
            'sentiment_score': 0.5,
            'error': str(e)
        })
