import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .utils import to_serializable

def perform_technical_analysis(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Calculate technical indicators and generate simple signals."""
    if data is None or data.empty:
        print("Error: No data available for technical analysis.")
        return None
    if len(data) < 200:
        print("Warning: Insufficient data for 200-day SMA. Some indicators might be NaN.")

    print("Performing technical analysis...")
    tech_data = data.copy()

    tech_data['SMA_50'] = tech_data['Adj Close'].rolling(window=50).mean()
    tech_data['SMA_200'] = tech_data['Adj Close'].rolling(window=200).mean()
    tech_data['EMA_20'] = tech_data['Adj Close'].ewm(span=20, adjust=False).mean()

    try:
        sma_20 = tech_data['Adj Close'].rolling(window=20).mean()
        rolling_std_20 = tech_data['Adj Close'].rolling(window=20).std()
        tech_data['upper_band'] = sma_20 + (rolling_std_20 * 2)
        tech_data['lower_band'] = sma_20 - (rolling_std_20 * 2)
        tech_data['20d_rolling_std'] = rolling_std_20
        print("Bollinger Bands calculated successfully.")
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        tech_data['upper_band'] = np.nan
        tech_data['lower_band'] = np.nan
        tech_data['20d_rolling_std'] = np.nan
        import traceback
        traceback.print_exc()

    delta = tech_data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0.0).fillna(0.0)
    loss = -delta.where(delta < 0, 0.0).fillna(0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    tech_data['RSI'] = 100 - (100 / (1 + rs))
    tech_data['RSI'] = tech_data['RSI'].fillna(50)

    ema_12 = tech_data['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_26 = tech_data['Adj Close'].ewm(span=26, adjust=False).mean()
    tech_data['MACD'] = ema_12 - ema_26
    tech_data['MACD_Signal'] = tech_data['MACD'].ewm(span=9, adjust=False).mean()
    tech_data['MACD_Hist'] = tech_data['MACD'] - tech_data['MACD_Signal']

    if 'Volume' in tech_data.columns and pd.api.types.is_numeric_dtype(tech_data['Volume']):
        tech_data['OBV'] = (np.sign(tech_data['Adj Close'].diff().fillna(0)) * tech_data['Volume']).cumsum()
    else:
        tech_data['OBV'] = np.nan
        print("Warning: Volume data missing or non-numeric, OBV not calculated.")

    tech_data['Momentum'] = tech_data['Adj Close'] / tech_data['Adj Close'].shift(10) - 1

    required_cols = ['High', 'Low', 'Adj Close']
    if all(col in tech_data.columns for col in required_cols):
        try:
            high_low = tech_data['High'] - tech_data['Low']
            high_close = np.abs(tech_data['High'] - tech_data['Adj Close'].shift())
            low_close = np.abs(tech_data['Low'] - tech_data['Adj Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range_np = np.nanmax(ranges, axis=1)
            true_range_series = pd.Series(true_range_np, index=tech_data.index, name='TrueRange')
            tech_data['ATR'] = true_range_series.rolling(14, min_periods=1).mean()
            print("ATR calculated successfully.")
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            tech_data['ATR'] = np.nan
            import traceback
            traceback.print_exc()
    else:
        tech_data['ATR'] = np.nan
        print(f"Warning: Required columns ({', '.join(required_cols)}) missing for ATR calculation.")

    tech_data['MA_Signal'] = np.where(tech_data['SMA_50'] > tech_data['SMA_200'], 1, np.where(tech_data['SMA_50'] < tech_data['SMA_200'], -1, 0))
    tech_data['RSI_Signal'] = np.where(tech_data['RSI'] < 30, 1, np.where(tech_data['RSI'] > 70, -1, 0))
    tech_data['MACD_Signal_Line'] = np.where(tech_data['MACD'] > tech_data['MACD_Signal'], 1, np.where(tech_data['MACD'] < tech_data['MACD_Signal'], -1, 0))

    tech_data['Tech_Signal'] = tech_data[['MA_Signal', 'RSI_Signal', 'MACD_Signal_Line']].mean(axis=1, skipna=True)

    tech_data = tech_data.dropna(subset=['Adj Close'])

    results = {
        'technical_data': tech_data,
        'analysis': to_serializable({
            'current_price': tech_data['Adj Close'].iloc[-1] if not tech_data.empty else None,
            'current_rsi': tech_data['RSI'].iloc[-1] if not tech_data.empty and 'RSI' in tech_data.columns else None,
            'ma_signal': tech_data['MA_Signal'].iloc[-1] if not tech_data.empty and 'MA_Signal' in tech_data.columns and pd.notna(tech_data['MA_Signal'].iloc[-1]) else 0,
            'macd_signal': tech_data['MACD_Signal_Line'].iloc[-1] if not tech_data.empty and 'MACD_Signal_Line' in tech_data.columns and pd.notna(tech_data['MACD_Signal_Line'].iloc[-1]) else 0,
            'combined_signal': tech_data['Tech_Signal'].iloc[-1] if not tech_data.empty and 'Tech_Signal' in tech_data.columns else None
        })
    }
    
    print("Technical analysis complete.")
    return results
