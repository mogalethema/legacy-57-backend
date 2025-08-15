from flask import Flask, jsonify, request
from strategies import all_strategies
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime

app = Flask(__name__)

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialize() failed")
    mt5.shutdown()

def get_candles(symbol: str, timeframe: int, count: int = 500):
    """
    Fetch candle data for a symbol and timeframe from MT5.
    :param symbol: e.g., "EURUSD"
    :param timeframe: mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, etc.
    :param count: number of candles
    :return: pandas DataFrame with OHLCV and time index
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Map string timeframes to mt5 constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    if df['close'].iloc[-1] > ema.iloc[-1] and width.iloc[-1] < width.mean()*0.5 and hist.iloc[-1] > 0 and rsi.iloc[-1] < 55:
        return "BUY", "EMA Pullback Long# Placeholder for strategy functions, will be added next

# === Register Strategies 999–1000
allStrategies.extend([strategy999, strategy1000])


    # Call all strategies here and collect results
    signals = {}
    explanations = {}

    # Strategies will populate signals and explanations dicts here

    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe_str,
        "signals": signals,
        "explanations": explanations,
        "timestamp": datetime.datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
