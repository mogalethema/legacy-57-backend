import MetaTrader5 as mt5
import pandas as pd
import numpy as np

def get_symbol_data(symbol, timeframe=mt5.TIMEFRAME_M15, bars=500):
    # initialize MT5 connection
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError("No data retrieved for symbol")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Strategy 1: Moving Average Crossover
def strategy_moving_average(df, short_window=20, long_window=50):
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    if df['ma_short'].iloc[-2] < df['ma_long'].iloc[-2] and df['ma_short'].iloc[-1] > df['ma_long'].iloc[-1]:
        return "buy"
    elif df['ma_short'].iloc[-2] > df['ma_long'].iloc[-2] and df['ma_short'].iloc[-1] < df['ma_long'].iloc[-1]:
        return "sell"
    else:
        return "hold"

# Strategy 2: RSI Overbought/Oversold
def strategy_rsi(df, period=14, overbought=70, oversold=30):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    if rsi.iloc[-1] > overbought:
        return "sell"
    elif rsi.iloc[-1] < oversold:
        return "buy"
    else:
        return "hold"

# Strategy 3: Bollinger Bands Breakout
def strategy_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    if prev_close < lower_band.iloc[-2] and last_close > lower_band.iloc[-1]:
        return "buy"
    elif prev_close > upper_band.iloc[-2] and last_close < upper_band.iloc[-1]:
        return "sell"
    else:
        return "hold"

# Strategy 4: MACD Crossover
def strategy_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    df['macd'] = macd
    df['signal_line'] = signal_line
    if df['macd'].iloc[-2] < df['signal_line'].iloc[-2] and df['macd'].iloc[-1] > df['signal_line'].iloc[-1]:
        return "buy"
    elif df['macd'].iloc[-2] > df['signal_line'].iloc[-2] and df['macd'].iloc[-1] < df['signal_line'].iloc[-1]:
        return "sell"
    else:
        return "hold"

# Combine signals from all strategies
def combined_signal(symbol):
    df = get_symbol_data(symbol)
    signals = {
        "ma_crossover": strategy_moving_average(df),
        "rsi": strategy_rsi(df),
        "bollinger": strategy_bollinger_bands(df),
        "macd": strategy_macd(df)
    }
    # Simple consensus: if 2 or more say buy => buy; 2 or more say sell => sell; else hold
    buy_count = sum(1 for s in signals.values() if s == "buy")
    sell_count = sum(1 for s in signals.values() if s == "sell")
    if buy_count >= 2:
        overall = "buy"
    elif sell_count >= 2:
        overall = "sell"
    else:
        overall = "hold"
    return {"overall": overall, "details": signals}
