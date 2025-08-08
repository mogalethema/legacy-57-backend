from flask import Flask, jsonify, request
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

# Placeholder for strategy functions, will be added next
# === STRATEGY 1: Moving Average Crossover ===
def moving_average_crossover(df):
    df['ma_fast'] = df['close'].rolling(window=9).mean()
    df['ma_slow'] = df['close'].rolling(window=21).mean()
    if df['ma_fast'].iloc[-2] < df['ma_slow'].iloc[-2] and df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
        return "BUY", "Fast MA crossed above Slow MA - potential uptrend."
    elif df['ma_fast'].iloc[-2] > df['ma_slow'].iloc[-2] and df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1]:
        return "SELL", "Fast MA crossed below Slow MA - potential downtrend."
    else:
        return "HOLD", "No crossover signal."

# === STRATEGY 2: RSI Overbought/Oversold ===
def rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_strategy(df):
    df['rsi'] = rsi(df)
    latest_rsi = df['rsi'].iloc[-1]
    if latest_rsi < 30:
        return "BUY", f"RSI at {latest_rsi:.2f} indicates oversold conditions."
    elif latest_rsi > 70:
        return "SELL", f"RSI at {latest_rsi:.2f} indicates overbought conditions."
    else:
        return "HOLD", f"RSI at {latest_rsi:.2f} - no clear signal."

# === STRATEGY 3: MACD Crossover ===
def macd_strategy(df):
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return "BUY", "MACD crossed above signal line - bullish momentum."
    elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return "SELL", "MACD crossed below signal line - bearish momentum."
    else:
        return "HOLD", "No MACD signal."

# === STRATEGY 4: Bollinger Bands Bounce ===
def bollinger_bands_strategy(df, period=20):
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper'] = df['ma'] + 2 * df['std']
    df['lower'] = df['ma'] - 2 * df['std']
    price = df['close'].iloc[-1]
    if price < df['lower'].iloc[-1]:
        return "BUY", "Price below lower Bollinger Band - potential bounce."
    elif price > df['upper'].iloc[-1]:
        return "SELL", "Price above upper Bollinger Band - potential reversal."
    else:
        return "HOLD", "Price within Bollinger Bands."

# === STRATEGY 5: Donchian Channel Breakout ===
def donchian_breakout(df, period=20):
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    if df['close'].iloc[-1] > upper.iloc[-2]:
        return "BUY", "Breakout above Donchian channel - bullish signal."
    elif df['close'].iloc[-1] < lower.iloc[-2]:
        return "SELL", "Breakdown below Donchian channel - bearish signal."
    else:
        return "HOLD", "No Donchian breakout."
# === STRATEGY 6: Stochastic Oscillator ===
def stochastic_oscillator_strategy(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    df['%K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    if df['%K'].iloc[-1] < 20 and df['%D'].iloc[-1] < 20:
        return "BUY", "Stochastic indicates oversold condition."
    elif df['%K'].iloc[-1] > 80 and df['%D'].iloc[-1] > 80:
        return "SELL", "Stochastic indicates overbought condition."
    else:
        return "HOLD", "Stochastic neutral."

# === STRATEGY 7: Ichimoku Cloud Cross ===
def ichimoku_strategy(df):
    nine_high = df['high'].rolling(window=9).max()
    nine_low = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (nine_high + nine_low) / 2

    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    if df['tenkan_sen'].iloc[-1] > df['kijun_sen'].iloc[-1]:
        return "BUY", "Tenkan-sen crossed above Kijun-sen."
    elif df['tenkan_sen'].iloc[-1] < df['kijun_sen'].iloc[-1]:
        return "SELL", "Tenkan-sen crossed below Kijun-sen."
    else:
        return "HOLD", "Ichimoku lines aligned."

# === STRATEGY 8: EMA Trend Following ===
def ema_trend_strategy(df, short_period=10, long_period=50):
    df['ema_short'] = df['close'].ewm(span=short_period).mean()
    df['ema_long'] = df['close'].ewm(span=long_period).mean()
    if df['ema_short'].iloc[-1] > df['ema_long'].iloc[-1]:
        return "BUY", "Short EMA above long EMA - uptrend."
    elif df['ema_short'].iloc[-1] < df['ema_long'].iloc[-1]:
        return "SELL", "Short EMA below long EMA - downtrend."
    else:
        return "HOLD", "EMAs converging."

# === STRATEGY 9: Heikin-Ashi Reversal ===
def heikin_ashi_strategy(df):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_close[i-1]) / 2)
    df['ha_open'] = ha_open
    df['ha_close'] = ha_close
    if df['ha_close'].iloc[-1] > df['ha_open'].iloc[-1]:
        return "BUY", "Heikin-Ashi indicates bullish trend."
    elif df['ha_close'].iloc[-1] < df['ha_open'].iloc[-1]:
        return "SELL", "Heikin-Ashi indicates bearish trend."
    else:
        return "HOLD", "Heikin-Ashi neutral."

# === STRATEGY 10: Volume Spike ===
def volume_spike_strategy(df):
    avg_volume = df['real_volume'].rolling(window=20).mean()
    if df['real_volume'].iloc[-1] > 1.5 * avg_volume.iloc[-1]:
        return "BUY", "Unusual volume spike detected."
    else:
        return "HOLD", "Volume normal."

# === STRATEGY 11: CCI Strategy ===
def cci_strategy(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - ma) / (0.015 * md)
    df['cci'] = cci
    if cci.iloc[-1] > 100:
        return "BUY", f"CCI = {cci.iloc[-1]:.2f} (bullish)"
    elif cci.iloc[-1] < -100:
        return "SELL", f"CCI = {cci.iloc[-1]:.2f} (bearish)"
    else:
        return "HOLD", f"CCI = {cci.iloc[-1]:.2f} (neutral)"

# === STRATEGY 12: Envelope Strategy ===
def envelope_strategy(df, period=20, deviation=0.02):
    ma = df['close'].rolling(window=period).mean()
    upper = ma * (1 + deviation)
    lower = ma * (1 - deviation)
    price = df['close'].iloc[-1]
    if price < lower.iloc[-1]:
        return "BUY", "Price below lower envelope - potential bounce."
    elif price > upper.iloc[-1]:
        return "SELL", "Price above upper envelope - potential reversal."
    else:
        return "HOLD", "Price within envelope."

# === STRATEGY 13: Price Action Engulfing ===
def engulfing_strategy(df):
    open1, close1 = df['open'].iloc[-2], df['close'].iloc[-2]
    open2, close2 = df['open'].iloc[-1], df['close'].iloc[-1]
    if close1 < open1 and close2 > open2 and close2 > open1 and open2 < close1:
        return "BUY", "Bullish engulfing pattern."
    elif close1 > open1 and close2 < open2 and close2 < open1 and open2 > close1:
        return "SELL", "Bearish engulfing pattern."
    else:
        return "HOLD", "No engulfing pattern."

# === STRATEGY 14: TEMA Cross ===
def tema_strategy(df):
    ema1 = df['close'].ewm(span=10).mean()
    ema2 = ema1.ewm(span=10).mean()
    tema = 3 * ema1 - 3 * ema2 + ema2.ewm(span=10).mean()
    df['tema'] = tema
    if df['close'].iloc[-2] < tema.iloc[-2] and df['close'].iloc[-1] > tema.iloc[-1]:
        return "BUY", "Price crossed above TEMA."
    elif df['close'].iloc[-2] > tema.iloc[-2] and df['close'].iloc[-1] < tema.iloc[-1]:
        return "SELL", "Price crossed below TEMA."
    else:
        return "HOLD", "No TEMA signal."

# === STRATEGY 15: ATR Breakout ===
def atr_strategy(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    df['atr'] = atr
    if df['close'].iloc[-1] > df['close'].iloc[-2] + atr.iloc[-2]:
        return "BUY", "Price broke above ATR level."
    elif df['close'].iloc[-1] < df['close'].iloc[-2] - atr.iloc[-2]:
        return "SELL", "Price broke below ATR level."
    else:
        return "HOLD", "No ATR breakout."
# === STRATEGY 16: Bollinger Band Squeeze ===
def bollinger_squeeze_strategy(df, period=20):
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    squeeze = (upper - lower).iloc[-1]
    avg_squeeze = (upper - lower).mean()
    if squeeze < 0.5 * avg_squeeze:
        return "WAIT", "Bollinger Bands squeezed - possible breakout."
    elif df['close'].iloc[-1] > upper.iloc[-1]:
        return "BUY", "Breakout above Bollinger Band."
    elif df['close'].iloc[-1] < lower.iloc[-1]:
        return "SELL", "Breakout below Bollinger Band."
    else:
        return "HOLD", "No breakout yet."

# === STRATEGY 17: Triple EMA (TRIX) ===
def trix_strategy(df, period=15):
    ema1 = df['close'].ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    df['trix'] = trix
    if trix.iloc[-1] > 0:
        return "BUY", f"TRIX above 0: {trix.iloc[-1]:.2f}"
    elif trix.iloc[-1] < 0:
        return "SELL", f"TRIX below 0: {trix.iloc[-1]:.2f}"
    else:
        return "HOLD", "TRIX at zero."

# === STRATEGY 18: DMI / ADX Trend Strength ===
def adx_strategy(df, period=14):
    up_move = df['high'].diff()
    down_move = df['low'].diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=period).mean()

    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).sum() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean().iloc[-1]

    if adx > 25:
        return "TRENDING", f"ADX = {adx:.2f} (strong trend)"
    else:
        return "RANGING", f"ADX = {adx:.2f} (weak trend)"

# === STRATEGY 19: Donchian Channel Breakout ===
def donchian_breakout_strategy(df, period=20):
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    if df['close'].iloc[-1] > upper.iloc[-2]:
        return "BUY", "Donchian breakout upwards."
    elif df['close'].iloc[-1] < lower.iloc[-2]:
        return "SELL", "Donchian breakout downwards."
    else:
        return "HOLD", "Inside Donchian channel."

# === STRATEGY 20: Keltner Channel ===
def keltner_channel_strategy(df, ema_period=20, atr_period=10, multiplier=2):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    ema = typical_price.ewm(span=ema_period).mean()
    tr = df['high'] - df['low']
    atr = tr.rolling(window=atr_period).mean()
    upper_band = ema + (multiplier * atr)
    lower_band = ema - (multiplier * atr)
    close = df['close'].iloc[-1]

    if close > upper_band.iloc[-1]:
        return "SELL", "Price above Keltner channel (overbought)."
    elif close < lower_band.iloc[-1]:
        return "BUY", "Price below Keltner channel (oversold)."
    else:
        return "HOLD", "Price inside Keltner channel."

# === STRATEGY 21: Hull Moving Average (HMA) ===
def hma_strategy(df, period=21):
    wma_half = df['close'].rolling(window=int(period/2)).mean()
    wma_full = df['close'].rolling(window=period).mean()
    hma = 2 * wma_half - wma_full
    hma = hma.rolling(window=int(np.sqrt(period))).mean()
    df['hma'] = hma
    if df['close'].iloc[-1] > hma.iloc[-1]:
        return "BUY", "Price above HMA"
    elif df['close'].iloc[-1] < hma.iloc[-1]:
        return "SELL", "Price below HMA"
    else:
        return "HOLD", "Price at HMA"

# === STRATEGY 22: Z-Score Reversion ===
def zscore_strategy(df, period=20):
    mean = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    zscore = (df['close'] - mean) / std
    latest = zscore.iloc[-1]
    if latest > 2:
        return "SELL", f"Z-score = {latest:.2f} (price overextended)"
    elif latest < -2:
        return "BUY", f"Z-score = {latest:.2f} (price undervalued)"
    else:
        return "HOLD", f"Z-score = {latest:.2f} (normal range)"

# === STRATEGY 23: Parabolic SAR ===
def parabolic_sar_strategy(df, af=0.02, max_af=0.2):
    sar = df['low'][0]
    ep = df['high'][0]
    trend = 1
    af_current = af
    for i in range(1, len(df)):
        sar = sar + af_current * (ep - sar)
        if trend == 1:
            if df['low'][i] < sar:
                trend = -1
                sar = ep
                ep = df['low'][i]
                af_current = af
            else:
                if df['high'][i] > ep:
                    ep = df['high'][i]
                    af_current = min(af_current + af, max_af)
        else:
            if df['high'][i] > sar:
                trend = 1
                sar = ep
                ep = df['high'][i]
                af_current = af
            else:
                if df['low'][i] < ep:
                    ep = df['low'][i]
                    af_current = min(af_current + af, max_af)

    if trend == 1:
        return "BUY", "Parabolic SAR indicates uptrend."
    else:
        return "SELL", "Parabolic SAR indicates downtrend."

# === STRATEGY 24: Ultimate Oscillator ===
def ultimate_oscillator_strategy(df):
    bp = df['close'] - np.minimum(df['low'], df['close'].shift())
    tr = np.maximum(df['high'], df['close'].shift()) - np.minimum(df['low'], df['close'].shift())
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    uo_val = uo.iloc[-1]
    if uo_val < 30:
        return "BUY", f"UO = {uo_val:.2f} (bullish divergence)"
    elif uo_val > 70:
        return "SELL", f"UO = {uo_val:.2f} (bearish divergence)"
    else:
        return "HOLD", f"UO = {uo_val:.2f} (neutral)"

# === STRATEGY 25: Williams %R ===
def williams_r_strategy(df, period=14):
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    wr = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    latest = wr.iloc[-1]
    if latest < -80:
        return "BUY", f"Williams %R = {latest:.2f} (oversold)"
    elif latest > -20:
        return "SELL", f"Williams %R = {latest:.2f} (overbought)"
    else:
        return "HOLD", f"Williams %R = {latest:.2f} (neutral)"
# === STRATEGY 26: Price Action - Engulfing Pattern ===
def engulfing_pattern_strategy(df):
    body_prev = df['close'].shift(1) - df['open'].shift(1)
    body_curr = df['close'] - df['open']
    is_bullish = (body_prev < 0) & (body_curr > 0) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    is_bearish = (body_prev > 0) & (body_curr < 0) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))

    if is_bullish.iloc[-1]:
        return "BUY", "Bullish engulfing pattern detected"
    elif is_bearish.iloc[-1]:
        return "SELL", "Bearish engulfing pattern detected"
    else:
        return "HOLD", "No engulfing pattern"

# === STRATEGY 27: Price Action - Doji Reversal ===
def doji_reversal_strategy(df):
    body = abs(df['close'] - df['open'])
    range_total = df['high'] - df['low']
    doji = (body / range_total) < 0.1
    if doji.iloc[-1]:
        return "CAUTION", "Doji candle detected - possible reversal"
    return "HOLD", "No doji pattern"

# === STRATEGY 28: Gap Trading ===
def gap_trading_strategy(df):
    prev_close = df['close'].shift(1)
    gap = df['open'] - prev_close
    if gap.iloc[-1] > 0.005 * prev_close.iloc[-1]:
        return "SELL", "Gap up detected - possible mean reversion"
    elif gap.iloc[-1] < -0.005 * prev_close.iloc[-1]:
        return "BUY", "Gap down detected - possible mean reversion"
    else:
        return "HOLD", "No significant gap"

# === STRATEGY 29: EMA Ribbon (Multi-EMA Signal) ===
def ema_ribbon_strategy(df):
    ema8 = df['close'].ewm(span=8).mean()
    ema13 = df['close'].ewm(span=13).mean()
    ema21 = df['close'].ewm(span=21).mean()
    if ema8.iloc[-1] > ema13.iloc[-1] > ema21.iloc[-1]:
        return "BUY", "EMA ribbon bullish crossover"
    elif ema8.iloc[-1] < ema13.iloc[-1] < ema21.iloc[-1]:
        return "SELL", "EMA ribbon bearish crossover"
    else:
        return "HOLD", "EMAs not aligned"

# === STRATEGY 30: Ichimoku Kinko Hyo ===
def ichimoku_strategy(df):
    nine_high = df['high'].rolling(window=9).max()
    nine_low = df['low'].rolling(window=9).min()
    tenkan_sen = (nine_high + nine_low) / 2

    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1]:
        return "BUY", "Tenkan > Kijun - bullish Ichimoku signal"
    elif tenkan_sen.iloc[-1] < kijun_sen.iloc[-1]:
        return "SELL", "Tenkan < Kijun - bearish Ichimoku signal"
    else:
        return "HOLD", "Ichimoku neutral"

# === STRATEGY 31: TEMA (Triple EMA) ===
def tema_strategy(df, period=14):
    ema1 = df['close'].ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    if df['close'].iloc[-1] > tema.iloc[-1]:
        return "BUY", "Price above TEMA"
    elif df['close'].iloc[-1] < tema.iloc[-1]:
        return "SELL", "Price below TEMA"
    else:
        return "HOLD", "Price near TEMA"

# === STRATEGY 32: Stochastic Momentum Index (SMI) ===
def smi_strategy(df, period=14):
    min_low = df['low'].rolling(window=period).min()
    max_high = df['high'].rolling(window=period).max()
    median = (max_high + min_low) / 2
    smi = 100 * ((df['close'] - median) / ((max_high - min_low) / 2))
    val = smi.iloc[-1]
    if val > 40:
        return "BUY", f"SMI = {val:.2f} (bullish)"
    elif val < -40:
        return "SELL", f"SMI = {val:.2f} (bearish)"
    else:
        return "HOLD", f"SMI = {val:.2f} (neutral)"

# === STRATEGY 33: Money Flow Index (MFI) ===
def mfi_strategy(df, period=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['tick_volume']
    positive_flow = np.where(typical_price > typical_price.shift(), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(), raw_money_flow, 0)
    pos_sum = pd.Series(positive_flow).rolling(window=period).sum()
    neg_sum = pd.Series(negative_flow).rolling(window=period).sum()
    mfi = 100 - (100 / (1 + (pos_sum / neg_sum)))
    val = mfi.iloc[-1]
    if val < 20:
        return "BUY", f"MFI = {val:.2f} (oversold)"
    elif val > 80:
        return "SELL", f"MFI = {val:.2f} (overbought)"
    else:
        return "HOLD", f"MFI = {val:.2f} (neutral)"

# === STRATEGY 34: Fractal Breakout ===
def fractal_breakout_strategy(df):
    highs = df['high']
    lows = df['low']
    recent_high = highs.iloc[-3]
    recent_low = lows.iloc[-3]
    if df['close'].iloc[-1] > recent_high:
        return "BUY", "Fractal breakout up"
    elif df['close'].iloc[-1] < recent_low:
        return "SELL", "Fractal breakout down"
    else:
        return "HOLD", "No fractal breakout"

# === STRATEGY 35: Volume Spike Detection ===
def volume_spike_strategy(df, threshold=2.5):
    mean_vol = df['tick_volume'].rolling(window=20).mean()
    vol = df['tick_volume'].iloc[-1]
    if vol > threshold * mean_vol.iloc[-1]:
        return "VOLUME SPIKE", f"Volume spike detected: {vol}"
    else:
        return "NORMAL", f"Volume normal: {vol}"
# === STRATEGY 36: ADX (Average Directional Index) ===
def adx_strategy(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    val = adx.iloc[-1]
    if val > 25:
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            return "BUY", f"ADX = {val:.2f} Strong Trend Up"
        else:
            return "SELL", f"ADX = {val:.2f} Strong Trend Down"
    else:
        return "HOLD", f"ADX = {val:.2f} Weak trend"

# === STRATEGY 37: Triple Top/Bottom Detection ===
def triple_pattern_strategy(df):
    tops = df['high'].nlargest(3)
    bottoms = df['low'].nsmallest(3)
    if tops.std() < 0.0005:
        return "SELL", "Possible Triple Top - Reversal Down"
    elif bottoms.std() < 0.0005:
        return "BUY", "Possible Triple Bottom - Reversal Up"
    else:
        return "HOLD", "No triple pattern detected"

# === STRATEGY 38: Parabolic SAR ===
def parabolic_sar_strategy(df):
    try:
        import ta
        psar = ta.trend.psar_up(df['high'], df['low'], df['close'], step=0.02, max_step=0.2)
        if df['close'].iloc[-1] > psar.iloc[-1]:
            return "BUY", "Price above PSAR - Bullish"
        elif df['close'].iloc[-1] < psar.iloc[-1]:
            return "SELL", "Price below PSAR - Bearish"
        else:
            return "HOLD", "Price near PSAR"
    except:
        return "HOLD", "PSAR requires `ta` library installed"

# === STRATEGY 39: Pivot Point Strategy ===
def pivot_point_strategy(df):
    high = df['high'].iloc[-2]
    low = df['low'].iloc[-2]
    close = df['close'].iloc[-2]

    pivot = (high + low + close) / 3
    support = pivot - (high - low)
    resistance = pivot + (high - low)

    price = df['close'].iloc[-1]
    if price > resistance:
        return "BUY", "Price above resistance - bullish breakout"
    elif price < support:
        return "SELL", "Price below support - bearish breakout"
    else:
        return "HOLD", "Price within pivot range"

# === STRATEGY 40: Keltner Channel ===
def keltner_channel_strategy(df, ema_period=20, atr_period=10):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    ema = typical_price.ewm(span=ema_period).mean()
    atr = (df['high'] - df['low']).rolling(window=atr_period).mean()

    upper = ema + 2 * atr
    lower = ema - 2 * atr
    price = df['close'].iloc[-1]

    if price > upper.iloc[-1]:
        return "SELL", "Price above Keltner Channel - reversal likely"
    elif price < lower.iloc[-1]:
        return "BUY", "Price below Keltner Channel - reversal likely"
    else:
        return "HOLD", "Price inside Keltner Channel"

# === STRATEGY 41: ZigZag Pattern Detection ===
def zigzag_strategy(df):
    # A very simplified zigzag logic
    price = df['close']
    zigzag = price.diff().abs().rolling(window=5).mean()
    signal = "HOLD"
    if zigzag.iloc[-1] > zigzag.mean():
        signal = "POTENTIAL ENTRY", "High volatility detected via ZigZag"
    return signal if isinstance(signal, tuple) else ("HOLD", signal)

# === STRATEGY 42: OBV (On Balance Volume) ===
def obv_strategy(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i-1]:
            obv.append(obv[-1] + df['tick_volume'][i])
        elif df['close'][i] < df['close'][i-1]:
            obv.append(obv[-1] - df['tick_volume'][i])
        else:
            obv.append(obv[-1])
    if obv[-1] > np.mean(obv):
        return "BUY", "OBV rising - buyers in control"
    elif obv[-1] < np.mean(obv):
        return "SELL", "OBV falling - sellers in control"
    else:
        return "HOLD", "OBV stable"

# === STRATEGY 43: Williams %R ===
def williams_r_strategy(df, period=14):
    high = df['high'].rolling(period).max()
    low = df['low'].rolling(period).min()
    wr = -100 * ((high - df['close']) / (high - low))
    val = wr.iloc[-1]
    if val < -80:
        return "BUY", f"Williams %R = {val:.2f} (Oversold)"
    elif val > -20:
        return "SELL", f"Williams %R = {val:.2f} (Overbought)"
    else:
        return "HOLD", f"Williams %R = {val:.2f} (Neutral)"

# === STRATEGY 44: Bollinger Band Width Breakout ===
def bb_width_breakout_strategy(df, period=20):
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    width = upper - lower
    if width.iloc[-1] < width.mean() * 0.5:
        return "BREAKOUT COMING", "Low volatility detected - prepare for breakout"
    return "HOLD", "Volatility normal"

# === STRATEGY 45: Heiken Ashi Trend Detection ===
def heiken_ashi_strategy(df):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
    if ha_close.iloc[-1] > ha_open.iloc[-1]:
        return "BUY", "Heiken Ashi shows bullish trend"
    elif ha_close.iloc[-1] < ha_open.iloc[-1]:
        return "SELL", "Heiken Ashi shows bearish trend"
    else:
        return "HOLD", "Neutral Heiken Ashi candle"
        
def bollinger_bands_strategy(df):
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['stddev'] = df['close'].rolling(window=20).std()
    df['upper'] = df['ma20'] + (2 * df['stddev'])
    df['lower'] = df['ma20'] - (2 * df['stddev'])

    latest_close = df['close'].iloc[-1]
    upper = df['upper'].iloc[-1]
    lower = df['lower'].iloc[-1]

    if latest_close > upper:
        return "sell", "Price is above the upper Bollinger Band. Overbought."
    elif latest_close < lower:
        return "buy", "Price is below the lower Bollinger Band. Oversold."
    else:
        return "hold", "Price is within the Bollinger Bands range."


def stochastic_oscillator_strategy(df):
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['%K'] = 100 * ((df['close'] - low14) / (high14 - low14))
    df['%D'] = df['%K'].rolling(3).mean()

    k = df['%K'].iloc[-1]
    d = df['%D'].iloc[-1]

    if k < 20 and d < 20 and k > d:
        return "buy", "Stochastic Oscillator is oversold and turning upward."
    elif k > 80 and d > 80 and k < d:
        return "sell", "Stochastic Oscillator is overbought and turning downward."
    else:
        return "hold", "No strong signal from Stochastic Oscillator."


def aroon_strategy(df):
    period = 25
    df['aroon_up'] = df['high'].rolling(period).apply(
        lambda x: float(x.argmax()) / period * 100, raw=True)
    df['aroon_down'] = df['low'].rolling(period).apply(
        lambda x: float(x.argmin()) / period * 100, raw=True)

    up = df['aroon_up'].iloc[-1]
    down = df['aroon_down'].iloc[-1]

    if up > 70 and down < 30:
        return "buy", "Aroon Up is high and Aroon Down is low. Uptrend signal."
    elif down > 70 and up < 30:
        return "sell", "Aroon Down is high and Aroon Up is low. Downtrend signal."
    else:
        return "hold", "Aroon shows no clear trend."

@app.route("/api/strategy-signals", methods=["GET"])
def strategy_signals():
    symbol = request.args.get("symbol", default="EURUSD")
    timeframe_str = request.args.get("timeframe", default="M15")
    timeframe = TIMEFRAME_MAP.get(timeframe_str.upper(), mt5.TIMEFRAME_M15)

    df = get_candles(symbol, timeframe, 500)
    if df is None or df.empty:
        return jsonify({"error": "Failed to fetch candle data"}), 400

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
