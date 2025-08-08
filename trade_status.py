from flask import Flask, request, jsonify
import MetaTrader5 as mt5
import pandas as pd
import datetime

app = Flask(__name__)

# Initialize MT5 connection
def mt5_initialize():
    if not mt5.initialize():
        return False, str(mt5.last_error())
    return True, None

# Shutdown MT5 connection
def mt5_shutdown():
    mt5.shutdown()

# Fetch historical bars (candles) for a symbol and timeframe
def get_bars(symbol, timeframe, n=100):
    timeframe_dict = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }
    tf = timeframe_dict.get(timeframe.upper(), mt5.TIMEFRAME_M15)

    utc_from = datetime.datetime.now() - datetime.timedelta(days=10)
    rates = mt5.copy_rates_from(symbol, tf, utc_from, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Moving Average Crossover
def strategy_ma_crossover(df, fast=9, slow=21):
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()

    if len(df) < slow:
        return "hold"

    if df['ma_fast'].iloc[-2] < df['ma_slow'].iloc[-2] and df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
        return "buy"
    elif df['ma_fast'].iloc[-2] > df['ma_slow'].iloc[-2] and df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1]:
        return "sell"
    else:
        return "hold"

# RSI
def strategy_rsi(df, period=14, oversold=30, overbought=70):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi

    if len(df) < period:
        return "hold"

    last_rsi = df['rsi'].iloc[-1]
    if last_rsi < oversold:
        return "buy"
    elif last_rsi > overbought:
        return "sell"
    else:
        return "hold"

# 🧠 Natural Language Explanation: Moving Average
def explain_ma_signal(signal):
    return {
        "buy": "The fast moving average crossed above the slow one — this could mean a potential bullish trend.",
        "sell": "The fast moving average crossed below the slow one — possibly indicating a bearish trend.",
        "hold": "No clear crossover trend detected by moving averages."
    }.get(signal, "No explanation available.")

# 🧠 Natural Language Explanation: RSI
def explain_rsi_signal(signal):
    return {
        "buy": "RSI shows oversold conditions — this might be a good time to buy.",
        "sell": "RSI indicates the asset is overbought — might be time to sell.",
        "hold": "RSI is neutral — no strong buy or sell signal."
    }.get(signal, "No explanation available.")

# API: Account status
@app.route('/api/trade-status', methods=['GET'])
def trade_status():
    ok, err = mt5_initialize()
    if not ok:
        return jsonify({"status": "failed", "error": err}), 500

    account_info = mt5.account_info()
    positions = mt5.positions_get()
    mt5_shutdown()

    if account_info is None:
        return jsonify({"status": "failed", "error": "No account info"}), 500

    return jsonify({
        "status": "connected",
        "login": account_info.login,
        "balance": account_info.balance,
        "equity": account_info.equity,
        "open_positions": len(positions) if positions else 0
    })

# API: Place trade
@app.route('/api/place-trade', methods=['POST'])
def place_trade():
    data = request.json
    symbol = data.get("symbol", "EURUSD")
    volume = data.get("volume", 0.01)
    order_type = data.get("type", "buy")

    ok, err = mt5_initialize()
    if not ok:
        return jsonify({"status": "failed", "error": err}), 500

    if not mt5.symbol_select(symbol, True):
        mt5_shutdown()
        return jsonify({"status": "failed", "error": f"Failed to select symbol {symbol}"}), 400

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info or not symbol_info.visible:
        mt5_shutdown()
        return jsonify({"status": "failed", "error": "Invalid or invisible symbol"}), 400

    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type == "buy" else tick.bid

    action_map = {
        "buy": mt5.ORDER_TYPE_BUY,
        "sell": mt5.ORDER_TYPE_SELL,
        "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
        "sell_stop": mt5.ORDER_TYPE_SELL_STOP
    }
    action = action_map.get(order_type.lower(), mt5.ORDER_TYPE_BUY)

    request_trade = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": action,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "Placed via Flask API",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": symbol_info.filling_mode,
    }

    result = mt5.order_send(request_trade)
    mt5_shutdown()

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return jsonify({
            "status": "failed",
            "retcode": result.retcode,
            "comment": result.comment
        }), 400

    return jsonify({
        "status": "success",
        "order": {
            "ticket": result.order,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price
        }
    })

# API: Get strategy signals + explanations
@app.route('/api/strategy-signals', methods=['GET'])
def strategy_signals():
    symbol = request.args.get('symbol', 'EURUSD')
    timeframe = request.args.get('timeframe', 'M15')

    ok, err = mt5_initialize()
    if not ok:
        return jsonify({"status": "failed", "error": err}), 500

    bars = get_bars(symbol, timeframe, n=100)
    mt5_shutdown()

    if bars is None or bars.empty:
        return jsonify({"status": "failed", "error": "Failed to get bars"}), 500

    ma_signal = strategy_ma_crossover(bars)
    rsi_signal = strategy_rsi(bars)

    return jsonify({
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "signals": {
            "moving_average_crossover": ma_signal,
            "rsi": rsi_signal
        },
        "explanations": {
            "moving_average_crossover": explain_ma_signal(ma_signal),
            "rsi": explain_rsi_signal(rsi_signal)
        }
    })

if __name__ == '__main__':
    print("🚀 Flask server running...")
    app.run(port=5000, debug=True)
