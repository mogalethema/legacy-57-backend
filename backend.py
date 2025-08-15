# backend.py (UPGRADED)
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# IMPORTANT: MetaTrader5 package must be installed and MT5 must be running.
# pip install MetaTrader5 fastapi uvicorn pandas numpy
import MetaTrader5 as mt5
import pandas as pd

# ---------- CONFIG ----------
DEFAULT_SYMBOL = "EURUSD"
# MetaTrader5 timeframe constants (e.g., mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5)
DEFAULT_TIMEFRAME = mt5.TIMEFRAME_M1
HISTORY_LIMIT = 500
POLL_INTERVAL = 1.0  # seconds for checking MT5 for new candles
HEARTBEAT_INTERVAL = 15.0  # seconds - heartbeat to WS clients
LOG_LEVEL = logging.INFO
# ----------------------------

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory store of the latest candles per (symbol, timeframe) to avoid hitting MT5 too often
# key -> (symbol, timeframe) as string "EURUSD|M1"
CANDLES_STORE: Dict[str, List[Dict]] = {}

# Connected clients: maps WebSocket -> subscription tuple (symbol, timeframe)
clients_lock = asyncio.Lock()
clients: Dict[WebSocket, Tuple[str, int]] = {}

# Strategy queue: push events here and worker(s) will compute signals asynchronously
strategy_queue: "asyncio.Queue[Dict]" = asyncio.Queue()

# -----------------------
# Helper functions
# -----------------------
def key_for(symbol: str, timeframe: int) -> str:
    return f"{symbol.upper()}|{timeframe}"

def mt5_init():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed, error = {mt5.last_error()}")
    logging.info("MT5 initialized.")

def fetch_history(symbol: str = DEFAULT_SYMBOL, timeframe: int = DEFAULT_TIMEFRAME, n: int = HISTORY_LIMIT):
    """Return last n candles from MT5 as list of dicts (time iso, open, high, low, close, volume)."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if rates is None or len(rates) == 0:
            return []
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        candles: List[Dict] = []
        for _, row in df.iterrows():
            candles.append({
                "time": row['time'].isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row.get('tick_volume', 0))
            })
        return candles
    except Exception as e:
        logging.exception("fetch_history error")
        return []

# Simple indicators (kept small & safe)
def ma(series: pd.Series, period: int = 20) -> Optional[float]:
    if len(series) < period:
        return None
    return float(series.rolling(period).mean().iloc[-1])

def rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])

def bollinger(series: pd.Series, period: int = 20, stds: float = 2.0):
    if len(series) < period:
        return None, None, None
    m = series.rolling(period).mean()
    s = series.rolling(period).std()
    upper = m + stds * s
    lower = m - stds * s
    return float(upper.iloc[-1]), float(lower.iloc[-1]), float(m.iloc[-1])

# -----------------------
# Broadcasting helpers
# -----------------------
async def broadcast_to_subscribers(message: dict, symbol: str, timeframe: int):
    """Send 'message' to all clients subscribed to symbol/timeframe."""
    text = json.dumps(message)
    to_remove = []
    async with clients_lock:
        for ws, (sym, tf) in list(clients.items()):
            if sym.upper() == symbol.upper() and tf == timeframe:
                try:
                    await ws.send_text(text)
                except Exception as e:
                    logging.warning("Broadcast error, scheduling removal: %s", e)
                    to_remove.append(ws)
        for ws in to_remove:
            clients.pop(ws, None)
            try:
                await ws.close()
            except Exception:
                pass

# -----------------------
# Candle poller (per symbol/timeframe)
# -----------------------
async def candle_poller(symbol: str = DEFAULT_SYMBOL, timeframe: int = DEFAULT_TIMEFRAME, poll_interval: float = POLL_INTERVAL):
    """Continuously poll MT5 for the given symbol/timeframe and broadcast new candles."""
    key = key_for(symbol, timeframe)
    logging.info("Starting poller for %s", key)
    CANDLES_STORE[key] = fetch_history(symbol, timeframe, HISTORY_LIMIT)
    last_time = CANDLES_STORE[key][-1]["time"] if CANDLES_STORE[key] else None

    while True:
        try:
            latest = fetch_history(symbol, timeframe, HISTORY_LIMIT)
            if not latest:
                await asyncio.sleep(poll_interval)
                continue

            # If new last candle found, update store and broadcast
            if last_time is None:
                CANDLES_STORE[key] = latest
                last_time = latest[-1]["time"]
            elif latest[-1]["time"] != last_time:
                CANDLES_STORE[key] = latest
                last_time = latest[-1]["time"]

                # Prepare indicators for the newest candle and a summary payload
                df = pd.DataFrame(latest)
                df['close'] = pd.to_numeric(df['close'])
                ma20 = ma(df['close'], 20)
                rsi14 = rsi(df['close'], 14)
                bb_upper, bb_lower, bb_mid = bollinger(df['close'], 20)

                newest = latest[-1]
                payload = {
                    "type": "candle",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": newest,
                    "indicators": {
                        "ma20": ma20,
                        "rsi14": rsi14,
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower,
                        "bb_mid": bb_mid
                    },
                    # signals left empty — strategy workers will add signals by pushing into strategy_queue
                    "signals": []
                }

                # Enqueue event for strategy workers to compute signals (non-blocking)
                await strategy_queue.put({"event": "new_candle", "symbol": symbol, "timeframe": timeframe, "candles": latest})

                # Broadcast the candle now (fast) — strategies will push signal messages separately
                await broadcast_to_subscribers(payload, symbol, timeframe)

            await asyncio.sleep(poll_interval)
        except Exception as e:
            logging.exception("candle_poller error")
            await asyncio.sleep(1.0)

# -----------------------
# Strategy worker(s)
# -----------------------
async def demo_strategy_processor():
    """
    Example strategy worker. In production:
      - Replace with real strategy loader that imports user strategies or strategy modules.
      - Or use Redis/Celery to scale many workers.
    This worker consumes candle events from strategy_queue and, for each new candle,
    computes demo signals and broadcasts them as small signal messages.
    """
    logging.info("Strategy worker started (demo).")
    while True:
        try:
            item = await strategy_queue.get()
            if item is None:
                await asyncio.sleep(0.1)
                continue

            # Only process new candle events here (demo)
            if item.get("event") == "new_candle":
                symbol = item["symbol"]
                timeframe = item["timeframe"]
                candles = item["candles"]
                # compute a simple demo MA crossover signal (safe & fast)
                df = pd.DataFrame(candles)
                df['close'] = pd.to_numeric(df['close'])
                if len(df) >= 22:
                    ma20 = ma(df['close'], 20)
                    last_close = float(df['close'].iloc[-1])
                    prev_close = float(df['close'].iloc[-2])
                    signals = []
                    if ma20 is not None:
                        if prev_close < ma20 and last_close > ma20:
                            signals.append({"strategy_id": "demo-ma-cross", "action": "BUY", "confidence": 0.6, "reason": "price crossed above MA20"})
                        elif prev_close > ma20 and last_close < ma20:
                            signals.append({"strategy_id": "demo-ma-cross", "action": "SELL", "confidence": 0.6, "reason": "price crossed below MA20"})
                    # broadcast signals (small payload)
                    if signals:
                        message = {
                            "type": "signals",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "time": candles[-1]["time"],
                            "signals": signals
                        }
                        await broadcast_to_subscribers(message, symbol, timeframe)

            strategy_queue.task_done()
        except Exception:
            logging.exception("demo_strategy_processor error")
            await asyncio.sleep(0.1)

# -----------------------
# WebSocket endpoint
# -----------------------
@app.websocket("/ws/chart")
async def websocket_endpoint(websocket: WebSocket, symbol: str = Query(DEFAULT_SYMBOL), tf: int = Query(DEFAULT_TIMEFRAME)):
    """
    Connect with: wss://<ngrok-host>/ws/chart?symbol=EURUSD&tf=512   (example tf value is mt5.TIMEFRAME_M1)
    Note: client should use the same integer values for timeframe as mt5 constants.
    """
    await websocket.accept()
    sub_symbol = symbol.upper()
    sub_tf = int(tf)
    logging.info("WS connect %s %s", sub_symbol, sub_tf)

    # Register client
    async with clients_lock:
        clients[websocket] = (sub_symbol, sub_tf)

    try:
        # Send initial history snapshot (fast)
        key = key_for(sub_symbol, sub_tf)
        if key not in CANDLES_STORE:
            # warm-up: fetch history once for this symbol/timeframe
            CANDLES_STORE[key] = fetch_history(sub_symbol, sub_tf, HISTORY_LIMIT)
        await websocket.send_text(json.dumps({
            "type": "history_snapshot",
            "symbol": sub_symbol,
            "timeframe": sub_tf,
            "candles": CANDLES_STORE.get(key, [])
        }))

        # Keep connection alive; server pushes updates autonomously.
        # Accept control messages (optional) from client (e.g., client can request symbol switch)
        while True:
            try:
                # wait for client message or timeout for heartbeat
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=HEARTBEAT_INTERVAL * 2)
                # optional: handle control messages like {"cmd":"subscribe","symbol":"GBPUSD","tf":512}
                try:
                    parsed = json.loads(msg)
                    if isinstance(parsed, dict) and parsed.get("cmd") == "subscribe":
                        new_sym = parsed.get("symbol", sub_symbol).upper()
                        new_tf = int(parsed.get("tf", sub_tf))
                        async with clients_lock:
                            clients[websocket] = (new_sym, new_tf)
                        # send new history snapshot
                        key2 = key_for(new_sym, new_tf)
                        if key2 not in CANDLES_STORE:
                            CANDLES_STORE[key2] = fetch_history(new_sym, new_tf, HISTORY_LIMIT)
                        await websocket.send_text(json.dumps({
                            "type": "history_snapshot",
                            "symbol": new_sym,
                            "timeframe": new_tf,
                            "candles": CANDLES_STORE.get(key2, [])
                        }))
                except json.JSONDecodeError:
                    # ignore plain text pings
                    await websocket.send_text(json.dumps({"type": "pong", "received": msg}))
            except asyncio.TimeoutError:
                # send heartbeat to client to keep connection alive
                try:
                    await websocket.send_text(json.dumps({"type": "heartbeat", "time": datetime.now(timezone.utc).isoformat()}))
                except Exception:
                    logging.info("Websocket heartbeat failed -> disconnecting")
                    raise

    except WebSocketDisconnect:
        logging.info("Websocket disconnected: %s", websocket.client)
    except Exception:
        logging.exception("WS error")
    finally:
        async with clients_lock:
            if websocket in clients:
                clients.pop(websocket, None)
            try:
                await websocket.close()
            except Exception:
                pass

# -----------------------
# HTTP endpoints
# -----------------------
@app.get("/history")
async def get_history(symbol: str = Query(DEFAULT_SYMBOL), tf: int = Query(DEFAULT_TIMEFRAME), limit: int = Query(HISTORY_LIMIT)):
    limit = min(max(50, limit), 2000)
    candles = fetch_history(symbol, tf, limit)
    return {"symbol": symbol.upper(), "timeframe": tf, "candles": candles}

# -----------------------
# Startup: initialize MT5 and start pollers & workers
# -----------------------
@app.on_event("startup")
async def startup_event():
    logging.info("Starting backend - initializing MT5...")
    mt5_init()

    # Start poller for default symbol/timeframe immediately
    asyncio.create_task(candle_poller(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, POLL_INTERVAL))

    # Optionally: if you want additional default pairs, start pollers here (e.g., GBPUSD)
    # asyncio.create_task(candle_poller("GBPUSD", DEFAULT_TIMEFRAME, POLL_INTERVAL))

    # Start a few strategy workers (demo). For scale, increase number or move to external worker system.
    for _ in range(2):  # two workers for demo; change to more workers later if needed
        asyncio.create_task(demo_strategy_processor())

    logging.info("Background pollers & strategy workers started.")

# -----------------------
# Shutdown: cleanup MT5
# -----------------------
@app.on_event("shutdown")
def shutdown_event():
    try:
        mt5.shutdown()
        logging.info("MT5 shutdown complete.")
    except Exception:
        logging.exception("Error shutting down MT5.")

# -----------------------
# CLI Run
# -----------------------
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
