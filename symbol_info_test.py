import MetaTrader5 as mt5
import pprint

mt5.initialize()

symbol = "EURUSD"  # Change this if you want another symbol
info = mt5.symbol_info(symbol)
if info is None:
    print(f"Symbol {symbol} not found")
else:
    pprint.pprint(info._asdict())

mt5.shutdown()
