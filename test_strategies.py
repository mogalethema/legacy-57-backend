import pandas as pd
import numpy as np
import time

# === IMPORT ALL STRATEGIES ===
# from app import allStrategies  # Make sure allStrategies contains your 1000 strategies

# === 1. Prepare Initial Test Data ===
np.random.seed(42)
bars = 50  # start with 50 historical candles
df_test = pd.DataFrame({
    'open': np.random.rand(bars)*100,
    'high': np.random.rand(bars)*100,
    'low': np.random.rand(bars)*100,
    'close': np.random.rand(bars)*100,
    'volume': np.random.rand(bars)*1000
})

# === 2. Simulate “Live Candles” ===
new_candles = 20  # number of new candles to simulate
results_over_time = []

for i in range(new_candles):
    # Simulate new candle
    new_open = np.random.rand()*100
    new_high = new_open + np.random.rand()*5
    new_low = new_open - np.random.rand()*5
    new_close = np.random.rand()*100
    new_volume = np.random.rand()*1000

    new_row = pd.DataFrame({
        'open': [new_open],
        'high': [new_high],
        'low': [new_low],
        'close': [new_close],
        'volume': [new_volume]
    })
    
    df_test = pd.concat([df_test, new_row], ignore_index=True)

    # Run all strategies on updated DataFrame
    step_results = []
    for idx, strategy in enumerate(allStrategies, start=1):
        try:
            signal, reason = strategy(df_test)
            step_results.append((i+1, idx, signal, reason))
        except Exception as e:
            step_results.append((i+1, idx, "ERROR", str(e)))
    
    results_over_time.extend(step_results)
    
    # Optional: simulate real-time delay
    # time.sleep(0.1)

# === 3. Convert to DataFrame and Review ===
results_df = pd.DataFrame(results_over_time, columns=["CandleStep", "Strategy", "Signal", "Reason"])

print("=== Last 10 Results ===")
print(results_df.tail(10))

print("\n=== Signal Counts ===")
print(results_df['Signal'].value_counts())

# === 4. Save Results for Full Review ===
results_df.to_csv("strategy_live_simulation.csv", index=False)
print("\nResults saved to strategy_live_simulation.csv")
