import requests

def test_strategy_signals():
    url = "http://127.0.0.1:5000/api/strategy-signals"
    params = {
        "symbol": "EURUSD",
        "timeframe": "M15"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        print("Response status:", response.status_code)
        print("Response JSON:")
        print(data)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_strategy_signals()
