import json
import os
import requests
from typing import List, Dict
from datetime import datetime

COINGLASS_API_URL = "https://open-api-v4.coinglass.com/api/index/fear-greed-history"
COINGECKO_BTC_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"


def fetch_btc_price() -> float:
    """Fetch real-time BTC price from CoinGecko API."""
    response = requests.get(COINGECKO_BTC_URL, timeout=10)
    response.raise_for_status()
    return float(response.json()["bitcoin"]["usd"])


def fetch_fgi_data() -> List[Dict]:
    """
    Fetch FGI and BTC price data from CoinGlass API.

    Requires COINGLASS_API_KEY environment variable.

    Returns:
        List of dicts with keys: date, fgi_value, fgi_class, btc_price
    """
    api_key = os.environ.get("COINGLASS_API_KEY")
    if not api_key:
        raise ValueError("COINGLASS_API_KEY environment variable not set")

    headers = {"CG-API-KEY": api_key}
    response = requests.get(COINGLASS_API_URL, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    if data["code"] != "0":
        raise ValueError(f"CoinGlass API error: {data.get('msg', 'Unknown error')}")

    times = data["data"]["time_list"]
    values = data["data"]["data_list"]
    prices = data["data"]["price_list"]

    result = []
    for i in range(len(times)):
        ts = times[i] / 1000 if times[i] > 1e12 else times[i]
        fgi = int(values[i])

        # Classify FGI value
        if fgi <= 25:
            fgi_class = "Extreme Fear"
        elif fgi <= 45:
            fgi_class = "Fear"
        elif fgi <= 55:
            fgi_class = "Neutral"
        elif fgi <= 75:
            fgi_class = "Greed"
        else:
            fgi_class = "Extreme Greed"

        result.append({
            "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
            "fgi_value": fgi,
            "fgi_class": fgi_class,
            "btc_price": prices[i]
        })

    # Sort by date ascending
    result.sort(key=lambda x: x["date"])
    return result


def save_json(data: List[Dict], filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> List[Dict]:
    """Load data from JSON file. Returns empty list if file doesn't exist."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
