"""Polygon API data fetcher"""
import os
import time
import requests
from datetime import datetime, timedelta

API_KEY = os.environ.get("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"


def _request_with_retry(url: str, params: dict, retries: int = 3) -> dict:
    """Make request with retry logic"""
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            return resp.json()
        except Exception as e:
            if i < retries - 1:
                time.sleep(1)
            else:
                raise e
    return {}


def fetch_rsi(ticker: str, days: int = 730) -> dict[str, float]:
    """Fetch historical RSI data, return {date: value}"""
    end = datetime.now()
    start = end - timedelta(days=days)

    url = f"{BASE_URL}/v1/indicators/rsi/{ticker}"
    params = {
        "timespan": "day",
        "adjusted": "true",
        "window": 14,
        "series_type": "close",
        "order": "asc",
        "limit": 5000,
        "timestamp.gte": start.strftime("%Y-%m-%d"),
        "timestamp.lte": end.strftime("%Y-%m-%d"),
        "apiKey": API_KEY
    }

    data = _request_with_retry(url, params)

    if "results" not in data or "values" not in data["results"]:
        return {}

    result = {}
    for r in data["results"]["values"]:
        date = datetime.fromtimestamp(r["timestamp"] / 1000).strftime("%Y-%m-%d")
        result[date] = r["value"]
    return result


def fetch_price_volume(ticker: str, days: int = 730) -> tuple[dict[str, float], dict[str, float]]:
    """Fetch historical price and volume data"""
    end = datetime.now()
    start = end - timedelta(days=days)

    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": API_KEY
    }

    data = _request_with_retry(url, params)

    if "results" not in data:
        return {}, {}

    prices = {}
    volumes = {}
    for r in data["results"]:
        date = datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
        prices[date] = r["c"]
        volumes[date] = r["v"]

    return prices, volumes


def fetch_put_call_ratio(ticker: str) -> float | None:
    """Fetch current Put/Call Ratio from options snapshot

    Returns:
        PCR value (puts/calls), higher = more bearish
        None if data unavailable
    """
    url = f"{BASE_URL}/v3/snapshot/options/{ticker}"
    params = {"apiKey": API_KEY}

    data = _request_with_retry(url, params)

    if "results" not in data:
        return None

    puts = 0
    calls = 0

    for contract in data["results"]:
        contract_type = contract.get("details", {}).get("contract_type")
        open_interest = contract.get("open_interest", 0)

        if contract_type == "put":
            puts += open_interest
        elif contract_type == "call":
            calls += open_interest

    if calls == 0:
        return None

    return puts / calls


def fetch_news_sentiment(ticker: str, limit: int = 10) -> float | None:
    """Fetch news sentiment score

    Returns:
        Sentiment score 0-100 (0 = very negative, 100 = very positive)
        None if no news available
    """
    url = f"{BASE_URL}/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": limit,
        "apiKey": API_KEY
    }

    data = _request_with_retry(url, params)

    if "results" not in data or not data["results"]:
        return None

    sentiment_scores = []

    for article in data["results"]:
        insights = article.get("insights", [])
        for insight in insights:
            if insight.get("ticker") == ticker:
                sentiment = insight.get("sentiment")
                if sentiment == "positive":
                    sentiment_scores.append(80)
                elif sentiment == "negative":
                    sentiment_scores.append(20)
                elif sentiment == "neutral":
                    sentiment_scores.append(50)

    if not sentiment_scores:
        return None

    return sum(sentiment_scores) / len(sentiment_scores)


def fetch_current(ticker: str, include_options: bool = True, include_news: bool = True) -> dict | None:
    """Fetch current day data for a ticker

    Args:
        ticker: Stock ticker symbol
        include_options: Whether to fetch Put/Call Ratio (slower)
        include_news: Whether to fetch news sentiment (slower)
    """
    rsi_data = fetch_rsi(ticker, days=30)
    prices, volumes = fetch_price_volume(ticker, days=30)

    if not rsi_data or not prices:
        return None

    # Find common dates that have both RSI and price data
    common_dates = set(rsi_data.keys()) & set(prices.keys())
    if not common_dates:
        return None

    latest_date = max(common_dates)

    result = {
        "ticker": ticker,
        "date": latest_date,
        "price": prices.get(latest_date),
        "rsi": rsi_data.get(latest_date),
        "volume": volumes.get(latest_date),
        "pcr": None,
        "news_sentiment": None
    }

    # Fetch optional data
    if include_options:
        result["pcr"] = fetch_put_call_ratio(ticker)

    if include_news:
        result["news_sentiment"] = fetch_news_sentiment(ticker)

    return result
