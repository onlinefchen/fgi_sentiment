"""Backtest sentiment-based trading strategy"""
from .sentiment import SentimentCalculator


def backtest(
    ticker: str,
    rsi_data: dict[str, float],
    prices: dict[str, float],
    volumes: dict[str, float],
    threshold: int = 30,
    targets: list[int] = None
) -> dict:
    """Run backtest for a single ticker

    Args:
        ticker: Stock ticker symbol
        rsi_data: {date: rsi_value}
        prices: {date: close_price}
        volumes: {date: volume}
        threshold: Sentiment threshold for buy signal
        targets: Target percentages to track

    Returns:
        {
            "ticker": str,
            "signals": [...],
            "summary": {...},
            "date_range": {...}
        }
    """
    if targets is None:
        targets = [5, 10, 20, 30]

    calculator = SentimentCalculator()

    # Prepare data dict for sentiment calculation
    data = {
        "rsi": rsi_data,
        "prices": prices,
        "volumes": volumes
    }

    signals = []
    sorted_dates = sorted(prices.keys())

    for date in sorted_dates:
        if date not in rsi_data or date not in volumes:
            continue

        result = calculator.calculate(data, date)
        sentiment = result["score"]

        if sentiment is None:
            continue

        if sentiment < threshold:
            price = prices[date]

            # Calculate days to reach each target
            days_to_target = {}
            for target in targets:
                target_price = price * (1 + target / 100)
                days = None

                future_dates = [d for d in sorted_dates if d > date]
                for i, future_date in enumerate(future_dates):
                    if prices[future_date] >= target_price:
                        days = i + 1
                        break

                days_to_target[str(target)] = days

            signals.append({
                "date": date,
                "sentiment": sentiment,
                "components": result["components"],
                "price": price,
                "days_to_target": days_to_target
            })

    # Calculate summary
    summary = {}
    for t in targets:
        successes = [s["days_to_target"][str(t)] for s in signals if s["days_to_target"][str(t)] is not None]
        total = len(signals)

        summary[str(t)] = {
            "success_count": len(successes),
            "total": total,
            "success_rate": len(successes) / total if total > 0 else 0,
            "avg_days": sum(successes) / len(successes) if successes else None
        }

    return {
        "ticker": ticker,
        "threshold": threshold,
        "signals": signals,
        "summary": summary,
        "date_range": {
            "from": sorted_dates[0] if sorted_dates else None,
            "to": sorted_dates[-1] if sorted_dates else None
        }
    }
