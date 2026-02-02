# src/backtester.py
from typing import List, Dict, Optional

def backtest(
    data: List[Dict],
    fgi_threshold: int = 25,
    targets: List[int] = None,
    date_from: str = None,
    date_to: str = None
) -> Dict:
    """
    Backtest FGI-based buy strategy.

    Args:
        data: List of {date, fgi_value, btc_price}
        fgi_threshold: Buy when FGI < threshold
        targets: List of target percentages (e.g., [10, 20, 30])
        date_from: Start date filter (YYYY-MM-DD)
        date_to: End date filter (YYYY-MM-DD)

    Returns:
        Dict with signals and summary statistics
    """
    if targets is None:
        targets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Filter by date range
    filtered_data = data
    if date_from:
        filtered_data = [d for d in filtered_data if d["date"] >= date_from]
    if date_to:
        filtered_data = [d for d in filtered_data if d["date"] <= date_to]

    signals = []

    for i, row in enumerate(filtered_data):
        if row["fgi_value"] <= fgi_threshold:
            signal = {
                "date": row["date"],
                "fgi": row["fgi_value"],
                "price": row["btc_price"],
                "days_to_target": {}
            }

            # Look forward to find days to reach each target
            buy_price = row["btc_price"]
            for target in targets:
                target_price = buy_price * (1 + target / 100)
                days = None

                for j in range(i + 1, len(filtered_data)):
                    if filtered_data[j]["btc_price"] >= target_price:
                        days = j - i
                        break

                signal["days_to_target"][str(target)] = days

            signals.append(signal)

    # Calculate summary statistics
    summary = {}
    for target in targets:
        target_key = str(target)
        success_signals = [(s, s["days_to_target"][target_key]) for s in signals if s["days_to_target"][target_key] is not None]

        if success_signals:
            days_list = [d for _, d in success_signals]
            min_signal = min(success_signals, key=lambda x: x[1])
            max_signal = max(success_signals, key=lambda x: x[1])
            summary[target_key] = {
                "avg_days": round(sum(days_list) / len(days_list), 1),
                "min_days": min(days_list),
                "max_days": max(days_list),
                "min_date": min_signal[0]["date"],
                "max_date": max_signal[0]["date"],
                "success_count": len(days_list),
                "total": len(signals),
                "success_rate": round(len(days_list) / len(signals), 2) if signals else 0
            }
        else:
            summary[target_key] = {
                "avg_days": None,
                "min_days": None,
                "max_days": None,
                "min_date": None,
                "max_date": None,
                "success_count": 0,
                "total": len(signals),
                "success_rate": 0
            }

    return {
        "threshold": fgi_threshold,
        "targets": targets,
        "date_range": {
            "from": date_from or (filtered_data[0]["date"] if filtered_data else None),
            "to": date_to or (filtered_data[-1]["date"] if filtered_data else None)
        },
        "signals": signals,
        "summary": summary
    }
