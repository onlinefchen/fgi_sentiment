#!/usr/bin/env python3
"""FGI Sentiment Analysis CLI - BTC & Stocks"""
import json
import os
import click
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "config"


@click.group()
def cli():
    """FGI Sentiment Analysis Tool

    Analyze Fear & Greed for BTC and individual stocks.

    \b
    Commands:
      btc     - BTC Fear & Greed Index
      stock   - Individual stock sentiment
    """
    pass


# =============================================================================
# BTC Commands
# =============================================================================

@cli.group()
def btc():
    """BTC Fear & Greed Index commands"""
    pass


@btc.command("status")
def btc_status():
    """Show current BTC FGI and price."""
    import pandas as pd
    from src.crypto.fetcher import fetch_fgi_data

    data = fetch_fgi_data()
    if not data:
        click.echo("Failed to fetch data")
        return

    latest = data[-1]
    click.echo(f"BTC FGI: {latest['fgi_value']} ({latest['fgi_class']}) | ${latest['btc_price']:,.0f} | {latest['date']}")


@btc.command("update")
def btc_update():
    """Fetch latest BTC FGI data from CoinGlass."""
    import pandas as pd
    from src.crypto.fetcher import fetch_fgi_data

    DATA_DIR.mkdir(exist_ok=True)
    merged_path = DATA_DIR / "btc_merged.csv"

    click.echo("Fetching BTC FGI from CoinGlass...")
    data = fetch_fgi_data()
    click.echo(f"  Fetched {len(data)} records")

    df = pd.DataFrame(data)
    df.to_csv(merged_path, index=False)
    click.echo(f"  Saved to {merged_path}")


@btc.command("backtest")
@click.option("-t", "--threshold", default=15, help="FGI threshold for buy signal")
@click.option("--targets", default="50,100", help="Target percentages")
def btc_backtest(threshold, targets):
    """Run BTC FGI backtest."""
    import pandas as pd
    from tabulate import tabulate
    from src.crypto.backtester import backtest

    merged_path = DATA_DIR / "btc_merged.csv"
    if not merged_path.exists():
        click.echo("No data. Run 'python cli.py btc update' first.")
        return

    df = pd.read_csv(merged_path)
    data = df.to_dict("records")
    target_list = [int(t.strip()) for t in targets.split(",")]

    result = backtest(data, threshold, target_list)

    click.echo(f"\nFound {len(result['signals'])} buy signals (FGI < {threshold})")

    headers = ["Date", "FGI", "Price"] + [f"+{t}%" for t in target_list]
    rows = []
    for s in result["signals"][-20:]:  # Last 20
        row = [s["date"], s["fgi"], f"${s['price']:,.0f}"]
        for t in target_list:
            days = s["days_to_target"].get(str(t))
            row.append(f"{days}d" if days else "--")
        rows.append(row)

    click.echo(tabulate(rows, headers=headers, tablefmt="simple"))

    click.echo("\nSummary:")
    summary_rows = []
    for t in target_list:
        s = result["summary"][str(t)]
        rate = f"{s['success_rate']*100:.0f}% ({s['success_count']}/{s['total']})"
        avg = f"{s['avg_days']:.0f}d" if s['avg_days'] else "--"
        summary_rows.append([f"+{t}%", avg, rate])
    click.echo(tabulate(summary_rows, headers=["Target", "Avg Days", "Success Rate"], tablefmt="simple"))


@btc.command("alert")
@click.option("-t", "--threshold", default=15, help="FGI threshold for alert")
@click.option("--notify", is_flag=True, help="Send notifications")
def btc_alert(threshold, notify):
    """Check BTC FGI and alert if below threshold."""
    from src.crypto.fetcher import fetch_fgi_data
    from src.notifier import MultiNotifier

    data = fetch_fgi_data()
    if not data:
        click.echo("Failed to fetch data")
        return

    latest = data[-1]
    fgi = latest["fgi_value"]
    price = latest["btc_price"]
    date = latest["date"]

    click.echo(f"BTC FGI: {fgi} ({latest['fgi_class']}) | ${price:,.0f} | {date}")

    if fgi < threshold:
        click.echo(f"ALERT: FGI ({fgi}) is below threshold ({threshold})!")

        if notify:
            notifier = MultiNotifier()
            message = f"""BTC Fear & Greed Index Alert!

Current FGI: {fgi} ({latest['fgi_class']})
BTC Price: ${price:,.0f}
Date: {date}

This is a good time to consider buying BTC according to your FGI < {threshold} strategy.

---
Sent by GitHub Actions"""
            notifier.send(f"BTC FGI Alert - {latest['fgi_class']}!", message)
    else:
        click.echo(f"FGI ({fgi}) is above threshold ({threshold}). No alert.")


# =============================================================================
# Stock Commands
# =============================================================================

@cli.group()
def stock():
    """Individual stock sentiment commands"""
    pass


def load_watchlist() -> list[str]:
    path = CONFIG_DIR / "stock_watchlist.json"
    if path.exists():
        with open(path) as f:
            return json.load(f).get("stocks", [])
    return []


def save_watchlist(stocks: list[str]):
    CONFIG_DIR.mkdir(exist_ok=True)
    path = CONFIG_DIR / "stock_watchlist.json"
    with open(path, "w") as f:
        json.dump({"stocks": stocks}, f, indent=2)


@stock.command("status")
@click.argument("ticker", required=False)
def stock_status(ticker):
    """Show current sentiment for a stock or watchlist."""
    from src.stock.fetcher import fetch_current
    from src.stock.sentiment import SentimentCalculator

    calculator = SentimentCalculator()

    if ticker:
        tickers = [ticker.upper()]
    else:
        tickers = load_watchlist()
        if not tickers:
            click.echo("Watchlist empty. Add with: python cli.py stock add AAPL")
            return

    import time
    for i, t in enumerate(tickers):
        if i > 0:
            time.sleep(1.5)

        data = fetch_current(t)
        if not data:
            click.echo(f"{t}: Failed to fetch data")
            continue

        full_data = {
            "rsi": {data["date"]: data["rsi"]},
            "volumes": {data["date"]: data["volume"]},
            "pcr": data.get("pcr"),
            "news_sentiment": data.get("news_sentiment")
        }
        result = calculator.calculate(full_data, data["date"])

        score = result["score"]
        components = result["components"]

        status_text = "Extreme Fear" if score and score < 30 else "Normal"
        parts = [f"{t}: Score {score} ({status_text})"]

        if "rsi" in components:
            parts.append(f"RSI:{components['rsi']:.0f}")
        if "pcr" in components:
            parts.append(f"PCR:{components['pcr']:.0f}")
        if "news" in components:
            parts.append(f"News:{components['news']:.0f}")
        parts.append(f"${data['price']:.2f}")

        click.echo(" | ".join(parts))


@stock.command("add")
@click.argument("ticker")
def stock_add(ticker):
    """Add a stock to watchlist."""
    stocks = load_watchlist()
    ticker = ticker.upper()

    if ticker in stocks:
        click.echo(f"{ticker} already in watchlist")
        return

    stocks.append(ticker)
    save_watchlist(stocks)
    click.echo(f"Added {ticker}")


@stock.command("remove")
@click.argument("ticker")
def stock_remove(ticker):
    """Remove a stock from watchlist."""
    stocks = load_watchlist()
    ticker = ticker.upper()

    if ticker not in stocks:
        click.echo(f"{ticker} not in watchlist")
        return

    stocks.remove(ticker)
    save_watchlist(stocks)
    click.echo(f"Removed {ticker}")


@stock.command("list")
def stock_list():
    """List watchlist stocks."""
    stocks = load_watchlist()
    if not stocks:
        click.echo("Watchlist empty")
        return
    click.echo("Watchlist: " + ", ".join(stocks))


@stock.command("backtest")
@click.argument("ticker")
@click.option("-t", "--threshold", default=30, help="Sentiment threshold")
@click.option("--targets", default="5,10,20,30", help="Target percentages")
@click.option("--days", default=730, help="Days of history")
def stock_backtest(ticker, threshold, targets, days):
    """Run backtest for a stock."""
    from tabulate import tabulate
    from src.stock.fetcher import fetch_rsi, fetch_price_volume
    from src.stock.backtester import backtest

    ticker = ticker.upper()
    target_list = [int(t.strip()) for t in targets.split(",")]

    click.echo(f"Fetching {ticker} data ({days} days)...")

    rsi_data = fetch_rsi(ticker, days=days)
    prices, volumes = fetch_price_volume(ticker, days=days)

    if not rsi_data or not prices:
        click.echo("Failed to fetch data")
        return

    click.echo(f"  RSI: {len(rsi_data)} | Prices: {len(prices)}")

    result = backtest(ticker, rsi_data, prices, volumes, threshold, target_list)

    click.echo(f"\nFound {len(result['signals'])} buy signals (Sentiment < {threshold})")

    headers = ["Date", "Score", "RSI", "Vol", "Price"] + [f"+{t}%" for t in target_list]
    rows = []
    for s in result["signals"]:
        rsi = s["components"].get("rsi", "--")
        vol = s["components"].get("volume", "--")
        row = [
            s["date"],
            s["sentiment"],
            f"{rsi:.1f}" if isinstance(rsi, float) else rsi,
            int(vol) if isinstance(vol, float) else vol,
            f"${s['price']:.2f}"
        ]
        for t in target_list:
            days_val = s["days_to_target"].get(str(t))
            row.append(f"{days_val}d" if days_val else "--")
        rows.append(row)

    click.echo(tabulate(rows, headers=headers, tablefmt="simple"))

    click.echo("\nSummary:")
    summary_rows = []
    for t in target_list:
        s = result["summary"][str(t)]
        rate = f"{s['success_rate']*100:.0f}% ({s['success_count']}/{s['total']})"
        avg = f"{s['avg_days']:.0f}d" if s['avg_days'] else "--"
        summary_rows.append([f"+{t}%", avg, rate])
    click.echo(tabulate(summary_rows, headers=["Target", "Avg Days", "Success Rate"], tablefmt="simple"))


@stock.command("alert")
@click.option("-t", "--threshold", default=None, type=int, help="Override threshold")
@click.option("--notify", is_flag=True, help="Send notifications")
def stock_alert(threshold, notify):
    """Check watchlist stocks and alert if sentiment is low."""
    from src.stock.fetcher import fetch_current
    from src.stock.sentiment import SentimentCalculator
    from src.notifier import MultiNotifier

    calculator = SentimentCalculator()
    threshold = threshold or calculator.alert_threshold

    stocks = load_watchlist()
    if not stocks:
        click.echo("Watchlist empty")
        return

    alerts = []
    import time

    for i, ticker in enumerate(stocks):
        if i > 0:
            time.sleep(1.5)

        data = fetch_current(ticker)
        if not data:
            click.echo(f"{ticker}: Failed to fetch")
            continue

        full_data = {
            "rsi": {data["date"]: data["rsi"]},
            "volumes": {data["date"]: data["volume"]},
            "pcr": data.get("pcr"),
            "news_sentiment": data.get("news_sentiment")
        }
        result = calculator.calculate(full_data, data["date"])
        score = result["score"]
        components = result["components"]

        if score and score < threshold:
            alerts.append({
                "ticker": ticker,
                "score": score,
                "components": components,
                "price": data["price"]
            })

        parts = [f"{ticker}: Score {score}"]
        if "rsi" in components:
            parts.append(f"RSI:{components['rsi']:.0f}")
        parts.append(f"${data['price']:.2f}")
        click.echo(" | ".join(parts))

    if alerts:
        click.echo(f"\nALERT: {len(alerts)} stock(s) below threshold ({threshold})!")

        if notify:
            notifier = MultiNotifier()
            lines = [f"{len(alerts)} stock(s) showing extreme fear:"]
            for a in alerts:
                lines.append(f"  {a['ticker']}: Score {a['score']} @ ${a['price']:.2f}")
            lines.append("\n---\nSent by GitHub Actions")
            notifier.send("Stock Sentiment Alert!", "\n".join(lines))
    else:
        click.echo(f"\nNo alerts. All above threshold ({threshold}).")


if __name__ == "__main__":
    cli()
