#!/usr/bin/env python3
"""FGI Sentiment Analysis CLI - BTC & Stocks"""
import json
import os
import click
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "config"


@click.group()
def cli():
    """FGI Sentiment Analysis Tool

    Analyze Fear & Greed for BTC and individual stocks.

    \b
    Quick Reference (copy & paste):
    ════════════════════════════════════════════════
    # BTC
    python cli.py btc status
    python cli.py btc update
    python cli.py btc backtest -t 15 --targets 50,100
    python cli.py btc alert -t 20 --notify
    \b
    # Stock - 查看
    python cli.py stock status
    python cli.py stock status AAPL
    python cli.py stock status AAPL -v
    \b
    # Stock - Watchlist
    python cli.py stock list
    python cli.py stock add TSLA
    python cli.py stock remove TSLA
    \b
    # Stock - 回测
    python cli.py stock backtest NVDA
    python cli.py stock backtest NVDA -t 30
    python cli.py stock backtest NVDA --days 365
    python cli.py stock backtest NVDA --from 2024-01-01
    \b
    # Stock - 缓存
    python cli.py stock cache status
    python cli.py stock cache clear
    python cli.py stock cache clear AAPL
    \b
    # Stock - 警报
    python cli.py stock alert -t 25 --notify
    ════════════════════════════════════════════════
    """
    pass


# =============================================================================
# BTC Commands
# =============================================================================

@cli.group()
def btc():
    """BTC Fear & Greed Index commands

    \b
    Quick Reference:
      python cli.py btc status
      python cli.py btc update
      python cli.py btc history -d 7
      python cli.py btc backtest -t 15 --targets 50,100
      python cli.py btc alert -t 20 --notify
    """
    pass


@btc.command("status")
def btc_status():
    """Show current BTC FGI and price."""
    import pandas as pd
    from src.crypto.fetcher import fetch_fgi_data, fetch_btc_price

    data = fetch_fgi_data()
    if not data:
        click.echo("Failed to fetch data")
        return

    latest = data[-1]
    try:
        btc_price = fetch_btc_price()
        price_str = f"${btc_price:,.0f} (snapshot: ${latest['btc_price']:,.0f})"
    except Exception:
        price_str = f"${latest['btc_price']:,.0f} (snapshot)"
    click.echo(f"BTC FGI: {latest['fgi_value']} ({latest['fgi_class']}) | {price_str} | {latest['date']}")


@btc.command("history")
@click.option("--days", "-d", default=7, help="Number of days to show (default: 7)")
def btc_history(days):
    """Show recent BTC FGI history."""
    from src.crypto.fetcher import fetch_fgi_data

    data = fetch_fgi_data()
    if not data:
        click.echo("Failed to fetch data")
        return

    recent = data[-days:]
    click.echo(f"{'Date':<12} {'FGI':>4}  {'Class':<14} {'Price':>10}")
    click.echo("-" * 46)
    for row in recent:
        fgi = row['fgi_value']
        line = f"{row['date']:<12} {fgi:>4}  {row['fgi_class']:<14} ${row['btc_price']:>9,.0f}"
        if fgi <= 15:
            click.echo(click.style(f"🔥 {line}", fg="red", bold=True))
        else:
            click.echo(f"   {line}")


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
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
def btc_backtest(threshold, targets, start):
    """Run BTC FGI backtest."""
    import pandas as pd
    from tabulate import tabulate
    from src.crypto.backtester import backtest

    merged_path = DATA_DIR / "btc_merged.csv"
    if not merged_path.exists():
        click.echo("No local data, fetching from CoinGlass...")
        from src.crypto.fetcher import fetch_fgi_data
        DATA_DIR.mkdir(exist_ok=True)
        data_raw = fetch_fgi_data()
        df = pd.DataFrame(data_raw)
        df.to_csv(merged_path, index=False)
        click.echo(f"  Fetched {len(data_raw)} records, saved to {merged_path}")
    else:
        df = pd.read_csv(merged_path)
    if start:
        df = df[df["date"] >= start]
    data = df.to_dict("records")
    target_list = [int(t.strip()) for t in targets.split(",")]

    result = backtest(data, threshold, target_list)

    date_from = result['date_range']['from']
    date_to = result['date_range']['to']
    click.echo(f"\nFGI ≤{threshold} | {date_from} ~ {date_to} | {len(result['signals'])} buy signals")

    headers = ["Date", "FGI", "Price"] + [f"+{t}%" for t in target_list]
    rows = []
    for s in result["signals"]:
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
        mn = f"{s['min_days']}d" if s.get('min_days') else "--"
        mx = f"{s['max_days']}d" if s.get('max_days') else "--"
        min_date = s.get('min_date') or ""
        max_date = s.get('max_date') or ""
        summary_rows.append([f"+{t}%", avg, f"{mn} ({min_date})" if min_date else mn, f"{mx} ({max_date})" if max_date else mx, rate])
    click.echo(tabulate(summary_rows, headers=["Target", "Avg", "Min (date)", "Max (date)", "Success Rate"], tablefmt="simple"))


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
            message = f"""🚨 BTC 恐惧与贪婪指数警报

━━━━━━━━━━━━━━━━━━━━━━
📊 当前指标
━━━━━━━━━━━━━━━━━━━━━━
• FGI 指数: {fgi} ({latest['fgi_class']})
• BTC 价格: ${price:,.0f}
• 数据日期: {date}

━━━━━━━━━━━━━━━━━━━━━━
💡 策略提示
━━━━━━━━━━━━━━━━━━━━━━
当前 FGI ({fgi}) 低于阈值 ({threshold})，
市场处于极度恐惧状态。

根据历史回测 (FGI ≤15):
• +50% 成功率: 93%
• +100% 成功率: 93%

这可能是一个买入 BTC 的好时机。

━━━━━━━━━━━━━━━━━━━━━━
⚙️ 由 GitHub Actions 自动发送"""
            notifier.send(f"🚨 BTC FGI 警报 - {fgi} 极度恐惧!", message)
    else:
        click.echo(f"FGI ({fgi}) is above threshold ({threshold}). No alert.")


# =============================================================================
# Stock Commands
# =============================================================================

@cli.group()
def stock():
    """Individual stock sentiment commands

    \b
    Quick Reference:
      python cli.py stock status                  # 查看 watchlist
      python cli.py stock status AAPL -v          # 详细模式
      python cli.py stock list
      python cli.py stock add TSLA
      python cli.py stock backtest NVDA -t 30 --days 365
      python cli.py stock cache status
      python cli.py stock alert -t 25 --notify
    """
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


def get_rsi_status(rsi: float) -> str:
    if rsi < 30:
        return "超卖"
    elif rsi < 40:
        return "偏弱"
    elif rsi <= 60:
        return "中性"
    elif rsi <= 70:
        return "偏强"
    else:
        return "超买"


def get_pcr_status(pcr: float) -> str:
    if pcr >= 1.5:
        return "极度恐慌"
    elif pcr >= 1.2:
        return "看跌"
    elif pcr >= 0.8:
        return "中性"
    elif pcr >= 0.5:
        return "看涨"
    else:
        return "极度贪婪"


def get_news_status(news: float) -> str:
    if news < 30:
        return "非常负面"
    elif news < 45:
        return "负面"
    elif news <= 55:
        return "中性"
    elif news <= 70:
        return "正面"
    else:
        return "非常正面"


def get_sentiment_status(score: float) -> str:
    if score < 25:
        return "极度恐惧"
    elif score < 40:
        return "恐惧"
    elif score <= 60:
        return "中性"
    elif score <= 75:
        return "贪婪"
    else:
        return "极度贪婪"


@stock.command("status")
@click.argument("ticker", required=False)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed breakdown")
def stock_status(ticker, verbose):
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

        if verbose or len(tickers) == 1:
            # Detailed view for single stock or verbose mode
            click.echo(f"\n{'='*50}")
            click.echo(f"  {t}")
            click.echo(f"{'='*50}")

            # Price info
            if data.get("realtime"):
                chg = data.get("change_percent", 0)
                sign = "+" if chg >= 0 else ""
                click.echo(f"  价格:      ${data['price']:.2f} ({sign}{chg:.2f}%) [实时]")
            else:
                click.echo(f"  价格:      ${data['price']:.2f}  [{data['date']}]")

            # Sentiment score
            status_text = get_sentiment_status(score) if score else "N/A"
            click.echo(f"  情绪指数:  {score} ({status_text})")

            click.echo(f"\n  指标明细")
            click.echo(f"  {'-'*46}")

            # RSI
            if data.get("rsi"):
                rsi = data["rsi"]
                rsi_status = get_rsi_status(rsi)
                rsi_score = components.get("rsi", "N/A")
                click.echo(f"  RSI:       {rsi:.1f} ({rsi_status}) → 得分: {rsi_score:.0f}" if isinstance(rsi_score, float) else f"  RSI:       {rsi:.1f} ({rsi_status})")

            # PCR
            if data.get("pcr"):
                pcr = data["pcr"]
                pcr_status = get_pcr_status(pcr)
                pcr_score = components.get("pcr", "N/A")
                click.echo(f"  看跌/看涨: {pcr:.2f} ({pcr_status}) → 得分: {pcr_score:.0f}" if isinstance(pcr_score, float) else f"  看跌/看涨: {pcr:.2f} ({pcr_status})")

            # Volume
            if data.get("volume"):
                vol = data["volume"]
                vol_score = components.get("volume", "N/A")
                vol_display = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
                click.echo(f"  成交量:    {vol_display} → 得分: {vol_score:.0f}" if isinstance(vol_score, float) else f"  成交量:    {vol_display}")

            # News
            if data.get("news_sentiment"):
                news = data["news_sentiment"]
                news_status = get_news_status(news)
                news_score = components.get("news", "N/A")
                click.echo(f"  新闻情绪:  {news:.0f} ({news_status}) → 得分: {news_score:.0f}" if isinstance(news_score, float) else f"  新闻情绪:  {news:.0f} ({news_status})")

            click.echo("")
        else:
            # Compact view for watchlist
            status_text = get_sentiment_status(score) if score else "N/A"
            parts = [f"{t}: {score} ({status_text})"]

            if "rsi" in components:
                parts.append(f"RSI:{data['rsi']:.0f}")
            if data.get("pcr"):
                parts.append(f"PCR:{data['pcr']:.2f}")

            price_str = f"${data['price']:.2f}"
            if data.get("realtime") and data.get("change_percent") is not None:
                chg = data["change_percent"]
                sign = "+" if chg >= 0 else ""
                price_str += f" ({sign}{chg:.2f}%)"
            parts.append(price_str)

            if data.get("realtime"):
                parts.append("[实时]")
            else:
                parts.append(f"[{data['date']}]")

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


@stock.command("cache")
@click.argument("action", type=click.Choice(["status", "clear"]))
@click.argument("ticker", required=False)
def stock_cache(action, ticker):
    """Manage data cache.

    \b
    Actions:
      status  - Show cache status for all tickers or a specific ticker
      clear   - Clear cache for all tickers or a specific ticker
    """
    from pathlib import Path
    import json

    cache_dir = Path(__file__).parent / "data" / "cache"

    if action == "status":
        if not cache_dir.exists():
            click.echo("No cache directory found")
            return

        files = list(cache_dir.glob("*.json"))
        if ticker:
            files = [f for f in files if f.stem.upper() == ticker.upper()]

        if not files:
            click.echo("No cached data found")
            return

        for f in sorted(files):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                rsi_count = len(data.get("rsi", {}))
                price_count = len(data.get("prices", {}))

                rsi_dates = sorted(data.get("rsi", {}).keys())
                date_range = f"{rsi_dates[0]} to {rsi_dates[-1]}" if rsi_dates else "N/A"

                size_kb = f.stat().st_size / 1024
                click.echo(f"{f.stem}: {price_count} days ({date_range}) [{size_kb:.1f}KB]")
            except Exception as e:
                click.echo(f"{f.stem}: Error reading cache - {e}")

    elif action == "clear":
        if not cache_dir.exists():
            click.echo("No cache to clear")
            return

        if ticker:
            cache_file = cache_dir / f"{ticker.upper()}.json"
            if cache_file.exists():
                cache_file.unlink()
                click.echo(f"Cleared cache for {ticker.upper()}")
            else:
                click.echo(f"No cache found for {ticker.upper()}")
        else:
            files = list(cache_dir.glob("*.json"))
            for f in files:
                f.unlink()
            click.echo(f"Cleared cache for {len(files)} tickers")


@stock.command("backtest")
@click.argument("ticker")
@click.option("-t", "--threshold", default=30, help="Sentiment threshold")
@click.option("--targets", default="10,20,30,40,50,60,70,80,90,100", help="Target percentages")
@click.option("--days", default=730, help="Days of history")
@click.option("--from", "date_from", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--to", "date_to", default=None, help="End date (YYYY-MM-DD)")
def stock_backtest(ticker, threshold, targets, days, date_from, date_to):
    """Run backtest for a stock.

    Examples:
        python cli.py stock backtest TSLA
        python cli.py stock backtest TSLA --from 2024-01-01
        python cli.py stock backtest TSLA --from 2024-01-01 --to 2024-12-31
    """
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

    result = backtest(ticker, rsi_data, prices, volumes, threshold, target_list, date_from, date_to)

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
        mn = f"{s['min_days']}d" if s.get('min_days') else "--"
        mx = f"{s['max_days']}d" if s.get('max_days') else "--"
        summary_rows.append([f"+{t}%", mn, avg, mx, rate])
    click.echo(tabulate(summary_rows, headers=["Target", "Min", "Avg", "Max", "Success Rate"], tablefmt="simple"))


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

        status_text = get_sentiment_status(score) if score else "N/A"
        parts = [f"{ticker}: {score} ({status_text})"]
        if "rsi" in components:
            parts.append(f"RSI:{components['rsi']:.0f}")

        price_str = f"${data['price']:.2f}"
        if data.get("realtime") and data.get("change_percent") is not None:
            chg = data["change_percent"]
            sign = "+" if chg >= 0 else ""
            price_str += f" ({sign}{chg:.2f}%)"
        parts.append(price_str)

        if data.get("realtime"):
            parts.append("[实时]")
        else:
            parts.append(f"[{data['date']}]")

        click.echo(" | ".join(parts))

    if alerts:
        click.echo(f"\nALERT: {len(alerts)} stock(s) below threshold ({threshold})!")

        if notify:
            notifier = MultiNotifier()

            lines = [
                "🚨 个股情绪警报",
                "",
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"📊 {len(alerts)} 只股票触发极度恐惧信号",
                "━━━━━━━━━━━━━━━━━━━━━━",
                ""
            ]

            for a in alerts:
                comp = a["components"]
                lines.append(f"【{a['ticker']}】")
                lines.append(f"  • 综合情绪: {a['score']:.1f} (阈值: {threshold})")
                lines.append(f"  • 当前价格: ${a['price']:.2f}")

                details = []
                if comp.get("rsi"):
                    rsi = comp["rsi"]
                    rsi_status = "超卖" if rsi < 30 else "正常"
                    details.append(f"RSI: {rsi:.0f} ({rsi_status})")
                if comp.get("volume"):
                    details.append(f"成交量: {comp['volume']:.0f}")
                if comp.get("pcr"):
                    pcr = comp["pcr"]
                    pcr_status = "看跌" if pcr < 50 else "看涨"
                    details.append(f"PCR: {pcr:.0f} ({pcr_status})")
                if comp.get("news"):
                    news = comp["news"]
                    news_status = "负面" if news < 40 else "中性" if news < 60 else "正面"
                    details.append(f"新闻: {news:.0f} ({news_status})")

                if details:
                    lines.append(f"  • 指标明细: {' | '.join(details)}")
                lines.append("")

            lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━",
                "💡 策略提示",
                "━━━━━━━━━━━━━━━━━━━━━━",
                "市场处于极度恐惧状态，",
                "根据历史回测，这可能是买入机会。",
                "",
                "TSLA/NVDA 回测成功率: 100%",
                "AAPL 回测成功率: 82%",
                "",
                "━━━━━━━━━━━━━━━━━━━━━━",
                "⚙️ 由 GitHub Actions 自动发送"
            ])

            notifier.send(f"🚨 个股警报 - {len(alerts)}只股票极度恐惧!", "\n".join(lines))
    else:
        click.echo(f"\nNo alerts. All above threshold ({threshold}).")


if __name__ == "__main__":
    cli()
