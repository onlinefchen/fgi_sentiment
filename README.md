# FGI Sentiment Analysis Tool

统一的恐惧与贪婪指数分析工具，支持 BTC 和个股。

## 功能

| 类型 | 数据源 | 指标 |
|------|--------|------|
| BTC | CoinGlass | Fear & Greed Index |
| 个股 | Polygon | RSI + Volume + Put/Call Ratio + News |

## 安装

```bash
pip install -r requirements.txt

# BTC
export COINGLASS_API_KEY="your_key"

# 个股
export POLYGON_API_KEY="your_key"
```

## BTC 命令

```bash
# 当前状态
python cli.py btc status

# 更新数据
python cli.py btc update

# 回测
python cli.py btc backtest -t 15

# 警报
python cli.py btc alert -t 15 --notify
```

## 个股命令

```bash
# 管理 watchlist
python cli.py stock add AAPL
python cli.py stock add TSLA
python cli.py stock list

# 当前状态
python cli.py stock status AAPL
python cli.py stock status  # 所有 watchlist

# 回测
python cli.py stock backtest AAPL
python cli.py stock backtest NVDA -t 25

# 警报
python cli.py stock alert --notify
```

## 回测结果

### BTC (FGI < 15)

| 目标 | 成功率 | 平均天数 |
|------|--------|----------|
| +50% | 93% | ~200d |
| +100% | 93% | ~400d |

### 个股 (Sentiment < 30)

| 股票 | +10%成功率 | +20%成功率 |
|------|------------|------------|
| AAPL | 82% | 82% |
| TSLA | 100% | 100% |
| NVDA | 100% | 100% |

## GitHub Actions

自动检查并发送通知：

- BTC: 每小时检查
- 个股: 美股交易时间每小时检查

### 配置 Secrets

| Secret | 说明 |
|--------|------|
| `COINGLASS_API_KEY` | CoinGlass API |
| `POLYGON_API_KEY` | Polygon API |
| `EMAIL_USERNAME` | Gmail |
| `EMAIL_PASSWORD` | Gmail 应用密码 |
| `EMAIL_TO` | 接收邮箱 |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot |
| `TELEGRAM_CHAT_ID` | Telegram Chat ID |
