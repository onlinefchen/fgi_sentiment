"""Extensible sentiment score calculation"""
import json
import os
import statistics
from abc import ABC, abstractmethod
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "stock_sentiment.json"


class Indicator(ABC):
    """Base class for all sentiment indicators"""
    name: str = "base"
    weight: float = 1.0

    @abstractmethod
    def calculate(self, data: dict, date: str) -> float | None:
        """Calculate indicator score (0-100, lower = more fear)

        Args:
            data: Dict containing all fetched data (rsi, prices, volumes, etc.)
            date: Date to calculate for

        Returns:
            Score 0-100 or None if cannot calculate
        """
        pass


class RSIIndicator(Indicator):
    """RSI-based fear indicator"""
    name = "rsi"
    weight = 0.6

    def calculate(self, data: dict, date: str) -> float | None:
        rsi = data.get("rsi", {}).get(date)
        if rsi is None:
            return None
        # RSI is already 0-100, lower = more fear
        return rsi


class VolumeIndicator(Indicator):
    """Volume anomaly indicator - high volume during drop = capitulation"""
    name = "volume"
    weight = 0.4
    lookback: int = 20

    def calculate(self, data: dict, date: str) -> float | None:
        volumes = data.get("volumes", {})
        if date not in volumes:
            return None

        sorted_dates = sorted(volumes.keys())
        try:
            idx = sorted_dates.index(date)
        except ValueError:
            return None

        if idx < self.lookback:
            return 50  # Not enough data, return neutral

        recent_dates = sorted_dates[idx - self.lookback:idx]
        recent_volumes = [volumes[d] for d in recent_dates]
        avg_volume = statistics.mean(recent_volumes)

        current_volume = volumes[date]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # High volume = capitulation = fear = low score
        if volume_ratio >= 2.0:
            return 10
        elif volume_ratio >= 1.5:
            return 30
        elif volume_ratio >= 1.2:
            return 40
        elif volume_ratio >= 0.8:
            return 50
        else:
            return 70


class PutCallRatioIndicator(Indicator):
    """Put/Call Ratio indicator - high PCR = bearish sentiment"""
    name = "pcr"
    weight = 0.2

    def calculate(self, data: dict, date: str) -> float | None:
        pcr = data.get("pcr")
        if pcr is None:
            return None

        # Convert PCR to 0-100 score
        # PCR > 1.5 = extreme fear (low score)
        # PCR < 0.5 = extreme greed (high score)
        # PCR = 1 = neutral (50)
        if pcr >= 1.5:
            return 15
        elif pcr >= 1.2:
            return 30
        elif pcr >= 0.8:
            return 50
        elif pcr >= 0.5:
            return 70
        else:
            return 85


class NewsIndicator(Indicator):
    """News sentiment indicator"""
    name = "news"
    weight = 0.15

    def calculate(self, data: dict, date: str) -> float | None:
        # News sentiment is already 0-100
        return data.get("news_sentiment")


class SentimentCalculator:
    """Calculate combined sentiment score using multiple indicators"""

    def __init__(self, config_path: Path | None = None):
        self.indicators: list[Indicator] = []
        self.config = self._load_config(config_path or CONFIG_PATH)
        self._init_indicators()

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        # Default config
        return {
            "indicators": {
                "rsi": {"enabled": True, "weight": 0.6},
                "volume": {"enabled": True, "weight": 0.4}
            },
            "alert_threshold": 30
        }

    def _init_indicators(self):
        """Initialize enabled indicators with configured weights"""
        indicator_classes = {
            "rsi": RSIIndicator,
            "volume": VolumeIndicator,
            "pcr": PutCallRatioIndicator,
            "news": NewsIndicator
        }

        for name, settings in self.config.get("indicators", {}).items():
            if settings.get("enabled", True) and name in indicator_classes:
                indicator = indicator_classes[name]()
                indicator.weight = settings.get("weight", indicator.weight)
                self.indicators.append(indicator)

    def calculate(self, data: dict, date: str) -> dict:
        """Calculate combined sentiment score

        Returns:
            {
                "score": float,  # Combined score 0-100
                "components": {  # Individual indicator scores
                    "rsi": float,
                    "volume": float
                }
            }
        """
        components = {}
        total_weight = 0
        weighted_sum = 0

        for indicator in self.indicators:
            score = indicator.calculate(data, date)
            if score is not None:
                components[indicator.name] = score
                weighted_sum += score * indicator.weight
                total_weight += indicator.weight

        if total_weight == 0:
            return {"score": None, "components": components}

        final_score = weighted_sum / total_weight
        return {
            "score": round(final_score, 1),
            "components": components
        }

    @property
    def alert_threshold(self) -> int:
        return self.config.get("alert_threshold", 30)
