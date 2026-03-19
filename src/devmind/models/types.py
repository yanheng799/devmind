"""Type definitions for DEVMIND."""

from enum import Enum
from typing import Literal


class EventType(str, Enum):
    """Types of events that can affect stock prices."""

    MONETARY_POLICY = "monetary_policy"
    FISCAL_POLICY = "fiscal_policy"
    EARNINGS = "earnings"
    GEOPOLITICAL = "geopolitical"
    REGULATORY = "regulatory"
    INDUSTRY = "industry"
    MARKET_SENTIMENT = "market_sentiment"
    TECHNOLOGY = "technology"
    SUPPLY_CHAIN = "supply_chain"
    COMMODITY = "commodity"
    OTHER = "other"


class TimeHorizon(str, Enum):
    """Time horizon for price impact."""

    IMMEDIATE = "immediate"  # < 1 day
    SHORT = "short"  # 1-5 days
    MEDIUM = "medium"  # 5-20 days
    LONG = "long"  # 20+ days


class Direction(str, Enum):
    """Price direction prediction."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SourceType(str, Enum):
    """News source types."""

    PRIMARY = "primary"  # Official announcements
    SECONDARY = "secondary"  # News outlets


class Sentiment(str, Enum):
    """Sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
