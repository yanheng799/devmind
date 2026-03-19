"""Base data models for DEVMIND."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from devmind.models.types import Direction, EventType, Sentiment, SourceType, TimeHorizon


class NewsArticle(BaseModel):
    """Raw news article data.

    Attributes:
        article_id: Unique identifier
        title: Article title
        content: Full article content
        source: Source name (e.g., "eastmoney", "xinhua")
        source_type: PRIMARY or SECONDARY
        publish_time: Publication timestamp
        url: Article URL
        related_stocks: List of related stock codes (extracted or annotated)
        metadata: Additional metadata
    """

    article_id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Article title")
    content: str = Field(..., min_length=1, description="Article content")
    source: str = Field(..., min_length=1, description="Source name")
    source_type: SourceType = Field(default=SourceType.SECONDARY, description="Source type")
    publish_time: datetime = Field(..., description="Publication timestamp")
    url: str = Field(..., min_length=1, description="Article URL")
    related_stocks: list[str] = Field(default_factory=list, description="Related stock codes")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("related_stocks")
    @classmethod
    def validate_stock_codes(cls, value: list[str]) -> list[str]:
        """Validate stock codes are in correct format."""
        for code in value:
            if not isinstance(code, str):
                raise ValueError(f"Stock code must be string, got {type(code)}")
            # A-share codes are 6 digits, optionally with suffix
            if not code.replace(".", "").replace("SH", "").replace("SZ", "").isdigit():
                raise ValueError(f"Invalid stock code format: {code}")
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Validate URL format."""
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http:// or https://, got: {value}")
        return value


class ExtractedEvent(BaseModel):
    """Event extracted from news article.

    Attributes:
        event_id: Unique identifier
        article_id: Source article ID
        event_type: Category of event
        entities: Named entities (companies, sectors, etc.)
        magnitude: Impact magnitude (e.g., "high", "medium", "low")
        timeframe: Expected impact timeframe
        transmission_chain: Chain of causal reasoning
        confidence: Extraction confidence (0-1)
        raw_evidence: Supporting text from article
    """

    event_id: str = Field(..., description="Unique event identifier")
    article_id: str = Field(..., description="Source article ID")
    event_type: EventType = Field(..., description="Event category")
    entities: list[str] = Field(..., min_length=1, description="Named entities")
    magnitude: str = Field(..., description="Impact magnitude: high/medium/low")
    timeframe: TimeHorizon = Field(..., description="Expected impact timeframe")
    transmission_chain: list[str] = Field(
        ...,
        min_length=1,
        description="Chain of causal reasoning",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    raw_evidence: str = Field(..., description="Supporting text from article")

    @field_validator("magnitude")
    @classmethod
    def validate_magnitude(cls, value: str) -> str:
        """Validate magnitude value."""
        valid_values = {"high", "medium", "low"}
        value_lower = value.lower()
        if value_lower not in valid_values:
            raise ValueError(f"Magnitude must be one of {valid_values}, got: {value}")
        return value_lower


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result.

    Attributes:
        article_id: Source article ID
        sentiment: POSITIVE, NEGATIVE, or NEUTRAL
        score: Sentiment score (-1 to 1)
        confidence: Analysis confidence (0-1)
        aspects: Aspect-based sentiment (optional)
    """

    article_id: str = Field(..., description="Source article ID")
    sentiment: Sentiment = Field(..., description="Sentiment classification")
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1 to 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    aspects: dict[str, str] = Field(
        default_factory=dict,
        description="Aspect-based sentiment results",
    )


class HistoricalEvent(BaseModel):
    """Historical event for similarity search.

    Similar to ExtractedEvent but includes actual outcome data.

    Attributes:
        event_id: Unique identifier
        event_type: Category of event
        description: Event description
        entities: Related entities
        magnitude: Impact magnitude
        timeframe: Time horizon
        stock_code: Affected stock
        actual_direction: Actual price movement (BULLISH/BEARISH/NEUTRAL)
        actual_change_pct: Actual price change percentage
        event_date: Date of event
        vector_id: Vector store ID
    """

    event_id: str = Field(..., description="Unique event identifier")
    event_type: EventType = Field(..., description="Event category")
    description: str = Field(..., description="Event description")
    entities: list[str] = Field(..., description="Related entities")
    magnitude: str = Field(..., description="Impact magnitude")
    timeframe: TimeHorizon = Field(..., description="Time horizon")
    stock_code: str = Field(..., description="Affected stock code")
    actual_direction: Direction = Field(..., description="Actual price movement")
    actual_change_pct: float = Field(..., description="Actual price change %")
    event_date: datetime = Field(..., description="Event date")
    vector_id: str | None = Field(default=None, description="Vector store ID")

    @field_validator("magnitude")
    @classmethod
    def validate_magnitude(cls, value: str) -> str:
        """Validate magnitude value."""
        valid_values = {"high", "medium", "low"}
        value_lower = value.lower()
        if value_lower not in valid_values:
            raise ValueError(f"Magnitude must be one of {valid_values}, got: {value}")
        return value_lower
